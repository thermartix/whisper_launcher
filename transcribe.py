"""Run the configured Whisper preset on one local media file."""

import argparse
import json
import os
from dataclasses import asdict
from html import escape
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = ROOT / "whisper-settings.json"
MIXED_LANGUAGES = "multilingual"
SAMPLE_RATE = 16000
OUTPUT_FORMATS = ("txt", "srt", "vtt", "tsv", "json")


def load_settings(path=DEFAULT_SETTINGS):
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    keys = {"model", "language", "task", "output_format"}
    if not isinstance(settings, dict) or settings.keys() - {"device", "word_timestamps"} != keys:
        raise ValueError("whisper-settings.json requires model, language, task, and output_format.")

    model = settings["model"]
    language = settings["language"]
    task = settings["task"]
    if language == MIXED_LANGUAGES:
        from faster_whisper import available_models
    else:
        from whisper import available_models
        from whisper.tokenizer import LANGUAGES

        if language is not None and (not isinstance(language, str) or language not in LANGUAGES):
            raise ValueError(
                "Use null to auto-detect one language, 'multilingual' for mixed speech, or a code such as 'en'."
            )

    if model not in available_models():
        raise ValueError(f"Unknown Whisper model {model!r}.")
    if task not in ("transcribe", "translate"):
        raise ValueError("Set task to 'transcribe' or 'translate'.")
    if model.endswith(".en") and (language not in (None, "en") or task != "transcribe"):
        raise ValueError("English-only models support English transcription only.")
    if task == "translate" and model in ("turbo", "large-v3-turbo"):
        raise ValueError("Choose a multilingual model such as 'small' for English translation.")
    if settings["output_format"] not in (*OUTPUT_FORMATS, "all"):
        raise ValueError("Set output_format to 'all', 'txt', 'srt', 'vtt', 'tsv', or 'json'.")
    word_timestamps = settings.get("word_timestamps", False)
    if not isinstance(word_timestamps, bool):
        raise ValueError("Set word_timestamps to true or false.")
    settings["word_timestamps"] = word_timestamps and settings["output_format"] in ("json", "vtt")
    return settings


def run_model(audio, settings, device):
    import torch
    import whisper

    hardware = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"Loading Whisper {settings['model']} on {hardware} (downloads the model on first use)...")
    # Keep the checkpoint out of VRAM while the model moves to CUDA.
    model = whisper.load_model(
        settings["model"], device="cpu", download_root=str(ROOT / ".whisper-runner" / "models")
    ).to(device)
    print(f"Running {settings['task']}...")
    return model.transcribe(
        audio,
        language=settings["language"],
        task=settings["task"],
        fp16=device == "cuda",
        verbose=False,
        word_timestamps=settings.get("word_timestamps", False),
    )


def infer(audio, settings, preference):
    import torch

    device = preference
    if preference == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        return run_model(audio, settings, device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        if preference == "cuda":
            raise SystemExit("The GPU ran out of memory. Try a smaller Whisper model or set device to 'auto'.") from None
        print("The GPU ran out of memory. Retrying on CPU with the same Whisper model.")

    # The exception's traceback may hold GPU tensors until control leaves the handler.
    torch.cuda.empty_cache()
    return run_model(audio, settings, "cpu")


def run_mixed_model(audio, clips, settings, device):
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    print(f"Loading faster-whisper {settings['model']} on {device.upper()} (downloads the model on first use)...")
    model = WhisperModel(
        settings["model"],
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
        download_root=str(ROOT / ".whisper-runner" / "models-fast"),
    )
    print(f"Running mixed-language {settings['task']}...")
    # Re-detect the language for each speech clip and reset the prompt between windows.
    segments, info = model.transcribe(
        audio,
        task=settings["task"],
        multilingual=True,
        condition_on_previous_text=False,
        clip_timestamps=clips,
        word_timestamps=settings.get("word_timestamps", False),
    )
    # Track the original timeline, including silence skipped by the speech detector.
    decoded = []
    with tqdm(total=round(info.duration), desc="Transcribing", unit="s") as progress:
        for segment in segments:
            decoded.append(segment)
            end = min(round(segment.end), progress.total)
            progress.update(max(0, end - progress.n))
        progress.update(progress.total - progress.n)
    return decoded


def infer_mixed(audio, settings, preference):
    from ctranslate2 import get_cuda_device_count
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # Split near pauses so a language switch does not have to wait for a 30-second window.
    speech = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=650, max_speech_duration_s=15), sampling_rate=SAMPLE_RATE
    )
    if not speech:
        print("No speech detected.")
        return []
    clips = [span[edge] / SAMPLE_RATE for span in speech for edge in ("start", "end")]

    if preference != "cpu":
        try:
            if not get_cuda_device_count():
                raise RuntimeError("CTranslate2 found no CUDA device.")
            return run_mixed_model(audio, clips, settings, "cuda")
        except RuntimeError as error:
            if preference == "cuda":
                raise SystemExit(
                    f"faster-whisper could not use CUDA. {error}\n"
                    "Check that CUDA 12 cuBLAS and cuDNN 9 are available on PATH, or choose CPU."
                ) from None
            print(f"faster-whisper could not use CUDA. {error}\nRetrying with faster-whisper on CPU.")
    return run_mixed_model(audio, clips, settings, "cpu")


def write_word_vtt(segments, file, format_timestamp):
    print("WEBVTT\n", file=file)
    for segment in segments:
        start = max(0, round(segment["start"] * 1000))
        end = max(start + 1, round(segment["end"] * 1000))
        previous = start
        parts = []
        for word in segment.get("words") or ():
            timestamp = round(word["start"] * 1000)
            # Inline timestamps must increase strictly and be inside the cue.
            if previous < timestamp < end:
                marker = format_timestamp(timestamp / 1000, always_include_hours=True)
                parts.append(f"<{marker}>")
                previous = timestamp
            parts.append(escape(word["word"], quote=False))
        text = "".join(parts) if parts else escape(segment["text"], quote=False)
        text = text.replace("\r", " ").replace("\n", " ").strip()
        if text:
            cue_start = format_timestamp(start / 1000, always_include_hours=True)
            cue_end = format_timestamp(end / 1000, always_include_hours=True)
            print(f"{cue_start} --> {cue_end}\n{text}\n", file=file)


def write_mixed_outputs(segments, media, staging, output_format, word_timestamps=False):
    from faster_whisper.utils import format_timestamp

    formats = OUTPUT_FORMATS if output_format == "all" else (output_format,)
    for extension in formats:
        path = Path(staging) / f"{media.stem}.{extension}"
        with path.open("w", encoding="utf-8") as file:
            if extension == "json":
                result = {
                    "text": "".join(segment.text for segment in segments),
                    "segments": [asdict(segment) for segment in segments],
                    "language": None,
                }
                json.dump(result, file, ensure_ascii=False)
                continue
            if extension == "vtt":
                if word_timestamps:
                    write_word_vtt((asdict(segment) for segment in segments), file, format_timestamp)
                    continue
                print("WEBVTT\n", file=file)
            if extension == "tsv":
                print("start\tend\ttext", file=file)
            for index, segment in enumerate(segments, start=1):
                text = segment.text.strip()
                if extension == "txt":
                    print(text, file=file)
                elif extension == "tsv":
                    start, end = round(1000 * segment.start), round(1000 * segment.end)
                    print(start, end, text.replace("\t", " "), sep="\t", file=file)
                else:
                    srt = extension == "srt"
                    separator = "," if srt else "."
                    start = format_timestamp(segment.start, srt, separator)
                    end = format_timestamp(segment.end, srt, separator)
                    if srt:
                        print(index, file=file)
                    print(f"{start} --> {end}\n{text.replace('-->', '->')}\n", file=file)


def transcribe(media, settings, device):
    # Stage the files next to the input, then publish them together after all writers succeed.
    with TemporaryDirectory(dir=media.parent, prefix=".whisper-") as staging:
        print(f"Reading audio from {media.name}...")
        output_format = settings["output_format"]
        word_timestamps = settings.get("word_timestamps", False)
        if settings["language"] == MIXED_LANGUAGES:
            from av.error import FFmpegError
            from faster_whisper import decode_audio

            try:
                audio = decode_audio(str(media), sampling_rate=SAMPLE_RATE)
                if not audio.size:
                    raise ValueError("The file contains no audio samples.")
            except (FFmpegError, IndexError, ValueError) as error:
                raise SystemExit(f"Could not read audio from {media.name}. {error}") from error
            segments = infer_mixed(audio, settings, device)
            write_mixed_outputs(segments, media, staging, output_format, word_timestamps)
        else:
            import whisper
            from whisper.utils import format_timestamp, get_writer

            try:
                audio = whisper.load_audio(str(media))
            except RuntimeError as error:
                detail = "\n".join(str(error).splitlines()[-3:])
                raise SystemExit(
                    f"Could not read audio from {media.name}. Choose a playable file with an audio track.\n{detail}"
                ) from error
            result = infer(audio, settings, device)
            if output_format == "vtt" and word_timestamps:
                with (Path(staging) / f"{media.stem}.vtt").open("w", encoding="utf-8") as file:
                    write_word_vtt(result["segments"], file, format_timestamp)
            else:
                get_writer(output_format, staging)(result, str(media))

        # On Windows rename fails if the destination exists, including during a concurrent run.
        for run in count(1):
            suffix = "" if run == 1 else f".{run}"
            output = media.with_name(f"{media.name}.whisper{suffix}")
            try:
                return Path(staging).rename(output)
            except FileExistsError:
                continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--open-output", action="store_true")
    args = parser.parse_args()
    media = args.media.absolute()
    if not media.is_file():
        parser.error("Choose an existing audio or video file.")

    try:
        settings = load_settings(args.settings)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Check {args.settings}. {error}\n")

    output = transcribe(media, settings, args.device)
    print(f"Saved to {output}")
    if args.open_output:
        try:
            os.startfile(output)
        except OSError as error:
            print(f"Could not open the output folder. {error}")


if __name__ == "__main__":
    main()
