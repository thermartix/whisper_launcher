"""Run the configured Whisper preset on one local media file."""

import argparse
import json
import math
import os
import re
import unicodedata
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
PROMPT_TOKEN_LIMIT = 200

# Windows can cache models without symlinks; no elevation or Developer Mode is needed.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def configure_fast_cuda():
    """Find the launcher-local NVIDIA DLLs before CTranslate2 starts inference."""
    if os.name != "nt":
        return []

    import ctypes
    from importlib.metadata import PackageNotFoundError, distribution

    handles = []
    for package, folder in (("nvidia-cublas-cu12", "cublas"), ("nvidia-cudnn-cu12", "cudnn")):
        try:
            directory = distribution(package).locate_file(f"nvidia/{folder}/bin")
        except PackageNotFoundError:
            continue
        if directory.is_dir():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            handles.append(os.add_dll_directory(str(directory)))

    # Some missing cuDNN components can abort native inference instead of raising in Python.
    for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"):
        try:
            handles.append(ctypes.WinDLL(name, winmode=0))
        except OSError as error:
            raise RuntimeError(f"Could not load {name}. Run Run-Whisper.ps1 to repair the CUDA libraries.") from error
    # Keep directory registrations and loaded libraries alive throughout inference.
    return handles


def prepare_mixed_model(name):
    from faster_whisper.utils import _MODELS
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tqdm import tqdm

    # Use the repository mapping from our pinned faster-whisper release, but show downloads.
    options = dict(
        repo_id=_MODELS[name],
        cache_dir=str(ROOT / ".whisper-runner" / "models-fast"),
        allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
    )

    def complete(path):
        # HF can return a snapshot directory even if a previous model download was interrupted.
        return all((Path(path) / file).is_file() and (Path(path) / file).stat().st_size
                   for file in ("config.json", "model.bin", "tokenizer.json")) and any(Path(path).glob("vocabulary.*"))

    try:
        cached = snapshot_download(**options, local_files_only=True)
        if complete(cached):
            print(f"Using cached faster-whisper {name} model.")
            return cached
    except LocalEntryNotFoundError:
        pass

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            # Redirected WinForms logs support carriage returns, not terminal cursor movement.
            kwargs.update(position=0, leave=False, ncols=90, mininterval=0.5, ascii=True)
            super().__init__(*args, **kwargs)

    print(f"Downloading faster-whisper {name}. The first download can take several minutes; retries reuse the cache.")
    try:
        path = snapshot_download(**options, tqdm_class=DownloadProgress)
        if not complete(path):
            raise OSError("The model cache is incomplete.")
    except Exception as error:
        raise SystemExit(
            f"Could not download faster-whisper {name}. Check your internet connection and free disk space, then retry.\n{error}"
        ) from None
    print("Model download complete.")
    return path


def load_settings(path=DEFAULT_SETTINGS):
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    keys = {"model", "language", "task", "output_format"}
    optional = {"device", "word_timestamps", "normalize_audio", "decoding_preset",
                "condition_on_previous_text", "hallucination_filter", "vocabulary", "glossary_path"}
    if not isinstance(settings, dict) or settings.keys() - optional != keys:
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
    # Skip alignment work only when the output cannot carry word timing.
    settings["word_timestamps"] = word_timestamps and settings["output_format"] in ("srt", "vtt", "json", "all")
    for name in ("normalize_audio", "condition_on_previous_text", "hallucination_filter"):
        if name in settings and not isinstance(settings[name], bool):
            raise ValueError(f"Set {name} to true or false.")
    if settings.get("decoding_preset", "standard") not in ("standard", "accuracy"):
        raise ValueError("Set decoding_preset to 'standard' or 'accuracy'.")
    if not isinstance(settings.get("vocabulary", ""), str):
        raise ValueError("Set vocabulary to text containing names or short phrases.")
    glossary = settings.get("glossary_path")
    if glossary is not None and not isinstance(glossary, str):
        raise ValueError("Set glossary_path to a text-file path or null.")
    # Read the glossary before loading a model; relative paths always refer to the launcher.
    settings["vocabulary_terms"] = vocabulary_terms(settings)
    return settings


def vocabulary_terms(settings):
    """Keep user names first, then unique glossary phrases, without treating comments as hints."""
    entries = re.split(r"[,;\n]+", settings.get("vocabulary", ""))
    glossary = settings.get("glossary_path")
    if glossary:
        path = Path(glossary)
        if not path.is_absolute():
            path = ROOT / path
        if path.stat().st_size > 65536:
            raise ValueError("Keep the glossary under 64 KiB; short, relevant vocabulary works best.")
        entries.extend(line for line in path.read_text(encoding="utf-8-sig").splitlines()
                       if not line.lstrip().startswith("#"))
    terms = {}
    for entry in entries:
        phrase = unicodedata.normalize("NFC", " ".join(entry.split()))
        if phrase:
            terms.setdefault(phrase.casefold(), phrase)
    return list(terms.values())


def vocabulary_prompt(settings, encode):
    """Fit whole phrases to both backends' context budget instead of silently losing names."""
    terms = settings.get("vocabulary_terms")
    if terms is None:
        terms = vocabulary_terms(settings)
    selected = []
    for term in terms:
        candidate = ", ".join([*selected, term])
        if len(encode(" " + candidate)) <= PROMPT_TOKEN_LIMIT:
            selected.append(term)
    if len(selected) < len(terms):
        print(f"Vocabulary: using {len(selected)} of {len(terms)} phrases within the {PROMPT_TOKEN_LIMIT}-token limit. Put important names first.")
    elif selected:
        print(f"Vocabulary: using {len(selected)} names and phrases throughout the recording.")
    return ", ".join(selected) or None


def decoding_options(settings):
    """Keep legacy presets compatible; alignment can also serve the optional silence filter."""
    accuracy = settings.get("decoding_preset", "standard") == "accuracy"
    silence_filter = settings.get("hallucination_filter", False)
    options = {
        "condition_on_previous_text": settings.get(
            "condition_on_previous_text", not accuracy and settings.get("language") != MIXED_LANGUAGES
        ),
        "word_timestamps": settings.get("word_timestamps", False) or silence_filter,
        "hallucination_silence_threshold": 2.0 if silence_filter else None,
    }
    if accuracy:
        # OpenAI's Python API defaults to greedy decoding, unlike faster-whisper.
        options.update(beam_size=5, best_of=5)
    # Leave temperature fallback and quality thresholds at the pinned backends' defaults.
    return options


def limit_prompt_history(model):
    """Share faster-whisper's half-context budget between persistent hints and past text."""
    original = model.get_prompt

    def get_prompt(tokenizer, previous_tokens, **options):
        budget = model.max_length // 2 - 1
        hotwords = options.get("hotwords")
        if hotwords and not options.get("prefix"):
            budget -= min(budget, len(tokenizer.encode(" " + hotwords.strip())))
        # The pinned backend otherwise budgets hotwords and previous text independently.
        previous_tokens = previous_tokens[-budget:] if budget else []
        return original(tokenizer, previous_tokens, **options)

    model.get_prompt = get_prompt


def normalize_region(audio):
    """Raise quiet speech with bounded gain; reduce stationary noise only before boosting it."""
    import numpy as np

    size, hop = 512, 256
    if audio.size < size or not np.any(audio):
        return audio, 1.0, False
    padded = np.pad(audio, (hop, hop + (-audio.size) % hop))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    window = np.hanning(size)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Spectral flatness is only a conservative speech/noise heuristic, not a VAD model.
    band = power[:, 10:225] + 1e-15
    flatness = np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)
    voiced = (rms > max(1e-4, float(rms.max()) * 0.1)) & (flatness < 0.3)
    if np.count_nonzero(voiced) < 3:
        return audio, 1.0, False
    level = float(np.percentile(rms[voiced], 75))
    # Target -20 dBFS RMS, add at most 12 dB, and leave peak headroom without clipping.
    gain = min(0.1 / level, 10 ** (12 / 20), 0.95 / float(np.max(np.abs(audio))))
    if gain < 1.05:
        return audio, 1.0, False

    quiet = (rms < level * 0.25) & (rms > 1e-7) & (flatness > 0.35)
    reduced = False
    cleaned = audio
    # Require at least 250 ms of similar low-energy noise and a consequential boosted floor.
    if np.count_nonzero(quiet) * hop / SAMPLE_RATE >= 0.25:
        floor = float(np.median(rms[quiet]))
        spread = float(np.percentile(rms[quiet], 90) / np.percentile(rms[quiet], 10))
        if floor * gain >= 10 ** (-50 / 20) and level / floor <= 32 and spread <= 2:
            noise_power = np.median(power[quiet], axis=0)
            # A 6 dB attenuation cap protects consonants when the noise estimate is imperfect.
            mask = np.sqrt(np.maximum(0.25, 1 - 0.8 * noise_power / (power + 1e-15)))
            restored = np.fft.irfft(spectrum * mask, n=size, axis=1) * window
            accumulated = np.zeros(padded.size)
            weights = np.zeros(padded.size)
            for index, frame in enumerate(restored):
                offset = index * hop
                accumulated[offset:offset + size] += frame
                weights[offset:offset + size] += window ** 2
            cleaned = (accumulated[hop:hop + audio.size] /
                       weights[hop:hop + audio.size]).astype(np.float32)
            reduced = True
    # Reduction can alter peaks slightly, so check headroom again before applying gain.
    gain = min(gain, 0.95 / max(float(np.max(np.abs(cleaned))), 1e-12))
    if gain <= 1:
        return audio, 1.0, False
    return cleaned * gain, gain, reduced


def prepare_audio(audio, settings):
    """Normalize overlapping regions without moving samples or changing subtitle timestamps."""
    if not settings.get("normalize_audio", False):
        return audio
    import numpy as np

    if not audio.size:
        return audio
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("Normalization requires finite mono audio samples.")
    print("Checking quiet audio and noise before normalization...")
    output = audio.copy()
    region_size = 8 * SAMPLE_RATE
    step = region_size // 2
    max_gain, reduced_regions = 1.0, 0
    for offset in range(-step, audio.size, step):
        start, end = max(0, offset), min(audio.size, offset + region_size)
        region = audio[start:end]
        normalized, gain, reduced = normalize_region(region)
        if gain > 1:
            # Half-overlapping cosine weights sum to one, avoiding gain steps at boundaries.
            position = np.arange(start - offset, end - offset) + 0.5
            weight = (0.5 - 0.5 * np.cos(2 * np.pi * position / region_size)).astype(np.float32)
            output[start:end] += (normalized - region) * weight
            max_gain = max(max_gain, gain)
            reduced_regions += int(reduced)
    print(f"Normalization: maximum gain {20 * math.log10(max_gain):.1f} dB; "
          f"conditional noise reduction in {reduced_regions} overlapping regions.")
    return output


def run_model(audio, settings, device):
    import torch
    import whisper

    hardware = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"Loading Whisper {settings['model']} on {hardware} (downloads the model on first use)...")
    # Keep the checkpoint out of VRAM while the model moves to CUDA.
    model = whisper.load_model(
        settings["model"], device="cpu", download_root=str(ROOT / ".whisper-runner" / "models")
    ).to(device)
    from whisper.tokenizer import get_tokenizer

    prompt = vocabulary_prompt(settings, lambda text: get_tokenizer(model.is_multilingual).encode(text))
    print(f"Running {settings['task']}...")
    return model.transcribe(
        audio,
        language=settings["language"],
        task=settings["task"],
        fp16=device == "cuda",
        verbose=False,
        initial_prompt=prompt,
        carry_initial_prompt=bool(prompt),
        **decoding_options(settings),
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


def run_mixed_model(audio, clips, settings, device, model_path):
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    cuda_libraries = configure_fast_cuda() if device == "cuda" else []
    print(f"Loading faster-whisper {settings['model']} on {device.upper()}...")
    model = WhisperModel(
        model_path,
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
        local_files_only=True,
    )
    prompt = vocabulary_prompt(settings, lambda text: model.hf_tokenizer.encode(text).ids)
    if prompt:
        limit_prompt_history(model)
    print(f"Running mixed-language {settings['task']}...")
    # Re-detect the language for each speech clip; context resets follow the selected preset.
    segments, info = model.transcribe(
        audio,
        task=settings["task"],
        multilingual=True,
        clip_timestamps=clips,
        # Hotwords are supplied in every window, even when previous text is reset.
        hotwords=prompt,
        **decoding_options(settings),
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
    # Download once, independently of CUDA selection and any CPU retry.
    model_path = prepare_mixed_model(settings["model"])

    if preference != "cpu":
        try:
            if not get_cuda_device_count():
                raise RuntimeError("CTranslate2 found no CUDA device.")
            return run_mixed_model(audio, clips, settings, "cuda", model_path)
        except RuntimeError as error:
            if preference == "cuda":
                raise SystemExit(
                    f"faster-whisper could not use CUDA. {error}\n"
                    "Run Run-Whisper.ps1 to repair CUDA 12 cuBLAS and cuDNN 9, or choose CPU."
                ) from None
            print(f"faster-whisper could not use CUDA. {error}\nRetrying with faster-whisper on CPU.")
    return run_mixed_model(audio, clips, settings, "cpu", model_path)


def write_word_srt(segments, file, format_timestamp):
    """Write one cue per aligned word, falling back to segments without word data."""
    index = 0
    for segment in segments:
        for item in segment.get("words") or (segment,):
            text = item.get("word", item.get("text", "")).replace("\r", " ").replace("\n", " ").strip()
            if not text:
                continue
            # Rounding can collapse a very short word; keep every cue at least 1 ms long.
            start = max(0, round(item["start"] * 1000))
            end = max(start + 1, round(item["end"] * 1000))
            cue_start = format_timestamp(start / 1000, always_include_hours=True, decimal_marker=",")
            cue_end = format_timestamp(end / 1000, always_include_hours=True, decimal_marker=",")
            index += 1
            print(f"{index}\n{cue_start} --> {cue_end}\n{text.replace('-->', '->')}\n", file=file)


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
            if extension == "srt" and word_timestamps:
                write_word_srt((asdict(segment) for segment in segments), file, format_timestamp)
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
            print("Audio decoded. Finding speech for mixed-language transcription...")
            segments = infer_mixed(prepare_audio(audio, settings), settings, device)
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
            result = infer(prepare_audio(audio, settings), settings, device)
            # Apply the same word-aware subtitle writers to individual and All formats runs.
            formats = OUTPUT_FORMATS if output_format == "all" else (output_format,)
            for extension in formats:
                if word_timestamps and extension in ("srt", "vtt"):
                    writer = write_word_srt if extension == "srt" else write_word_vtt
                    with (Path(staging) / f"{media.stem}.{extension}").open("w", encoding="utf-8") as file:
                        writer(result["segments"], file, format_timestamp)
                else:
                    get_writer(extension, staging)(result, str(media))

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
    parser.add_argument("media", type=Path, nargs="?")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--open-output", action="store_true")
    parser.add_argument("--check-fast-cuda", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_fast_cuda:
        try:
            cuda_libraries = configure_fast_cuda()
            from ctranslate2 import get_cuda_device_count

            if not get_cuda_device_count():
                raise RuntimeError("CTranslate2 found no CUDA device.")
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
        return
    if args.media is None:
        parser.error("Choose an existing audio or video file.")
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
