"""Run the configured Whisper preset on one local media file."""

import argparse
import json
import os
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import whisper
from whisper.tokenizer import LANGUAGES
from whisper.utils import get_writer


ROOT = Path(__file__).resolve().parent


def load_settings():
    settings = json.loads((ROOT / "whisper-settings.json").read_text(encoding="utf-8-sig"))
    keys = {"model", "language", "task", "output_format"}
    if not isinstance(settings, dict) or settings.keys() - {"device"} != keys:
        raise ValueError("whisper-settings.json requires model, language, task, and output_format.")

    model = settings["model"]
    language = settings["language"]
    task = settings["task"]
    if model not in whisper.available_models():
        raise ValueError(f"Unknown Whisper model {model!r}.")
    if language is not None and (not isinstance(language, str) or language not in LANGUAGES):
        raise ValueError("Set language to null for auto-detection or a Whisper code such as 'en' or 'cs'.")
    if task not in ("transcribe", "translate"):
        raise ValueError("Set task to 'transcribe' or 'translate'.")
    if model.endswith(".en") and (language not in (None, "en") or task != "transcribe"):
        raise ValueError("English-only models support English transcription only.")
    if task == "translate" and model in ("turbo", "large-v3-turbo"):
        raise ValueError("Choose a multilingual model such as 'small' for English translation.")
    if settings["output_format"] not in ("all", "txt", "srt", "vtt", "tsv", "json"):
        raise ValueError("Set output_format to 'all', 'txt', 'srt', 'vtt', 'tsv', or 'json'.")
    return settings


def run_model(audio, settings, device):
    hardware = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"Loading Whisper {settings['model']} on {hardware} (downloads the model on first use)...")
    # Keep the checkpoint out of VRAM while the model moves to CUDA.
    model = whisper.load_model(
        settings["model"], device="cpu", download_root=str(ROOT / ".whisper-runner" / "models")
    ).to(device)
    print(f"Running {settings['task']}...")
    return model.transcribe(
        audio, language=settings["language"], task=settings["task"], fp16=device == "cuda", verbose=False
    )


def infer(audio, settings, preference):
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


def transcribe(media, settings, device):
    # Stage the files next to the input, then publish them together after all writers succeed.
    with TemporaryDirectory(dir=media.parent, prefix=".whisper-") as staging:
        print(f"Reading audio from {media.name}...")
        try:
            audio = whisper.load_audio(str(media))
        except RuntimeError as error:
            detail = "\n".join(str(error).splitlines()[-3:])
            raise SystemExit(
                f"Could not read audio from {media.name}. Choose a playable MP3 or MP4 with audio.\n{detail}"
            ) from error

        result = infer(audio, settings, device)
        get_writer(settings["output_format"], staging)(result, str(media))

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
    parser.add_argument("--open-output", action="store_true")
    args = parser.parse_args()
    media = args.media.absolute()
    if media.suffix.lower() not in (".mp3", ".mp4") or not media.is_file():
        parser.error("Choose an existing MP3 or MP4 file.")

    try:
        settings = load_settings()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Check whisper-settings.json. {error}\n")

    output = transcribe(media, settings, args.device)
    print(f"Saved to {output}")
    if args.open_output:
        try:
            os.startfile(output)
        except OSError as error:
            print(f"Could not open the output folder. {error}")
            input("Press Enter to close.")


if __name__ == "__main__":
    main()
