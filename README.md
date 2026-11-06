# Whisper Runner for Windows

Double-click **Run Whisper.cmd** and choose an MP3 or MP4. Whisper processes the audio locally using the settings in `whisper-settings.json`. The console shows setup and transcription progress, and the runner opens the results in File Explorer when it finishes.

For `Interview.mp4`, look for `Interview.mp4.whisper` beside the original. Each repeat run gets a numbered folder, such as `Interview.mp4.whisper.2`, so earlier results are preserved. You need write access to the folder containing the media.

## First run

- Use 64-bit Windows 10 or 11 on an Intel or AMD PC. Extract the runner to a writable folder and keep its files together.
- The runner uses existing 64-bit Python 3.13 and FFmpeg when available. Otherwise it installs them for your Windows user with WinGet, included with Microsoft App Installer. [WinGet setup](https://learn.microsoft.com/windows/package-manager/winget/)
- If the Microsoft Visual C++ runtime required by PyTorch is missing, WinGet installs it and Windows may prompt for administrator approval.
- The first run needs internet access to install Whisper and PyTorch and to download the selected model. The CUDA build takes several GB of disk space. The runner caches the CPU and CUDA environments separately and shares the downloaded models between them. An installed environment and cached model can run offline. [OpenAI Whisper](https://github.com/openai/whisper) · [PyTorch](https://pytorch.org/get-started/locally/)

## Settings

Edit `whisper-settings.json` before opening the runner to change the preset.

| Setting | Default | Options |
| --- | --- | --- |
| `model` | `medium` | `small` needs less memory and works well as a faster everyday option. `base` is lighter still. [Available models](https://github.com/openai/whisper#available-models-and-languages) |
| `device` | `auto` | `auto` tries NVIDIA CUDA and uses CPU if CUDA setup fails or the GPU runs out of memory. `cpu` always uses CPU. `cuda` requires a working NVIDIA GPU and reports errors if it cannot use it. Settings files without `device` use CPU. |
| `language` | `null` | `null` detects the spoken language. Use a language code such as `"en"` or `"cs"` to set it explicitly. |
| `task` | `transcribe` | `transcribe` keeps the spoken language. `translate` produces English and requires a multilingual model such as `small`. |
| `output_format` | `srt` | `srt` writes subtitles. Set `txt`, `vtt`, `tsv`, or `json` for a different format, or `all` for all five. |

## NVIDIA GPU setup

CUDA mode uses the PyTorch CUDA 13.0 build. It requires a Turing-generation or newer NVIDIA GPU and Windows driver 580.88 or newer. The runner installs the PyTorch build in its own environment using your installed NVIDIA driver. It uses the CPU to read and decode media with FFmpeg and the GPU for Whisper inference. [PyTorch CUDA support](https://pytorch.org/blog/pytorch-2-12-release-blog/)

OpenAI estimates about 2 GB of VRAM for `small`, 5 GB for `medium`, 6 GB for `turbo`, and 10 GB for `large`. Available VRAM varies with other open apps. If `auto` falls back to CPU due to memory, try a smaller model. [Whisper model sizes](https://github.com/openai/whisper#available-models-and-languages)

## Troubleshooting

- If WinGet is unavailable, install 64-bit Python 3.13 and FFmpeg manually and add both to `PATH`. Install the x64 Microsoft Visual C++ Redistributable if the runner reports it is missing, then reopen the runner. [Python](https://www.python.org/downloads/windows/) · [FFmpeg](https://ffmpeg.org/download.html#build-windows) · [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
- The runner leaves the console open on errors. If setup was interrupted, reopen the runner to retry. If the local environment is damaged or the runner was moved, close the runner and delete `.whisper-runner/venv` for CPU or `.whisper-runner/venv-cuda` for CUDA to rebuild it on the next launch.

The runner version is recorded in `VERSION`.
