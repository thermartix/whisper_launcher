# Whisper Launcher for Windows

Double-click **Whisper Launcher.exe** to open the launcher directly. Keep the executable with the other project files. Click **Browse...** or paste the path to an audio or video file. Choose the model, device, source language, task, and output format, then click **Transcribe**. The window shows setup and transcription progress and opens the results in File Explorer when the run finishes.

Choose **File → Exit** to close the launcher once any active transcription has finished. Choose **Help → About...** to see the launcher version and a summary.

The file picker lists common formats, including MP3, WAV, M4A, FLAC, MP4, MKV, MOV, and WebM. Choose **All files** for other formats that FFmpeg can decode. Video files need an audio track.

For `Interview.mp4`, look for `Interview.mp4.whisper` beside the original. Each repeat run gets a numbered folder, such as `Interview.mp4.whisper.2`, so earlier results are preserved. You need write access to the folder containing the media.

## First run

- Use 64-bit Windows 10 or 11 on an Intel or AMD PC. Extract the launcher to a writable folder and keep its files together.
- The launcher uses existing 64-bit Python 3.13 when available. The standard Whisper backend also uses FFmpeg. The launcher installs missing prerequisites with WinGet, included with Microsoft App Installer. [WinGet setup](https://learn.microsoft.com/windows/package-manager/winget/)
- If the Microsoft Visual C++ runtime is missing, WinGet installs it and Windows may prompt for administrator approval.
- The first run with each backend needs internet access to install its packages and download the selected model. The standard Whisper CUDA build takes several GB of disk space. Standard Whisper shares its model cache across its CPU and CUDA environments. An installed environment and cached model can run offline. [OpenAI Whisper](https://github.com/openai/whisper) · [PyTorch](https://pytorch.org/get-started/locally/)

## Settings

The launcher loads its defaults from `whisper-settings.json`. Changes in the form apply to the next run. Click **Save as defaults** to write your selections to the file for future launches. You can also edit the JSON directly.

| Setting | Default | Options |
| --- | --- | --- |
| `model` | `medium` | Choose from all model names supported by this Whisper release, including `tiny`, `base`, `small`, `medium`, `large`, `turbo`, and their available variants. English-only models end in `.en`. [Available models](https://github.com/openai/whisper#available-models-and-languages) |
| `device` | `auto` | `auto` tries NVIDIA CUDA and falls back to CPU within the selected backend. `cpu` always uses CPU. `cuda` requires a working NVIDIA GPU and the libraries for that backend. Settings files without `device` use CPU. |
| `language` | `null` | **Auto-detect** (`null`) detects one language for the file. Choose a specific language such as Czech (`cs`), or choose **Mixed languages (faster-whisper)** (`multilingual`) for a recording that switches languages. |
| `task` | `transcribe` | `transcribe` keeps the spoken language. `translate` produces English and requires a multilingual model such as `small`. |
| `output_format` | `srt` | `srt` writes subtitles. Set `txt`, `vtt`, `tsv`, or `json` for a different format, or `all` for all five. |

To run from PowerShell with the saved defaults, use `& .\Run-Whisper.ps1 -MediaPath 'C:\Media\Interview.mp4'`. Add `-OpenOutput` to open the output folder automatically.

## Mixed-language recordings

Choose **Mixed languages (faster-whisper)** under **Source language** and select a multilingual model such as `small` or `medium`. With **Transcribe**, the launcher splits speech near pauses so faster-whisper can detect the language for each part and write the spoken languages to the same transcript or subtitle file. Short switches within a sentence may be harder to recognize. **Translate to English** and all output formats are available with compatible models. [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

The first mixed-language run installs a separate environment in `.whisper-runner/venv-fast` and downloads CTranslate2 model files to `.whisper-runner/models-fast`. PyAV supplies the audio decoder for both audio and video files in this mode.

## NVIDIA GPU setup

Standard Whisper uses the PyTorch CUDA 13.0 build. It requires a Turing-generation or newer NVIDIA GPU and Windows driver 580.88 or newer. The launcher installs the PyTorch build in its own environment using your installed NVIDIA driver. It uses the CPU to read and decode media with FFmpeg and the GPU for Whisper inference. [PyTorch CUDA support](https://pytorch.org/blog/pytorch-2-12-release-blog/)

Faster-whisper uses CTranslate2. For CUDA in mixed-language mode, install CUDA 12 cuBLAS and cuDNN 9 for CUDA 12 and make their DLLs available on `PATH`. **Automatic** retries on CPU if faster-whisper cannot use CUDA. [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)

OpenAI estimates about 2 GB of VRAM for `small`, 5 GB for `medium`, 6 GB for `turbo`, and 10 GB for `large`. Available VRAM varies with other open apps. If `auto` falls back to CPU due to memory, try a smaller model. [Whisper model sizes](https://github.com/openai/whisper#available-models-and-languages)

## Troubleshooting

- If WinGet is unavailable, install 64-bit Python 3.13 and FFmpeg manually and add both to `PATH`. Install the x64 Microsoft Visual C++ Redistributable if the launcher reports it is missing, then reopen the launcher. [Python](https://www.python.org/downloads/windows/) · [FFmpeg](https://ffmpeg.org/download.html#build-windows) · [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
- If a run fails, check the launcher log for details and click **Transcribe** to retry. If the local environment is damaged or the launcher was moved, close the launcher and delete the affected environment. The launcher recreates it on the next run. The folders are `.whisper-runner/venv` for standard CPU, `.whisper-runner/venv-cuda` for standard CUDA, and `.whisper-runner/venv-fast` for mixed-language mode.

## Rebuilding the executable

Run `& .\Build-Launcher.ps1` from PowerShell to rebuild **Whisper Launcher.exe** using the Windows .NET Framework compiler. The build reads `VERSION` for the executable's file version.
