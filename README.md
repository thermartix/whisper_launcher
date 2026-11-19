# Whisper Launcher for Windows

Double-click **Whisper Launcher.exe** to open the launcher directly. Keep the executable with the other project files. Click **Browse...** or paste the path to an audio or video file. Choose the model, device, source language, task, and output format. Enable **Word-by-word timing** for word-level subtitle or JSON timings, then click **Transcribe**. The status shows elapsed time during setup and transcription, and download and transcription progress refresh in the log. The launcher opens the results in File Explorer when the run finishes.

Choose **File → Exit** to close the launcher once any active transcription has finished. Choose **Help → About...** to see the launcher version and a summary.

The file picker lists common formats, including MP3, WAV, M4A, FLAC, MP4, MKV, MOV, and WebM. Choose **All files** for other formats that FFmpeg can decode. Video files need an audio track.

For `Interview.mp4`, look for `Interview.mp4.whisper` beside the original. Each repeat run gets a numbered folder, such as `Interview.mp4.whisper.2`, so earlier results are preserved. You need write access to the folder containing the media.

## First run

- Use 64-bit Windows 10 or 11 on an Intel or AMD PC. Extract the launcher to a writable folder and keep its files together.
- The launcher uses existing 64-bit Python 3.13 when available. The standard Whisper backend also uses FFmpeg. The launcher installs missing prerequisites with WinGet, included with Microsoft App Installer. [WinGet setup](https://learn.microsoft.com/windows/package-manager/winget/)
- If the Microsoft Visual C++ runtime is missing, WinGet installs it and Windows may prompt for administrator approval.
- The first run with each backend needs internet access to install its packages and download the selected model. The standard Whisper CUDA build takes several GB of disk space. Standard Whisper shares its model cache across its CPU and CUDA environments. An installed environment and cached model can run offline. [OpenAI Whisper](https://github.com/openai/whisper) · [PyTorch](https://pytorch.org/get-started/locally/)

## Settings

The launcher loads its defaults from `whisper-settings.json`. Changes in the form apply to the next run. Click **Save as defaults** to write your selections to the file for future launches. You can also edit the JSON directly. Version 1.7.0 ships with the Spanish Immunotec preset below; older settings files retain their earlier behavior until you enable the new options.

| Setting | Default | Options |
| --- | --- | --- |
| `model` | `large-v3-turbo` | Choose from all model names supported by this Whisper release, including `tiny`, `base`, `small`, `medium`, `large`, `turbo`, and their available variants. English-only models end in `.en`. [Available models](https://github.com/openai/whisper#available-models-and-languages) |
| `device` | `auto` | `auto` tries NVIDIA CUDA and falls back to CPU within the selected backend. `cpu` always uses CPU. `cuda` requires a working NVIDIA GPU and the libraries for that backend. Settings files without `device` use CPU. |
| `language` | `es` | Spanish (`es`) for the supplied preset. **Auto-detect** (`null`) detects one language for the file. Choose another language such as Czech (`cs`), or choose **Mixed languages (faster-whisper)** (`multilingual`) for a recording that switches languages. |
| `task` | `transcribe` | `transcribe` keeps the spoken language. `translate` produces English and requires a multilingual model such as `small`. |
| `output_format` | `srt` | `srt` writes subtitles. Set `txt`, `vtt`, `tsv`, or `json` for a different format, or `all` for all five. |
| `word_timestamps` | `true` | Enable **Word-by-word timing** for **SubRip subtitles (.srt)**, **WebVTT subtitles (.vtt)**, **JSON (.json)**, or **All formats**. SRT writes one subtitle cue per word using its start and end times. WebVTT adds inline timestamps for word starts within subtitle cues. JSON adds a `words` list to each segment, with a start and end time in seconds for each word. |
| `normalize_audio` | `true` | Raise quiet speech conservatively before either backend, with conditional noise checks as described below. |
| `decoding_preset` | `accuracy` | **Accuracy** uses five beams and five sampling candidates during temperature fallback. **Standard** preserves each backend's default decoding search. |
| `condition_on_previous_text` | `false` | **Use previous text as context** controls continuity across windows. Off reduces the risk of repetition loops; on can improve continuity. Selecting Accuracy turns it off, and you can override the checkbox. |
| `hallucination_filter` | `true` | **Filter possible hallucinations near long silences** enables the backend's word-alignment-based silence heuristic with a 2-second threshold. |
| `vocabulary` | `Ainhoa, Tzeitel` | Editable names and short phrases, separated by commas, semicolons or new lines. These take priority over the glossary. |
| `glossary_path` | `vocabulary/immunotec-es.txt` | Optional UTF-8 text file containing one phrase per line. Clear the field or use `null` to disable it. Relative paths refer to the launcher folder, including when run settings are stored in a temporary file. |

**Word-by-word timing** stays visible below **Output format**, including with the default SRT format. It works with both standard Whisper and mixed-language faster-whisper. The checkbox is disabled for TXT and TSV, which do not use word timings; switching formats keeps your selection. Click **Save as defaults** to remember it. Word alignment can increase transcription time, and its estimated times may need adjustment.

To run from PowerShell with the saved defaults, use `& .\Run-Whisper.ps1 -MediaPath 'C:\Media\Interview.mp4'`. Add `-OpenOutput` to open the output folder automatically.

## Accuracy, normalization and vocabulary

The **Accuracy** preset applies to both backends. It keeps the existing temperature fallback and standard no-speech, compression and probability thresholds. The silence filter is a separate checkbox and computes word alignment even if your output format does not use word timings. Filtering can omit quiet speech; compare a representative excerpt with the filter on and off. No particular WER improvement is guaranteed without a corrected reference transcript. The preset does not change your selected model; compare `large-v3` and `large-v2` against turbo on your own recordings if needed. [OpenAI transcription options](https://github.com/openai/whisper/blob/v20250625/whisper/transcribe.py)

**Normalize quiet speech** processes overlapping 8-second regions, targets approximately -20 dBFS speech RMS, adds at most 12 dB, and protects peaks at 0.95 full scale. Silence, stationary noise without detected speech, and already loud regions are left alone. Smooth region blending preserves the sample count and original timeline; the source media is never modified. This is a conservative signal-level heuristic, not loudness mastering or a speech enhancement model.

There is no general denoising pass. Only regions that will be amplified are checked for a consequential stationary broadband noise floor, estimated from at least 250 ms of quieter frames. When that check succeeds, light spectral reduction (at most 6 dB per frequency bin) is applied before gain. Otherwise the region receives gain alone. The log reports the maximum gain and the number of overlapping regions receiving conditional noise reduction. Turning normalization off bypasses both steps. This heuristic may miss changing noise or overlapping voices; use the original audio to assess difficult passages.

Enter recurring names in **Names / vocabulary**, for example `Ainhoa, Tzeitel`. Edit `vocabulary/immunotec-es.txt` in a text editor to add or remove domain terms. Its starter list includes Immunotec product spellings and Spanish compensation-plan terminology, including BDN, VVP, VGP, VVC, VTOC, commissions, bonuses and ranks. It is vocabulary, not a description of current compensation rules. Sources: [Immunotec product catalog](https://www.immunotec.com/en-US/MVP/products) and [Spanish compensation-plan terminology](https://cdn.immunotec.com/docs/WK/SP/ES/CompPlan-ES.pdf).

Glossary lines beginning with `#` are comments; blank lines and duplicate phrases are ignored. User names come first. Both backends fit whole phrases into a 200-token hint budget; if the list is too long, the log reports omitted phrases. Keep important terms near the top and clear the glossary for unrelated recordings. Standard Whisper carries the vocabulary prompt into every decoding window; mixed-language faster-whisper supplies it as persistent hotwords, independently of previous-text conditioning. Hints improve the chance of the desired spelling but do not force names into the output or replace recognized words after transcription.

## Mixed-language recordings

Choose **Mixed languages (faster-whisper)** under **Source language** and select a multilingual model such as `small` or `medium`. With **Transcribe**, the launcher splits speech near pauses so faster-whisper can detect the language for each part and write the spoken languages to the same transcript or subtitle file. Short switches within a sentence may be harder to recognize. **Translate to English** and all output formats are available with compatible models. [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

The first mixed-language run installs a separate environment in `.whisper-runner/venv-fast` and downloads CTranslate2 model files to `.whisper-runner/models-fast`. PyAV supplies the audio decoder for both audio and video files in this mode.

Model downloads are shown separately from loading and transcription. A first download can take several minutes, especially for `medium` and larger models. Retrying reuses the download cache; a complete cached model loads without contacting Hugging Face. An incomplete snapshot is downloaded before loading. Public models do not require a Hugging Face account or token; the unauthenticated-request warning concerns download rate limits. Windows caching works without symlinks, administrator rights, or Developer Mode. [Hugging Face cache limitations](https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations)

## NVIDIA GPU setup

Standard Whisper uses the PyTorch CUDA 13.0 build. It requires a Turing-generation or newer NVIDIA GPU and Windows driver 580.88 or newer. The launcher installs the PyTorch build in its own environment using your installed NVIDIA driver. It uses the CPU to read and decode media with FFmpeg and the GPU for Whisper inference. [PyTorch CUDA support](https://pytorch.org/blog/pytorch-2-12-release-blog/)

Faster-whisper uses CTranslate2. For CUDA in mixed-language mode, the launcher checks for CUDA 12 cuBLAS and cuDNN 9. If they are missing, it installs NVIDIA's Windows packages (`nvidia-cublas-cu12==12.9.1.4` and `nvidia-cudnn-cu12==9.10.2.21`) in `venv-fast`, requiring about 1.2 GB of downloads and additional disk space. Their DLL directories are enabled only in the runner process. No system CUDA installation or permanent `PATH` change is needed. A working NVIDIA driver is still required. **Automatic** uses CPU if GPU setup fails and retries on CPU if inference raises a CUDA error. **NVIDIA CUDA** reports a setup error instead. [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)

OpenAI estimates about 2 GB of VRAM for `small`, 5 GB for `medium`, 6 GB for `turbo`, and 10 GB for `large`. Available VRAM varies with other open apps. If `auto` falls back to CPU due to memory, try a smaller model. [Whisper model sizes](https://github.com/openai/whisper#available-models-and-languages)

## Troubleshooting

- If the log stops at model loading on an older launcher, it may be downloading silently. Version 1.6.2 shows model download progress and elapsed time. Close the older launcher after its active run ends, then reopen the updated executable. If a run fails, retry to resume its cached download.
- If WinGet is unavailable, install 64-bit Python 3.13 and FFmpeg manually and add both to `PATH`. Install the x64 Microsoft Visual C++ Redistributable if the launcher reports it is missing, then reopen the launcher. [Python](https://www.python.org/downloads/windows/) · [FFmpeg](https://ffmpeg.org/download.html#build-windows) · [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
- If a run fails, check the launcher log for details and click **Transcribe** to retry. If the local environment is damaged or the launcher was moved, close the launcher and delete the affected environment. The launcher recreates it on the next run. The folders are `.whisper-runner/venv` for standard CPU, `.whisper-runner/venv-cuda` for standard CUDA, and `.whisper-runner/venv-fast` for mixed-language mode.

## Rebuilding the executable

Run `& .\Build-Launcher.ps1` from PowerShell to rebuild **Whisper Launcher.exe** using the Windows .NET Framework compiler. The build reads `VERSION` for the executable's file version.

After installing the mixed-language environment, run regression checks with `& .\.whisper-runner\venv-fast\Scripts\python.exe -I -m unittest discover -s tests -v` and `& .\tests\Test-FasterSetup.ps1`. These cover complete and interrupted model caches, visible download progress, download failures, CUDA setup, and CPU fallback without downloading models.

Run `& .\.whisper-runner\venv\Scripts\python.exe -I -m unittest discover -s tests -p test_word_timestamps.py -v` to check the standard Whisper writers too. The word-timing checks cover both backends' model options, SRT word cues, WebVTT inline timestamps, JSON word data, All formats, and compatibility with older settings. Each environment skips the other backend's writer tests.

Run `& .\.whisper-runner\venv\Scripts\python.exe -I -m unittest discover -s tests -p test_accuracy.py -v` for audio and quality checks in the standard environment. The faster environment's full suite includes these checks too. They cover bounded gain, conditional noise reduction, sample-count preservation, glossary encoding and priority, prompt limits, legacy settings, accuracy options, and vocabulary delivery to both backends without model downloads. They do not measure WER.

Run `& .\tests\Test-LauncherSettings.ps1` in Windows PowerShell to check the native form's settings, legacy defaults, Unicode saves and layout. It uses temporary settings and creates the form offscreen. Add `-RenderPath 'C:\dev\Whisper\.whisper-runner\launcher.png'` to render the default form for visual inspection.
