#Requires -Version 5.1
param(
    [string]$MediaPath,
    [string]$SettingsPath = (Join-Path $PSScriptRoot 'whisper-settings.json'),
    [switch]$OpenOutput
)

$ErrorActionPreference = 'Stop'

if (-not $PSBoundParameters.ContainsKey('MediaPath')) {
    try {
        & (Join-Path $PSScriptRoot 'Whisper-Launcher.ps1') -SettingsPath $SettingsPath
    } catch {
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Whisper Launcher', 'OK', 'Error')
        exit 1
    }
    exit
}

function Find-Python {
    # Skip Store placeholder aliases and verify the interpreter before creating the venv.
    $ErrorActionPreference = 'SilentlyContinue'
    $probe = "import platform, sys; assert sys.version_info[:2] == (3, 13) and platform.machine() == 'AMD64' and sys.maxsize > 2**32; print(sys.executable)"
    $launcher = Get-Command py.exe -CommandType Application | Select-Object -First 1
    if ($launcher) {
        $path = & $launcher.Source -3.13 -I -c $probe 2>$null
        if ($LASTEXITCODE -eq 0 -and $path) { return $path }
    }

    $standardPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    $pathPythons = Get-Command python.exe -CommandType Application | Select-Object -ExpandProperty Source
    foreach ($candidate in (@($standardPath) + $pathPythons)) {
        if ($candidate -and $candidate -notlike '*\WindowsApps\*' -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $path = & $candidate -I -c $probe 2>$null
            if ($LASTEXITCODE -eq 0 -and $path) { return $path }
        }
    }
}

function Install-WinGetPackage([string]$Id, [string[]]$Options) {
    $winget = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $winget) {
        throw "WinGet is required to install $Id. Install Microsoft App Installer or follow the manual setup in README.md."
    }

    Write-Host "Installing $Id with WinGet..."
    & $winget.Source install --id $Id --exact --source winget --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity @Options | Out-Host
    $installCode = $LASTEXITCODE

    # WinGet can update the registry PATH while this PowerShell process is already running.
    $env:Path = @(
        $env:Path
        [Environment]::GetEnvironmentVariable('Path', 'Machine')
        [Environment]::GetEnvironmentVariable('Path', 'User')
    ) -join ';'
    return $installCode
}

function Test-PythonCommand([string]$Python, [string[]]$Arguments) {
    $ErrorActionPreference = 'SilentlyContinue'
    & $Python -I @Arguments *> $null
    return $LASTEXITCODE -eq 0
}

function Test-SupportedNvidiaGpu {
    $ErrorActionPreference = 'SilentlyContinue'
    $nvidiaSmi = Get-Command nvidia-smi.exe -CommandType Application | Select-Object -First 1
    if (-not $nvidiaSmi) { return $false }

    # Check the CUDA 13 driver and GPU generation before downloading the larger PyTorch build.
    $gpus = & $nvidiaSmi.Source '--query-gpu=driver_version,compute_cap' '--format=csv,noheader,nounits' 2>$null
    if ($LASTEXITCODE -eq 0) {
        foreach ($gpu in $gpus) {
            if ($gpu -match '^\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*$' -and
                [version]$Matches[1] -ge [version]'580.88' -and [version]$Matches[2] -ge [version]'7.5') {
                return $true
            }
        }
    }
    return $false
}

function Get-RunnerPython([string]$Cache, [string]$Build) {
    # Keep CTranslate2 and each PyTorch build in separate environments.
    $venvName = switch ($Build) {
        'faster-whisper' { 'venv-fast' }
        'cpu' { 'venv' }
        default { 'venv-cuda' }
    }
    $venv = Join-Path $Cache $venvName
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf) -or -not (Test-PythonCommand $python @('-m', 'pip', '--version'))) {
        $basePython = Find-Python
        if (-not $basePython) {
            $installCode = Install-WinGetPackage 'Python.Python.3.13' @('--scope', 'user', '--architecture', 'x64')
            $basePython = Find-Python
            if (-not $basePython) { throw "64-bit Python 3.13 is unavailable after WinGet setup (exit $installCode). See README.md." }
        }
        Write-Host 'Creating the Python environment...'
        & $basePython -I -m venv $venv | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed. See the output above.' }
    }

    # The native Python packages need the Microsoft C++ runtime on a clean Windows install.
    $runtimeCheck = "import ctypes; ctypes.CDLL('msvcp140.dll'); ctypes.CDLL('vcruntime140.dll'); ctypes.CDLL('vcruntime140_1.dll')"
    if (-not (Test-PythonCommand $python @('-c', $runtimeCheck))) {
        $installCode = Install-WinGetPackage 'Microsoft.VCRedist.2015+.x64' @('--scope', 'machine')
        if (-not (Test-PythonCommand $python @('-c', $runtimeCheck))) {
            throw "The Microsoft Visual C++ runtime is unavailable after WinGet setup (exit $installCode). See README.md."
        }
    }

    if ($Build -eq 'faster-whisper') {
        $fasterVersion = '1.2.1'
        $ct2Version = '4.8.5'
        # Faster-whisper 1.2.1 passes a metadata option that PyAV 19 removed.
        $avVersion = '18.0.0'
        $check = "from importlib.metadata import version; assert version('faster-whisper') == '$fasterVersion' and version('ctranslate2') == '$ct2Version' and version('av') == '$avVersion'; import faster_whisper"
        if (-not (Test-PythonCommand $python @('-c', $check))) {
            Write-Host 'Installing faster-whisper for mixed-language transcription...'
            & $python -I -m pip --isolated --disable-pip-version-check --no-cache-dir install --index-url https://pypi.org/simple "faster-whisper==$fasterVersion" "ctranslate2==$ct2Version" "av==$avVersion" | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-PythonCommand $python @('-c', $check))) {
                throw 'Setting up faster-whisper failed. See the output above.'
            }
        }
        return $python
    }

    $torchVersion = "2.14.1+$Build"
    # Update the launcher's model and language catalog when changing the Whisper release.
    $whisperVersion = '20250625'
    $check = "from importlib.metadata import version; assert version('torch') == '$torchVersion' and version('openai-whisper') == '$whisperVersion'"
    if (-not (Test-PythonCommand $python @('-c', $check))) {
        Write-Host "Installing Whisper and PyTorch ($Build). The first setup can take several minutes..."
        & $python -I -m pip --isolated --disable-pip-version-check --no-cache-dir install --index-url "https://download.pytorch.org/whl/$Build" "torch==$torchVersion" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch ($Build) failed. See the output above." }
        & $python -I -m pip --isolated --disable-pip-version-check --no-cache-dir install --index-url https://pypi.org/simple "openai-whisper==$whisperVersion" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Installing Whisper failed. See the output above.' }
    }
    return $python
}

try {
    if ([Console]::IsOutputRedirected) {
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    }

    $media = Get-Item -LiteralPath $MediaPath
    if ($media -isnot [IO.FileInfo]) {
        throw 'Choose an existing audio or video file.'
    }

    $version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION') -Raw).Trim()
    Write-Host "Whisper Launcher $version"
    Write-Host "Selected $($media.FullName)"

    $settings = Get-Content -LiteralPath $SettingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $device = 'cpu'
    if ($settings -is [pscustomobject] -and $settings.PSObject.Properties.Name -ccontains 'device') {
        $device = $settings.device
        if ($device -isnot [string] -or $device -cnotin 'auto', 'cpu', 'cuda') {
            throw "Set device in whisper-settings.json to 'auto', 'cpu', or 'cuda'."
        }
    }

    $cache = Join-Path $PSScriptRoot '.whisper-runner'
    [void][IO.Directory]::CreateDirectory($cache)
    try {
        # A second launch must not install into the same venv or write the same model cache.
        $lock = [IO.File]::Open((Join-Path $cache 'runner.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw 'The launcher cache could not be locked. Close any other Whisper Launcher window and try again.'
    }

    if ($settings.language -ceq 'multilingual') {
        # PyAV supplies audio decoding for this backend, including video files.
        $python = Get-RunnerPython $cache 'faster-whisper'
    } else {
        if (-not (Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue)) {
            $installCode = Install-WinGetPackage 'Gyan.FFmpeg.Essentials' @('--scope', 'user')
            if (-not (Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue)) {
                throw "FFmpeg is unavailable on PATH after WinGet setup (exit $installCode). See README.md."
            }
        }

        $python = $null
        if ($device -ne 'cpu') {
            if (Test-SupportedNvidiaGpu) {
                try {
                    $python = Get-RunnerPython $cache 'cu130'
                    # Verify the wheel can execute on the GPU PyTorch selected.
                    if (-not (Test-PythonCommand $python @('-c', "import torch; assert torch.ones(1, device='cuda').item() == 1"))) {
                        throw 'PyTorch could not run a CUDA operation. Check the NVIDIA driver and GPU availability.'
                    }
                } catch {
                    if ($device -eq 'cuda') { throw }
                    Write-Warning "CUDA setup failed. Using CPU. $($_.Exception.Message)"
                    $python = $null
                }
            } elseif ($device -eq 'cuda') {
                throw 'CUDA requires a supported NVIDIA GPU and driver 580.88 or newer. See README.md.'
            }
        }
        if (-not $python) {
            $device = 'cpu'
            $python = Get-RunnerPython $cache 'cpu'
        }
    }

    $arguments = @((Join-Path $PSScriptRoot 'transcribe.py'), $media.FullName, '--device', $device, '--settings', $SettingsPath)
    if ($OpenOutput) { $arguments += '--open-output' }
    & $python -I -X utf8 -u @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Transcription failed. See the output above.' }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($lock) { $lock.Dispose() }
}
