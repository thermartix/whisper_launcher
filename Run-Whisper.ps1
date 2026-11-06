#Requires -Version 5.1
param([string]$MediaPath)

$ErrorActionPreference = 'Stop'

function Select-MediaFile {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $dialog.Title = 'Select an MP3 or MP4 to transcribe with Whisper'
        $dialog.Filter = 'MP3 and MP4 files (*.mp3;*.mp4)|*.mp3;*.mp4'
        $dialog.CheckFileExists = $true
        $dialog.RestoreDirectory = $true
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            return $dialog.FileName
        }
    } finally {
        $dialog.Dispose()
    }
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

try {
    $interactive = -not $PSBoundParameters.ContainsKey('MediaPath')
    if ($interactive) {
        $MediaPath = Select-MediaFile
        if (-not $MediaPath) { exit 0 }
    }

    $media = Get-Item -LiteralPath $MediaPath
    if ($media -isnot [IO.FileInfo] -or $media.Extension -notin '.mp3', '.mp4') {
        throw 'Choose an existing MP3 or MP4 file.'
    }

    $version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION') -Raw).Trim()
    Write-Host "Whisper Runner $version"
    Write-Host "Selected $($media.FullName)"

    $cache = Join-Path $PSScriptRoot '.whisper-runner'
    [void][IO.Directory]::CreateDirectory($cache)
    try {
        # A second launch must not install into the same venv or write the same model cache.
        $lock = [IO.File]::Open((Join-Path $cache 'runner.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw 'The runner cache could not be locked. Close any other Whisper Runner window and try again.'
    }

    $venv = Join-Path $cache 'venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf) -or -not (Test-PythonCommand $python @('-m', 'pip', '--version'))) {
        $basePython = Find-Python
        if (-not $basePython) {
            $installCode = Install-WinGetPackage 'Python.Python.3.13' @('--scope', 'user', '--architecture', 'x64')
            $basePython = Find-Python
            if (-not $basePython) { throw "64-bit Python 3.13 is unavailable after WinGet setup (exit $installCode). See README.md." }
        }
        Write-Host 'Creating the Python environment...'
        & $basePython -I -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed. See the output above.' }
    }

    if (-not (Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue)) {
        $installCode = Install-WinGetPackage 'Gyan.FFmpeg.Essentials' @('--scope', 'user')
        if (-not (Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue)) {
            throw "FFmpeg is unavailable on PATH after WinGet setup (exit $installCode). See README.md."
        }
    }

    # The CPU PyTorch wheel needs the Microsoft C++ runtime on a clean Windows install.
    $runtimeCheck = "import ctypes; ctypes.CDLL('msvcp140.dll'); ctypes.CDLL('vcruntime140.dll'); ctypes.CDLL('vcruntime140_1.dll')"
    if (-not (Test-PythonCommand $python @('-c', $runtimeCheck))) {
        $installCode = Install-WinGetPackage 'Microsoft.VCRedist.2015+.x64' @('--scope', 'machine')
        if (-not (Test-PythonCommand $python @('-c', $runtimeCheck))) {
            throw "The Microsoft Visual C++ runtime is unavailable after WinGet setup (exit $installCode). See README.md."
        }
    }

    $torchVersion = '2.14.1+cpu'
    $whisperVersion = '20250625'
    $check = "from importlib.metadata import version; assert version('torch') == '$torchVersion' and version('openai-whisper') == '$whisperVersion'"
    if (-not (Test-PythonCommand $python @('-c', $check))) {
        Write-Host 'Installing Whisper and CPU PyTorch. The first setup can take several minutes...'
        & $python -I -m pip --isolated --disable-pip-version-check --no-cache-dir install --index-url https://download.pytorch.org/whl/cpu "torch==$torchVersion"
        if ($LASTEXITCODE -ne 0) { throw 'Installing CPU PyTorch failed. See the output above.' }
        & $python -I -m pip --isolated --disable-pip-version-check --no-cache-dir install --index-url https://pypi.org/simple "openai-whisper==$whisperVersion"
        if ($LASTEXITCODE -ne 0) { throw 'Installing Whisper failed. See the output above.' }
    }

    $arguments = @((Join-Path $PSScriptRoot 'transcribe.py'), $media.FullName)
    if ($interactive) { $arguments += '--open-output' }
    & $python -I -X utf8 -u @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Transcription failed. See the output above.' }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($lock) { $lock.Dispose() }
}
