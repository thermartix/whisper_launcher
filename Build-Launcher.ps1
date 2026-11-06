#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$compiler = Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION') -Raw).Trim()
$staging = Join-Path ([IO.Path]::GetTempPath()) ('whisper-launcher-' + [guid]::NewGuid())
$assemblyInfo = Join-Path $staging 'AssemblyInfo.cs'
$executable = Join-Path $staging 'Whisper Launcher.exe'
[void][IO.Directory]::CreateDirectory($staging)

try {
    # Use the same version for the About dialog and the executable's Windows file details.
    [IO.File]::WriteAllText($assemblyInfo, @"
using System.Reflection;

[assembly: AssemblyTitle("Whisper Launcher")]
[assembly: AssemblyProduct("Whisper Launcher")]
[assembly: AssemblyVersion("${version}.0")]
[assembly: AssemblyFileVersion("${version}.0")]
"@)
    & $compiler /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll `
        "/out:$executable" (Join-Path $PSScriptRoot 'WhisperLauncher.cs') $assemblyInfo
    if ($LASTEXITCODE -ne 0) { throw 'Building Whisper Launcher.exe failed.' }
    Copy-Item -LiteralPath $executable -Destination $PSScriptRoot -Force
} finally {
    [IO.File]::Delete($assemblyInfo)
    [IO.File]::Delete($executable)
    [IO.Directory]::Delete($staging)
}
