#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# Load only the setup function so these checks never install packages or launch the UI.
$tokens = $null
$errors = $null
$script:RunnerRoot = Split-Path $PSScriptRoot
$runner = Join-Path $script:RunnerRoot 'Run-Whisper.ps1'
$ast = [Management.Automation.Language.Parser]::ParseFile($runner, [ref]$tokens, [ref]$errors)
if ($errors) { throw ($errors | Out-String) }
$definition = $ast.Find({ param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-FasterCuda'
}, $true)
# Script roots are empty in dynamically evaluated functions; retain the real runner location.
Invoke-Expression ($definition.Extent.Text.Replace('$PSScriptRoot', '$script:RunnerRoot'))

function Test-PythonCommand {
    param($Python, $Arguments)
    return $script:Checks.Dequeue()
}

function Invoke-FakePython {
    $script:Installs++
    $global:LASTEXITCODE = $script:InstallExitCode
}

function Assert-Setup($Preference, [bool[]]$Checks, $Expected, $ExpectedInstalls, $InstallExitCode = 0) {
    $script:Checks = [Collections.Generic.Queue[bool]]::new()
    foreach ($value in $Checks) { $script:Checks.Enqueue($value) }
    $script:Installs = 0
    $script:InstallExitCode = $InstallExitCode
    $actual = try { Initialize-FasterCuda 'Invoke-FakePython' $Preference } catch { 'error' }
    if ($actual -ne $Expected -or $script:Installs -ne $ExpectedInstalls) {
        throw "Preference $Preference expected $Expected / $ExpectedInstalls installs, got $actual / $script:Installs."
    }
}

Assert-Setup 'cpu' @() 'cpu' 0
Assert-Setup 'auto' @($false) 'cpu' 0
Assert-Setup 'cuda' @($false) 'error' 0
Assert-Setup 'auto' @($true, $true) 'cuda' 0
Assert-Setup 'auto' @($true, $false, $true) 'cuda' 1
Assert-Setup 'auto' @($true, $false) 'cpu' 1 1
Assert-Setup 'cuda' @($true, $false) 'error' 1 1
Assert-Setup 'auto' @($true, $false, $false) 'cpu' 1
Write-Output 'Passed 8 faster-whisper CUDA setup checks.'
