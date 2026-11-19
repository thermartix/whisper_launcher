#Requires -Version 5.1
param([string]$RenderPath)
$ErrorActionPreference = 'Stop'

# Exercise real WinForms controls offscreen, without starting a runner or modifying defaults.
$script:LauncherRoot = Split-Path $PSScriptRoot
$source = [IO.File]::ReadAllText((Join-Path $script:LauncherRoot 'Whisper-Launcher.ps1'))
$tokens = $null
$parseErrors = $null
[void][Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors) { throw ($parseErrors | Out-String) }
$source = $source.Replace('$PSScriptRoot', '$script:LauncherRoot').Replace('[void]$form.ShowDialog()', '& $script:AssertLauncher')
$script:SettingsCase = 'current'
$script:RenderPath = $RenderPath
$script:AssertLauncher = {
    # Showing offscreen creates child handles so layout and bitmap rendering are realistic.
    $form.StartPosition = 'Manual'
    $form.Location = [Drawing.Point]::new(-32000, -32000)
    $form.ShowInTaskbar = $false
    $form.Show()
    $form.PerformLayout()
    $layout.PerformLayout()
    $selected = Get-SelectedSettings
    if ($script:SettingsCase -eq 'current') {
        if ($selected.decoding_preset -ne 'accuracy' -or -not $selected.normalize_audio -or
            $selected.condition_on_previous_text -or -not $selected.hallucination_filter -or
            $selected.vocabulary -ne 'Ainhoa, Tzeitel' -or $selected.glossary_path -ne 'vocabulary/immunotec-es.txt') {
            throw 'The form did not restore the supplied accuracy and vocabulary settings.'
        }
        if ($log.Height -lt 100) { throw 'The default layout leaves too little space for the log.' }
        if ($script:RenderPath) {
            $bitmap = [Drawing.Bitmap]::new($form.Width, $form.Height)
            try {
                $form.DrawToBitmap($bitmap, [Drawing.Rectangle]::new(0, 0, $form.Width, $form.Height))
                $bitmap.Save($script:RenderPath, [Drawing.Imaging.ImageFormat]::Png)
            } finally { $bitmap.Dispose() }
        }
    } else {
        if ($selected.decoding_preset -ne 'standard' -or $selected.normalize_audio -or
            -not $selected.condition_on_previous_text -or $selected.hallucination_filter -or
            $selected.vocabulary -or $selected.glossary_path -or $selected.device -ne 'cpu') {
            throw 'Opening old settings changed the legacy behavior.'
        }
    }
    $fields['decoding_preset'].SelectedIndex = 0
    $fields['decoding_preset'].SelectedIndex = 1
    if ($previousText.Checked) { throw 'Accuracy did not reset previous-text conditioning.' }
    $previousText.Checked = $true
    $testNames = 'Ainhoa; Tzeitel; Jos' + [char]0xE9
    $vocabulary.Text = $testNames
    $glossaryPath.Text = ''
    $normalizeAudio.Checked = $false
    $selected = Get-SelectedSettings
    Save-Settings ($selected | ConvertTo-Json)
    $saved = Get-Content -LiteralPath $SettingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $saved.condition_on_previous_text -or $saved.normalize_audio -or
        $saved.glossary_path -ne $null -or $saved.vocabulary -ne $testNames) {
        throw 'Saving lost an override, Unicode vocabulary, or an empty glossary.'
    }
    $form.Size = $form.MinimumSize
    $form.PerformLayout()
    $layout.PerformLayout()
    foreach ($control in @($wordTimestamps, $normalizeAudio, $previousText, $hallucinationFilter, $vocabulary, $glossaryPath, $actions, $log)) {
        if ($control.Right -gt $layout.ClientSize.Width - $layout.Padding.Right -or
            $control.Bottom -gt $layout.ClientSize.Height - $layout.Padding.Bottom) {
            throw "Control outside the minimum-size layout: $($control.AccessibleName) $($control.Text)"
        }
    }
    if ($log.Height -lt 60) { throw 'The minimum-size layout leaves too little space for the log.' }
}

$testDirectory = Join-Path $script:LauncherRoot ('.whisper-runner/launcher-settings-test-' + [guid]::NewGuid())
[void][IO.Directory]::CreateDirectory($testDirectory)
try {
    $configuration = Join-Path $testDirectory 'settings.json'
    Copy-Item -LiteralPath (Join-Path $script:LauncherRoot 'whisper-settings.json') -Destination $configuration
    & ([scriptblock]::Create($source)) -SettingsPath $configuration
    $script:SettingsCase = 'legacy'
    [IO.File]::WriteAllText($configuration, '{"model":"small","language":null,"task":"transcribe","output_format":"srt"}')
    & ([scriptblock]::Create($source)) -SettingsPath $configuration
    Write-Output 'Passed launcher settings, legacy defaults, Unicode save, preset override and layout checks.'
} finally {
    [IO.File]::Delete((Join-Path $testDirectory 'settings.json'))
    [IO.Directory]::Delete($testDirectory)
}
