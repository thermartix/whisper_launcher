#Requires -Version 5.1
param([string]$SettingsPath)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

function Add-Label($row, $text) {
    $label = [System.Windows.Forms.Label]::new()
    $label.Text = $text
    $label.AutoSize = $true
    $label.Anchor = 'Left'
    $layout.Controls.Add($label, 0, $row)
}

function Add-Setting($row, $name, $label, [Collections.IDictionary]$choices) {
    Add-Label $row $label
    $combo = [System.Windows.Forms.ComboBox]::new()
    $combo.AccessibleName = $label
    $combo.Dock = 'Fill'
    $combo.DropDownStyle = 'DropDownList'
    $combo.Tag = $choices
    $combo.Items.AddRange([object[]]@($choices.Keys))
    foreach ($choice in $choices.GetEnumerator()) {
        if ($choice.Value -ceq $settings.$name) {
            $combo.SelectedItem = $choice.Key
            break
        }
    }
    $layout.Controls.Add($combo, 1, $row)
    $layout.SetColumnSpan($combo, 2)
    $fields[$name] = $combo
}

function Get-SelectedSettings {
    $selected = [ordered]@{}
    foreach ($name in $fields.Keys) {
        $combo = $fields[$name]
        if ($combo.SelectedIndex -lt 0) { throw "Choose a value for $($combo.AccessibleName)." }
        $selected[$name] = $combo.Tag[$combo.SelectedItem]
    }
    $selected['word_timestamps'] = $wordTimestamps.Checked
    if ($selected.model.EndsWith('.en') -and
        ($selected.language -notin $null, 'en' -or $selected.task -ne 'transcribe')) {
        throw 'English-only models support English transcription only.'
    }
    if ($selected.task -eq 'translate' -and $selected.model -in 'turbo', 'large-v3-turbo') {
        throw "Choose a multilingual model such as 'small' for English translation."
    }
    return $selected
}

function Save-Settings([string]$Json) {
    # Replace the file in one operation so other launches always read a complete preset.
    $temporary = Join-Path ([IO.Path]::GetDirectoryName($SettingsPath)) ([IO.Path]::GetRandomFileName())
    try {
        [IO.File]::WriteAllText($temporary, "$Json`r`n")
        if ([IO.File]::Exists($SettingsPath)) {
            [IO.File]::Replace($temporary, $SettingsPath, [NullString]::Value)
        } else {
            [IO.File]::Move($temporary, $SettingsPath)
        }
    } finally {
        [IO.File]::Delete($temporary)
    }
}

function Add-LogChunk($stream, [string]$chunk, [Collections.Generic.List[string]]$lines) {
    foreach ($part in [regex]::Matches($chunk, '[^\r\n]+|[\r\n]')) {
        switch ($part.Value) {
            "`r" {
                # Wait for the next character so CRLF still ends a normal log line.
                $stream.CarriageReturn = $true
            }
            "`n" {
                $line = $stream.Line.ToString()
                if (-not [string]::IsNullOrWhiteSpace($line)) { $lines.Add($line) }
                [void]$stream.Line.Clear()
                $stream.CarriageReturn = $false
            }
            default {
                if ($stream.CarriageReturn) {
                    [void]$stream.Line.Clear()
                    $stream.CarriageReturn = $false
                }
                [void]$stream.Line.Append($part.Value)
            }
        }
    }
}

$SettingsPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SettingsPath)
$settingsWarning = ''
try {
    $settings = Get-Content -LiteralPath $SettingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($settings -isnot [pscustomobject]) { throw 'The settings must be a JSON object.' }
    if ($settings.PSObject.Properties.Name -cnotcontains 'device') {
        $settings | Add-Member -NotePropertyName device -NotePropertyValue 'cpu'
    }
} catch {
    $settingsWarning = "Could not read the saved settings. $($_.Exception.Message)"
    $settings = [pscustomobject]@{model='medium'; device='auto'; language=$null; task='transcribe'; output_format='srt'}
}

# The catalog matches the models and language codes in the pinned Whisper release.
$options = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'whisper-options.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$models = [ordered]@{}
foreach ($model in $options.models) { $models[$model] = $model }
$languages = [ordered]@{
    'Auto-detect' = $null
    'Mixed languages (faster-whisper)' = 'multilingual'
}
foreach ($language in $options.languages.PSObject.Properties) {
    $languages["$($language.Value) ($($language.Name))"] = $language.Name
}

$version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'VERSION') -Raw).Trim()
$form = [System.Windows.Forms.Form]::new()
$form.Text = 'Whisper Launcher'
$form.StartPosition = 'CenterScreen'
$form.Font = [System.Drawing.Font]::new('Segoe UI', 10)
$form.AutoScaleMode = 'Font'
$form.ClientSize = [System.Drawing.Size]::new(680, 515)
$form.MinimumSize = [System.Drawing.Size]::new(580, 470)

$layout = [System.Windows.Forms.TableLayoutPanel]::new()
$layout.Dock = 'Fill'
$layout.Padding = [System.Windows.Forms.Padding]::new(16)
$layout.ColumnCount = 3
$layout.RowCount = 10
[void]$layout.ColumnStyles.Add([System.Windows.Forms.ColumnStyle]::new('Absolute', 124))
[void]$layout.ColumnStyles.Add([System.Windows.Forms.ColumnStyle]::new('Percent', 100))
[void]$layout.ColumnStyles.Add([System.Windows.Forms.ColumnStyle]::new('AutoSize'))
for ($row = 0; $row -lt 9; $row++) {
    [void]$layout.RowStyles.Add([System.Windows.Forms.RowStyle]::new('AutoSize'))
}
[void]$layout.RowStyles.Add([System.Windows.Forms.RowStyle]::new('Percent', 100))
$form.Controls.Add($layout)

# Add the menu after the fill panel so the panel is docked below it.
$menu = [System.Windows.Forms.MenuStrip]::new()
$fileMenu = [System.Windows.Forms.ToolStripMenuItem]::new('&File')
$exitItem = [System.Windows.Forms.ToolStripMenuItem]::new('E&xit')
$exitItem.Add_Click({ $form.Close() })
[void]$fileMenu.DropDownItems.Add($exitItem)

$helpMenu = [System.Windows.Forms.ToolStripMenuItem]::new('&Help')
$aboutItem = [System.Windows.Forms.ToolStripMenuItem]::new('&About...')
$aboutItem.Add_Click({
    $aboutText = @"
Whisper Launcher $version

Transcribe or translate audio and video files locally on Windows.
Uses OpenAI Whisper for single-language recordings and faster-whisper for mixed-language recordings.
"@
    [void][System.Windows.Forms.MessageBox]::Show($form, $aboutText, 'About Whisper Launcher', 'OK', 'Information')
})
[void]$helpMenu.DropDownItems.Add($aboutItem)
$menu.Items.AddRange(@($fileMenu, $helpMenu))
$form.MainMenuStrip = $menu
$form.Controls.Add($menu)

Add-Label 0 'Audio or video'
$mediaPath = [System.Windows.Forms.TextBox]::new()
$mediaPath.AccessibleName = 'Audio or video file path'
$mediaPath.Dock = 'Fill'
$layout.Controls.Add($mediaPath, 1, 0)
$browse = [System.Windows.Forms.Button]::new()
$browse.Text = 'Browse...'
$browse.AutoSize = $true
$layout.Controls.Add($browse, 2, 0)

$fields = [ordered]@{}
Add-Setting 1 'model' 'Model' $models
Add-Setting 2 'device' 'Device' ([ordered]@{'Automatic (NVIDIA or CPU)'='auto'; 'CPU'='cpu'; 'NVIDIA CUDA'='cuda'})
Add-Setting 3 'language' 'Source language' $languages
Add-Setting 4 'task' 'Task' ([ordered]@{'Transcribe'='transcribe'; 'Translate to English'='translate'})
Add-Setting 5 'output_format' 'Output format' ([ordered]@{
    'SubRip subtitles (.srt)'='srt'
    'Plain text (.txt)'='txt'
    'WebVTT subtitles (.vtt)'='vtt'
    'Tab-separated text (.tsv)'='tsv'
    'JSON (.json)'='json'
    'All formats'='all'
})

$wordTimestamps = [System.Windows.Forms.CheckBox]::new()
$wordTimestamps.Text = 'Word timestamps'
$wordTimestamps.AutoSize = $true
$wordTimestamps.Checked = $settings.word_timestamps -is [bool] -and $settings.word_timestamps
$layout.Controls.Add($wordTimestamps, 1, 6)
$layout.SetColumnSpan($wordTimestamps, 2)
$outputFormat = $fields['output_format']
$updateWordTimestamps = {
    # Keep the saved preference while the selected format hides the checkbox.
    $wordTimestamps.Visible = $outputFormat.SelectedIndex -ge 0 -and
        $outputFormat.Tag[$outputFormat.SelectedItem] -in 'json', 'vtt'
}
$outputFormat.Add_SelectedIndexChanged($updateWordTimestamps)
& $updateWordTimestamps

$actions = [System.Windows.Forms.FlowLayoutPanel]::new()
$actions.AutoSize = $true
$actions.Dock = 'Fill'
$start = [System.Windows.Forms.Button]::new()
$start.Text = 'Transcribe'
$start.AutoSize = $true
$save = [System.Windows.Forms.Button]::new()
$save.Text = 'Save as defaults'
$save.AutoSize = $true
$actions.Controls.AddRange(@($start, $save))
$layout.Controls.Add($actions, 0, 7)
$layout.SetColumnSpan($actions, 3)
$form.AcceptButton = $start

$status = [System.Windows.Forms.Label]::new()
$status.Text = 'Choose an audio or video file to get started.'
$status.AutoSize = $true
$layout.Controls.Add($status, 0, 8)
$layout.SetColumnSpan($status, 3)

$log = [System.Windows.Forms.TextBox]::new()
$log.AccessibleName = 'Transcription log'
$log.Multiline = $true
$log.ReadOnly = $true
$log.ScrollBars = 'Vertical'
$log.Dock = 'Fill'
$layout.Controls.Add($log, 0, 9)
$layout.SetColumnSpan($log, 3)
if ($settingsWarning) { $log.Text = $settingsWarning }

$browse.Add_Click({
    $dialog = [System.Windows.Forms.OpenFileDialog]::new()
    try {
        $dialog.Title = 'Choose audio or video to transcribe'
        $dialog.Filter = 'Audio and video files|*.aac;*.aiff;*.avi;*.flac;*.m4a;*.m4v;*.mkv;*.mov;*.mp3;*.mp4;*.mpeg;*.mpg;*.ogg;*.opus;*.wav;*.webm;*.wma;*.wmv|All files (*.*)|*.*'
        $dialog.CheckFileExists = $true
        $dialog.RestoreDirectory = $true
        if ($dialog.ShowDialog($form) -eq 'OK') { $mediaPath.Text = $dialog.FileName }
    } finally {
        $dialog.Dispose()
    }
})

$save.Add_Click({
    try {
        Save-Settings (Get-SelectedSettings | ConvertTo-Json)
        $status.Text = 'Saved these settings as defaults.'
    } catch {
        $status.Text = $_.Exception.Message
    }
})

$run = @{Process=$null; Settings=$null; Streams=[ordered]@{}; LogTail=''}
$timer = [System.Windows.Forms.Timer]::new()
$timer.Interval = 150
$timer.Add_Tick({
    # Poll both redirected streams without blocking the form or filling either pipe.
    $lines = [Collections.Generic.List[string]]::new()
    foreach ($stream in @($run.Streams.Values)) {
        for ($i = 0; $i -lt 16 -and $stream.Pending -and $stream.Pending.IsCompleted; $i++) {
            $length = $stream.Pending.GetAwaiter().GetResult()
            $stream.Pending = $null
            if ($length -eq 0) {
                Add-LogChunk $stream "`n" $lines
                break
            }
            Add-LogChunk $stream ([string]::new($stream.Buffer, 0, $length)) $lines
            $stream.Pending = $stream.Reader.ReadAsync($stream.Buffer, 0, $stream.Buffer.Length)
        }
    }

    # Only replace the unfinished lines at the end, leaving completed messages in place.
    $unfinished = foreach ($stream in $run.Streams.Values) {
        $line = $stream.Line.ToString()
        if (-not [string]::IsNullOrWhiteSpace($line)) { $line }
    }
    $tail = $unfinished -join "`r`n"
    if ($lines.Count -or $tail -cne $run.LogTail) {
        $completed = if ($lines.Count) { ($lines -join "`r`n") + "`r`n" } else { '' }
        $log.Select($log.TextLength - $run.LogTail.Length, $run.LogTail.Length)
        $log.SelectedText = $completed + $tail
        $log.SelectionStart = $log.TextLength
        $log.ScrollToCaret()
        $run.LogTail = $tail
    }

    if ($run.Process.HasExited -and -not ($run.Streams.Values | Where-Object { $_.Pending })) {
        $timer.Stop()
        $status.Text = if ($run.Process.ExitCode -eq 0) {
            'Transcription complete. The log shows where your files were saved.'
        } else {
            'Transcription failed. See the log for details.'
        }
        $run.Process.Dispose()
        $run.Process = $null
        $run.Streams.Clear()
        [IO.File]::Delete($run.Settings)
        $run.Settings = $null
        $start.Enabled = $true
    }
})

$start.Add_Click({
    try {
        $json = Get-SelectedSettings | ConvertTo-Json
        $path = $mediaPath.Text.Trim().Trim('"')
        $file = if ($path) { Get-Item -LiteralPath $path -ErrorAction SilentlyContinue }
        if ($file -isnot [IO.FileInfo]) { throw 'Choose an existing audio or video file.' }

        # Give each run its own preset so saving defaults cannot change a run during setup.
        $run.Settings = [IO.Path]::GetTempFileName()
        [IO.File]::WriteAllText($run.Settings, $json)
        $runner = Join-Path $PSScriptRoot 'Run-Whisper.ps1'
        $info = [Diagnostics.ProcessStartInfo]::new()
        $info.FileName = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $info.Arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -MediaPath "{1}" -SettingsPath "{2}" -OpenOutput' -f $runner, $file.FullName, $run.Settings
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.StandardOutputEncoding = [Text.Encoding]::UTF8
        $info.StandardErrorEncoding = [Text.Encoding]::UTF8

        $run.Process = [Diagnostics.Process]::Start($info)
        foreach ($name in 'StandardOutput', 'StandardError') {
            $reader = $run.Process.$name
            $buffer = [char[]]::new(4096)
            $run.Streams[$name] = @{
                Reader=$reader
                Buffer=$buffer
                Pending=$reader.ReadAsync($buffer, 0, $buffer.Length)
                Line=[Text.StringBuilder]::new()
                CarriageReturn=$false
            }
        }
        $log.Clear()
        $run.LogTail = ''
        $start.Enabled = $false
        $status.Text = 'Running setup and transcription...'
        $timer.Start()
    } catch {
        if ($run.Settings) {
            [IO.File]::Delete($run.Settings)
            $run.Settings = $null
        }
        $status.Text = $_.Exception.Message
    }
})

$form.Add_FormClosing({
    param($sender, $eventArgs)
    if ($run.Process -and $eventArgs.CloseReason -eq 'UserClosing') {
        $eventArgs.Cancel = $true
        $status.Text = 'Transcription is running. You can close this window when it finishes.'
    }
})

try {
    [void]$form.ShowDialog()
} finally {
    $timer.Dispose()
    $form.Dispose()
}
