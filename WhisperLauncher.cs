using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class WhisperLauncher
{
    [STAThread]
    private static int Main()
    {
        string directory = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(directory, "Run-Whisper.ps1");
        if (!File.Exists(script))
        {
            return ShowError("Run-Whisper.ps1 is missing. Keep Whisper Launcher.exe with the other launcher files.");
        }

        // Start the existing WinForms launcher without creating a PowerShell console.
        var startInfo = new ProcessStartInfo
        {
            FileName = Path.Combine(Environment.SystemDirectory, @"WindowsPowerShell\v1.0\powershell.exe"),
            Arguments = "-NoProfile -STA -ExecutionPolicy Bypass -File \"" + script + "\"",
            WorkingDirectory = directory,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        try
        {
            using (Process launcher = Process.Start(startInfo))
            {
                launcher.WaitForExit();
                return launcher.ExitCode;
            }
        }
        catch (Win32Exception error)
        {
            return ShowError("Could not start Windows PowerShell.\r\n\r\n" + error.Message);
        }
    }

    private static int ShowError(string message)
    {
        MessageBox.Show(message, "Whisper Launcher", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
