@echo off
setlocal
title Whisper Runner
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0Run-Whisper.ps1"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)
