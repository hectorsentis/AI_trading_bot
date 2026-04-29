@echo off
setlocal
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File ".tools\run.ps1" %*
endlocal
