@echo off
REM Legacy entrypoint kept for compatibility. Use .tools\run.cmd for the full autonomous platform.
cd /d "%~dp0\.."
call .tools\run.cmd %*
