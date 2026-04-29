@echo off
setlocal
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
.tools\py314\python.exe src\runtime_status.py --show
endlocal
