@echo off
rem One-click start on Windows: makes a private Python environment the first time, then opens the app.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First start: installing Kemudi. This downloads about 300 MB once.
  python -m venv .venv || goto :failed
  ".venv\Scripts\python.exe" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || goto :failed
  ".venv\Scripts\python.exe" -m pip install --quiet -e . || goto :failed
)
".venv\Scripts\python.exe" -m kemudi serve --open
goto :eof
:failed
echo Install failed. Check that Python 3.10 or newer is installed and that you are online.
pause
