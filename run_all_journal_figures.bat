@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found in the project root.
  echo Create the virtual environment first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" build_all_journal_figures.py

if errorlevel 1 (
  echo.
  echo Figure generation failed. Read the error above.
  pause
  exit /b 1
)

echo.
echo All available journal figures were generated.
pause
