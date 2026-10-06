@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)
"%PYTHON_EXE%" build_extended_experiment_figures.py
pause
