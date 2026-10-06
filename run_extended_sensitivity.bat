@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo Running poisoning-rate sensitivity study...
"%PYTHON_EXE%" run_extended_experiments.py poison --datasets cora pubmed reddit --poison-rates 0.05 0.10 0.20 0.30 --seed 42
if errorlevel 1 goto :failed

echo Running trigger-size sensitivity study...
"%PYTHON_EXE%" run_extended_experiments.py trigger --datasets cora pubmed reddit --trigger-sizes 2 3 4 5 --seed 42
if errorlevel 1 goto :failed

"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo SENSITIVITY STUDY FAILED. Existing successful outputs were preserved.
pause
exit /b 1
