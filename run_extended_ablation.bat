@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo Running real ETD-FGL component ablation study...
"%PYTHON_EXE%" run_extended_experiments.py ablation --datasets cora pubmed reddit --seed 42
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo ABLATION STUDY FAILED. Existing successful outputs were preserved.
pause
exit /b 1
