@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo Running client-count scalability study: 5, 10, 20, 30, 50 clients...
"%PYTHON_EXE%" run_extended_experiments.py clients --datasets cora pubmed reddit --client-counts 5 10 20 30 50 --methods attacked,pdfl,flpurifier,etdfgl --seed 42
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo CLIENT SCALABILITY STUDY FAILED. Existing successful outputs were preserved.
pause
exit /b 1
