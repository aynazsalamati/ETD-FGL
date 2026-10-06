@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo Running Dirichlet non-IID study: alpha = 0.1, 0.5, 1.0...
"%PYTHON_EXE%" run_extended_experiments.py noniid --datasets cora pubmed reddit --alphas 0.1 0.5 1.0 --methods attacked,pdfl,flpurifier,etdfgl
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo NON-IID STUDY FAILED. Existing successful outputs were preserved.
pause
exit /b 1
