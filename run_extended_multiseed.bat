@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo Running multi-seed statistical evaluation on Cora, PubMed, and Reddit...
"%PYTHON_EXE%" run_extended_experiments.py multiseed --datasets cora pubmed reddit --seeds 42 43 44 --methods all
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo MULTI-SEED STUDY FAILED. Existing successful outputs were preserved.
pause
exit /b 1
