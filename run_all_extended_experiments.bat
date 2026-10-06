@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo ================================================================================================
echo ETD-FGL EXTENDED JOURNAL EVALUATION
 echo This is a long real-experiment suite. Existing completed conditions are skipped automatically.
echo ================================================================================================

echo [1/6] Multi-seed evaluation...
"%PYTHON_EXE%" run_extended_experiments.py multiseed --datasets cora pubmed reddit --seeds 42 43 44 --methods all
if errorlevel 1 goto :failed

echo [2/6] Non-IID Dirichlet evaluation...
"%PYTHON_EXE%" run_extended_experiments.py noniid --datasets cora pubmed reddit --alphas 0.1 0.5 1.0 --methods attacked,pdfl,flpurifier,etdfgl
if errorlevel 1 goto :failed

echo [3/6] Ablation study...
"%PYTHON_EXE%" run_extended_experiments.py ablation --datasets cora pubmed reddit --seed 42
if errorlevel 1 goto :failed

echo [4/6] Sensitivity analysis...
"%PYTHON_EXE%" run_extended_experiments.py poison --datasets cora pubmed reddit --poison-rates 0.05 0.10 0.20 0.30 --seed 42
if errorlevel 1 goto :failed
"%PYTHON_EXE%" run_extended_experiments.py trigger --datasets cora pubmed reddit --trigger-sizes 2 3 4 5 --seed 42
if errorlevel 1 goto :failed

echo [5/6] Client-count scalability...
"%PYTHON_EXE%" run_extended_experiments.py clients --datasets cora pubmed reddit --client-counts 5 10 20 30 50 --methods attacked,pdfl,flpurifier,etdfgl --seed 42
if errorlevel 1 goto :failed

echo [6/6] Building final extended tables and journal figures...
"%PYTHON_EXE%" build_extended_experiment_figures.py
if errorlevel 1 goto :failed

echo ================================================================================================
echo ALL EXTENDED EXPERIMENTS COMPLETED
 echo Results: outputs\extended_experiments\journal_summary
echo ================================================================================================
pause
exit /b 0

:failed
echo ================================================================================================
echo EXTENDED PIPELINE STOPPED AT A FAILED STUDY.
echo Existing successful outputs were not deleted.
echo ================================================================================================
pause
exit /b 1
