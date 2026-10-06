@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: .venv Python was not found.
  pause
  exit /b 1
)

echo ====================================================================================================
echo FINAL REAL ETD-FGL RUN: Cora + PubMed + Reddit
 echo ====================================================================================================

"%PYTHON_EXE%" run_real_dataset_pipeline.py --dataset cora
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_real_comparisons_and_figures.py --dataset cora
if errorlevel 1 goto :failed

"%PYTHON_EXE%" run_real_dataset_pipeline.py --dataset pubmed
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_real_comparisons_and_figures.py --dataset pubmed
if errorlevel 1 goto :failed

"%PYTHON_EXE%" run_real_dataset_pipeline.py --dataset reddit
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_real_comparisons_and_figures.py --dataset reddit
if errorlevel 1 goto :failed

"%PYTHON_EXE%" build_real_comparisons_and_figures.py --dataset all
if errorlevel 1 goto :failed

"%PYTHON_EXE%" build_section4_extra_figures.py
if errorlevel 1 goto :failed

echo ====================================================================================================
echo ALL REAL DATASET RUNS AND FIGURES COMPLETED
 echo Outputs: outputs\final_multidataset
 echo ====================================================================================================
pause
exit /b 0

:failed
echo ====================================================================================================
echo PIPELINE STOPPED. Read the error above.
echo Completed outputs were not deleted.
echo ====================================================================================================
pause
exit /b 1
