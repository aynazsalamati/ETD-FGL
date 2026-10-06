@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe run_real_dataset_pipeline.py --dataset reddit
if errorlevel 1 pause & exit /b 1
.venv\Scripts\python.exe build_real_comparisons_and_figures.py --dataset reddit
pause
