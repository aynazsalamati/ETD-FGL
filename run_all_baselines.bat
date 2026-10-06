@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "BASE_OUTPUT=%~dp0outputs\federated_cora"

if not exist "%PYTHON_EXE%" (
    echo ERROR: .venv Python was not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

echo ================================================================================================
echo ETD-FGL: run all selected baselines on Cora
echo ================================================================================================

if not exist "%BASE_OUTPUT%\partitions\iid_stratified_5_clients\client_node_indices.pt" (
    echo [1/10] Creating the shared Cora partition...
    "%PYTHON_EXE%" partition_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [1/10] Shared Cora partition already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\fedavg_clean\best_fedavg_cora.pt" (
    echo [2/10] Training Clean FedAvg...
    "%PYTHON_EXE%" train_fedavg_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [2/10] Clean FedAvg result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\fedavg_structural_backdoor\best_backdoored_fedavg_cora.pt" (
    echo [3/10] Training Attacked FedAvg...
    "%PYTHON_EXE%" train_fedavg_backdoor_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [3/10] Attacked FedAvg result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\defended_fedavg_structural_backdoor\final_summary.json" (
    if exist "%BASE_OUTPUT%\pilot_trust_fusion\pilot_trust_report.json" (
        echo [4/10] Training ETD-FGL from the existing trust report...
        "%PYTHON_EXE%" train_defended_fedavg_cora.py
        if errorlevel 1 goto :failed
    ) else (
        echo ERROR: ETD-FGL final result and pilot trust report are both missing.
        echo Keep your existing outputs folder, or rebuild the ETD-FGL trust pipeline first.
        goto :failed
    )
) else (
    echo [4/10] ETD-FGL result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\pdfl_calibration\pdfl_detector.joblib" (
    echo [5/10] Calibrating PD-FL on clean GCN updates...
    "%PYTHON_EXE%" calibrate_pdfl_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [5/10] PD-FL detector already exists. Skipping calibration.
)

if not exist "%BASE_OUTPUT%\baseline_pdfl\final_summary.json" (
    echo [6/10] Training PD-FL...
    "%PYTHON_EXE%" train_pdfl_fedavg_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [6/10] PD-FL result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\baseline_gsp\final_summary.json" (
    echo [7/10] Training GSP-FL...
    "%PYTHON_EXE%" train_gsp_fedavg_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [7/10] GSP-FL result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\baseline_flpurifier\final_summary.json" (
    echo [8/10] Training FLPurifier-GNN...
    "%PYTHON_EXE%" train_flpurifier_fedavg_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [8/10] FLPurifier-GNN result already exists. Skipping.
)

if not exist "%BASE_OUTPUT%\baseline_dmgnn\final_summary.json" (
    echo [9/10] Training DMGNN-FL...
    "%PYTHON_EXE%" train_dmgnn_fedavg_cora.py
    if errorlevel 1 goto :failed
) else (
    echo [9/10] DMGNN-FL result already exists. Skipping.
)

echo [10/10] Building the final comparison and journal figures...
"%PYTHON_EXE%" build_all_baseline_comparison.py
if errorlevel 1 goto :failed
"%PYTHON_EXE%" build_all_baseline_journal_figures.py
if errorlevel 1 goto :failed

echo.
echo ================================================================================================
echo ALL BASELINES COMPLETED SUCCESSFULLY
echo Results:
echo %BASE_OUTPUT%\all_baseline_comparison
echo ================================================================================================
pause
exit /b 0

:failed
echo.
echo ================================================================================================
echo PIPELINE STOPPED BECAUSE A STEP FAILED.
echo Read the error above. Existing successful outputs were not deleted.
echo ================================================================================================
pause
exit /b 1
