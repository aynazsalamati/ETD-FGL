@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: .venv Python was not found:
    echo %PYTHON_EXE%
    echo Create the virtual environment and install a compatible PyTorch build first.
    pause
    exit /b 1
)

echo Installing ETD-FGL Python dependencies...
"%PYTHON_EXE%" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo Installation failed.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully.
pause
