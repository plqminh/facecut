@echo off
echo ==============================================
echo      FaceCut AI - Environment Installer
echo ==============================================
echo.

if exist venv (
    echo [INFO] Virtual environment 'venv' already exists.
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Make sure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [INFO] Removing potential CPU versions of torch...
pip uninstall -y torch torchvision torchaudio

echo.
echo [INFO] Cleaning pip cache...
pip cache purge

echo.
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo [SUCCESS] FaceCut AI environment is ready!
echo You can now run 'run_facecut.bat' to start the app.
echo ==============================================
pause
