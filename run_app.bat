@echo off
title LungCT UNet++ Diagnostic Studio Launcher
color 0B
cls
echo =====================================================================
echo           LungCT UNet++ Next-Gen AI Diagnostic Studio
echo =====================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b
)

echo [2/3] Checking dependencies...
python -c "import torch, flask, cv2, matplotlib, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing missing requirements...
    pip install -r requirements.txt
)

echo [3/3] Starting Flask Application...
echo.
echo =====================================================================
echo  Open your browser at: http://localhost:5000
echo =====================================================================
echo.
python app.py --port 5000
pause
