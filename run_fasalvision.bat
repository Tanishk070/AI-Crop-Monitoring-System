@echo off
title 🌿 FASALVision Launcher - SIH 2025
echo.
echo ================================================
echo       🌾 Launching FASALVision Prototype
echo ================================================
echo.

:: Navigate to your project directory
cd /d C:\Programming\SIH_2025\AI-Crop-Monitoring-System

:: Activate virtual environment
echo Activating virtual environment...
call .\venv310\Scripts\activate

:: Start backend server in a new terminal window
echo Starting FastAPI backend...
start "FASALVision Backend" cmd /k "uvicorn python.api.app:app --reload --host 127.0.0.1 --port 8000"

:: Wait for backend to initialize
timeout /t 5 /nobreak >nul

:: Start Streamlit frontend (dashboard.py) in another window
echo Launching Streamlit frontend...
cd python\dashboard
start "FASALVision Dashboard" cmd /k "streamlit run dashboard.py"

echo.
echo ✅ All systems started successfully!
echo You can now access the dashboard at: http://localhost:8501
echo.

pause
