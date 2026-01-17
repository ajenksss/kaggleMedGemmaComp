@echo off
echo ===================================================
echo 🏥 MedGemma Triage Copilot - Starting Demo...
echo ===================================================
cd /d "%~dp0"

echo Activating environment and launching Streamlit...
echo.

:: Check if streamlit is in path, just in case
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Streamlit not found in PATH.
    echo Please ensure you have installed the dependencies: pip install -r requirements.txt
    pause
    exit /b
)

streamlit run src/app.py

pause
