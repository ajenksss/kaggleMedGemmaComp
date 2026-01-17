@echo off
echo ===================================================
echo 🏥 MedGemma Triage Copilot - Starting Demo...
echo ===================================================

:: Ensure we are running from the script's directory
cd /d "%~dp0"

echo Launching Streamlit via Python helper...
echo Working Directory: %CD%
echo.

:: Launch using python module directly
python -m streamlit run src/app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Demo failed to start.
    echo Please verify that 'pip install -r requirements.txt' has been run.
    echo.
    pause
)
