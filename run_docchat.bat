@echo off
echo =======================================
echo     Starting DocChat AI Servers...
echo =======================================
echo.

echo [1/2] Starting FastAPI Backend on Port 8000...
start "DocChat Backend (FastAPI)" cmd /c ".venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8000"

echo Waiting 5 seconds for the backend to initialize...
timeout /t 5 /nobreak > NUL

echo [2/2] Starting Streamlit Frontend on Port 8501...
start "DocChat Frontend (Streamlit)" cmd /c ".venv\Scripts\streamlit.exe run ui.py"

echo.
echo =======================================
echo   Servers are launching!
echo   A browser tab should open shortly.
echo =======================================
echo You can safely close this black window.
timeout /t 3 > NUL
