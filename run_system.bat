@echo off
echo Starting Hyperlocal Air Quality Forecasting Engine...
echo.
echo Launching FastAPI Backend...
start cmd /k "python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000"
echo.
echo Launching Streamlit Dashboard...
timeout /t 3
start cmd /k "streamlit run src/dashboard/app.py"
echo.
echo Both services launched successfully!