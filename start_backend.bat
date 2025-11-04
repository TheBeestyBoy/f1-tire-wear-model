@echo off
echo ================================================================================
echo F1 Tire Wear AI - Starting Backend Server
echo ================================================================================
echo.

cd backend\app
echo [1/2] Starting FastAPI server on http://localhost:8000...
echo.
"C:\Users\Soren\AppData\Local\Programs\Python\Python312\python.exe" f1_main.py

pause
