@echo off
echo ================================================================================
echo F1 Tire Wear AI - Starting Frontend
echo ================================================================================
echo.

cd frontend

echo [1/3] Setting up F1 application files...
if exist src\index_f1.tsx (
    copy /Y src\index_f1.tsx src\index.tsx >nul
    echo [OK] Entry point updated
) else (
    echo [OK] Using existing index.tsx
)
echo     Files ready
echo.

echo [2/3] Checking dependencies...
if not exist node_modules (
    echo [WARNING] Dependencies not installed. Run setup_frontend.bat first!
    pause
    exit /b 1
)
echo [OK] Dependencies found
echo.

echo [3/3] Starting React development server on http://localhost:3000...
echo.
echo NOTE: The page will auto-reload when you make changes.
echo Press CTRL+C to stop the server.
echo.
call npm start

pause
