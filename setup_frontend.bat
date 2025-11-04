@echo off
echo ================================================================================
echo F1 Tire Wear AI - Frontend Setup
echo ================================================================================
echo.

cd frontend

echo Installing dependencies (this may take a few minutes)...
echo.
call npm install

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo Next steps:
echo   1. Run start_backend.bat (in one terminal)
echo   2. Run start_frontend.bat (in another terminal)
echo   3. Open http://localhost:3000 in your browser
echo.

pause
