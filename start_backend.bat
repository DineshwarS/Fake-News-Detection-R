@echo off
title Fake News Detection - R Backend Server
echo ============================================================
echo Starting Fake News Detection Plumber REST API Server...
echo ============================================================

set "RSCRIPT=C:\Program Files\R\R-4.6.1\bin\Rscript.exe"

if not exist "%RSCRIPT%" (
    where Rscript >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "RSCRIPT=Rscript"
    ) else (
        echo ERROR: Rscript.exe not found at "%RSCRIPT%"!
        echo Please make sure R is installed or update the path in this script.
        pause
        exit /b 1
    )
)

echo Using R from: %RSCRIPT%
echo API URL: http://127.0.0.1:8000
echo Swagger UI: http://127.0.0.1:8000/__docs__/
echo.
"%RSCRIPT%" "Back End\run_api.R"

pause
