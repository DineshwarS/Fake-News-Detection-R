@echo off
title Fake News Detection - Training Pipeline
echo ============================================================
echo Starting Model Training Pipeline (80/20 Split, TF-IDF, glmnet)...
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
echo Training in progress... Please wait a couple minutes.
echo.
"%RSCRIPT%" "Back End\train_model.R"

echo.
echo ============================================================
echo Training finished. Model saved in Back End\models\
echo ============================================================
pause
