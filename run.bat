@echo off
setlocal
cd /d "%~dp0"
title Hand Gesture Recognition - Launcher

echo ================================================
echo   Hand Gesture Recognition - Project Launcher
echo ================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py -3.12"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found.
        echo Install Python 3.9 to 3.12 and enable "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PYTHON% -m venv venv
    if errorlevel 1 goto :error
) else (
    echo [1/3] Virtual environment already exists.
)

echo.
echo [2/3] Installing/checking required packages...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [3/3] Starting the application...
echo.
"venv\Scripts\python.exe" app.py
if errorlevel 1 goto :error

echo.
echo Application closed.
pause
exit /b 0

:error
echo.
echo [ERROR] Something went wrong. Check the messages above.
pause
exit /b 1
