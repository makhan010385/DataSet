@echo off
setlocal
title TrustGraph-X v2

cd /d "%~dp0"
set "DATASET_DIR=E:\2026\Lincoln Conference\Third Coference\Dataset"

echo ============================================
echo  TrustGraph-X v2 - local launcher
echo  Folder : %CD%
echo  Dataset: %DATASET_DIR%
echo ============================================
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 (set "PY=py") else (set "PY=python")

%PY% --version >nul 2>&1
if not %ERRORLEVEL%==0 (
  echo [ERROR] Python is not installed or not on PATH.
  echo Install Python 3.10+ from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  echo.
  pause
  exit /b 1
)

if not exist "trustgraph_app.py" (
  echo [ERROR] trustgraph_app.py is not in this folder.
  echo Put RUN_TrustGraphX.bat, trustgraph_app.py, trustgraph_pipeline.py and
  echo requirements.txt together in the same folder.
  echo.
  pause
  exit /b 1
)

if not exist "%DATASET_DIR%" (
  echo [WARNING] Dataset folder not found:
  echo   %DATASET_DIR%
  echo You can still change it in the app sidebar after it opens.
  echo.
)

echo Installing dependencies ^(first run downloads ~500 MB, be patient^)...
%PY% -m pip install --upgrade pip
%PY% -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
if not %ERRORLEVEL%==0 (
  echo.
  echo [ERROR] Dependency installation failed - copy the text above and send it over.
  pause
  exit /b 1
)

echo.
echo Starting the app. Your browser opens at http://localhost:8501
echo KEEP THIS WINDOW OPEN while you use the app. Press Ctrl+C to stop.
echo.
%PY% -m streamlit run trustgraph_app.py --server.port 8501

echo.
echo The app stopped.
pause
