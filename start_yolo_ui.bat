@echo off
set "PY=C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe"
set YOLO_AUTOINSTALL=0
set YOLO_OFFLINE=1

title YOLO Building Extraction UI

echo ============================================
echo   YOLO Building Boundary Extraction
echo ============================================
echo.

netstat -ano | findstr ":7862" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo   UI already running, opening browser...
    start "" "http://127.0.0.1:7862"
    echo.
    pause
    exit /b 0
)

echo   Starting service (loading model, 30~60s)...
echo   Keep the hidden service window alive.
echo.
start "" /min "%PY%" "D:\Gaofen_Building_SegFormer\app_yolo.py" > "D:\Gaofen_Building_SegFormer\work_dirs\ui_yolo_app.log" 2> "D:\Gaofen_Building_SegFormer\work_dirs\ui_yolo_err.log"

echo   Waiting for service...
:wait
timeout /t 3 /nobreak >nul
netstat -ano | findstr ":7862" | findstr "LISTENING" >nul
if errorlevel 1 goto wait

echo   Ready, opening browser...
start "" "http://127.0.0.1:7862"
echo.
pause
