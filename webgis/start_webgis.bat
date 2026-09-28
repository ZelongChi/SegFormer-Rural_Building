@echo off
set "PY=C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe"
set YOLO_AUTOINSTALL=0
set YOLO_OFFLINE=1

title BuildingInsight WebGIS

echo ============================================
echo   BuildingInsight WebGIS Server
echo ============================================
echo.

netstat -ano | findstr ":8080" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo   Server already running, opening browser...
    start "" "http://127.0.0.1:8080"
    pause
    exit /b 0
)

echo   Starting server (loading model, 30~60s)...
start "" "http://127.0.0.1:8080"
"%PY%" "D:\Gaofen_Building_SegFormer\webgis\app.py"
pause
