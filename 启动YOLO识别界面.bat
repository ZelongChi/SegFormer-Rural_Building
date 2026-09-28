@echo off
chcp 65001 >nul
title YOLO 单栋建筑识别界面
set "PY=C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe"
set YOLO_AUTOINSTALL=0
set YOLO_OFFLINE=1

echo ============================================
echo   YOLO-seg 单栋建筑识别界面
echo ============================================
echo.

rem 界面已在运行? 直接开浏览器
netstat -ano | findstr ":7862" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo   界面已在运行, 正在打开浏览器...
    start "" "http://127.0.0.1:7862"
    echo.
    pause
    exit /b 0
)

echo   正在启动界面服务(加载模型约需 30~60 秒)...
echo   服务会另开一个小窗口运行, 请勿关闭
echo.
start "" /min "%PY%" "D:\Gaofen_Building_SegFormer\app_yolo.py" > "D:\Gaofen_Building_SegFormer\work_dirs\ui_yolo_app.log" 2> "D:\Gaofen_Building_SegFormer\work_dirs\ui_yolo_err.log"

echo   等待服务就绪...
:wait
timeout /t 3 /nobreak >nul
netstat -ano | findstr ":7862" | findstr "LISTENING" >nul
if errorlevel 1 goto wait

echo   界面已就绪, 正在打开浏览器...
start "" "http://127.0.0.1:7862"
echo.
echo   使用完毕后关闭"界面服务"小窗口即可退出。
pause
