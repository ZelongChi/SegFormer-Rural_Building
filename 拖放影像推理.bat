@echo off
chcp 65001 >nul
title YOLO 单栋建筑识别 - 拖放推理
set "PY=C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe"
set YOLO_AUTOINSTALL=0
set YOLO_OFFLINE=1

echo ============================================
echo   YOLO-seg 单栋建筑识别 - 拖放推理
echo ============================================
echo.
if "%~1"=="" (
    echo   用法: 把要识别的影像文件(TIF/PNG/JPG)直接拖到这个图标上
    echo.
    pause
    exit /b 1
)

echo   正在识别: %~1
echo   输出目录: D:\Gaofen_Building_SegFormer\work_dirs\result_seg_bat
echo.

"%PY%" "D:\Gaofen_Building_SegFormer\postprocess\infer_seg_yolo.py" --tif "%~1" --out "D:\Gaofen_Building_SegFormer\work_dirs\result_seg_bat"
echo.
echo   完成! 产物在 work_dirs\result_seg_bat\ 下:
echo     building.shp / building.dxf (南方CASS可直接打开)
echo     instance_mask.tif (每栋一个ID)
echo.
pause
