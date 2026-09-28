@echo off
set "PY=C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe"
set YOLO_AUTOINSTALL=0
set YOLO_OFFLINE=1

title YOLO Building Inference

echo ============================================
echo   YOLO Building Boundary Inference
echo ============================================
echo.

if "%~1"=="" (
    echo   Drag an image file (TIF/PNG/JPG) onto this icon.
    echo.
    pause
    exit /b 1
)

echo   Processing: %~1
echo   Output: D:\Gaofen_Building_SegFormer\work_dirs\result_seg_bat
echo.

"%PY%" "D:\Gaofen_Building_SegFormer\postprocess\infer_seg_yolo.py" --tif "%~1" --out "D:\Gaofen_Building_SegFormer\work_dirs\result_seg_bat"
echo.
echo   Done. Output files:
echo     building.shp / building.dxf (open in CASS)
echo     instance_mask.tif (one ID per building)
echo.
pause
