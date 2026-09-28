# -*- coding: utf-8 -*-
"""把 test 集 16 张图拼成 4x4 大图（1024x1024），带模拟地理坐标"""
import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine

DATA_ROOT = r'D:\Gaofen_Dataset'
OUT = r'D:\Gaofen_Building_SegFormer\work_dirs\test_mosaic.tif'
GRID = 4

with open(os.path.join(DATA_ROOT, 'split', 'test.txt'), encoding='utf-8') as f:
    names = [l.strip() for l in f if l.strip()]

# 优先选含建筑的
sel = []
for n in names:
    base = os.path.splitext(n)[0]
    mask = cv2.imread(os.path.join(DATA_ROOT, 'mask', base + '.tif'), 0)
    if mask is not None and mask.max() > 0:
        sel.append(n)
    if len(sel) >= GRID * GRID:
        break
print('选用:', sel)

cell = 256
H = W = cell * GRID
canvas = np.zeros((H, W, 3), np.uint8)
for i, n in enumerate(sel):
    r, c = divmod(i, GRID)
    img = cv2.imread(os.path.join(DATA_ROOT, 'img', n))
    img = cv2.resize(img, (cell, cell))
    canvas[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = img

# 模拟地理坐标：1m 分辨率，CGCS2000 地理参考（伪坐标，仅演示链路）
transform = Affine(1.0, 0, 0, 0, -1.0, H)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with rasterio.open(OUT, 'w', driver='GTiff', height=H, width=W, count=3,
                   dtype='uint8', crs='EPSG:4490', transform=transform) as dst:
    dst.write(np.transpose(canvas, (2, 0, 1)))
print(f'[完成] 大图 {W}x{H}: {OUT}')
