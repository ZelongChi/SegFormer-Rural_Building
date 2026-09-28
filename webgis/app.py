# -*- coding: utf-8 -*-
"""
webgis/app.py —— 建筑边界识别 WebGIS 后端
FastAPI + YOLO-seg: 上传影像 -> 瓦片推理 -> 返回 GeoJSON 矢量 + 预览图 + SHP/DXF
启动: python webgis/app.py   访问 http://127.0.0.1:8080
"""
import os
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['YOLO_OFFLINE'] = '1'

import io
import json
import time
import uuid
import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(r'D:\Gaofen_Building_SegFormer')
WEIGHTS = ROOT / 'yolo_data' / 'seg_building' / 'weights' / 'best.pt'
WEBGIS_DIR = ROOT / 'webgis'
JOBS_DIR = WEBGIS_DIR / 'jobs'
STATIC_DIR = WEBGIS_DIR / 'static'
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='BuildingInsight WebGIS')
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS))
    return _model


# 像素坐标 -> 球面坐标(MapLibre 自定义 extent)
def px_to_lon(x, W):
    return -180.0 + (x / W) * 360.0


def px_to_lat(y, H):
    return 85.05112878 - (y / H) * 170.10225756


def run_inference(img_path: Path, conf: float):
    """瓦片推理 -> 合并 -> 返回 buildings GeoJSON + 预览图 + 矢量文件"""
    model = get_model()
    with rasterio.open(str(img_path)) as src:
        crs = src.crs
        tr = src.transform
        H, W = src.height, src.width
        if src.count >= 3:
            arr = src.read([1, 2, 3])
        else:
            arr = np.repeat(src.read(1)[None, :, :], 3, axis=0)
    rgb = np.transpose(arr, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # 像元分辨率(m), 无投影时为 0
    px_m2 = abs(tr[0] * tr[4]) if crs is not None else 0.0

    tile, overlap = 512, 128
    stride = tile - overlap
    ys = list(range(0, H - tile + 1, stride))
    xs = list(range(0, W - tile + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))

    polys_xy = []
    for y0 in ys:
        for x0 in xs:
            t = rgb[y0:y0 + tile, x0:x0 + tile]
            res = model.predict(t, conf=conf, verbose=False)[0]
            if res.masks is None:
                continue
            for p in res.masks.xy:
                p = p + np.array([x0, y0])
                if len(p) >= 4:
                    polys_xy.append(p)

    # 过滤小碎片 + 多边形级 IoU 合并 + 简化
    cands = []
    for p in polys_xy:
        poly = Polygon(p)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 20:
            continue
        cands.append(poly)
    merged = []
    for poly in cands:
        hit = None
        for i, m in enumerate(merged):
            inter = poly.intersection(m).area
            union = poly.union(m).area
            if union > 0 and inter / union > 0.30:
                hit = i
                break
        if hit is None:
            merged.append(poly)
        else:
            merged[hit] = merged[hit].union(poly)
    kept = []
    for poly in merged:
        parts = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
        for part in parts:
            if part.area < 30:
                continue
            part = part.simplify(1.0, preserve_topology=True)
            if part.geom_type == 'Polygon' and part.area >= 30:
                kept.append(part)

    # 预览图
    vis = rgb.copy()
    rng = np.random.RandomState(7)
    colors = rng.randint(60, 255, (max(len(kept), 1), 3)).tolist()
    for i, poly in enumerate(kept, 1):
        c = colors[i - 1]
        pts = np.array(poly.exterior.coords, np.int32)
        cv2.drawContours(vis, [pts], -1, c, 2)

    buildings = []
    for i, poly in enumerate(kept, 1):
        area_px = poly.area
        coords = [[px_to_lon(x, W), px_to_lat(y, H)] for x, y in poly.exterior.coords]
        buildings.append({
            'id': i,
            'area_px2': round(area_px, 1),
            'area_m2': round(area_px * px_m2, 1) if px_m2 > 0 else None,
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            }
        })

    return {
        'buildings': buildings,
        'W': W, 'H': H,
        'rgb': rgb,
        'vis': vis,
        'kept_polys': kept,
        'crs': crs,
        'tr': tr,
        'px_m2': px_m2
    }


def save_vector_files(job_dir: Path, result):
    """保存 SHP / DXF"""
    import geopandas as gpd
    import ezdxf
    kept = result['kept_polys']
    crs = result['crs']
    tr = result['tr']
    inv = ~tr
    buildings_out = []
    for poly in kept:
        geo = Polygon([tuple(inv * (float(x), float(y))) for x, y in poly.exterior.coords])
        buildings_out.append(geo)
    gdf = gpd.GeoDataFrame(
        {'id': list(range(1, len(buildings_out) + 1)),
         'area_m2': [round(g.area, 1) for g in buildings_out]},
        geometry=buildings_out, crs=crs)
    shp = job_dir / 'building.shp'
    gdf.to_file(shp, encoding='utf-8')
    dxf = job_dir / 'building.dxf'
    doc = ezdxf.new('R2000')
    msp = doc.modelspace()
    for poly in kept:
        msp.add_lwpolyline(
            [tuple(inv * (float(x), float(y))) for x, y in poly.exterior.coords], close=True)
    doc.saveas(str(dxf))
    return shp.name, dxf.name


@app.post('/api/infer')
async def infer(file: UploadFile = File(...), conf: float = Form(0.35)):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = 'job_' + ts + '_' + uuid.uuid4().hex[:6]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    src_path = job_dir / ('input' + suffix)
    with open(src_path, 'wb') as f:
        f.write(await file.read())

    result = run_inference(src_path, conf)

    # 预览图(带轮廓)
    vis_png = job_dir / 'preview.png'
    cv2.imwrite(str(vis_png), cv2.cvtColor(result['vis'], cv2.COLOR_RGB2BGR))
    # 原图(作为底图)
    base_png = job_dir / 'base.png'
    cv2.imwrite(str(base_png), cv2.cvtColor(result['rgb'], cv2.COLOR_RGB2BGR))

    shp_name, dxf_name = save_vector_files(job_dir, result)

    return JSONResponse({
        'job_id': job_id,
        'count': len(result['buildings']),
        'width': result['W'],
        'height': result['H'],
        'px_m2': result['px_m2'],
        'buildings': result['buildings'],
        'preview_url': f'/files/{job_id}/preview.png',
        'base_url': f'/files/{job_id}/base.png',
        'shp_url': f'/files/{job_id}/{shp_name}',
        'dxf_url': f'/files/{job_id}/{dxf_name}'
    })


@app.get('/files/{job_id}/{name}')
async def get_file(job_id: str, name: str):
    p = JOBS_DIR / job_id / name
    if not p.exists():
        return JSONResponse({'error': 'not found'}, status_code=404)
    media = 'image/png' if name.endswith('.png') else \
            'application/octet-stream'
    return FileResponse(str(p), media_type=media, filename=name)


app.mount('/', StaticFiles(directory=str(STATIC_DIR), html=True), name='static')


if __name__ == '__main__':
    import uvicorn
    print('BuildingInsight WebGIS running at http://127.0.0.1:8080')
    uvicorn.run(app, host='127.0.0.1', port=8080)
