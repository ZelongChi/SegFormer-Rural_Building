# -*- coding: utf-8 -*-
"""
app_yolo.py —— YOLO-seg 单栋建筑识别 本地可视化界面 (Gradio)
==============================================================
上传遥感影像 -> 自动识别每栋建筑 -> 输出:
  - 叠加预览图(每栋彩色编号)
  - building.shp / building.dxf (每栋独立多边形, 可导入南方CASS)
  - instance_mask.tif (每栋一个ID)

用法:  python app_yolo.py    然后浏览器打开 http://127.0.0.1:7862
"""
import os

# 必须在 import ultralytics 之前禁用其自动更新(否则每次启动都会 pip 更新并退出)
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['YOLO_OFFLINE'] = '1'

import datetime
import traceback

import numpy as np
import cv2
import rasterio
import geopandas as gpd
import ezdxf
import gradio as gr
from shapely.geometry import Polygon
from ultralytics import YOLO

ROOT = r'D:\Gaofen_Building_SegFormer'
WEIGHTS = os.path.join(ROOT, 'yolo_data', 'seg_building', 'weights', 'best.pt')
UI_DIR = os.path.join(ROOT, 'work_dirs', 'ui_yolo')

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS)
    return _model


def _read_rgb(path):
    """读影像为 RGB (自动拉伸 2%-98% 分位, 支持单波段/多波段)"""
    with rasterio.open(path) as s:
        if s.count >= 3:
            arr = s.read([1, 2, 3])
        else:
            arr = np.repeat(s.read(1)[None, :, :], 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    p = np.percentile(arr, [2, 98])
    arr = np.clip((arr - p[0]) / max(p[1] - p[0], 1e-6), 0, 1) * 255
    return np.ascontiguousarray(arr.astype(np.uint8))


def _bbox(p):
    return [float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max())]


def _iou_boxes(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def run_job(img_path, conf, progress=gr.Progress(track_tqdm=False)):
    """完整流程: 瓦片推理 -> 实例去重 -> 掩码 -> SHP/DXF -> 可视化"""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    job = os.path.join(UI_DIR, 'job_' + ts)
    os.makedirs(job, exist_ok=True)
    try:
        progress(0.1, desc='① 读取影像...')
        with rasterio.open(img_path) as src:
            crs, tr, H, W = src.crs, src.transform, src.height, src.width
            if src.count >= 3:
                arr = src.read([1, 2, 3])
            else:
                arr = np.repeat(src.read(1)[None, :, :], 3, axis=0)
        rgb = np.transpose(arr, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        progress(0.3, desc='② 模型推理(瓦片)中...')
        model = get_model()
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

        progress(0.7, desc='③ 合并重叠实例/去碎片...')
        # 过滤小碎片 + 多边形级IoU合并瓦片切碎实例 + 简化
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

        if not kept:
            return None, None, None, None, '⚠️ 未检测到建筑。可调低置信度阈值(如 0.20)重试。'

        progress(0.8, desc='④ 写实例掩码...')
        inst = np.zeros((H, W), np.uint8)
        for i, poly in enumerate(kept, 1):
            cv2.fillPoly(inst, [np.array(poly.exterior.coords, np.int32)], i)
        mask_tif = os.path.join(job, 'instance_mask.tif')
        with rasterio.open(mask_tif, 'w', driver='GTiff', height=H, width=W,
                           count=1, dtype='uint8', crs=crs, transform=tr) as dst:
            dst.write(inst, 1)

        progress(0.9, desc='⑤ 导出矢量(SHP/DXF)...')
        inv = ~tr

        def to_geo(poly):
            return [tuple(inv * (float(x), float(y))) for x, y in poly.exterior.coords]

        geoms = [Polygon(to_geo(p)) for p in kept]
        gdf = gpd.GeoDataFrame(
            {'id': list(range(1, len(kept) + 1)),
             'area_m2': [round(g.area, 1) for g in geoms]},
            geometry=geoms, crs=crs)
        shp = os.path.join(job, 'building.shp')
        gdf.to_file(shp, encoding='utf-8')
        dxf = os.path.join(job, 'building.dxf')
        doc = ezdxf.new('R2000')
        msp = doc.modelspace()
        for poly in kept:
            msp.add_lwpolyline(
                [tuple(inv * (float(x), float(y))) for x, y in poly.exterior.coords],
                close=True)
        doc.saveas(dxf)

        # 可视化: 每栋彩色轮廓 + 编号
        vis = _read_rgb(img_path)
        rng = np.random.RandomState(7)
        colors = rng.randint(60, 255, (len(kept), 3)).tolist()
        for i, poly in enumerate(kept, 1):
            c = colors[i - 1]
            pts = np.array(poly.exterior.coords, np.int32)
            cv2.drawContours(vis, [pts], -1, c, 2)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.putText(vis, str(i), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        vis_p = os.path.join(job, 'vis.png')
        cv2.imwrite(vis_p, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        total_area = sum(g.area for g in geoms)
        summary = (f'✅ 识别到 **{len(kept)} 栋**建筑\n\n'
                   f'| 栋号 | 面积(m²) |\n|---|---|\n' +
                   '\n'.join(f'| {i} | {gdf.loc[i-1, "area_m2"]} |'
                             for i in range(1, len(kept) + 1)) +
                   f'\n\n**合计面积: {total_area:.1f} m²**')
        files = [shp, dxf, mask_tif]
        return vis_p, files, summary, None, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, f'❌ 处理失败:\n{e}\n\n{tb[-1200:]}', None, None


# ============ 界面 ============
with gr.Blocks(title='YOLO-seg 单栋建筑识别') as demo:
    gr.Markdown('## 🏠 YOLO-seg 单栋建筑识别（农村建筑）\n'
                '上传遥感影像 → 自动识别**每一栋**建筑 → 输出每栋独立矢量（SHP/DXF，可直接导入南方CASS）')
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type='filepath', label='上传影像 (TIF/PNG/JPG)')
            conf_sl = gr.Slider(0.20, 0.60, value=0.35, step=0.05,
                                label='置信度阈值 (低=检得多, 高=检得准)')
            btn = gr.Button('🚀 开始识别', variant='primary')
        with gr.Column(scale=2):
            img_out = gr.Image(type='filepath', label='识别结果（每栋彩色编号）')
            with gr.Row():
                out_files = gr.File(label='矢量产物 (SHP/DXF/掩码)')
            txt_out = gr.Markdown()

    btn.click(run_job, inputs=[img_in, conf_sl],
              outputs=[img_out, out_files, txt_out, img_in, img_in])

demo.queue(default_concurrency_limit=1)
if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7862,
                allowed_paths=[UI_DIR], inbrowser=True)
    print('界面已启动: http://127.0.0.1:7862')
