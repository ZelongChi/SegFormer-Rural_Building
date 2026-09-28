# -*- coding: utf-8 -*-
"""
infer_seg_yolo.py —— YOLO-seg 实例分割: 大图 -> 每栋建筑独立多边形 -> SHP/DXF
================================================================================
用训练好的 YOLO-seg 权重对任意大图做实例分割，输出每栋建筑独立边界：
  - instance_mask.tif : 每栋一个 ID 的实例掩码(带地理坐标)
  - building.shp      : 每栋多边形 + area_m2（可用 QGIS/ArcGIS/CASS 打开）
  - building.dxf      : R2000 格式（可直接导入南方CASS）

用法:
  python postprocess/infer_seg_yolo.py --tif 你的影像.tif
  python postprocess/infer_seg_yolo.py --tif 影像.tif --weights 你的best.pt --conf 0.35
"""
import argparse
import os

# 必须在 import ultralytics 之前禁用其自动更新(否则每次启动都会 pip 更新并退出)
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['YOLO_OFFLINE'] = '1'

import cv2
import numpy as np
import rasterio
import ezdxf
import geopandas as gpd
from shapely.geometry import Polygon
from ultralytics import YOLO

DEFAULT_WEIGHTS = r'D:\Gaofen_Building_SegFormer\yolo_data\seg_building\weights\best.pt'
DEFAULT_OUT = r'D:\Gaofen_Building_SegFormer\work_dirs\result_seg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--tif', required=True, help='待识别的大图(TIF/PNG/JPG均可)')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--tile', type=int, default=512, help='推理瓦片大小')
    ap.add_argument('--overlap', type=int, default=128, help='瓦片重叠')
    ap.add_argument('--conf', type=float, default=0.35, help='置信度阈值(低=多检, 高=少检)')
    ap.add_argument('--iou', type=float, default=0.45, help='NMS IoU')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.weights):
        print(f'[错误] 找不到权重: {args.weights}')
        print('请先训练, 或检查训练输出目录。')
        return

    print('[1/4] 加载模型...')
    model = YOLO(args.weights)

    print('[2/4] 读取影像...')
    with rasterio.open(args.tif) as src:
        crs, tr, H, W = src.crs, src.transform, src.height, src.width
        n_bands = src.count
        if n_bands >= 3:
            arr = src.read([1, 2, 3])          # (3, H, W)
            rgb = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
        else:  # 单波段灰度 -> 复制成 RGB
            arr = src.read(1)                  # (H, W)
            rgb = np.stack([arr] * 3, axis=-1)  # (H, W, 3)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    print(f'  尺寸: {W}x{H}, 波段: {n_bands}')

    print('[3/4] 瓦片推理...')
    stride = args.tile - args.overlap
    ys = list(range(0, H - args.tile + 1, stride))
    xs = list(range(0, W - args.tile + 1, stride))
    if not ys or ys[-1] + args.tile < H:
        ys.append(max(0, H - args.tile))
    if not xs or xs[-1] + args.tile < W:
        xs.append(max(0, W - args.tile))

    polys_xy = []  # 全图像素坐标多边形
    for y0 in ys:
        for x0 in xs:
            tile = rgb[y0:y0 + args.tile, x0:x0 + args.tile]
            res = model.predict(tile, conf=args.conf, iou=args.iou, verbose=False)[0]
            if res.masks is None:
                continue
            for p in res.masks.xy:          # 每实例 (N,2) 瓦片内坐标
                p = p + np.array([x0, y0])  # 转全图坐标
                if len(p) >= 4:
                    polys_xy.append(p)

    # ---- 后处理: 过滤碎片 + 合并瓦片切碎的实例 + 简化轮廓 ----
    # 1) 转 shapely 多边形, 过滤小碎片/无效几何
    cands = []
    for p in polys_xy:
        poly = Polygon(p)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 20:
            continue
        cands.append(poly)

    # 2) 多边形级 IoU 合并: 同一栋被瓦片切成两半, 交集占比高则合并
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

    # 3) 拆开 MultiPolygon(掩码孔洞产生的多环), 再过滤/简化
    kept = []
    for poly in merged:
        parts = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
        for part in parts:
            if part.area < 30:
                continue
            part = part.simplify(1.0, preserve_topology=True)  # 简化,去锯齿
            if part.geom_type == 'Polygon' and part.area >= 30:
                kept.append(part)
    print(f'  检测到建筑实例: {len(kept)} 栋')

    if not kept:
        print('  [提示] 未检测到建筑。可降低 --conf (如 0.20) 重试。')
        return

    # 实例掩码
    inst = np.zeros((H, W), np.uint8)
    for i, poly in enumerate(kept, 1):
        pts = np.array(poly.exterior.coords, np.int32)
        cv2.fillPoly(inst, [pts], i)
    with rasterio.open(os.path.join(args.out, 'instance_mask.tif'), 'w',
                       driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs=crs, transform=tr) as dst:
        dst.write(inst, 1)

    print('[4/4] 导出矢量...')
    inv = ~tr

    def to_geo(poly):
        return [tuple(inv * (float(x), float(y))) for x, y in poly.exterior.coords]

    geoms = [Polygon(to_geo(p)) for p in kept]
    gdf = gpd.GeoDataFrame(
        {'area_m2': [round(g.area, 1) for g in geoms]},
        geometry=geoms, crs=crs)
    gdf.to_file(os.path.join(args.out, 'building.shp'), encoding='utf-8')

    doc = ezdxf.new('R2000')
    msp = doc.modelspace()
    for p in kept:
        msp.add_lwpolyline([tuple(inv * (float(x), float(y))) for x, y in p.exterior.coords],
                           close=True)
    doc.saveas(os.path.join(args.out, 'building.dxf'))

    print(f'[完成] 输出目录: {args.out}')
    print('  building.shp / building.dxf（可导入南方CASS）')
    print('  instance_mask.tif（每栋一个ID）')


if __name__ == '__main__':
    main()
