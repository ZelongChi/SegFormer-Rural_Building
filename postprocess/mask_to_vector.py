# -*- coding: utf-8 -*-
"""
掩码 -> 单栋建筑矢量（核心后处理）
===========================================
语义分割只给"建筑/非建筑"，本脚本把建筑连通域拆成独立单栋并导出矢量：
  掩码 -> 形态学去噪 -> 连通域 -> 距离变换+分水岭拆分连片农房
       -> 外轮廓 -> 多边形简化 -> 房角直角规整
       -> 像素转地理坐标 -> SHP + DXF(南方CASS可直接打开)

用法（Windows PowerShell，工程根目录执行）:
  python postprocess/mask_to_vector.py ^
      --mask D:/Gaofen_Building_SegFormer/work_dirs/result/result_mask.tif ^
      --dom D:/Gaofen_Dataset/test_dom.tif ^
      --out D:/Gaofen_Building_SegFormer/work_dirs/result/building.shp

参数:
  --mask                推理得到的建筑二值掩码 GeoTIFF
  --dom                 原始正射影像 TIF（用于读取地理坐标，若掩码已带坐标可省略）
  --epsg                输出坐标系（默认4490=CGCS2000地理坐标）
  --min_area_px         小于该像素面积的噪点剔除
  --watershed_thresh    分水岭前景阈值比例（0~1，连片农房多调低=0.25~0.35）
  --simplify_tol        轮廓简化容差(像素)
  --force_right_angle   是否开启房角直角规整(测绘DLG要求)
"""
import argparse
import os

import cv2
import numpy as np
import rasterio

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    _HAS_GEO = True
except Exception:
    _HAS_GEO = False

try:
    import ezdxf
    _HAS_DXF = True
except Exception:
    _HAS_DXF = False


def load_mask_geo(mask_path, dom_path):
    """优先从原始DOM读地理坐标，否则用掩码自带坐标"""
    if dom_path and os.path.exists(dom_path):
        with rasterio.open(dom_path) as src:
            transform, crs = src.transform, src.crs
    else:
        with rasterio.open(mask_path) as src:
            transform, crs = src.transform, src.crs
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask, transform, crs


def watershed_split(bin_mask, fg_thresh):
    """距离变换 + 分水岭，拆分连片农房，返回 markers（每栋一个ID）"""
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    if dist.max() < 1e-6:
        return np.zeros_like(bin_mask, np.int32)

    # 前景种子：距离图上较亮的区域 = 房屋核心
    _, sure_fg = cv2.threshold(dist, fg_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(closed, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # 分水岭需要 3 通道彩色图作为输入
    markers = cv2.watershed(
        cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR), markers)
    return markers


def right_angle_regularize(pts):
    """房角直角规整：主方向聚类 + 正交直线求交重建，消除像素锯齿。
    步骤：1) 求轮廓主方向(直方图)与正交次方向
         2) 每条边归属到最近的主/次方向
         3) 相邻边求交点得到规整顶点（正交多边形）
    """
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)

    vecs = np.roll(pts, -1, axis=0) - pts
    lens = np.hypot(vecs[:, 0], vecs[:, 1])
    valid = lens > 1e-6
    dirs = vecs[valid] / lens[valid][:, None]
    angs = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180

    # 主方向：方向角直方图峰值
    hist, edges = np.histogram(angs, bins=36, range=(0, 180))
    i0 = int(np.argmax(hist))
    theta0 = np.radians((edges[i0] + edges[i0 + 1]) / 2.0)
    d0 = np.array([np.cos(theta0), np.sin(theta0)])
    d1 = np.array([-np.sin(theta0), np.cos(theta0)])  # 正交方向

    # 每条边归属到最近主方向
    edge_dirs = np.zeros((n, 2))
    for k in range(n):
        if lens[k] < 1e-6:
            continue
        v = vecs[k] / lens[k]
        # 与 d0 的夹角(0~90)，决定归属
        cos_a = abs(float(np.dot(v, d0)))
        edge_dirs[k] = d0 if cos_a > np.cos(np.pi / 4) else d1
    # 方向符号统一（保持原始走向）
    for k in range(n):
        if lens[k] < 1e-6:
            continue
        if np.dot(vecs[k], edge_dirs[k]) < 0:
            edge_dirs[k] = -edge_dirs[k]

    def line_intersect(p, d, q, e):
        """直线 p+t*d 与 q+s*e 的交点（2D 叉积法）"""
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-9:
            return q.copy()
        t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / denom
        return p + t * d

    # 顶点 = 相邻两条边所在直线的交点
    new_pts = np.zeros_like(pts)
    for i in range(n):
        p_prev = pts[i - 1]
        p_cur = pts[i]
        new_pts[i] = line_intersect(p_prev, edge_dirs[i - 1],
                                    p_cur, edge_dirs[i])

    # 去除重合点并闭合
    keep = [0]
    for i in range(1, n):
        if np.hypot(*(new_pts[i] - new_pts[keep[-1]])) > 1e-6:
            keep.append(i)
    out = new_pts[keep]
    if len(out) >= 3 and not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0]])
    return out


def extract_polygon(markers, lbl, simplify_tol, min_area_px, force_right_angle):
    comp = (markers == lbl).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area_px:
        return None

    approx = cv2.approxPolyDP(cnt, max(simplify_tol, 2.0), True)
    pts = approx.reshape(-1, 2).astype(float)
    if force_right_angle and len(pts) >= 4:
        pts = right_angle_regularize(pts)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def to_geo(pts_pixel, transform):
    pts = np.array(pts_pixel)
    xs = transform.a * pts[:, 0] + transform.b * pts[:, 1] + transform.c
    ys = transform.d * pts[:, 0] + transform.e * pts[:, 1] + transform.f
    return np.stack([xs, ys], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', required=True)
    ap.add_argument('--dom', default=None)
    ap.add_argument('--out', default=r'D:\Gaofen_Building_SegFormer\work_dirs\result\building.shp')
    ap.add_argument('--epsg', type=int, default=4490)
    ap.add_argument('--min_area_px', type=float, default=100)
    ap.add_argument('--watershed_thresh', type=float, default=0.4)
    ap.add_argument('--simplify_tol', type=float, default=3.0)
    ap.add_argument('--force_right_angle', action='store_true', default=True)
    ap.add_argument('--no_dxf', action='store_true')
    args = ap.parse_args()

    mask, transform, crs = load_mask_geo(args.mask, args.dom)
    print(f'[1/4] 掩码 {mask.shape}, 过滤 <{args.min_area_px}px')

    print('[2/4] 分水岭拆分连片农房...')
    markers = watershed_split(mask, args.watershed_thresh)
    labels = [l for l in np.unique(markers) if l > 1]
    print(f'  候选实例 {len(labels)} 个')

    print('[3/4] 轮廓提取 + 简化 + 直角矫正...')
    polys_geo, areas_m2 = [], []
    for lbl in labels:
        pts = extract_polygon(markers, lbl, args.simplify_tol,
                              args.min_area_px, args.force_right_angle)
        if pts is None:
            continue
        geo = to_geo(pts, transform)
        polys_geo.append(Polygon([(float(x), float(y)) for x, y in geo]))
        areas_m2.append(polys_geo[-1].area)

    print(f'  有效建筑 {len(polys_geo)} 栋')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # ---- SHP ----
    if _HAS_GEO and polys_geo:
        gdf = gpd.GeoDataFrame(
            {'id': range(1, len(polys_geo) + 1),
             'area_m2': areas_m2},
            geometry=polys_geo, crs=f'EPSG:{args.epsg}')
        gdf.to_file(args.out, encoding='utf-8')
        print(f'  SHP -> {args.out}')
    else:
        print('  [Warn] 未安装 geopandas/shapely 或无有效多边形，跳过SHP')

    # ---- DXF（R2000+ 支持 LWPOLYLINE，CASS 兼容）----
    if not args.no_dxf and _HAS_DXF and polys_geo:
        dxf_path = os.path.splitext(args.out)[0] + '.dxf'
        doc = ezdxf.new('R2000')
        msp = doc.modelspace()
        for poly in polys_geo:
            msp.add_lwpolyline(list(poly.exterior.coords), close=True)
        doc.saveas(dxf_path)
        print(f'  DXF -> {dxf_path} (可导入南方CASS)')
    else:
        print('  [Warn] 未安装 ezdxf 或已禁用，跳过DXF')

    print('\n[完成] 矢量成果可在 CASS / QGIS / ArcGIS 中打开质检。')


if __name__ == '__main__':
    main()
