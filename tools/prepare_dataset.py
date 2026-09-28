# -*- coding: utf-8 -*-
"""
数据集准备脚本：Gaofen 高分农村建筑 -> mmseg 标准格式
支持两种标注：栅格掩码(PNG/TIF) / SHP矢量

用法（Windows PowerShell，在工程根目录执行）：
  python tools/prepare_dataset.py ^
      --src_image D:\Gaofen_Dataset\raw_img ^
      --src_label D:\Gaofen_Dataset\raw_mask ^
      --label_type mask --class_value 255
"""
import argparse
import os
import shutil

import cv2
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
MASK_EXTS = ('.png', '.tif', '.tiff')


def collect_images(src_image):
    return [f for f in sorted(os.listdir(src_image))
            if f.lower().endswith(IMG_EXTS)]


def norm_mask(mask, class_value):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.where(mask == class_value, 1, 0).astype(np.uint8)


def rasterize_shp(shp_path, ref_img_path, out_mask_path):
    """SHP -> 与参考影像对齐的掩码"""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(ref_img_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=out_shape, transform=transform,
                     fill=0, dtype=np.uint8)
    cv2.imwrite(out_mask_path, mask)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_image', required=True, help='原始影像目录')
    ap.add_argument('--src_label', required=True, help='原始标注目录')
    ap.add_argument('--label_type', required=True, choices=['mask', 'shp'])
    ap.add_argument('--out', default='D:/Gaofen_Dataset',
                    help='输出根目录(默认 D:/Gaofen_Dataset)')
    ap.add_argument('--class_value', type=int, default=255,
                    help='掩码中建筑像素值(通常1或255)')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'mask'), exist_ok=True)

    images = collect_images(args.src_image)
    if not images:
        raise SystemExit(f'[错误] {args.src_image} 下没有影像文件')

    label_files = {os.path.splitext(f)[0]: f
                   for f in os.listdir(args.src_label)
                   if f.lower().endswith(MASK_EXTS)}

    print(f'[Info] 共 {len(images)} 张影像，开始转换...')
    converted = 0
    for i, img in enumerate(images):
        base, ext = os.path.splitext(img)
        img_src = os.path.join(args.src_image, img)
        img_dst = os.path.join(args.out, 'img', f'{base}.png')
        mask_dst = os.path.join(args.out, 'mask', f'{base}.png')

        if os.path.exists(img_dst) and os.path.exists(mask_dst):
            converted += 1
            continue

        # 读原图（含带地理坐标 TIF）
        if ext.lower() in ('.tif', '.tiff'):
            import rasterio
            with rasterio.open(img_src) as src:
                arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_dst, arr)
        else:
            shutil.copyfile(img_src, img_dst)

        # 生成掩码
        if args.label_type == 'mask':
            if base not in label_files:
                print(f'[Warn] {img} 缺少同名掩码，跳过')
                continue
            m = cv2.imread(os.path.join(args.src_label, label_files[base]),
                           cv2.IMREAD_GRAYSCALE)
            if m is None:
                print(f'[Warn] 掩码读取失败: {label_files[base]}')
                continue
            cv2.imwrite(mask_dst, norm_mask(m, args.class_value))
        else:  # shp
            shp = next((os.path.join(args.src_label, c)
                        for c in os.listdir(args.src_label)
                        if c.lower().endswith('.shp')
                        and os.path.splitext(c)[0] == base), None)
            if shp is None:
                print(f'[Warn] {img} 缺少同名SHP，跳过')
                continue
            rasterize_shp(shp, img_src, mask_dst)

        converted += 1
        print(f'  [{i+1}/{len(images)}] {img} -> 完成')

    print(f'\n[Info] 成功转换 {converted} 张。')
    print('下一步: python tools/split_dataset.py 划分 train/val/test')


if __name__ == '__main__':
    main()
