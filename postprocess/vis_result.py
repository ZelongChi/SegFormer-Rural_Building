# -*- coding: utf-8 -*-
"""把影像与实例掩码叠加, 生成可视化预览图
用法: python vis_result.py --tif 影像.tif --mask 实例掩码.tif --out 输出.png
"""
import argparse
import cv2
import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tif', required=True)
    ap.add_argument('--mask', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with rasterio.open(args.tif) as s:
        if s.count >= 3:
            img = np.transpose(s.read([1, 2, 3]), (1, 2, 0))
        else:
            g = s.read(1)
            img = np.stack([g] * 3, axis=-1)
    img = np.clip(img, 0, 255).astype(np.uint8)

    with rasterio.open(args.mask) as s:
        m = s.read(1)

    n = int(m.max())
    rng = np.random.RandomState(7)
    colors = rng.randint(60, 255, (max(n, 1), 3)).tolist()

    vis = img.copy()
    for i in range(1, n + 1):
        mask = (m == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = colors[i - 1]
        for cnt in cnts:
            cv2.drawContours(vis, [cnt], -1, c, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(vis, str(i), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    cv2.imwrite(args.out, vis)
    print(f'可视化已保存: {args.out}')
    print(f'建筑栋数: {n}')


if __name__ == '__main__':
    main()
