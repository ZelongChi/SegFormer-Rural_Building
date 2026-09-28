# -*- coding: utf-8 -*-
"""
大图 TIF 推理（transformers 版 SegFormer）：
重叠切片 + 加权融合 -> 建筑二值掩码 GeoTIFF
用法:
  python postprocess/infer_tif.py --tif D:/Gaofen_Dataset/test_dom.tif
输出:
  result_mask.tif (0/1 二值)   result_prob.tif (0~1 概率)

变更：自动处理小于 tile 的小图——先 resize 到 tile 尺寸推理，
      再把结果 resize 回原始尺寸，地理参考保持不变。
"""
import argparse
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\Gaofen_Building_SegFormer\hf_cache'

import numpy as np
import rasterio
import torch
from PIL import Image
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

CKPT = r'D:\Gaofen_Building_SegFormer\work_dirs\segformer_mitb2\best_mIoU.pth'
PRETRAINED = 'nvidia/segformer-b2-finetuned-ade-512-512'


def load_image_geo(tif_path):
    """读取影像为 RGB uint8，自动处理：
    - P 模式调色板 TIFF（count=1 / colorinterp=palette）-> PIL convert RGB
    - 单波段灰度图 -> 复制成 3 通道
    - 16-bit/高位深 -> 2%-98% 百分位拉伸到 8-bit
    - 多波段 -> 取前 3 波段，保证 RGB 顺序
    """
    Image.MAX_IMAGE_PIXELS = None

    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform
        h, w = src.height, src.width
        count = src.count
        is_palette = any(ci == rasterio.enums.ColorInterp.palette
                         for ci in src.colorinterp)

    # P 模式调色板 / 单波段：走 PIL，能正确还原调色板
    if count < 3 or is_palette:
        pil = Image.open(tif_path)
        arr = np.array(pil.convert("RGB"))
        print(f'  [读取] 调色板/单波段图，已 convert 为 RGB ({arr.shape})')
    else:
        # 多波段真彩色：rasterio 读前 3 波段
        with rasterio.open(tif_path) as src:
            raw = src.read([1, 2, 3])
            arr = np.transpose(raw, (1, 2, 0))

    # 位深统一：非 uint8 做 2%-98% 拉伸
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, [2, 98])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        arr = arr.astype(np.uint8)
        print(f'  [读取] 高位深 -> 8-bit（2-98% 拉伸）')

    return arr, transform, crs, h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tif', required=True)
    ap.add_argument('--out', default=r'D:\Gaofen_Building_SegFormer\work_dirs\result')
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--prob_thr', type=float, default=0.45)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if not os.path.exists(CKPT):
        raise SystemExit(f'[错误] 未找到模型权重: {CKPT}\n请先运行 tools/train.py 完成训练。')

    os.makedirs(args.out, exist_ok=True)
    print('[1/4] 加载模型...')
    processor = SegformerImageProcessor.from_pretrained(PRETRAINED)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(CKPT, map_location=args.device))
    model.to(args.device).eval()

    print('[2/4] 读取影像...')
    img, transform, crs, H0, W0 = load_image_geo(args.tif)
    print(f'  原始影像 {W0}x{H0}, 瓦片 {args.tile}, 重叠 {args.overlap}')

    # ---- 小图自动 resize 到 tile 尺寸（两边都不足一个 tile 时才需要）----
    NEED_RESIZE = (H0 < args.tile) and (W0 < args.tile)
    if NEED_RESIZE:
        img = np.array(Image.fromarray(img).resize(
            (args.tile, args.tile), Image.BICUBIC))
        H, W = args.tile, args.tile
        print(f'  [小图] 原图 {W0}x{H0} < tile {args.tile}，已双三次 resize 到 {W}x{H}')
    else:
        H, W = H0, W0

    prob_acc = np.zeros((H, W), np.float32)
    w_acc = np.zeros((H, W), np.float32)
    stride = args.tile - args.overlap

    ys = list(range(0, max(H - args.tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - args.tile, 0) + 1, stride)) or [0]
    if ys[-1] + args.tile < H:
        ys.append(H - args.tile)
    if xs[-1] + args.tile < W:
        xs.append(W - args.tile)

    print('[3/4] 切片推理...')
    with torch.no_grad():
        for i, y0 in enumerate(ys):
            for j, x0 in enumerate(xs):
                tile = img[y0:y0 + args.tile, x0:x0 + args.tile]
                th, tw = tile.shape[:2]
                inputs = processor(images=tile, return_tensors='pt').to(args.device)
                out = model(**inputs)
                logits = nn.functional.interpolate(
                    out.logits, size=(th, tw), mode='bilinear',
                    align_corners=False)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

                # 边缘权重斜坡，消除拼接缝
                ramp = np.ones_like(prob, np.float32)
                r = min(16, args.overlap // 2)
                ramp[:r, :] *= np.linspace(0.2, 1.0, r)[:, None]
                ramp[-r:, :] *= np.linspace(1.0, 0.2, r)[:, None]
                ramp[:, :r] *= np.linspace(0.2, 1.0, r)[None, :]
                ramp[:, -r:] *= np.linspace(1.0, 0.2, r)[None, :]

                prob_acc[y0:y0 + th, x0:x0 + tw] += prob * ramp
                w_acc[y0:y0 + th, x0:x0 + tw] += ramp
                print(f'  瓦片 [{i+1}/{len(ys)}][{j+1}/{len(xs)}]', end='\r')

    w_acc[w_acc < 1e-6] = 1.0
    prob = prob_acc / w_acc
    mask = (prob >= args.prob_thr).astype(np.uint8)

    # ---- 如果之前 resize 过，把结果缩回原始尺寸 ----
    if NEED_RESIZE:
        prob = np.array(Image.fromarray((prob * 255).astype(np.uint8))
                        .resize((W0, H0), Image.BILINEAR)).astype(np.float32) / 255.0
        mask = (prob >= args.prob_thr).astype(np.uint8)
        H, W = H0, W0
        print(f'\n  [小图] 结果已缩回原始尺寸 {W}x{H}')

    print('\n[4/4] 写出 GeoTIFF...')
    with rasterio.open(os.path.join(args.out, 'result_mask.tif'), 'w',
                       driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(mask, 1)
    with rasterio.open(os.path.join(args.out, 'result_prob.tif'), 'w',
                       driver='GTiff', height=H, width=W, count=1,
                       dtype='float32', crs=crs, transform=transform) as dst:
        dst.write(prob.astype(np.float32), 1)

    print(f'[完成] 输出: {args.out}\\result_mask.tif 与 result_prob.tif')
    print('下一步: python postprocess/mask_to_vector.py 提取单栋建筑矢量')


if __name__ == '__main__':
    main()
