# -*- coding: utf-8 -*-
"""
单图预测可视化（transformers 版 SegFormer）
用法:
  python tools/visualize_pred.py --image D:/Gaofen_Dataset/img/xxx.png
"""
import argparse
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\Gaofen_Building_SegFormer\hf_cache'

import cv2
import numpy as np
import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

CKPT = r'D:\Gaofen_Building_SegFormer\work_dirs\segformer_mitb2\best_mIoU.pth'
PRETRAINED = 'nvidia/segformer-b2-finetuned-ade-512-512'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', default=r'D:\Gaofen_Building_SegFormer\work_dirs\pred_overlay.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if not os.path.exists(CKPT):
        raise SystemExit(f'[错误] 未找到模型权重: {CKPT}\n请先完成训练。')

    processor = SegformerImageProcessor.from_pretrained(PRETRAINED)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(CKPT, map_location=args.device))
    model.to(args.device).eval()

    img = cv2.imread(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors='pt').to(args.device)

    with torch.no_grad():
        out = model(**inputs)
        logits = nn.functional.interpolate(
            out.logits, size=(img.shape[0], img.shape[1]),
            mode='bilinear', align_corners=False)
        seg = logits.argmax(dim=1)[0].cpu().numpy()

    overlay = img.copy()
    overlay[seg == 1] = (0, 0, 255)
    out_img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    contours, _ = cv2.findContours((seg == 1).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, contours, -1, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, out_img)
    n = sum(1 for c in contours if cv2.contourArea(c) > 20)
    print(f'[完成] 检出建筑连通域约 {n} 个，结果: {args.out}')


if __name__ == '__main__':
    main()
