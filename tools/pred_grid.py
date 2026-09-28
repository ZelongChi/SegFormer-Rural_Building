# -*- coding: utf-8 -*-
"""test 集推理对比可视化：原图 | 真值 | 预测（4 张三联拼图）"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\Gaofen_Building_SegFormer\hf_cache'

import cv2
import numpy as np
import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DATA_ROOT = r'D:\Gaofen_Dataset'
WORK_DIR = r'D:\Gaofen_Building_SegFormer\work_dirs'
CKPT = os.path.join(WORK_DIR, 'segformer_mitb2', 'best_mIoU.pth')
PRETRAINED = 'nvidia/segformer-b2-finetuned-ade-512-512'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = SegformerImageProcessor.from_pretrained(PRETRAINED)
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

with open(os.path.join(DATA_ROOT, 'split', 'test.txt'), encoding='utf-8') as f:
    names = [l.strip() for l in f if l.strip()]

rows, miou_list = [], []
n_show = 0
for name in names:
    base = os.path.splitext(name)[0]
    img = cv2.imread(os.path.join(DATA_ROOT, 'img', name))
    mask = cv2.imread(os.path.join(DATA_ROOT, 'mask', base + '.tif'), 0)
    if mask is None:
        mask = cv2.imread(os.path.join(DATA_ROOT, 'mask', base + '.png'), 0)
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:      # 跳过纯背景，只展示含建筑样本
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_r = cv2.resize(rgb, (512, 512))
    mask_r = cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)

    with torch.no_grad():
        out = model(pixel_values=processor(images=rgb_r, return_tensors='pt')['pixel_values'].to(DEVICE))
    logits = nn.functional.interpolate(out.logits, size=(512, 512), mode='bilinear', align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    ious = []
    for c in range(2):
        p, t = (pred == c), (mask_r == c)
        inter = (p & t).sum(); union = (p | t).sum()
        ious.append(float(inter) / float(union) if union else 1.0)
    miou = float(np.mean(ious))
    miou_list.append(miou)

    cell = 256
    row = np.concatenate([
        cv2.resize(rgb_r, (cell, cell)),
        cv2.cvtColor(cv2.resize(mask_r * 255, (cell, cell),
                                interpolation=cv2.INTER_NEAREST).astype(np.uint8),
                     cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(cv2.resize((pred * 255).astype(np.uint8), (cell, cell),
                                interpolation=cv2.INTER_NEAREST),
                     cv2.COLOR_GRAY2RGB),
    ], axis=1)
    rows.append(row)
    n_show += 1
    print(f'{name:24s} mIoU={miou:.3f}')
    if n_show >= 4:
        break

grid = np.concatenate(rows, axis=0)
out_path = os.path.join(WORK_DIR, 'test_pred_grid.png')
cv2.imwrite(out_path, grid)
print(f'\n[完成] 4 张平均 mIoU={np.mean(miou_list):.4f}，拼图: {out_path}')
