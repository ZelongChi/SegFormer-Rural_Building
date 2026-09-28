# -*- coding: utf-8 -*-
"""诊断：为什么训练 loss≈0 但验证 mIoU 只有 0.1？"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\Gaofen_Building_SegFormer\hf_cache'

import cv2
import numpy as np
import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DATA_ROOT = r'D:\Gaofen_Dataset'
WORK_DIR = r'D:\Gaofen_Building_SegFormer\work_dirs\segformer_mitb2'
PRETRAINED = 'nvidia/segformer-b2-finetuned-ade-512-512'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 512

processor = SegformerImageProcessor.from_pretrained(PRETRAINED)
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED, num_labels=2, ignore_mismatched_sizes=True,
    id2label={0: 'background', 1: 'building'},
    label2id={'background': 0, 'building': 1}).to(DEVICE)

# 用最新 best_mIoU.pth
ckpt = torch.load(os.path.join(WORK_DIR, 'best_mIoU.pth'), map_location=DEVICE)
model.load_state_dict(ckpt if 'model' not in ckpt else ckpt['model'])
model.eval()

with open(os.path.join(DATA_ROOT, 'split', 'val.txt'), encoding='utf-8') as f:
    names = [l.strip() for l in f if l.strip()]

print(f'=== 诊断: 前 12 张验证图 ===')
miou_sum, n = 0.0, 0
bld_gt_tot, bld_pred_tot = 0, 0
for name in names[:12]:
    base = os.path.splitext(name)[0]
    img = cv2.imread(os.path.join(DATA_ROOT, 'img', name))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(DATA_ROOT, 'mask', base + '.tif'), 0)
    if mask is None:
        mask = cv2.imread(os.path.join(DATA_ROOT, 'mask', base + '.png'), 0)
    mask_bin = (mask > 0).astype(np.int64)

    img_r = cv2.resize(img_rgb, (SIZE, SIZE))
    mask_r = cv2.resize(mask_bin, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    enc = processor(images=img_r, return_tensors='pt')
    with torch.no_grad():
        out = model(pixel_values=enc['pixel_values'].to(DEVICE))
    logits = nn.functional.interpolate(
        out.logits, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    gt_bld = (mask_r == 1).sum()
    pd_bld = (pred == 1).sum()
    inter = ((pred == 1) & (mask_r == 1)).sum()
    union = ((pred == 1) | (mask_r == 1)).sum()
    iou_b = float(inter) / float(union) if union else 1.0
    # mIoU 与脚本一致
    ious = []
    for c in range(2):
        p, t = (pred == c), (mask_r == c)
        inter_c = (p & t).sum(); union_c = (p | t).sum()
        ious.append(float(inter_c) / float(union_c) if union_c else 1.0)
    miou = float(np.mean(ious))
    miou_sum += miou; n += 1
    bld_gt_tot += gt_bld; bld_pred_tot += pd_bld
    print(f'{name:24s} 真值建筑 {gt_bld/262144*100:5.1f}%  预测建筑 {pd_bld/262144*100:5.1f}%  '
          f'建筑IoU {iou_b:.3f}  mIoU {miou:.3f}')

    if n <= 2:  # 保存前两张可视化
        vis = np.concatenate([
            cv2.resize(img_rgb, (256, 256)),
            cv2.cvtColor((mask_r * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
            cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        ], axis=1)
        cv2.imwrite(os.path.join(WORK_DIR, f'diag_{n}.png'), vis)

print(f'\n=== 汇总(12张): 平均mIoU {miou_sum/n:.4f} | '
      f'真值建筑像素 {bld_gt_tot/12/262144*100:.1f}% | '
      f'预测建筑像素 {bld_pred_tot/12/262144*100:.1f}% ===')
