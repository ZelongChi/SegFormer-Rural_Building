# -*- coding: utf-8 -*-
"""
SegFormer MiT-B2 农村建筑语义分割训练（HuggingFace transformers 实现）
======================================================================
任务：二分类语义分割（0=背景, 1=建筑），Gaofen 高分农村建筑数据集
环境：系统 Python 3.10 / torch 2.4.1+cu121 / transformers 4.44.2

用法：
  python tools/train.py

输出：
  D:\Gaofen_Building_SegFormer\work_dirs\segformer_mitb2\
    ├─ best_mIoU.pth     # 最优权重（推理脚本默认加载它）
    └─ last.pth
"""
import os
import time
import argparse

# 权重缓存与下载镜像全部放 D 盘
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'D:\Gaofen_Building_SegFormer\hf_cache'

import cv2
import numpy as np
import albumentations as A
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ============ 配置 ============
DATA_ROOT = r'D:\Gaofen_Dataset'
WORK_DIR = r'D:\Gaofen_Building_SegFormer\work_dirs\segformer_mitb2'
PRETRAINED = 'nvidia/segformer-b2-finetuned-ade-512-512'  # MiT-B2 预训练(ADE20K)
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 60
VAL_INTERVAL = 5
PATIENCE = 8              # 早停：val mIoU 连续 N 次验证不提升则停止
LR = 1.5e-4
BACKBONE_LR_MULT = 0.1
NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(WORK_DIR, exist_ok=True)

# 数据增强（albumentations）：只做不破坏几何的增强（旋转90°倍数、翻转）
# 不做缩放/透视/任意角度旋转——高分影像建筑几何关系必须保持
# 注意：增强过强会拖慢 head 学习，这里刻意从弱开始
TRAIN_AUG = A.Compose([
    A.RandomRotate90(p=0.3),
    A.HorizontalFlip(p=0.3),
])


class BuildingDataset(Dataset):
    """读取 D:\Gaofen_Dataset 下 img/mask + split/*.txt"""

    def __init__(self, split, processor, size=IMG_SIZE, train=True):
        self.train = train
        self.size = size
        with open(os.path.join(DATA_ROOT, 'split', f'{split}.txt'),
                  encoding='utf-8') as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.processor = processor

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # 纯背景样本降权：50% 概率保留（正则化抑制误检），否则重采样
        if self.train and np.random.rand() > 0.5:
            name0 = self.names[idx]
            base0 = os.path.splitext(name0)[0]
            probe = None
            for ext in ('.png', '.tif', '.tiff'):
                p = os.path.join(DATA_ROOT, 'mask', base0 + ext)
                if os.path.exists(p):
                    probe = p
                    break
            if probe is not None:
                m0 = cv2.imread(probe, 0)
                if m0 is not None and m0.max() == 0:
                    return self[(idx + np.random.randint(1, len(self))) % len(self)]

        name = self.names[idx]
        base = os.path.splitext(name)[0]

        img = cv2.imread(os.path.join(DATA_ROOT, 'img', name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 掩码：支持 .png / .tif（与影像同名）
        mask_path = None
        for ext in ('.png', '.tif', '.tiff'):
            p = os.path.join(DATA_ROOT, 'mask', base + ext)
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
            raise FileNotFoundError(f'缺少掩码: {base}')
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 0).astype(np.int64)          # 建筑像素 -> 1

        h, w = img.shape[:2]
        if self.train:
            # 随机裁剪（保证裁剪内非背景占比不过高）
            for _ in range(10):
                if h >= self.size and w >= self.size:
                    y0 = np.random.randint(0, h - self.size + 1)
                    x0 = np.random.randint(0, w - self.size + 1)
                    crop = mask[y0:y0 + self.size, x0:x0 + self.size]
                    if crop.mean() < 0.95:
                        break
                else:
                    y0 = x0 = 0
                    break
            else:
                y0 = np.random.randint(0, max(h - self.size, 0) + 1)
                x0 = np.random.randint(0, max(w - self.size, 0) + 1)

            img = img[y0:y0 + self.size, x0:x0 + self.size]
            mask = mask[y0:y0 + self.size, x0:x0 + self.size]

            # albumentations 增强（img/mask 同步）
            aug = TRAIN_AUG(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        else:
            # 验证：resize 到 512
            img = cv2.resize(img, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size),
                              interpolation=cv2.INTER_NEAREST)

        encoded = self.processor(
            images=img, segmentation_maps=mask, return_tensors='pt')
        return encoded['pixel_values'].squeeze(0), encoded['labels'].squeeze(0)


def compute_miou(pred, label, num_classes=NUM_CLASSES):
    """像素级 mIoU"""
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (label == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(float(inter) / float(union) if union > 0 else 1.0)
    return float(np.mean(ious))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    miou_sum, n = 0.0, 0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        out = model(pixel_values=pixel_values)
        # transformers logits 为输入的 1/4 分辨率，resize 回原尺寸
        logits = nn.functional.interpolate(
            out.logits, size=labels.shape[-2:], mode='bilinear',
            align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        for i in range(pred.shape[0]):
            miou_sum += compute_miou(pred[i], labels[i].numpy())
            n += 1
    model.train()
    return miou_sum / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批次大小')
    ap.add_argument('--lr', type=float, default=LR, help='学习率')
    ap.add_argument('--val_interval', type=int, default=VAL_INTERVAL)
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='早停耐心值（epoch），val mIoU 连续不升则停')
    ap.add_argument('--amp', action='store_true',
                    help='启用混合精度(fp16)。默认关闭：fp32更稳，SegFormer在fp16下可能loss溢出为nan')
    ap.add_argument('--resume', action='store_true',
                    help='从 work_dirs 的 last.pth 断点续训')
    args = ap.parse_args()
    epochs, batch_size, lr = args.epochs, args.batch_size, args.lr
    use_amp = args.amp and DEVICE == 'cuda'

    print(f'[Info] Device: {DEVICE} | torch {torch.__version__}')
    if DEVICE == 'cuda':
        print(f'[Info] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[Info] 精度: {"AMP(fp16)" if use_amp else "FP32"} | 早停 patience={args.patience}')

    processor = SegformerImageProcessor.from_pretrained(PRETRAINED)
    # 关键修复：ADE 预训练的 processor 默认 do_reduce_labels=True（label-1），
    # 会把"建筑=1"错位成 0、背景转成 255(忽略)，导致验证 mIoU 完全失真。
    # 本任务只有 2 类(背景/建筑)，必须关闭该选项，保持 0=背景, 1=建筑。
    processor.do_reduce_labels = False
    train_ds = BuildingDataset('train', processor, train=True)
    val_ds = BuildingDataset('val', processor, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f'[Info] 训练 {len(train_ds)} 张 / 验证 {len(val_ds)} 张 | '
          f'epochs={epochs} batch={batch_size} lr={lr}')

    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={0: 'background', 1: 'building'},
        label2id={'background': 0, 'building': 1},
    ).to(DEVICE)

    # 优化器：backbone 学习率 x0.1，decode head 全量
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith('segformer.encoder')
         else head_params).append(p)
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * BACKBONE_LR_MULT},
        {'params': head_params, 'lr': lr},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_miou = 0.0
    nan_count = 0
    no_improve = 0        # 早停计数
    start_epoch = 1
    start = time.time()

    # ---------- 断点续训 ----------
    ckpt_path = os.path.join(WORK_DIR, 'last.pth')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if 'model' not in ckpt:
            raise SystemExit('[错误] last.pth 是旧格式(纯权重)，无法续训。'
                             '请删除该文件后从头训练，或使用最新版脚本产生的断点。')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        best_miou = ckpt.get('best_miou', 0.0)
        no_improve = ckpt.get('no_improve', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'[Resume] 从 Epoch {ckpt.get("epoch", 0)} 续训，'
              f'已保存最优 mIoU={best_miou:.4f}')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        valid_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast('cuda', enabled=use_amp):
                out = model(pixel_values=pixel_values)
                logits = nn.functional.interpolate(
                    out.logits, size=labels.shape[-2:], mode='bilinear',
                    align_corners=False)
                loss = criterion(logits, labels)

            # NaN 保护：跳过异常批次，避免污染训练
            if not torch.isfinite(loss):
                nan_count += 1
                pbar.set_postfix(loss=f'NAN(skip {nan_count})')
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / valid_batches if valid_batches else float('nan')
        print(f'  Epoch {epoch} 平均loss: {avg_loss:.4f}'
              + (f' (跳过nan批次 {nan_count})' if nan_count else ''))

        if epoch % args.val_interval == 0:
            miou = evaluate(model, val_loader, DEVICE)
            print(f'  [Val] Epoch {epoch} mIoU: {miou:.4f}')
            if miou > best_miou:
                best_miou = miou
                no_improve = 0
                torch.save(model.state_dict(),
                           os.path.join(WORK_DIR, 'best_mIoU.pth'))
                print(f'  [Save] 新最优 mIoU={miou:.4f} -> best_mIoU.pth')
            else:
                no_improve += args.val_interval
                print(f'  [EarlyStop] 连续 {no_improve} epoch 未提升'
                      f' (patience={args.patience})')
                if no_improve >= args.patience:
                    print(f'  触发早停，停止训练。最优 mIoU={best_miou:.4f}')
                    break

        # 每 epoch 保存完整 checkpoint（含优化器，供断点续训）
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_miou': best_miou,
            'no_improve': no_improve,
        }, ckpt_path)

    print(f'\n[完成] 训练结束，耗时 {(time.time()-start)/60:.1f} 分钟，'
          f'最优 mIoU={best_miou:.4f}，跳过nan批次 {nan_count} 次')
    print(f'权重: {WORK_DIR}\\best_mIoU.pth')
    print(f'断点(可 --resume 续训): {WORK_DIR}\\last.pth')


if __name__ == '__main__':
    main()
