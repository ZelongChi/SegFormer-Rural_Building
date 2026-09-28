# -*- coding: utf-8 -*-
"""
convert_to_yolo_seg.py —— 语义掩码 -> 实例标注（YOLO-seg格式），无需人工标注
"""
import os, glob, random
import cv2, numpy as np

IMG_SRC = r'D:\Rural_Building_Dataset\origin\input'
MSK_SRC = r'D:\Rural_Building_Dataset\origin\target'
OUT     = r'D:\Gaofen_Building_SegFormer\yolo_data'
MIN_AREA = 30
VAL_RATIO = 0.2
SEED = 42

def collect(d):
    r = {}
    for p in glob.glob(os.path.join(d, '*')):
        if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            r[os.path.splitext(os.path.basename(p))[0]] = p
    return r

def mask_to_polygons(mask):
    polys = []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        m = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 4:
                continue
            c = c.reshape(-1, 2).astype(np.float32)
            eps = 0.002 * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(c) >= 4:
                polys.append(c)
    return polys

def main():
    imgs = collect(IMG_SRC)
    msks = collect(MSK_SRC)
    names = sorted(set(imgs) & set(msks))
    if not names:
        print(f'[错误] 未找到匹配的影像/掩码: {IMG_SRC} 和 {MSK_SRC}')
        return
    random.seed(SEED); random.shuffle(names)
    n_val = max(1, int(len(names) * VAL_RATIO))
    val_names = set(names[:n_val]); train_names = names[n_val:]
    print(f'总共 {len(names)} 张 -> 训练 {len(train_names)} / 验证 {len(val_names)}')
    for split, part in (('train', train_names), ('val', val_names)):
        d_img = os.path.join(OUT, 'images', split)
        d_lbl = os.path.join(OUT, 'labels', split)
        os.makedirs(d_img, exist_ok=True); os.makedirs(d_lbl, exist_ok=True)
        n_img = 0; n_inst = 0
        for name in part:
            img = cv2.imread(imgs[name]); msk = cv2.imread(msks[name], 0)
            if img is None or msk is None:
                print(f'  [跳过] 读取失败: {name}'); continue
            H, W = msk.shape[:2]
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            msk_b = (msk > 0).astype(np.uint8)
            polys = mask_to_polygons(msk_b)
            if not polys:
                continue
            cv2.imwrite(os.path.join(d_img, name + '.png'), img)
            lines = []
            for p in polys:
                p[:, 0] = np.clip(p[:, 0] / W, 0, 1)
                p[:, 1] = np.clip(p[:, 1] / H, 0, 1)
                lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in p))
            with open(os.path.join(d_lbl, name + '.txt'), 'w') as f:
                f.write('\n'.join(lines))
            n_img += 1; n_inst += len(polys)
        print(f'{split}: {n_img} 张, 共 {n_inst} 个建筑实例')
    with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['building']\n")
    print('完成! 输出目录:', OUT)

if __name__ == '__main__':
    main()