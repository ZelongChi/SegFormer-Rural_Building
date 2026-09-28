# -*- coding: utf-8 -*-
"""
按【原图组】划分 train/val/test（防止数据泄漏）
========================================================================
Rural_Building_Dataset 的 augmented 包含 origin 的 5 种增强变体
(_horflip/_verflip/_rorate90/_rorate180/_rorate270)。
若随机划分，同一建筑的增强变体会同时进 train 和 val，导致指标虚高。

本脚本按"原图基名"分组，整组进入同一集合，保证泛化评估真实。

用法: python tools/split_dataset.py
"""
import os
import random

img_dir = r'D:\Gaofen_Dataset\img'
mask_dir = r'D:\Gaofen_Dataset\mask'
split_dir = r'D:\Gaofen_Dataset\split'
os.makedirs(split_dir, exist_ok=True)

random.seed(42)

AUG_SUFFIXES = ('_horflip', '_verflip', '_rorate90', '_rorate180', '_rorate270')


def base_key(name):
    """去掉增强后缀，得到原图基名"""
    base = os.path.splitext(name)[0]
    for s in AUG_SUFFIXES:
        if base.endswith(s):
            return base[:-len(s)]
    return base


img_list = [f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.tif', '.jpg'))]

# 只保留有对应掩码的
valid = []
for f in img_list:
    base = os.path.splitext(f)[0]
    if any(os.path.exists(os.path.join(mask_dir, base + e))
           for e in ('.png', '.tif', '.tiff')):
        valid.append(f)
img_list = valid

# 按基名分组
groups = {}
for f in img_list:
    groups.setdefault(base_key(f), []).append(f)
group_keys = list(groups.keys())
random.shuffle(group_keys)

n = len(group_keys)
n_train = int(n * 0.7)
n_val = int(n * 0.2)
train_keys = group_keys[:n_train]
val_keys = group_keys[n_train:n_train + n_val]
test_keys = group_keys[n_train + n_val:]


def write_txt(keys, savepath):
    files = []
    for k in keys:
        files.extend(sorted(groups[k]))
    with open(savepath, 'w', encoding='utf-8') as f:
        for name in files:
            f.write(name + '\n')
    return len(files)


n_train_f = write_txt(train_keys, os.path.join(split_dir, 'train.txt'))
n_val_f = write_txt(val_keys, os.path.join(split_dir, 'val.txt'))
n_test_f = write_txt(test_keys, os.path.join(split_dir, 'test.txt'))

print(f'[完成] 原图组 {n} 个 -> train {len(train_keys)}组/{n_train_f}张, '
      f'val {len(val_keys)}组/{n_val_f}张, test {len(test_keys)}组/{n_test_f}张')
print(f'列表已写入: {split_dir}')
