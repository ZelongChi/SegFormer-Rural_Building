# -*- coding: utf-8 -*-
"""YOLO-seg 单栋建筑识别训练脚本"""
from ultralytics import YOLO


def main():
    m = YOLO(r'D:\Gaofen_Building_SegFormer\yolo11s-seg.pt')
    m.train(
        data=r'D:\Gaofen_Building_SegFormer\yolo_data\data.yaml',
        project=r'D:\Gaofen_Building_SegFormer\yolo_data',
        name='seg_building',
        epochs=100, imgsz=512, batch=8,
        device=0, workers=0, seed=42, patience=20,
    )
    print('训练完成')


if __name__ == '__main__':
    main()
