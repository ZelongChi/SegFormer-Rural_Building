# -*- coding: utf-8 -*-
"""SegFormer 农村建筑提取 —— 本地可视化界面 (Gradio)
用法:  python app.py   然后浏览器打开 http://127.0.0.1:7860
"""
import os
import sys
import subprocess
import datetime
import traceback

import numpy as np
import cv2
import rasterio
import geopandas as gpd
import gradio as gr

ROOT = r'D:\Gaofen_Building_SegFormer'
PY = sys.executable
INFER = os.path.join(ROOT, 'postprocess', 'infer_tif.py')
VEC = os.path.join(ROOT, 'postprocess', 'mask_to_vector.py')
UI_DIR = os.path.join(ROOT, 'work_dirs', 'ui')


def _read_rgb(path):
    """读取影像为 RGB 显示图（自动拉伸到 2%-98% 分位）"""
    with rasterio.open(path) as s:
        if s.count >= 3:
            arr = s.read([1, 2, 3])
        else:
            arr = np.repeat(s.read(1)[None, :, :], 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    p = np.percentile(arr, [2, 98])
    arr = np.clip((arr - p[0]) / max(p[1] - p[0], 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)


def run_job(img_path, prob_thr, ws_thr, simplify_tol,
            progress=gr.Progress(track_tqdm=False)):
    """完整流程: 推理 -> 矢量化 -> 可视化 -> 统计"""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    job = os.path.join(UI_DIR, 'job_' + ts)
    os.makedirs(job, exist_ok=True)
    try:
        progress(0.05, desc='① 模型推理中...')
        r1 = subprocess.run(
            [PY, INFER, '--tif', img_path, '--out', job,
             '--prob_thr', str(prob_thr)],
            capture_output=True, text=True, timeout=3600)
        if r1.returncode != 0:
            return None, None, f'❌ 推理失败:\n{r1.stderr[-1500:]}', None, None

        progress(0.55, desc='② 拆分单栋建筑 + 矢量化...')
        shp = os.path.join(job, 'building.shp')
        r2 = subprocess.run(
            [PY, VEC, '--mask', os.path.join(job, 'result_mask.tif'),
             '--dom', img_path, '--out', shp,
             '--watershed_thresh', str(ws_thr),
             '--simplify_tol', str(simplify_tol)],
            capture_output=True, text=True, timeout=3600)
        if r2.returncode != 0:
            return None, None, f'❌ 矢量化失败:\n{r2.stderr[-1500:]}', None, None

        progress(0.85, desc='③ 生成可视化与统计...')
        rgb = _read_rgb(img_path)
        with rasterio.open(os.path.join(job, 'result_mask.tif')) as s:
            mask = s.read(1)
        vis = rgb.copy()
        cnts, _ = cv2.findContours(
            (mask == 1).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        vis_p = os.path.join(job, 'vis.png')
        cv2.imwrite(vis_p, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        mask_p = os.path.join(job, 'mask.png')
        cv2.imwrite(mask_p, (mask == 1).astype(np.uint8) * 255)

        log = '\n'.join([l for l in (r1.stdout + r2.stdout).splitlines()
                         if l.strip()][-10:])
        if not os.path.exists(shp):
            info = ('⚠️ **未检测到建筑**\n\n'
                    '可能原因与建议：\n'
                    '1. 影像中确实没有建筑 → 换一张含建筑的影像\n'
                    '2. 概率阈值过高导致漏检 → 调低到 0.30~0.35 重试\n'
                    '3. 建筑太小/太模糊 → 使用更高分辨率影像\n\n'
                    f'```\n{log}\n```')
            return vis_p, mask_p, info, None, None

        gdf = gpd.read_file(shp)
        n = len(gdf)
        if 'area_m2' in gdf.columns:
            total = float(gdf['area_m2'].sum())
        else:
            total = float(gdf.area.sum())
        info = (f'✅ 识别完成\n\n'
                f'**检测建筑：{n} 栋**\n\n'
                f'**总面积：{total:,.1f} m²**\n\n'
                f'```\n{log}\n```')
        dxf = os.path.splitext(shp)[0] + '.dxf'
        return vis_p, mask_p, info, dxf, shp
    except Exception as e:
        return None, None, f'❌ 出错:\n{traceback.format_exc()[-1200:]}', None, None


with gr.Blocks(title='SegFormer 农村建筑提取', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🏠 SegFormer 农村建筑提取')
    gr.Markdown(
        '上传遥感影像（tif / png / jpg），自动识别建筑并导出 **DXF / SHP** 矢量（可导入南方 CASS）。'
        '模型：SegFormer MiT-B2（Rural_Building_Dataset 训练，验证 mIoU 0.9024）')
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type='filepath', label='上传影像')
            with gr.Accordion('高级参数', open=False):
                prob = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                                 label='概率阈值（低=多检，高=少检）')
                ws = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                               label='拆分阈值（连片农房调低更细）')
                tol = gr.Slider(1.0, 8.0, value=3.0, step=0.5,
                                label='轮廓简化容差（大=更平滑）')
            btn = gr.Button('🚀 开始识别', variant='primary')
        with gr.Column(scale=2):
            out_vis = gr.Image(label='识别结果（绿色=建筑轮廓）')
            out_info = gr.Markdown()
    with gr.Row():
        out_mask = gr.Image(label='建筑掩码')
    with gr.Row():
        f_dxf = gr.File(label='⬇ DXF（南方CASS）')
        f_shp = gr.File(label='⬇ SHP')

    btn.click(run_job, [inp, prob, ws, tol],
              [out_vis, out_mask, out_info, f_dxf, f_shp])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7861, inbrowser=True,
                allowed_paths=[UI_DIR])
