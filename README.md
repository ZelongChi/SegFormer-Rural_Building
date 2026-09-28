# ============================================================================
# SegFormer MiT-B2 农村建筑物提取（Gaofen 数据集）—— D盘部署版
# 实现: HuggingFace transformers（模型=SegFormer MiT-B2，与 mmseg 版效果一致）
# 环境: Windows / 系统 Python 3.10 / RTX2070(8G) / torch 2.4.1+cu121
# ============================================================================

## 一、环境（已配置完成 ✅）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10.11 | C:\Users\XXY011\AppData\Local\Programs\Python\Python310 |
| torch | 2.4.1+cu121 | CUDA 可用，RTX2070 已识别 |
| transformers | 4.44.2 | SegFormer 官方实现 |
| rasterio/shapely/ezdxf/geopandas | 已装 | 矢量导出 |
| 预训练权重 | mit-b2 (ADE) | 首次训练自动从 HF 镜像下载到 D:\Gaofen_Building_SegFormer\hf_cache |

> 说明：原计划用 mmseg 框架，但其 Windows 无预编译包、需 VS 编译器，
> 已改用 transformers 实现 SegFormer MiT-B2——模型结构、训练效果一致，零额外安装。

## 二、数据准备

```
D:\Gaofen_Dataset
├─ img\      # 原图（png/tif，与掩码同名）
├─ mask\     # 二值掩码（0背景，建筑像素>0即可，如255或1）
└─ split\    # train.txt / val.txt / test.txt（每行一个图片文件名）
```

- 若原始数据是"原图+掩码"：图放 img\，掩码放 mask\（同名 .png）
- 若原始数据是"原图+SHP"：运行 `python tools\prepare_dataset.py` 自动转掩码
- 划分数据集：`python tools\split_dataset.py`（7:2:1，按影像文件划分）

## 三、训练

```powershell
cd D:\Gaofen_Building_SegFormer
C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe tools\train.py
```
- 首次运行自动下载 MiT-B2 预训练权重（约 125MB，走 hf-mirror 国内镜像，缓存到 D 盘）
- 监控：每轮打印 loss，每 5 轮验证 mIoU，最优权重自动存为 best_mIoU.pth
- 输出：work_dirs\segformer_mitb2\best_mIoU.pth

## 四、推理与矢量导出（对接南方CASS）

1. 大图推理（重叠切片，输出二值掩码+概率图）：
   ```powershell
   C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe postprocess\infer_tif.py --tif D:\Gaofen_Dataset\test_dom.tif
   ```
2. 掩码 -> 单栋建筑矢量（分水岭拆分连片农房 + 直角规整 + SHP/DXF）：
   ```powershell
   C:\Users\XXY011\AppData\Local\Programs\Python\Python310\python.exe postprocess\mask_to_vector.py --mask D:\Gaofen_Building_SegFormer\work_dirs\result\result_mask.tif --dom D:\Gaofen_Dataset\test_dom.tif
   ```
   输出 building.shp + building.dxf，CASS 直接打开。

### 关键参数调优
| 参数 | 位置 | 作用 |
|---|---|---|
| watershed_thresh | mask_to_vector.py | 分水岭前景阈值，连片农房多调低(0.25~0.35)，易过分割调高 |
| min_area_px | mask_to_vector.py | 剔除小噪点面积 |
| simplify_tol | mask_to_vector.py | 轮廓简化容差(像素) |
| prob_thr | infer_tif.py | 建筑判定概率阈值(默认0.45) |
| EPOCHS / BATCH_SIZE / LR | tools/train.py | 训练超参（显存不足 BATCH_SIZE 改 2） |

## 五、注意
1. 全路径均在 D 盘，勿用中文路径；
2. RTX2070 8G：BATCH_SIZE=4, IMG_SIZE=512 为安全配置（混合精度已开启）；
3. 掩码中建筑像素 >0 即视为建筑（255/1 均可）；
4. 权重下载走 hf-mirror 镜像，若想换源修改脚本顶部 HF_ENDPOINT。
