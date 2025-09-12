# MMDetection3D 简化训练脚本

基于官方MMDetection3D框架，提供简化的KITTI数据集3D目标检测训练脚本，支持PointPillars、SECOND、Part-A2等主流模型。

## 🚀 快速开始

### 环境安装

#### 1. 创建Conda环境

```bash
conda create --name openmmlab python=3.11 -y
conda activate openmmlab
```

#### 2. 安装PyTorch

```bash
# 方法1：使用本地wheel文件（如果有的话）
pip install D:\download\torch-2.8.0+cu128-cp311-cp311-win_amd64.whl

# 方法2：从官方源安装
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 3. 安装OpenMMLab工具链

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
```

#### 4. 安装MMDetection3D

**重要：必须使用Visual Studio命令提示符**

1. 打开 **"x64 Native Tools Command Prompt for VS 2022"**
   - 在Windows开始菜单中搜索 "x64 Native Tools Command Prompt for VS 2022"
   - **以管理员身份运行**
   - ⚠️ 不要使用普通的PowerShell或CMD

2. 在VS命令提示符中执行：

```cmd
# 激活Conda基础环境
E:\Conda\condabin\conda.bat activate base

# 激活openmmlab环境
E:\Conda\condabin\conda.bat activate openmmlab

# 导航到项目目录
cd "D:\U盘备份\毕设\21网络1张金翔 毕业设计\基于RT-DETR的遥感卫星图像目标检测系统\mmdetection3d"

# 安装mmdet3d（解决matplotlib 3.5.3版本兼容性问题）
mim install "mmdet3d>=1.1.0rc0"
```

### 数据准备

1. **准备KITTI数据集**
   
   将KITTI数据集放置在以下目录结构：
   ```
   data/kitti/
   ├── ImageSets/
   │   ├── train.txt
   │   ├── val.txt
   │   └── test.txt
   ├── training/
   │   ├── calib/
   │   ├── image_2/
   │   ├── label_2/
   │   └── velodyne/
   └── testing/
       ├── calib/
       ├── image_2/
       └── velodyne/
   ```

2. **创建tools包（一次性设置）**
   
   首先，确保tools目录是一个Python包：
   ```bash
   # 如果没有tools/__init__.py文件，创建一个空文件
   New-Item -Path tools -Name __init__.py -ItemType File(在tools目录中新建一个__init__.py，里面什么内容都不要写)
   ```

3. **运行官方数据预处理**
   
   使用MMDetection3D官方提供的数据转换工具：
   ```bash
   python -m tools.create_data kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
   ```
   
   **注意**：已移除 `--with-plane` 参数，因为大多数KITTI数据集不包含地面平面信息。训练脚本会自动禁用地面平面使用以避免错误。
   
   这将生成以下文件：
   - `kitti_infos_train.pkl` - 训练集信息
   - `kitti_infos_val.pkl` - 验证集信息
   - `kitti_infos_test.pkl` - 测试集信息
   - `kitti_infos_trainval.pkl` - 训练验证集信息
   - `kitti_dbinfos_train.pkl` - 数据增强数据库
   - `kitti_gt_database/` - Ground Truth数据库
   - `training/velodyne_reduced/` - 预处理后的点云数据
   - `testing/velodyne_reduced/` - 测试集预处理后的点云数据

   **优势**：
   - ✅ 官方工具，兼容性最佳
   - ✅ 生成标准MMDetection3D v2格式
   - ✅ 无需自定义脚本，减少错误

## 📦 模型自动下载

所有训练脚本已集成模型自动下载功能：

### 自动下载特性

- **🚀 启动即下载**：首次运行训练时自动下载所需的预训练权重
- **📁 统一存储**：模型自动保存到 `./models/` 目录下，按模型类型分类
- **⚡ 智能检测**：已存在的模型文件不会重复下载
- **📊 进度显示**：下载过程显示进度条和速度

### 支持的模型权重

#### 基于图像的模型权重
- **SMOKE**：DLA34骨干网络 + KITTI预训练权重（推荐新手入门）
- **MonoFlex**：DLA34骨干网络 + KITTI预训练权重（高精度单目检测）
- **FCOS3D**：ResNet-101 + nuScenes预训练权重（锚框自由检测）
- **ImVoxelNet**：ResNet-50 + KITTI预训练权重（图像到体素投影）

#### 基于点云的模型权重
- **PointPillars**：KITTI 3类和Car类检测权重（点云检测，推荐入门）
- **SECOND**：KITTI 3类和Car类检测权重（点云检测，经典模型）
- **Part-A2**：KITTI 3类检测权重（点云检测，两阶段）
- **CenterPoint**：nuScenes预训练权重（点云检测，锚框自由）
- **PV-RCNN**：KITTI 3类检测权重（点云检测，高精度）

## 🚀 快速开始（推荐新手）

如果您是3D目标检测的新手，强烈推荐从**SMOKE**模型开始：

### 1. 数据准备
```bash
# 使用官方工具处理KITTI数据（一次性操作）
python -m tools.create_data kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### 2. 训练SMOKE模型
```bash
# 一键训练（自动下载预训练权重 + AMP加速）
python train_SMOKE.py --dataset kitti-mono3d --amp
```

### 3. 快速验证效果
```bash
# 用您的图像验证检测效果
python pred_image_simple.py --img test1.png

# 查看输出结果：result_SMOKE_3d.png
```

**为什么选择SMOKE作为入门？**
- 🖼️ **只需图像**：无需复杂的点云数据处理
- ⚡ **训练快速**：相比点云模型训练时间更短
- 💾 **资源友好**：普通GPU即可运行，显存需求低
- 🎯 **验证简单**：直接输出可视化结果图像

---

## 🎯 模型训练

### 支持的模型

#### 🖼️ **基于图像的模型（推荐新手）**

| 模型 | 数据集支持 | 训练脚本 | 特点 |
|------|-----------|----------|------|
| **SMOKE** | KITTI | `train_SMOKE.py` | 🌟**新手首选**：单阶段检测，快速简单 |
| **MonoFlex** | KITTI | `train_MonoFlex.py` | 🎯**高精度**：柔性检测，处理截断物体更好 |
| **FCOS3D** | nuScenes | `train_FCOS3D.py` | 🆓**锚框自由**：全卷积单阶段检测 |
| **ImVoxelNet** | KITTI | `train_ImVoxelNet.py` | 🔄**图像到体素**：投影到3D空间检测 |

#### ☁️ **基于点云的模型**

| 模型 | 数据集支持 | 训练脚本 | 特点 |
|------|-----------|----------|------|
| **PointPillars** | KITTI, nuScenes, Waymo | `train_PointPillars.py` | 🔰**点云入门**：简单快速，易于理解 |
| **SECOND** | KITTI, Waymo | `train_SECOND.py` | 📚**经典模型**：平衡速度和精度 |
| **Part-A2** | KITTI | `train_PartA2.py` | 🎯**高精度**：两阶段检测，精度优秀 |
| **PV-RCNN** | KITTI | `train_PV_RCNN.py` | 🏆**最高精度**：点体素融合，业界顶尖 |
| **CenterPoint** | nuScenes | `train_CenterPoint.py` | 🆓**锚框自由**：基于中心点检测 |
| **3DSSD** | KITTI | `train_3DSSD.py` | ⚡**高效率**：单阶段快速检测 |
| **VoteNet** | ScanNet, SUNRGBD | `train_VoteNet.py` | 🏠**室内场景**：专为室内设计 |

### 训练命令

#### PointPillars（推荐新手）

```bash
# KITTI 3类检测（Car, Pedestrian, Cyclist）
python train_PointPillars.py --dataset kitti-3class --amp

# KITTI 汽车检测（仅Car类）
python train_PointPillars.py --dataset kitti-car --amp

# 自定义工作目录
python train_PointPillars.py --dataset kitti-3class --work-dir ./my_experiment --amp
```

#### SECOND

```bash
# KITTI 3类检测
python train_SECOND.py --dataset kitti-3class --amp

# KITTI 汽车检测
python train_SECOND.py --dataset kitti-car --amp
```

#### Part-A2

```bash
# KITTI 3类检测
python train_PartA2.py --dataset kitti-3class --amp

# KITTI 汽车检测
python train_PartA2.py --dataset kitti-car --amp
```

#### 其他模型

```bash
# PV-RCNN（高精度）
python train_PV_RCNN.py --amp

# 3DSSD（高效率）
python train_3DSSD.py --amp

# CenterPoint（多模态融合）
python train_CenterPoint.py --amp

# VoteNet（室内场景，需要相应数据集）
python train_VoteNet.py --dataset scannet --amp
```

#### SMOKE单目3D检测（推荐新手）

**SMOKE**是基于图像的单目3D目标检测模型，特别适合初学者：

```bash
# SMOKE单目3D检测（基于图像，自动下载DLA34权重）
python train_SMOKE.py --dataset kitti-mono3d --amp
```

**单目3D检测模型对比**：

| 模型 | 特点 | 精度 | 适用场景 |
|------|------|------|----------|
| **SMOKE** | 🌟 单阶段，快速简单 | 中等 | 新手入门，快速验证 |
| **MonoFlex** | 🎯 柔性检测，处理截断物体 | 较高 | 需要更高精度的应用 |
| **FCOS3D** | 🆓 锚框自由，全卷积 | 高 | nuScenes大规模数据集 |
| **ImVoxelNet** | 🔄 图像到体素投影 | 中等 | 多视角场景 |

**共同优势**：
- ✅ **输入简单**：只需要RGB图像，无需点云数据
- ✅ **资源友好**：相比点云模型，显存和计算需求更低
- ✅ **部署容易**：不依赖昂贵的激光雷达硬件
- ⚠️ **精度限制**：单目检测精度通常低于基于点云的方法

#### 全部训练命令汇总

```bash
# 一键复制运行（选择其中一个）

# PointPillars（推荐新手，基于点云）
python train_PointPillars.py --dataset kitti-3class --amp

# SECOND（基于点云）
python train_SECOND.py --dataset kitti-3class --amp

# Part-A2（基于点云）
python train_PartA2.py --dataset kitti-3class --amp

# 基于图像的单目3D检测模型
python train_SMOKE.py --dataset kitti-mono3d        # 推荐新手
python train_MonoFlex.py --dataset kitti-mono3d     # 高精度
python train_FCOS3D.py --dataset nus-mono3d         # nuScenes数据集
python train_ImVoxelNet.py --dataset kitti-3d-car   # 图像到体素

# PV-RCNN（高精度，基于点云）
python train_PV_RCNN.py --amp

# 3DSSD（高效率，基于点云）
python train_3DSSD.py --amp

# CenterPoint（基于点云）
python train_CenterPoint.py --amp

# VoteNet（需要相应数据集）
python train_VoteNet.py --dataset scannet --amp
```

### 📋 训练参数配置

#### 基本参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 选择数据集配置 | `kitti-3class`, `kitti-car` |
| `--config` | 自定义配置文件 | `configs/pointpillars/xxx.py` |
| `--work-dir` | 自定义输出目录 | `./my_experiment` |
| `--amp` | 🚀 **启用混合精度训练（强烈推荐）** | - |
| `--resume auto` | 自动恢复训练 | - |
| `--cfg-options` | 覆盖配置参数 | 见下方详细说明 |

#### 🔧 自定义训练参数（通过--cfg-options）

**修改训练轮数（Epoch）：**
```bash
# 设置训练80个epoch
python train_PointPillars.py --dataset kitti-3class --amp --cfg-options train_cfg.max_epochs=80

# 设置训练200个epoch（长时间训练）
python train_SMOKE.py --dataset kitti-mono3d --amp --cfg-options train_cfg.max_epochs=200
```

**修改批次大小（Batch Size）：**
```bash
# 设置batch_size为4（显存小的情况）
python train_PointPillars.py --dataset kitti-3class --amp --cfg-options train_dataloader.batch_size=4

# 设置batch_size为16（显存大的情况）
python train_SMOKE.py --dataset kitti-mono3d --amp --cfg-options train_dataloader.batch_size=16
```

**修改学习率：**
```bash
# 设置初始学习率
python train_PointPillars.py --dataset kitti-3class --amp --cfg-options optim_wrapper.optimizer.lr=0.001
```

**组合多个参数：**
```bash
# 同时修改epoch、batch_size和学习率
python train_PointPillars.py --dataset kitti-3class --amp \
  --cfg-options train_cfg.max_epochs=100 \
                train_dataloader.batch_size=8 \
                optim_wrapper.optimizer.lr=0.002
```

#### 🚀 AMP混合精度训练支持

**所有训练脚本均已支持AMP混合精度训练！**

| 模型类型 | 训练脚本 | AMP支持 | 推荐使用 |
|---------|---------|---------|----------|
| **LiDAR模型** | `train_PointPillars.py` | ✅ | 🌟 推荐新手 |
| | `train_SECOND.py` | ✅ | 高性能 |
| | `train_PartA2.py` | ✅ | 两阶段检测 |
| | `train_PV_RCNN.py` | ✅ | 最高精度 |
| | `train_3DSSD.py` | ✅ | 单阶段高效 |
| | `train_CenterPoint.py` | ✅ | 中心点检测 |
| **单目模型** | `train_SMOKE.py` | ✅ | 🌟 推荐新手 |
| | `train_MonoFlex.py` | ✅ | 高精度 |
| | `train_FCOS3D.py` | ✅ | nuScenes |
| | `train_ImVoxelNet.py` | ✅ | 图像到体素 |
| **点云分割** | `train_VoteNet.py` | ✅ | 点云场景理解 |

**💡 重要提示**：
- ✅ **强烈推荐使用`--amp`**：可显著加速训练并节省显存（通常能节省30-50%显存）
- ⚠️ **批次大小调整**：显存不足时降低batch_size，显存充足时可适当增加
- 🎯 **训练轮数建议**：KITTI数据集通常80-100个epoch足够，复杂数据集可增加到200+
- 📊 **学习率调整**：默认学习率通常已经调优，除非有特殊需求否则不建议修改

## 📁 输出目录结构

训练结果自动保存到 `runs/outputs/` 目录：

```
runs/outputs/
├── PointPillars_kitti-3class/     # PointPillars KITTI 3类
│   ├── 20241212_140000.log        # 训练日志
│   ├── pointpillars_xxx.py        # 配置文件副本
│   ├── epoch_1.pth                # 模型检查点
│   ├── epoch_2.pth
│   └── latest.pth                 # 最新检查点
├── SECOND_kitti-3class/           # SECOND KITTI 3类
├── PartA2_kitti-3class/           # Part-A2 KITTI 3类
└── PV_RCNN/                       # PV-RCNN
```

## 🔧 常见问题解决

### 1. 环境问题

**问题**: `ImportError: cannot import name 'Config' from 'mmcv'`
```bash
# 解决方案：重新安装正确版本的mmcv
mim install "mmcv>=2.0.0,<2.2.0"
```

**问题**: `TypeError: The annotations loaded from annotation file should be a dict`
```bash
# 解决方案：重新运行官方数据预处理
python -m tools.create_data kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### 2. 训练问题

**问题**: CUDA内存不足
```bash
# 解决方案：使用混合精度训练和减小batch size
python train_PointPillars.py --dataset kitti-3class --amp --cfg-options train_dataloader.batch_size=2
```

**问题**: 训练速度慢
```bash
# 解决方案：启用混合精度训练
python train_PointPillars.py --dataset kitti-3class --amp
```

### 3. 数据问题

**问题**: 找不到数据文件
- 确保KITTI数据集放在 `data/kitti/` 目录下
- 运行 `python -m tools.create_data kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti` 生成必要的信息文件

## 📊 性能参考

在RTX 5060 Laptop GPU上的训练时间参考：

| 模型 | 数据集 | 训练时间 | 显存占用 | mAP |
|------|--------|----------|----------|-----|
| PointPillars | KITTI 3类 | ~8-12小时 | ~6GB | ~60% |
| SECOND | KITTI 3类 | ~6-10小时 | ~8GB | ~65% |
| Part-A2 | KITTI 3类 | ~10-15小时 | ~10GB | ~70% |
| PV-RCNN | KITTI 3类 | ~15-20小时 | ~12GB | ~75% |

*注意：具体时间和性能取决于硬件配置和数据集大小*

## 🎯 推荐训练流程

### 新手推荐
1. 从PointPillars开始：`python train_PointPillars.py --dataset kitti-3class --amp`
2. 验证环境和数据无误后，尝试其他模型

### 高精度需求
1. 使用PV-RCNN：`python train_PV_RCNN.py --amp`
2. 或Part-A2：`python train_PartA2.py --dataset kitti-3class --amp`

### 快速实验
1. 使用汽车单类检测：`python train_PointPillars.py --dataset kitti-car --amp`
2. 训练时间更短，适合调试和验证

## 📝 模型测试

训练完成后，可以使用简化的测试脚本评估模型性能：

### 测试命令

#### 🖼️ 基于图像的模型测试（推荐新手优先尝试）

**为什么推荐单目模型作为入门？**
- ✅ **门槛最低**：只需要RGB图像，无需复杂的点云数据
- ✅ **验证简单**：可以直接用test1.png快速验证效果
- ✅ **资源友好**：显存需求小，普通GPU即可运行
- ✅ **理解容易**：单目3D检测原理相对简单

```bash
# SMOKE - 推荐新手第一次尝试
python test_SMOKE.py --dataset kitti-mono3d

# MonoFlex - 更高精度的单目检测
python test_MonoFlex.py --dataset kitti-mono3d

# FCOS3D - nuScenes数据集（如果有的话）
python test_FCOS3D.py --dataset nus-mono3d

# ImVoxelNet - 图像到体素投影
python test_ImVoxelNet.py --dataset kitti-3d-car

# 快速图像推理验证
python pred_image_simple.py --img test1.png
```

#### PointPillars测试

```bash
# 测试KITTI 3类检测模型
python test_PointPillars.py --dataset kitti-3class

# 测试KITTI 汽车检测模型
python test_PointPillars.py --dataset kitti-car

# 自定义配置和检查点
python test_PointPillars.py --config configs/pointpillars/xxx.py --checkpoint path/to/checkpoint.pth
```

#### SECOND测试

```bash
# 测试KITTI 3类检测模型
python test_SECOND.py --dataset kitti-3class

# 测试KITTI 汽车检测模型
python test_SECOND.py --dataset kitti-car
```

#### 其他模型测试

```bash
# Part-A2
python test_PartA2.py --dataset kitti-3class

# SMOKE单目3D检测
python test_SMOKE.py --dataset kitti-mono3d

# PV-RCNN
python test_PV_RCNN.py

# 3DSSD
python test_3DSSD.py

# CenterPoint
python test_CenterPoint.py

# VoteNet
python test_VoteNet.py --dataset scannet
```

### 测试参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 选择数据集配置 | `kitti-3class`, `kitti-car` |
| `--config` | 自定义配置文件 | `configs/pointpillars/xxx.py` |
| `--checkpoint` | 自定义检查点文件 | `runs/outputs/xxx/latest.pth` |
| `--show` | 显示可视化结果 | - |
| `--show-dir` | 保存可视化结果目录 | `./test_results` |
| `--score-thr` | 检测分数阈值 | `0.3` |

## 🔮 模型推理

### 🖼️ 单目3D检测推理（推荐新手首次尝试）

**最简单的3D检测体验**：只需要一张图片即可进行3D目标检测！

#### 一键式推理（推荐新手）

```bash
# 最简单的推理方式（自动处理test1.png）
python pred_image_simple.py --model smoke

# 指定其他模型
python pred_image_simple.py --model monoflex
python pred_image_simple.py --model fcos3d
python pred_image_simple.py --model imvoxelnet

# 自定义图像和置信度阈值
python pred_image_simple.py --model smoke --img your_image.jpg --score-thr 0.1

# 输出：result_{model}_{imgname}.png（带3D框的检测结果）
```

#### 模型专用推理脚本

```bash
# SMOKE推理（推荐新手）
python pred_SMOKE.py --dataset kitti-mono3d --img test1.png

# MonoFlex推理（高精度）
python pred_MonoFlex.py --dataset kitti-mono3d --img test1.png

# FCOS3D推理（nuScenes数据集）
python pred_FCOS3D.py --dataset nus-mono3d --img test1.png

# ImVoxelNet推理（图像到体素投影）
python pred_ImVoxelNet.py --dataset kitti-3d-car --img test1.png

# 自动选择示例图像
python pred_SMOKE.py --dataset kitti-mono3d
python pred_MonoFlex.py --dataset kitti-mono3d
python pred_FCOS3D.py --dataset nus-mono3d
python pred_ImVoxelNet.py --dataset kitti-3d-car
```

**单目推理特点**：
- ✅ **输入简单**：任何RGB图像文件即可
- ✅ **输出直观**：生成带3D边界框的可视化图像  
- ✅ **快速验证**：几秒钟即可看到检测效果
- ✅ **无需点云**：不需要复杂的激光雷达数据

### ☁️ 基于点云的模型推理

#### PointPillars推理

```bash
# 自动选择示例点云文件
python pred_PointPillars.py --dataset kitti-3class

# 指定点云文件
python pred_PointPillars.py --pcd data/kitti/training/velodyne_reduced/000001.bin --dataset kitti-3class

# 保存结果到指定目录
python pred_PointPillars.py --dataset kitti-3class --out-dir ./inference_results
```

#### 其他模型推理

```bash
# SECOND
python pred_SECOND.py --dataset kitti-3class

# Part-A2
python pred_PartA2.py --dataset kitti-3class

# PV-RCNN（高精度）
python pred_PV_RCNN.py

# 3DSSD（高效率）
python pred_3DSSD.py
```

### 推理参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--pcd` | 点云文件路径 | `data/kitti/training/velodyne_reduced/000001.bin` |
| `--dataset` | 数据集配置 | `kitti-3class`, `kitti-car` |
| `--config` | 自定义配置文件 | `configs/pointpillars/xxx.py` |
| `--checkpoint` | 自定义检查点文件 | `runs/outputs/xxx/latest.pth` |
| `--pred-score-thr` | 预测分数阈值 | `0.3` |
| `--out-dir` | 输出目录 | `./inference_results` |
| `--show` | 显示结果窗口 | - |

### 推理结果

推理完成后，结果会保存为：
- `result_PointPillars_kitti-3class.png` - 可视化结果图像（保存到项目根目录）
- `outputs/vis_data/` - 详细的可视化结果
- `outputs/pred/` - 预测结果JSON文件

### 使用官方测试脚本

如需使用官方测试脚本：

```bash
python tools/test.py \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    runs/outputs/PointPillars_kitti-3class/latest.pth
```

## 🤝 技术支持

如遇问题，请：
1. 检查环境安装是否正确
2. 确认数据预处理是否完成
3. 查看训练日志中的错误信息
4. 参考官方MMDetection3D文档

## 📚 相关资源

- [MMDetection3D官方文档](https://mmdetection3d.readthedocs.io/)
- [KITTI数据集](http://www.cvlibs.net/datasets/kitti/)
- [OpenMMLab](https://openmmlab.com/)

---

**简化特性**：
- ✅ 官方数据预处理工具
- ✅ 简化的训练命令
- ✅ 自动输出目录管理
- ✅ 混合精度训练支持
- ✅ 多模型统一接口
- ✅ 详细的错误处理指南
