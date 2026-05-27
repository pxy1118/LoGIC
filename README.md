# LoGIC

本项目是论文 **LoGIC-Design: A Low-Dimensional Grid-Controlled Framework for Heterogeneous TPMS Architectures and Data-Driven Performance Prediction** 的官方代码库。

LoGIC-Design（Low-dimensional Grid-based Intent-Controlled Design）是一种基于低分辨率网格控制的异质周期性极小曲面（TPMS）生成与性能预测框架。该项目不仅提供了高效的 TPMS 结构生成算法，还包含了一整套自动化的有限元分析（FEA）和计算流体力学（CFD）数据集生成流水线，以及用于预测 TPMS 结构力学和流体力学性能的深度学习模型（CNN 和 GNN）。

## 🌟 主要特性

- **异质 TPMS 生成**: 支持通过低分辨率网格（如 3x3x3）控制多种 TPMS 结构的权重、密度和旋转场，实现复杂异质结构的平滑过渡，并支持同时生成结构域和流体域。
- **自动化数据集生成**:
  - **FEA (结构力学)**: 自动化调用 Abaqus 进行批量建模、网格划分、求解和后处理。
  - **CFD (流体力学)**: 结合 Blender、3-matic 和 COMSOL 进行流体域提取、网格重划分和流体仿真。
- **深度学习性能预测**:
  - **CNN 模型**: 基于三维体素数据的卷积神经网络，用于快速预测结构性能。
  - **GNN 模型**: 基于孔隙网络提取的图神经网络（Pore GNN），结合 CNN 提取的局部特征，实现高精度的性能预测。
- **可视化 UI**: 提供基于 Tkinter 的图形用户界面，方便用户交互式设计和预览 TPMS 结构。

## 📁 目录结构

```text
LoGIC/
├── tpms_hybrid.py              # 核心 TPMS 生成器算法
├── UI/                         # 图形用户界面 (Tkinter)
│   └── tpms_ui.py              # UI 启动脚本
├── dataset_generate_FEA/       # 结构力学 (FEA) 数据集生成流水线
│   ├── 1_generate_dataset.py   # 生成 TPMS 样本
│   ├── 3_batch_run_abaqus.py   # 批量运行 Abaqus 仿真
│   └── ...
├── dataset_generate_CFD/       # 流体力学 (CFD) 数据集生成流水线
│   ├── 1_generate_dataset.py   # 生成流体域样本
│   ├── 2_process_fluid_blender.py # Blender 处理脚本
│   ├── 4_run_comsol_simulation_v2.py # COMSOL 仿真脚本
│   └── ...
├── CNN/                        # 卷积神经网络预测模型
│   ├── configs/                # 模型配置文件
│   ├── models/                 # CNN 网络结构定义
│   └── scripts/                # 训练、预测、评估脚本
└── GNN/                        # 图神经网络预测模型
    ├── configs/                # 模型配置文件
    ├── models/                 # Pore GNN 网络结构定义
    └── scripts/                # 图构建、训练、评估脚本
```

## 数据集 (Datasets)

本项目的数据集托管在 Hugging Face：
🔗 [https://huggingface.co/datasets/pxy1118/LoGIC-Dataset](https://huggingface.co/datasets/pxy1118/LoGIC-Dataset)

请将下载的文件放置在以下相应目录中：
- `CNN/dataset/`
- `GNN/dataset/`
- `dataset_generate_CFD/` (需放置 `fluid.mph`)

## 版本与 DOI

本仓库已准备通过 GitHub Release 和 Zenodo GitHub integration 归档生成 DOI。计划首个正式归档版本为：

- Tag: `v1.0.0`
- Release title: `LoGIC-Design v1.0.0`
- Release description: paper-submission version of the LoGIC-Design codebase

发布 `v1.0.0` release 后，Zenodo 将归档该版本并生成 DOI；获得 DOI 后，可在本节补充 DOI badge 和正式 DOI 链接。

## 环境依赖

### 1. Python 环境
推荐使用 Python 3.11+。可以通过以下命令安装项目所需的所有依赖：

```bash
# 安装所有依赖
pip install -r requirements.txt
```

*注意：如果您需要使用 GPU 加速深度学习模型，请根据您的 CUDA 版本安装对应的 PyTorch。*

### 2. 外部软件依赖 (用于数据集生成)
- **Abaqus**: 用于 FEA 仿真 (`dataset_generate_FEA`)。
- **Blender**: 用于流体域网格初步处理 (`dataset_generate_CFD`)。
- **Materialise 3-matic**: 用于流体域网格重划分和优化 (`dataset_generate_CFD`)。
- **COMSOL Multiphysics**: 用于 CFD 仿真 (`dataset_generate_CFD`)。

## 🚀 使用指南

### 1. 交互式设计 (UI)
运行以下命令启动图形用户界面，进行 TPMS 结构的交互式设计和预览：
```bash
python UI/tpms_ui.py
```

### 2. 数据集生成
进入相应的目录并按脚本编号顺序运行。

**FEA 数据集生成:**
```bash
cd dataset_generate_FEA
python 1_generate_dataset.py
python 2_process_volmesh.py
python 3_batch_run_abaqus.py
python 4_batch_postprocess.py
python 5_aggregate_postprocessed_curves.py
```

**CFD 数据集生成:**
```bash
cd dataset_generate_CFD
python 1_generate_dataset.py
# 后续步骤需要调用 Blender, 3-matic 和 COMSOL，请参考脚本内的具体说明
```

### 3. 深度学习模型训练与评估

**CNN 模型:**
```bash
cd CNN
python scripts/1_train_cnn.py
python scripts/2_predict.py
python scripts/3_evaluate_all.py
python scripts/4_visualize_results.py
```

**GNN 模型:**
```bash
cd GNN
python scripts/1_pretrain_cnn.py
python scripts/2_extract_features.py
python scripts/3_build_graphs.py
python scripts/4_train_gnn.py
python scripts/5_predict.py
python scripts/6_evaluate_all.py
```

## 📝 引用 (Citation)

如果您在研究中使用了本代码库，请引用本仓库的固定发布版本。GitHub 会读取根目录下的 [`CITATION.cff`](CITATION.cff) 并显示引用入口；Zenodo 归档 `v1.0.0` release 后，请优先使用 Zenodo 生成的 DOI。

```bibtex
@software{pan_logic_design_2026,
  author  = {Pan, Xiaoyue},
  title   = {LoGIC-Design: A Low-Dimensional Grid-Controlled Framework for Heterogeneous TPMS Architectures and Data-Driven Performance Prediction},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/pxy1118/LoGIC}
}
```

论文正式发表后，请同时引用论文版本的 DOI 或期刊引用信息。

## 📄 许可证 (License)

[MIT License](LICENSE)
