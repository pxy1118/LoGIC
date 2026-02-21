#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 CAE 训练用的混合权重体数据，仅保存 12x12x12 的权重网格：
- dataset/3D_CAE_Train_Weights_12x12x12xN.npy   (S, 12, 12, 12, N)
- dataset/3D_CAE_Train_Weights_meta.json        生成参数元数据

实现：先随机生成 one-hot 的 (3,3,3,N)，再按轴重复 4 倍扩展为 (12,12,12,N)。
这与参考中的 density12 方法一致（每个低分辨率格点扩成 4x4x4 小块）。
"""
import os
import json
from typing import List

import numpy as np

from tpms_mixup import (
    primitive, gyroid, diamond, neovius, schoen_iwp, lidinoid,
)

# ================== 基本配置 ==================
DATASET_ROOT = 'dataset'

# TPMS 组合（固定 N，便于聚合保存）。建议 N>=2。
TPMS_FUNCS: List = [primitive, gyroid]  # 可改为 [primitive, gyroid, diamond]
N_CHANNELS = len(TPMS_FUNCS)

# 随机与数量
NUM_SAMPLES = 17835
GLOBAL_RANDOM_SEED = 42  # 设为 None 每次不同

# 输出文件名（仅 12x12x12xN）
OUTPUT_FILE_12 = '3D_CAE_Train_Weights.npy'   # (S, 12, 12, 12, N)
META_JSON = '3D_CAE_Train_Weights_meta.json'

# ================== 工具函数 ==================

def _rng():
    return np.random.default_rng(GLOBAL_RANDOM_SEED) if GLOBAL_RANDOM_SEED is not None else np.random.default_rng()


def _random_weight_grid_3x3x3xN(rng: np.random.Generator, n_channels: int) -> np.ndarray:
    """生成 shape=(3,3,3,N) 的 one-hot 权重网格：每格只激活一个通道。"""
    if n_channels <= 0:
        raise ValueError('n_channels 必须 > 0')
    choice_idx = rng.integers(0, n_channels, size=(3, 3, 3))
    eye = np.eye(n_channels, dtype=np.float32)
    grid = eye[choice_idx].astype(np.float32)
    return grid  # (3,3,3,N)


def _expand_3_to_12(arr_3: np.ndarray) -> np.ndarray:
    """将 (S,3,3,3,N) 通过重复扩展为 (S,12,12,12,N)。"""
    assert arr_3.ndim == 5 and arr_3.shape[1:4] == (3, 3, 3)
    arr_12 = np.repeat(arr_3, 4, axis=1)
    arr_12 = np.repeat(arr_12, 4, axis=2)
    arr_12 = np.repeat(arr_12, 4, axis=3)
    return arr_12.astype(np.float32)


# ================== 主流程 ==================

def generate_cae_weight_train():
    os.makedirs(DATASET_ROOT, exist_ok=True)

    rng = _rng()

    # 先生成 one-hot 的 3x3x3xN，再扩展为 12x12x12xN
    lowres_grids: list[np.ndarray] = []  # (3,3,3,N)

    for _ in range(NUM_SAMPLES):
        grid_N = _random_weight_grid_3x3x3xN(rng, N_CHANNELS)
        lowres_grids.append(grid_N)

    arr3 = np.stack(lowres_grids, axis=0).astype(np.float32)  # (S,3,3,3,N)
    arr12 = _expand_3_to_12(arr3)                              # (S,12,12,12,N)

    # 仅保存 12x12x12xN
    out12 = os.path.join(DATASET_ROOT, OUTPUT_FILE_12)
    np.save(out12, arr12)

    # 写入元数据
    meta = {
        'num_samples': NUM_SAMPLES,
        'grid_shape': [12, 12, 12],
        'channels': N_CHANNELS,
        'tpms': [f.__name__ for f in TPMS_FUNCS],
        'seed': GLOBAL_RANDOM_SEED,
        'output': out12.replace('\\', '/'),
        'dtype': 'float32',
        'expanded_from': [3, 3, 3],
        'expand_factor': 4,
        'lowres_one_hot': True,
        'note': '12x 由 3x 逐轴 repeat(4) 得到（最近邻扩展）',
    }
    # with open(os.path.join(DATASET_ROOT, META_JSON), 'w', encoding='utf-8') as f:
    #     json.dump(meta, f, ensure_ascii=False, indent=2)

    print('已生成 12x12x12xN ->', meta['output'], ' shape=', tuple(arr12.shape))


if __name__ == '__main__':
    generate_cae_weight_train()
