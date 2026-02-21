#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展版 TPMS 混合生成器 —— 支持任意尺寸的 N×M×K 权重网格 (例如 9×9×3)
新增：提供手工配置的 5×5×5 权重/密度网格，避免通过 5×5×3 平铺再裁剪。
"""
import numpy as np
import os
from scipy.ndimage import zoom, gaussian_filter
from typing import Tuple, List, Callable, Optional, Sequence
import matplotlib.pyplot as plt

# 从原始模块导入共享功能
from tpms_hybrid import (
    Gyroid, Gyroid_R, FKS, Diamond, Primitive, Neovius, Schoen_IWP, Lidinoid, my_tpms,
    TPMS_COLOR_PAIRS, TPMS_FUNCTIONS,
    expanded_ranges, finalize_mesh,
    create_fluid_domain_from_solid_mask, export_stl,
    _resolve_tpms_name, get_tpms_color_pairs, get_tpms_functions,
    map_density_to_local_porosity
)

# ================== 增强的网格创建功能 ==================
def create_composite_weight_grid(subgrids: List[np.ndarray], layout: Tuple[int, ...] = (3, 3)) -> np.ndarray:
    """
    将多个 3×3×3 子网格组合成更大的网格，支持在 X/Y/Z 三个方向重复。
    参数:
        subgrids: 形状为 (sx, sy, sz, N) 的子网格列表。
        layout: (rows, cols[, layers]) 布局，例如 (3,3,2) 生成 9×9×6 网格。
    返回:
        组合后的大网格，形状为 (rows*sx, cols*sy, layers*sz, N)
    """
    if len(layout) == 2:
        rows, cols = layout
        layers = 1
    elif len(layout) == 3:
        rows, cols, layers = layout
    else:
        raise ValueError("layout 必须是长度为 2 或 3 的元组")

    if rows <= 0 or cols <= 0 or layers <= 0:
        raise ValueError("layout 中的维度必须为正整数")

    num_subgrids = rows * cols * layers
    if len(subgrids) < num_subgrids:
        raise ValueError(f"需要{num_subgrids}个子网格，但只提供了{len(subgrids)}个")

    sx, sy, sz, channel_count = subgrids[0].shape
    composite = np.zeros((rows * sx, cols * sy, layers * sz, channel_count), dtype=subgrids[0].dtype)

    for index in range(num_subgrids):
        i = index // (cols * layers)
        j = (index % (cols * layers)) // layers
        k = index % layers
        composite[
            i * sx:(i + 1) * sx,
            j * sy:(j + 1) * sy,
            k * sz:(k + 1) * sz,
            :
        ] = subgrids[index]

    return composite

def create_composite_density_grid(subgrids: List[np.ndarray], layout: Tuple[int, ...] = (3, 3)) -> np.ndarray:
    """将多个 3×3×3 密度子网格组合成更大的网格，支持 X/Y/Z 三个方向重复。"""
    if len(layout) == 2:
        rows, cols = layout
        layers = 1
    elif len(layout) == 3:
        rows, cols, layers = layout
    else:
        raise ValueError("layout 必须是长度为 2 或 3 的元组")

    if rows <= 0 or cols <= 0 or layers <= 0:
        raise ValueError("layout 中的维度必须为正整数")

    num_subgrids = rows * cols * layers
    if len(subgrids) < num_subgrids:
        raise ValueError(f"需要{num_subgrids}个子网格，但只提供了{len(subgrids)}个")

    sx, sy, sz = subgrids[0].shape
    composite = np.zeros((rows * sx, cols * sy, layers * sz), dtype=subgrids[0].dtype)

    for index in range(num_subgrids):
        i = index // (cols * layers)
        j = (index % (cols * layers)) // layers
        k = index % layers
        composite[
            i * sx:(i + 1) * sx,
            j * sy:(j + 1) * sy,
            k * sz:(k + 1) * sz
        ] = subgrids[index]

    return composite

def generate_random_weight_grid_custom(N: int, grid_size: Tuple[int, int, int] = (3, 3, 3), 
                                     seed=None, smooth_bias=False, boundary_fixed=False, 
                                     normalize_channels: bool = True, uniform_when_single: bool = True):
    """
    生成指定尺寸的随机权重网格
    """
    if seed is not None:
        np.random.seed(seed)
        
    x_size, y_size, z_size = grid_size
    
    if N == 1 and uniform_when_single:
        return np.ones((x_size, y_size, z_size, 1), dtype=np.float32)
    
    grid = np.random.rand(x_size, y_size, z_size, N).astype(np.float32)
    
    if boundary_fixed and N >= 2:
        # 仅当网格足够大时才固定边界
        if x_size >= 3 and y_size >= 3 and z_size >= 3:
            # 固定角落为第一种TPMS
            for i in [0, x_size-1]:
                for j in [0, y_size-1]:
                    for k in [0, z_size-1]:
                        grid[i,j,k,:] = 0.0
                        grid[i,j,k,0] = 1.0
            
            # 中心区域为第二种TPMS
            if x_size > 2 and y_size > 2 and z_size > 2:
                center_i = x_size // 2
                center_j = y_size // 2
                center_k = z_size // 2
                grid[center_i, center_j, center_k, :] = 0.0
                grid[center_i, center_j, center_k, 1] = 1.0
    
    if smooth_bias:
        for n in range(N):
            grid[..., n] = gaussian_filter(grid[..., n], sigma=0.8)
    
    if normalize_channels:
        grid_sum = grid.sum(axis=-1, keepdims=True)
        grid = np.divide(grid, grid_sum, out=np.zeros_like(grid), where=grid_sum != 0)
    
    return np.clip(grid, 0.0, 1.0)

def generate_weight_matrix(resolution: int, x_range, y_range, z_range, grid_N: np.ndarray, 
                          smooth_sigma=None, normalize_channels: bool = True):
    """
    通用权重矩阵生成，支持任意尺寸的输入网格
    """
    if grid_N.ndim != 4:
        raise ValueError("权重网格必须是4D: (X,Y,Z,TPMS_TYPES)")
    
    grid_size = grid_N.shape[:3]
    
    # 动态计算缩放因子
    zoom_factor = (
        resolution / grid_size[0],
        resolution / grid_size[1],
        resolution / grid_size[2],
        1
    )
    
    W = zoom(grid_N.astype(np.float32), zoom_factor, order=1)
    
    if smooth_sigma:
        for i in range(W.shape[-1]):
            W[..., i] = gaussian_filter(W[..., i], sigma=smooth_sigma)
        if normalize_channels and W.shape[-1] > 1:
            W_sum = W.sum(axis=-1, keepdims=True)
            W = np.divide(W, W_sum, out=np.zeros_like(W), where=W_sum != 0)
    
    return np.clip(W, 0.0, 1.0)

def generate_density_field(resolution: int, x_range, y_range, z_range,
                          density_grid: np.ndarray,
                          smooth_sigma: Optional[float] = None,
                          method: str = 'linear') -> np.ndarray:
    """
    通用密度场生成，支持任意尺寸的输入网格
    """
    if density_grid.ndim != 3:
        raise ValueError("密度网格必须是3D: (X,Y,Z)")
    
    grid_size = density_grid.shape
    
    # 动态计算缩放因子
    zoom_factor = (
        resolution / grid_size[0],
        resolution / grid_size[1],
        resolution / grid_size[2]
    )
    
    order = 1 if method == 'linear' else 0
    D = zoom(density_grid.astype(np.float32), zoom_factor, order=order)
    
    if smooth_sigma:
        D = gaussian_filter(D, sigma=smooth_sigma)
    
    return np.clip(D, 0.0, 1.0)

def tile_grid_to_target(grid: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """沿空间轴重复 grid 至 target_shape，必要时裁剪。"""
    spatial_dims = len(target_shape)
    if grid.ndim < spatial_dims:
        raise ValueError("grid 维度不足以扩展到目标尺寸")

    base_shape = grid.shape[:spatial_dims]
    repeats: List[int] = []
    for axis, target in enumerate(target_shape):
        size = base_shape[axis]
        if size <= 0:
            raise ValueError("grid 的尺寸必须为正整数")
        if target % size != 0:
            raise ValueError("目标尺寸必须是基础尺寸的整数倍")
        repeats.append(target // size)

    tile_factors = repeats + [1] * (grid.ndim - spatial_dims)
    tiled = np.tile(grid, tile_factors)

    slices = tuple(slice(0, target_shape[i]) for i in range(spatial_dims))
    if grid.ndim > spatial_dims:
        slices += tuple(slice(None) for _ in range(grid.ndim - spatial_dims))

    return tiled[slices]

def drop_layer_z(arr: np.ndarray, drop_index: int = 2) -> np.ndarray:
    """
    从 Z 轴删除指定层（默认删除第 3 层，0-based 索引为 2）。
    支持 3D (X,Y,Z) 或 4D (X,Y,Z,C) 数组。
    """
    if arr.ndim not in (3, 4):
        raise ValueError("只能对 3D 或 4D 数组删除层")
    if arr.shape[2] <= drop_index:
        raise ValueError(f"Z 轴长度为 {arr.shape[2]}，无法删除索引 {drop_index} 的层")
    return np.delete(arr, drop_index, axis=2)


# MANUAL_WEIGHT_GRID_5x5x6 = np.array([
#     [ 
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
#     [  # 深度1
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#     ],
#     [  # 深度2
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
#     [  # 深度3
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#     ],
#     [  # 深度4
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
# ], dtype=np.float32)
# # 5×5×6密度网格（可直接调整各层密度分布）
# MANUAL_DENSITY_GRID_5x5x6 = np.array([
#     [  # 深度0
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#     ],
#     [  # 深度1
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#     ],
#     [  # 深度2
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#     ],
#     [  # 深度3
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#     ],
#     [  # 深度4
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#         [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
#         [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
#     ],
# ], dtype=np.float32)
MANUAL_WEIGHT_GRID_5x5x6 = np.array([
    [ 
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ],
    [  # 深度1
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ],
    [  # 深度2
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ],
    [  # 深度3
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ],
    [  # 深度4
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ],
], dtype=np.float32)
# 5×5×6密度网格（可直接调整各层密度分布）
MANUAL_DENSITY_GRID_5x5x6 = np.array([
    [  # 深度0
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
    ],
    [  # 深度1
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
    ],
    [  # 深度2
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
    ],
    [  # 深度3
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.4, 0.4, 0.2, 0.2],
    ],
    [  # 深度4
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
        [0.2, 0.2, 0.3, 0.3, 0.2, 0.2],
        [0.4, 0.4, 0.3, 0.3, 0.4, 0.4],
    ],
], dtype=np.float32)

def get_manual_weight_grid_5x5x6(tpms_list: Sequence[Callable]) -> np.ndarray:
    """准备 5×5×6 的手工权重网格；如需自定义请修改 MANUAL_WEIGHT_GRID_5x5x6。"""
    channel_count = len(tpms_list)
    if channel_count == 0:
        raise ValueError("tpms_list 不能为空")
    manual_values = MANUAL_WEIGHT_GRID_5x5x6.copy()
    if manual_values.shape[-1] != channel_count:
        raise ValueError("MANUAL_WEIGHT_GRID_5x5x6 通道数与当前 tpms_list 不一致，请手动调整")
    return manual_values

def get_manual_density_grid_5x5x6() -> np.ndarray:
    """准备 5×5×6 的手工密度网格；如需自定义请修改 MANUAL_DENSITY_GRID_5x5x6。"""
    return MANUAL_DENSITY_GRID_5x5x6.copy()

# ================== 通用可视化函数 ==================
def visualize_weight_grid(weight_grid: np.ndarray, 
                         tpms_names: List[str], 
                         density_grid: Optional[np.ndarray] = None,
                         title="TPMS Weight & Density Distribution",
                         show_values=True,
                         base_colors=None,
                         base_colors_light=None,
                         base_colors_dark=None,
                         figsize=10,
                         alpha_power=1.0,
                         edgecolor="#222",
                         linewidth=0.6,
                         pane_color="#f7f7f7",
                         bg_color="#ffffff",
                         elev=22,
                         azim=38,
                         density_darken=True,
                         density_effect_strength=1.0,
                         density_auto_scale=True,
                         density_scale_minmax=None):
    """
    可视化任意尺寸的权重网格
    """
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import colorsys
    
    if weight_grid.ndim != 4:
        raise ValueError("权重网格必须是4D: (X,Y,Z,TPMS_TYPES)")
    
    grid_size = weight_grid.shape[:3]
    x_size, y_size, z_size = grid_size
    N = weight_grid.shape[3]
    
    def to_rgb_array(colors, n_expected):
        if colors is None:
            return None
        rgb_list = []
        for c in colors:
            if isinstance(c, str):
                rgb_list.append(to_rgb(c))
            else:
                rgb_list.append(tuple(c[:3]))
        arr = np.array(rgb_list, dtype=np.float32)
        if arr.shape[0] != n_expected:
            raise ValueError("颜色列表长度需与 TPMS 数量一致")
        return arr
    
    base_rgb_light = to_rgb_array(base_colors_light, N)
    base_rgb_dark  = to_rgb_array(base_colors_dark, N)
    
    if base_rgb_light is None or base_rgb_dark is None:
        if base_colors is None:
            import matplotlib.cm as cm
            base_colors = [cm.tab10(i % 10) for i in range(N)]
        base_rgb = to_rgb_array(base_colors, N)
        light_list, dark_list = [], []
        for rgb in base_rgb:
            h, s, v = colorsys.rgb_to_hsv(*rgb)
            light = colorsys.hsv_to_rgb(h, max(0.0, s*0.6), min(1.0, v*1.2))
            dark  = colorsys.hsv_to_rgb(h, min(1.0, s*1.1), max(0.0, v*0.65))
            light_list.append(light)
            dark_list.append(dark)
        base_rgb_light = np.array(light_list, dtype=np.float32)
        base_rgb_dark  = np.array(dark_list,  dtype=np.float32)
    
    fig = plt.figure(figsize=(figsize, figsize * 0.8), constrained_layout=True)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)
    
    if hasattr(ax, 'set_box_aspect'):
        max_dim = max(x_size, y_size, z_size)
        ax.set_box_aspect((x_size/max_dim, y_size/max_dim, z_size/max_dim))
    
    try:
        ax.xaxis.pane.set_facecolor(pane_color)
        ax.yaxis.pane.set_facecolor(pane_color)
        ax.zaxis.pane.set_facecolor(pane_color)
        ax.xaxis.pane.set_edgecolor('#dddddd')
        ax.yaxis.pane.set_edgecolor('#dddddd')
        ax.zaxis.pane.set_edgecolor('#dddddd')
    except Exception:
        pass
    
    def cube_vertices(x, y, z, scale=0.9):
        dx, dy, dz = scale/2, scale/2, scale/2
        vertices = np.array([
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz],
            [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz],
            [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
        ])
        faces = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5],
            [2,3,7,6], [0,3,7,4]
        ]
        return vertices, faces
    
    def luminance(rgb):
        r, g, b = rgb
        return 0.2126*r + 0.7152*g + 0.0722*b
    
    if density_grid is not None:
        d_norm = np.clip(density_grid, 0.0, 1.0)
    else:
        d_norm = np.ones((x_size, y_size, z_size))
    
    if density_auto_scale:
        if density_scale_minmax is not None:
            d_min, d_max = float(density_scale_minmax[0]), float(density_scale_minmax[1])
        else:
            d_min, d_max = float(np.min(d_norm)), float(np.max(d_norm))
        if (d_max - d_min) < 1e-8:
            d_min, d_max = 0.0, 1.0
        def _scale_density(v: float) -> float:
            return float(np.clip((v - d_min) / (d_max - d_min), 0.0, 1.0))
    else:
        def _scale_density(v: float) -> float:
            return float(np.clip(v, 0.0, 1.0))
    
    # 绘制网格单元
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                w = weight_grid[x, y, z]  # shape (N,)
                total_weight = float(np.sum(w))
                if total_weight < 1e-6:
                    continue
                
                # 按权重混合颜色
                w_norm = w / max(total_weight, 1e-8)  # 归一化权重
                mixed_light = np.sum(w_norm[:, None] * base_rgb_light, axis=0)
                mixed_dark  = np.sum(w_norm[:, None] * base_rgb_dark,  axis=0)
                
                d_raw = float(d_norm[x, y, z])
                t = float(np.clip((d_raw - np.min(d_norm)) / max(np.ptp(d_norm), 1e-8), 0.0, 1.0)) if density_auto_scale else float(np.clip(d_raw, 0.0, 1.0))
                t = t if density_darken else (1.0 - t)
                t = float(np.clip(t, 0.0, 1.0))
                shaded_color = (1.0 - t) * mixed_light + t * mixed_dark
                
                alpha = float(np.clip(total_weight ** alpha_power, 0.15, 1.0))
                
                vertices, faces = cube_vertices(x, y, z, scale=0.9)
                poly_faces = [vertices[f] for f in faces]
                poly = Poly3DCollection(
                    poly_faces,
                    facecolors=np.append(shaded_color, alpha),
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    antialiased=True,
                    alpha=alpha,
                    clip_on=False
                )
                ax.add_collection3d(poly)
                
                # 显示前两个主导 TPMS
                if show_values:
                    # 简化显示
                    sorted_indices = np.argsort(w)[::-1]
                    top1_idx = sorted_indices[0]
                    top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx
                    label_text = f"{tpms_names[top1_idx][:3]}/{tpms_names[top2_idx][:3]}"
                    ax.text(x + 0.05, y + 0.05, z + 0.05,
                            label_text,
                            fontsize=6, color='black', ha='center', va='center')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.6, x_size-0.4)
    ax.set_ylim(-0.6, y_size-0.4)
    ax.set_zlim(-0.6, z_size-0.4)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xticks(list(range(x_size)))
    ax.set_yticks(list(range(y_size)))
    ax.set_zticks(list(range(z_size)))
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 视角与标题
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=14, pad=12, fontweight='bold')
    
    # 图例
    legend_colors = (base_rgb_light + base_rgb_dark) / 2.0
    handles = [plt.Rectangle((0,0), 1, 1, facecolor=legend_colors[i], label=name)
               for i, name in enumerate(tpms_names)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)
    
    return fig, ax

# ================== 重用原始模块中的核心功能 ==================
def create_hybrid_tpms_solid(tpms_funcs: List[Callable], x_range, y_range, z_range,
                              weight_volume: np.ndarray,
                              density_field: Optional[np.ndarray] = None,
                              global_target_porosity: Optional[float] = None,
                              resolution=90,
                              solid_threshold=0.3,
                              min_threshold=None,
                              smooth=True, smooth_iter=10,
                              morph_close=True, close_iter=1,
                              remove_small=True, min_voxels=50,
                              verbose=True,
                              refine_threshold=True, refine_iters=10,
                              porosity_tol=0.005, refine_ignore_morph=True,
                              return_mask=False):
    """
    从原始模块导入此函数，因为它已经支持所需的通用功能
    """
    from tpms_hybrid import create_hybrid_tpms_solid as original_func
    return original_func(
        tpms_funcs, x_range, y_range, z_range,
        weight_volume=weight_volume,
        density_field=density_field,
        global_target_porosity=global_target_porosity,
        resolution=resolution,
        solid_threshold=solid_threshold,
        min_threshold=min_threshold,
        smooth=smooth, smooth_iter=smooth_iter,
        morph_close=morph_close, close_iter=close_iter,
        remove_small=remove_small, min_voxels=min_voxels,
        verbose=verbose,
        refine_threshold=refine_threshold, refine_iters=refine_iters,
        porosity_tol=porosity_tol, refine_ignore_morph=refine_ignore_morph,
        return_mask=return_mask
    )

# ================== 主流程 ==================
def main():
    # === 配置区 ===
    base_x = base_y = base_z = (-1.5, 1.5)
    resolution = 120
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    verbose = True

    # 复制/扩域系数（与权重/密度网格尺寸无强绑定，可按需调整）
    replicate = (5, 5, 6)

    # 需求：直接使用手工配置的 5×5×6 权重/密度网格
    weight_grid_size = (5, 5, 6)

    # 动态调整用于体素化的分辨率（按 replicate 比例缩放，维持大致体素密度）
    total_resolution_x = resolution * replicate[0] // 3 
    total_resolution_y = resolution * replicate[1] // 3
    total_resolution_z = resolution * replicate[2] // 3
    resolution = max(total_resolution_x, total_resolution_y, total_resolution_z)

    # 选择 TPMS
    tpms_list = [Schoen_IWP, Gyroid, Schoen_IWP]
    N = len(tpms_list)
    
    # ========= 权重网格：直接取得 5×5×6 =========
    manual_weight_grid = get_manual_weight_grid_5x5x6(tpms_list)  # (5,5,6,N)

    if manual_weight_grid.shape[:3] != weight_grid_size:
        raise ValueError(f"权重网格尺寸应为{weight_grid_size}，但得到{manual_weight_grid.shape[:3]}")
    
    # ========= 密度网格：直接取得 5×5×6 =========
    density_grid = get_manual_density_grid_5x5x6()  # (5,5,6)

    if density_grid.shape != weight_grid_size:
        raise ValueError(f"密度网格尺寸应为{weight_grid_size}，但得到{density_grid.shape}")
    
    # 可视化
    print(f"生成 {weight_grid_size[0]}x{weight_grid_size[1]}x{weight_grid_size[2]} 权重+密度网格可视化...")
    tpms_names = [f.__name__ for f in tpms_list]
    tpms_colors_light, tpms_colors_dark = get_tpms_color_pairs(tpms_names)
    
    fig, ax = visualize_weight_grid(
        manual_weight_grid, 
        tpms_names,
        density_grid=density_grid,
        title=f"TPMS Type & Density ({weight_grid_size[0]}×{weight_grid_size[1]}×{weight_grid_size[2]})",
        alpha_power=1.2,
        base_colors_light=tpms_colors_light,
        base_colors_dark=tpms_colors_dark,
        density_darken=True,
        density_effect_strength=2.0,
        show_values=True,
        figsize=12
    )
    
    cube_img = os.path.join(output_dir, f"weight_density_grid_{weight_grid_size[0]}x{weight_grid_size[1]}x{weight_grid_size[2]}.png")
    fig.savefig(cube_img, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"网格可视化已保存: {cube_img}")
    
    # 域扩展
    if replicate != (1,1,1):
        x_range, y_range, z_range = expanded_ranges(base_x, base_y, base_z, replicate)
    else:
        x_range, y_range, z_range = base_x, base_y, base_z
    
    # 生成密度场
    density_field = generate_density_field(
        resolution, x_range, y_range, z_range,
        density_grid,
        smooth_sigma=None, 
        method='linear'
    )
    
    # 权重平滑
    smooth_sigma = 0.5
    weight_volume = generate_weight_matrix(
        resolution, x_range, y_range, z_range, 
        manual_weight_grid, 
        smooth_sigma=smooth_sigma, 
        normalize_channels=(N>1)
    )
    
    # 保存字段数据
    combined_fields = np.concatenate([
        density_field[..., None].astype(np.float32),
        weight_volume.astype(np.float32)
    ], axis=-1).astype(np.float32)
    
    weight_path = os.path.join(output_dir, "fields.npy")
    np.save(weight_path, combined_fields)
    if verbose:
        print("密度/权重体已保存: fields.npy (通道0=密度, 其余=权重)")
    
    # 实体参数
    global_target_porosity = None
    solid_threshold = 0.3
    min_threshold = None
    smooth = True; smooth_iter = 10
    final_taubin_iter = 10
    morph_close = True; close_iter = 1
    remove_small = True; min_voxels = 80
    desired_size = 17.0
    align_origin = True
    
    # 生成结构域（实体）并返回掩码
    print(f"生成结构域: {[f.__name__ for f in tpms_list]}...")
    solid_mesh, actual_porosity, used_threshold, solid_mask = create_hybrid_tpms_solid(
        tpms_list, x_range, y_range, z_range,
        weight_volume=weight_volume,
        density_field=density_field,
        global_target_porosity=global_target_porosity,
        resolution=resolution,
        solid_threshold=solid_threshold,
        min_threshold=min_threshold,
        smooth=smooth, smooth_iter=smooth_iter,
        morph_close=morph_close, close_iter=close_iter,
        remove_small=remove_small, min_voxels=min_voxels,
        verbose=verbose,
        return_mask=True
    )
    
    voxel_path = os.path.join(output_dir, "voxel.npy")
    np.save(voxel_path, solid_mask.astype(np.uint8))
    if verbose:
        print(f"实体体素掩码已保存: {voxel_path} (值域: 0=流体, 1=实体)")
    
    # 最终清理
    if verbose: print("结构域最终清理/平滑...")
    solid_mesh = finalize_mesh(solid_mesh, smooth_taubin_iter=final_taubin_iter, do_clean=True, verbose=verbose)
    if verbose: print(f"结构域最终网格: 点 {solid_mesh.n_points}, 单元 {solid_mesh.n_cells}")
    
    # 生成流体域
    print("生成流体域...")
    fluid_mesh, fluid_porosity = create_fluid_domain_from_solid_mask(
        solid_mask,
        x_range, y_range, z_range,
        resolution=resolution,
        add_boundary_box=False,
        boundary_thickness=0,
        z_extension=0.1,
        smooth=True,
        smooth_iter=smooth_iter,
        remove_small=remove_small,
        min_voxels=min_voxels,
        verbose=verbose
    )
    
    # 最终清理流体域
    if verbose: print("流体域最终清理/平滑...")
    fluid_mesh = finalize_mesh(fluid_mesh, smooth_taubin_iter=final_taubin_iter, do_clean=True, verbose=verbose)
    if verbose: print(f"流体域最终网格: 点 {fluid_mesh.n_points}, 单元 {fluid_mesh.n_cells}")
    
    # 缩放与对齐原点（两个网格同步）
    if desired_size is not None and solid_mesh is not None and solid_mesh.n_points > 0:
        b = solid_mesh.bounds
        lx, ly, lz = b[1]-b[0], b[3]-b[2], b[5]-b[4]
        if lx > 0 and ly > 0 and lz > 0:
            scale_factor = desired_size / lz
            solid_mesh.scale([scale_factor]*3, inplace=True)
            fluid_mesh.scale([scale_factor]*3, inplace=True)
            nb = solid_mesh.bounds
            if verbose:
                print(f"已缩放: 原尺寸=({lx:.3f},{ly:.3f},{lz:.3f}) -> 新尺寸≈{desired_size}")
            if align_origin:
                offset = [-nb[0], -nb[2], -nb[4]]
                solid_mesh.translate(offset, inplace=True)
                fluid_mesh.translate(offset, inplace=True)
    
    # 导出 STL
    suffix = f"_{weight_grid_size[0]}x{weight_grid_size[1]}x{weight_grid_size[2]}"
    if replicate != (1,1,1):
        suffix += f"_rep{replicate[0]}x{replicate[1]}x{replicate[2]}"
    
    solid_filename = os.path.join(output_dir, f"model_solid{suffix}.stl")
    export_stl(solid_mesh, solid_filename)
    
    fluid_filename = os.path.join(output_dir, f"model_fluid{suffix}.stl")
    export_stl(fluid_mesh, fluid_filename)
    
    print("\n" + "="*60)
    print(f"✅ 生成完成!")
    print(f"   结构域: {solid_filename}")
    print(f"   流体域: {fluid_filename}")
    print(f"   实际孔隙率: {actual_porosity*100:.2f}%")
    print(f"   流体体积分数: {fluid_porosity*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()