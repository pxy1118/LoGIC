#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os
import time
from scipy.ndimage import gaussian_filter, zoom, binary_closing, binary_dilation, generate_binary_structure, label
from typing import Tuple, List, Callable, Optional, TYPE_CHECKING, Any, Sequence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import colorsys

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    pv = None
    _HAS_PYVISTA = False

if TYPE_CHECKING:
    import pyvista as _pv_typing

# ================== 基本 TPMS 隐式函数 ==================

def Gyroid(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Gyroid (使用旋转后的坐标)
    return (np.sin(scale*x)*np.cos(scale*y) +
            np.sin(scale*y)*np.cos(scale*z) +
            np.sin(scale*z)*np.cos(scale*x))



def FKS(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 FKS (使用旋转后的坐标)
    return (
        np.cos(2*scale*x) * np.sin(scale*y) * np.cos(scale*z)
        + np.cos(scale*x) * np.cos(2*scale*y) * np.sin(scale*z)
        + np.sin(scale*x) * np.cos(scale*y) * np.cos(2*scale*z)
    )
    
def Diamond(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Diamond (使用旋转后的坐标)
    return (np.sin(scale*y)*np.sin(scale*z) +
            np.sin(scale*x)*np.cos(scale*y)*np.cos(scale*z) +
            np.cos(scale*x)*np.sin(scale*y)*np.cos(scale*z) +
            np.cos(scale*x)*np.cos(scale*y)*np.sin(scale*z))

def Primitive(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    """Schwarz Primitive (P-type)"""
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Primitive (使用旋转后的坐标)
    return np.cos(scale*x) + np.cos(scale*y) + np.cos(scale*z)

def Neovius(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    """Neovius surface"""
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Neovius (使用旋转后的坐标)
    return (3 * (np.cos(scale*x) + np.cos(scale*y) + np.cos(scale*z)) +
            4 * np.cos(scale*x) * np.cos(scale*y) * np.cos(scale*z))

def Schoen_IWP(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    """Schoen I-WP"""
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Schoen_IWP (使用旋转后的坐标)
    return (2 * (np.cos(scale*x)*np.cos(scale*y) +
                 np.cos(scale*y)*np.cos(scale*z) +
                 np.cos(scale*z)*np.cos(scale*x)) -
            (np.cos(2*scale*x) + np.cos(2*scale*y) + np.cos(2*scale*z)))

def Lidinoid(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    """Lidinoid surface"""
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 Lidinoid (使用旋转后的坐标)
    sx, cx = np.sin(scale*x), np.cos(scale*x)
    sy, cy = np.sin(scale*y), np.cos(scale*y)
    sz, cz = np.sin(scale*z), np.cos(scale*z)
    return (0.5 * (sx*cz + sy*cx + sz*cy) -
            0.5 * (cx*cz + cy*cx + cz*cy) +
            0.15 * (cx + cy + cz))

def my_tpms(x, y, z, scale=2*np.pi/3, rot_x=0, rot_y=0, rot_z=0):
    # 1. 绕 X 轴旋转
    cos_rx, sin_rx = np.cos(rot_x), np.sin(rot_x)
    y_new = y * cos_rx - z * sin_rx
    z_new = y * sin_rx + z * cos_rx
    y, z = y_new, z_new
    
    # 2. 绕 Y 轴旋转
    cos_ry, sin_ry = np.cos(rot_y), np.sin(rot_y)
    x_new = x * cos_ry + z * sin_ry
    z_new = -x * sin_ry + z * cos_ry
    x, z = x_new, z_new
    
    # 3. 绕 Z 轴旋转
    cos_rz, sin_rz = np.cos(rot_z), np.sin(rot_z)
    x_new = x * cos_rz - y * sin_rz
    y_new = x * sin_rz + y * cos_rz
    x, y = x_new, y_new

    # 4. 计算 my_tpms (使用旋转后的坐标)
    return (np.sin(scale*x)*np.cos(scale*y) +
            np.sin(scale*y)*np.cos(scale*z) +
            np.sin(scale*z)*np.cos(scale*x))
# 全局配色方案
TPMS_COLOR_PAIRS = {
    'Gyroid':      ("#bde0fe", "#8ac6fd"),
    'Primitive':   ("#ffc8dd", "#ff75a7"),
    'FKS':         ("#cdb4db", "#c299da"),
    'Diamond':     ("#ffdfc4", "#ffb77a"),
    'Neovius':     ("#bfcd90", "#bcd276"),
    'Schoen_IWP':  ("#c2e9e6", "#7accc8"),
    'Lidinoid':    ("#d4edda", "#8fd9a8"),
    'my_tpms':     ("#f0efeb", "#c1c0b9"),
}

TPMS_FUNCTIONS = {
    'Gyroid': Gyroid,
    'Primitive': Primitive,
    'FKS': FKS,
    'Diamond': Diamond,
    'Neovius': Neovius,
    'Schoen_IWP': Schoen_IWP,
    'Lidinoid': Lidinoid,
    'my_tpms': my_tpms,
}


# MANUAL_WEIGHT_GRID = np.array([
#     [
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
#     [
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
#     [
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#         [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     ],
# ], dtype=np.float32)
MANUAL_WEIGHT_GRID = np.array([
    [
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ],
    [
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    ],
    [
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ],
], dtype=np.float32)
# MANUAL_WEIGHT_GRID = np.array([
#     [
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#     ],
#     [
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#     ],
#     [
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#         [[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]],
#     ],
# ], dtype=np.float32)

_MANUAL_WEIGHT_GRID_MAP = {
    ('Gyroid','Schoen_IWP'): MANUAL_WEIGHT_GRID,
}
tpms_list = [
    Gyroid, Schoen_IWP
]

# MANUAL_DENSITY_GRID = np.array([
#     [
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#     ],
#     [
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#     ],
#     [
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#         [0.3, 0.3, 0.3],
#     ],
# ], dtype=np.float32)

MANUAL_DENSITY_GRID = np.array([
    [
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
    ],
    [
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
    ],
    [
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
        [0.4, 0.3, 0.4],
    ],
], dtype=np.float32)

# MANUAL_DENSITY_GRID = np.array([
#     [
#         [0.2, 0.4, 0.2],
#         [0.4, 0.3, 0.4],
#         [0.2, 0.4, 0.2],
#     ],
#     [
#         [0.4, 0.3, 0.4],
#         [0.2, 0.3, 0.2],
#         [0.4, 0.3, 0.4],
#     ],
#     [
#         [0.2, 0.4, 0.2],
#         [0.4, 0.3, 0.4],
#         [0.2, 0.4, 0.2],
#     ],
# ], dtype=np.float32)

# MANUAL_DENSITY_GRID = np.array([
#     [
#         [0.19, 0.39, 0.19],
#         [0.39, 0.24, 0.39],
#         [0.19, 0.39, 0.19],
#     ],
#     [
#         [0.39, 0.24, 0.39],
#         [0.29, 0.39, 0.29],
#         [0.39, 0.24, 0.39],
#     ],
#     [
#         [0.19, 0.39, 0.19],
#         [0.39, 0.24, 0.39],
#         [0.19, 0.39, 0.19],
#     ],
# ], dtype=np.float32)

# ================== 旋转控制网格 ==================
# 形状: (3, 3, 3, 3)，最后一维为 [rot_x, rot_y, rot_z]（角度）
MANUAL_ROTATION_GRID = np.array([
    [
        [[45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [45.0, 0.0, 0.0]],
        [[0.0, 45.0, 0.0], [0.0, 45.0, 0.0], [0.0, 45.0, 0.0]],
        [[0.0, 0.0, 45.0], [0.0, 0.0, 45.0], [0.0, 0.0, 45.0]],
    ],
    [
        [[45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [45.0, 0.0, 0.0]],
        [[0.0, 45.0, 0.0], [0.0, 45.0, 0.0], [0.0, 45.0, 0.0]],
        [[0.0, 0.0, 45.0], [0.0, 0.0, 45.0], [0.0, 0.0, 45.0]],
    ],
    [
        [[45.0, 0.0, 0.0], [45.0, 0.0, 0.0], [45.0, 0.0, 0.0]],
        [[0.0, 45.0, 0.0], [0.0, 45.0, 0.0], [0.0, 45.0, 0.0]],
        [[0.0, 0.0, 45.0], [0.0, 0.0, 45.0], [0.0, 0.0, 45.0]],
    ],
], dtype=np.float32)

def _resolve_tpms_name(tpms_item: Any, fallback: str) -> str:
    if isinstance(tpms_item, str):
        return tpms_item
    name = getattr(tpms_item, "__name__", None)
    if name is None:
        return fallback
    return name

def get_manual_weight_grid(tpms_sequence: Sequence[Any]) -> Optional[np.ndarray]:
    if not tpms_sequence:
        return None
    names = tuple(
        _resolve_tpms_name(tpms_item, f"TPMS_{idx}")
        for idx, tpms_item in enumerate(tpms_sequence)
    )
    if names in _MANUAL_WEIGHT_GRID_MAP:
        return _MANUAL_WEIGHT_GRID_MAP[names].copy()
    name_set = set(names)
    for base_order, grid in _MANUAL_WEIGHT_GRID_MAP.items():
        if len(base_order) == len(names) and set(base_order) == name_set:
            perm = [base_order.index(name) for name in names]
            return grid[..., perm].copy()
    return None

def get_manual_density_grid(tpms_sequence: Optional[Sequence[Any]] = None) -> np.ndarray:
    _ = tpms_sequence
    return MANUAL_DENSITY_GRID.copy()

def get_manual_rotation_grid(tpms_sequence: Optional[Sequence[Any]] = None) -> np.ndarray:
    """获取手动旋转网格 (3x3x3x3)，最后一维为 [rot_x, rot_y, rot_z]（角度）"""
    _ = tpms_sequence
    return MANUAL_ROTATION_GRID.copy()

def get_tpms_color_pairs(tpms_names: List[str],
                         default_light: str = "#eeeeee",
                         default_dark: str = "#777777") -> Tuple[List[str], List[str]]:
    lights, darks = [], []
    for name in tpms_names:
        light, dark = TPMS_COLOR_PAIRS.get(name, (default_light, default_dark))
        lights.append(light)
        darks.append(dark)
    return lights, darks

def get_tpms_functions(tpms_names: Sequence[str]) -> List[Callable]:
    funcs: List[Callable] = []
    for name in tpms_names:
        func = TPMS_FUNCTIONS.get(name)
        if func is None:
            raise KeyError(f"未知的 TPMS 函数名称: {name}")
        funcs.append(func)
    return funcs

# ================== 权重矩阵生成 ==================

def generate_weight_matrix(resolution: int, x_range, y_range, z_range, grid_N: np.ndarray, 
                          smooth_sigma=None, normalize_channels: bool = True):
    if grid_N.ndim != 4 or grid_N.shape[:3] != (3,3,3):
        raise ValueError("需要提供 shape=(3,3,3,N) 的手动权重网格 grid_N")

    zoom_factor = (resolution/3, resolution/3, resolution/3, 1)
    W = zoom(grid_N.astype(np.float32), zoom_factor, order=1)

    if smooth_sigma:
        for i in range(W.shape[-1]):
            W[..., i] = gaussian_filter(W[..., i], sigma=smooth_sigma)
        if normalize_channels and W.shape[-1] > 1:
            W_sum = W.sum(axis=-1, keepdims=True)
            W = np.divide(W, W_sum, out=np.zeros_like(W), where=W_sum != 0)

    return np.clip(W, 0.0, 1.0)

def generate_random_weight_grid(N: int, seed=None, smooth_bias=False, boundary_fixed=False, 
                               normalize_channels: bool = True, uniform_when_single: bool = True):
    if seed is not None:
        np.random.seed(seed)

    if N == 1 and uniform_when_single:
        return np.ones((3, 3, 3, 1), dtype=np.float32)

    grid = np.random.rand(3, 3, 3, N).astype(np.float32)

    if boundary_fixed and N >= 2:
        for i in [0,2]:
            for j in [0,2]:
                for k in [0,2]:
                    grid[i,j,k,:] = 0.0
                    grid[i,j,k,0] = 1.0
        grid[1,1,1,:] = 0.0
        grid[1,1,1,1] = 1.0

    if smooth_bias:
        for n in range(N):
            grid[..., n] = gaussian_filter(grid[..., n], sigma=0.8)

    if normalize_channels:
        grid_sum = grid.sum(axis=-1, keepdims=True)
        grid = np.divide(grid, grid_sum, out=np.zeros_like(grid), where=grid_sum != 0)

    return np.clip(grid, 0.0, 1.0)

# ================== 密度场生成 ==================

def generate_density_field_from_3x3x3(resolution: int, x_range, y_range, z_range,
                                      density_grid_3x3x3: np.ndarray,
                                      smooth_sigma: Optional[float] = None,
                                      method: str = 'linear') -> np.ndarray:
    if density_grid_3x3x3.shape != (3, 3, 3):
        raise ValueError("density_grid_3x3x3 必须是 (3,3,3)")

    zoom_factor = (resolution / 3.0, resolution / 3.0, resolution / 3.0)
    order = 1 if method == 'linear' else 0
    D = zoom(density_grid_3x3x3.astype(np.float32), zoom_factor, order=order)

    if smooth_sigma:
        D = gaussian_filter(D, sigma=smooth_sigma)

    return np.clip(D, 0.0, 1.0)

def generate_rotation_field_from_3x3x3(resolution: int, x_range, y_range, z_range,
                                       rotation_grid_3x3x3: np.ndarray,
                                       smooth_sigma: Optional[float] = None,
                                       method: str = 'linear') -> np.ndarray:
    """
    从 3x3x3x3 旋转网格生成高分辨率旋转场
    
    参数:
        resolution: 目标分辨率
        rotation_grid_3x3x3: 形状 (3,3,3,3)，最后一维为 [rot_x, rot_y, rot_z]（角度）
        smooth_sigma: 可选高斯平滑
        method: 'linear' 或 'nearest'
    
    返回:
        形状 (resolution, resolution, resolution, 3) 的旋转场（角度）
    """
    if rotation_grid_3x3x3.shape != (3, 3, 3, 3):
        raise ValueError("rotation_grid_3x3x3 必须是 (3,3,3,3)")

    zoom_factor = (resolution / 3.0, resolution / 3.0, resolution / 3.0, 1)
    order = 1 if method == 'linear' else 0
    R = zoom(rotation_grid_3x3x3.astype(np.float32), zoom_factor, order=order)

    if smooth_sigma:
        for i in range(3):
            R[..., i] = gaussian_filter(R[..., i], sigma=smooth_sigma)

    return R

def map_density_to_local_porosity(density_field: np.ndarray,
                                  min_porosity: float = 0.15,
                                  max_porosity: float = 0.95) -> np.ndarray:
    return max_porosity - density_field * (max_porosity - min_porosity)

# ================== 域扩展 ==================

def expanded_ranges(x_range, y_range, z_range, replicate: Tuple[int,int,int]):
    nx, ny, nz = replicate
    Lx = x_range[1]-x_range[0]; Ly = y_range[1]-y_range[0]; Lz = z_range[1]-z_range[0]
    return (x_range[0], x_range[0] + nx*Lx), (y_range[0], y_range[0] + ny*Ly), (z_range[0], z_range[0] + nz*Lz)

# ================== 最终清理/平滑 ==================

def finalize_mesh(mesh: Any, smooth_taubin_iter=10, do_clean=True, verbose=True) -> Any:
    if mesh is None or mesh.n_points == 0:
        return mesh
    if smooth_taubin_iter>0 and hasattr(mesh, 'smooth_taubin'):
        try:
            mesh = mesh.smooth_taubin(n_iter=smooth_taubin_iter)
        except Exception as e:
            if verbose: print(f"Taubin 平滑失败: {e}")
    if do_clean:
        try:
            mesh = mesh.clean(tolerance=1e-6)
        except Exception as e:
            if verbose: print(f"clean 失败: {e}")
    return mesh

# ================== 孔隙率阈值自动计算 ==================

def _auto_threshold(absF, target_porosity, verbose=False):
    solid_fraction = 1 - target_porosity
    flat = absF.ravel()
    k = int(solid_fraction * flat.size)
    if k <= 0:
        th = flat.min()
    elif k >= flat.size:
        th = flat.max()
    else:
        th = np.partition(flat, k)[k]
    if verbose:
        print(f"自动阈值 th={th:.5f} (目标孔隙率 {target_porosity*100:.2f}%)")
    return th

# ================== 掩码后处理 ==================

def _postprocess_mask(mask, morph_close=True, close_iter=1, remove_small=True, min_voxels=50, verbose=False):
    if morph_close:
        struct = generate_binary_structure(3,1)
        mask = binary_closing(mask, structure=struct, iterations=close_iter)
    if remove_small:
        struct = generate_binary_structure(3,1)
        lbl, num = label(mask, structure=struct)
        if num > 1:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            keep = np.argmax(counts)
            before = mask.sum()
            mask = (lbl == keep)
            if verbose:
                print(f"去除小组件: 保留 {keep} 号, 体素 {mask.sum()} / 原 {before}")
    return mask

# ================== 🆕 流体域生成函数 ==================

def create_fluid_domain_from_solid_mask(
    solid_mask: np.ndarray,
    x_range, y_range, z_range,
    resolution: int,
    add_boundary_box: bool = True,
    boundary_thickness: int = 2,
    z_extension: float = 0.0,  # ← 新增：Z 方向物理扩展长度（单位：与坐标系一致）
    smooth: bool = True,
    smooth_iter: int = 10,
    remove_small: bool = True,
    min_voxels: int = 50,
    verbose: bool = True
) -> Any:
    """
    从实体掩码生成流体域网格，支持 Z 轴方向扩展以提供入口/出口缓冲区。
    
    参数:
        solid_mask: 实体结构的布尔掩码 (True=实体, False=流体)，shape=(res, res, res)
        x_range, y_range, z_range: 原始物理坐标范围，如 (-1.5, 1.5)
        resolution: 原始分辨率（用于 solid_mask）
        add_boundary_box: 是否添加 X/Y 方向的封闭边界盒
        boundary_thickness: 边界盒厚度（体素数）
        z_extension: Z 方向上下各扩展的物理长度（如 5.0 表示总高 +10.0）
        ...
    """
    if verbose:
        print(f"生成流体域: 分辨率 {resolution}^3, Z 扩展 = {z_extension:.2f}")
    t0 = time.time()
    
    # 计算原始体素尺寸
    orig_spacing = (
        (x_range[1] - x_range[0]) / (resolution - 1),
        (y_range[1] - y_range[0]) / (resolution - 1),
        (z_range[1] - z_range[0]) / (resolution - 1)
    )
    
    # 流体掩码 = 实体掩码的反向
    fluid_mask = ~solid_mask  # shape: (res, res, res)

    # === 🆕 Z 轴扩展逻辑 ===
    if z_extension > 0:
        # 计算需要扩展的体素数（向上取整）
        z_ext_voxels = int(np.ceil(z_extension / orig_spacing[2]))
        if z_ext_voxels <= 0:
            z_ext_voxels = 1
        
        # 创建扩展后的掩码
        new_z_size = resolution + 2 * z_ext_voxels
        extended_fluid_mask = np.zeros((resolution, resolution, new_z_size), dtype=bool)
        
        # 将原始流体掩码放入中间
        extended_fluid_mask[:, :, z_ext_voxels:-z_ext_voxels] = fluid_mask
        
        # 上下扩展区域设为流体（True）
        extended_fluid_mask[:, :, :z_ext_voxels] = True   # 下方扩展区
        extended_fluid_mask[:, :, -z_ext_voxels:] = True  # 上方扩展区
        
        fluid_mask = extended_fluid_mask
        
        # 更新 z_range 以反映物理扩展
        z_range = (
            z_range[0] - z_extension,
            z_range[1] + z_extension
        )
        
        # 更新 Z 方向体素尺寸（总长度变长，但体素数也变多）
        new_z_length = z_range[1] - z_range[0]
        new_z_voxels = fluid_mask.shape[2]
        new_z_spacing = new_z_length / (new_z_voxels - 1) if new_z_voxels > 1 else orig_spacing[2]
        spacing = (orig_spacing[0], orig_spacing[1], new_z_spacing)
    else:
        spacing = orig_spacing
        z_ext_voxels = 0

    # === 添加 X/Y 方向边界盒（Z 方向保持开放）===
    if add_boundary_box:
        boundary_mask = np.zeros_like(fluid_mask)
        t = boundary_thickness
        
        # 仅封闭 X 和 Y 方向（侧壁）
        boundary_mask[:t, :, :] = True   # -X
        boundary_mask[-t:, :, :] = True  # +X
        boundary_mask[:, :t, :] = True   # -Y
        boundary_mask[:, -t:, :] = True  # +Y
        # Z 方向不封闭（保持开放，用于流体流入流出）
        
        # 从边界中减去实体部分（注意：solid_mask 仍是原始大小）
        # 需要将 solid_mask 扩展到新尺寸（仅 Z 向）
        if z_ext_voxels > 0:
            extended_solid_mask = np.zeros_like(fluid_mask, dtype=bool)
            extended_solid_mask[:, :, z_ext_voxels:-z_ext_voxels] = solid_mask
            solid_for_boundary = extended_solid_mask
        else:
            solid_for_boundary = solid_mask
        
        boundary_mask = boundary_mask & (~solid_for_boundary)
        fluid_mask = fluid_mask | boundary_mask

    # === 后处理流体掩码 ===
    fluid_mask = _postprocess_mask(
        fluid_mask, 
        morph_close=True, 
        close_iter=1,
        remove_small=remove_small, 
        min_voxels=min_voxels, 
        verbose=verbose
    )
    
    # === 构建 PyVista 网格 ===
    # 计算 origin（X/Y 扩展用于边界盒，Z 不扩展）
    origin = (
        x_range[0] - boundary_thickness * spacing[0],
        y_range[0] - boundary_thickness * spacing[1],
        z_range[0] - spacing[2]  # Z 方向只 pad=1，不加边界厚度
    )
    
    # Padding（所有方向 +1）
    padded_mask = np.pad(fluid_mask, pad_width=1, mode='constant', constant_values=False)
    
    # 创建标量场
    S_fluid = padded_mask.astype(np.float32) - 0.5
    
    # 设置网格
    grid = pv.ImageData()
    grid.dimensions = [
        resolution + 2,           # X
        resolution + 2,           # Y
        fluid_mask.shape[2] + 2   # Z（动态）
    ]
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data['fluid'] = S_fluid.flatten(order='F')
    
    # 提取等值面
    try:
        surface = grid.contour([0.0], 'fluid', method='flying_edges')
    except Exception:
        surface = grid.contour([0.0], 'fluid')
    
    # 清理小组件
    if remove_small and surface is not None and surface.n_cells > 0:
        try:
            surface = surface.connectivity(extraction_mode='largest')
        except Exception as e:
            if verbose: 
                print(f"流体域连通域清理失败: {e}")
    
    # 平滑
    if smooth and surface is not None and surface.n_points > 0:
        try:
            if hasattr(surface, 'smooth_taubin'):
                surface = surface.smooth_taubin(n_iter=smooth_iter)
            else:
                surface = surface.smooth(n_iter=smooth_iter, relaxation_factor=0.01)
        except Exception as e:
            if verbose: 
                print(f"流体域平滑失败: {e}")
    
    fluid_porosity = fluid_mask.mean()
    if verbose:
        print(f"流体域完成: 点 {surface.n_points if surface else 0}, "
              f"单元 {surface.n_cells if surface else 0}, "
              f"流体体积分数 {fluid_porosity*100:.2f}%, "
              f"耗时 {time.time()-t0:.2f}s")
    
    return surface, fluid_porosity
# ================== 核心：多 TPMS 混合实体生成（返回掩码） ==================

def create_hybrid_tpms_solid(tpms_funcs: List[Callable], x_range, y_range, z_range,
                              weight_volume: np.ndarray,
                              density_field: Optional[np.ndarray] = None,
                              rotation_field: Optional[np.ndarray] = None,
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
                              return_mask=False):  # 🆕 新增参数
    """
    从多个 TPMS 函数 + 权重体 + 可选密度场 + 可选旋转场 生成实体结构
    
    🆕 rotation_field: 形状 (res,res,res,3) 的旋转场，每个体素的 [rot_x, rot_y, rot_z]（角度）
    🆕 return_mask=True 时同时返回实体掩码，用于生成流体域
    """
    if verbose:
        names = [f.__name__ for f in tpms_funcs]
        print(f"生成混合实体: 分辨率 {resolution}^3, 使用 {len(tpms_funcs)} 个 TPMS: {names}")
    t0 = time.time()

    # 生成空间网格
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 插值权重体
    if weight_volume.shape[:3] != (resolution, resolution, resolution):
        if verbose: print(f"插值权重体: {weight_volume.shape} -> {(resolution, resolution, resolution, -1)}")
        sx = resolution / weight_volume.shape[0]
        sy = resolution / weight_volume.shape[1]
        sz = resolution / weight_volume.shape[2]
        zoom_factors = (sx, sy, sz) + (1,) * (weight_volume.ndim - 3)
        weight_volume = zoom(weight_volume.astype(np.float32), zoom_factors, order=1)

    # 确保权重归一化
    W = np.clip(weight_volume, 0, 1)
    if W.shape[-1] > 1:
        W_sum = W.sum(axis=-1, keepdims=True)
        W = np.divide(W, W_sum, out=np.zeros_like(W), where=W_sum != 0)

    # 插值旋转场（如果提供）
    if rotation_field is not None:
        if rotation_field.shape[:3] != (resolution, resolution, resolution):
            if verbose: print(f"插值旋转场: {rotation_field.shape} -> {(resolution, resolution, resolution, 3)}")
            sx = resolution / rotation_field.shape[0]
            sy = resolution / rotation_field.shape[1]
            sz = resolution / rotation_field.shape[2]
            rotation_field = zoom(rotation_field.astype(np.float32), (sx, sy, sz, 1), order=1)

    # 计算混合标量场 F = Σ (W_i * F_i)
    # 优化：为了防止旋转时产生的坐标“拉扯”现象（由于绕原点旋转导致的杠杆效应），
    # 我们将旋转中心移动到当前计算域的几何中心。
    # 这样可以最大程度减小旋转带来的位移变形。
    if rotation_field is not None:
        # 1. 准备旋转场 (角度 -> 弧度)
        rot_x_rad = np.deg2rad(rotation_field[..., 0])
        rot_y_rad = np.deg2rad(rotation_field[..., 1])
        rot_z_rad = np.deg2rad(rotation_field[..., 2])

        # 2. 计算几何中心
        cx = (x_range[0] + x_range[1]) / 2.0
        cy = (y_range[0] + y_range[1]) / 2.0
        cz = (z_range[0] + z_range[1]) / 2.0

        if verbose:
            print(f"应用坐标旋转优化: 旋转中心 set to ({cx:.2f}, {cy:.2f}, {cz:.2f})")

        # 3. 将坐标移动到中心相对位置
        X_rel = X - cx
        Y_rel = Y - cy
        Z_rel = Z - cz

        # 4. 执行旋转 (顺序与 TPMS 函数内部一致: X -> Y -> Z)
        # 绕 X 轴
        c_x, s_x = np.cos(rot_x_rad), np.sin(rot_x_rad)
        Y_new = Y_rel * c_x - Z_rel * s_x
        Z_new = Y_rel * s_x + Z_rel * c_x
        Y_rel, Z_rel = Y_new, Z_new
        
        # 绕 Y 轴
        c_y, s_y = np.cos(rot_y_rad), np.sin(rot_y_rad)
        X_new = X_rel * c_y + Z_rel * s_y
        Z_new = -X_rel * s_y + Z_rel * c_y
        X_rel, Z_rel = X_new, Z_new
        
        # 绕 Z 轴
        c_z, s_z = np.cos(rot_z_rad), np.sin(rot_z_rad)
        X_new = X_rel * c_z - Y_rel * s_z
        Y_new = X_rel * s_z + Y_rel * c_z
        X_rel, Y_rel = X_new, Y_new

        # 5. 恢复绝对坐标 (但相对于旋转后的中心)
        # 注意：这里我们将旋转后的相对坐标加上中心，得到“在原位置发生自旋”的效果
        X_in = X_rel + cx
        Y_in = Y_rel + cy
        Z_in = Z_rel + cz
    else:
        X_in, Y_in, Z_in = X, Y, Z

    F_total = np.zeros_like(X)
    for i, func in enumerate(tpms_funcs):
        # 由于坐标已经预先旋转，这里不再传入 rot 参数 (默认为 0)
        Fi = func(X_in, Y_in, Z_in)
        W_i = W[..., i]
        F_total += W_i * Fi

    F = F_total
    absF = np.abs(F)
    flat_abs = absF.ravel()

    # 计算阈值
    if density_field is not None:
        s_local = np.clip(density_field.astype(np.float32), 0.0, 1.0)
        arr = np.sort(flat_abs)
        q = np.linspace(0.0, 1.0, arr.size)
        th_local = np.interp(s_local.ravel(), q, arr).reshape(s_local.shape)
        mask = (absF <= th_local)
        if verbose:
            exp_porosity = 1.0 - float(s_local.mean())
            print(f"使用密度→固体分数量化映射，th范围: {th_local.min():.5f} ~ {th_local.max():.5f}，期望孔隙率≈{exp_porosity*100:.2f}%")
        used_threshold = "local_quantile"
    else:
        if global_target_porosity is not None:
            th = _auto_threshold(absF, global_target_porosity, verbose=verbose)
        else:
            th = solid_threshold
            if verbose: print(f"使用固定阈值 th={th:.5f}")

        if min_threshold is not None and th < min_threshold:
            if verbose: print(f"阈值提升到 min_threshold {min_threshold}")
            th = min_threshold

        if global_target_porosity is not None and refine_threshold:
            lo = flat_abs.min(); hi = flat_abs.max()
            th = float(np.clip(th, lo, hi))

            def porosity_for(th_val):
                mask_tmp = (absF <= th_val)
                if not refine_ignore_morph and morph_close:
                    struct = generate_binary_structure(3,1)
                    mask_tmp = binary_closing(mask_tmp, structure=struct, iterations=1)
                return 1 - mask_tmp.mean()

            target_p = float(global_target_porosity)
            best_th = th
            best_p = porosity_for(th)
            best_err = abs(best_p - target_p)
            if verbose:
                print(f"细化开始: 初始孔隙率 {best_p*100:.2f}% 目标 {target_p*100:.2f}")

            for i in range(refine_iters):
                mid = 0.5 * (lo + hi)
                p_mid = porosity_for(mid)
                if p_mid > target_p:
                    lo = mid
                else:
                    hi = mid

                cand_pairs = [(mid, p_mid), (best_th, best_p)]
                for th_c, p_c in cand_pairs:
                    err = abs(p_c - target_p)
                    if err < best_err:
                        best_err = err; best_th = th_c; best_p = p_c

                if verbose:
                    print(f"  迭代 {i+1}: mid_th={mid:.5f} porosity={p_mid*100:.2f}% best_th={best_th:.5f} err={best_err*100:.2f}%")
                if best_err <= porosity_tol:
                    if verbose:
                        print(f"达到容差 {porosity_tol*100:.2f}% 提前停止")
                    break

            th = best_th
            if verbose:
                print(f"细化完成: 最终 th={th:.5f} 估计孔隙率 {best_p*100:.2f}% 误差 {best_err*100:.2f}%")
            if min_threshold is not None and th < min_threshold:
                th = min_threshold

        mask = (absF <= th)
        used_threshold = th

    # 生成二值掩码并后处理
    mask = _postprocess_mask(mask, morph_close=morph_close, close_iter=close_iter,
                             remove_small=remove_small, min_voxels=min_voxels, verbose=verbose)

    # 使用连续标量场生成等值面
    spacing = ((x_range[1]-x_range[0])/(resolution-1),
               (y_range[1]-y_range[0])/(resolution-1),
               (z_range[1]-z_range[0])/(resolution-1))
    origin = (x_range[0]-spacing[0], y_range[0]-spacing[1], z_range[0]-spacing[2])

    if density_field is not None:
        S = (th_local - absF).astype(np.float32)
    else:
        S = (used_threshold - absF).astype(np.float32)

    paddedS = np.full((resolution+2, resolution+2, resolution+2), -1.0, dtype=np.float32)
    paddedS[1:-1,1:-1,1:-1] = S

    grid = pv.ImageData()
    grid.dimensions = [resolution+2] * 3
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data['S'] = paddedS.flatten(order='F')
    try:
        surface = grid.contour([0.0], 'S', method='flying_edges')
    except Exception:
        surface = grid.contour([0.0], 'S')

    # 网格连通域清理
    if remove_small and surface is not None and surface.n_cells > 0:
        try:
            surface = surface.connectivity(extraction_mode='largest')
        except Exception as e:
            if verbose: print(f"连通域清理失败: {e}")

    # 网格平滑
    if smooth and surface.n_points > 0:
        try:
            if hasattr(surface, 'smooth_taubin'):
                surface = surface.smooth_taubin(n_iter=smooth_iter)
            else:
                surface = surface.smooth(n_iter=smooth_iter, relaxation_factor=0.01)
        except Exception as e:
            if verbose: print(f"平滑失败: {e}")

    actual_porosity = 1 - mask.mean()
    if verbose:
        print(f"完成: 点 {surface.n_points}, 单元 {surface.n_cells}, 实际孔隙率 {actual_porosity*100:.2f}%, 耗时 {time.time()-t0:.2f}s")

    if return_mask:
        return surface, actual_porosity, used_threshold, mask  # 🆕 返回掩码
    else:
        return surface, actual_porosity, used_threshold

# ================== 导出 & 可视化 ==================

def export_stl(mesh: Any, filename):
    mesh.save(filename)
    print(f"STL 已保存: {os.path.abspath(filename)}")

def visualize_weight_cube_3d(weight_grid_3x3x3: np.ndarray, 
                            tpms_names: List[str], 
                            density_grid_3x3x3: Optional[np.ndarray] = None,
                            title="TPMS Weight & Density",
                            show_values=True,
                            base_colors=None,
                            base_colors_light=None,
                            base_colors_dark=None,
                            figsize=7,
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
    N = weight_grid_3x3x3.shape[-1]

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

    fig = plt.figure(figsize=(figsize*1.2, figsize), constrained_layout=True)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)

    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect((1, 1, 1))
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

    if density_grid_3x3x3 is not None:
        d_norm = np.clip(density_grid_3x3x3, 0.0, 1.0)
    else:
        d_norm = np.ones((3,3,3))

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

    for x in range(3):
        for y in range(3):
            for z in range(3):
                w = weight_grid_3x3x3[x, y, z]  # shape (N,)
                total_weight = float(np.sum(w))
                if total_weight < 1e-6:
                    continue

                # === 🎨 关键修改：按权重混合颜色 ===
                w_norm = w / total_weight  # 归一化权重
                mixed_light = np.sum(w_norm[:, None] * base_rgb_light, axis=0)
                mixed_dark  = np.sum(w_norm[:, None] * base_rgb_dark,  axis=0)

                d_raw = float(d_norm[x, y, z])
                d_scaled = _scale_density(d_raw)
                t = d_scaled if density_darken else (1.0 - d_scaled)
                t = float(np.clip(t, 0.0, 1.0))
                t = t * float(np.clip(density_effect_strength, 0.0, 1.0))
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

                # === 🏷️ 改进标签：显示前两个主导 TPMS ===
                if show_values:
                    txt_color = 'black' if luminance(shaded_color) > 0.5 else 'white'
                    
                    # 获取前两个最大权重的索引
                    sorted_indices = np.argsort(w)[::-1]
                    top1_idx, top2_idx = sorted_indices[0], sorted_indices[1] if N > 1 else None
                    top1_w = w[top1_idx]
                    
                    if N == 1 or (top2_idx is not None and w[top2_idx] < 0.05):
                        # 几乎是纯的
                        label_text = f"{tpms_names[top1_idx][:4]}\n{top1_w:.1f}"
                    else:
                        # 显示两个
                        top2_w = w[top2_idx]
                        name1 = tpms_names[top1_idx][:2]
                        name2 = tpms_names[top2_idx][:2]
                        label_text = f"{name1}/{name2}\n{top1_w:.1f}/{top2_w:.1f}"
                    
                    label_text += f"\n(d={d_raw:.1f})"
                    
                    ax.text(x + 0.05, y + 0.05, z + 0.05,
                            label_text,
                            fontsize=7, color=txt_color, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor=np.append(shaded_color, 0.75),
                                      edgecolor='none'))

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, 3.6)
    ax.set_zlim(-0.6, 3.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])
    ax.tick_params(axis='both', which='major', labelsize=8)

    corners = np.array([
        [-0.6, -0.6, -0.6], [ 3.6, -0.6, -0.6],
        [-0.6,  3.6, -0.6], [ 3.6,  3.6, -0.6],
        [-0.6, -0.6,  3.6], [ 3.6, -0.6,  3.6],
        [-0.6,  3.6,  3.6], [ 3.6,  3.6,  3.6]
    ])
    edges = [
        (0,1),(0,2),(1,3),(2,3),
        (4,5),(4,6),(5,7),(6,7),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        xs, ys, zs = zip(corners[i], corners[j])
        ax.plot(xs, ys, zs, color="#666", linewidth=1.0, alpha=0.6)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=14, pad=12, fontweight='bold')

    legend_colors = (base_rgb_light + base_rgb_dark) / 2.0
    handles = [plt.Rectangle((0,0), 1, 1, facecolor=legend_colors[i], label=name)
               for i, name in enumerate(tpms_names)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=False)

    return fig, ax

# ================== 🆕 主流程（生成结构域 + 流体域） ==================

def main():
    # === 配置区 ===
    base_x = base_y = base_z = (-1.5, 1.5)
    resolution = 120
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    verbose = True
    replicate = (3, 3, 3)

    # 选择 TPMS
    
    N = len(tpms_list)

    # 权重网格
    manual_weight_grid = get_manual_weight_grid(tpms_list)
    if manual_weight_grid is None:
        manual_weight_grid = generate_random_weight_grid(N, seed=42, normalize_channels=(N>1))
    
    # 密度网格
    density_grid_3x3x3 = get_manual_density_grid(tpms_list)

    # 旋转网格
    rotation_grid_3x3x3 = get_manual_rotation_grid(tpms_list)

    # 可视化
    print("生成 3x3x3 权重+密度立方体可视化...")
    tpms_names = [f.__name__ for f in tpms_list]
    tpms_colors_light, tpms_colors_dark = get_tpms_color_pairs(tpms_names)

    fig, ax = visualize_weight_cube_3d(
        manual_weight_grid, 
        tpms_names,
        density_grid_3x3x3=density_grid_3x3x3,
        title="TPMS Type & Density",
        alpha_power=1.2,
        base_colors_light=tpms_colors_light,
        base_colors_dark=tpms_colors_dark,
        density_darken=True,
        density_effect_strength=2.0,
        show_values=True
    )
    cube_img = os.path.join(output_dir, "weight_density_cube.png")
    fig.savefig(cube_img, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"立方体可视化已保存: {cube_img}")
    
    # 域扩展
    if replicate != (1,1,1):
        x_range, y_range, z_range = expanded_ranges(base_x, base_y, base_z, replicate)
    else:
        x_range, y_range, z_range = base_x, base_y, base_z

    # 生成密度场
    density_field = generate_density_field_from_3x3x3(
        resolution, x_range, y_range, z_range,
        density_grid_3x3x3,
        smooth_sigma=None, 
        method='linear'
    )

    # 生成旋转场
    rotation_field = generate_rotation_field_from_3x3x3(
        resolution, x_range, y_range, z_range,
        rotation_grid_3x3x3,
        smooth_sigma=None,
        method='linear'
    )

    # 权重平滑
    smooth_sigma = 0.5
    weight_volume = generate_weight_matrix(resolution, x_range, y_range, z_range, 
                                          manual_weight_grid, smooth_sigma=smooth_sigma, 
                                          normalize_channels=(N>1))
    combined_fields = np.concatenate([
        density_field[..., None].astype(np.float32),
        weight_volume.astype(np.float32)
    ], axis=-1).astype(np.float32)
    weight_path = os.path.join(output_dir, "fields.npy")
    np.save(weight_path, combined_fields)

    combined_fields_3x3x3 = np.concatenate([
        density_grid_3x3x3[..., None].astype(np.float32),
        manual_weight_grid.astype(np.float32)
    ], axis=-1)
    text_path = os.path.join(output_dir, "fields.txt")
    flat_fields = combined_fields_3x3x3.reshape(-1, combined_fields_3x3x3.shape[-1])
    channel_labels = ["density"] + [f"weight_{name}" for name in tpms_names]
    header = " ".join(channel_labels)
    np.savetxt(text_path, flat_fields, fmt="%.6f", header=header, comments='')

    if verbose:
        print("密度/权重体已保存: fields.npy (通道0=密度, 其余=权重)")
        print("3x3x3 密度/权重表已保存: fields.txt (每行: 密度 + 对应权重)")

    # 实体参数
    global_target_porosity = None
    solid_threshold = 0.3
    min_threshold = None
    smooth = True; smooth_iter = 10
    final_taubin_iter = 10
    morph_close = True; close_iter = 1
    remove_small = True; min_voxels = 80
    desired_size = 20.0
    align_origin = True

    # === 🆕 生成结构域（实体）并返回掩码 ===
    print(f"生成结构域: {[f.__name__ for f in tpms_list]}...")
    solid_mesh, actual_porosity, used_threshold, solid_mask = create_hybrid_tpms_solid(
        tpms_list, x_range, y_range, z_range,
        weight_volume=weight_volume,
        density_field=density_field,
        rotation_field=rotation_field,
        global_target_porosity=global_target_porosity,
        resolution=resolution,
        solid_threshold=solid_threshold,
        min_threshold=min_threshold,
        smooth=smooth, smooth_iter=smooth_iter,
        morph_close=morph_close, close_iter=close_iter,
        remove_small=remove_small, min_voxels=min_voxels,
        verbose=verbose,
        return_mask=True  # 🆕 返回掩码
    )

    voxel_path = os.path.join(output_dir, "voxel.npy")
    np.save(voxel_path, solid_mask.astype(np.uint8))
    if verbose:
        print(f"实体体素掩码已保存: {voxel_path} (值域: 0=流体, 1=实体)")

    # 最终清理
    if verbose: print("结构域最终清理/平滑...")
    solid_mesh = finalize_mesh(solid_mesh, smooth_taubin_iter=final_taubin_iter, 
                               do_clean=True, verbose=verbose)
    if verbose: print(f"结构域最终网格: 点 {solid_mesh.n_points}, 单元 {solid_mesh.n_cells}")

    # === 🆕 生成流体域 ===
    print("生成流体域...")
    fluid_mesh, fluid_porosity = create_fluid_domain_from_solid_mask(
        solid_mask,
        x_range, y_range, z_range,
        resolution=resolution,
        add_boundary_box=False,
        boundary_thickness=0,
        z_extension=0.1,  # ← 关键：上下各扩展 0.1 单位（如 mm）
        smooth=True,
        smooth_iter=smooth_iter,
        remove_small=remove_small,
        min_voxels=min_voxels,
        verbose=verbose
    )

    # 最终清理流体域
    if verbose: print("流体域最终清理/平滑...")
    fluid_mesh = finalize_mesh(fluid_mesh, smooth_taubin_iter=final_taubin_iter, 
                               do_clean=True, verbose=verbose)
    if verbose: print(f"流体域最终网格: 点 {fluid_mesh.n_points}, 单元 {fluid_mesh.n_cells}")

    # 缩放与对齐原点（两个网格同步）
    if desired_size is not None and solid_mesh is not None and solid_mesh.n_points > 0:
        b = solid_mesh.bounds
        lx, ly, lz = b[1]-b[0], b[3]-b[2], b[5]-b[4]
        if lx > 0 and ly > 0 and lz > 0:
            scale_factor = desired_size / max(lx, ly, lz)
            solid_mesh.scale([scale_factor]*3, inplace=True)
            fluid_mesh.scale([scale_factor]*3, inplace=True)
            
            nb = solid_mesh.bounds
            if verbose:
                print(f"已缩放: 原尺寸=({lx:.3f},{ly:.3f},{lz:.3f}) -> 新尺寸≈{desired_size}")
            
            if align_origin:
                offset = [-nb[0], -nb[2], -nb[4]]
                solid_mesh.translate(offset, inplace=True)
                fluid_mesh.translate(offset, inplace=True)
                if verbose:
                    nb2 = solid_mesh.bounds
                    print(f"已对齐原点: bounds=({nb2[0]:.3f},...,{nb2[5]:.3f})")

    # === 🆕 导出两个 STL ===
    suffix = f"_{replicate[0]}x{replicate[1]}x{replicate[2]}" if replicate != (1,1,1) else ""
    
    # 导出结构域（实体）
    solid_filename = os.path.join(output_dir, f"model_solid{suffix}.stl")
    export_stl(solid_mesh, solid_filename)
    
    # 导出流体域
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