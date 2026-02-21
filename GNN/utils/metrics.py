"""
评估指标
"""

import numpy as np
import torch
from typing import Union


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """转换为 numpy 数组"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error"""
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    """Mean Squared Error"""
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    """R² Score (Coefficient of Determination)"""
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))) * 100
