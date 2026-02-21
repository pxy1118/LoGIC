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



class CompositeCurveLoss(torch.nn.Module):
    """
    复合曲线损失函数
    
    结合多个损失项来更好地学习应力应变曲线：
    1. Base Loss: 点对点的基础损失 (MAE/MSE/SmoothL1)
    2. Slope Loss: 一阶导数损失，保持曲线斜率连续性
    3. Curvature Loss: 二阶导数损失，保持曲线曲率平滑
    """
    
    def __init__(
        self,
        base_loss: str = 'SmoothL1',
        base_weight: float = 1.0,
        slope_weight: float = 0.2,
        curvature_weight: float = 0.05,
        smooth_l1_beta: float = 1.0
    ):
        """
        Args:
            base_loss: 基础损失类型 ('MAE', 'MSE', 'SmoothL1')
            base_weight: 基础损失权重
            slope_weight: 斜率损失权重
            curvature_weight: 曲率损失权重
            smooth_l1_beta: SmoothL1Loss的beta参数
        """
        super().__init__()
        
        self.base_weight = base_weight
        self.slope_weight = slope_weight
        self.curvature_weight = curvature_weight
        
        # 选择基础损失
        if base_loss == 'MAE':
            self.base_loss_fn = torch.nn.L1Loss()
        elif base_loss == 'MSE':
            self.base_loss_fn = torch.nn.MSELoss()
        elif base_loss == 'SmoothL1':
            self.base_loss_fn = torch.nn.SmoothL1Loss(beta=smooth_l1_beta)
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N) 预测曲线
            target: (B, N) 真实曲线
            
        Returns:
            加权总损失
        """
        # 1. 基础损失 (点对点)
        loss_base = self.base_loss_fn(pred, target)
        
        # 2. 斜率损失 (一阶导数)
        if self.slope_weight > 0:
            pred_slope = pred[:, 1:] - pred[:, :-1]
            target_slope = target[:, 1:] - target[:, :-1]
            loss_slope = torch.nn.functional.l1_loss(pred_slope, target_slope)
        else:
            loss_slope = 0.0
        
        # 3. 曲率损失 (二阶导数)
        if self.curvature_weight > 0:
            pred_curvature = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
            target_curvature = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            loss_curvature = torch.nn.functional.l1_loss(pred_curvature, target_curvature)
        else:
            loss_curvature = 0.0
        
        # 加权总损失
        total_loss = (
            self.base_weight * loss_base +
            self.slope_weight * loss_slope +
            self.curvature_weight * loss_curvature
        )
        
        return total_loss
