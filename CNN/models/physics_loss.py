"""
Physics-Informed Loss Functions for Stress-Strain Curve Prediction

This module implements physics-based loss functions that enforce physical
constraints on predicted stress-strain curves, ensuring predictions are
physically plausible and consistent with material mechanics principles.
"""

import torch
import torch.nn as nn
from typing import Optional


class MonotonicityLoss(nn.Module):
    """
    Monotonicity Loss for Stress-Strain Curves
    
    Enforces that stress values should be non-decreasing (monotonically increasing)
    in the elastic region of the stress-strain curve. This reflects the physical
    constraint that stress increases with strain in the elastic regime.
    
    The loss computes differences between consecutive stress values and penalizes
    negative differences (where stress decreases). For perfectly monotonic curves,
    the loss is zero.
    
    Args:
        elastic_points: Number of initial points considered as elastic region.
                       Default is 10, representing the initial linear portion.
        reduction: Specifies the reduction to apply to the output:
                  'mean' (default) | 'sum' | 'none'
    
    Shape:
        - Input: (B, N) where B is batch size, N is sequence length (41 points)
        - Output: scalar if reduction='mean' or 'sum', (B,) if reduction='none'
    
    Example:
        >>> loss_fn = MonotonicityLoss(elastic_points=10)
        >>> pred = torch.randn(4, 41)  # Batch of 4 predictions
        >>> loss = loss_fn(pred)
        >>> print(loss.item())  # Scalar loss value
    
    Validates: Requirements 4.1
    """
    
    def __init__(self, elastic_points: int = 10, reduction: str = 'mean'):
        super().__init__()
        
        if elastic_points < 2:
            raise ValueError(
                f"elastic_points must be at least 2 to compute differences, "
                f"got {elastic_points}"
            )
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        
        self.elastic_points = elastic_points
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute monotonicity loss for predicted stress-strain curves.
        
        Args:
            pred: Predicted stress values, shape (B, N)
            target: Target stress values (unused, for API compatibility)
        
        Returns:
            Monotonicity loss value
        
        Raises:
            ValueError: If pred has invalid shape or elastic_points exceeds sequence length
        """
        if pred.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (B, N), got {pred.dim()}D tensor with shape {pred.shape}"
            )
        
        batch_size, seq_len = pred.shape
        
        if self.elastic_points > seq_len:
            raise ValueError(
                f"elastic_points ({self.elastic_points}) cannot exceed sequence length ({seq_len})"
            )
        
        # Extract elastic region (first N points)
        elastic_region = pred[:, :self.elastic_points]  # (B, elastic_points)
        
        # Compute differences between consecutive stress values
        # diff[i] = stress[i+1] - stress[i]
        # For monotonic curves, all differences should be >= 0
        diffs = elastic_region[:, 1:] - elastic_region[:, :-1]  # (B, elastic_points-1)
        
        # Penalize negative differences (non-increasing stress)
        # Use ReLU to only penalize violations: max(0, -diff)
        violations = torch.relu(-diffs)  # (B, elastic_points-1)
        
        # Compute loss per sample
        if self.reduction == 'none':
            # Return loss per sample (mean over sequence dimension)
            loss = violations.mean(dim=1)  # (B,)
        elif self.reduction == 'sum':
            # Sum over all violations
            loss = violations.sum()
        else:  # reduction == 'mean'
            # Mean over all violations (default)
            loss = violations.mean()
        
        return loss


class SmoothnessLoss(nn.Module):
    """
    Smoothness Loss for Stress-Strain Curves
    
    Penalizes large second derivatives in the stress-strain curve, enforcing
    smooth transitions without sharp discontinuities. This reflects the physical
    expectation that material behavior changes gradually rather than abruptly.
    
    The loss computes second derivatives (rate of change of the slope) and
    penalizes their magnitude using L2 norm. For perfectly smooth curves with
    constant or gradually changing slopes, the loss is minimal.
    
    Second derivative approximation:
        f''(x_i) ≈ (f(x_{i+1}) - 2*f(x_i) + f(x_{i-1})) / h^2
    
    For stress-strain curves with uniform spacing, we use:
        second_deriv[i] = stress[i+1] - 2*stress[i] + stress[i-1]
    
    Args:
        reduction: Specifies the reduction to apply to the output:
                  'mean' (default) | 'sum' | 'none'
    
    Shape:
        - Input: (B, N) where B is batch size, N is sequence length (41 points)
        - Output: scalar if reduction='mean' or 'sum', (B,) if reduction='none'
    
    Example:
        >>> loss_fn = SmoothnessLoss()
        >>> pred = torch.randn(4, 41)  # Batch of 4 predictions
        >>> loss = loss_fn(pred)
        >>> print(loss.item())  # Scalar loss value
    
    Validates: Requirements 4.2
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute smoothness loss for predicted stress-strain curves.
        
        Args:
            pred: Predicted stress values, shape (B, N)
            target: Target stress values (unused, for API compatibility)
        
        Returns:
            Smoothness loss value (L2 norm of second derivatives)
        
        Raises:
            ValueError: If pred has invalid shape or sequence length < 3
        """
        if pred.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (B, N), got {pred.dim()}D tensor with shape {pred.shape}"
            )
        
        batch_size, seq_len = pred.shape
        
        if seq_len < 3:
            raise ValueError(
                f"Sequence length must be at least 3 to compute second derivatives, "
                f"got {seq_len}"
            )
        
        # Compute second derivatives using finite differences
        # second_deriv[i] = stress[i+1] - 2*stress[i] + stress[i-1]
        # This gives us (N-2) second derivative values
        second_derivs = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # (B, N-2)
        
        # Compute L2 norm (squared magnitude) of second derivatives
        # This penalizes large changes in slope
        squared_second_derivs = second_derivs ** 2  # (B, N-2)
        
        # Apply reduction
        if self.reduction == 'none':
            # Return loss per sample (mean over sequence dimension)
            loss = squared_second_derivs.mean(dim=1)  # (B,)
        elif self.reduction == 'sum':
            # Sum over all second derivatives
            loss = squared_second_derivs.sum()
        else:  # reduction == 'mean'
            # Mean over all second derivatives (default)
            loss = squared_second_derivs.mean()
        
        return loss


class ElasticLinearityLoss(nn.Module):
    """
    Elastic Linearity Loss for Stress-Strain Curves
    
    Enforces linear behavior in the elastic region of the stress-strain curve.
    In the elastic regime, stress and strain have a linear relationship (Hooke's Law),
    so the initial portion of the curve should be well-approximated by a straight line.
    
    The loss fits a linear regression to the first N points (elastic region) and
    computes the mean squared error (MSE) between the actual stress values and the
    fitted line. For perfectly linear elastic regions, the loss is zero.
    
    Linear fit: stress = slope * strain + intercept
    where strain indices are [0, 1, 2, ..., N-1]
    
    Args:
        elastic_points: Number of initial points considered as elastic region.
                       Default is 10, representing the initial linear portion.
        reduction: Specifies the reduction to apply to the output:
                  'mean' (default) | 'sum' | 'none'
    
    Shape:
        - Input: (B, N) where B is batch size, N is sequence length (41 points)
        - Output: scalar if reduction='mean' or 'sum', (B,) if reduction='none'
    
    Example:
        >>> loss_fn = ElasticLinearityLoss(elastic_points=10)
        >>> pred = torch.randn(4, 41)  # Batch of 4 predictions
        >>> loss = loss_fn(pred)
        >>> print(loss.item())  # Scalar loss value
    
    Validates: Requirements 4.3
    """
    
    def __init__(self, elastic_points: int = 10, reduction: str = 'mean'):
        super().__init__()
        
        if elastic_points < 2:
            raise ValueError(
                f"elastic_points must be at least 2 to fit a line, "
                f"got {elastic_points}"
            )
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        
        self.elastic_points = elastic_points
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute elastic linearity loss for predicted stress-strain curves.
        
        Args:
            pred: Predicted stress values, shape (B, N)
            target: Target stress values (unused, for API compatibility)
        
        Returns:
            Elastic linearity loss value (MSE from linear fit)
        
        Raises:
            ValueError: If pred has invalid shape or elastic_points exceeds sequence length
        """
        if pred.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (B, N), got {pred.dim()}D tensor with shape {pred.shape}"
            )
        
        batch_size, seq_len = pred.shape
        
        if self.elastic_points > seq_len:
            raise ValueError(
                f"elastic_points ({self.elastic_points}) cannot exceed sequence length ({seq_len})"
            )
        
        # Extract elastic region (first N points)
        elastic_region = pred[:, :self.elastic_points]  # (B, elastic_points)
        
        # Create strain indices as x-coordinates: [0, 1, 2, ..., N-1]
        # Shape: (elastic_points,)
        strain_indices = torch.arange(
            self.elastic_points, 
            dtype=pred.dtype, 
            device=pred.device
        )
        
        # Fit linear regression: stress = slope * strain + intercept
        # Using least squares solution:
        # slope = (N * sum(x*y) - sum(x) * sum(y)) / (N * sum(x^2) - sum(x)^2)
        # intercept = (sum(y) - slope * sum(x)) / N
        
        N = self.elastic_points
        sum_x = strain_indices.sum()  # Sum of strain indices
        sum_x2 = (strain_indices ** 2).sum()  # Sum of squared strain indices
        
        # Compute sums for each sample in batch
        sum_y = elastic_region.sum(dim=1)  # (B,) - sum of stress values
        sum_xy = (elastic_region * strain_indices.unsqueeze(0)).sum(dim=1)  # (B,) - sum of x*y
        
        # Compute slope and intercept for each sample
        # slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x^2)
        numerator = N * sum_xy - sum_x * sum_y
        denominator = N * sum_x2 - sum_x ** 2
        
        # Avoid division by zero (though denominator should never be zero for N >= 2)
        slope = numerator / (denominator + 1e-8)  # (B,)
        
        # intercept = (sum_y - slope * sum_x) / N
        intercept = (sum_y - slope * sum_x) / N  # (B,)
        
        # Compute fitted line values: y_fit = slope * x + intercept
        # Broadcasting: (B, 1) * (elastic_points,) + (B, 1) -> (B, elastic_points)
        fitted_line = slope.unsqueeze(1) * strain_indices.unsqueeze(0) + intercept.unsqueeze(1)
        
        # Compute MSE between actual and fitted values
        squared_errors = (elastic_region - fitted_line) ** 2  # (B, elastic_points)
        
        # Apply reduction
        if self.reduction == 'none':
            # Return loss per sample (mean over sequence dimension)
            loss = squared_errors.mean(dim=1)  # (B,)
        elif self.reduction == 'sum':
            # Sum over all squared errors
            loss = squared_errors.sum()
        else:  # reduction == 'mean'
            # Mean over all squared errors (default)
            loss = squared_errors.mean()
        
        return loss



class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Composite Loss Function
    
    Combines a base prediction loss (MAE or MSE) with physics-based constraints
    to ensure predicted stress-strain curves are both accurate and physically plausible.
    
    The total loss is a weighted sum of:
    1. Base loss: MAE or MSE between predictions and targets
    2. Monotonicity loss: Penalizes non-increasing stress in elastic region
    3. Smoothness loss: Penalizes sharp discontinuities (large second derivatives)
    4. Elastic linearity loss: Enforces linear behavior in elastic region
    
    Each component can be individually enabled/disabled and weighted via configuration.
    
    Total Loss = base_loss + 
                 monotonicity_weight * monotonicity_loss +
                 smoothness_weight * smoothness_loss +
                 elastic_weight * elastic_linearity_loss
    
    Args:
        base_loss_type: Type of base loss, 'mae' (L1) or 'mse' (L2). Default: 'mae'
        monotonicity_weight: Weight for monotonicity loss. Set to 0 to disable. Default: 0.1
        smoothness_weight: Weight for smoothness loss. Set to 0 to disable. Default: 0.05
        elastic_weight: Weight for elastic linearity loss. Set to 0 to disable. Default: 0.1
        elastic_points: Number of initial points for elastic region. Default: 10
        expected_seq_len: Expected sequence length for validation. Default: 41
        reduction: Reduction method for losses: 'mean' | 'sum' | 'none'. Default: 'mean'
    
    Shape:
        - pred: (B, N) where B is batch size, N is sequence length
        - target: (B, N) where B is batch size, N is sequence length
        - Output: scalar if reduction='mean' or 'sum', (B,) if reduction='none'
    
    Example:
        >>> # Default configuration with all physics losses enabled
        >>> loss_fn = PhysicsInformedLoss()
        >>> pred = torch.randn(4, 41)
        >>> target = torch.randn(4, 41)
        >>> loss = loss_fn(pred, target)
        >>> print(loss.item())
        
        >>> # Custom weights with smoothness disabled
        >>> loss_fn = PhysicsInformedLoss(
        ...     base_loss_type='mse',
        ...     monotonicity_weight=0.2,
        ...     smoothness_weight=0.0,  # Disabled
        ...     elastic_weight=0.15
        ... )
        >>> loss = loss_fn(pred, target)
    
    Validates: Requirements 4.4, 4.5
    """
    
    def __init__(
        self,
        base_loss_type: str = 'mae',
        monotonicity_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        elastic_weight: float = 0.1,
        elastic_points: int = 10,
        expected_seq_len: int = 41,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        # Validate base loss type
        if base_loss_type not in ['mae', 'mse']:
            raise ValueError(
                f"base_loss_type must be 'mae' or 'mse', got '{base_loss_type}'"
            )
        
        # Validate weights are non-negative
        if monotonicity_weight < 0:
            raise ValueError(
                f"monotonicity_weight must be non-negative, got {monotonicity_weight}"
            )
        if smoothness_weight < 0:
            raise ValueError(
                f"smoothness_weight must be non-negative, got {smoothness_weight}"
            )
        if elastic_weight < 0:
            raise ValueError(
                f"elastic_weight must be non-negative, got {elastic_weight}"
            )
        
        # Validate expected sequence length
        if expected_seq_len < 3:
            raise ValueError(
                f"expected_seq_len must be at least 3, got {expected_seq_len}"
            )
        
        # Validate elastic points
        if elastic_points < 2:
            raise ValueError(
                f"elastic_points must be at least 2, got {elastic_points}"
            )
        if elastic_points > expected_seq_len:
            raise ValueError(
                f"elastic_points ({elastic_points}) cannot exceed expected_seq_len ({expected_seq_len})"
            )
        
        self.base_loss_type = base_loss_type
        self.monotonicity_weight = monotonicity_weight
        self.smoothness_weight = smoothness_weight
        self.elastic_weight = elastic_weight
        self.elastic_points = elastic_points
        self.expected_seq_len = expected_seq_len
        self.reduction = reduction
        
        # Initialize base loss
        if base_loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction=reduction)
        else:  # mse
            self.base_loss = nn.MSELoss(reduction=reduction)
        
        # Initialize physics losses (only if weights > 0)
        self.monotonicity_loss = None
        if monotonicity_weight > 0:
            self.monotonicity_loss = MonotonicityLoss(
                elastic_points=elastic_points,
                reduction=reduction
            )
        
        self.smoothness_loss = None
        if smoothness_weight > 0:
            self.smoothness_loss = SmoothnessLoss(reduction=reduction)
        
        self.elastic_loss = None
        if elastic_weight > 0:
            self.elastic_loss = ElasticLinearityLoss(
                elastic_points=elastic_points,
                reduction=reduction
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss combining base loss and physics constraints.
        
        Args:
            pred: Predicted stress values, shape (B, N)
            target: Target stress values, shape (B, N)
        
        Returns:
            Total weighted loss combining all enabled components
        
        Raises:
            ValueError: If pred or target have invalid shapes or sequence length mismatch
        """
        # Validate input shapes
        if pred.dim() != 2:
            raise ValueError(
                f"Expected 2D prediction tensor (B, N), got {pred.dim()}D tensor with shape {pred.shape}"
            )
        
        if target.dim() != 2:
            raise ValueError(
                f"Expected 2D target tensor (B, N), got {target.dim()}D tensor with shape {target.shape}"
            )
        
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got pred: {pred.shape}, target: {target.shape}"
            )
        
        # Validate sequence length
        batch_size, seq_len = pred.shape
        if seq_len != self.expected_seq_len:
            raise ValueError(
                f"Expected sequence length {self.expected_seq_len}, got {seq_len}. "
                f"Physics loss requires predictions with {self.expected_seq_len} points."
            )
        
        # Compute base loss (always enabled)
        base_loss_raw = self.base_loss(pred, target)
        
        # For reduction='none', base_loss returns (B, N) but physics losses return (B,)
        # We need to reduce base_loss to (B,) to match
        if self.reduction == 'none':
            total_loss = base_loss_raw.mean(dim=1)  # (B, N) -> (B,)
        else:
            total_loss = base_loss_raw  # Already scalar or (B,)
        
        # Add physics losses if enabled (weight > 0)
        if self.monotonicity_loss is not None:
            mono_loss = self.monotonicity_loss(pred)
            total_loss = total_loss + self.monotonicity_weight * mono_loss
        
        if self.smoothness_loss is not None:
            smooth_loss = self.smoothness_loss(pred)
            total_loss = total_loss + self.smoothness_weight * smooth_loss
        
        if self.elastic_loss is not None:
            elastic_loss = self.elastic_loss(pred)
            total_loss = total_loss + self.elastic_weight * elastic_loss
        
        return total_loss
    
    def get_loss_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute and return individual loss components for logging/debugging.
        
        This method is useful for monitoring the contribution of each loss component
        during training and understanding which constraints are being violated.
        
        Args:
            pred: Predicted stress values, shape (B, N)
            target: Target stress values, shape (B, N)
        
        Returns:
            Dictionary containing individual loss components:
            {
                'base_loss': float,
                'monotonicity_loss': float or None,
                'smoothness_loss': float or None,
                'elastic_loss': float or None,
                'total_loss': float
            }
        
        Example:
            >>> loss_fn = PhysicsInformedLoss()
            >>> pred = torch.randn(4, 41)
            >>> target = torch.randn(4, 41)
            >>> components = loss_fn.get_loss_components(pred, target)
            >>> print(f"Base: {components['base_loss']:.4f}")
            >>> print(f"Monotonicity: {components['monotonicity_loss']:.4f}")
            >>> print(f"Total: {components['total_loss']:.4f}")
        """
        components = {}
        
        # Base loss
        components['base_loss'] = self.base_loss(pred, target).item()
        
        # Physics losses (if enabled)
        if self.monotonicity_loss is not None:
            components['monotonicity_loss'] = self.monotonicity_loss(pred).item()
        else:
            components['monotonicity_loss'] = None
        
        if self.smoothness_loss is not None:
            components['smoothness_loss'] = self.smoothness_loss(pred).item()
        else:
            components['smoothness_loss'] = None
        
        if self.elastic_loss is not None:
            components['elastic_loss'] = self.elastic_loss(pred).item()
        else:
            components['elastic_loss'] = None
        
        # Total loss
        components['total_loss'] = self.forward(pred, target).item()
        
        return components
