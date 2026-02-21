"""
Learning Rate Schedulers with Warmup Support

Provides learning rate schedulers with warmup functionality for stable training.
"""

import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Warmup
    
    Gradually increases learning rate from a small value to the base learning rate
    during the warmup period, then applies the main scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Initial learning rate at the start of warmup (default: 1e-6)
        main_scheduler: Main scheduler to use after warmup (optional)
        last_epoch: The index of last epoch (default: -1)
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> main_scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> scheduler = WarmupScheduler(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=1e-6,
        ...     main_scheduler=main_scheduler
        ... )
        >>> for epoch in range(epochs):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-6,
        main_scheduler=None,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.main_scheduler = main_scheduler
        
        # Store base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # After warmup: use main scheduler if provided
            if self.main_scheduler is not None:
                # Adjust main scheduler's epoch
                self.main_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                return self.main_scheduler.get_last_lr()
            else:
                # No main scheduler: keep base learning rate
                return self.base_lrs
    
    def step(self, epoch=None):
        """Step the scheduler"""
        super().step(epoch)
        
        # Also step the main scheduler if we're past warmup
        if self.main_scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self.main_scheduler.step()


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Warmup
    
    Combines linear warmup with cosine annealing decay. This is a common
    configuration for training deep neural networks.
    
    Learning rate schedule:
    - Warmup (0 to warmup_epochs): Linear increase from warmup_start_lr to base_lr
    - Main (warmup_epochs to T_max): Cosine annealing from base_lr to eta_min
    
    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations (total epochs)
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Initial learning rate at the start of warmup (default: 1e-6)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> scheduler = CosineAnnealingWarmupLR(
        ...     optimizer,
        ...     T_max=100,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=1e-6,
        ...     eta_min=1e-6
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        # Store base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            if self.warmup_epochs == 0:
                return self.base_lrs
            
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            # Adjust epoch to start from 0 after warmup
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_T_max = self.T_max - self.warmup_epochs
            
            if cosine_T_max <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * cosine_epoch / cosine_T_max)) / 2
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(_LRScheduler):
    """
    Simple Linear Warmup Learning Rate Scheduler
    
    Linearly increases learning rate from warmup_start_lr to base_lr over
    warmup_epochs, then keeps base_lr constant.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Initial learning rate at the start of warmup (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupLR(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     warmup_start_lr=1e-6
        ... )
        >>> for epoch in range(epochs):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        # Store base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            if self.warmup_epochs == 0:
                return self.base_lrs
            
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # After warmup: keep base learning rate
            return self.base_lrs


def create_scheduler_with_warmup(
    optimizer,
    scheduler_type: str = 'cosine',
    total_epochs: int = 100,
    warmup_epochs: int = 0,
    warmup_start_lr: float = 1e-6,
    eta_min: float = 1e-6,
    **kwargs
):
    """
    Factory function to create learning rate scheduler with warmup
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'step', 'none')
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Initial learning rate for warmup
        eta_min: Minimum learning rate (for cosine annealing)
        **kwargs: Additional arguments for specific schedulers
    
    Returns:
        Learning rate scheduler
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> scheduler = create_scheduler_with_warmup(
        ...     optimizer,
        ...     scheduler_type='cosine',
        ...     total_epochs=100,
        ...     warmup_epochs=10,
        ...     eta_min=1e-6
        ... )
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingWarmupLR(
            optimizer,
            T_max=total_epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min
        )
    
    elif scheduler_type == 'linear':
        return LinearWarmupLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr
        )
    
    elif scheduler_type == 'none':
        # No scheduling, just warmup
        if warmup_epochs > 0:
            return LinearWarmupLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr
            )
        else:
            # No warmup, no scheduling - return a dummy scheduler
            from torch.optim.lr_scheduler import LambdaLR
            return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Supported: 'cosine', 'linear', 'none'"
        )
