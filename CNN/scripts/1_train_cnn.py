"""
阶段 1: 3D CNN Backbone 预训练脚本

训练目标: 回归预测孔隙率/渗透率
损失函数: MAE (Mean Absolute Error)
优化器: Adam
数据划分: 80% 训练, 20% 验证

使用方法:
    python scripts/1_pretrain_cnn.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone, ImprovedCNN3DBackbone
from models.physics_loss import PhysicsInformedLoss
from data.dataset import create_data_loaders
from utils.lr_scheduler import create_scheduler_with_warmup


class WeightedCurveLoss(nn.Module):
    """
    Curve loss with higher weight for the initial elastic region
    """
    def __init__(self, weight_first_k=10, weight_factor=10.0):
        super().__init__()
        self.k = weight_first_k
        self.factor = weight_factor
        self.base = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        loss = self.base(pred, target) # (B, num_outputs)
        
        # Create weight mask
        weights = torch.ones_like(loss)
        if pred.shape[1] >= self.k:
            weights[:, :self.k] = self.factor
            
        # Weighted mean -> Normalize by sum of weights to keep loss magnitude reasonable
        # Total weight per sample = k*factor + (N-k)*1
        # To avoid altering learning rate scale too much
        return torch.mean(loss * weights) / weights.mean()

class AutoWeightedLoss(nn.Module):
    """
    自动加权多任务损失函数
    
    基于不确定性的任务加权，自动平衡不同数量级的任务
    参考: Multi-Task Learning Using Uncertainty to Weigh Losses
    """
    def __init__(self, num_outputs: int = 2):
        super().__init__()
        # 可学习的对数方差参数
        self.log_vars = nn.Parameter(torch.zeros(num_outputs))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_outputs) 预测值
            target: (B, num_outputs) 真实值
        
        Returns:
            加权损失
        """
        # 确保 log_vars 在正确的设备上
        log_vars = self.log_vars.to(pred.device)
        
        # 精度 = exp(-log_var)
        precision = torch.exp(-log_vars)
        
        # 加权 MAE: precision * |pred - target| + 0.5 * log_var
        # 第一项：加权误差，第二项：正则化（防止 log_var 过大）
        loss = precision * torch.abs(pred - target) + 0.5 * log_vars
        
        return loss.mean()
    
    def get_weights(self) -> torch.Tensor:
        """获取当前任务权重"""
        return torch.exp(-self.log_vars).detach().cpu()


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain 3D CNN Backbone')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='数据集目录 (覆盖配置文件)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (覆盖配置文件)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (覆盖配置文件)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (覆盖配置文件)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练 (AMP)')
    parser.add_argument('--augment_rotate_z', action='store_true',
                        help='启用Z轴旋转数据增强')
    parser.add_argument('--decoder', type=str, default=None,
                        choices=['mlp', 'lstm'],
                        help='解码器类型: mlp/lstm (覆盖配置文件)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: str,
    epoch: int,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    scaler: GradScaler = None
) -> dict:
    """训练一个 epoch
    
    Args:
        grad_clip: 梯度裁剪阈值，防止梯度爆炸
        use_amp: 是否使用混合精度训练
        scaler: AMP GradScaler
    """
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch in pbar:
        voxels = batch['voxel'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 混合精度前向传播
        if use_amp and scaler is not None:
            # Use torch.amp.autocast for compatibility with newer PyTorch versions
            with torch.amp.autocast('cuda'):
                outputs = model(voxels)
                loss = criterion(outputs, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / n_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: str,
    dataset_stats: dict
) -> dict:
    """验证"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    for batch in loader:
        voxels = batch['voxel'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(voxels)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        n_batches += 1
        
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    # 合并预测结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 反归一化计算真实 MAE
    label_mean = np.array(dataset_stats['label_mean'])
    label_std = np.array(dataset_stats['label_std'])
    num_outputs = dataset_stats.get('num_outputs', 1)
    
    preds_denorm = all_preds * label_std + label_mean
    labels_denorm = all_labels * label_std + label_mean
    
    # MAE
    if num_outputs > 1:
        # 分别计算每个输出的 MAE
        mae_per_output = np.mean(np.abs(preds_denorm - labels_denorm), axis=0)
        mae_real = mae_per_output.tolist()  # [渗透率MAE, 压降 MAE]
    else:
        mae_real = float(np.mean(np.abs(preds_denorm - labels_denorm)))
    
    # 计算 R² 分数
    if num_outputs > 1:
        r2_list = []
        for i in range(num_outputs):
            ss_res = np.sum((labels_denorm[:, i] - preds_denorm[:, i]) ** 2)
            ss_tot = np.sum((labels_denorm[:, i] - np.mean(labels_denorm[:, i])) ** 2)
            
            # 如果方差极小（例如恒为0的点），则跳过该点或设为NaN
            if ss_tot < 1e-6:
                r2_list.append(np.nan)
            else:
                r2_list.append(1 - (ss_res / ss_tot))
        r2 = r2_list
    else:
        ss_res = np.sum((labels_denorm - preds_denorm) ** 2)
        ss_tot = np.sum((labels_denorm - np.mean(labels_denorm)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'loss': total_loss / n_batches,
        'mae_real': mae_real,
        'r2': r2
    }


def main():
    args = parse_args()
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, args.config)
    config = load_config(config_path)
    
    # 命令行参数覆盖配置
    dataset_dir = args.dataset_dir or os.path.join(ROOT_DIR, config['data']['dataset_dir'])
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    lr = args.lr or config['training']['lr']
    
    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    checkpoint_dir = os.path.join(ROOT_DIR, config['logging']['checkpoint_dir'])
    log_dir = os.path.join(ROOT_DIR, config['logging']['log_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # ===== 数据加载 =====
    print(f"\n加载数据集: {dataset_dir}")
    augment_rotate_z = args.augment_rotate_z or config['data'].get('augment_rotate_z', False)
    print(f"Z轴旋转增强: {'启用' if augment_rotate_z else '禁用'}")
    
    # 多尺度输入增强配置
    use_multiscale = config['data'].get('augment_multiscale', False)
    multiscale_range = config['data'].get('multiscale_range', [60, 120])
    multiscale_prob = config['data'].get('multiscale_prob', 0.5)
    
    if use_multiscale:
        print(f"多尺度输入增强: 启用 (分辨率范围: {multiscale_range[0]}-{multiscale_range[1]}, 概率: {multiscale_prob})")
    else:
        print(f"多尺度输入增强: 禁用")
    
    # 获取预测目标
    target_names = config['data'].get('prediction_targets', ['E', 'yield'])
    curve_points = config['data'].get('curve_points', None)
    
    train_loader, val_loader, stats = create_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        train_ratio=config['data']['train_ratio'],
        num_workers=0,  # Windows 下使用 0
        use_augmentation=augment_rotate_z,  # 使用 augment_rotate_z 控制
        seed=42,
        target_names=target_names,
        curve_points=curve_points,
        use_multiscale=use_multiscale,
        multiscale_range=tuple(multiscale_range),
        multiscale_prob=multiscale_prob
    )
    
    print(f"训练样本: {stats['n_train']}, 验证样本: {stats['n_val']}")
    num_outputs = stats.get('num_outputs', 1)
    print(f"输出维度: {num_outputs}")
    if num_outputs == 1:
        print(f"标签统计: mean={stats['label_mean']:.4f}, std={stats['label_std']:.4f}")
    else:
        print(f"标签统计: mean={stats['label_mean']}, std={stats['label_std']}")
    
    # ===== 模型创建 =====
    print("\n创建模型...")
    activation = config['cnn'].get('activation', 'elu')
    
    # Check if using improved architecture
    architecture_type = config['cnn'].get('architecture_type', 'simple')
    use_improved = architecture_type == 'resnet' or \
                   config['cnn'].get('attention', {}).get('enabled', False) or \
                   config['cnn'].get('multi_scale_fusion', {}).get('enabled', False)
    
    if use_improved:
        print(f"使用改进架构: {architecture_type.upper()}")
        
        # Decoder configuration
        decoder_config = config['cnn'].get('decoder', {})
        decoder_type = args.decoder or decoder_config.get('type', 'lstm')
        decoder_hidden_dim = decoder_config.get('hidden_dim', 256)
        decoder_num_layers = decoder_config.get('num_layers', 2)
        decoder_num_heads = decoder_config.get('num_heads', 4)
        
        # Attention configuration
        attention_config = config['cnn'].get('attention', {})
        use_attention = attention_config.get('enabled', False)
        attention_type = attention_config.get('type', 'channel')
        attention_positions = attention_config.get('positions', None)
        
        # Multi-scale fusion configuration
        fusion_config = config['cnn'].get('multi_scale_fusion', {})
        use_multi_scale = fusion_config.get('enabled', False)
        fusion_layers = fusion_config.get('fusion_layers', None)
        
        # Residual configuration
        residual_config = config['cnn'].get('residual', {})
        use_bottleneck = residual_config.get('use_bottleneck', False)
        
        print(f"  - 激活函数: {activation.upper()}")
        print(f"  - 解码器: {decoder_type.upper()}")
        if use_attention:
            print(f"  - 注意力机制: {attention_type.upper()} at layers {attention_positions}")
        if use_multi_scale:
            print(f"  - 多尺度融合: layers {fusion_layers}")
        
        model = ImprovedCNN3DBackbone(
            in_channels=1,
            filters=tuple(config['cnn']['filters']),
            fc_dim=config['cnn']['fc_dim'],
            adaptive_pool_size=config['cnn']['adaptive_pool_size'],
            dropout=config['cnn']['dropout'],
            num_outputs=num_outputs,
            activation=activation,
            decoder_type=decoder_type,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_num_heads=decoder_num_heads,
            architecture_type=architecture_type,
            use_attention=use_attention,
            attention_type=attention_type,
            attention_positions=attention_positions,
            use_multi_scale=use_multi_scale,
            fusion_layers=fusion_layers,
            use_bottleneck=use_bottleneck
        )
    else:
        print("使用基础架构 (CNN3DBackbone)")
        decoder_type = args.decoder or config['cnn'].get('decoder_type', 'mlp')
        print(f"  - 激活函数: {activation.upper()}")
        print(f"  - 解码器: {decoder_type.upper()}")
        
        model = CNN3DBackbone(
            in_channels=1,
            filters=tuple(config['cnn']['filters']),
            fc_dim=config['cnn']['fc_dim'],
            adaptive_pool_size=config['cnn']['adaptive_pool_size'],
            dropout=config['cnn']['dropout'],
            num_outputs=num_outputs,
            activation=activation,
            decoder_type=decoder_type
        )
    
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # ===== 损失函数 =====
    # 根据配置选择损失函数
    curve_loss_config = config['training'].get('curve_loss', None)
    physics_loss_config = config['training'].get('physics_loss', {})
    use_physics_loss = physics_loss_config.get('enabled', False)
    
    # Physics-informed loss and composite curve loss are mutually exclusive
    # Physics loss has its own base loss, while composite curve loss is standalone
    if use_physics_loss and num_outputs > 5:
        print("使用物理约束损失 (Physics-Informed Loss)")
        print(f"  - Base loss type: MAE")
        print(f"  - Monotonicity weight: {physics_loss_config.get('monotonicity_weight', 0.1)}")
        print(f"  - Smoothness weight: {physics_loss_config.get('smoothness_weight', 0.05)}")
        print(f"  - Elastic linearity weight: {physics_loss_config.get('elastic_weight', 0.1)}")
        
        criterion = PhysicsInformedLoss(
            base_loss_type='mae',
            monotonicity_weight=physics_loss_config.get('monotonicity_weight', 0.1),
            smoothness_weight=physics_loss_config.get('smoothness_weight', 0.05),
            elastic_weight=physics_loss_config.get('elastic_weight', 0.1),
            elastic_points=physics_loss_config.get('elastic_points', 10),
            expected_seq_len=num_outputs
        ).to(device)
    elif curve_loss_config and num_outputs > 5:  # 曲线预测任务
        print(f"使用复合曲线损失 (Curve Loss)")
        print(f"  - Base: {curve_loss_config.get('base', 'SmoothL1')}")
        print(f"  - Weights: base={curve_loss_config.get('base_weight', 1.0)}, "
              f"slope={curve_loss_config.get('slope_weight', 0.2)}, "
              f"curvature={curve_loss_config.get('curvature_weight', 0.05)}")
        
        # 使用复合损失
        from utils.metrics import CompositeCurveLoss
        criterion = CompositeCurveLoss(
            base_loss=curve_loss_config.get('base', 'SmoothL1'),
            base_weight=curve_loss_config.get('base_weight', 1.0),
            slope_weight=curve_loss_config.get('slope_weight', 0.2),
            curvature_weight=curve_loss_config.get('curvature_weight', 0.05),
            smooth_l1_beta=curve_loss_config.get('smooth_l1_beta', 1.0)
        ).to(device)
    else:
        # 标量任务或简单损失
        criterion = nn.L1Loss().to(device)
        print("使用标准 L1 Loss (MAE)")
    
    # ===== 优化器与调度器 =====
    # 将 criterion 的参数也加入优化器（如果有）
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    
    if isinstance(criterion, AutoWeightedLoss):
        optimizer = AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器 (支持预热)
    warmup_config = config['training'].get('warmup', {})
    warmup_epochs = int(warmup_config.get('epochs', 0))
    warmup_start_lr = float(warmup_config.get('start_lr', 1e-6))
    scheduler_type = warmup_config.get('scheduler', 'cosine')
    
    scheduler = create_scheduler_with_warmup(
        optimizer,
        scheduler_type=scheduler_type,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=1e-6
    )
    
    # ===== 混合精度训练 =====
    use_amp = args.amp or config['training'].get('use_amp', False)
    # Use torch.amp.GradScaler for compatibility
    scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None
    print(f"混合精度训练 (AMP): {'启用' if use_amp else '禁用'}")
    
    # ===== 恢复训练 =====
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"从 epoch {start_epoch} 继续训练")
    
    # ===== 训练循环 =====
    print(f"\n开始训练 (共 {epochs} 轮)...")
    warmup_info = f" | Warmup: {warmup_epochs} epochs" if warmup_epochs > 0 else ""
    print(f"优化器: AdamW | 学习率调度: {scheduler_type.upper()}{warmup_info} | 梯度裁剪: 1.0")
    print(f"激活函数: {activation.upper()} | AMP: {use_amp} | Z轴旋转: {augment_rotate_z}")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    
    # 早停计数器
    patience = config['training']['early_stopping_patience']
    patience_counter = 0
    grad_clip = 1.0  # 梯度裁剪阈值
    
    for epoch in range(start_epoch, epochs + 1):
        t_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, 
            grad_clip, use_amp, scaler
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, stats)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        t_cost = time.time() - t_start
        
        # 打印日志
        current_lr = optimizer.param_groups[0]['lr']
        
        # 格式化 MAE 和 R² (支持多输出)
        mae_val = val_metrics['mae_real']
        r2_val = val_metrics['r2']
        
        if isinstance(mae_val, list):
            # 过滤掉 NaN 的 R2 (即方差为0的点)
            valid_r2 = [x for x in r2_val if not np.isnan(x)]
            
            if len(mae_val) > 5:
                # 输出维度过高时显示平均值
                mae_avg = np.mean(mae_val)
                r2_avg = np.mean(valid_r2) if valid_r2 else 0.0
                mae_str = f"Mean:{mae_avg:.4f}"
                r2_str = f"Mean:{r2_avg:.4f}"
            elif len(mae_val) == 2 and 'E' in target_names and 'yield' in target_names:
                # 特殊处理默认的 E/yield 任务
                mae_str = f"E:{mae_val[0]:.2f} yield:{mae_val[1]:.2f}"
                r2_str = f"E:{r2_val[0]:.3f} yield:{r2_val[1]:.3f}"
            else:
                # 其他情况显示列表
                mae_str = "[" + ", ".join([f"{v:.2f}" for v in mae_val]) + "]"
                r2_str = "[" + ", ".join([f"{v:.3f}" for v in r2_val]) + "]"
        else:
            mae_str = f"{mae_val:.4f}"
            r2_str = f"{r2_val:.4f}"
        
        # 显示任务权重（如果使用自动加权）
        weight_str = ""
        if isinstance(criterion, AutoWeightedLoss):
            weights = criterion.get_weights()
            weight_str = f" | Weights: E:{weights[0]:.3f} Y:{weights[1]:.3f}"
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_metrics['loss']:.4f} | "
              f"Val: {val_metrics['loss']:.4f} | "
              f"MAE: {mae_str} | "
              f"R²: {r2_str} | "
              f"LR: {current_lr:.2e}{weight_str} | "
              f"Time: {t_cost:.1f}s")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_path = os.path.join(checkpoint_dir, 'cnn_backbone_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'stats': stats
            }, best_path)
            print(f"  >> 保存最佳模型 (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if epoch % config['logging']['save_interval'] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'cnn_backbone_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, ckpt_path)
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n早停触发: 验证损失 {patience} 轮未改善")
            break
    
    # ===== 训练结束 =====
    print("=" * 60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存至: {os.path.join(checkpoint_dir, 'cnn_backbone_best.pth')}")
    
    # 保存训练曲线
    np.savez(
        os.path.join(log_dir, 'cnn_training_history.npz'),
        train_losses=train_losses,
        val_losses=val_losses
    )
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.title('Training Curve (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'cnn_training_curve.png'), dpi=150)
        print(f"训练曲线保存至: {os.path.join(log_dir, 'cnn_training_curve.png')}")
        plt.close()
        
    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == '__main__':
    main()
