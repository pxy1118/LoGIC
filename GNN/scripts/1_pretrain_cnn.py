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
from tqdm import tqdm

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone
from data.dataset import create_data_loaders


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
    parser.add_argument('--label_type', type=str, default='permeability',
                        choices=['porosity', 'permeability'],
                        help='标签类型')
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
    grad_clip: float = 1.0
) -> dict:
    """训练一个 epoch
    
    Args:
        grad_clip: 梯度裁剪阈值，防止梯度爆炸
    """
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch in pbar:
        voxels = batch['voxel'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(voxels)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
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
            r2_list.append(1 - (ss_res / (ss_tot + 1e-8)))
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
    epochs = args.epochs or config['training']['cnn_epochs']
    batch_size = args.batch_size or config['training']['cnn_batch_size']
    lr = args.lr or config['training']['cnn_lr']
    
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
    train_loader, val_loader, stats = create_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        train_ratio=config['data']['train_ratio'],
        label_type=args.label_type,
        num_workers=0,  # Windows 下使用 0
        use_augmentation=True,
        seed=42
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
    model = CNN3DBackbone(
        in_channels=1,
        filters=tuple(config['cnn']['filters']),
        fc_dim=config['cnn']['fc_dim'],
        adaptive_pool_size=config['cnn']['adaptive_pool_size'],
        dropout=config['cnn']['dropout'],
        num_outputs=num_outputs
    )
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # ===== 优化器与损失函数 =====
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()  # MAE Loss
    
    # ===== 恢复训练 =====
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"从 epoch {start_epoch} 继续训练")
    
    # ===== 训练循环 =====
    print(f"\n开始训练 (共 {epochs} 轮)...")
    print(f"优化器: AdamW | 学习率调度: CosineAnnealingLR | 梯度裁剪: 1.0")
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
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, grad_clip)
        
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
            # 多输出: [log(渗透率), 压降]
            mae_str = f"logK:{mae_val[0]:.3f} dP:{mae_val[1]:.2f}"
            r2_str = f"logK:{r2_val[0]:.3f} dP:{r2_val[1]:.3f}"
        else:
            mae_str = f"{mae_val:.4f}"
            r2_str = f"{r2_val:.4f}"
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_metrics['loss']:.4f} | "
              f"Val: {val_metrics['loss']:.4f} | "
              f"MAE: {mae_str} | "
              f"R²: {r2_str} | "
              f"LR: {current_lr:.2e} | "
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
