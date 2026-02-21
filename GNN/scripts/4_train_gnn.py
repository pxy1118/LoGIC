"""
阶段 3: GNN 训练脚本

支持两种模式:
1. Node-Level: 在单个大图上进行节点回归
2. Graph-Level: 在图列表上进行图回归 (推荐)

使用方法:
    # Graph-level (推荐)
    python scripts/4_train_gnn.py --mode graph --epochs 1000
    
    # Node-level
    python scripts/4_train_gnn.py --mode node --epochs 1000
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
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.pore_gnn import PoreGNN, NodeLevelGNN, GraphLevelGNN
from utils.metrics import mae, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pore-GNN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--graphs_dir', type=str, default='graphs',
                        help='图数据目录')
    parser.add_argument('--mode', type=str, default='graph',
                        choices=['node', 'graph'],
                        help='训练模式: node/graph')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小 (仅 graph 模式)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--conv_type', type=str, default=None,
                        choices=['ChebConv', 'GCNConv', 'SAGEConv'],
                        help='图卷积类型')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='隐藏层维度 (override config)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='GNN层数 (override config)')
    parser.add_argument('--pooling', type=str, default=None,
                        choices=['sum', 'mean', 'max'],
                        help='图池化方式 (override config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_node_level(
    model: nn.Module,
    data,
    optimizer,
    criterion
) -> dict:
    """训练 Node-Level GNN (一个大图)"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    
    # 只计算训练集节点的损失
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return {'loss': loss.item()}


@torch.no_grad()
def eval_node_level(
    model: nn.Module,
    data,
    criterion,
    mask_name: str = 'val_mask'
) -> dict:
    """评估 Node-Level GNN"""
    model.eval()
    
    out = model(data)
    mask = getattr(data, mask_name)
    
    loss = criterion(out[mask], data.y[mask])
    
    preds = out[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    
    return {
        'loss': loss.item(),
        'mae': mae(labels, preds),
        'r2': r2_score(labels, preds)
    }


def train_graph_level(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: str,
    grad_clip: float = 1.0,
    num_outputs: int = 1
) -> dict:
    """训练 Graph-Level GNN
    
    Args:
        grad_clip: 梯度裁剪阈值
        num_outputs: 输出维度
    """
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        
        # PyG 批处理会将多输出标签拼接，需要重新 reshape
        # batch.y shape: (batch_size * num_outputs,) → (batch_size, num_outputs)
        batch_size = out.shape[0]
        if num_outputs > 1:
            target = batch.y.view(batch_size, num_outputs)
        else:
            target = batch.y.view(batch_size, 1)
        
        loss = criterion(out, target)
        
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {'loss': total_loss / n_batches}


@torch.no_grad()
def eval_graph_level(
    model: nn.Module,
    loader,
    criterion,
    device: str,
    num_outputs: int = 1,
    label_mean: np.ndarray = None,
    label_std: np.ndarray = None
) -> dict:
    """评估 Graph-Level GNN (支持多输出和反归一化)"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch)
        
        # PyG 批处理会将多输出标签拼接，需要重新 reshape
        batch_size = out.shape[0]
        if num_outputs > 1:
            target = batch.y.view(batch_size, num_outputs)
        else:
            target = batch.y.view(batch_size, 1)
        
        loss = criterion(out, target)
        
        total_loss += loss.item()
        n_batches += 1
        
        all_preds.append(out.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    # 反归一化计算真实 MAE
    if label_mean is not None and label_std is not None:
        preds_denorm = preds * label_std + label_mean
        labels_denorm = labels * label_std + label_mean
    else:
        preds_denorm = preds
        labels_denorm = labels
    
    # 计算每个输出的指标 (使用反归一化后的值)
    if num_outputs > 1:
        mae_list = [mae(labels_denorm[:, i], preds_denorm[:, i]) for i in range(num_outputs)]
        r2_list = [r2_score(labels_denorm[:, i], preds_denorm[:, i]) for i in range(num_outputs)]
        return {
            'loss': total_loss / n_batches,
            'mae': mae_list,  # [log(K) MAE, 压降 MAE]
            'r2': r2_list    # [log(K) R², 压降 R²]
        }
    else:
        return {
            'loss': total_loss / n_batches,
            'mae': mae(labels_denorm, preds_denorm),
            'r2': r2_score(labels_denorm, preds_denorm)
        }


def main():
    args = parse_args()
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, args.config)
    config = load_config(config_path)
    
    # 参数覆盖
    graphs_dir = os.path.join(ROOT_DIR, args.graphs_dir)
    epochs = args.epochs or config['training']['gnn_epochs']
    batch_size = args.batch_size or config['training']['gnn_batch_size']
    lr = args.lr or config['training']['gnn_lr']
    conv_type = args.conv_type or config['gnn']['conv_type']
    hidden_dim = args.hidden_dim or config['gnn']['hidden_dim']
    num_layers = args.num_layers or config['gnn']['num_layers']
    pooling = args.pooling or config['gnn']['pooling']
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"训练模式: {args.mode.upper()}-Level GNN")
    print(f"配置: Conv={conv_type}, Hidden={hidden_dim}, Layers={num_layers}, Pool={pooling}, LR={lr}")
    
    # 创建输出目录
    checkpoint_dir = os.path.join(ROOT_DIR, config['logging']['checkpoint_dir'])
    log_dir = os.path.join(ROOT_DIR, config['logging']['log_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # ===== 加载数据 =====
    print(f"\n加载图数据: {graphs_dir}")
    
    # 加载元信息（包含归一化统计）
    meta_path = os.path.join(graphs_dir, 'meta.npy')
    meta = np.load(meta_path, allow_pickle=True).item()
    label_mean = np.array(meta.get('label_mean', [0, 0]))
    label_std = np.array(meta.get('label_std', [1, 1]))
    is_normalized = meta.get('normalized', False)
    
    if is_normalized:
        print(f"标签已归一化: mean={label_mean}, std={label_std}")
    
    if args.mode == 'node':
        # Node-Level: 加载单个大图
        node_graph = torch.load(os.path.join(graphs_dir, 'node_level_graph.pt'), weights_only=False)
        node_graph = node_graph.to(device)
        
        print(f"节点数: {node_graph.num_nodes}")
        print(f"边数: {node_graph.num_edges}")
        print(f"训练节点: {node_graph.train_mask.sum().item()}")
        print(f"验证节点: {node_graph.val_mask.sum().item()}")
        
        train_loader = None
        val_loader = None
        data = node_graph
        
    else:
        # Graph-Level: 加载图列表
        train_graphs = torch.load(os.path.join(graphs_dir, 'train_graphs.pt'), weights_only=False)
        val_graphs = torch.load(os.path.join(graphs_dir, 'val_graphs.pt'), weights_only=False)
        
        print(f"训练图: {len(train_graphs)}")
        print(f"验证图: {len(val_graphs)}")
        print(f"节点/图: {train_graphs[0].num_nodes}")
        
        # 检测输出维度
        sample_y = train_graphs[0].y
        num_outputs = sample_y.shape[0] if len(sample_y.shape) == 1 else sample_y.shape[1]
        print(f"输出维度: {num_outputs}")
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        data = None
    
    # ===== 创建模型 =====
    print("\n创建模型...")
    
    # 获取 num_outputs (node-level 从 data.y 获取)
    if args.mode == 'node':
        num_outputs = data.y.shape[1] if len(data.y.shape) > 1 else 1
        print(f"输出维度: {num_outputs}")
    
    model = PoreGNN(
        mode=args.mode,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        conv_type=conv_type,
        cheb_k=config['gnn']['cheb_k'],
        pooling=pooling,
        dropout=0.0,
        num_outputs=num_outputs
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # ===== 优化器与损失 =====
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()  # MAE
    grad_clip = 1.0  # 梯度裁剪阈值
    
    # ===== 训练循环 =====
    print(f"\n开始训练 (共 {epochs} 轮)...")
    print(f"优化器: AdamW | 学习率调度: CosineAnnealingLR | 梯度裁剪: {grad_clip}")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        
        # 训练
        if args.mode == 'node':
            train_metrics = train_node_level(model, data, optimizer, criterion)
            val_metrics = eval_node_level(model, data, criterion, 'val_mask')
        else:
            train_metrics = train_graph_level(model, train_loader, optimizer, criterion, device, grad_clip, num_outputs)
            val_metrics = eval_graph_level(model, val_loader, criterion, device, num_outputs, label_mean, label_std)
        
        scheduler.step()  # CosineAnnealingLR 不需要 loss
        
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        t_cost = time.time() - t_start
        
        # 日志
        if epoch % config['logging']['log_interval'] == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            
            # 格式化多输出指标
            mae_val = val_metrics['mae']
            r2_val = val_metrics['r2']
            
            if isinstance(mae_val, list):
                mae_str = f"logK:{mae_val[0]:.3f} dP:{mae_val[1]:.2f}"
                r2_str = f"logK:{r2_val[0]:.3f} dP:{r2_val[1]:.3f}"
            else:
                mae_str = f"{mae_val:.4f}"
                r2_str = f"{r2_val:.4f}"
            
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train: {train_metrics['loss']:.4f} | "
                  f"Val: {val_metrics['loss']:.4f} | "
                  f"MAE: {mae_str} | "
                  f"R2: {r2_str} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {t_cost:.2f}s")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_path = os.path.join(checkpoint_dir, f'pore_gnn_{args.mode}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'mode': args.mode,
                'conv_type': conv_type,
                'num_outputs': num_outputs,
                'label_mean': label_mean.tolist() if hasattr(label_mean, 'tolist') else label_mean,
                'label_std': label_std.tolist() if hasattr(label_std, 'tolist') else label_std,
                'normalized': is_normalized
            }, best_path)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发: 验证损失 {patience} 轮未改善")
            break
    
    # ===== 训练结束 =====
    print("=" * 70)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型: {os.path.join(checkpoint_dir, f'pore_gnn_{args.mode}_best.pth')}")
    
    # 保存训练历史
    np.savez(
        os.path.join(log_dir, f'gnn_{args.mode}_training_history.npz'),
        train_losses=train_losses,
        val_losses=val_losses
    )
    
    # 绘制曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train', alpha=0.7)
        plt.plot(val_losses, label='Validation', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.title(f'{args.mode.upper()}-Level GNN Training')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # 平滑曲线
        window = min(50, len(train_losses) // 10 + 1)
        if window > 1:
            train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            plt.plot(train_smooth, label='Train (smoothed)')
            plt.plot(val_smooth, label='Val (smoothed)')
        else:
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.title('Smoothed Curves')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'gnn_{args.mode}_training_curve.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == '__main__':
    main()
