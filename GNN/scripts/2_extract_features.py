"""
阶段 2-1: 特征提取脚本

使用预训练的 CNN Backbone 提取特征：
- Layer4: (N, 32, 6, 6, 6) → 用于 Graph-level GNN
- Layer6: (N, 128) → 用于 Node-level GNN

使用方法:
    python scripts/2_extract_features.py --checkpoint checkpoints/cnn_backbone_best.pth
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone
from data.dataset import VoxelDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CNN features')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cnn_backbone_best.pth',
                        help='预训练模型检查点路径')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='特征输出目录')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    return parser.parse_args()


def extract_features(
    model: CNN3DBackbone,
    dataloader: DataLoader,
    device: str
) -> dict:
    """
    提取所有样本的 Layer4 和 Layer6 特征
    
    Returns:
        {
            'layer4': (N, 32, 6, 6, 6),
            'layer6': (N, 128),
            'sample_ids': (N,)
        }
    """
    model.eval()
    
    all_layer4 = []
    all_layer6 = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            voxels = batch['voxel'].to(device)
            sample_ids = batch['sample_id']
            
            # 提取特征
            features = model.forward_features(voxels)
            
            all_layer4.append(features['layer4'].cpu().numpy())
            all_layer6.append(features['layer6'].cpu().numpy())
            all_ids.extend(sample_ids.tolist() if isinstance(sample_ids, torch.Tensor) else sample_ids)
    
    # 合并
    layer4 = np.concatenate(all_layer4, axis=0)  # (N, 32, 6, 6, 6)
    layer6 = np.concatenate(all_layer6, axis=0)  # (N, 128)
    sample_ids = np.array(all_ids)
    
    return {
        'layer4': layer4,
        'layer6': layer6,
        'sample_ids': sample_ids
    }


def main():
    args = parse_args()
    
    # 路径处理
    checkpoint_path = os.path.join(ROOT_DIR, args.checkpoint)
    dataset_dir = os.path.join(ROOT_DIR, args.dataset_dir)
    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # ===== 加载模型 =====
    print(f"\n加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"[错误] 找不到检查点文件: {checkpoint_path}")
        print("请先运行 1_pretrain_cnn.py 进行预训练")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        stats = checkpoint.get('stats', {})
        num_outputs = stats.get('num_outputs', 1)
        
        model = CNN3DBackbone(
            in_channels=1,
            filters=tuple(config['cnn']['filters']),
            fc_dim=config['cnn']['fc_dim'],
            adaptive_pool_size=config['cnn']['adaptive_pool_size'],
            num_outputs=num_outputs
        )
        print(f"  配置: filters={config['cnn']['filters']}, fc_dim={config['cnn']['fc_dim']}, num_outputs={num_outputs}")
    else:
        # 使用默认配置
        model = CNN3DBackbone()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功 (Epoch {checkpoint.get('epoch', 'N/A')})")
    
    # ===== 加载数据 =====
    print(f"\n加载数据集: {dataset_dir}")
    
    # 加载完整数据集 (不划分，使用 permeability 标签)
    dataset = VoxelDataset(
        dataset_dir=dataset_dir,
        label_type='permeability',  # 使用渗透率标签（已对数变换）
        indices=None,  # 使用全部数据
        transform=None,
        normalize_label=False  # 保存原始值，归一化在训练时处理
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 保持顺序
        num_workers=0,
        pin_memory=True
    )
    
    print(f"样本总数: {len(dataset)}")
    
    # ===== 提取特征 =====
    print("\n开始提取特征...")
    features = extract_features(model, dataloader, device)
    
    print(f"\n特征形状:")
    print(f"  Layer4: {features['layer4'].shape}")
    print(f"  Layer6: {features['layer6'].shape}")
    
    # ===== 保存特征 =====
    print(f"\n保存特征到: {output_dir}")
    
    np.save(os.path.join(output_dir, 'features_layer4.npy'), features['layer4'])
    np.save(os.path.join(output_dir, 'features_layer6.npy'), features['layer6'])
    np.save(os.path.join(output_dir, 'sample_ids.npy'), features['sample_ids'])
    
    # 同时保存标签 (用于后续 GNN 训练)
    labels = dataset.labels
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(f"  Labels: {labels.shape}")
    if len(labels.shape) > 1:
        print(f"    [log(K), 压降]: 第1列 [{labels[:,0].min():.2f}, {labels[:,0].max():.2f}], "
              f"第2列 [{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")
    
    # 保存元信息
    meta = {
        'n_samples': len(dataset),
        'layer4_shape': features['layer4'].shape,
        'layer6_shape': features['layer6'].shape,
        'checkpoint': args.checkpoint,
        'label_type': 'permeability',
        'log_transform': getattr(dataset, 'log_transform', False),
        'num_outputs': labels.shape[1] if len(labels.shape) > 1 else 1
    }
    np.save(os.path.join(output_dir, 'meta.npy'), meta)
    
    print("\n✅ 特征提取完成!")
    print(f"  features_layer4.npy: {features['layer4'].shape} - 用于 Graph-level GNN")
    print(f"  features_layer6.npy: {features['layer6'].shape} - 用于 Node-level GNN")
    print(f"  labels.npy: {labels.shape} - 标签 [log(K), 压降]")


if __name__ == '__main__':
    main()
