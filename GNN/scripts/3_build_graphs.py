"""
阶段 2-2: 构建图数据脚本

从提取的特征构建图数据:
1. Node-level 图: 整个数据集形成一个大图
2. Graph-level 图: 每个样本形成一个独立的图

使用方法:
    python scripts/3_build_graphs.py --features_dir features --output_dir graphs
"""

import argparse
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from data.graph_builder import (
    build_node_level_graph,
    build_graph_level_graphs
)


def parse_args():
    parser = argparse.ArgumentParser(description='Build graph data from CNN features')
    parser.add_argument('--features_dir', type=str, default='features',
                        help='特征目录')
    parser.add_argument('--output_dir', type=str, default='graphs',
                        help='图数据输出目录')
    parser.add_argument('--k', type=int, default=5,
                        help='KNN 的 K 值')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['node', 'graph', 'both'],
                        help='构建模式: node/graph/both')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 路径
    features_dir = os.path.join(ROOT_DIR, args.features_dir)
    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 加载特征 =====
    print(f"加载特征: {features_dir}")
    
    features_layer4 = np.load(os.path.join(features_dir, 'features_layer4.npy'))
    features_layer6 = np.load(os.path.join(features_dir, 'features_layer6.npy'))
    labels = np.load(os.path.join(features_dir, 'labels.npy'))
    sample_ids = np.load(os.path.join(features_dir, 'sample_ids.npy'))
    
    n_samples = len(labels)
    num_outputs = labels.shape[1] if len(labels.shape) > 1 else 1
    print(f"样本数: {n_samples}")
    print(f"输出维度: {num_outputs}")
    if num_outputs > 1:
        print(f"标签范围: log(K) [{labels[:,0].min():.2f}, {labels[:,0].max():.2f}], "
              f"压降 [{labels[:,1].min():.2f}, {labels[:,1].max():.2f}]")
    print(f"Layer4 特征: {features_layer4.shape}")
    print(f"Layer6 特征: {features_layer6.shape}")
    
    # ===== 数据划分 =====
    print(f"\n划分数据集 (train={args.train_ratio:.0%})...")
    
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices, 
        train_size=args.train_ratio,
        random_state=args.seed
    )
    
    # 创建掩码
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    print(f"训练集: {train_mask.sum()}, 验证集: {val_mask.sum()}")
    
    # ===== 标签归一化 (基于训练集统计) =====
    print("\n计算标签归一化统计...")
    train_labels = labels[train_idx]
    
    if num_outputs > 1:
        label_mean = train_labels.mean(axis=0)
        label_std = train_labels.std(axis=0)
        label_std = np.where(label_std < 1e-8, 1.0, label_std)  # 避免除零
    else:
        label_mean = train_labels.mean()
        label_std = train_labels.std()
        if label_std < 1e-8:
            label_std = 1.0
    
    # 归一化所有标签
    labels_normalized = (labels - label_mean) / label_std
    
    print(f"  原始标签范围: {labels.min(axis=0)} ~ {labels.max(axis=0)}")
    print(f"  归一化后范围: {labels_normalized.min(axis=0)} ~ {labels_normalized.max(axis=0)}")
    print(f"  均值: {label_mean}, 标准差: {label_std}")
    
    # ===== 构建 Node-level 图 =====
    if args.mode in ['node', 'both']:
        print(f"\n构建 Node-level 图 (K={args.k})...")
        
        node_graph = build_node_level_graph(
            features_layer6,
            labels_normalized,  # 使用归一化后的标签
            k=args.k,
            train_mask=train_mask,
            val_mask=val_mask
        )
        
        print(f"  节点数: {node_graph.num_nodes}")
        print(f"  边数: {node_graph.num_edges}")
        print(f"  节点特征维度: {node_graph.x.shape[1]}")
        print(f"  平均度: {node_graph.num_edges / node_graph.num_nodes:.2f}")
        
        # 保存
        node_path = os.path.join(output_dir, 'node_level_graph.pt')
        torch.save(node_graph, node_path)
        print(f"  保存至: {node_path}")
    
    # ===== 构建 Graph-level 图列表 =====
    if args.mode in ['graph', 'both']:
        print(f"\n构建 Graph-level 图 (K={args.k})...")
        
        graph_list = build_graph_level_graphs(
            features_layer4,
            labels_normalized,  # 使用归一化后的标签
            k=args.k
        )
        
        print(f"  图数量: {len(graph_list)}")
        print(f"  单图节点数: {graph_list[0].num_nodes}")
        print(f"  单图边数: {graph_list[0].num_edges}")
        print(f"  节点特征维度: {graph_list[0].x.shape[1]}")
        
        # 添加划分信息
        train_graphs = [graph_list[i] for i in train_idx]
        val_graphs = [graph_list[i] for i in val_idx]
        
        # 保存完整列表和划分
        graph_path = os.path.join(output_dir, 'graph_level_data.pt')
        torch.save({
            'graphs': graph_list,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'sample_ids': sample_ids
        }, graph_path)
        print(f"  保存至: {graph_path}")
        
        # 额外保存划分后的数据 (方便直接加载)
        torch.save(train_graphs, os.path.join(output_dir, 'train_graphs.pt'))
        torch.save(val_graphs, os.path.join(output_dir, 'val_graphs.pt'))
    
    # ===== 保存元信息 =====
    meta = {
        'n_samples': n_samples,
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'k': args.k,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'layer4_shape': features_layer4.shape,
        'layer6_shape': features_layer6.shape,
        'num_outputs': num_outputs,
        'label_type': 'permeability',
        'label_mean': label_mean.tolist() if hasattr(label_mean, 'tolist') else float(label_mean),
        'label_std': label_std.tolist() if hasattr(label_std, 'tolist') else float(label_std),
        'normalized': True
    }
    np.save(os.path.join(output_dir, 'meta.npy'), meta)
    
    print(f"\n✅ 图数据构建完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
