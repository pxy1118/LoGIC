"""
预测脚本

使用训练好的 CNN + GNN 模型预测 TPMS 结构的渗透率和压降

使用方法:
    # 预测单个样本 (体素文件)
    python scripts/5_predict.py --input path/to/voxel.npy
    
    # 预测整个目录
    python scripts/5_predict.py --input path/to/voxels_dir --output results.csv
    
    # 预测数据集中的指定样本
    python scripts/5_predict.py --dataset dataset --sample_ids 0 1 2 3
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone
from models.pore_gnn import PoreGNN
from data.graph_builder import build_knn_edges
from torch_geometric.data import Data, Batch


def parse_args():
    parser = argparse.ArgumentParser(description='Predict permeability and pressure drop')
    parser.add_argument('--input', type=str, default=None,
                        help='输入体素文件(.npy)或目录')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集目录 (与 --sample_ids 配合使用)')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=None,
                        help='要预测的样本ID列表')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径')
    parser.add_argument('--cnn_checkpoint', type=str, default='checkpoints/cnn_backbone_best.pth',
                        help='CNN模型检查点')
    parser.add_argument('--gnn_checkpoint', type=str, default='checkpoints/pore_gnn_graph_best.pth',
                        help='GNN模型检查点')
    parser.add_argument('--use_gnn', action='store_true', default=True,
                        help='使用GNN进行预测 (默认True)')
    parser.add_argument('--use_cnn_only', action='store_true',
                        help='仅使用CNN预测')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--k', type=int, default=5,
                        help='图构建的KNN参数')
    return parser.parse_args()


class Predictor:
    """TPMS 渗透率/压降预测器"""
    
    def __init__(
        self,
        cnn_checkpoint: str,
        gnn_checkpoint: str = None,
        device: str = 'cuda',
        k: int = 5
    ):
        """
        Args:
            cnn_checkpoint: CNN模型检查点路径
            gnn_checkpoint: GNN模型检查点路径 (可选)
            device: 计算设备
            k: 图构建的KNN参数
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.k = k
        
        # 加载CNN模型
        print(f"加载CNN模型: {cnn_checkpoint}")
        cnn_ckpt = torch.load(cnn_checkpoint, map_location=self.device, weights_only=False)
        
        config = cnn_ckpt.get('config', {})
        stats = cnn_ckpt.get('stats', {})
        self.num_outputs = stats.get('num_outputs', 2)
        self.label_mean = np.array(stats.get('label_mean', [0, 0]))
        self.label_std = np.array(stats.get('label_std', [1, 1]))
        
        # 创建CNN模型
        cnn_config = config.get('cnn', {})
        self.cnn = CNN3DBackbone(
            in_channels=1,
            filters=tuple(cnn_config.get('filters', [8, 16, 32])),
            fc_dim=cnn_config.get('fc_dim', 128),
            adaptive_pool_size=cnn_config.get('adaptive_pool_size', 6),
            num_outputs=self.num_outputs
        ).to(self.device)
        
        self.cnn.load_state_dict(cnn_ckpt['model_state_dict'])
        self.cnn.eval()
        print(f"  CNN加载成功 (Epoch {cnn_ckpt.get('epoch', 'N/A')})")
        
        # 加载GNN模型 (可选)
        self.gnn = None
        self.gnn_label_mean = None
        self.gnn_label_std = None
        self.gnn_normalized = False
        
        if gnn_checkpoint and os.path.exists(gnn_checkpoint):
            print(f"加载GNN模型: {gnn_checkpoint}")
            gnn_ckpt = torch.load(gnn_checkpoint, map_location=self.device, weights_only=False)
            
            gnn_config = gnn_ckpt.get('config', {}).get('gnn', {})
            
            # 获取GNN的归一化统计信息
            self.gnn_normalized = gnn_ckpt.get('normalized', False)
            if self.gnn_normalized:
                self.gnn_label_mean = np.array(gnn_ckpt.get('label_mean', [0, 0]))
                self.gnn_label_std = np.array(gnn_ckpt.get('label_std', [1, 1]))
                print(f"  GNN归一化: mean={self.gnn_label_mean}, std={self.gnn_label_std}")
            
            self.gnn = PoreGNN(
                mode='graph',
                hidden_channels=gnn_config.get('hidden_dim', 32),
                num_layers=gnn_config.get('num_layers', 3),
                conv_type=gnn_ckpt.get('conv_type', 'ChebConv'),
                cheb_k=gnn_config.get('cheb_k', 2),
                pooling=gnn_config.get('pooling', 'sum'),
                num_outputs=gnn_ckpt.get('num_outputs', self.num_outputs)
            ).to(self.device)
            
            self.gnn.load_state_dict(gnn_ckpt['model_state_dict'])
            self.gnn.eval()
            print(f"  GNN加载成功 (Epoch {gnn_ckpt.get('epoch', 'N/A')})")
    
    def preprocess_voxel(self, voxel: np.ndarray) -> torch.Tensor:
        """预处理体素数据"""
        # 确保形状正确
        if voxel.ndim == 3:
            voxel = voxel[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
        elif voxel.ndim == 4:
            voxel = voxel[np.newaxis, ...]  # (1, 1, D, H, W)
        
        # 转换为float32并归一化到[0,1]
        voxel = voxel.astype(np.float32)
        if voxel.max() > 1:
            voxel = voxel / 255.0
        
        return torch.tensor(voxel, dtype=torch.float32).to(self.device)
    
    def build_graph_from_features(self, layer4_features: np.ndarray) -> Data:
        """从Layer4特征构建图"""
        # layer4_features: (32, 6, 6, 6)
        # 重塑为节点特征: (216, 32)
        node_features = layer4_features.transpose(1, 2, 3, 0).reshape(-1, layer4_features.shape[0])
        
        # 构建边
        edge_index, edge_weight = build_knn_edges(node_features, k=self.k)
        
        # 创建图
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32).view(-1, 1)
        )
        
        return graph
    
    def denormalize(self, pred: np.ndarray, from_gnn: bool = False) -> dict:
        """反归一化预测结果并转换回原始尺度
        
        Args:
            pred: 预测值
            from_gnn: 是否来自GNN
        """
        if from_gnn and self.gnn_normalized:
            # GNN 输出是归一化值，需要反归一化
            pred_denorm = pred * self.gnn_label_std + self.gnn_label_mean
            log_k = pred_denorm[0]
            pressure_drop = pred_denorm[1] if len(pred_denorm) > 1 else None
        elif from_gnn:
            # GNN 未归一化，直接使用
            log_k = pred[0]
            pressure_drop = pred[1] if len(pred) > 1 else None
        else:
            # CNN 输出是归一化值，需要反归一化
            pred_denorm = pred * self.label_std + self.label_mean
            log_k = pred_denorm[0]
            pressure_drop = pred_denorm[1] if len(pred_denorm) > 1 else None
        
        # log(渗透率) 转回原始渗透率 (m²)
        permeability = np.exp(log_k)
        
        return {
            'log_permeability': log_k,
            'permeability_m2': permeability,
            'pressure_drop_N_m3': pressure_drop
        }
    
    @torch.no_grad()
    def predict_single(self, voxel: np.ndarray, use_gnn: bool = True) -> dict:
        """
        预测单个样本
        
        Args:
            voxel: 体素数据 (D, H, W) 或 (1, D, H, W)
            use_gnn: 是否使用GNN (需要先加载GNN模型)
            
        Returns:
            预测结果字典
        """
        # 预处理
        voxel_tensor = self.preprocess_voxel(voxel)
        
        if use_gnn and self.gnn is not None:
            # 使用 CNN 提取特征 + GNN 预测
            features = self.cnn.forward_features(voxel_tensor)
            layer4 = features['layer4'].cpu().numpy()[0]  # (32, 6, 6, 6)
            
            # 构建图
            graph = self.build_graph_from_features(layer4).to(self.device)
            
            # GNN 预测
            pred = self.gnn(graph)
            pred = pred.cpu().numpy().flatten()
            
            # GNN 输出是原始值，不需要反归一化
            result = self.denormalize(pred, from_gnn=True)
        else:
            # 仅使用 CNN 预测
            pred = self.cnn(voxel_tensor)
            pred = pred.cpu().numpy().flatten()
            
            # CNN 输出是归一化值，需要反归一化
            result = self.denormalize(pred, from_gnn=False)
        
        result['raw_prediction'] = pred.tolist()
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, voxels: list, use_gnn: bool = True) -> list:
        """
        批量预测
        
        Args:
            voxels: 体素数据列表
            use_gnn: 是否使用GNN
            
        Returns:
            预测结果列表
        """
        results = []
        
        if use_gnn and self.gnn is not None:
            # GNN 批量预测
            graphs = []
            
            for voxel in voxels:
                voxel_tensor = self.preprocess_voxel(voxel)
                features = self.cnn.forward_features(voxel_tensor)
                layer4 = features['layer4'].cpu().numpy()[0]
                graph = self.build_graph_from_features(layer4)
                graphs.append(graph)
            
            # 批处理
            batch = Batch.from_data_list(graphs).to(self.device)
            preds = self.gnn(batch)
            preds = preds.cpu().numpy()
            
            for pred in preds:
                results.append(self.denormalize(pred, from_gnn=True))
        else:
            # CNN 批量预测
            for voxel in voxels:
                results.append(self.predict_single(voxel, use_gnn=False))
        
        return results


def load_voxel_from_file(filepath: str) -> np.ndarray:
    """从文件加载体素数据"""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        # 尝试常见的键名
        for key in ['voxel', 'voxels', 'data', 'arr_0']:
            if key in data:
                return data[key]
        raise ValueError(f"无法从 {filepath} 中找到体素数据")
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


def main():
    args = parse_args()
    
    # 路径处理
    cnn_checkpoint = os.path.join(ROOT_DIR, args.cnn_checkpoint)
    gnn_checkpoint = os.path.join(ROOT_DIR, args.gnn_checkpoint) if not args.use_cnn_only else None
    
    # 创建预测器
    predictor = Predictor(
        cnn_checkpoint=cnn_checkpoint,
        gnn_checkpoint=gnn_checkpoint,
        device=args.device,
        k=args.k
    )
    
    use_gnn = not args.use_cnn_only and predictor.gnn is not None
    print(f"\n预测模式: {'CNN + GNN' if use_gnn else 'CNN only'}")
    
    results = []
    sample_names = []
    
    # ===== 从数据集预测 =====
    if args.dataset and args.sample_ids:
        dataset_dir = os.path.join(ROOT_DIR, args.dataset)
        voxels_path = os.path.join(dataset_dir, 'dataset_voxels.npy')
        
        print(f"\n从数据集加载: {voxels_path}")
        all_voxels = np.load(voxels_path)
        
        for sid in args.sample_ids:
            if sid < len(all_voxels):
                voxel = all_voxels[sid]
                result = predictor.predict_single(voxel, use_gnn=use_gnn)
                results.append(result)
                sample_names.append(f"sample_{sid:04d}")
                print(f"  样本 {sid}: K={result['permeability_m2']:.4e} m², "
                      f"dP={result['pressure_drop_N_m3']:.2f} N/m³")
    
    # ===== 从文件/目录预测 =====
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 单个文件
            print(f"\n预测文件: {input_path}")
            voxel = load_voxel_from_file(str(input_path))
            result = predictor.predict_single(voxel, use_gnn=use_gnn)
            results.append(result)
            sample_names.append(input_path.stem)
            
        elif input_path.is_dir():
            # 目录中的所有 .npy 文件
            npy_files = sorted(input_path.glob('*.npy'))
            print(f"\n预测目录: {input_path} ({len(npy_files)} 个文件)")
            
            for npy_file in npy_files:
                voxel = load_voxel_from_file(str(npy_file))
                result = predictor.predict_single(voxel, use_gnn=use_gnn)
                results.append(result)
                sample_names.append(npy_file.stem)
        else:
            print(f"[错误] 输入路径不存在: {input_path}")
            sys.exit(1)
    
    else:
        print("[错误] 请指定 --input 或 --dataset + --sample_ids")
        sys.exit(1)
    
    # ===== 输出结果 =====
    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"{'样本名':<20} {'渗透率 (m²)':<15} {'log(K)':<10} {'压降 (N/m³)':<12}")
    print("-" * 60)
    
    for name, result in zip(sample_names, results):
        k_str = f"{result['permeability_m2']:.4e}"
        log_k_str = f"{result['log_permeability']:.3f}"
        dp_str = f"{result['pressure_drop_N_m3']:.2f}" if result['pressure_drop_N_m3'] else "N/A"
        print(f"{name:<20} {k_str:<15} {log_k_str:<10} {dp_str:<12}")
    
    # ===== 保存到CSV =====
    if args.output:
        import csv
        
        output_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT_DIR, args.output)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_name', 'permeability_m2', 'log_permeability', 'pressure_drop_N_m3'])
            
            for name, result in zip(sample_names, results):
                writer.writerow([
                    name,
                    result['permeability_m2'],
                    result['log_permeability'],
                    result['pressure_drop_N_m3']
                ])
        
        print(f"\n结果已保存至: {output_path}")
    
    print("\n✅ 预测完成!")


if __name__ == '__main__':
    main()
