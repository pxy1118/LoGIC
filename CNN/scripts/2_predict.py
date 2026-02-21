"""
预测脚本

使用训练好的 CNN 模型预测 TPMS 结构的力学性质

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


def parse_args():
    parser = argparse.ArgumentParser(description='Predict mechanical properties')
    parser.add_argument('--input', type=str, default=None,
                        help='输入体素文件(.npy)或目录')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集目录 (与 --sample_ids 配合使用)')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=None,
                        help='要预测的样本ID列表')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cnn_backbone_best.pth',
                        help='CNN模型检查点')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    return parser.parse_args()


class Predictor:
    """TPMS 力学性质预测器"""
    
    def __init__(self, checkpoint: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 加载CNN模型
        print(f"加载CNN模型: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        
        config = ckpt.get('config', {})
        stats = ckpt.get('stats', {})
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
            num_outputs=self.num_outputs,
            decoder_type=cnn_config.get('decoder_type', 'mlp')
        ).to(self.device)
        
        self.cnn.load_state_dict(ckpt['model_state_dict'])
        self.cnn.eval()
        print(f"  CNN加载成功 (Epoch {ckpt.get('epoch', 'N/A')})")
    
    def preprocess_voxel(self, voxel: np.ndarray) -> torch.Tensor:
        """预处理体素数据"""
        if voxel.ndim == 3:
            voxel = voxel[np.newaxis, np.newaxis, ...]
        elif voxel.ndim == 4:
            voxel = voxel[np.newaxis, ...]
        
        voxel = voxel.astype(np.float32)
        if voxel.max() > 1:
            voxel = voxel / 255.0
        
        return torch.tensor(voxel, dtype=torch.float32).to(self.device)
    
    def denormalize(self, pred: np.ndarray) -> dict:
        """反归一化预测结果"""
        pred_denorm = pred * self.label_std + self.label_mean
        
        result = {'raw_prediction': pred.tolist()}
        
        # 根据输出数量解析结果
        if self.num_outputs == 2:
            result['E'] = pred_denorm[0]
            result['yield'] = pred_denorm[1]
        elif self.num_outputs > 2:
            result['E'] = pred_denorm[0]
            result['yield'] = pred_denorm[1]
            result['stress_strain_curve'] = pred_denorm[2:].tolist()
        else:
            result['value'] = pred_denorm[0]
        
        return result
    
    @torch.no_grad()
    def predict_single(self, voxel: np.ndarray) -> dict:
        """预测单个样本"""
        voxel_tensor = self.preprocess_voxel(voxel)
        pred = self.cnn(voxel_tensor)
        pred = pred.cpu().numpy().flatten()
        return self.denormalize(pred)
    
    @torch.no_grad()
    def predict_batch(self, voxels: list) -> list:
        """批量预测"""
        results = []
        for voxel in voxels:
            results.append(self.predict_single(voxel))
        return results


def load_voxel_from_file(filepath: str) -> np.ndarray:
    """从文件加载体素数据"""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        for key in ['voxel', 'voxels', 'data', 'arr_0']:
            if key in data:
                return data[key]
        raise ValueError(f"无法从 {filepath} 中找到体素数据")
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


def main():
    args = parse_args()
    
    checkpoint = os.path.join(ROOT_DIR, args.checkpoint)
    predictor = Predictor(checkpoint=checkpoint, device=args.device)
    
    print(f"\n预测模式: CNN only")
    
    results = []
    sample_names = []
    
    # 从数据集预测
    if args.dataset and args.sample_ids:
        dataset_dir = os.path.join(ROOT_DIR, args.dataset)
        voxels_path = os.path.join(dataset_dir, 'dataset_voxels.npy')
        
        print(f"\n从数据集加载: {voxels_path}")
        all_voxels = np.load(voxels_path)
        
        for sid in args.sample_ids:
            if sid < len(all_voxels):
                voxel = all_voxels[sid]
                result = predictor.predict_single(voxel)
                results.append(result)
                sample_names.append(f"sample_{sid:04d}")
                print(f"  样本 {sid}: {result}")
    
    # 从文件/目录预测
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            print(f"\n预测文件: {input_path}")
            voxel = load_voxel_from_file(str(input_path))
            result = predictor.predict_single(voxel)
            results.append(result)
            sample_names.append(input_path.stem)
            
        elif input_path.is_dir():
            npy_files = sorted(input_path.glob('*.npy'))
            print(f"\n预测目录: {input_path} ({len(npy_files)} 个文件)")
            
            for npy_file in npy_files:
                voxel = load_voxel_from_file(str(npy_file))
                result = predictor.predict_single(voxel)
                results.append(result)
                sample_names.append(npy_file.stem)
        else:
            print(f"[错误] 输入路径不存在: {input_path}")
            sys.exit(1)
    
    else:
        print("[错误] 请指定 --input 或 --dataset + --sample_ids")
        sys.exit(1)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    
    for name, result in zip(sample_names, results):
        print(f"{name}: {result}")
    
    # 保存到CSV
    if args.output:
        import csv
        
        output_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT_DIR, args.output)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            if results:
                header = ['sample_name'] + list(results[0].keys())
                writer.writerow(header)
                
                for name, result in zip(sample_names, results):
                    row = [name] + list(result.values())
                    writer.writerow(row)
        
        print(f"\n结果已保存至: {output_path}")
    
    print("\n✅ 预测完成!")


if __name__ == '__main__':
    main()
