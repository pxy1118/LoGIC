"""
评估脚本 - 仅使用CNN模型
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone
from data.dataset import VoxelDataset
from utils.stress_strain_analysis import StressStrainAnalyzer
import logging

logging.getLogger("StressStrainAnalysis").setLevel(logging.ERROR)

def print_metric(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<15} | Avg MAE: {mae:.4f} | R²: {r2:.4f}")
    return mae, r2

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--dataset_dir', type=str, default=os.path.join(ROOT_DIR, 'dataset'))
    parser.add_argument('--checkpoint', type=str, default=os.path.join(ROOT_DIR, 'checkpoints', 'cnn_backbone_best.pth'))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'evaluation'))
    return parser.parse_args()

def load_model(checkpoint_path, num_outputs, device):
    """加载CNN模型"""
    print(f"Loading CNN from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cnn_config = ckpt.get('config', {}).get('cnn', {})
    
    # Detect if using improved architecture
    state_dict = ckpt['model_state_dict']
    use_improved = 'encoder.0.0.weight' in state_dict or \
                   'attention_modules.attn_2.mlp.0.weight' in state_dict
    
    if use_improved:
        print("  检测到改进架构 (ImprovedCNN3DBackbone)")
        from models.cnn_backbone import ImprovedCNN3DBackbone
        
        # Decoder configuration
        decoder_config = cnn_config.get('decoder', {})
        decoder_type = decoder_config.get('type', 'lstm')
        decoder_hidden_dim = decoder_config.get('hidden_dim', 256)
        decoder_num_layers = decoder_config.get('num_layers', 2)
        decoder_num_heads = decoder_config.get('num_heads', 4)
        
        # Architecture type
        architecture_type = cnn_config.get('architecture_type', 'resnet')
        
        # Attention configuration
        attention_config = cnn_config.get('attention', {})
        use_attention = attention_config.get('enabled', False)
        attention_type = attention_config.get('type', 'channel')
        attention_positions = attention_config.get('positions', None)
        
        # Multi-scale fusion configuration
        fusion_config = cnn_config.get('multi_scale_fusion', {})
        use_multi_scale = fusion_config.get('enabled', False)
        fusion_layers = fusion_config.get('fusion_layers', None)
        
        # Residual configuration
        residual_config = cnn_config.get('residual', {})
        use_bottleneck = residual_config.get('use_bottleneck', False)
        
        print(f"  - 架构类型: {architecture_type}")
        print(f"  - 解码器: {decoder_type} (hidden_dim={decoder_hidden_dim}, layers={decoder_num_layers})")
        if use_attention:
            print(f"  - 注意力: {attention_type} at {attention_positions}")
        if use_multi_scale:
            print(f"  - 多尺度融合: {fusion_layers}")
        
        cnn = ImprovedCNN3DBackbone(
            in_channels=1,
            filters=tuple(cnn_config.get('filters', [8, 16, 32])),
            fc_dim=cnn_config.get('fc_dim', 128),
            adaptive_pool_size=cnn_config.get('adaptive_pool_size', 6),
            dropout=cnn_config.get('dropout', 0.0),
            num_outputs=num_outputs,
            activation=cnn_config.get('activation', 'elu'),
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
        ).to(device)
    else:
        print("  检测到基础架构 (CNN3DBackbone)")
        from models.cnn_backbone import CNN3DBackbone
        
        decoder_type = cnn_config.get('decoder_type', 'mlp')
        if 'decoder.rnn.weight_ih_l0' in state_dict:
            decoder_type = 'lstm'
            print(f"  - 解码器: {decoder_type}")
        
        cnn = CNN3DBackbone(
            in_channels=1,
            filters=tuple(cnn_config.get('filters', [8, 16, 32])),
            fc_dim=cnn_config.get('fc_dim', 128),
            adaptive_pool_size=cnn_config.get('adaptive_pool_size', 6),
            num_outputs=num_outputs,
            decoder_type=decoder_type
        ).to(device)
    
    cnn.load_state_dict(state_dict)
    cnn.eval()
    return cnn

def plot_parity(y_true, y_pred, title, unit, filename):
    """绘制parity图"""
    plt.figure(figsize=(7, 6))
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=15, edgecolors='k', linewidth=0.3)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', alpha=0.8)
    
    plt.xlabel(f'True {unit}')
    plt.ylabel(f'Predicted {unit}')
    plt.title(f'{title}\nMAE={mae:.3f}, R²={r2:.3f}, RMSE={rmse:.3f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


class CurveAnalyzer:
    """分析应力应变曲线"""
    def __init__(self, num_points):
        self.num_points = num_points
        self.strain_step = None
        
    def calibrate_strain_step(self, curves, true_Es):
        """估算应变步长"""
        # steps = []
        # for i in range(len(curves)):
        #     if pd.isna(true_Es[i]) or true_Es[i] == 0:
        #         continue
        #     stress_val = curves[i][1]
        #     estimated_step = stress_val / true_Es[i]
        #     if estimated_step > 0 and estimated_step < 0.1:
        #         steps.append(estimated_step)
        
        # if steps:
        #     self.strain_step = np.median(steps)
        #     print(f"[CurveAnalyzer] 校准应变步长: {self.strain_step:.6f}")
        # else:
        print("[CurveAnalyzer] 使用默认应变步长 0.0025")
        self.strain_step = 0.0025


    def calculate_properties(self, curve):
        """从曲线计算E和屈服强度 - 修正版：正确处理应力-应变数据"""
        if self.strain_step is None:
            return 0.0, 0.0
            
        # curve 是应力值 (MPa)，需要构建对应的应变数组
        strain = np.arange(len(curve)) * self.strain_step
        stress = curve
        
        # 关键修正：StressStrainAnalyzer 需要位移(mm)和力(N)作为输入
        # 我们需要反推这些值
        # 使用与 4_batch_postprocess.py 一致的参数
        gauge_length = 24.0  # mm (标距长度)
        cross_sectional_area = 576.0  # mm² (横截面积)
        
        # 从应变反推位移: displacement = strain * gauge_length
        displacement = strain * gauge_length
        
        # 从应力反推力: force = stress * cross_sectional_area
        force = stress * cross_sectional_area
        
        analyzer = StressStrainAnalyzer(
            displacement=displacement,
            force=force,
            cross_sectional_area=cross_sectional_area,
            gauge_length=gauge_length
        )
        
        max_strain = strain[-1] if len(strain) > 0 else 0.0
        strain_threshold = min(0.015, max_strain * 0.5)
        
        E = analyzer.calculate_elastic_modulus(strain_threshold=strain_threshold, min_points=2)
        
        if E and E > 0:
            yield_strength = analyzer.calculate_yield_strength_02(E)
            if yield_strength is None:
                # 回退：使用最大应力的70%作为屈服强度估计
                yield_strength = np.max(stress) * 0.7 if len(stress) > 0 else 0.0
        else:
            E = 0.0
            yield_strength = 0.0
        
        if E is None: E = 0.0
        if yield_strength is None:
            yield_strength = np.max(stress) if len(stress) > 0 else 0.0
            
        return E, yield_strength

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载真实标签
    e_dict, yield_dict = {}, {}
    try:
        e_df = pd.read_csv(os.path.join(args.dataset_dir, 'E.csv'))
        id_col = 'sample' if 'sample' in e_df.columns else e_df.columns[0]
        val_col = 'E' if 'E' in e_df.columns else e_df.columns[1]
        for _, row in e_df.iterrows():
            try:
                val = row[id_col]
                if isinstance(val, str):
                    sid = int(val.split('_')[-1]) if '_' in val else int(float(val))
                else:
                    sid = int(val)
                e_dict[sid] = float(row[val_col])
            except:
                pass
            
        y_df = pd.read_csv(os.path.join(args.dataset_dir, 'yield.csv'))
        id_col = 'sample' if 'sample' in y_df.columns else y_df.columns[0]
        val_col = 'yield' if 'yield' in y_df.columns else y_df.columns[1]
        for _, row in y_df.iterrows():
            try:
                val = row[id_col]
                if isinstance(val, str):
                    sid = int(val.split('_')[-1]) if '_' in val else int(float(val))
                else:
                    sid = int(val)
                yield_dict[sid] = float(row[val_col])
            except:
                pass
        print(f"加载了 {len(e_dict)} 个E值和 {len(yield_dict)} 个屈服强度值")
    except Exception as e:
        print(f"警告: 无法加载CSV: {e}")

    # 检测目标类型
    print(f"从 {args.checkpoint} 读取配置...")
    curve_points = None
    try:
        temp_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        target_names = temp_ckpt.get('config', {}).get('data', {}).get('prediction_targets', ['E', 'yield'])
        curve_points = temp_ckpt.get('config', {}).get('data', {}).get('curve_points', None)
        if curve_points:
            print(f"检测到曲线点数配置: {curve_points}")
    except Exception as e:
        print(f"警告: 无法读取配置: {e}")
        target_names = ['E', 'yield']
    print(f"检测到目标: {target_names}")

    # 加载数据集
    print(f"从 {args.dataset_dir} 加载数据集")
    dataset = VoxelDataset(
        dataset_dir=args.dataset_dir,
        indices=None,
        transform=None,
        normalize_label=True,
        target_names=target_names,
        curve_points=curve_points
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    stats = dataset.get_statistics()
    num_outputs = stats['num_outputs']
    print(f"数据集统计: {len(dataset)} 个样本")
    print(f"归一化 - Mean: {dataset.label_mean}, Std: {dataset.label_std}")

    # 加载模型
    cnn = load_model(args.checkpoint, num_outputs, args.device)

    # 评估
    results = {
        'sample_id': [],
        'y_true': [],
        'y_pred': []
    }

    print("开始评估...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            voxels = batch['voxel'].to(args.device)
            labels = batch['label'].cpu().numpy()
            sample_ids = batch['sample_id'].cpu().numpy()
            
            pred_norm = cnn(voxels).cpu().numpy()
            
            true_denorm = dataset.denormalize_label(labels)
            pred_denorm = dataset.denormalize_label(pred_norm)

            for i in range(len(sample_ids)):
                results['sample_id'].append(sample_ids[i])
                results['y_true'].append(true_denorm[i])
                results['y_pred'].append(pred_denorm[i])

    # 分析结果
    y_true = np.array(results['y_true'])
    y_pred = np.array(results['y_pred'])
    
    is_multitask = ('E' in target_names and 'yield' in target_names and 'curves' in target_names)
    curve_start_idx = 2 if is_multitask else 0
    num_curve_points = num_outputs - curve_start_idx
    has_curves = num_curve_points > 5

    if has_curves:
        # 曲线分析 - 使用CSV中的真实值而不是从曲线计算
        analyzer = CurveAnalyzer(num_points=num_curve_points)
        
        # 仅用于校准应变步长（用于从预测曲线计算物理量）
        calib_Es = [e_dict.get(int(sid), np.nan) for sid in results['sample_id']]
        analyzer.calibrate_strain_step(y_true[:, curve_start_idx:], calib_Es)
        
        props = {
            'true_E': [], 'true_Yield': [],
            'cnn_E': [], 'cnn_Yield': [],
            'cnn_E_pred': [], 'cnn_Yield_pred': []
        }
        
        for i in range(len(y_true)):
            sid = int(results['sample_id'][i])
            
            # 直接使用CSV中的真实值
            true_E = e_dict.get(sid, 0.0)
            true_Yield = yield_dict.get(sid, 0.0)
            props['true_E'].append(true_E)
            props['true_Yield'].append(true_Yield)
            
            # 从预测曲线计算物理量
            e_cnn, y_cnn = analyzer.calculate_properties(y_pred[i, curve_start_idx:])
            props['cnn_E'].append(e_cnn)
            props['cnn_Yield'].append(y_cnn)
            
            if is_multitask:
                props['cnn_E_pred'].append(y_pred[i, 0])
                props['cnn_Yield_pred'].append(y_pred[i, 1])
        
        # 保存结果
        prop_df = pd.DataFrame({
            'sample_id': results['sample_id'],
            'True_E': props['true_E'], 'True_Yield': props['true_Yield'],
            'CNN_E_Derived': props['cnn_E'], 'CNN_Yield_Derived': props['cnn_Yield']
        })
        if is_multitask:
            prop_df['CNN_E_Pred'] = props['cnn_E_pred']
            prop_df['CNN_Yield_Pred'] = props['cnn_Yield_pred']
            
        prop_csv = os.path.join(args.output_dir, 'evaluation_properties.csv')
        prop_df.to_csv(prop_csv, index=False)
        print(f"保存属性结果到 {prop_csv}")
        
        # 报告
        print("\n" + "="*50)
        print("评估报告")
        print("="*50)

        print("\n--- 力学性质 ---")
        
        # 过滤掉零值样本（如果CSV中没有对应数据）
        valid_mask = (np.array(props['true_E']) > 0)
        if np.sum(valid_mask) > 0:
            te = np.array(props['true_E'])[valid_mask]
            print(f"\n弹性模量 (E) - 有效样本: {np.sum(valid_mask)}/{len(props['true_E'])}")
            print_metric("CNN Derived", te, np.array(props['cnn_E'])[valid_mask])
            if is_multitask:
                print_metric("CNN Direct", te, np.array(props['cnn_E_pred'])[valid_mask])
        else:
            print("\n弹性模量 (E): 无有效样本")

        valid_mask_y = (np.array(props['true_Yield']) > 0)
        if np.sum(valid_mask_y) > 0:
            ty = np.array(props['true_Yield'])[valid_mask_y]
            print(f"\n屈服强度 - 有效样本: {np.sum(valid_mask_y)}/{len(props['true_Yield'])}")
            print_metric("CNN Derived", ty, np.array(props['cnn_Yield'])[valid_mask_y])
            if is_multitask:
                print_metric("CNN Direct", ty, np.array(props['cnn_Yield_pred'])[valid_mask_y])
        else:
            print("\n屈服强度: 无有效样本")

        print("\n--- 应力应变曲线 ---")
        print_metric("CNN", y_true[:, curve_start_idx:], y_pred[:, curve_start_idx:])
        
        # 绘制样本曲线
        print("生成曲线图...")
        indices = np.linspace(0, len(y_true)-1, 5, dtype=int)
        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(indices):
            plt.subplot(1, 5, i+1)
            plt.plot(y_true[idx, curve_start_idx:], label='True', color='black')
            plt.plot(y_pred[idx, curve_start_idx:], label='CNN', linestyle='--')
            plt.title(f'Sample {results["sample_id"][idx]}')
            if i == 0: plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sample_curves.png'))
        
    else:
        # 标量任务
        print("\n" + "="*50)
        print("评估报告 (标量)")
        print("="*50)
        
        res_dict = {'sample_id': results['sample_id']}
        for i in range(num_outputs):
            col_name = target_names[i] if i < len(target_names) else f'Out_{i}'
            res_dict[f'true_{col_name}'] = y_true[:, i]
            res_dict[f'pred_{col_name}'] = y_pred[:, i]
        
        df_res = pd.DataFrame(res_dict)
        csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        df_res.to_csv(csv_path, index=False)
        print(f"\n保存结果到 {csv_path}")
        
        for i in range(num_outputs):
            col_name = target_names[i] if i < len(target_names) else f'Out_{i}'
            print(f"\n--- {col_name} ---")
            print_metric("CNN", y_true[:, i], y_pred[:, i])
    
    print("完成.")

if __name__ == '__main__':
    main()
