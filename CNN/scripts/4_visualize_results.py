"""
可视化评估结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 配置matplotlib支持中文
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 12})

def plot_parity(df, true_col, pred_col, title, unit, ax=None):
    """绘制parity图"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # 过滤掉零值和无效值
    valid_mask = (df[true_col] > 0) & (df[pred_col].notna())
    valid_df = df[valid_mask]
    
    if len(valid_df) == 0:
        ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return

    y_true = valid_df[true_col].values
    y_pred = valid_df[pred_col].values
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
    
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.1
    ax.plot([vmin-margin, vmax+margin], [vmin-margin, vmax+margin], 'r--', alpha=0.8, linewidth=2)
    
    # 添加统计信息
    stats_text = f"MAE: {mae:.2f} {unit}\nR²: {r2:.3f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.1f}%\nN: {len(valid_df)}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"True {title} ({unit})", fontsize=12)
    ax.set_ylabel(f"Predicted {title} ({unit})", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return mae, r2

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(root_dir, 'evaluation')
    prop_csv = os.path.join(eval_dir, 'evaluation_properties.csv')
    
    if not os.path.exists(prop_csv):
        print(f"文件未找到: {prop_csv}. 请先运行 3_evaluate_all.py")
        return

    df = pd.read_csv(prop_csv)
    print(f"加载了 {len(df)} 个样本的属性")
    print(f"列名: {df.columns.tolist()}")
    
    # 数据统计
    print("\n数据统计:")
    print(f"  True_E 有效样本: {(df['True_E'] > 0).sum()}/{len(df)}")
    print(f"  True_Yield 有效样本: {(df['True_Yield'] > 0).sum()}/{len(df)}")
    
    # 解析列名
    col_map = {
        'True_E': ['True_E'],
        'True_Yield': ['True_Yield'],
        'CNN_E': ['CNN_E_Derived', 'CNN_E', 'CNN_E_Pred'],
        'CNN_Yield': ['CNN_Yield_Derived', 'CNN_Yield', 'CNN_Yield_Pred']
    }
    
    cols = {}
    for key, candidates in col_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        cols[key] = found
        
    print(f"\n映射的列: {cols}")

    # 1. 力学性质的Parity图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics = {}
    
    # 弹性模量
    if cols['True_E'] and cols['CNN_E']:
        print("\n绘制弹性模量 parity 图...")
        result = plot_parity(df, cols['True_E'], cols['CNN_E'], "Elastic Modulus", "MPa", ax=axes[0])
        if result:
            metrics['E_MAE'], metrics['E_R2'] = result
    else:
        print("警告: 缺少弹性模量列")
    
    # 屈服强度
    if cols['True_Yield'] and cols['CNN_Yield']:
        print("绘制屈服强度 parity 图...")
        result = plot_parity(df, cols['True_Yield'], cols['CNN_Yield'], "Yield Strength", "MPa", ax=axes[1])
        if result:
            metrics['Yield_MAE'], metrics['Yield_R2'] = result
    else:
        print("警告: 缺少屈服强度列")
        
    plt.tight_layout()
    parity_path = os.path.join(eval_dir, 'physics_parity_plots.png')
    plt.savefig(parity_path, dpi=150, bbox_inches='tight')
    print(f"\n保存 parity 图到: {parity_path}")
    plt.close()
    
    # 2. 误差分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if cols['True_E'] and cols['CNN_E']:
        valid_mask = (df[cols['True_E']] > 0) & (df[cols['CNN_E']].notna())
        err_E = (df[valid_mask][cols['CNN_E']] - df[valid_mask][cols['True_E']]).values
        
        axes[0].hist(err_E, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f"Elastic Modulus Error Distribution\nMean: {err_E.mean():.2f}, Std: {err_E.std():.2f}")
        axes[0].set_xlabel("Error (Predicted - True) [MPa]")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)
        
    if cols['True_Yield'] and cols['CNN_Yield']:
        valid_mask = (df[cols['True_Yield']] > 0) & (df[cols['CNN_Yield']].notna())
        err_Yield = (df[valid_mask][cols['CNN_Yield']] - df[valid_mask][cols['True_Yield']]).values
        
        axes[1].hist(err_Yield, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title(f"Yield Strength Error Distribution\nMean: {err_Yield.mean():.2f}, Std: {err_Yield.std():.2f}")
        axes[1].set_xlabel("Error (Predicted - True) [MPa]")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_path = os.path.join(eval_dir, 'error_distributions.png')
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    print(f"保存误差分布图到: {error_path}")
    plt.close()
    
    # 3. 生成评估报告
    print("\n" + "="*60)
    print("评估指标总结")
    print("="*60)
    if 'E_MAE' in metrics:
        print(f"弹性模量 (E):")
        print(f"  MAE:  {metrics['E_MAE']:.2f} MPa")
        print(f"  R²:   {metrics['E_R2']:.4f}")
    if 'Yield_MAE' in metrics:
        print(f"\n屈服强度 (Yield):")
        print(f"  MAE:  {metrics['Yield_MAE']:.2f} MPa")
        print(f"  R²:   {metrics['Yield_R2']:.4f}")
    print("="*60)
    
    print("\n可视化完成!")
    print(f"结果保存在: {eval_dir}")

if __name__ == "__main__":
    main()
