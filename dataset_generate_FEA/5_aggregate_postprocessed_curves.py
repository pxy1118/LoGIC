#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合所有样本的应力-应变曲线为一个数据集文件 (.npz)
功能:
  1. 遍历 dataset_fea 下所有 output/stress_strain_curve.csv
  2. 读取应力-应变数据
  3. 插值到统一的 strain 网格 (例如 0~0.2, 1000点)
  4. 按样本ID排序并保存为 compressed npz 文件
  5. 记录有效的样本索引, 以便与 design parameters 对齐

输出文件包含:
  - strain: (N_points,)  应变网格
  - stress: (N_samples, N_points) 应力矩阵
  - sample_ids: (N_samples,)  样本字符串ID (e.g. "0001")
  - sample_indices: (N_samples,) 样本整数索引 (e.g. 1)

使用方法:
  python 5_aggregate_postprocessed_curves.py
"""

import os
import glob
import numpy as np
import pandas as pd
import argparse
import re
from tqdm import tqdm

def parse_sample_id(path):
    # 尝试从路径中提取数字ID
    # 假设路径结构 .../0123/output/...
    parts = path.replace('\\', '/').split('/')
    if 'output' in parts:
        try:
            # 取 output 上一级的文件夹名
            idx = parts.index('output')
            name = parts[idx-1]
            if name.isdigit():
                return int(name), name
        except:
            pass
    return None, None

def aggregate_curves(root_dir, output_file, num_points, max_strain_limit):
    print(f"Scanning {root_dir}...")
    
    # 查找所有 csv
    search_pattern = os.path.join(root_dir, "**", "output", "stress_strain_curve.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} potential curve files.")
    
    # 收集有效文件
    sample_data = []
    
    all_max_strain = 0.0
    
    for f in tqdm(files, desc="Reading Files"):
        sid, sname = parse_sample_id(f)
        if sid is None:
            continue
            
        try:
            df = pd.read_csv(f)
            if 'Strain' not in df.columns or 'Stress(MPa)' not in df.columns:
                continue
            
            # 简单清洗
            df = df.dropna()
            if len(df) < 5:
                continue
                
            # 获取当前最大应变用于统计
            local_max = df['Strain'].max()
            if local_max > all_max_strain:
                all_max_strain = local_max
                
            sample_data.append({
                'id': sid,
                'name': sname,
                'path': f,
                'df': df
            })
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")

    # 按 ID 排序
    sample_data.sort(key=lambda x: x['id'])
    
    if not sample_data:
        print("No valid data found.")
        return

    # 确定插值网格
    # 如果用户指定了 max_strain_limit > 0，则使用该值
    # 否则使用数据中的最大值 (取一个合理的上限，比如 95% 分位数，或者直接最大值)
    if max_strain_limit <= 0:
        target_max = all_max_strain
        print(f"Auto-detected max strain: {target_max:.4f}")
    else:
        target_max = max_strain_limit
        print(f"Using defined max strain: {target_max:.4f}")

    interp_strain = np.linspace(0, target_max, num_points)
    
    stress_matrix = []
    sample_ids = []
    sample_indices = []
    
    for item in tqdm(sample_data, desc="Interpolating"):
        df = item['df']
        # 确保 Strain 单调递增 (去重)
        df = df.drop_duplicates(subset=['Strain']).sort_values('Strain')
        
        s_raw = df['Strain'].values
        sigma_raw = df['Stress(MPa)'].values
        
        # 插值
        # left=0: 起始点前补0
        # right=0: 超过实验最大应变后补0 (视为失效) 或者补 NaN
        # 这里补 0，表示无法承受载荷
        # 注意：如果曲线在这里断掉，说明失效了
        interp_sigma = np.interp(interp_strain, s_raw, sigma_raw, left=0.0, right=0.0)
        
        stress_matrix.append(interp_sigma)
        sample_ids.append(item['name'])
        sample_indices.append(item['id'])
        
    stress_matrix = np.array(stress_matrix, dtype=np.float32)
    sample_ids = np.array(sample_ids)
    sample_indices = np.array(sample_indices, dtype=np.int32)
    
    print(f"Aggregation complete.")
    print(f"Samples: {len(sample_ids)}")
    print(f"Curve shape: {stress_matrix.shape}")
    
    # Check for missing IDs
    if len(sample_indices) > 0:
        max_id = sample_indices.max()
        missing = []
        existing_set = set(sample_indices)
        for i in range(max_id + 1):
            if i not in existing_set:
                missing.append(i)
        if missing:
            print(f"Warning: {len(missing)} samples are missing in the range [0, {max_id}].")
            if len(missing) < 20:
                print(f"Missing IDs: {missing}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez_compressed(output_file, 
                        strain=interp_strain, 
                        stress=stress_matrix, 
                        sample_ids=sample_ids,
                        sample_indices=sample_indices,
                        info="Interpolated Stress-Strain Curves")
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate FEA Stress-Strain Curves")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "dataset_fea"), help="Dataset root directory")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset_fea", "dataset_curves_aggregated.npz"), help="Output .npz file")
    parser.add_argument("--points", type=int, default=41, help="Number of interpolation points")
    parser.add_argument("--max_strain", type=float, default=0.1, help="Max strain for interpolation grid (0 for auto)")
    
    args = parser.parse_args()
    aggregate_curves(args.root, args.out, args.points, args.max_strain)
