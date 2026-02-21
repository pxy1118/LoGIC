import os
import shutil
import pandas as pd
import numpy as np

# 配置路径
DATASET_DIR = "dataset_ml"
CSV_FILENAME = "simulation_results.csv"

def main():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, DATASET_DIR)
    csv_path = os.path.join(dataset_path, CSV_FILENAME)

    if not os.path.exists(csv_path):
        print(f"错误: 找不到结果文件 {csv_path}")
        return

    print(f"正在读取仿真结果: {csv_path}")
    
    # 1. 读取成功的样本ID
    # 假设 CSV 格式为: sample_id, 渗透率, ...
    try:
        df = pd.read_csv(csv_path)
        # 确保 sample_id 是字符串格式并补零 (例如 '0' -> '0000')
        valid_ids = set(df['sample_id'].astype(str).str.zfill(4).unique())
        print(f"找到 {len(valid_ids)} 个成功仿真的样本。")
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 2. 扫描数据集目录中的所有样本文件夹
    if not os.path.exists(dataset_path):
        print(f"数据集目录不存在: {dataset_path}")
        return

    all_items = os.listdir(dataset_path)
    # 筛选出 4 位数字命名的文件夹
    sample_dirs = [d for d in all_items if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(dataset_path, d))]
    
    deleted_count = 0
    remaining_samples = []

    print(f"扫描到 {len(sample_dirs)} 个样本文件夹，开始清理...")

    for s_id in sample_dirs:
        dir_full_path = os.path.join(dataset_path, s_id)
        
        if s_id not in valid_ids:
            # 如果不在成功列表中，则删除
            print(f"  [删除] 样本 {s_id} (未在仿真结果中)")
            try:
                shutil.rmtree(dir_full_path)
                deleted_count += 1
            except Exception as e:
                print(f"    无法删除 {s_id}: {e}")
        else:
            remaining_samples.append(s_id)

    print(f"\n清理完成。删除: {deleted_count} 个, 剩余: {len(remaining_samples)} 个。")

    # 3. 重新聚合 .npy 文件 (保持 dataset_*.npy 与现有文件夹一致)
    if remaining_samples:
        print("\n正在重新聚合剩余样本的 .npy 数据...")
        remaining_samples.sort() # 确保顺序

        all_weights = []
        all_densities = []
        all_rotations = []
        all_voxels = []

        for s_id in remaining_samples:
            sample_dir = os.path.join(dataset_path, s_id)
            try:
                # 读取单个样本的数据
                w = np.load(os.path.join(sample_dir, "input_weight_3x3x3.npy"))
                d = np.load(os.path.join(sample_dir, "input_density_3x3x3.npy"))
                v = np.load(os.path.join(sample_dir, "output_voxel.npy"))
                
                # 旋转数据可能不存在（旧数据），做兼容处理
                rot_path = os.path.join(sample_dir, "input_rotation_3x3x3.npy")
                if os.path.exists(rot_path):
                    r = np.load(rot_path)
                else:
                    # 如果缺失，使用全0填充 (3,3,3,3)
                    r = np.zeros((3, 3, 3, 3), dtype=np.float32)

                all_weights.append(w)
                all_densities.append(d)
                all_rotations.append(r)
                all_voxels.append(v)

            except Exception as e:
                print(f"  [警告] 读取样本 {s_id} 数据失败: {e}")

        # 保存聚合文件
        if all_weights:
            dataset_voxels = np.array(all_voxels, dtype=bool)
            dataset_weights = np.array(all_weights, dtype=np.float32)
            dataset_densities = np.array(all_densities, dtype=np.float32)
            dataset_rotations = np.array(all_rotations, dtype=np.float32)

            np.save(os.path.join(dataset_path, "dataset_voxels.npy"), dataset_voxels)
            np.save(os.path.join(dataset_path, "dataset_params_weight.npy"), dataset_weights)
            np.save(os.path.join(dataset_path, "dataset_params_density.npy"), dataset_densities)
            np.save(os.path.join(dataset_path, "dataset_params_rotation.npy"), dataset_rotations)

            print("聚合数据已更新:")
            print(f"  Voxels: {dataset_voxels.shape}")
            print(f"  Weights: {dataset_weights.shape}")
            print(f"  Densities: {dataset_densities.shape}")
            print(f"  Rotations: {dataset_rotations.shape}")
    else:
        print("没有剩余样本，跳过聚合。")

if __name__ == "__main__":
    main()
