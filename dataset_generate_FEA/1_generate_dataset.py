import numpy as np
import os
import time
import sys

# 尝试导入同目录下的 tpms_hybrid 模块
import matplotlib.pyplot as plt
try:
    from tpms_hybrid_v2 import (
        Gyroid, Schoen_IWP,
        generate_weight_matrix, generate_density_field_from_3x3x3, generate_rotation_field_from_3x3x3,
        create_hybrid_tpms_solid, expanded_ranges,
        visualize_weight_cube_3d, get_tpms_color_pairs, export_stl
    )
except ImportError:
    # 如果运行路径不对，尝试添加当前路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from tpms_hybrid_v2 import (
        Gyroid, Schoen_IWP,
        generate_weight_matrix, generate_density_field_from_3x3x3, generate_rotation_field_from_3x3x3,
        create_hybrid_tpms_solid, expanded_ranges,
        visualize_weight_cube_3d, get_tpms_color_pairs, export_stl
    )

# ================= 配置区域 =================
DATASET_SIZE = 300           # 生成样本数量
RESOLUTION = 120            # 分辨率设为 120
OUTPUT_DIR = "dataset_fea"   # 输出文件夹
TPMS_FUNCS = [Gyroid, Schoen_IWP] # 指定使用的 TPMS 类型
REPLICATE = (3, 3, 3)       # 周期性重复次数

# 坐标范围 (保持与原脚本一致)
BASE_RANGE = (-1.5, 1.5)

def generate_random_inputs():
    """
    随机生成 3x3x3 的控制参数
    """
    # 1. 随机权重网格 (3, 3, 3, 2)
    # 对应 [Gyroid, Schoen_IWP]，每个位置严格二选一(1.0/0.0)
    # 生成 0 或 1 的随机索引
    choices = np.random.randint(0, 2, size=(3, 3, 3))
    
    # 转换为 One-hot 编码: 0->[1,0], 1->[0,1]
    weight_grid = np.eye(2, dtype=np.float32)[choices]
    
#     weight_grid = np.array([
#     [
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#     ],
#     [
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#     ],
#     [
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#     ],
# ], dtype=np.float32)

    # 2. 随机密度网格 (3, 3, 3)
    # 离散值: [0.3, 0.4, 0.5, 0.6]
    # 动态期望: 在 0.3 到 0.5 之间随机选择一个目标期望
    density_choices = [0.25, 0.3, 0.4, 0.5]
    
    # 混合两个基准分布来实现动态期望
    # Dist_A (偏低): [0.8, 0.1, 0.05, 0.05]
    # Dist_B (偏高): [0.05, 0.15, 0.4, 0.4]
    alpha = np.random.rand() # 0~1 随机系数
    probs = (1 - alpha) * np.array([0.8, 0.1, 0.05, 0.05]) + alpha * np.array([0.05, 0.15, 0.4, 0.4])
    probs /= probs.sum() # Normalize
    
    density_grid = np.random.choice(density_choices, size=(3, 3, 3), p=probs).astype(np.float32)

    # 3. 随机旋转网格 (3, 3, 3, 3)
    # 要求：相邻网格的旋转角度差值不超过 15 度，范围 0-180
    # 采用随机平面场 (Linear Gradient) + 截断的方式，保证梯度约束
    rotation_grid = np.zeros((3, 3, 3, 3), dtype=np.float32)
    
    # 预计算坐标网格 (3, 3, 3)
    xx, yy, zz = np.indices((3, 3, 3))
    
    for axis in range(3): # 对Rx, Ry, Rz三个分量分别生成
        base_val = np.random.uniform(-180, 180)
        # 随机生成每个方向的梯度，范围 [-15, 15]
        # 这样任意相邻网格的差值最大为 15
        d_ang = np.random.choice(np.linspace(-15, 15, num=7), size=3)
        
        # 计算线性场: V = Base + x*dx + y*dy + z*dz
        vals = base_val + xx * d_ang[0] + yy * d_ang[1] + zz * d_ang[2]
        
 
        
        rotation_grid[..., axis] = vals
    # rotation_grid = np.array([
    #     [
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #     ],
    #     [
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #     ],
    #     [
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #     ],
    # ], dtype=np.float32)
    
    return weight_grid, density_grid, rotation_grid

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    print(f"开始生成数据集，共 {DATASET_SIZE} 个样本...")
    print(f"TPMS 类型: {[f.__name__ for f in TPMS_FUNCS]}")
    print(f"分辨率: {RESOLUTION}^3")

    total_start = time.time()

    # 用于聚合所有样本的列表
    all_weights = []
    all_densities = []
    all_rotations = []
    all_voxels = []

    for i in range(DATASET_SIZE):
        sample_id = f"{i:04d}"
        t_start = time.time()
        
        # 0. 检查已存在跳过
        sample_dir = os.path.join(OUTPUT_DIR, sample_id)
        stl_path = os.path.join(sample_dir, "model_solid.stl")
        
        # 如果模型文件已存在，则认为已完成，跳过
        if os.path.exists(stl_path):
            print(f"[跳过] 样本 {sample_id} 已存在。")
            
            # 为了保证最终的聚合数组长度正确，这里需要尝试读取现有的npy文件
            # 否则 all_weights 等列表会缺失数据，导致最后聚合 shape 对不上
            try:
                w_exist = np.load(os.path.join(sample_dir, "input_weight_3x3x3.npy"))
                d_exist = np.load(os.path.join(sample_dir, "input_density_3x3x3.npy"))
                
                # 兼容旧数据：如果有旋转文件则读取，否则全0默认
                rot_path = os.path.join(sample_dir, "input_rotation_3x3x3.npy")
                if os.path.exists(rot_path):
                    r_exist = np.load(rot_path)
                else:
                    r_exist = np.zeros((3, 3, 3, 3), dtype=np.float32)

                v_exist = np.load(os.path.join(sample_dir, "output_voxel.npy"))
                all_weights.append(w_exist)
                all_densities.append(d_exist)
                all_rotations.append(r_exist)
                all_voxels.append(v_exist)
            except Exception as e_load:
                print(f"  [警告] 读取已存在样本数据失败: {e_load}，可能导致通过聚合数据不完整")
                # 如果没法读取，为了安全起见可能需要重新生成，或者插入占位符
                # 这里我们假设文件完整性没问题，如果出错就重新生成这个样本
                print(f"  --> 重新生成样本 {sample_id}")
            else:
                # 成功读取，直接进入下一轮循环
                continue

        # 1. 生成随机输入参数
        weight_grid_3x3, density_grid_3x3, rotation_grid_3x3 = generate_random_inputs() 
        # 计算扩展后的坐标范围
        x_range, y_range, z_range = expanded_ranges(BASE_RANGE, BASE_RANGE, BASE_RANGE, REPLICATE)

        # 2. 从 3x3x3 插值到全分辨率场
        # 生成密度场
        density_field = generate_density_field_from_3x3x3(
            RESOLUTION, x_range, y_range, z_range, 
            density_grid_3x3, smooth_sigma=None
        )
        
        # 生成权重场
        weight_volume = generate_weight_matrix(
            RESOLUTION, x_range, y_range, z_range, 
            weight_grid_3x3, smooth_sigma=0.5
        )

        # 生成旋转场
        rotation_field = generate_rotation_field_from_3x3x3(
            RESOLUTION, x_range, y_range, z_range, 
            rotation_grid_3x3, smooth_sigma=None
        )
        
        # 3. 生成实体结构 
        
        # 注意：create_hybrid_tpms_solid 需要 return_mask=True
        try:
            solid_mesh, porosity, _, solid_mask = create_hybrid_tpms_solid(
                TPMS_FUNCS, x_range, y_range, z_range,
                weight_volume=weight_volume,
                density_field=density_field,
                rotation_field=rotation_field,
                resolution=RESOLUTION,
                solid_threshold=0.0, # 使用 density_field 控制，这里设个默认值
                return_mask=True,     # 关键：获取体素数据
                verbose=False,        # 关闭详细日志
                remove_small=True,    # 保持物理连通性
                smooth=False          # 关闭平滑以加速，ML通常用原始体素
            )
        except Exception as e:
            print(f"样本 {sample_id} 生成失败: {e}")
            continue

        # 4. 保存数据
        sample_dir = os.path.join(OUTPUT_DIR, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # 保存单样本数据
        np.save(os.path.join(sample_dir, "input_weight_3x3x3.npy"), weight_grid_3x3)
        np.save(os.path.join(sample_dir, "input_density_3x3x3.npy"), density_grid_3x3)
        np.save(os.path.join(sample_dir, "input_rotation_3x3x3.npy"), rotation_grid_3x3)
        
        # 处理模型：缩放至 24mm 并对齐原点
        if solid_mesh is not None and solid_mesh.n_points > 0:
            # 1. 缩放
            b = solid_mesh.bounds
            max_dim = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
            if max_dim > 0:
                scale_factor = 24.0 / max_dim
                solid_mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
            
            # 2. 移动到原点
            b = solid_mesh.bounds
            offset = (-b[0], -b[2], -b[4])
            solid_mesh.translate(offset, inplace=True)

        # 保存 STL 模型
        stl_path = os.path.join(sample_dir, "model_solid.stl")
        export_stl(solid_mesh, stl_path)

        voxel_data = solid_mask # Bool 类型，节省空间 (0/1)
        np.save(os.path.join(sample_dir, "output_voxel.npy"), voxel_data)

        # 5. 生成参考图
        tpms_names = [f.__name__ for f in TPMS_FUNCS]
        colors_light, colors_dark = get_tpms_color_pairs(tpms_names)
        
        fig, ax = visualize_weight_cube_3d(
            weight_grid_3x3, 
            tpms_names,
            density_grid_3x3x3=density_grid_3x3,
            title=f"Sample {sample_id}\nPorosity: {porosity*100:.1f}%",
            base_colors_light=colors_light,
            base_colors_dark=colors_dark,
            show_values=True,
            figsize=5
        )
        fig.savefig(os.path.join(sample_dir, "ref_cube.png"), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # 收集到列表中用于最终聚合
        all_weights.append(weight_grid_3x3)
        all_densities.append(density_grid_3x3)
        all_rotations.append(rotation_grid_3x3)
        all_voxels.append(voxel_data)
        
        # 如果需要，也可以保存完整的权重场/密度场 (数据量较大，视需求而定)
        # np.save(os.path.join(sample_dir, "field_density_full.npy"), density_field)
        
        t_cost = time.time() - t_start
        print(f"[Sample {sample_id}] 孔隙率: {porosity*100:.1f}% | 耗时: {t_cost:.2f}s")
    
    # === 汇总保存为单一的大型 .npy 文件 (N, 120, 120, 120) ===
    print("\n正在聚合所有样本...")
    dataset_voxels = np.array(all_voxels, dtype=bool)           # (N, 120, 120, 120) - Bool 类型
    dataset_weights = np.array(all_weights, dtype=np.float32)   # (N, 3, 3, 3, 2)
    dataset_densities = np.array(all_densities, dtype=np.float32) # (N, 3, 3, 3)
    dataset_rotations = np.array(all_rotations, dtype=np.float32) # (N, 3, 3, 3, 3)

    print(f"保存聚合数据集到 {OUTPUT_DIR} ...")
    np.save(os.path.join(OUTPUT_DIR, "dataset_voxels.npy"), dataset_voxels)
    np.save(os.path.join(OUTPUT_DIR, "dataset_params_weight.npy"), dataset_weights)
    np.save(os.path.join(OUTPUT_DIR, "dataset_params_density.npy"), dataset_densities)
    np.save(os.path.join(OUTPUT_DIR, "dataset_params_rotation.npy"), dataset_rotations)

    print(f"  Voxel Shape: {dataset_voxels.shape}")
    print(f"  Weight Param Shape: {dataset_weights.shape}")
    print(f"  Density Param Shape: {dataset_densities.shape}")
    print(f"  Rotation Param Shape: {dataset_rotations.shape}")

    print(f"\n✅ 数据集生成完成! 总耗时: {time.time() - total_start:.2f}s")
    print(f"数据保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
