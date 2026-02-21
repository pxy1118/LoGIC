import numpy as np
import os
import sys
import argparse

# 尝试导入 PyVista
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not installed. Please install it with 'pip install pyvista' for 3D visualization.")

# 默认配置
DEFAULT_DATASET_DIR = "dataset_ml"

def visualize_voxel(voxel_data, sample_idx, use_volume=False):
    """
    使用 PyVista 可视化体素数据
    :param voxel_data: (Nx, Ny, Nz) 的 numpy 数组，值为 0 或 1
    :param sample_idx: 样本索引用于标题
    :param use_volume: 是否使用体渲染 (Volume Rendering)，否则使用阈值网格 (Threshold Mesh)
    """
    if not HAS_PYVISTA:
        print("Error: Visualization requires PyVista.")
        return

    print(f"Processing voxel grid of shape {voxel_data.shape}...")
    
    # 创建 PyVista ImageData
    # 注意：我们把数据当作 Cell Data (体素)，所以 Grid 的 Dimensions (节点数) 需要比 Voxel Shape (单元数) 大 1
    grid = pv.ImageData()
    grid.dimensions = np.array(voxel_data.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1) # 这里使用体素坐标，不还原回物理坐标以简化显示
    
    # 填充数据
    # VTK 默认数据顺序是 Fortran-like (column-major)，即 x 变化最快
    # 假设 voxel_data 是 [x, y, z] indexing='ij' 生成的
    grid.cell_data["Density"] = voxel_data.flatten(order="F")
    
    # 设置可视化
    p = pv.Plotter(window_size=[1024, 768])
    p.add_axes()
    p.add_title(f"Sample {sample_idx} Voxel Visualization\nShape: {voxel_data.shape}")

    if use_volume:
        # 体渲染模式 (看起来像云雾或半透明实体)
        vol_opacity = [0, 1.0] # 0透明，1不透明
        p.add_volume(grid, scalars="Density", cmap="viridis", opacity=vol_opacity, shade=True)
    else:
        # 阈值模式 (Minecraft 风格的方块)
        # 提取值 >= 0.5 的部分
        thres = grid.threshold(0.5, scalars="Density")
        
        # 显示边缘让体素感更强
        p.add_mesh(thres, color="#4287f5", show_edges=True, edge_color="#222222", opacity=1.0, lighting=True)
        # 添加包围盒
        p.add_mesh(grid.outline(), color="k")

    print("Opening PyVista window...")
    print("  - Use mouse to rotate, pan, and zoom")
    print("  - Press 'q' to close window")
    p.show()

def load_sample_data(dataset_dir, idx):
    """
    加载指定索引的样本数据，优先读取聚合的大文件，其次读取单独文件
    """
    # 1. 尝试读取聚合的大文件 (使用 mmap 避免内存爆炸)
    big_file_path = os.path.join(dataset_dir, "dataset_voxels.npy")
    if os.path.exists(big_file_path):
        try:
            print(f"Found aggregated dataset: {big_file_path}")
            # mmap_mode='r' 允许我们像数组一样访问文件上的数据而不必全部读入内存
            data_mmap = np.load(big_file_path, mmap_mode='r')
            if idx < 0 or idx >= data_mmap.shape[0]:
                print(f"Error: Index {idx} out of range (0-{data_mmap.shape[0]-1})")
                return None
            return np.array(data_mmap[idx]) # 复制一份到内存
        except Exception as e:
            print(f"Error reading aggregated file: {e}")
    
    # 2. 尝试读取单独的样本文件
    sample_path = os.path.join(dataset_dir, f"{idx:04d}", "output_voxel.npy")
    if os.path.exists(sample_path):
        print(f"Found single sample file: {sample_path}")
        return np.load(sample_path)
    
    print(f"Error: Could not find data for sample index {idx} in {dataset_dir}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize a voxel sample from the generated dataset.")
    parser.add_argument("index", type=int, help="The index of the sample to visualize (e.g., 0, 5, 12)")
    parser.add_argument("--dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to the dataset directory")
    parser.add_argument("--volume", action="store_true", help="Use volume rendering instead of solid blocks")
    
    args = parser.parse_args()
    
    voxel_data = load_sample_data(args.dir, args.index)
    
    if voxel_data is not None:
        visualize_voxel(voxel_data, args.index, use_volume=args.volume)

if __name__ == "__main__":
    main()
