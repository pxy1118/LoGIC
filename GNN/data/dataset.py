"""
Voxel Dataset for Pore-GNN

提供体素数据的 PyTorch Dataset 封装:
1. 从 .npy 文件加载聚合数据
2. 支持孔隙率作为临时标签 (当渗透率数据不足时)
3. 支持数据增强 (可选的随机翻转/旋转)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Union
import os


class VoxelDataset(Dataset):
    """
    体素数据集
    
    数据结构:
        - voxels: (N, 120, 120, 120) bool 数组
        - weights: (N, 3, 3, 3, 2) 控制参数
        - densities: (N, 3, 3, 3) 密度参数
        - labels: (N, 2) [渗透率, 压降] 或 (N,) 孔隙率
    """
    
    def __init__(
        self,
        dataset_dir: str,
        label_type: str = 'porosity',  # 'porosity' 或 'permeability'
        indices: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        normalize_label: bool = True,
        return_params: bool = False
    ):
        """
        Args:
            dataset_dir: 数据集目录路径
            label_type: 标签类型 ('porosity' 使用孔隙率, 'permeability' 使用渗透率)
            indices: 使用的样本索引 (用于 train/val/test 划分)
            transform: 数据增强函数
            normalize_label: 是否对标签进行归一化
            return_params: 是否返回控制参数 (weight, density)
        """
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.label_type = label_type
        self.transform = transform
        self.normalize_label = normalize_label
        self.return_params = return_params
        
        # 加载数据
        self._load_data()
        
        # 应用索引过滤
        if indices is not None:
            self.voxels = self.voxels[indices]
            self.weights = self.weights[indices]
            self.densities = self.densities[indices]
            self.labels = self.labels[indices]
            self.sample_ids = [self.sample_ids[i] for i in indices]
        
        # 计算归一化参数 (支持多维标签)
        if normalize_label:
            if self.labels.ndim == 1:
                self.label_mean = self.labels.mean()
                self.label_std = self.labels.std()
                if self.label_std < 1e-8:
                    self.label_std = 1.0
            else:
                # 多输出: (N, num_outputs) - 按列归一化
                self.label_mean = self.labels.mean(axis=0)  # (num_outputs,)
                self.label_std = self.labels.std(axis=0)    # (num_outputs,)
                self.label_std = np.where(self.label_std < 1e-8, 1.0, self.label_std)
        else:
            if self.labels.ndim == 1:
                self.label_mean = 0.0
                self.label_std = 1.0
            else:
                self.label_mean = np.zeros(self.labels.shape[1], dtype=np.float32)
                self.label_std = np.ones(self.labels.shape[1], dtype=np.float32)
    
    def _load_data(self):
        """从 .npy 文件加载数据"""
        # 加载体素
        voxel_path = os.path.join(self.dataset_dir, 'dataset_voxels.npy')
        self.voxels = np.load(voxel_path)  # (N, 120, 120, 120) bool
        
        # 加载控制参数
        weight_path = os.path.join(self.dataset_dir, 'dataset_params_weight.npy')
        density_path = os.path.join(self.dataset_dir, 'dataset_params_density.npy')
        self.weights = np.load(weight_path)    # (N, 3, 3, 3, 2)
        self.densities = np.load(density_path)  # (N, 3, 3, 3)
        
        n_samples = len(self.voxels)
        self.sample_ids = list(range(n_samples))
        
        # 加载/计算标签
        if self.label_type == 'permeability':
            self._load_permeability_labels()
        else:
            # 使用孔隙率作为标签 (从体素直接计算)
            self._compute_porosity_labels()
    
    def _load_permeability_labels(self):
        """加载渗透率和压降标签，并过滤掉仿真失败的样本"""
        csv_path = os.path.join(self.dataset_dir, 'simulation_results.csv')
        
        if not os.path.exists(csv_path):
            print(f"[警告] 未找到仿真数据 {csv_path}，使用孔隙率代替")
            self._compute_porosity_labels()
            return
        
        df = pd.read_csv(csv_path)
        
        # 记录原始样本数
        n_total = len(self.voxels)
        
        # 找出有效的 sample_id (对应 voxels 的索引)
        valid_indices = []
        valid_permeability = []
        valid_pressure_drop = []
        
        # 确保 sample_id 是 int
        if 'sample_id' not in df.columns:
             print("[错误] CSV 中缺少 'sample_id' 列")
             self._compute_porosity_labels()
             return

        # 检测列名
        perm_col = '渗透率 (m^2)' if '渗透率 (m^2)' in df.columns else 'permeability'
        drop_col = '压降 (N/m^3)' if '压降 (N/m^3)' in df.columns else 'pressure_drop'
        
        has_pressure_drop = drop_col in df.columns

        for _, row in df.iterrows():
            try:
                sid = int(row['sample_id'])
                # 确保 ID 对应有效的数据索引
                if 0 <= sid < n_total:
                    valid_indices.append(sid)
                    valid_permeability.append(row[perm_col])
                    if has_pressure_drop:
                        valid_pressure_drop.append(row[drop_col])
            except (ValueError, KeyError):
                continue
        
        if not valid_indices:
             print("[警告] 未找到有效标签，使用孔隙率代替")
             self._compute_porosity_labels()
             return
             
        # 转换为 numpy 数组
        valid_indices = np.array(valid_indices, dtype=np.int64)
        
        # 过滤数据
        n_dropped = n_total - len(valid_indices)
        if n_dropped > 0:
            print(f"[信息] 过滤掉 {n_dropped} 个未仿真/失败样本 (保留 {len(valid_indices)}/{n_total})")
            
            self.voxels = self.voxels[valid_indices]
            self.weights = self.weights[valid_indices]
            self.densities = self.densities[valid_indices]
            self.sample_ids = [self.sample_ids[i] for i in valid_indices]
        
        # 构建标签数组: [log(渗透率), 压降]
        # 对渗透率取对数变换，使其从 ~10^-8 变为 ~-18 (自然对数)
        if has_pressure_drop:
            log_permeability = np.log(np.array(valid_permeability, dtype=np.float32))
            pressure_drop = np.array(valid_pressure_drop, dtype=np.float32)
            
            self.labels = np.stack([log_permeability, pressure_drop], axis=1)  # (N, 2)
            self.log_transform = True  # 标记渗透率使用了对数变换
            
            print(f"[信息] 加载双输出标签 (渗透率已取对数):")
            print(f"       log(K): [{self.labels[:,0].min():.2f}, {self.labels[:,0].max():.2f}]")
            print(f"       压降:   [{self.labels[:,1].min():.2f}, {self.labels[:,1].max():.2f}]")
        else:
            self.labels = np.log(np.array(valid_permeability, dtype=np.float32))
            self.log_transform = True
            print(f"[信息] 加载渗透率标签 (已取对数): [{self.labels.min():.2f}, {self.labels.max():.2f}]")
    
    def _compute_porosity_labels(self):
        """从体素计算孔隙率作为标签"""
        # 孔隙率 = 孔隙体素数 / 总体素数
        # 注意: voxels 中 True=实体, False=孔隙
        total_voxels = np.prod(self.voxels.shape[1:])  # 120^3
        solid_counts = self.voxels.sum(axis=(1, 2, 3))  # 每个样本的实体体素数
        
        # 孔隙率 = 1 - 固体占比
        porosity = 1.0 - (solid_counts / total_voxels)
        self.labels = porosity.astype(np.float32)
        
        print(f"[信息] 使用孔隙率作为标签: 范围 [{self.labels.min():.3f}, {self.labels.max():.3f}]")
    
    def __len__(self) -> int:
        return len(self.voxels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            字典包含:
            - 'voxel': (1, 120, 120, 120) float32 张量
            - 'label': (1,) 标签张量
            - 'sample_id': 样本ID
            可选:
            - 'weight': (3, 3, 3, 2) 权重参数
            - 'density': (3, 3, 3) 密度参数
        """
        # 获取体素 (bool -> float32)
        voxel = self.voxels[idx].astype(np.float32)
        
        # 应用数据增强
        if self.transform is not None:
            voxel = self.transform(voxel)
        
        # 添加通道维度: (120, 120, 120) -> (1, 120, 120, 120)
        voxel = voxel[np.newaxis, ...]
        
        # 获取标签并归一化
        label = self.labels[idx].copy() if self.labels.ndim > 1 else self.labels[idx]
        if self.normalize_label:
            label = (label - self.label_mean) / self.label_std
        
        result = {
            'voxel': torch.from_numpy(voxel),
            'label': torch.tensor(label, dtype=torch.float32) if self.labels.ndim > 1 else torch.tensor([label], dtype=torch.float32),
            'sample_id': self.sample_ids[idx]
        }
        
        if self.return_params:
            result['weight'] = torch.from_numpy(self.weights[idx])
            result['density'] = torch.from_numpy(self.densities[idx])
        
        return result
    
    def denormalize_label(self, normalized_label: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """将归一化的标签还原为原始值"""
        return normalized_label * self.label_std + self.label_mean
    
    def get_statistics(self) -> Dict[str, float]:
        """获取数据集统计信息"""
        stats = {
            'n_samples': len(self),
            'voxel_shape': self.voxels.shape[1:],
        }
        
        if self.labels.ndim == 1:
            stats['label_mean'] = float(self.label_mean)
            stats['label_std'] = float(self.label_std)
            stats['label_min'] = float(self.labels.min())
            stats['label_max'] = float(self.labels.max())
            stats['num_outputs'] = 1
        else:
            # 多输出
            stats['num_outputs'] = self.labels.shape[1]
            stats['label_mean'] = self.label_mean.tolist()
            stats['label_std'] = self.label_std.tolist()
            stats['label_min'] = self.labels.min(axis=0).tolist()
            stats['label_max'] = self.labels.max(axis=0).tolist()
        
        return stats


class VoxelTransform:
    """体素数据增强
    
    对于沿 z 轴流动的流体仿真:
    - 绕 z 轴旋转 (0°, 90°, 180°, 270°) 不改变渗透率
    - x, y 方向翻转不改变渗透率
    - z 方向翻转会交换入口/出口，但对称结构不影响结果
    - 随机分辨率重采样 (模拟多尺度输入)
    """
    
    def __init__(
        self,
        random_flip_xy: bool = True,
        random_flip_z: bool = False,
        random_rotate_z: bool = True,
        random_rotate_all: bool = False,
        random_resolution: bool = False,
        resolution_range: Tuple[int, int] = (64, 120)
    ):
        """
        Args:
            random_flip_xy: 随机翻转 x, y 轴
            random_flip_z: 随机翻转 z 轴
            random_rotate_z: 随机绕 z 轴旋转
            random_rotate_all: 随机绕任意轴旋转
            random_resolution: 是否启用随机分辨率增强
            resolution_range: 分辨率范围 (min_dim, max_dim)
        """
        self.random_flip_xy = random_flip_xy
        self.random_flip_z = random_flip_z
        self.random_rotate_z = random_rotate_z
        self.random_rotate_all = random_rotate_all
        self.random_resolution = random_resolution
        self.resolution_range = resolution_range
    
    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        
        Args:
            voxel: (D, H, W) 体素数组 (bool 或 float)
            
        Returns:
            增强后的体素数组
        """
        # 1. 几何变换
        # 随机翻转 x, y 轴
        if self.random_flip_xy:
            if np.random.random() > 0.5:
                voxel = np.flip(voxel, axis=1)  # 翻转 y
            if np.random.random() > 0.5:
                voxel = np.flip(voxel, axis=2)  # 翻转 x
        
        # 随机翻转 z 轴 (交换入口/出口)
        if self.random_flip_z:
            if np.random.random() > 0.5:
                voxel = np.flip(voxel, axis=0)
        
        # 随机绕 z 轴旋转 (在 xy 平面内旋转)
        if self.random_rotate_z:
            k = np.random.randint(0, 4)  # 0, 90, 180, 270 度
            if k > 0:
                voxel = np.rot90(voxel, k=k, axes=(1, 2))  # 在 xy 平面旋转
        
        # 随机绕任意轴旋转 (会改变流动方向)
        if self.random_rotate_all:
            k = np.random.randint(0, 4)
            axes = [(0, 1), (0, 2), (1, 2)]
            axis = axes[np.random.randint(0, 3)]
            voxel = np.rot90(voxel, k=k, axes=axis)
        
        # 2. 分辨率增强 (多尺度输入模拟)
        # 将体素下采样到随机分辨率，然后再上采样回原始尺寸
        if self.random_resolution and np.random.random() > 0.2:  # 80% 概率应用
            min_dim, max_dim = self.resolution_range
            target_dim = np.random.randint(min_dim, max_dim + 1)
            
            # 只有当目标分辨率小于原始分辨率时才处理
            if target_dim < voxel.shape[0]:
                original_shape = voxel.shape
                
                # 转换为 Tensor (N, C, D, H, W)
                tensor = torch.from_numpy(voxel.astype(np.float32))
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                
                # 下采样
                downsampled = torch.nn.functional.interpolate(
                    tensor, 
                    size=(target_dim, target_dim, target_dim),
                    mode='trilinear',  # 使用三线性插值
                    align_corners=False
                )
                
                # 上采样回原始尺寸
                upsampled = torch.nn.functional.interpolate(
                    downsampled,
                    size=original_shape,
                    mode='trilinear',  # 使用三线性插值保持平滑
                    align_corners=False
                )
                
                # 转回 Numpy
                # 注意：插值后不再是 0/1 二值，而是平滑的 float 值 [0, 1]
                # 这对 CNN 来说是很好的"软输入"
                voxel = upsampled.squeeze().numpy()

        # 确保内存连续
        return np.ascontiguousarray(voxel)


def create_data_loaders(
    dataset_dir: str,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    label_type: str = 'porosity',
    num_workers: int = 0,
    use_augmentation: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器
    
    Args:
        dataset_dir: 数据集目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        label_type: 标签类型
        num_workers: 数据加载线程数
        use_augmentation: 是否使用数据增强
        seed: 随机种子
        
    Returns:
        (train_loader, val_loader, stats_dict)
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 获取实际有效的样本数 (通过实例化一个临时数据集)
    # 注意: 这里 indices=None 会触发 _load_data 进行过滤
    temp_dataset = VoxelDataset(dataset_dir, label_type=label_type, indices=None)
    n_samples = len(temp_dataset)
    
    if n_samples == 0:
        raise ValueError("数据集为空，请检查数据源")
    
    print(f"有效样本总数: {n_samples}")
    del temp_dataset # 释放内存
    
    # 划分索引
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"数据集划分: 训练 {len(train_indices)} / 验证 {len(val_indices)}")
    
    # 创建数据增强 (仅绕 z 轴旋转 + xy 翻转，保持流动方向不变)
    # 启用 random_resolution 以支持多尺度输入训练
    train_transform = VoxelTransform(
        random_flip_xy=True,
        random_flip_z=False,
        random_rotate_z=True,
        random_rotate_all=False,
        random_resolution=True,        # 启用分辨率增强
        resolution_range=(64, 120)     # 分辨率范围
    ) if use_augmentation else None
    
    # 创建数据集
    # 注意: 这里传入 indices 是针对已过滤后的数据集的索引 (0..n_samples-1)
    train_dataset = VoxelDataset(
        dataset_dir=dataset_dir,
        label_type=label_type,
        indices=train_indices,
        transform=train_transform,
        normalize_label=True
    )
    
    val_dataset = VoxelDataset(
        dataset_dir=dataset_dir,
        label_type=label_type,
        indices=val_indices,
        transform=None,  # 验证集不使用增强
        normalize_label=True
    )
    
    # 同步归一化参数 (使用训练集的统计量)
    val_dataset.label_mean = train_dataset.label_mean
    val_dataset.label_std = train_dataset.label_std
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 收集统计信息
    stats = train_dataset.get_statistics()
    stats['n_train'] = len(train_dataset)
    stats['n_val'] = len(val_dataset)
    
    return train_loader, val_loader, stats


# === 测试代码 ===
if __name__ == "__main__":
    import sys
    
    # 测试数据集
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    print(f"测试数据集: {dataset_dir}")
    
    # 创建数据加载器
    train_loader, val_loader, stats = create_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=4,
        train_ratio=0.8,
        label_type='porosity',
        num_workers=0
    )
    
    print(f"\n数据集统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # 测试一个批次
    print(f"\n测试批次加载:")
    batch = next(iter(train_loader))
    
    print(f"  voxel shape: {batch['voxel'].shape}")
    print(f"  label shape: {batch['label'].shape}")
    print(f"  sample_ids: {batch['sample_id']}")
    print(f"  labels (normalized): {batch['label'].squeeze().tolist()}")
