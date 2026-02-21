"""
Voxel Dataset for Pore-GNN

提供体素数据的 PyTorch Dataset 封装:
1. 从 .npy 文件加载体素数据
2. 从 CSV 文件加载力学性质标签（杨氏模量E和屈服强度yield）
3. 支持数据增强 (可选的随机翻转/旋转)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Union
import os


class VoxelDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        indices: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        normalize_label: bool = True,
        return_params: bool = False,
        target_names: Optional[List[str]] = None,
        curve_points: Optional[int] = None
    ):
        super().__init__()
        
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.normalize_label = normalize_label
        self.return_params = return_params
        self.target_names = target_names if target_names is not None else ['E', 'yield']
        self.curve_points = curve_points
        
        # 1. 加载体素和参数
        self._load_geometry_data()
        
        # 2. 根据配置加载标签 (CSV 或 NPZ)
        self._load_labels()
        
        # 3. 应用索引过滤 (Train/Val Split)
        if indices is not None:
            self._apply_indices(indices)
        
        # 4. 计算归一化参数
        self._compute_normalization_stats()
    
    def _load_geometry_data(self):
        """加载体素、权重和密度"""
        self.voxels = np.load(os.path.join(self.dataset_dir, 'dataset_voxels.npy'))
        self.weights = np.load(os.path.join(self.dataset_dir, 'dataset_params_weight.npy'))
        self.densities = np.load(os.path.join(self.dataset_dir, 'dataset_params_density.npy'))
        self.sample_ids = np.arange(len(self.voxels))

    def _load_labels(self):
        """根据 target_names 分发加载逻辑，支持多任务（曲线+标量）"""
        n_total = len(self.voxels)
        
        # 准备标量数据容器
        scalar_labels = None
        scalar_valid = []
        
        # 准备曲线数据容器
        curve_labels = None
        curve_valid = []

        is_curve_task = 'curves' in self.target_names
        # 检查是否包含标量任务 (E 或 yield)
        scalar_targets = [t for t in self.target_names if t in ['E', 'yield']]
        is_scalar_task = len(scalar_targets) > 0
        
        # 1. 加载曲线数据
        if is_curve_task:
            npz_path = os.path.join(self.dataset_dir, 'dataset_curves_aggregated.npz')
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"曲线数据未找到: {npz_path}")
            
            try:
                data = np.load(npz_path)
                keys = list(data.keys())
                curve_key = next((k for k in ['curves', 'stress', 'data', 'arr_0'] if k in keys), None)
                id_key = next((k for k in ['sample_ids', 'ids', 'id', 'sample'] if k in keys), None)
                
                if curve_key is None: raise KeyError(f"Missing curve data in npz, keys: {keys}")
                curves_raw = data[curve_key].astype(np.float32)
                
                # 如果指定了 curve_points，则截取前 N 个点
                if self.curve_points is not None and self.curve_points > 0:
                    if self.curve_points < curves_raw.shape[1]:
                        # print(f"[Info] 截取前 {self.curve_points} 个曲线点 (原始: {curves_raw.shape[1]})")
                        curves_raw = curves_raw[:, :self.curve_points]
                
                curve_map = {}
                if id_key is not None:
                    ids = data[id_key]
                    if ids.ndim > 1: ids = ids.flatten()
                    for sid, val in zip(ids, curves_raw):
                        curve_map[int(sid)] = val
                else:
                    # 如果没有ID，假设顺序对应
                    for i, val in enumerate(curves_raw):
                        curve_map[i] = val
                
                curve_labels = curve_map
            except Exception as e:
                raise RuntimeError(f"加载曲线数据失败: {e}")

        # 2. 加载标量数据
        if is_scalar_task:
            e_path = os.path.join(self.dataset_dir, 'E.csv')
            y_path = os.path.join(self.dataset_dir, 'yield.csv')
            
            # 使用更鲁棒的读取方式（处理可能的ID格式差异）
            try:
                e_df = pd.read_csv(e_path)
                y_df = pd.read_csv(y_path)
                
                # 标准化ID列名为 'sample'
                if 'sample' not in e_df.columns: e_df.columns = ['sample', 'E']
                if 'sample' not in y_df.columns: y_df.columns = ['sample', 'yield']
                
                # 提取数字ID
                def parse_id(val):
                    s = str(val)
                    if '_'in s: return int(s.split('_')[-1])
                    return int(float(s))
                
                e_df['sid'] = e_df['sample'].apply(parse_id)
                y_df['sid'] = y_df['sample'].apply(parse_id)
                
                merged = pd.merge(e_df, y_df, on='sid', how='inner')
                scalar_map = {}
                for _, row in merged.iterrows():
                    vals = []
                    for t in scalar_targets:
                        if t == 'E': vals.append(row['E'])
                        elif t == 'yield': vals.append(row['yield'])
                    scalar_map[int(row['sid'])] = np.array(vals, dtype=np.float32)
                
                scalar_labels = scalar_map
                
            except Exception as e:
                raise RuntimeError(f"加载标量数据失败: {e}")

        # 3. 合并数据并对齐
        self.valid_indices = []
        final_labels = []
        
        for i in range(n_total):
            # 检查该样本是否拥有所有需要的数据
            has_curve = (not is_curve_task) or (curve_labels is not None and i in curve_labels)
            has_scalar = (not is_scalar_task) or (scalar_labels is not None and i in scalar_labels)
            
            if has_curve and has_scalar:
                self.valid_indices.append(i)
                
                parts = []
                # 注意顺序：先放标量 (E, yield)，再放曲线
                # 这样 loss function 可以容易地切分前 k 个
                if is_scalar_task:
                    parts.append(scalar_labels[i])
                if is_curve_task:
                    parts.append(curve_labels[i])
                
                combined = np.concatenate(parts) if len(parts) > 1 else parts[0]
                final_labels.append(combined)

        self.labels = np.array(final_labels, dtype=np.float32)
        self.valid_indices = np.array(self.valid_indices, dtype=np.int64)

        # 应用过滤
        if len(self.valid_indices) < n_total:
            print(f"[Info] 过滤有效样本 (Multitask): {len(self.valid_indices)}/{n_total}")
            self.voxels = self.voxels[self.valid_indices]
            self.weights = self.weights[self.valid_indices]
            self.densities = self.densities[self.valid_indices]
            self.sample_ids = self.valid_indices 
 

    def _apply_indices(self, indices):
        self.voxels = self.voxels[indices]
        self.weights = self.weights[indices]
        self.densities = self.densities[indices]
        self.labels = self.labels[indices]
        self.sample_ids = self.sample_ids[indices]

    def _compute_normalization_stats(self):
        if self.normalize_label:
            if self.labels.ndim == 1:
                self.label_mean = self.labels.mean()
                self.label_std = self.labels.std() or 1.0
            else:
                self.label_mean = self.labels.mean(axis=0)
                self.label_std = self.labels.std(axis=0)
                self.label_std[self.label_std < 1e-8] = 1.0
        else:
            dim = 1 if self.labels.ndim == 1 else self.labels.shape[1]
            self.label_mean = np.zeros(dim) if dim > 1 else 0.0
            self.label_std = np.ones(dim) if dim > 1 else 1.0
    
    def __len__(self) -> int:
        return len(self.voxels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        voxel = self.voxels[idx].astype(np.float32)
        if self.transform is not None:
            voxel = self.transform(voxel)
        voxel = voxel[np.newaxis, ...]
        
        label = self.labels[idx].copy()
        if self.normalize_label:
            label = (label - self.label_mean) / self.label_std
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if label_tensor.ndim == 0:
             label_tensor = label_tensor.unsqueeze(0)

        result = {
            'voxel': torch.from_numpy(voxel),
            'label': label_tensor,
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
        stats = {
            'n_samples': len(self),
            'voxel_shape': self.voxels.shape[1:],
            'num_outputs': 1 if self.labels.ndim == 1 else self.labels.shape[1]
        }
        if stats['num_outputs'] > 5:
            stats['label_mean_avg'] = float(np.mean(self.label_mean))
            stats['label_std_avg'] = float(np.mean(self.label_std))
            stats['label_mean'] = self.label_mean
            stats['label_std'] = self.label_std
        else:
             if self.labels.ndim == 1:
                stats['label_mean'] = float(self.label_mean)
                stats['label_std'] = float(self.label_std)
             else:
                stats['label_mean'] = self.label_mean.tolist()
                stats['label_std'] = self.label_std.tolist()
        return stats


class VoxelTransform:
    """
    体素数据增强变换
    
    支持的增强方式：
    - 随机翻转 (XY平面和Z轴)
    - 随机旋转 (Z轴90度旋转)
    - 多尺度输入 (随机下采样+上采样)
    
    Args:
        random_flip_xy: 是否随机翻转XY平面
        random_flip_z: 是否随机翻转Z轴
        random_rotate_z: 是否随机Z轴旋转
        random_rotate_all: 是否随机全方向旋转（暂未实现）
        random_resolution: 是否启用多尺度输入增强
        resolution_range: 多尺度输入的分辨率范围 (min_size, max_size)
        resolution_prob: 应用多尺度增强的概率 (0.0-1.0)
    """
    def __init__(
        self, 
        random_flip_xy=True, 
        random_flip_z=False, 
        random_rotate_z=True, 
        random_rotate_all=False, 
        random_resolution=False, 
        resolution_range=(64, 120),
        resolution_prob=0.5
    ):
        self.random_flip_xy = random_flip_xy
        self.random_flip_z = random_flip_z
        self.random_rotate_z = random_rotate_z
        self.random_rotate_all = random_rotate_all
        self.random_resolution = random_resolution
        self.resolution_range = resolution_range
        self.resolution_prob = resolution_prob
        
        # 验证参数
        if self.random_resolution:
            assert len(resolution_range) == 2, "resolution_range 必须是 (min_size, max_size)"
            assert resolution_range[0] > 0 and resolution_range[1] > resolution_range[0], \
                "resolution_range 必须满足 0 < min_size < max_size"
    
    def _downsample_upsample(self, voxel):
        """
        多尺度输入增强：下采样到低分辨率，然后上采样回原始尺寸
        
        这种增强可以：
        1. 模拟不同扫描分辨率的数据
        2. 增强模型对尺度变化的鲁棒性
        3. 起到类似模糊的正则化效果
        
        Args:
            voxel: (D, H, W) 体素数据
        
        Returns:
            处理后的体素数据，形状不变
        """
        from scipy.ndimage import zoom
        
        original_shape = voxel.shape
        min_size, max_size = self.resolution_range
        
        # 随机选择目标分辨率
        # 使用离散的分辨率级别，避免过于随机
        possible_sizes = [32, 40, 48, 60, 80, 96]
        possible_sizes = [s for s in possible_sizes if min_size <= s < max_size]
        
        if not possible_sizes:
            # 如果没有合适的离散级别，使用随机值
            target_size = np.random.randint(min_size, max_size)
        else:
            target_size = np.random.choice(possible_sizes)
        
        # 计算缩放因子
        scale_factor = target_size / original_shape[0]
        
        # 下采样：使用最近邻插值保持二值特性
        downsampled = zoom(voxel, scale_factor, order=0)
        
        # 上采样回原始尺寸
        upsampled = zoom(downsampled, 1.0 / scale_factor, order=0)
        
        # 确保形状完全匹配（处理可能的舍入误差）
        if upsampled.shape != original_shape:
            # 裁剪或填充到原始形状
            result = np.zeros(original_shape, dtype=voxel.dtype)
            slices = tuple(slice(0, min(upsampled.shape[i], original_shape[i])) for i in range(3))
            result[slices] = upsampled[slices]
            upsampled = result
        
        return upsampled
    
    def __call__(self, voxel):
        """
        应用数据增强变换
        
        Args:
            voxel: (D, H, W) 体素数据
        
        Returns:
            增强后的体素数据
        """
        # 1. 多尺度输入增强（如果启用）
        if self.random_resolution and np.random.random() < self.resolution_prob:
            voxel = self._downsample_upsample(voxel)
        
        # 2. 随机翻转
        if self.random_flip_xy:
            if np.random.random() > 0.5: 
                voxel = np.flip(voxel, axis=1)
            if np.random.random() > 0.5: 
                voxel = np.flip(voxel, axis=2)
        
        if self.random_flip_z and np.random.random() > 0.5:
            voxel = np.flip(voxel, axis=0)
        
        # 3. 随机旋转
        if self.random_rotate_z:
            k = np.random.randint(0, 4)
            if k > 0: 
                voxel = np.rot90(voxel, k=k, axes=(1, 2))
        
        return np.ascontiguousarray(voxel)

def create_data_loaders(
    dataset_dir: str,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    use_augmentation: bool = True,
    seed: int = 42,
    target_names: Optional[List[str]] = None,
    curve_points: Optional[int] = None,
    use_multiscale: bool = False,
    multiscale_range: Tuple[int, int] = (60, 120),
    multiscale_prob: float = 0.5
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器
    
    Args:
        dataset_dir: 数据集目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        num_workers: 数据加载线程数
        use_augmentation: 是否使用基础数据增强（翻转、旋转）
        seed: 随机种子
        target_names: 预测目标名称列表
        curve_points: 曲线点数（如果预测曲线）
        use_multiscale: 是否使用多尺度输入增强
        multiscale_range: 多尺度增强的分辨率范围 (min_size, max_size)
        multiscale_prob: 应用多尺度增强的概率
    
    Returns:
        train_loader, val_loader, stats
    """
    np.random.seed(seed)
    
    temp_dataset = VoxelDataset(dataset_dir, indices=None, target_names=target_names, curve_points=curve_points)
    n_samples = len(temp_dataset)
    del temp_dataset
    
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # 创建训练集的数据增强变换
    if use_augmentation:
        transform = VoxelTransform(
            random_flip_xy=True, 
            random_flip_z=True, 
            random_rotate_z=True,
            random_resolution=use_multiscale,
            resolution_range=multiscale_range,
            resolution_prob=multiscale_prob
        )
    else:
        transform = None
    
    train_dataset = VoxelDataset(
        dataset_dir, 
        indices=train_indices, 
        transform=transform, 
        normalize_label=True, 
        target_names=target_names, 
        curve_points=curve_points
    )
    
    val_dataset = VoxelDataset(
        dataset_dir, 
        indices=val_indices, 
        transform=None,  # 验证集不使用数据增强
        normalize_label=True, 
        target_names=target_names, 
        curve_points=curve_points
    )
    
    val_dataset.label_mean = train_dataset.label_mean
    val_dataset.label_std = train_dataset.label_std
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
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
