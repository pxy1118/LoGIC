"""
图构建模块

实现两种图构建策略:
1. Node-level: 每个样本作为大图中的一个节点 (使用 Layer6 特征)
2. Graph-level: 每个样本作为一张独立的图 (使用 Layer4 特征)

建边方式: KNN (K-Nearest Neighbors) 基于特征空间距离
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Optional, Dict
import os


def build_knn_edges(
    features: np.ndarray,
    k: int = 5,
    include_self: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 KNN 构建边
    
    Args:
        features: (N, D) 特征矩阵
        k: 近邻数量
        include_self: 是否包含自环
        
    Returns:
        edge_index: (2, E) 边索引
        edge_weight: (E,) 边权重 (距离的倒数)
    """
    n_samples = len(features)
    
    # 确保 k 不超过样本数
    k_actual = min(k + 1, n_samples)  # +1 因为最近邻包含自身
    
    # KNN 搜索
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    
    # 构建边列表
    src_nodes = []
    dst_nodes = []
    weights = []
    
    for i in range(n_samples):
        for j_idx in range(k_actual):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            
            # 跳过自环 (除非明确要求)
            if i == j and not include_self:
                continue
            
            src_nodes.append(i)
            dst_nodes.append(j)
            
            # 边权重: 距离越近权重越大
            # 使用高斯核或简单的倒数
            weight = 1.0 / (dist + 1e-6)
            weights.append(weight)
    
    edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    
    return edge_index, edge_weight


def build_node_level_graph(
    features_layer6: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    train_mask: Optional[np.ndarray] = None,
    val_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None
) -> Data:
    """
    构建 Node-level 图 (策略 A)
    
    整个数据集形成一个大图:
    - 每个样本是一个节点
    - 节点特征为 Layer6 输出 (128-d)
    - 使用 KNN 建边
    
    Args:
        features_layer6: (N, 128) Layer6 特征
        labels: (N,) 标签
        k: KNN 的 K 值
        train_mask: 训练集掩码
        val_mask: 验证集掩码
        test_mask: 测试集掩码
        
    Returns:
        PyG Data 对象
    """
    n_samples = len(features_layer6)
    
    # 构建边
    edge_index, edge_weight = build_knn_edges(features_layer6, k=k)
    
    # 创建 PyG Data
    # 处理多输出标签
    if len(labels.shape) > 1:
        y_tensor = torch.tensor(labels, dtype=torch.float32)  # (N, num_outputs)
    else:
        y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # (N, 1)
    
    data = Data(
        x=torch.tensor(features_layer6, dtype=torch.float32),
        y=y_tensor,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32).view(-1, 1)
    )
    
    # 添加掩码
    if train_mask is not None:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    if test_mask is not None:
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    # 统计信息
    data.num_nodes = n_samples
    data.num_edges = edge_index.shape[1]
    
    return data


def build_graph_level_graphs(
    features_layer4: np.ndarray,
    labels: np.ndarray,
    k: int = 5
) -> List[Data]:
    """
    构建 Graph-level 图列表 (策略 B)
    
    每个样本形成一张独立的图:
    - Layer4 输出 (32, 6, 6, 6) 中每个空间位置是一个节点
    - 共 6*6*6 = 216 个节点
    - 节点特征为 32-d
    - 使用 KNN 建边 (在 32-d 特征空间)
    
    Args:
        features_layer4: (N, 32, 6, 6, 6) Layer4 特征
        labels: (N,) 或 (N, num_outputs) 标签
        k: KNN 的 K 值
        
    Returns:
        PyG Data 对象列表
    """
    n_samples = len(features_layer4)
    num_outputs = labels.shape[1] if len(labels.shape) > 1 else 1
    graphs = []
    
    for i in range(n_samples):
        # 获取单个样本的特征: (32, 6, 6, 6)
        feat = features_layer4[i]
        
        # 重塑为节点特征: (6*6*6, 32) = (216, 32)
        # 先转置 (32, 6, 6, 6) -> (6, 6, 6, 32)，再展平
        node_features = feat.transpose(1, 2, 3, 0).reshape(-1, feat.shape[0])
        # 现在 node_features: (216, 32)
        
        # 构建边 (基于特征空间的 KNN)
        edge_index, edge_weight = build_knn_edges(node_features, k=k)
        
        # 处理标签
        if num_outputs > 1:
            label_tensor = torch.tensor(labels[i], dtype=torch.float32)  # (num_outputs,)
        else:
            label_tensor = torch.tensor([labels[i]], dtype=torch.float32)  # (1,)
        
        # 创建 PyG Data
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            y=label_tensor,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32).view(-1, 1)
        )
        
        graphs.append(graph)
    
    return graphs


def add_spatial_edges(
    grid_shape: Tuple[int, int, int] = (6, 6, 6),
    node_features: np.ndarray = None,
    k: int = 5,
    spatial_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    添加基于空间位置的边 (可选的增强方法)
    
    除了特征空间的 KNN 边，还可以添加基于 3D 网格位置的邻接边
    
    Args:
        grid_shape: 网格形状 (6, 6, 6)
        node_features: 节点特征 (用于特征 KNN)
        k: 特征 KNN 的 K 值
        spatial_weight: 空间边的权重系数
        
    Returns:
        edge_index, edge_weight
    """
    d, h, w = grid_shape
    n_nodes = d * h * w
    
    # 6-邻接 (面相邻)
    src_list = []
    dst_list = []
    
    for z in range(d):
        for y in range(h):
            for x in range(w):
                idx = z * h * w + y * w + x
                
                # 6 个方向
                neighbors = []
                if x > 0: neighbors.append((z, y, x - 1))
                if x < w - 1: neighbors.append((z, y, x + 1))
                if y > 0: neighbors.append((z, y - 1, x))
                if y < h - 1: neighbors.append((z, y + 1, x))
                if z > 0: neighbors.append((z - 1, y, x))
                if z < d - 1: neighbors.append((z + 1, y, x))
                
                for nz, ny, nx in neighbors:
                    n_idx = nz * h * w + ny * w + nx
                    src_list.append(idx)
                    dst_list.append(n_idx)
    
    spatial_edges = np.array([src_list, dst_list], dtype=np.int64)
    spatial_weights = np.ones(len(src_list), dtype=np.float32) * spatial_weight
    
    # 如果提供了特征，添加特征 KNN 边
    if node_features is not None:
        feat_edges, feat_weights = build_knn_edges(node_features, k=k)
        
        # 合并边
        edge_index = np.concatenate([spatial_edges, feat_edges], axis=1)
        edge_weight = np.concatenate([spatial_weights, feat_weights])
        
        # 去重 (可选)
        # ...
        
        return edge_index, edge_weight
    
    return spatial_edges, spatial_weights


class PoreGraphDataset(InMemoryDataset):
    """
    Pore-GNN 图数据集 (用于 Graph-level 任务)
    
    继承 PyG 的 InMemoryDataset，支持保存和加载处理后的数据
    """
    
    def __init__(
        self,
        root: str,
        features_layer4: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        k: int = 5,
        transform=None,
        pre_transform=None
    ):
        """
        Args:
            root: 数据集根目录
            features_layer4: Layer4 特征 (首次创建时需要)
            labels: 标签 (首次创建时需要)
            k: KNN 的 K 值
        """
        self.features_layer4 = features_layer4
        self.labels = labels
        self.k = k
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['graph_level_data.pt']
    
    def download(self):
        pass
    
    def process(self):
        if self.features_layer4 is None or self.labels is None:
            raise ValueError("首次创建数据集需要提供 features_layer4 和 labels")
        
        print(f"构建 Graph-level 图数据 (K={self.k})...")
        graphs = build_graph_level_graphs(
            self.features_layer4,
            self.labels,
            k=self.k
        )
        
        print(f"共 {len(graphs)} 个图，每个图 {graphs[0].num_nodes} 个节点")
        
        self.save(graphs, self.processed_paths[0])


# === 测试代码 ===
if __name__ == "__main__":
    # 测试 KNN 建边
    print("测试 KNN 建边...")
    features = np.random.randn(10, 32).astype(np.float32)
    edge_index, edge_weight = build_knn_edges(features, k=3)
    print(f"  节点数: 10, 边数: {edge_index.shape[1]}")
    
    # 测试 Node-level 图构建
    print("\n测试 Node-level 图构建...")
    features_l6 = np.random.randn(100, 128).astype(np.float32)
    labels = np.random.rand(100).astype(np.float32)
    
    graph = build_node_level_graph(features_l6, labels, k=5)
    print(f"  节点数: {graph.num_nodes}")
    print(f"  边数: {graph.num_edges}")
    print(f"  节点特征: {graph.x.shape}")
    print(f"  标签: {graph.y.shape}")
    
    # 测试 Graph-level 图构建
    print("\n测试 Graph-level 图构建...")
    features_l4 = np.random.randn(10, 32, 6, 6, 6).astype(np.float32)
    labels = np.random.rand(10).astype(np.float32)
    
    graphs = build_graph_level_graphs(features_l4, labels, k=5)
    print(f"  图数量: {len(graphs)}")
    print(f"  单图节点数: {graphs[0].num_nodes}")
    print(f"  单图边数: {graphs[0].num_edges}")
    print(f"  节点特征: {graphs[0].x.shape}")
