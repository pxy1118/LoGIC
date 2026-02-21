"""
Pore-GNN 模型

实现两种 GNN 架构:
1. Node-Level GNN: 节点回归任务 (在大图上预测每个节点的渗透率)
2. Graph-Level GNN: 图回归任务 (每个图预测一个渗透率值) [推荐]

支持的图卷积类型:
- ChebConv (Chebyshev Spectral CNN) - 论文推荐
- GCNConv (Graph Convolutional Network)
- SAGEConv (GraphSAGE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    ChebConv, GCNConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.data import Data, Batch
from typing import Optional, Literal


class NodeLevelGNN(nn.Module):
    """
    Node-Level Pore-GNN
    
    用于在单个大图上进行节点回归。
    每个节点代表一个样本，预测其渗透率。
    
    Architecture (论文):
        GConv(128 → 32) + ReLU
        GConv(32 → 32) + ReLU
        GConv(32 → 1)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 32,
        num_layers: int = 3,
        conv_type: Literal['ChebConv', 'GCNConv', 'SAGEConv'] = 'ChebConv',
        cheb_k: int = 2,
        dropout: float = 0.0,
        num_outputs: int = 1
    ):
        """
        Args:
            in_channels: 输入特征维度 (Layer6 = 128)
            hidden_channels: 隐藏层维度
            num_layers: GNN 层数
            conv_type: 图卷积类型
            cheb_k: Chebyshev 多项式阶数 (仅 ChebConv)
            dropout: Dropout 概率
            num_outputs: 输出维度 (1=单输出, 2=双输出)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_outputs = num_outputs
        
        # 构建图卷积层
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = num_outputs if i == num_layers - 1 else hidden_channels
            
            conv = self._create_conv(conv_type, in_dim, out_dim, cheb_k)
            self.convs.append(conv)
    
    def _create_conv(self, conv_type: str, in_dim: int, out_dim: int, cheb_k: int):
        """创建图卷积层"""
        if conv_type == 'ChebConv':
            return ChebConv(in_dim, out_dim, K=cheb_k)
        elif conv_type == 'GCNConv':
            return GCNConv(in_dim, out_dim)
        elif conv_type == 'SAGEConv':
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv type: {conv_type}")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG Data 对象，包含 x, edge_index, edge_attr (可选)
            
        Returns:
            节点预测值 (N, num_outputs)
        """
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 最后一层不加激活函数
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphLevelGNN(nn.Module):
    """
    Graph-Level Pore-GNN (推荐)
    
    用于图回归任务。每个图代表一个样本，预测其渗透率。
    
    Architecture (论文):
        GConv(32 → 32) + ReLU
        GConv(32 → 32) + ReLU
        GConv(32 → 32)
        Global Pooling (Sum/Mean/Max)
        FC(32 → num_outputs)
    """
    
    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 32,
        num_layers: int = 3,
        conv_type: Literal['ChebConv', 'GCNConv', 'SAGEConv'] = 'ChebConv',
        cheb_k: int = 2,
        pooling: Literal['sum', 'mean', 'max'] = 'sum',
        dropout: float = 0.0,
        num_outputs: int = 1
    ):
        """
        Args:
            in_channels: 输入特征维度 (Layer4 节点特征 = 32)
            hidden_channels: 隐藏层维度
            num_layers: GNN 层数
            conv_type: 图卷积类型
            cheb_k: Chebyshev 多项式阶数
            pooling: 全局池化类型
            dropout: Dropout 概率
            num_outputs: 输出维度 (1=单输出, 2=双输出)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling
        self.num_outputs = num_outputs
        
        # 图卷积层
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels
            
            conv = self._create_conv(conv_type, in_dim, out_dim, cheb_k)
            self.convs.append(conv)
        
        # 全局池化
        if pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # 预测头
        self.fc = nn.Linear(hidden_channels, num_outputs)
    
    def _create_conv(self, conv_type: str, in_dim: int, out_dim: int, cheb_k: int):
        """创建图卷积层"""
        if conv_type == 'ChebConv':
            return ChebConv(in_dim, out_dim, K=cheb_k)
        elif conv_type == 'GCNConv':
            return GCNConv(in_dim, out_dim)
        elif conv_type == 'SAGEConv':
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv type: {conv_type}")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG Data 或 Batch 对象
            
        Returns:
            图预测值 (B, 1)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 图卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 最后一层也加激活 (与论文一致，池化前)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 全局池化
        x = self.pool(x, batch)  # (B, hidden_channels)
        
        # 预测
        x = self.fc(x)  # (B, num_outputs)
        
        return x


class PoreGNN(nn.Module):
    """
    统一的 Pore-GNN 接口
    
    根据 mode 参数自动选择 Node-Level 或 Graph-Level 架构
    """
    
    def __init__(
        self,
        mode: Literal['node', 'graph'] = 'graph',
        in_channels: int = None,
        hidden_channels: int = 32,
        num_layers: int = 3,
        conv_type: str = 'ChebConv',
        cheb_k: int = 2,
        pooling: str = 'sum',
        dropout: float = 0.0,
        num_outputs: int = 1
    ):
        """
        Args:
            mode: 'node' (Node-Level) 或 'graph' (Graph-Level)
            in_channels: 输入维度 (None 时自动设置: node=128, graph=32)
            num_outputs: 输出维度 (1 或 2)
            其他参数同上
        """
        super().__init__()
        
        self.mode = mode
        self.num_outputs = num_outputs
        
        # 自动设置输入维度
        if in_channels is None:
            in_channels = 128 if mode == 'node' else 32
        
        if mode == 'node':
            self.model = NodeLevelGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                conv_type=conv_type,
                cheb_k=cheb_k,
                dropout=dropout
            )
        else:
            self.model = GraphLevelGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                conv_type=conv_type,
                cheb_k=cheb_k,
                pooling=pooling,
                dropout=dropout,
                num_outputs=num_outputs
            )
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.model(data)


# === 测试代码 ===
if __name__ == "__main__":
    from torch_geometric.data import Batch
    
    print("=" * 50)
    print("测试 Pore-GNN 模型")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # --- 测试 Node-Level GNN ---
    print("1. Node-Level GNN (ChebConv)")
    
    # 模拟大图数据
    n_nodes = 100
    x = torch.randn(n_nodes, 128)
    # 创建随机边
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 5))
    y = torch.randn(n_nodes, 1)
    
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    
    model_node = NodeLevelGNN(
        in_channels=128,
        hidden_channels=32,
        num_layers=3,
        conv_type='ChebConv'
    ).to(device)
    
    out = model_node(data)
    print(f"   输入: x={data.x.shape}, edges={data.edge_index.shape}")
    print(f"   输出: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in model_node.parameters()):,}")
    
    # --- 测试 Graph-Level GNN ---
    print("\n2. Graph-Level GNN (ChebConv)")
    
    # 模拟一批图
    graphs = []
    for _ in range(8):
        n_nodes = 216  # 6x6x6
        x = torch.randn(n_nodes, 32)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 5))
        y = torch.randn(1)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    batch = Batch.from_data_list(graphs).to(device)
    
    model_graph = GraphLevelGNN(
        in_channels=32,
        hidden_channels=32,
        num_layers=3,
        conv_type='ChebConv',
        pooling='sum'
    ).to(device)
    
    out = model_graph(batch)
    print(f"   输入: batch_size=8, nodes_per_graph=216")
    print(f"   输出: {out.shape}")
    print(f"   参数量: {sum(p.numel() for p in model_graph.parameters()):,}")
    
    # --- 测试统一接口 ---
    print("\n3. PoreGNN 统一接口")
    
    pore_gnn = PoreGNN(mode='graph', conv_type='ChebConv').to(device)
    out = pore_gnn(batch)
    print(f"   Graph-Level 输出: {out.shape}")
    
    print("\n✅ 所有测试通过!")
