"""
3D CNN Backbone for Pore-GNN

基于论文 Table 2 的网络结构：
- 3层 3D 卷积 + BatchNorm + ReLU + MaxPool
- Adaptive Max Pooling (输出固定尺寸 6x6x6)
- 全连接层 (128维特征)
- 输出层 (渗透率预测)

用途：
1. 预训练阶段：端到端回归渗透率
2. 特征提取阶段：提取 Layer4 (32x6x6x6) 或 Layer6 (128-d) 特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CNN3DBackbone(nn.Module):
    """
    3D CNN Backbone 用于体素数据的特征提取与渗透率预测
    
    Architecture:
        Input: (B, 1, 120, 120, 120)
        Layer1: Conv3D(1→8) + BN + ReLU + MaxPool(2) → (B, 8, 60, 60, 60)
        Layer2: Conv3D(8→16) + BN + ReLU + MaxPool(2) → (B, 16, 30, 30, 30)
        Layer3: Conv3D(16→32) + BN + ReLU → (B, 32, 30, 30, 30)
        AdaptivePool: (B, 32, 6, 6, 6) → [Layer4 Feature]
        Flatten: (B, 6912)
        FC1: (B, 128) + BN + ReLU → [Layer6 Feature]
        Output: (B, 1)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        filters: Tuple[int, ...] = (8, 16, 32),
        fc_dim: int = 128,
        adaptive_pool_size: int = 6,
        dropout: float = 0.0,
        num_outputs: int = 1
    ):
        """
        Args:
            in_channels: 输入通道数 (体素为1)
            filters: 各卷积层的通道数
            fc_dim: 全连接层维度 (Layer6 特征维度)
            adaptive_pool_size: Adaptive Pooling 输出尺寸
            dropout: Dropout 概率
            num_outputs: 输出维度 (渗透率预测为1)
        """
        super().__init__()
        
        self.filters = filters
        self.fc_dim = fc_dim
        self.adaptive_pool_size = adaptive_pool_size
        
        # === 卷积层 ===
        # Layer 1: Conv + BN + ReLU + MaxPool
        self.conv1 = nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(filters[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Layer 2: Conv + BN + ReLU + MaxPool
        self.conv2 = nn.Conv3d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(filters[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Layer 3: Conv + BN + ReLU (无池化，后接 Adaptive Pooling)
        self.conv3 = nn.Conv3d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(filters[2])
        
        # Adaptive Max Pooling → 固定输出尺寸
        self.adaptive_pool = nn.AdaptiveMaxPool3d(adaptive_pool_size)
        
        # === 全连接层 ===
        # 计算 flatten 后的特征维度
        flatten_dim = filters[2] * (adaptive_pool_size ** 3)  # 32 * 6^3 = 6912
        
        # Layer 6: FC + BN + ReLU
        self.fc1 = nn.Linear(flatten_dim, fc_dim)
        self.bn_fc = nn.BatchNorm1d(fc_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output Layer
        self.fc_out = nn.Linear(fc_dim, num_outputs)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播并返回中间特征
        
        Args:
            x: 输入张量 (B, 1, D, H, W) 或 (B, D, H, W)
            
        Returns:
            包含各层特征的字典:
            - 'layer4': (B, 32, 6, 6, 6) - 用于 Graph-level GNN
            - 'layer6': (B, 128) - 用于 Node-level GNN
            - 'output': (B, 1) - 渗透率预测
        """
        # 确保输入为 5D: (B, C, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        # 卷积层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 8, 60, 60, 60)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 16, 30, 30, 30)
        x = F.relu(self.bn3(self.conv3(x)))              # (B, 32, 30, 30, 30)
        
        # Adaptive Pooling → Layer 4 特征
        layer4 = self.adaptive_pool(x)  # (B, 32, 6, 6, 6)
        
        # Flatten
        x_flat = layer4.view(layer4.size(0), -1)  # (B, 6912)
        
        # FC → Layer 6 特征
        layer6 = F.relu(self.bn_fc(self.fc1(x_flat)))  # (B, 128)
        layer6 = self.dropout(layer6)
        
        # 输出
        output = self.fc_out(layer6)  # (B, 1)
        
        return {
            'layer4': layer4,
            'layer6': layer6,
            'output': output
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，仅返回最终输出
        
        Args:
            x: 输入张量 (B, 1, D, H, W) 或 (B, D, H, W)
            
        Returns:
            渗透率预测 (B, 1)
        """
        return self.forward_features(x)['output']


class CNN3DFeatureExtractor(nn.Module):
    """
    基于预训练 CNN 的特征提取器
    
    冻结 CNN 权重，仅用于提取特征
    """
    
    def __init__(
        self,
        backbone: CNN3DBackbone,
        feature_layer: str = 'layer4',
        freeze: bool = True
    ):
        """
        Args:
            backbone: 预训练的 CNN3DBackbone
            feature_layer: 要提取的特征层 ('layer4' 或 'layer6')
            freeze: 是否冻结权重
        """
        super().__init__()
        
        self.backbone = backbone
        self.feature_layer = feature_layer
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取指定层的特征
        
        Args:
            x: 输入张量 (B, 1, D, H, W)
            
        Returns:
            特征张量:
            - layer4: (B, 32, 6, 6, 6)
            - layer6: (B, 128)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.backbone.forward_features(x)
            return features[self.feature_layer]
    
    @property
    def output_dim(self) -> int:
        """获取输出特征维度"""
        if self.feature_layer == 'layer4':
            return self.backbone.filters[-1]  # 32
        elif self.feature_layer == 'layer6':
            return self.backbone.fc_dim  # 128
        else:
            raise ValueError(f"Unknown feature layer: {self.feature_layer}")
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """获取输出特征形状"""
        if self.feature_layer == 'layer4':
            s = self.backbone.adaptive_pool_size
            return (self.backbone.filters[-1], s, s, s)  # (32, 6, 6, 6)
        elif self.feature_layer == 'layer6':
            return (self.backbone.fc_dim,)  # (128,)
        else:
            raise ValueError(f"Unknown feature layer: {self.feature_layer}")


def load_pretrained_backbone(
    checkpoint_path: str,
    device: str = 'cuda',
    **kwargs
) -> CNN3DBackbone:
    """
    加载预训练的 CNN Backbone
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        **kwargs: CNN 构造参数
        
    Returns:
        加载权重后的 CNN3DBackbone
    """
    model = CNN3DBackbone(**kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 支持两种保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


# === 测试代码 ===
if __name__ == "__main__":
    # 创建模型
    model = CNN3DBackbone()
    print(f"Model created: {model.__class__.__name__}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 模拟输入 (B=2, C=1, D=120, H=120, W=120)
    x = torch.randn(2, 1, 120, 120, 120, device=device)
    
    print(f"\nInput shape: {x.shape}")
    
    # 获取各层特征
    features = model.forward_features(x)
    
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # 测试特征提取器
    print("\n--- Feature Extractor ---")
    extractor = CNN3DFeatureExtractor(model, feature_layer='layer4')
    feat4 = extractor(x)
    print(f"Layer4 features: {feat4.shape}")
    print(f"Output dim: {extractor.output_dim}")
    print(f"Output shape: {extractor.output_shape}")
