"""
3D CNN Backbone

基于论文 Table 2 的网络结构：
- 3层 3D 卷积 + BatchNorm + ReLU + MaxPool
- Adaptive Max Pooling (输出固定尺寸 6x6x6)
- 全连接层 (128维特征)
- 输出层 (渗透率预测或应力应变曲线预测)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with skip connections
    
    Architecture:
        Input → Conv3D → BN → Activation → Conv3D → BN → (+) → Activation
          |_______________________________________________|
                        (skip connection)
    
    If in_channels != out_channels or stride != 1, uses 1x1x1 conv for projection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (for downsampling)
        activation: Activation function type ('elu' or 'relu')
        use_bottleneck: Whether to use bottleneck design (not used in basic block)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'elu',
        use_bottleneck: bool = False
    ):
        super().__init__()
        
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.use_bottleneck = use_bottleneck
        
        if use_bottleneck:
            # Bottleneck: 1x1 → 3x3 → 1x1 (reduces parameters)
            mid_channels = out_channels // 4
            self.conv1 = nn.Conv3d(in_channels, mid_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(mid_channels)
            self.conv2 = nn.Conv3d(mid_channels, mid_channels, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm3d(mid_channels)
            self.conv3 = nn.Conv3d(mid_channels, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm3d(out_channels)
        else:
            # Basic: 3x3 → 3x3
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.conv3 = None
            self.bn3 = None
        
        # Projection shortcut if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection
        
        Args:
            x: Input tensor (B, in_channels, D, H, W)
            
        Returns:
            Output tensor (B, out_channels, D', H', W')
        """
        identity = self.shortcut(x)
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.conv3 is not None:  # Bottleneck
            out = self.bn3(self.conv3(out))
        
        out += identity  # Skip connection
        out = self.activation(out)
        
        return out


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention Module
    
    Computes attention weights for each spatial location using:
    1. Channel-wise pooling (max + avg)
    2. Convolutional attention map generation
    3. Sigmoid activation for weights in [0, 1]
    
    Architecture:
        Input (B, C, D, H, W)
          ↓
        MaxPool + AvgPool along channel → (B, 2, D, H, W)
          ↓
        Conv3D(2→1, kernel=7) → (B, 1, D, H, W)
          ↓
        Sigmoid → Attention Map
          ↓
        Element-wise multiply with input
    
    Args:
        kernel_size: Kernel size for attention convolution (default: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying spatial attention
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Attention-weighted tensor (B, C, D, H, W)
        """
        # Channel-wise pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, D, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (B, 1, D, H, W)
        
        # Concatenate pooled features
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, D, H, W)
        
        # Generate attention map
        attention = self.sigmoid(self.conv(pooled))  # (B, 1, D, H, W)
        
        # Apply attention
        return x * attention


class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention Module (Squeeze-and-Excitation)

    Computes attention weights for each channel using:
    1. Global spatial pooling (avg + max)
    2. Two-layer MLP with bottleneck
    3. Sigmoid activation

    Architecture:
        Input (B, C, D, H, W)
          ↓
        Global AvgPool + MaxPool → (B, C, 1, 1, 1)
          ↓
        MLP: Linear(C → C//reduction) → ReLU → Linear(C//reduction → C)
          ↓
        Sigmoid → Channel Attention Weights (B, C, 1, 1, 1)
          ↓
        Element-wise multiply with input

    More efficient than spatial attention (fewer parameters).

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying channel attention

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Attention-weighted tensor (B, C, D, H, W)
        """
        B, C, _, _, _ = x.shape

        # Global pooling
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))

        # Combine and activate
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)

        return x * attention


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D
    
    Combines channel and spatial attention sequentially:
    Input → Channel Attention → Spatial Attention → Output
    
    This module applies both channel and spatial attention mechanisms
    in sequence, allowing the network to focus on both "what" (channels)
    and "where" (spatial locations) are important.
    
    Architecture:
        Input (B, C, D, H, W)
          ↓
        Channel Attention → (B, C, D, H, W)
          ↓
        Spatial Attention → (B, C, D, H, W)
    
    More powerful but more computationally expensive than individual
    attention mechanisms.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention bottleneck (default: 16)
        kernel_size: Kernel size for spatial attention convolution (default: 7)
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attn = ChannelAttention3D(channels, reduction)
        self.spatial_attn = SpatialAttention3D(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying CBAM attention
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Attention-weighted tensor (B, C, D, H, W)
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion Module
    
    Combines features from multiple layers by:
    1. Adaptive pooling to common spatial size
    2. Concatenation along channel dimension
    3. 1x1x1 conv to reduce to target channels
    
    Example:
        Layer1: (B, 8, 60, 60, 60)   → Pool → (B, 8, 6, 6, 6)
        Layer2: (B, 16, 30, 30, 30)  → Pool → (B, 16, 6, 6, 6)
        Layer3: (B, 32, 30, 30, 30)  → Pool → (B, 32, 6, 6, 6)
        Concat: (B, 56, 6, 6, 6)
        Fusion: (B, 32, 6, 6, 6)  [via 1x1x1 conv]
    
    Args:
        in_channels_list: List of input channel counts for each feature
        out_channels: Target number of output channels
        target_size: Target spatial size after pooling (default: 6)
    """
    def __init__(
        self,
        in_channels_list: Tuple[int, ...],
        out_channels: int,
        target_size: int = 6
    ):
        super().__init__()
        
        self.target_size = target_size
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveMaxPool3d(target_size) for _ in in_channels_list
        ])
        
        total_channels = sum(in_channels_list)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU()
        )
    
    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass fusing multi-scale features
        
        Args:
            features: Tuple of feature tensors from different layers
                     Each tensor has shape (B, C_i, D_i, H_i, W_i)
        
        Returns:
            Fused feature tensor (B, out_channels, target_size, target_size, target_size)
        """
        # Pool all features to same size
        pooled = [pool(feat) for pool, feat in zip(self.adaptive_pools, features)]
        
        # Concatenate along channel dimension
        concat = torch.cat(pooled, dim=1)
        
        # Fuse with 1x1x1 convolution
        fused = self.fusion_conv(concat)
        
        return fused


class SequenceDecoder(nn.Module):
    """
    序列解码器 (LSTM/GRU)
    用于将CNN特征解码为完整的应力应变曲线序列
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        output_seq_len: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.seq_len = output_seq_len  # Alias for compatibility
        
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.feature_proj(x)
        inputs = x_proj.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(inputs)
        preds = self.output_head(out)
        return preds.squeeze(-1)


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for stress-strain curve prediction

    Uses positional encoding and self-attention to model
    dependencies in the output sequence.

    Architecture:
        CNN Features (B, input_dim)
          ↓
        Feature Projection → (B, 1, hidden_dim) [memory]
          ↓
        Query Tokens + Positional Embeddings → (B, seq_len, hidden_dim)
          ↓
        Transformer Decoder Layers (multi-head attention)
          ↓
        Output Projection → (B, seq_len, 1) → (B, seq_len)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_seq_len: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Dimension of CNN features
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer decoder layers
            output_seq_len: Length of output sequence (41 for stress-strain curve)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.seq_len = output_seq_len
        self.hidden_dim = hidden_dim

        # Project CNN features to hidden dim
        self.feature_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional embeddings for sequence positions
        self.pos_embedding = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection to single value per position
        self.output_head = nn.Linear(hidden_dim, 1)

        # Learnable query tokens for each sequence position
        self.query_tokens = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN features (B, input_dim)

        Returns:
            Predicted curve (B, seq_len)
        """
        B = x.size(0)

        # Project features and expand to sequence for memory
        memory = self.feature_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)

        # Prepare queries with positional encoding
        queries = self.query_tokens.expand(B, -1, -1) + self.pos_embedding

        # Transformer decoding
        out = self.transformer(queries, memory)  # (B, seq_len, hidden_dim)

        # Project to output
        preds = self.output_head(out).squeeze(-1)  # (B, seq_len)

        return preds


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
        num_outputs: int = 1,
        activation: str = 'elu',
        decoder_type: str = 'mlp',  # 新增: 解码器类型
        decoder_hidden_dim: int = 256,  # 解码器隐藏层维度
        decoder_num_layers: int = 2,  # 解码器层数
        decoder_num_heads: int = 4  # Transformer 注意力头数
    ):
        """
        Args:
            decoder_type: 'mlp', 'lstm', 或 'transformer' (用于多点曲线预测)
            decoder_hidden_dim: 解码器隐藏层维度
            decoder_num_layers: 解码器层数
            decoder_num_heads: Transformer 注意力头数 (仅用于 transformer)
            其他参数同上
        """
        super().__init__()
        
        self.filters = filters
        self.fc_dim = fc_dim
        self.adaptive_pool_size = adaptive_pool_size
        self.activation = activation
        self.num_outputs = num_outputs
        self.decoder_type = decoder_type
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads = decoder_num_heads
        
        # 选择激活函数
        if activation == 'elu':
            self.act_fn = nn.ELU()
        else:
            self.act_fn = nn.ReLU()
        
        # === 卷积层 ===
        # Layer 1: Conv + BN + Activation + MaxPool
        self.conv1 = nn.Conv3d(in_channels, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(filters[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Layer 2: Conv + BN + Activation + MaxPool
        self.conv2 = nn.Conv3d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(filters[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Layer 3: Conv + BN + Activation (无池化，后接 Adaptive Pooling)
        self.conv3 = nn.Conv3d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(filters[2])
        
        # Adaptive Max Pooling → 固定输出尺寸
        self.adaptive_pool = nn.AdaptiveMaxPool3d(adaptive_pool_size)
        
        # === 全连接层 ===
        # 计算 flatten 后的特征维度
        flatten_dim = filters[2] * (adaptive_pool_size ** 3)  # 32 * 6^3 = 6912
        
        # Layer 6: FC + BN + Activation
        self.fc1 = nn.Linear(flatten_dim, fc_dim)
        self.bn_fc = nn.BatchNorm1d(fc_dim)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output Layer (Decoder) - 使用工厂方法创建
        self.decoder = self._create_decoder(
            fc_dim=fc_dim,
            num_outputs=num_outputs,
            decoder_type=decoder_type,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            dropout=dropout
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_decoder(
        self,
        fc_dim: int,
        num_outputs: int,
        decoder_type: str,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0
    ) -> nn.Module:
        """
        解码器工厂方法
        
        创建指定类型的解码器，支持 'mlp', 'lstm', 'transformer' 三种类型。
        
        Args:
            fc_dim: 输入特征维度 (来自 FC 层)
            num_outputs: 输出序列长度或输出维度
            decoder_type: 解码器类型 ('mlp', 'lstm', 'transformer')
            hidden_dim: 解码器隐藏层维度 (用于 lstm 和 transformer)
            num_layers: 解码器层数 (用于 lstm 和 transformer)
            num_heads: 注意力头数 (仅用于 transformer)
            dropout: Dropout 概率
        
        Returns:
            nn.Module: 解码器模块
        
        Raises:
            ValueError: 如果 decoder_type 不是支持的类型
        
        Examples:
            >>> # MLP 解码器 (单点预测)
            >>> decoder = self._create_decoder(128, 1, 'mlp')
            
            >>> # LSTM 解码器 (序列预测)
            >>> decoder = self._create_decoder(128, 41, 'lstm', hidden_dim=256, num_layers=2)
            
            >>> # Transformer 解码器 (序列预测)
            >>> decoder = self._create_decoder(128, 41, 'transformer', 
            ...                                hidden_dim=256, num_layers=2, num_heads=4)
        """
        if decoder_type == 'mlp':
            # 简单的线性层，用于单点预测或简单的多点预测
            return nn.Linear(fc_dim, num_outputs)
        
        elif decoder_type == 'lstm':
            # LSTM 解码器，用于序列预测
            if num_outputs <= 1:
                raise ValueError(
                    f"LSTM decoder requires num_outputs > 1 for sequence prediction, "
                    f"got num_outputs={num_outputs}. Use 'mlp' decoder for single output."
                )
            return SequenceDecoder(
                input_dim=fc_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_seq_len=num_outputs,
                dropout=dropout
            )
        
        elif decoder_type == 'transformer':
            # Transformer 解码器，用于序列预测
            if num_outputs <= 1:
                raise ValueError(
                    f"Transformer decoder requires num_outputs > 1 for sequence prediction, "
                    f"got num_outputs={num_outputs}. Use 'mlp' decoder for single output."
                )
            return TransformerDecoder(
                input_dim=fc_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_seq_len=num_outputs,
                num_heads=num_heads,
                dropout=dropout
            )
        
        else:
            raise ValueError(
                f"Invalid decoder_type: '{decoder_type}'. "
                f"Must be 'mlp', 'lstm', or 'transformer'."
            )
    
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
            - 'layer4': (B, 32, 6, 6, 6) - 卷积特征
            - 'layer6': (B, 128) - 全连接特征
            - 'output': (B, num_outputs) - 预测输出
        """
        # 确保输入为 5D: (B, C, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        # 卷积层
        x = self.pool1(self.act_fn(self.bn1(self.conv1(x))))  # (B, 8, 60, 60, 60)
        x = self.pool2(self.act_fn(self.bn2(self.conv2(x))))  # (B, 16, 30, 30, 30)
        x = self.act_fn(self.bn3(self.conv3(x)))              # (B, 32, 30, 30, 30)
        
        # Adaptive Pooling → Layer 4 特征
        layer4 = self.adaptive_pool(x)  # (B, 32, 6, 6, 6)
        
        # Flatten
        x_flat = layer4.view(layer4.size(0), -1)  # (B, 6912)
        
        # FC → Layer 6 特征
        layer6 = self.act_fn(self.bn_fc(self.fc1(x_flat)))  # (B, 128)
        layer6 = self.dropout_layer(layer6)
        
        # 输出
        output = self.decoder(layer6)  # (B, num)
        
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


class ImprovedCNN3DBackbone(nn.Module):
    """
    Improved 3D CNN Backbone with:
    - Residual connections (optional)
    - Attention mechanisms (optional)
    - Multi-scale fusion (optional)
    - Multiple decoder options
    
    Backward compatible with original CNN3DBackbone
    
    Architecture modes:
    - 'simple': Original 3-layer CNN (backward compatible)
    - 'resnet': ResNet-style with residual blocks
    
    Args:
        in_channels: Number of input channels (default: 1)
        filters: Tuple of filter sizes for each layer (default: (8, 16, 32))
        fc_dim: Fully connected layer dimension (default: 128)
        adaptive_pool_size: Size after adaptive pooling (default: 6)
        dropout: Dropout probability (default: 0.0)
        num_outputs: Number of output values (default: 1)
        activation: Activation function 'elu' or 'relu' (default: 'elu')
        decoder_type: Decoder type 'mlp', 'lstm', or 'transformer' (default: 'lstm')
        decoder_hidden_dim: Decoder hidden dimension (default: 256)
        decoder_num_layers: Number of decoder layers (default: 2)
        decoder_num_heads: Number of attention heads for transformer (default: 4)
        architecture_type: 'simple' or 'resnet' (default: 'simple')
        use_attention: Enable attention mechanisms (default: False)
        attention_type: 'channel', 'spatial', or 'cbam' (default: 'channel')
        attention_positions: List of layer indices to apply attention (default: None)
        use_multi_scale: Enable multi-scale feature fusion (default: False)
        fusion_layers: List of layer indices to fuse (default: None)
        use_bottleneck: Use bottleneck design in residual blocks (default: False)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        filters: Tuple[int, ...] = (8, 16, 32),
        fc_dim: int = 128,
        adaptive_pool_size: int = 6,
        dropout: float = 0.0,
        num_outputs: int = 1,
        activation: str = 'elu',
        decoder_type: str = 'lstm',
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 4,
        # New parameters for improved architecture
        architecture_type: str = 'simple',
        use_attention: bool = False,
        attention_type: str = 'channel',
        attention_positions: Optional[List[int]] = None,
        use_multi_scale: bool = False,
        fusion_layers: Optional[List[int]] = None,
        use_bottleneck: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.architecture_type = architecture_type
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale
        self.filters = filters
        self.num_outputs = num_outputs
        self.fc_dim = fc_dim
        self.adaptive_pool_size = adaptive_pool_size
        self.activation = activation
        self.decoder_type = decoder_type
        
        # Validate architecture type
        if architecture_type not in ['simple', 'resnet']:
            raise ValueError(
                f"Invalid architecture_type: '{architecture_type}'. "
                f"Must be 'simple' or 'resnet'."
            )
        
        # Validate attention type
        if use_attention and attention_type not in ['channel', 'spatial', 'cbam']:
            raise ValueError(
                f"Invalid attention_type: '{attention_type}'. "
                f"Must be 'channel', 'spatial', or 'cbam'."
            )
        
        # Validate decoder type
        if decoder_type not in ['mlp', 'lstm', 'transformer']:
            raise ValueError(
                f"Invalid decoder_type: '{decoder_type}'. "
                f"Must be 'mlp', 'lstm', or 'transformer'."
            )
        
        # Build encoder based on architecture type
        if architecture_type == 'simple':
            self.encoder = self._build_simple_encoder(
                in_channels, filters, activation
            )
        elif architecture_type == 'resnet':
            self.encoder = self._build_resnet_encoder(
                in_channels, filters, activation, use_bottleneck
            )
        
        # Attention modules (optional)
        self.attention_modules = nn.ModuleDict()
        if use_attention:
            attention_positions = attention_positions or [2]  # Default: after layer 2
            for pos in attention_positions:
                if pos < len(filters):
                    self.attention_modules[f'attn_{pos}'] = self._create_attention(
                        filters[pos], attention_type
                    )
        
        # Multi-scale fusion (optional)
        if use_multi_scale:
            fusion_layers = fusion_layers or list(range(len(filters)))
            fusion_channels = [filters[i] for i in fusion_layers]
            self.fusion = MultiScaleFusion(
                in_channels_list=fusion_channels,
                out_channels=filters[-1],
                target_size=adaptive_pool_size
            )
            self.fusion_layer_indices = fusion_layers
        else:
            self.fusion = None
        
        # Adaptive pooling (used when fusion is disabled)
        self.adaptive_pool = nn.AdaptiveMaxPool3d(adaptive_pool_size)
        
        # FC layer
        flatten_dim = filters[-1] * (adaptive_pool_size ** 3)
        self.fc1 = nn.Linear(flatten_dim, fc_dim)
        self.bn_fc = nn.BatchNorm1d(fc_dim)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_fn = nn.ELU() if activation == 'elu' else nn.ReLU()
        
        # Decoder
        self.decoder = self._create_decoder(
            fc_dim=fc_dim,
            num_outputs=num_outputs,
            decoder_type=decoder_type,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _build_simple_encoder(
        self,
        in_channels: int,
        filters: Tuple[int, ...],
        activation: str
    ) -> nn.ModuleList:
        """
        Build original simple encoder (backward compatible)
        
        Architecture:
            Layer 1: Conv3D + BN + Activation + MaxPool(2)
            Layer 2: Conv3D + BN + Activation + MaxPool(2)
            Layer 3: Conv3D + BN + Activation (no pooling)
        
        Args:
            in_channels: Number of input channels
            filters: Tuple of filter sizes
            activation: Activation function type
        
        Returns:
            ModuleList of encoder layers
        """
        act_fn = nn.ELU() if activation == 'elu' else nn.ReLU()
        layers = nn.ModuleList()
        
        # Layer 1: Conv + BN + Activation + MaxPool
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, filters[0], 3, padding=1),
            nn.BatchNorm3d(filters[0]),
            act_fn,
            nn.MaxPool3d(2, 2)
        ))
        
        # Layer 2: Conv + BN + Activation + MaxPool
        layers.append(nn.Sequential(
            nn.Conv3d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm3d(filters[1]),
            act_fn,
            nn.MaxPool3d(2, 2)
        ))
        
        # Layer 3: Conv + BN + Activation (no pooling)
        layers.append(nn.Sequential(
            nn.Conv3d(filters[1], filters[2], 3, padding=1),
            nn.BatchNorm3d(filters[2]),
            act_fn
        ))
        
        return layers
    
    def _build_resnet_encoder(
        self,
        in_channels: int,
        filters: Tuple[int, ...],
        activation: str,
        use_bottleneck: bool
    ) -> nn.ModuleList:
        """
        Build ResNet-style encoder with residual blocks
        
        Architecture:
            Stem: Conv3D + BN + Activation
            Block 1: ResidualBlock (stride=2 for downsampling)
            Block 2: ResidualBlock (stride=2 for downsampling)
            Block 3: ResidualBlock (stride=1, no downsampling)
        
        Args:
            in_channels: Number of input channels
            filters: Tuple of filter sizes
            activation: Activation function type
            use_bottleneck: Use bottleneck design in residual blocks
        
        Returns:
            ModuleList of encoder layers
        """
        layers = nn.ModuleList()
        
        # Initial conv (stem)
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, filters[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(filters[0]),
            nn.ELU() if activation == 'elu' else nn.ReLU()
        ))
        
        # Residual blocks with downsampling
        for i in range(len(filters) - 1):
            stride = 2 if i < 2 else 1  # Downsample first two transitions
            layers.append(ResidualBlock3D(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                stride=stride,
                activation=activation,
                use_bottleneck=use_bottleneck
            ))
        
        # Final residual block (no downsampling)
        layers.append(ResidualBlock3D(
            in_channels=filters[-1],
            out_channels=filters[-1],
            stride=1,
            activation=activation,
            use_bottleneck=use_bottleneck
        ))
        
        return layers
    
    def _create_attention(
        self,
        channels: int,
        attention_type: str
    ) -> nn.Module:
        """
        Create attention module based on type
        
        Args:
            channels: Number of channels
            attention_type: Type of attention ('channel', 'spatial', 'cbam')
        
        Returns:
            Attention module
        """
        if attention_type == 'channel':
            return ChannelAttention3D(channels)
        elif attention_type == 'spatial':
            return SpatialAttention3D()
        elif attention_type == 'cbam':
            return CBAM3D(channels)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
    
    def _create_decoder(
        self,
        fc_dim: int,
        num_outputs: int,
        decoder_type: str,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0
    ) -> nn.Module:
        """
        Create decoder based on type
        
        Args:
            fc_dim: Input feature dimension
            num_outputs: Output sequence length or dimension
            decoder_type: Decoder type ('mlp', 'lstm', 'transformer')
            hidden_dim: Decoder hidden dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads (for transformer)
            dropout: Dropout probability
        
        Returns:
            Decoder module
        
        Raises:
            ValueError: If decoder_type is invalid or incompatible with num_outputs
        """
        if decoder_type == 'mlp':
            return nn.Linear(fc_dim, num_outputs)
        
        elif decoder_type == 'lstm':
            if num_outputs <= 1:
                raise ValueError(
                    f"LSTM decoder requires num_outputs > 1 for sequence prediction, "
                    f"got num_outputs={num_outputs}. Use 'mlp' decoder for single output."
                )
            return SequenceDecoder(
                input_dim=fc_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_seq_len=num_outputs,
                dropout=dropout
            )
        
        elif decoder_type == 'transformer':
            if num_outputs <= 1:
                raise ValueError(
                    f"Transformer decoder requires num_outputs > 1 for sequence prediction, "
                    f"got num_outputs={num_outputs}. Use 'mlp' decoder for single output."
                )
            return TransformerDecoder(
                input_dim=fc_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_seq_len=num_outputs,
                num_heads=num_heads,
                dropout=dropout
            )
        
        else:
            raise ValueError(
                f"Invalid decoder_type: '{decoder_type}'. "
                f"Must be 'mlp', 'lstm', or 'transformer'."
            )
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with intermediate features
        
        Args:
            x: Input tensor (B, 1, D, H, W) or (B, D, H, W)
        
        Returns:
            Dictionary containing:
            - 'layer4': Convolutional features after pooling (B, C, 6, 6, 6)
            - 'layer6': FC features (B, fc_dim)
            - 'output': Final predictions (B, num_outputs)
            - 'intermediate_features': List of features from each encoder layer
        
        Raises:
            ValueError: If input shape is invalid
        """
        # Validate and ensure 5D input: (B, C, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() != 5:
            raise ValueError(
                f"Expected 4D or 5D input tensor, got {x.dim()}D. "
                f"Input shape should be (B, D, H, W) or (B, C, D, H, W)"
            )
        
        # Encoder with feature collection
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            
            # Apply attention if configured
            if self.use_attention and f'attn_{i}' in self.attention_modules:
                x = self.attention_modules[f'attn_{i}'](x)
            
            features.append(x)
        
        # Multi-scale fusion or single-scale
        if self.use_multi_scale and self.fusion is not None:
            fusion_feats = tuple(features[i] for i in self.fusion_layer_indices)
            layer4 = self.fusion(fusion_feats)
        else:
            layer4 = self.adaptive_pool(features[-1])
        
        # FC layers
        x_flat = layer4.view(layer4.size(0), -1)
        layer6 = self.act_fn(self.bn_fc(self.fc1(x_flat)))
        layer6 = self.dropout_layer(layer6)
        
        # Decoder
        output = self.decoder(layer6)
        
        return {
            'layer4': layer4,
            'layer6': layer6,
            'output': output,
            'intermediate_features': features
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, returns only final output
        
        Args:
            x: Input tensor (B, 1, D, H, W) or (B, D, H, W)
        
        Returns:
            Predictions (B, num_outputs)
        """
        return self.forward_features(x)['output']


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
    

