"""
Object-Aware Adapter 模块
在冻结的 CLIP 视觉编码器后引入，用于融合全局特征和物体区域特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention 模块
    Query: CLIP 全局 [CLS] token
    Key/Value: 物体区域特征
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q 来自 CLS token, K/V 来自区域特征
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,  # (B, N_q, C) - CLS token
        key_value: torch.Tensor,  # (B, N_kv, C) - 区域特征
        attention_mask: Optional[torch.Tensor] = None  # (B, N_q, N_kv)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (B, N_q, C)
            key_value: Key/Value tensor (B, N_kv, C)
            attention_mask: Optional mask (B, N_q, N_kv)
        
        Returns:
            output: (B, N_q, C)
            attention_weights: (B, num_heads, N_q, N_kv)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # 线性投影
        Q = self.q_proj(query)  # (B, N_q, C)
        K = self.k_proj(key_value)  # (B, N_kv, C)
        V = self.v_proj(key_value)  # (B, N_kv, C)
        
        # 重塑为多头形式
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, D)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_kv, D)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_kv, D)
        
        # 计算注意力
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)
        
        # 应用 mask
        if attention_mask is not None:
            # mask: (B, N_q, N_kv) -> (B, 1, N_q, N_kv)
            attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = attn @ V  # (B, H, N_q, D)
        output = output.transpose(1, 2).contiguous().view(B, N_q, C)  # (B, N_q, C)
        output = self.out_proj(output)
        
        return output, attn


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AdapterBlock(nn.Module):
    """
    Adapter 基础块: Cross-Attention + FeedForward + LayerNorm + Residual
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = FeedForward(dim, hidden_dim, dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, N_q, C) - 通常是 CLS token
            key_value: (B, N_kv, C) - 区域特征
            attention_mask: (B, N_q, N_kv)
        
        Returns:
            output: (B, N_q, C)
            attention_weights: (B, H, N_q, N_kv)
        """
        # Cross-Attention with residual
        attn_out, attn_weights = self.cross_attn(
            self.norm1(query), key_value, attention_mask
        )
        query = query + attn_out
        
        # FeedForward with residual
        query = query + self.ffn(self.norm2(query))
        
        return query, attn_weights


class ObjectAwareAdapter(nn.Module):
    """
    Object-Aware Adapter 主模块
    多层 Cross-Attention 堆叠，融合全局特征和物体区域特征
    """
    
    def __init__(
        self,
        clip_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_objects: int = 20
    ):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # 投影层：将 CLIP 特征投影到 Adapter 维度
        self.query_proj = nn.Linear(clip_dim, hidden_dim)
        self.kv_proj = nn.Linear(clip_dim, hidden_dim)
        
        # 堆叠多个 Adapter Block
        self.layers = nn.ModuleList([
            AdapterBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影：将融合后的特征映射回 CLIP 维度
        self.output_proj = nn.Linear(hidden_dim, clip_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        cls_token: torch.Tensor,  # (B, C) - CLIP 全局特征
        region_features: torch.Tensor,  # (B, N, C) - 区域特征
        region_mask: Optional[torch.Tensor] = None  # (B, N) - 有效区域掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_token: CLIP 全局 CLS token (B, C)
            region_features: 物体区域特征 (B, N, C)
            region_mask: 有效区域掩码 (B, N)，1 表示有效，0 表示无效
        
        Returns:
            enhanced_feature: 增强后的全局特征 (B, C)
            attention_weights: 最后一层的注意力权重 (B, H, 1, N)
        """
        B = cls_token.shape[0]
        
        # 将 CLS token 扩展为序列形式 (B, 1, C)
        query = cls_token.unsqueeze(1)  # (B, 1, C)
        
        # 投影到 Adapter 维度
        query = self.query_proj(query)  # (B, 1, hidden_dim)
        key_value = self.kv_proj(region_features)  # (B, N, hidden_dim)
        
        # 构建 attention mask
        attention_mask = None
        if region_mask is not None:
            # mask: (B, N) -> (B, 1, N)
            attention_mask = region_mask.unsqueeze(1)  # (B, 1, N)
        
        # 通过多层 Adapter
        all_attention_weights = []
        for layer in self.layers:
            query, attn_weights = layer(query, key_value, attention_mask)
            all_attention_weights.append(attn_weights)
        
        # 最终归一化和投影
        query = self.norm(query)
        output = self.output_proj(query)  # (B, 1, clip_dim)
        
        # 残差连接：融合原始 CLS token
        enhanced_feature = cls_token.unsqueeze(1) + output
        enhanced_feature = enhanced_feature.squeeze(1)  # (B, C)
        
        # 返回最后一层的注意力权重用于可视化
        final_attention = all_attention_weights[-1]  # (B, H, 1, N)
        
        return enhanced_feature, final_attention


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    用于高效微调 CLIP
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征
            original_output: 原始线性层的输出
        
        Returns:
            output with LoRA: original_output + dropout(x @ A @ B) * scaling
        """
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return original_output + lora_output


if __name__ == "__main__":
    # 简单测试
    print("Testing ObjectAwareAdapter...")
    
    B, N, C = 2, 10, 512
    cls_token = torch.randn(B, C)
    region_features = torch.randn(B, N, C)
    region_mask = torch.ones(B, N)
    region_mask[0, 5:] = 0  # 第一个样本只有 5 个有效区域
    
    adapter = ObjectAwareAdapter(
        clip_dim=C,
        hidden_dim=256,
        num_heads=4,
        num_layers=2
    )
    
    enhanced, attn = adapter(cls_token, region_features, region_mask)
    
    print(f"Input CLS token: {cls_token.shape}")
    print(f"Region features: {region_features.shape}")
    print(f"Enhanced feature: {enhanced.shape}")
    print(f"Attention weights: {attn.shape}")
    print("✓ Adapter test passed!")
