"""
The Architect - 完整模型
集成 CLIP、物体检测器和 Object-Aware Adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from .adapter import ObjectAwareAdapter
from .model_loader import load_clip_model


class TheArchitect(nn.Module):
    """
    The Architect 主模型
    
    Architecture:
        Image -> CLIP Visual Encoder -> [CLS] token
                                      |
                                      v
        Image -> Object Detector -> RoI Features -> Projection
                                      |
                                      v
                    ObjectAwareAdapter (Cross-Attention)
                                      |
                                      v
                        Enhanced Visual Feature
                                      |
                                      v
                    Contrastive Learning with Text
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        adapter_config: Optional[Dict] = None,
        freeze_clip: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.freeze_clip = freeze_clip
        
        # 加载 CLIP 模型
        self.clip_model, self.preprocess, self.tokenizer = load_clip_model(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device
        )
        
        # 获取 CLIP 维度
        if hasattr(self.clip_model.visual, 'output_dim'):
            self.clip_dim = self.clip_model.visual.output_dim
        else:
            self.clip_dim = 512  # ViT-B/32 default
        
        # 冻结 CLIP
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("[OK] CLIP parameters frozen.")
        
        # 默认 Adapter 配置
        if adapter_config is None:
            adapter_config = {
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'max_objects': 20
            }
        
        # 初始化 Object-Aware Adapter
        self.adapter = ObjectAwareAdapter(
            clip_dim=self.clip_dim,
            hidden_dim=adapter_config.get('hidden_dim', 512),
            num_heads=adapter_config.get('num_heads', 8),
            num_layers=adapter_config.get('num_layers', 4),
            dropout=adapter_config.get('dropout', 0.1),
            max_objects=adapter_config.get('max_objects', 20)
        )
        
        # 区域特征投影层（将检测器特征映射到 CLIP 维度）
        self.region_proj = nn.Sequential(
            nn.Linear(2048, self.clip_dim),
            nn.LayerNorm(self.clip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_dim, self.clip_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        # 将整个模型移动到指定设备
        self.to(device)
        print(f"[OK] Model moved to {device}")
        
    def encode_image(
        self,
        images: torch.Tensor,
        region_features: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码图像
        
        Args:
            images: (B, 3, H, W) 预处理后的图像
            region_features: (B, N, D) 检测器提取的区域特征，可选
            region_mask: (B, N) 区域掩码，可选
            return_attention: 是否返回注意力权重
        
        Returns:
            image_features: (B, C) 增强后的图像特征
            attention_weights: (B, H, 1, N) 可选
        """
        B = images.shape[0]
        
        # 确保所有输入都在同一设备
        device = images.device
        if region_features is not None:
            region_features = region_features.to(device)
        if region_mask is not None:
            region_mask = region_mask.to(device)
        
        # CLIP 视觉编码
        with torch.set_grad_enabled(not self.freeze_clip):
            visual_features = self.clip_model.visual(images)  # (B, C)
        
        # 如果没有区域特征，直接返回 CLIP 特征
        if region_features is None:
            if return_attention:
                return visual_features, None
            return visual_features
        
        # 投影区域特征到 CLIP 维度
        region_features = self.region_proj(region_features)  # (B, N, C)
        
        # 通过 Adapter 融合
        enhanced_features, attention_weights = self.adapter(
            visual_features,
            region_features,
            region_mask
        )
        
        # 归一化
        enhanced_features = F.normalize(enhanced_features, dim=-1)
        
        if return_attention:
            return enhanced_features, attention_weights
        return enhanced_features
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本
        
        Args:
            text: 单个文本字符串或字符串列表
        
        Returns:
            text_features: (B, C) 文本特征
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        tokens = self.tokenizer(text).to(self.device)  # (B, L)
        
        # 编码
        text_features = self.clip_model.encode_text(tokens)  # (B, C)
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        region_features: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: (B, 3, H, W)
            texts: 文本列表，长度为 B
            region_features: (B, N, D) 可选
            region_mask: (B, N) 可选
        
        Returns:
            dict with keys:
                - image_features: (B, C)
                - text_features: (B, C)
                - logit_scale: scalar
                - attention_weights: (B, H, 1, N) 如果有 region_features
        """
        # 编码图像（带注意力权重）
        if region_features is not None:
            image_features, attention_weights = self.encode_image(
                images, region_features, region_mask, return_attention=True
            )
        else:
            image_features = self.encode_image(images)
            attention_weights = None
        
        # 编码文本
        text_features = self.encode_text(texts)
        
        output = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp()
        }
        
        if attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        return output
    
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            image_features: (B, C) 或 (N_i, C)
            text_features: (B, C) 或 (N_t, C)
        
        Returns:
            logits: (N_i, N_t)
        """
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        return logits


if __name__ == "__main__":
    print("Testing TheArchitect model...")
    
    # 测试配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建模型
    model = TheArchitect(
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
        adapter_config={
            'hidden_dim': 256,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        },
        freeze_clip=True,
        device=device
    )
    
    # 测试输入
    B = 2
    images = torch.randn(B, 3, 224, 224).to(device)
    texts = ["a red cat and a blue dog", "a person riding a bicycle"]
    
    # 模拟区域特征
    N = 10
    region_features = torch.randn(B, N, 2048).to(device)
    region_mask = torch.ones(B, N).to(device)
    region_mask[0, 5:] = 0  # 第一个样本只有 5 个有效区域
    
    # 前向传播
    output = model(images, texts, region_features, region_mask)
    
    print(f"\nModel outputs:")
    print(f"  Image features: {output['image_features'].shape}")
    print(f"  Text features: {output['text_features'].shape}")
    print(f"  Logit scale: {output['logit_scale'].item():.4f}")
    print(f"  Attention weights: {output['attention_weights'].shape}")
    
    # 计算相似度
    logits = model.compute_similarity(
        output['image_features'],
        output['text_features']
    )
    print(f"  Similarity logits: {logits.shape}")
    
    print("\n[OK] Model test passed!")
