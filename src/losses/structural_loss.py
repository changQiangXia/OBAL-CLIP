"""
Structural Loss 模块
专门惩罚属性与实体的错误绑定
包含 Hard Negative Mining 策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import random
import re


class StructuralLoss(nn.Module):
    """
    Structural Loss
    
    结合:
    1. 标准对比损失 (InfoNCE)
    2. 结构化损失 (专门惩罚属性绑定错误)
    3. Hard Negative Mining
    
    数学公式:
    L_total = λ1 * L_contrastive + λ2 * L_structural
    
    L_contrastive = -log(exp(sim(i,t+)/τ) / Σ_t exp(sim(i,t)/τ))
    
    L_structural = max(0, sim(i,t_hard) - sim(i,t+) + margin)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        structural_weight: float = 1.0,
        margin: float = 0.2,
        hard_negative_ratio: float = 0.5,
        mining_strategy: str = "attribute_swap"
    ):
        super().__init__()
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.structural_weight = structural_weight
        self.margin = margin
        self.hard_negative_ratio = hard_negative_ratio
        self.mining_strategy = mining_strategy
        
        # 属性词表（用于生成硬负样本）
        self.color_words = [
            'red', 'blue', 'green', 'yellow', 'black', 'white', 
            'brown', 'pink', 'purple', 'orange', 'gray', 'grey'
        ]
        self.size_words = ['big', 'small', 'large', 'tiny', 'huge']
        
    def forward(
        self,
        image_features: torch.Tensor,  # (B, C)
        text_features: torch.Tensor,   # (B, C) 正样本
        text_captions: List[str],      # 原始文本，用于生成硬负样本
        logit_scale: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None  # 用于编码硬负样本
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            image_features: 图像特征 (B, C)
            text_features: 正样本文本特征 (B, C)
            text_captions: 原始文本列表，长度 B
            logit_scale: 温度系数的指数
            model: 模型，用于编码硬负样本
        
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量的字典
        """
        B = image_features.shape[0]
        device = image_features.device
        
        if logit_scale is None:
            logit_scale = torch.tensor(1.0 / self.temperature).to(device)
        
        # 1. 标准对比损失
        contrastive_loss = self._contrastive_loss(
            image_features, text_features, logit_scale
        )
        
        # 2. 生成硬负样本
        hard_negatives = self._generate_hard_negatives(text_captions)
        
        # 3. 结构化损失（Triplet-like）
        structural_loss = torch.tensor(0.0).to(device)
        
        if model is not None and len(hard_negatives) > 0:
            structural_loss = self._structural_loss(
                image_features,
                text_features,
                hard_negatives,
                text_captions,
                logit_scale,
                model
            )
        
        # 4. 总损失
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.structural_weight * structural_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'structural': structural_loss.item(),
            'num_hard_negatives': len(hard_negatives) // B if hard_negatives else 0
        }
        
        return total_loss, loss_dict
    
    def _contrastive_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        标准 InfoNCE 对比损失
        
        Args:
            image_features: (B, C)
            text_features: (B, C)
            logit_scale: scalar
        
        Returns:
            loss: scalar
        """
        # 计算相似度矩阵
        logits_per_image = logit_scale * image_features @ text_features.T  # (B, B)
        logits_per_text = logits_per_image.T  # (B, B)
        
        # 标签：对角线是正样本
        B = image_features.shape[0]
        labels = torch.arange(B, device=image_features.device)
        
        # 交叉熵损失
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def _structural_loss(
        self,
        image_features: torch.Tensor,  # (B, C)
        pos_text_features: torch.Tensor,  # (B, C)
        hard_negatives: List[str],     # List of hard negative texts
        original_captions: List[str],  # Original texts
        logit_scale: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        结构化损失
        确保正样本比硬负样本更接近图像
        
        L_structural = max(0, sim(image, neg) - sim(image, pos) + margin)
        """
        device = image_features.device
        B = image_features.shape[0]
        
        # 硬负样本数量（每个正样本对应 K 个硬负样本）
        K = len(hard_negatives) // B if len(hard_negatives) >= B else 0
        
        if K == 0:
            return torch.tensor(0.0).to(device)
        
        # 编码硬负样本
        with torch.no_grad():
            neg_features = model.encode_text(hard_negatives)  # (B*K, C)
        
        loss = torch.tensor(0.0).to(device)
        valid_pairs = 0
        
        for i in range(B):
            img_feat = image_features[i:i+1]  # (1, C)
            pos_feat = pos_text_features[i:i+1]  # (1, C)
            
            # 获取该样本的硬负样本
            start_idx = i * K
            end_idx = start_idx + K
            neg_feats = neg_features[start_idx:end_idx]  # (K, C)
            
            # 计算相似度
            sim_pos = (img_feat @ pos_feat.T).squeeze() * logit_scale  # scalar
            sim_negs = (img_feat @ neg_feats.T).squeeze() * logit_scale  # (K,) or scalar
            
            # 确保 sim_negs 是 1D tensor
            if sim_negs.dim() == 0:
                sim_negs = sim_negs.unsqueeze(0)
            
            # Triplet loss: max(0, neg_sim - pos_sim + margin)
            for sim_neg in sim_negs:
                loss += F.relu(sim_neg - sim_pos + self.margin)
                valid_pairs += 1
        
        if valid_pairs > 0:
            loss = loss / valid_pairs
        
        return loss
    
    def _generate_hard_negatives(self, captions: List[str]) -> List[str]:
        """
        生成硬负样本
        
        策略:
        1. attribute_swap: 交换属性词
        2. entity_swap: 交换实体词
        3. mixed: 混合策略
        """
        hard_negatives = []
        
        for caption in captions:
            negs = []
            
            # 策略 1: 属性交换
            if self.mining_strategy in ["attribute_swap", "mixed"]:
                neg = self._swap_attributes(caption)
                if neg and neg != caption:
                    negs.append(neg)
            
            # 策略 2: 颜色替换
            if self.mining_strategy in ["mixed"]:
                neg = self._replace_color(caption)
                if neg and neg != caption:
                    negs.append(neg)
            
            # 策略 3: 实体交换（简单实现）
            if self.mining_strategy in ["entity_swap", "mixed"]:
                neg = self._swap_entities(caption)
                if neg and neg != caption:
                    negs.append(neg)
            
            # 如果生成的硬负样本不够，复制最后一个
            while len(negs) < max(1, int(self.hard_negative_ratio * 2)):
                negs.append(negs[-1] if negs else caption)
            
            hard_negatives.extend(negs[:max(1, int(self.hard_negative_ratio * 2))])
        
        return hard_negatives
    
    def _swap_attributes(self, caption: str) -> Optional[str]:
        """
        交换属性词
        例如: "red cat and blue dog" -> "blue cat and red dog"
        """
        words = caption.lower().split()
        
        # 找出颜色词的位置
        color_positions = []
        for i, word in enumerate(words):
            clean_word = word.strip(',.!?;')
            if clean_word in self.color_words:
                color_positions.append((i, clean_word))
        
        # 如果有两个或以上颜色词，交换它们
        if len(color_positions) >= 2:
            new_words = words.copy()
            pos1, color1 = color_positions[0]
            pos2, color2 = color_positions[1]
            
            # 交换（保留原始大小写格式）
            new_words[pos1] = words[pos2]
            new_words[pos2] = words[pos1]
            
            return ' '.join(new_words)
        
        return None
    
    def _replace_color(self, caption: str) -> Optional[str]:
        """
        替换颜色词为随机颜色
        例如: "red cat" -> "blue cat"
        """
        words = caption.split()
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip(',.!?;')
            if clean_word in self.color_words:
                # 随机选择一个不同的颜色
                new_color = random.choice([
                    c for c in self.color_words if c != clean_word
                ])
                
                # 保留原始格式
                if word[0].isupper():
                    new_color = new_color.capitalize()
                
                words[i] = new_color + word[len(clean_word):]
                return ' '.join(words)
        
        return None
    
    def _swap_entities(self, caption: str) -> Optional[str]:
        """
        交换实体位置
        例如: "cat on dog" -> "dog on cat"
        这是一个简化版本
        """
        # 简单的实体交换：寻找 "and" 连接的短语
        if ' and ' in caption.lower():
            parts = caption.split(' and ')
            if len(parts) == 2:
                # 找到冠词/限定词的位置
                def find_noun_phrase(text):
                    words = text.split()
                    # 跳过开头的冠词/限定词
                    start = 0
                    while start < len(words) and words[start].lower() in ['a', 'an', 'the', 'one', 'two']:
                        start += 1
                    
                    # 跳过形容词
                    while start < len(words) and words[start].lower() in self.color_words + self.size_words:
                        start += 1
                    
                    return ' '.join(words[start:]) if start < len(words) else text
                
                # 交换
                return parts[1] + ' and ' + parts[0]
        
        return None


class HardNegativeMiner:
    """
    硬负样本挖掘器
    支持离线生成和动态挖掘
    """
    
    def __init__(
        self,
        strategies: List[str] = ["attribute_swap", "color_replace", "entity_swap"],
        max_negatives_per_sample: int = 5
    ):
        self.strategies = strategies
        self.max_negatives_per_sample = max_negatives_per_sample
        
        # 属性词典
        self.attribute_dict = {
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                      'brown', 'pink', 'purple', 'orange', 'gray'],
            'sizes': ['big', 'small', 'large', 'tiny', 'huge', 'tall', 'short'],
            'materials': ['wooden', 'metal', 'plastic', 'glass', 'leather'],
        }
    
    def generate_hard_negatives(self, captions: List[str]) -> List[List[str]]:
        """
        为每个 caption 生成硬负样本列表
        
        Args:
            captions: 文本列表
        
        Returns:
            List[List[str]]: 每个 caption 对应的硬负样本列表
        """
        results = []
        
        for caption in captions:
            negatives = set()
            
            for strategy in self.strategies:
                if len(negatives) >= self.max_negatives_per_sample:
                    break
                
                if strategy == "attribute_swap":
                    neg = self._attribute_swap(caption)
                    if neg:
                        negatives.add(neg)
                
                elif strategy == "color_replace":
                    for _ in range(2):  # 生成多个颜色替换版本
                        neg = self._color_replace(caption)
                        if neg:
                            negatives.add(neg)
                        if len(negatives) >= self.max_negatives_per_sample:
                            break
                
                elif strategy == "entity_swap":
                    neg = self._entity_swap(caption)
                    if neg:
                        negatives.add(neg)
                
                elif strategy == "negation":
                    neg = self._negation(caption)
                    if neg:
                        negatives.add(neg)
            
            # 确保至少有一个硬负样本
            if not negatives:
                negatives.add(caption)  #  fallback
            
            results.append(list(negatives)[:self.max_negatives_per_sample])
        
        return results
    
    def _attribute_swap(self, caption: str) -> Optional[str]:
        """交换属性"""
        all_attrs = self.attribute_dict['colors'] + self.attribute_dict['sizes']
        words = caption.split()
        
        attr_positions = []
        for i, word in enumerate(words):
            clean = word.lower().strip(',.!?;')
            if clean in all_attrs:
                attr_positions.append((i, word, clean))
        
        if len(attr_positions) >= 2:
            new_words = words.copy()
            i1, w1, _ = attr_positions[0]
            i2, w2, _ = attr_positions[1]
            new_words[i1] = w2
            new_words[i2] = w1
            return ' '.join(new_words)
        
        return None
    
    def _color_replace(self, caption: str) -> Optional[str]:
        """替换颜色"""
        words = caption.split()
        colors = self.attribute_dict['colors']
        
        for i, word in enumerate(words):
            clean = word.lower().strip(',.!?;')
            if clean in colors:
                new_color = random.choice([c for c in colors if c != clean])
                if word[0].isupper():
                    new_color = new_color.capitalize()
                words[i] = new_color + word[len(clean):]
                return ' '.join(words)
        
        return None
    
    def _entity_swap(self, caption: str) -> Optional[str]:
        """交换实体"""
        if ' and ' in caption.lower():
            parts = caption.split(' and ')
            if len(parts) == 2:
                return parts[1] + ' and ' + parts[0]
        return None
    
    def _negation(self, caption: str) -> Optional[str]:
        """添加否定"""
        return f"not {caption}"


if __name__ == "__main__":
    print("Testing Structural Loss...")
    
    # 创建损失函数
    loss_fn = StructuralLoss(
        temperature=0.07,
        contrastive_weight=1.0,
        structural_weight=0.5,
        margin=0.2,
        hard_negative_ratio=0.5,
        mining_strategy="mixed"
    )
    
    # 模拟数据
    B, C = 4, 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_features = torch.randn(B, C).to(device)
    image_features = F.normalize(image_features, dim=-1)
    
    text_features = torch.randn(B, C).to(device)
    text_features = F.normalize(text_features, dim=-1)
    
    captions = [
        "a red cat and a blue dog",
        "a big elephant and a small mouse",
        "a green apple on a wooden table",
        "a black car and a white bicycle"
    ]
    
    # 测试硬负样本生成
    print("\nGenerating hard negatives:")
    hard_negs = loss_fn._generate_hard_negatives(captions)
    for cap, negs in zip(captions, [hard_negs[i:i+2] for i in range(0, len(hard_negs), 2)]):
        print(f"  Original: {cap}")
        for neg in negs[:2]:
            print(f"    -> {neg}")
    
    # 测试损失计算
    print("\nComputing loss:")
    
    # 创建简单的 mock model
    class MockModel(nn.Module):
        def encode_text(self, texts):
            return torch.randn(len(texts), C).to(device)
    
    mock_model = MockModel()
    
    loss, loss_dict = loss_fn(
        image_features,
        text_features,
        captions,
        logit_scale=torch.tensor(1/0.07).to(device),
        model=mock_model
    )
    
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  Contrastive Loss: {loss_dict['contrastive']:.4f}")
    print(f"  Structural Loss: {loss_dict['structural']:.4f}")
    print(f"  Num Hard Negatives: {loss_dict['num_hard_negatives']}")
    
    print("\n✓ Structural Loss test passed!")
