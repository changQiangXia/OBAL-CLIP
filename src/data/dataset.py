"""
Dataset 类
支持合成数据（Debug 模式）和真实数据集
包含 Hard Negative Mining
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import random
from PIL import Image
import torchvision.transforms as T


class SyntheticDataset(Dataset):
    """
    合成数据集（用于 Debug 模式）
    无需真实图像，生成随机图像和对应的文本描述
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 224,
        hard_negative_ratio: float = 0.5
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.hard_negative_ratio = hard_negative_ratio
        
        # 合成数据的词库
        self.colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'purple']
        self.objects = ['cat', 'dog', 'bird', 'car', 'tree', 'flower', 'house', 'person']
        self.actions = ['sitting', 'standing', 'running', 'flying', 'jumping']
        
        # 预处理（对于合成数据，输入已经是 Tensor，不需要 ToTensor）
        self.transform = T.Normalize(
            mean=[0.4814, 0.4578, 0.4082], 
            std=[0.2686, 0.2613, 0.2758]
        )
        
        # 生成合成数据
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Dict]:
        """生成合成数据"""
        data = []
        for i in range(self.num_samples):
            # 生成描述
            caption = self._generate_caption()
            
            data.append({
                'id': i,
                'caption': caption,
                # 随机种子确保可复现
                'seed': i
            })
        return data
    
    def _generate_caption(self) -> str:
        """生成随机描述"""
        templates = [
            "a {color1} {obj1} and a {color2} {obj2}",
            "a {color1} {obj1} {action}",
            "a {color1} {obj1} next to a {color2} {obj2}",
            "a {color1} {obj1} on a {color2} {obj2}",
            "a {action} {color1} {obj1}",
        ]
        
        template = random.choice(templates)
        
        color1 = random.choice(self.colors)
        color2 = random.choice([c for c in self.colors if c != color1])
        obj1 = random.choice(self.objects)
        obj2 = random.choice([o for o in self.objects if o != obj1])
        action = random.choice(self.actions)
        
        caption = template.format(
            color1=color1,
            color2=color2,
            obj1=obj1,
            obj2=obj2,
            action=action
        )
        
        return caption
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 使用种子确保图像可复现
        torch.manual_seed(item['seed'])
        
        # 生成随机图像（模拟真实图像）
        img = torch.randn(3, self.image_size, self.image_size)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
        img = self.transform(img)
        
        # 生成硬负样本
        hard_negatives = self._generate_hard_negatives(item['caption'])
        
        return {
            'image': img,
            'caption': item['caption'],
            'hard_negatives': hard_negatives,
            'id': item['id']
        }
    
    def _generate_hard_negatives(self, caption: str) -> List[str]:
        """生成硬负样本"""
        hard_negs = []
        
        # 策略 1: 颜色交换
        words = caption.split()
        color_positions = []
        for i, word in enumerate(words):
            clean = word.lower().strip(',.')
            if clean in self.colors:
                color_positions.append((i, word))
        
        if len(color_positions) >= 2:
            new_words = words.copy()
            i1, w1 = color_positions[0]
            i2, w2 = color_positions[1]
            new_words[i1] = w2
            new_words[i2] = w1
            hard_negs.append(' '.join(new_words))
        
        # 策略 2: 单个颜色替换
        if color_positions:
            for _, orig_color in color_positions[:1]:  # 只替换第一个颜色
                new_color = random.choice([c for c in self.colors if c != orig_color.lower()])
                new_caption = caption.replace(orig_color, new_color, 1)
                if new_caption != caption:
                    hard_negs.append(new_caption)
                    break
        
        # 如果没有生成任何硬负样本，添加一个默认的
        if not hard_negs:
            hard_negs.append(caption)
        
        return hard_negs if hard_negs else [caption]


class COCODataset(Dataset):
    """
    COCO 数据集
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        max_captions_per_image: int = 5
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.max_captions_per_image = max_captions_per_image
        
        # 预处理
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
        ])
        
        # 加载 COCO 标注
        self.data = self._load_coco()
    
    def _load_coco(self) -> List[Dict]:
        """加载 COCO 数据"""
        try:
            from pycocotools.coco import COCO
        except ImportError:
            print("Warning: pycocotools not installed. Please install: pip install pycocotools")
            return []
        
        ann_file = f"{self.root_dir}/annotations/captions_{self.split}2017.json"
        
        try:
            coco = COCO(ann_file)
        except FileNotFoundError:
            print(f"Warning: COCO annotations not found at {ann_file}")
            return []
        
        data = []
        img_ids = coco.getImgIds()
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            captions = [ann['caption'] for ann in anns[:self.max_captions_per_image]]
            
            data.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'captions': captions
            })
        
        print(f"Loaded {len(data)} images from COCO {self.split}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 加载图像
        img_path = f"{self.root_dir}/{self.split}2017/{item['file_name']}"
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像加载失败，返回随机图像
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        
        image = self.transform(image)
        
        # 随机选择一个 caption
        caption = random.choice(item['captions'])
        
        return {
            'image': image,
            'caption': caption,
            'image_id': item['image_id']
        }


class AttributeBindingDataset(Dataset):
    """
    属性绑定评估数据集（如 ComCO, Winoground）
    """
    
    def __init__(
        self,
        data_file: str,
        image_size: int = 224
    ):
        self.image_size = image_size
        
        # 预处理
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
        ])
        
        # 加载数据
        self.data = self._load_data(data_file)
    
    def _load_data(self, data_file: str) -> List[Dict]:
        """加载评估数据"""
        import json
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Data file not found: {data_file}")
            return []
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'caption_0': item['caption_0'],  # 正确描述
            'caption_1': item['caption_1'],  # 错误描述（属性绑定错误）
            'label': item.get('label', 0),   # 0 表示 caption_0 正确
            'image_id': item.get('id', idx)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 collate function
    
    处理变长的 hard negatives
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    # 收集所有 hard negatives
    all_hard_negs = []
    for item in batch:
        if 'hard_negatives' in item:
            all_hard_negs.extend(item['hard_negatives'])
    
    result = {
        'images': images,
        'captions': captions,
    }
    
    if all_hard_negs:
        result['hard_negatives'] = all_hard_negs
    
    # 可选字段
    if 'image_id' in batch[0]:
        result['image_ids'] = [item.get('image_id', item.get('id')) for item in batch]
    
    return result


if __name__ == "__main__":
    print("Testing Datasets...")
    
    # 测试合成数据集
    print("\n1. Testing SyntheticDataset:")
    dataset = SyntheticDataset(num_samples=10, image_size=224)
    print(f"   Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Caption: {sample['caption']}")
    print(f"   Hard negatives: {sample['hard_negatives']}")
    
    # 测试 DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\n   Batch images shape: {batch['images'].shape}")
    print(f"   Batch captions: {batch['captions']}")
    print(f"   Batch hard negatives: {len(batch.get('hard_negatives', []))}")
    
    print("\n✓ Dataset tests passed!")
