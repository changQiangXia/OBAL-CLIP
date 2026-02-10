#!/usr/bin/env python3
"""
ARO 数据集评估脚本
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import yaml
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List

from src.models.the_architect import TheArchitect
from src.models.detector_wrapper import DetectorWrapper


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_aro_data(json_path: str, image_root: str, image_size: int = 224):
    """加载 ARO 数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
    ])
    
    items = []
    for item in tqdm(data, desc="Loading images"):
        image_path = os.path.join(image_root, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            items.append({
                'image': image,
                'caption_0': item['caption_0'],  # 正确描述
                'caption_1': item['caption_1'],  # 错误描述（属性绑定错误）
            })
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            continue
    
    return items


def evaluate(model, detector, test_data, device):
    """评估属性绑定准确率"""
    model.eval()
    
    correct = 0
    total = 0
    
    print("\nEvaluating...")
    for item in tqdm(test_data):
        image = item['image'].unsqueeze(0).to(device)
        caption_pos = item['caption_0']
        caption_neg = item['caption_1']
        
        with torch.no_grad():
            # 提取区域特征
            region_features, region_mask, _ = detector.extract_features(image)
            region_features = region_features.to(device)
            region_mask = region_mask.to(device)
            
            # 编码
            img_features = model.encode_image(image, region_features, region_mask)
            text_features_pos = model.encode_text(caption_pos)
            text_features_neg = model.encode_text(caption_neg)
            
            # 计算相似度
            sim_pos = (img_features @ text_features_pos.T).item()
            sim_neg = (img_features @ text_features_neg.T).item()
        
        if sim_pos > sim_neg:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy * 100,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--aro_json', type=str, required=True, 
                       help='Path to ARO json file (e.g., coco_order.json)')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Root directory of images (e.g., data/coco/val2017)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"ARO Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.aro_json}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("Loading model...")
    model = TheArchitect(
        clip_model_name=config['model']['clip_model'],
        clip_pretrained=config['model']['clip_pretrained'],
        adapter_config=config['model']['adapter'],
        freeze_clip=True,
        device=device
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 构建检测器
    detector = DetectorWrapper(
        detector_type=config['model']['detector']['type'],
        model_name=config['model']['detector']['model_name'],
        device=config['model']['detector'].get('device', device),
        max_detections=config['model']['detector']['max_detections']
    )
    
    # 加载数据
    print(f"\nLoading ARO data from {args.aro_json}...")
    test_data = load_aro_data(
        args.aro_json, 
        args.image_root,
        config['data']['image_size']
    )
    print(f"[OK] Loaded {len(test_data)} samples")
    
    # 评估
    results = evaluate(model, detector, test_data, device)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"{'='*60}\n")
    
    # 保存结果
    output_path = f"outputs/aro_results_{os.path.basename(args.aro_json).replace('.json', '')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to {output_path}")


if __name__ == "__main__":
    main()
