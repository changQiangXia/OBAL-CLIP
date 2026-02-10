#!/usr/bin/env python3
"""
SugarCrepe 基准评估脚本
测试属性绑定、对象关系和替换理解能力
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
import numpy as np

from src.models.the_architect import TheArchitect
from src.models.detector_wrapper import DetectorWrapper


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_sugarcrepe_data(json_path: str, image_root: str, image_size: int = 224):
    """加载 SugarCrepe 数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
    ])
    
    items = []
    for key, item in tqdm(data.items(), desc="Loading images"):
        image_path = os.path.join(image_root, item['filename'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            items.append({
                'image': image,
                'caption_pos': item['caption'],
                'caption_neg': item['negative_caption'],
                'filename': item['filename']
            })
        except Exception as e:
            continue
    
    return items


def evaluate(model, detector, test_data, device):
    """评估属性绑定准确率"""
    model.eval()
    
    correct = 0
    total = 0
    error_cases = []
    
    print(f"\nEvaluating {len(test_data)} samples...")
    for idx, item in enumerate(tqdm(test_data, desc="Testing")):
        image = item['image'].unsqueeze(0).to(device)
        caption_pos = item['caption_pos']
        caption_neg = item['caption_neg']
        
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
        
        is_correct = sim_pos > sim_neg
        if is_correct:
            correct += 1
        else:
            error_cases.append({
                'index': idx,
                'filename': item['filename'],
                'caption_pos': caption_pos,
                'caption_neg': caption_neg,
                'sim_pos': sim_pos,
                'sim_neg': sim_neg,
                'margin': sim_pos - sim_neg
            })
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy * 100,
        'correct': correct,
        'total': total,
        'error_cases': error_cases[:20]  # 保存前20个错误案例
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on SugarCrepe Benchmark')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--image_root', type=str, default='data/coco/val2017',
                       help='Root directory of COCO validation images')
    parser.add_argument('--subset', type=str, default='all',
                       choices=['all', 'swap_att', 'swap_obj', 'replace_att', 'replace_obj', 'replace_rel'],
                       help='Which subset to evaluate')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"SugarCrepe Benchmark Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
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
    
    # 定义要评估的子集
    if args.subset == 'all':
        subsets = ['swap_att', 'swap_obj', 'replace_att', 'replace_obj', 'replace_rel']
    else:
        subsets = [args.subset]
    
    # 评估每个子集
    all_results = {}
    overall_correct = 0
    overall_total = 0
    
    for subset in subsets:
        json_path = f'data/sugarcrepe/{subset}.json'
        if not os.path.exists(json_path):
            print(f"⚠️  {json_path} not found, skipping...")
            continue
        
        print(f"\n{'-'*70}")
        print(f"Evaluating: {subset}")
        print(f"{'-'*70}")
        
        # 加载数据
        test_data = load_sugarcrepe_data(json_path, args.image_root, config['data']['image_size'])
        print(f"[OK] Loaded {len(test_data)} valid samples")
        
        if len(test_data) == 0:
            print(f"⚠️  No valid samples for {subset}, skipping...")
            continue
        
        # 评估
        results = evaluate(model, detector, test_data, device)
        all_results[subset] = results
        
        overall_correct += results['correct']
        overall_total += results['total']
        
        print(f"\n{subset} Results:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Correct: {results['correct']}/{results['total']}")
    
    # 计算总体准确率
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Overall Results:")
    print(f"  Accuracy: {overall_accuracy:.2f}%")
    print(f"  Correct: {overall_correct}/{overall_total}")
    print(f"{'='*70}\n")
    
    # 保存结果
    final_results = {
        'overall': {
            'accuracy': overall_accuracy,
            'correct': overall_correct,
            'total': overall_total
        },
        'subsets': all_results,
        'checkpoint': args.checkpoint,
        'config': config['experiment_name']
    }
    
    output_path = f'outputs/sugarcrepe_results_{args.subset}.json'
    os.makedirs('outputs', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"[OK] Results saved to {output_path}")


if __name__ == "__main__":
    main()
