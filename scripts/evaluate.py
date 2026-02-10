"""
评估脚本
支持在 ComCO, Winoground 等基准上测试属性绑定能力
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import yaml
from tqdm import tqdm
import json
import numpy as np
from typing import Dict, List

from src.models.the_architect import TheArchitect
from src.models.detector_wrapper import DetectorWrapper, DummyDetector
from src.utils.visualization import EvaluationVisualizer


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_binding_accuracy(
    model: TheArchitect,
    detector,
    test_data: List[Dict],
    device: str,
    save_errors: bool = True,
    save_dir: str = "outputs/error_analysis"
) -> Dict[str, float]:
    """
    评估属性绑定准确率
    
    支持两种数据格式：
    1. 评估格式: item['caption_0'], item['caption_1']
    2. 合成格式: item['caption'], item['hard_negatives']
    
    新增: 保存错误案例用于分析
    """
    model.eval()
    
    correct = 0
    total = 0
    error_cases = []  # 保存错误案例
    
    print("Evaluating binding accuracy...")
    
    for idx, item in enumerate(tqdm(test_data)):
        # 加载图像
        image = item['image'].unsqueeze(0).to(device)  # (1, C, H, W)
        
        # 判断数据格式
        if 'caption_0' in item and 'caption_1' in item:
            # 评估数据集格式 (ComCO 等)
            caption_pos = item['caption_0']
            caption_neg = item['caption_1']
        elif 'caption' in item and 'hard_negatives' in item:
            # 合成数据集格式
            caption_pos = item['caption']
            hard_negs = item['hard_negatives']
            if len(hard_negs) == 0:
                continue
            caption_neg = hard_negs[0]  # 使用第一个硬负样本
        else:
            print(f"[WARN] Unknown data format, skipping item")
            continue
        
        # 提取区域特征
        with torch.no_grad():
            region_features, region_mask, _ = detector.extract_features(image)
            region_features = region_features.to(device)
            region_mask = region_mask.to(device)
            
            # 编码图像
            img_features = model.encode_image(image, region_features, region_mask)
            
            # 编码文本
            text_features_pos = model.encode_text(caption_pos)
            text_features_neg = model.encode_text(caption_neg)
        
        # 计算相似度
        sim_pos = (img_features @ text_features_pos.T).item()
        sim_neg = (img_features @ text_features_neg.T).item()
        
        # 判断是否预测正确
        is_correct = sim_pos > sim_neg
        if is_correct:
            correct += 1
        else:
            # 保存错误案例
            error_cases.append({
                'index': idx,
                'caption_pos': caption_pos,
                'caption_neg': caption_neg,
                'sim_pos': sim_pos,
                'sim_neg': sim_neg,
                'margin': sim_pos - sim_neg  # 负值表示预测错误
            })
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # 保存错误案例
    if save_errors and error_cases:
        import json
        os.makedirs(save_dir, exist_ok=True)
        error_file = os.path.join(save_dir, 'binding_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy * 100,
                'correct': correct,
                'total': total,
                'error_cases': error_cases
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Error cases saved to {error_file}")
        
        # 打印部分错误案例
        print("\n=== Error Case Analysis (Sample) ===")
        print(f"Total errors: {len(error_cases)}/{total}")
        print("\nTop 5 most confident errors (largest negative margin):")
        error_cases_sorted = sorted(error_cases, key=lambda x: x['margin'])
        for i, case in enumerate(error_cases_sorted[:5]):
            print(f"\n{i+1}. Margin: {case['margin']:.4f}")
            print(f"   Positive: {case['caption_pos']}")
            print(f"   Negative: {case['caption_neg']}")
            print(f"   Sim (pos): {case['sim_pos']:.4f}, Sim (neg): {case['sim_neg']:.4f}")
    
    return {
        'binding_accuracy': accuracy * 100,
        'correct': correct,
        'total': total
    }


def evaluate_retrieval(
    model: TheArchitect,
    detector,
    images: torch.Tensor,
    texts: List[str],
    device: str
) -> Dict[str, float]:
    """
    评估图像-文本检索性能
    
    计算 Recall@K
    """
    model.eval()
    
    # 提取图像特征
    img_features_list = []
    for i in range(len(images)):
        img = images[i:i+1].to(device)
        
        with torch.no_grad():
            region_features, region_mask, _ = detector.extract_features(img)
            region_features = region_features.to(device)
            region_mask = region_mask.to(device)
            
            img_features = model.encode_image(img, region_features, region_mask)
            img_features_list.append(img_features)
    
    img_features = torch.cat(img_features_list, dim=0)  # (N, C)
    
    # 提取文本特征
    with torch.no_grad():
        text_features = model.encode_text(texts)  # (N, C)
    
    # 计算相似度矩阵
    similarity_matrix = img_features @ text_features.T  # (N, N)
    
    # 计算 Recall@K
    recalls = {}
    for k in [1, 5, 10]:
        if k > len(images):
            continue
        
        # 图像到文本的检索
        i2t_ranks = []
        for i in range(len(images)):
            # 排序后的索引
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            # 找到 ground truth 的 rank
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            i2t_ranks.append(rank)
        
        # 文本到图像的检索
        t2i_ranks = []
        for i in range(len(texts)):
            sorted_indices = torch.argsort(similarity_matrix[:, i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            t2i_ranks.append(rank)
        
        # 计算 Recall@K
        i2t_recall = sum(1 for r in i2t_ranks if r < k) / len(i2t_ranks)
        t2i_recall = sum(1 for r in t2i_ranks if r < k) / len(t2i_ranks)
        
        recalls[f'R@{k}_i2t'] = i2t_recall * 100
        recalls[f'R@{k}_t2i'] = t2i_recall * 100
    
    return recalls


def main():
    parser = argparse.ArgumentParser(description='Evaluate The Architect')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print(f"\n{'='*60}")
    print(f"Evaluation: {config['experiment_name']}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # 设备
    device_setting = config.get('device', 'auto')
    if device_setting == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_setting
    print(f"Using device: {device}")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # 构建模型
    print("\nLoading model...")
    model = TheArchitect(
        clip_model_name=config['model']['clip_model'],
        clip_pretrained=config['model']['clip_pretrained'],
        adapter_config=config['model']['adapter'],
        freeze_clip=True,  # 评估时冻结 CLIP
        device=device
    )
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # 构建检测器
    print("\nBuilding detector...")
    if config['data'].get('use_synthetic', False):
        detector = DummyDetector(
            device=device,
            max_detections=config['model']['detector']['max_detections']
        )
    else:
        detector = DetectorWrapper(
            detector_type=config['model']['detector']['type'],
            model_name=config['model']['detector']['model_name'],
            device=config['model']['detector'].get('device', device),
            max_detections=config['model']['detector']['max_detections']
        )
    
    # 加载测试数据
    if args.test_data:
        print(f"\nLoading test data from {args.test_data}...")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
    else:
        # 使用合成数据进行演示
        print("\nUsing synthetic test data...")
        from src.data.dataset import SyntheticDataset
        dataset = SyntheticDataset(num_samples=20, image_size=config['data']['image_size'])
        test_data = [dataset[i] for i in range(len(dataset))]
    
    # 评估属性绑定准确率
    print("\n" + "="*60)
    print("Evaluating Attribute Binding")
    print("="*60)
    
    binding_results = evaluate_binding_accuracy(model, detector, test_data, device, save_errors=True)
    
    print(f"\nBinding Accuracy Results:")
    print(f"  Accuracy: {binding_results['binding_accuracy']:.2f}%")
    print(f"  Correct: {binding_results['correct']}/{binding_results['total']}")
    
    # 可视化
    visualizer = EvaluationVisualizer(save_dir='outputs/visualizations/eval')
    
    # 保存结果
    results = {
        'binding_accuracy': binding_results,
        'checkpoint': args.checkpoint,
        'config': config['experiment_name']
    }
    
    output_path = 'outputs/eval_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
