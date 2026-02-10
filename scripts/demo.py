"""
演示脚本
快速测试模型推理和可视化功能
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import yaml
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from src.models.the_architect import TheArchitect
from src.models.detector_wrapper import DummyDetector
from src.utils.visualization import EvaluationVisualizer


def load_config(config_path: str):
    """加载配置"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def tensor_to_pil(tensor):
    """将 tensor 转换为 PIL Image"""
    # 反归一化
    mean = torch.tensor([0.4814, 0.4578, 0.4082]).view(3, 1, 1)
    std = torch.tensor([0.2686, 0.2613, 0.2758]).view(3, 1, 1)
    
    img = tensor.cpu().clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    to_pil = T.ToPILImage()
    return to_pil(img)


def demo_inference(model, detector, device):
    """演示推理"""
    print("\n" + "="*60)
    print("Demo: Inference")
    print("="*60)
    
    model.eval()
    
    # 创建测试样本
    test_cases = [
        {
            'image': torch.randn(1, 3, 224, 224),
            'caption': "a red cat and a blue dog",
            'hard_negs': ["a blue cat and a red dog", "a red dog and a blue cat"]
        },
        {
            'image': torch.randn(1, 3, 224, 224),
            'caption': "a big elephant and a small mouse",
            'hard_negs': ["a small elephant and a big mouse"]
        },
        {
            'image': torch.randn(1, 3, 224, 224),
            'caption': "a green apple on a wooden table",
            'hard_negs': ["a wooden apple on a green table"]
        }
    ]
    
    visualizer = EvaluationVisualizer(save_dir='outputs/visualizations/demo')
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Caption: {test_case['caption']}")
        
        image = test_case['image'].to(device)
        caption = test_case['caption']
        hard_negs = test_case['hard_negs']
        
        # 提取区域特征
        with torch.no_grad():
            region_features, region_mask, _ = detector.extract_features(image)
            region_features = region_features.to(device)
            region_mask = region_mask.to(device)
            
            # 前向传播
            output = model(
                image,
                [caption] + hard_negs,
                region_features,
                region_mask
            )
            
            img_feat = output['image_features'][0]  # (C,)
            text_feat_pos = output['text_features'][0]  # (C,)
            text_feat_negs = output['text_features'][1:]  # (K, C)
            
            # 计算相似度
            sim_pos = (img_feat @ text_feat_pos).item()
            sim_negs = [(img_feat @ feat).item() for feat in text_feat_negs]
            
            print(f"  Similarity to positive: {sim_pos:.4f}")
            for j, (neg, sim) in enumerate(zip(hard_negs, sim_negs)):
                print(f"  Similarity to negative {j+1}: {sim:.4f}")
            
            # 判断是否预测正确
            is_correct = all(sim_pos > sim_neg for sim_neg in sim_negs)
            print(f"  Prediction: {'✓ Correct' if is_correct else '✗ Wrong'}")
            
            # 获取注意力权重
            if 'attention_weights' in output:
                attn = output['attention_weights'][0]  # (H, 1, N)
                attn_mean = attn.mean(dim=0).squeeze().cpu().numpy()  # (N,)
                print(f"  Attention weights (mean): {attn_mean[:5]}")  # 显示前5个
        
        results.append({
            'image': tensor_to_pil(image[0]),
            'caption': caption,
            'hard_negs': hard_negs,
            'prediction': caption if is_correct else hard_negs[0]
        })
    
    # 可视化定性结果
    images = [r['image'] for r in results]
    captions = [(r['caption'], r['hard_negs'][0], r['prediction']) for r in results]
    
    visualizer.plot_qualitative_results(images, captions, "demo_qualitative.png")
    print("\n✓ Qualitative results saved to outputs/visualizations/demo/demo_qualitative.png")


def demo_attention_visualization(model, detector, device):
    """演示注意力可视化"""
    print("\n" + "="*60)
    print("Demo: Attention Visualization")
    print("="*60)
    
    model.eval()
    visualizer = EvaluationVisualizer(save_dir='outputs/visualizations/demo')
    
    # 创建一个测试样本
    image = torch.randn(1, 3, 224, 224).to(device)
    caption = "a red cat and a blue dog"
    
    # 模拟物体标签（实际中来自检测器）
    object_labels = ["region_1", "region_2", "region_3", "region_4", "region_5"]
    
    with torch.no_grad():
        region_features, region_mask, _ = detector.extract_features(image)
        region_features = region_features.to(device)
        region_mask = region_mask.to(device)
        
        output = model(image, [caption], region_features, region_mask)
        
        if 'attention_weights' in output:
            attn = output['attention_weights'][0]  # (H, 1, N)
            attn = attn.squeeze(1).cpu().numpy()  # (H, N)
            
            # 可视化（使用 TrainingVisualizer 的 plot_attention_heatmap）
            from src.utils.visualization import TrainingVisualizer
            train_viz = TrainingVisualizer(save_dir='outputs/visualizations/demo')
            train_viz.plot_attention_heatmap(
                attn,
                object_labels[:attn.shape[1]],
                caption,
                "demo_attention_heatmap.png"
            )
            print("\n✓ Attention heatmap saved to outputs/visualizations/demo/demo_attention_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description='Demo The Architect')
    parser.add_argument('--config', type=str, default='configs/debug_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (optional)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print(f"\n{'='*60}")
    print(f"The Architect - Demo")
    print(f"{'='*60}")
    
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
    print("\nBuilding model...")
    model = TheArchitect(
        clip_model_name=config['model']['clip_model'],
        clip_pretrained=config['model']['clip_pretrained'],
        adapter_config=config['model']['adapter'],
        freeze_clip=True,
        device=device
    )
    
    # 加载检查点（如果有）
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("✓ Using randomly initialized Adapter (CLIP frozen)")
    
    # 构建检测器
    print("\nBuilding detector...")
    detector = DummyDetector(
        device=device,
        max_detections=config['model']['detector']['max_detections']
    )
    
    # 运行演示
    demo_inference(model, detector, device)
    demo_attention_visualization(model, detector, device)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
