"""
环境测试脚本
验证所有依赖是否正确安装，模型能否正常运行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """测试所有必要的导入"""
    print("\n" + "="*60)
    print("1. Testing Imports")
    print("="*60)
    
    errors = []
    
    try:
        import torch
        cuda_info = f" (CUDA: {torch.cuda.is_available()})" if torch.cuda.is_available() else " (CPU only)"
        print(f"  ✓ torch {torch.__version__}{cuda_info}")
    except ImportError as e:
        errors.append(("torch", e))
        print(f"  ✗ torch: {e}")
    
    try:
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        errors.append(("torchvision", e))
        print(f"  ✗ torchvision: {e}")
    
    try:
        import open_clip
        print(f"  ✓ open_clip_torch")
    except ImportError as e:
        errors.append(("open_clip", e))
        print(f"  ✗ open_clip: {e}")
    
    try:
        from ultralytics import YOLO
        print(f"  ✓ ultralytics (YOLOv8)")
    except ImportError as e:
        errors.append(("ultralytics", e))
        print(f"  ✗ ultralytics: {e}")
    
    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(("matplotlib", e))
        print(f"  ✗ matplotlib: {e}")
    
    try:
        import seaborn
        print(f"  ✓ seaborn {seaborn.__version__}")
    except ImportError as e:
        errors.append(("seaborn", e))
        print(f"  ✗ seaborn: {e}")
    
    try:
        import yaml
        print(f"  ✓ pyyaml")
    except ImportError as e:
        errors.append(("pyyaml", e))
        print(f"  ✗ pyyaml: {e}")
    
    if errors:
        print(f"\n  Missing {len(errors)} package(s). Please run: pip install -r requirements.txt")
        return False
    
    print("\n  ✓ All imports successful!")
    return True


def test_models():
    """测试模型加载"""
    print("\n" + "="*60)
    print("2. Testing Model Loading")
    print("="*60)
    
    try:
        from src.models.the_architect import TheArchitect
        from src.models.adapter import ObjectAwareAdapter
        from src.models.detector_wrapper import DummyDetector
        
        print("  ✓ Model modules imported")
        
        # 测试模型创建
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        
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
        print("  ✓ TheArchitect model created")
        
        detector = DummyDetector(device=device, max_detections=10)
        print("  ✓ DummyDetector created")
        
        # 测试前向传播
        print("\n  Testing forward pass...")
        B = 2
        images = torch.randn(B, 3, 224, 224).to(device)
        texts = ["a red cat", "a blue dog"]
        
        region_features, region_mask, _ = detector.extract_features(images)
        
        with torch.no_grad():
            output = model(images, texts, region_features, region_mask)
        
        print(f"    Image features: {output['image_features'].shape}")
        print(f"    Text features: {output['text_features'].shape}")
        print(f"    Attention weights: {output['attention_weights'].shape}")
        
        print("\n  ✓ Model forward pass successful!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """测试损失函数"""
    print("\n" + "="*60)
    print("3. Testing Loss Function")
    print("="*60)
    
    try:
        from src.losses.structural_loss import StructuralLoss
        import torch.nn.functional as F
        
        loss_fn = StructuralLoss(
            temperature=0.07,
            contrastive_weight=1.0,
            structural_weight=0.5
        )
        print("  ✓ StructuralLoss created")
        
        # 测试损失计算
        B, C = 4, 512
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        image_features = torch.randn(B, C).to(device)
        text_features = torch.randn(B, C).to(device)
        
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        captions = [
            "a red cat and a blue dog",
            "a big elephant and a small mouse",
            "a green apple on a table",
            "a black car and a white bicycle"
        ]
        
        # 创建 mock model
        class MockModel:
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
        
        print(f"  Loss: {loss_dict['total']:.4f}")
        print(f"  Contrastive: {loss_dict['contrastive']:.4f}")
        print(f"  Structural: {loss_dict['structural']:.4f}")
        
        print("\n  ✓ Loss computation successful!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """测试数据集"""
    print("\n" + "="*60)
    print("4. Testing Dataset")
    print("="*60)
    
    try:
        from src.data.dataset import SyntheticDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # 设置随机种子确保可复现
        torch.manual_seed(42)
        
        dataset = SyntheticDataset(num_samples=10, image_size=224)
        print(f"  ✓ SyntheticDataset created with {len(dataset)} samples")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"  Sample image shape: {sample['image'].shape}")
        print(f"  Sample caption: {sample['caption']}")
        print(f"  Hard negatives: {sample['hard_negatives']}")
        
        # 测试 DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        
        print(f"\n  Batch images shape: {batch['images'].shape}")
        print(f"  Batch captions: {batch['captions']}")
        
        print("\n  ✓ Dataset test successful!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """测试可视化"""
    print("\n" + "="*60)
    print("5. Testing Visualization")
    print("="*60)
    
    try:
        from src.utils.visualization import TrainingVisualizer
        import numpy as np
        
        visualizer = TrainingVisualizer(save_dir='outputs/visualizations/test')
        
        # 模拟训练数据
        for epoch in range(5):
            metrics = {
                'train_loss': 2.0 * np.exp(-epoch * 0.3) + 0.1,
                'val_loss': 2.1 * np.exp(-epoch * 0.25) + 0.15,
                'learning_rate': 1e-4,
                'binding_accuracy': 50 + 30 * epoch
            }
            visualizer.update(metrics, epoch)
        
        # 生成可视化
        visualizer.plot_training_curves("test_curves.png")
        
        print("  ✓ Visualization test successful!")
        print("  Check outputs/visualizations/test/ for generated plots")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("The Architect - Environment Test")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Models", test_models()))
    results.append(("Loss", test_loss()))
    results.append(("Dataset", test_dataset()))
    results.append(("Visualization", test_visualization()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! You're ready to train.")
        print("\nQuick start:")
        print("  1. Debug mode: python scripts/train.py --config configs/debug_config.yaml")
        print("  2. Demo: python scripts/demo.py")
    else:
        print("Some tests failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import torch  # 提前导入，因为 test_imports 需要它
    sys.exit(main())
