"""
可视化工具演示脚本
运行此脚本可以快速验证可视化功能是否正常工作
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from src.utils.visualization import TrainingVisualizer, EvaluationVisualizer

def demo_training_visualizer():
    """演示训练过程可视化"""
    print("=" * 60)
    print("Demo: Training Visualizer")
    print("=" * 60)
    
    viz = TrainingVisualizer(save_dir="../outputs/visualizations/demo")
    
    # 模拟训练数据
    for epoch in range(10):
        metrics = {
            'train_loss': 2.5 * np.exp(-epoch * 0.3) + 0.1 * np.random.randn(),
            'val_loss': 2.7 * np.exp(-epoch * 0.25) + 0.15 * np.random.randn(),
            'learning_rate': 1e-4 * (0.5 ** (epoch // 3)),
            'binding_accuracy': 50 + 40 * (1 - np.exp(-epoch * 0.4)) + 2 * np.random.randn(),
        }
        # 模拟检索指标
        viz.history['retrieval_metrics']['R@1'].append(30 + 50 * (1 - np.exp(-epoch * 0.3)))
        viz.history['retrieval_metrics']['R@5'].append(45 + 45 * (1 - np.exp(-epoch * 0.3)))
        viz.history['retrieval_metrics']['R@10'].append(55 + 35 * (1 - np.exp(-epoch * 0.3)))
        
        viz.update(metrics, epoch)
    
    # 生成训练曲线
    viz.plot_training_curves("demo_training_curves.png")
    print("✓ Training curves generated\n")
    
    # 演示硬负样本分析
    n_samples = 200
    similarities = np.random.randn(n_samples, 3)
    similarities[:, 0] += 0.8  # 正样本相似度更高
    similarities[:, 1] += 0.5  # 硬负样本相似度中等
    similarities[:, 2] -= 0.2  # 简单负样本相似度更低
    similarities = np.clip(similarities, -1, 1)
    
    # 模拟预测结果 (60% 正确区分硬负样本)
    labels = (similarities[:, 0] > similarities[:, 1]).astype(int)
    
    viz.plot_hard_negative_analysis(similarities, labels, "demo_hard_negative.png")
    print("✓ Hard negative analysis generated\n")
    
    # 演示注意力热力图
    attention = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.2, 0.2],
        [0.1, 0.2, 0.5, 0.2],
        [0.15, 0.25, 0.3, 0.3],
    ])  # 4 heads, 4 objects
    objects = ["red cat", "blue dog", "green bird", "yellow car"]
    
    viz.plot_attention_heatmap(attention, objects, "CLS token", "demo_attention.png")
    print("✓ Attention heatmap generated\n")


def demo_evaluation_visualizer():
    """演示评估结果可视化"""
    print("=" * 60)
    print("Demo: Evaluation Visualizer")
    print("=" * 60)
    
    viz = EvaluationVisualizer(save_dir="../outputs/visualizations/demo")
    
    # 演示检索性能对比
    methods = ["CLIP Baseline", "The Architect", "The Architect + LoRA"]
    metrics = {
        'R@1': [42.5, 68.3, 72.1],
        'R@5': [58.2, 82.5, 85.3],
        'R@10': [67.1, 89.2, 91.5],
    }
    
    viz.plot_retrieval_comparison(methods, metrics, "demo_retrieval_comparison.png")
    print("✓ Retrieval comparison generated\n")
    
    # 演示损失组件分析
    iterations = 1000
    loss_components = {
        'total': 2.0 * np.exp(-np.arange(iterations) / 300) + 0.1 + 0.05 * np.random.randn(iterations),
        'contrastive': 1.5 * np.exp(-np.arange(iterations) / 300) + 0.05 * np.random.randn(iterations),
        'structural': 0.5 * np.exp(-np.arange(iterations) / 250) + 0.02 * np.random.randn(iterations),
    }
    
    viz.plot_loss_components(loss_components, "demo_loss_components.png")
    print("✓ Loss components generated\n")
    
    # 演示混淆矩阵
    from sklearn.metrics import confusion_matrix
    n_classes = 5
    y_true = np.random.randint(0, n_classes, 200)
    y_pred = y_true.copy()
    # 添加一些错误
    mask = np.random.rand(200) < 0.2
    y_pred[mask] = np.random.randint(0, n_classes, mask.sum())
    
    cm = confusion_matrix(y_true, y_pred)
    labels = ["attr swap", "entity swap", "order swap", "negation", "correct"]
    
    viz.plot_confusion_matrix(cm, labels, "demo_confusion_matrix.png")
    print("✓ Confusion matrix generated\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("The Architect - Visualization Demo")
    print("=" * 60 + "\n")
    
    demo_training_visualizer()
    demo_evaluation_visualizer()
    
    print("=" * 60)
    print("All visualizations saved to: outputs/visualizations/demo/")
    print("=" * 60)
