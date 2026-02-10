"""
可视化工具模块 - 用于实验结果观察
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import json
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置默认样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-paper')


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = "outputs/visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': [],
            'binding_accuracy': [],
            'retrieval_metrics': {'R@1': [], 'R@5': [], 'R@10': []}
        }
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """更新历史记录"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
            elif key in self.history['retrieval_metrics']:
                self.history['retrieval_metrics'][key].append(value)
                
    def plot_training_curves(self, save_name: str = "training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress Dashboard', fontsize=16, fontweight='bold')
        
        # Loss curves
        ax1 = axes[0, 0]
        epochs = self.history['epoch']
        if self.history['train_loss']:
            ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2 = axes[0, 1]
        if self.history['learning_rate']:
            ax2.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # Binding accuracy
        ax3 = axes[1, 0]
        if self.history['binding_accuracy']:
            ax3.plot(epochs, self.history['binding_accuracy'], 'm-', marker='o', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Attribute Binding Accuracy')
            ax3.grid(True, alpha=0.3)
        
        # Retrieval metrics
        ax4 = axes[1, 1]
        for metric_name, values in self.history['retrieval_metrics'].items():
            if values:
                ax4.plot(epochs[:len(values)], values, marker='o', label=metric_name, linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall (%)')
        ax4.set_title('Retrieval Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Training curves saved to {save_path}")
        return save_path
    
    def plot_hard_negative_analysis(self, 
                                    similarities: np.ndarray,
                                    labels: np.ndarray,
                                    save_name: str = "hard_negative_analysis.png"):
        """
        分析硬负样本的相似度分布
        
        Args:
            similarities: (N, 3) array - [pos_sim, hard_neg_sim, easy_neg_sim]
            labels: (N,) array - 是否正确区分
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hard Negative Mining Analysis', fontsize=14, fontweight='bold')
        
        # Similarity distribution
        ax1 = axes[0]
        ax1.hist(similarities[:, 0], bins=30, alpha=0.6, label='Positive', color='green')
        ax1.hist(similarities[:, 1], bins=30, alpha=0.6, label='Hard Negative', color='red')
        ax1.hist(similarities[:, 2], bins=30, alpha=0.6, label='Easy Negative', color='blue')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Count')
        ax1.set_title('Similarity Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Margin analysis
        ax2 = axes[1]
        hard_margin = similarities[:, 0] - similarities[:, 1]
        easy_margin = similarities[:, 0] - similarities[:, 2]
        
        ax2.scatter(hard_margin, easy_margin, c=labels, cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Pos - Hard Neg Margin')
        ax2.set_ylabel('Pos - Easy Neg Margin')
        ax2.set_title('Margin Analysis (Green=Correct, Red=Wrong)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Hard negative analysis saved to {save_path}")
        return save_path
    
    def plot_attention_heatmap(self, 
                               attention_weights: np.ndarray,
                               object_labels: List[str],
                               query_label: str = "Query",
                               save_name: str = "attention_heatmap.png"):
        """
        可视化 Cross-Attention 权重热力图
        
        Args:
            attention_weights: (num_heads, num_objects) or (num_objects,)
            object_labels: 物体标签列表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(attention_weights.shape) == 2:
            # Multi-head attention
            sns.heatmap(attention_weights, 
                       xticklabels=object_labels,
                       yticklabels=[f'Head {i+1}' for i in range(attention_weights.shape[0])],
                       annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Attention Weight'})
            ax.set_title(f'Multi-Head Cross-Attention: {query_label}')
        else:
            # Single attention
            sns.heatmap(attention_weights.reshape(1, -1),
                       xticklabels=object_labels,
                       yticklabels=[query_label],
                       annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Attention Weight'})
            ax.set_title(f'Cross-Attention Weights')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Attention heatmap saved to {save_path}")
        return save_path


class EvaluationVisualizer:
    """评估结果可视化器"""
    
    def __init__(self, save_dir: str = "outputs/visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_retrieval_comparison(self,
                                   methods: List[str],
                                   metrics: Dict[str, List[float]],
                                   save_name: str = "retrieval_comparison.png"):
        """
        对比不同方法的检索性能
        
        Args:
            methods: 方法名称列表
            metrics: {'R@1': [...], 'R@5': [...], 'R@10': [...]}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.25
        
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.bar(x + i * width, values, width, label=metric_name, color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Retrieval Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (metric_name, values) in enumerate(metrics.items()):
            for j, v in enumerate(values):
                ax.text(j + i * width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Retrieval comparison saved to {save_path}")
        return save_path
    
    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              labels: List[str],
                              save_name: str = "confusion_matrix.png"):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Attribute Binding Confusion Matrix')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Confusion matrix saved to {save_path}")
        return save_path
    
    def plot_qualitative_results(self,
                                  images: List[np.ndarray],
                                  captions: List[Tuple[str, str, str]],  # (positive, negative, prediction)
                                  save_name: str = "qualitative_results.png"):
        """
        可视化定性结果
        
        Args:
            images: 图像列表
            captions: (正样本描述, 负样本描述, 模型预测) 元组列表
        """
        n = min(len(images), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Qualitative Results: Correct (Green) / Incorrect (Red)', fontsize=14, fontweight='bold')
        
        for i, (ax, img, (pos, neg, pred)) in enumerate(zip(axes.flat, images[:n], captions[:n])):
            ax.imshow(img)
            
            # 判断预测是否正确
            is_correct = pred == pos
            color = 'green' if is_correct else 'red'
            
            title = f"✓ {pred}" if is_correct else f"✗ Pred: {pred}"
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')
            
            caption_text = f"GT: {pos}\nNeg: {neg}"
            ax.text(0.5, -0.15, caption_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=8, wrap=True)
            ax.axis('off')
        
        # Hide unused subplots
        for j in range(n, 6):
            axes.flat[j].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Qualitative results saved to {save_path}")
        return save_path
    
    def plot_loss_components(self,
                             loss_components: Dict[str, List[float]],
                             save_name: str = "loss_components.png"):
        """
        绘制损失函数各组成部分的变化
        
        Args:
            loss_components: {'total': [...], 'contrastive': [...], 'structural': [...]}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'total': 'black', 'contrastive': 'blue', 'structural': 'red', 
                 'regularization': 'orange', 'detector': 'purple'}
        
        for name, values in loss_components.items():
            if values:
                ax.plot(values, label=name.capitalize(), 
                       color=colors.get(name, 'gray'), linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Component Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Loss components saved to {save_path}")
        return save_path


def create_experiment_report(metrics: Dict, save_path: str = "outputs/experiment_report.json"):
    """创建实验报告 JSON"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'summary': {
            'final_binding_accuracy': metrics.get('binding_accuracy', [0])[-1] if metrics.get('binding_accuracy') else 0,
            'best_epoch': int(np.argmax(metrics.get('val_acc', [0]))),
            'total_epochs': len(metrics.get('train_loss', []))
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[Report] Experiment report saved to {save_path}")
    return report
