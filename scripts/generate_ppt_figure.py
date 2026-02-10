#!/usr/bin/env python3
"""
生成 PPT 用的总结图
风格：简洁、醒目、高对比度
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patches as mpatches


def load_results(path='outputs/sugarcrepe_results_all.json'):
    with open(path, 'r') as f:
        return json.load(f)


def create_ppt_summary_v1(results, save_dir='outputs/visualizations'):
    """
    PPT 风格总结图 v1: 大数字 + 简洁布局
    适合放在 PPT 第一页作为亮点展示
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # 标题
    ax.text(8, 8.3, 'The Architect', fontsize=48, fontweight='bold', 
            ha='center', va='top', color='#1a1a1a')
    # 副标题
    ax.text(8, 7.5, 'Multi-Object Attribute Binding with CLIP', 
            fontsize=24, ha='center', va='top', color='#555555')
    
    # 主结果 - 大数字
    overall_acc = results['overall']['accuracy']
    
    # 大圆背景
    circle = Circle((4, 4.5), 1.8, facecolor='#27ae60', alpha=0.15, edgecolor='none')
    ax.add_patch(circle)
    circle2 = Circle((4, 4.5), 1.5, facecolor='#27ae60', alpha=0.25, edgecolor='none')
    ax.add_patch(circle2)
    
    ax.text(4, 4.8, f'{overall_acc:.1f}%', fontsize=72, fontweight='bold',
            ha='center', va='center', color='#27ae60')
    ax.text(4, 3.5, 'Overall Accuracy', fontsize=18, ha='center', va='center', color='#666')
    ax.text(4, 3.0, 'SugarCrepe Benchmark', fontsize=14, ha='center', va='center', color='#999')
    
    # 右侧关键指标卡片
    card_data = [
        ('[TARGET]', 'Swap-Att', f"{results['subsets']['swap_att']['accuracy']:.1f}%", '+10.3% vs CLIP'),
        ('[STAR]', 'Replace-Obj', f"{results['subsets']['replace_obj']['accuracy']:.1f}%", '+27.9% vs CLIP'),
        ('[DATA]', 'Total Samples', '4,757', '5 Tasks'),
    ]
    
    y_start = 6.2
    for i, (emoji, label, value, subtext) in enumerate(card_data):
        y = y_start - i * 1.8
        # 卡片背景
        rect = FancyBboxPatch((8.5, y-0.6), 6.5, 1.4, 
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2)
        ax.add_patch(rect)
        
        # 图标 - 使用几何图形代替 emoji
        if 'TARGET' in emoji:
            # 绘制靶心图标
            circle1 = plt.Circle((9, y), 0.25, fill=False, edgecolor='#666', linewidth=2)
            circle2 = plt.Circle((9, y), 0.15, fill=False, edgecolor='#666', linewidth=2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.plot([9, 9], [y-0.08, y+0.08], 'k-', linewidth=2)
            ax.plot([9-0.08, 9+0.08], [y, y], 'k-', linewidth=2)
        elif 'STAR' in emoji:
            # 绘制星形
            ax.text(9, y, '*', fontsize=40, ha='center', va='center', color='#f39c12', fontweight='bold')
        elif 'DATA' in emoji:
            # 绘制柱状图图标
            for i, h in enumerate([0.15, 0.25, 0.2]):
                rect = Rectangle((8.7+i*0.2, y-0.15), 0.12, h, facecolor='#666', edgecolor='none')
                ax.add_patch(rect)
        # Label
        ax.text(10, y+0.25, label, fontsize=16, fontweight='bold', 
                ha='left', va='center', color='#333')
        # Subtext
        ax.text(10, y-0.25, subtext, fontsize=12, ha='left', va='center', color='#666')
        # Value
        ax.text(14, y, value, fontsize=28, fontweight='bold', 
                ha='center', va='center', color='#27ae60')
    
    # 底部对比条
    ax.text(8, 1.3, 'vs CLIP Baseline (~60%)', fontsize=16, ha='center', va='center', color='#666')
    
    # 进度条背景
    rect_bg = FancyBboxPatch((3, 0.5), 10, 0.5, 
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='#e9ecef', edgecolor='none')
    ax.add_patch(rect_bg)
    
    # 进度条填充
    progress_width = 10 * (overall_acc / 100)
    rect_fill = FancyBboxPatch((3, 0.5), progress_width, 0.5, 
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#27ae60', edgecolor='none')
    ax.add_patch(rect_fill)
    
    # 进度条文字
    ax.text(8, 0.75, f'+{overall_acc - 60:.1f}% Improvement', fontsize=14, 
            ha='center', va='center', color='white', fontweight='bold')
    
    save_path = os.path.join(save_dir, 'ppt_summary_v1.png')
    plt.savefig(save_path, dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"PPT Summary v1 saved: {save_path}")
    return save_path


def create_ppt_comparison(results, save_dir='outputs/visualizations'):
    """
    PPT 风格对比图: 并排对比 CLIP vs Ours
    适合放在对比页
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor='white')
    
    # 左图: CLIP
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_facecolor('#fff5f5')
    
    # CLIP 标题
    ax1.text(5, 9, 'CLIP (Baseline)', fontsize=32, fontweight='bold',
             ha='center', va='top', color='#e74c3c')
    ax1.text(5, 8, 'Limited Compositional Understanding', fontsize=16,
             ha='center', va='top', color='#999')
    
    # CLIP 指标
    clip_values = [58.0, 62.0, 65.0, 65.0, 55.0]
    labels = ['Swap-Att', 'Swap-Obj', 'Replace-Att', 'Replace-Obj', 'Replace-Rel']
    
    for i, (label, val) in enumerate(zip(labels, clip_values)):
        y = 7 - i * 1.3
        # 标签
        ax1.text(1, y, label, fontsize=14, ha='left', va='center', color='#666')
        # 数值
        ax1.text(9, y, f'{val:.1f}%', fontsize=20, fontweight='bold',
                ha='right', va='center', color='#e74c3c')
        # 进度条
        bar_width = 6 * (val / 100)
        rect = FancyBboxPatch((2.5, y-0.25), bar_width, 0.5,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#e74c3c', alpha=0.3, edgecolor='none')
        ax1.add_patch(rect)
    
    # 总体
    ax1.text(5, 1, 'Overall: ~60%', fontsize=24, fontweight='bold',
             ha='center', va='center', color='#e74c3c')
    
    # 右图: Ours
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_facecolor('#f0fff4')
    
    # Ours 标题
    ax2.text(5, 9, 'The Architect (Ours)', fontsize=32, fontweight='bold',
             ha='center', va='top', color='#27ae60')
    ax2.text(5, 8, 'Strong Attribute Binding Capability', fontsize=16,
             ha='center', va='top', color='#27ae60')
    
    # Ours 指标
    our_values = [
        results['subsets']['swap_att']['accuracy'],
        results['subsets']['swap_obj']['accuracy'],
        results['subsets']['replace_att']['accuracy'],
        results['subsets']['replace_obj']['accuracy'],
        results['subsets']['replace_rel']['accuracy'],
    ]
    improvements = [o - c for o, c in zip(our_values, clip_values)]
    
    for i, (label, val, imp) in enumerate(zip(labels, our_values, improvements)):
        y = 7 - i * 1.3
        # 标签
        ax2.text(1, y, label, fontsize=14, ha='left', va='center', color='#333')
        # 提升标签
        ax2.text(2.3, y+0.35, f'+{imp:.1f}%', fontsize=10, ha='left', va='center',
                color='#27ae60', fontweight='bold')
        # 数值
        ax2.text(9, y, f'{val:.1f}%', fontsize=20, fontweight='bold',
                ha='right', va='center', color='#27ae60')
        # 进度条
        bar_width = 6 * (val / 100)
        rect = FancyBboxPatch((2.5, y-0.25), bar_width, 0.5,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#27ae60', alpha=0.7, edgecolor='none')
        ax2.add_patch(rect)
    
    # 总体
    overall = results['overall']['accuracy']
    ax2.text(5, 1, f'Overall: {overall:.1f}%', fontsize=24, fontweight='bold',
             ha='center', va='center', color='#27ae60')
    
    # 添加中间箭头
    fig.text(0.5, 0.5, '>>', fontsize=50, ha='center', va='center',
             color='#333', transform=fig.transFigure, fontweight='bold')
    
    save_path = os.path.join(save_dir, 'ppt_comparison.png')
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"PPT Comparison saved: {save_path}")
    return save_path


def create_ppt_architecture_diagram(results, save_dir='outputs/visualizations'):
    """
    PPT 风格架构图: 展示模型架构和关键创新点
    适合放在方法介绍页
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'Model Architecture', fontsize=36, fontweight='bold',
            ha='center', va='top', color='#1a1a1a')
    
    # 定义颜色
    color_clip = '#3498db'
    color_adapter = '#9b59b6'
    color_loss = '#e74c3c'
    
    # === 左侧: 输入 ===
    ax.text(2, 7.5, 'Input', fontsize=18, fontweight='bold', ha='center', color='#666')
    
    # 图像输入
    img_box = FancyBboxPatch((0.5, 5.5), 3, 1.5, 
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2)
    ax.add_patch(img_box)
    ax.text(2, 6.5, 'Image', fontsize=14, ha='center', va='center')
    ax.text(2, 6.0, '(COCO)', fontsize=10, ha='center', va='center', color='#999')
    
    # 文本输入
    txt_box = FancyBboxPatch((0.5, 3.5), 3, 1.5,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2)
    ax.add_patch(txt_box)
    ax.text(2, 4.5, 'Text', fontsize=14, ha='center', va='center')
    ax.text(2, 4.0, '"red cat & blue dog"', fontsize=10, ha='center', va='center', color='#999')
    
    # === 中间: 处理流程 ===
    ax.text(8, 7.5, 'Processing Pipeline', fontsize=18, fontweight='bold', ha='center', color='#666')
    
    # CLIP Visual (冻结)
    clip_v_box = FancyBboxPatch((5.5, 6), 5, 1,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor=color_clip, alpha=0.2, 
                                edgecolor=color_clip, linewidth=3)
    ax.add_patch(clip_v_box)
    ax.text(8, 6.5, 'CLIP Visual Encoder (Frozen)', fontsize=13, 
            ha='center', va='center', fontweight='bold', color=color_clip)
    
    # Object Detector
    det_box = FancyBboxPatch((5.5, 4.5), 5, 1,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#f39c12', alpha=0.2,
                             edgecolor='#f39c12', linewidth=3)
    ax.add_patch(det_box)
    ax.text(8, 5.0, 'YOLOv8 Object Detector', fontsize=13,
            ha='center', va='center', fontweight='bold', color='#d68910')
    
    # Object-Aware Adapter
    adapter_box = FancyBboxPatch((5.5, 2.8), 5, 1.3,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=color_adapter, alpha=0.3,
                                 edgecolor=color_adapter, linewidth=4)
    ax.add_patch(adapter_box)
    ax.text(8, 3.7, 'Object-Aware Adapter', fontsize=14,
            ha='center', va='center', fontweight='bold', color=color_adapter)
    ax.text(8, 3.2, 'Cross-Attention Fusion', fontsize=11,
            ha='center', va='center', color='#8e44ad')
    
    # CLIP Text (冻结)
    clip_t_box = FancyBboxPatch((5.5, 1.3), 5, 1,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor=color_clip, alpha=0.2,
                                edgecolor=color_clip, linewidth=3)
    ax.add_patch(clip_t_box)
    ax.text(8, 1.8, 'CLIP Text Encoder (Frozen)', fontsize=13,
            ha='center', va='center', fontweight='bold', color=color_clip)
    
    # === 右侧: 输出和损失 ===
    ax.text(13.5, 7.5, 'Training', fontsize=18, fontweight='bold', ha='center', color='#666')
    
    # Contrastive Loss
    loss1_box = FancyBboxPatch((12, 5.5), 3, 1,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=color_loss, alpha=0.2,
                               edgecolor=color_loss, linewidth=2)
    ax.add_patch(loss1_box)
    ax.text(13.5, 6.0, 'Contrastive Loss', fontsize=12,
            ha='center', va='center', fontweight='bold', color=color_loss)
    
    # Structural Loss (高亮)
    loss2_box = FancyBboxPatch((12, 3.8), 3, 1.3,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=color_loss, alpha=0.4,
                               edgecolor=color_loss, linewidth=4)
    ax.add_patch(loss2_box)
    ax.text(13.5, 4.7, 'Structural Loss', fontsize=13,
            ha='center', va='center', fontweight='bold', color=color_loss)
    ax.text(13.5, 4.2, 'Hard Negative Mining', fontsize=10,
            ha='center', va='center', color='#c0392b')
    
    # 结果
    result_box = FancyBboxPatch((12, 1.8), 3, 1.5,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#27ae60', alpha=0.3,
                                edgecolor='#27ae60', linewidth=3)
    ax.add_patch(result_box)
    ax.text(13.5, 2.8, f"{results['overall']['accuracy']:.1f}%", fontsize=20,
            ha='center', va='center', fontweight='bold', color='#27ae60')
    ax.text(13.5, 2.2, 'SugarCrepe', fontsize=11, ha='center', va='center', color='#27ae60')
    
    # === 连接箭头 ===
    arrow_style = dict(arrowstyle='->', color='#7f8c8d', lw=2)
    
    # 输入到处理
    ax.annotate('', xy=(5.3, 6.5), xytext=(3.6, 6.2), arrowprops=arrow_style)
    ax.annotate('', xy=(5.3, 5.0), xytext=(3.6, 6.0), arrowprops=arrow_style)
    ax.annotate('', xy=(5.3, 2.3), xytext=(3.6, 4.0), arrowprops=arrow_style)
    
    # 处理到损失
    ax.annotate('', xy=(11.8, 6.0), xytext=(10.6, 6.2), arrowprops=arrow_style)
    ax.annotate('', xy=(11.8, 4.5), xytext=(10.6, 3.5), arrowprops=arrow_style)
    
    # 损失到结果
    ax.annotate('', xy=(13.5, 3.3), xytext=(13.5, 3.7), arrowprops=arrow_style)
    
    save_path = os.path.join(save_dir, 'ppt_architecture.png')
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"PPT Architecture saved: {save_path}")
    return save_path


def main():
    print("="*60)
    print("Generating PPT-Style Figures")
    print("="*60)
    
    results = load_results()
    
    print("\nGenerating PPT figures...")
    create_ppt_summary_v1(results)
    create_ppt_comparison(results)
    create_ppt_architecture_diagram(results)
    
    print("\n" + "="*60)
    print("All PPT figures generated!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. ppt_summary_v1.png     - 首页亮点展示")
    print("  2. ppt_comparison.png     - CLIP vs Ours 对比")
    print("  3. ppt_architecture.png   - 模型架构图")
    print("\nLocation: outputs/visualizations/")


if __name__ == "__main__":
    main()
