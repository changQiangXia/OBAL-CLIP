"""
Training Script for The Architect
支持 Debug 模式和全量训练
包含混合精度训练 (AMP)、梯度累积等优化
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import random

from src.models.the_architect import TheArchitect
from src.models.detector_wrapper import DetectorWrapper, DummyDetector
from src.losses.structural_loss import StructuralLoss, HardNegativeMiner
from src.data.dataset import SyntheticDataset, COCODataset, collate_fn
from src.utils.visualization import TrainingVisualizer, create_experiment_report


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: Dict, device: str):
    """构建模型"""
    model = TheArchitect(
        clip_model_name=config['model']['clip_model'],
        clip_pretrained=config['model']['clip_pretrained'],
        adapter_config=config['model']['adapter'],
        freeze_clip=config['model']['adapter'].get('freeze_clip', True),
        device=device
    )
    return model


def build_detector(config: Dict, device: str):
    """构建检测器"""
    if config['data'].get('use_synthetic', False):
        # Debug 模式使用 DummyDetector
        print("Using DummyDetector for synthetic data")
        return DummyDetector(
            device=device,
            max_detections=config['model']['detector']['max_detections']
        )
    
    detector_config = config['model']['detector']
    return DetectorWrapper(
        detector_type=detector_config['type'],
        model_name=detector_config['model_name'],
        device=detector_config.get('device', device),
        conf_threshold=detector_config['conf_threshold'],
        max_detections=detector_config['max_detections']
    )


def build_dataloaders(config: Dict):
    """构建 DataLoader"""
    data_config = config['data']
    
    # 训练数据集
    if data_config.get('use_synthetic', False):
        print(f"Using SyntheticDataset with {data_config['synthetic_samples']} samples")
        train_dataset = SyntheticDataset(
            num_samples=data_config['synthetic_samples'],
            image_size=data_config['image_size'],
            hard_negative_ratio=config['loss']['hard_negative_ratio']
        )
    else:
        # TODO: 支持多个数据集混合
        train_dataset = COCODataset(
            root_dir=data_config['train_datasets'][0]['root'],
            split=data_config['train_datasets'][0]['split'],
            image_size=data_config['image_size']
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=data_config.get('pin_memory', False),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return train_loader


def build_optimizer(model: nn.Module, config: Dict):
    """构建优化器"""
    opt_config = config['training']['optimizer']
    
    # 只优化 Adapter 和投影层的参数
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    if opt_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=tuple(opt_config['betas'])
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay']
        )
    
    return optimizer


def build_scheduler(optimizer, config: Dict, num_training_steps: int):
    """构建学习率调度器"""
    sched_config = config['training']['scheduler']
    
    if sched_config['type'].lower() == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # 简单的 cosine 退火
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(
    model: TheArchitect,
    detector,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: StructuralLoss,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    device: str,
    visualizer: TrainingVisualizer
) -> Dict[str, float]:
    """训练一个 epoch"""
    
    model.train()
    
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_structural_loss = 0.0
    num_batches = 0
    
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    log_every = config['logging']['log_every']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        captions = batch['captions']
        
        # 提取区域特征（在 no_grad 模式下以节省显存）
        with torch.no_grad():
            region_features, region_mask, _ = detector.extract_features(images)
            region_features = region_features.to(device)
            region_mask = region_mask.to(device)
        
        # 混合精度前向传播
        with autocast(enabled=config['training']['amp']):
            # 前向传播
            output = model(images, captions, region_features, region_mask)
            
            # 计算损失
            loss, loss_dict = loss_fn(
                output['image_features'],
                output['text_features'],
                captions,
                logit_scale=output['logit_scale'],
                model=model
            )
            
            # 梯度累积
            loss = loss / grad_accum_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积步骤
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            if config['training'].get('clip_grad_norm'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['clip_grad_norm']
                )
            
            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        # 记录损失
        total_loss += loss.item() * grad_accum_steps
        total_contrastive_loss += loss_dict['contrastive']
        total_structural_loss += loss_dict['structural']
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item() * grad_accum_steps:.4f}",
            'contrastive': f"{loss_dict['contrastive']:.4f}",
            'structural': f"{loss_dict['structural']:.4f}"
        })
        
        # 日志
        if batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [Step {batch_idx}] Loss: {loss.item() * grad_accum_steps:.4f}, "
                  f"LR: {current_lr:.6f}")
    
    avg_loss = total_loss / num_batches
    avg_contrastive = total_contrastive_loss / num_batches
    avg_structural = total_structural_loss / num_batches
    
    return {
        'train_loss': avg_loss,
        'contrastive_loss': avg_contrastive,
        'structural_loss': avg_structural,
        'learning_rate': optimizer.param_groups[0]['lr']
    }


@torch.no_grad()
def evaluate(
    model: TheArchitect,
    detector,
    val_loader: DataLoader,
    config: Dict,
    device: str
) -> Dict[str, float]:
    """
    评估模型
    计算属性绑定准确率
    """
    model.eval()
    
    correct = 0
    total = 0
    
    print("\nEvaluating...")
    for batch in tqdm(val_loader, desc="Validation"):
        images = batch['images'].to(device)
        captions = batch['captions']
        hard_negs = batch.get('hard_negatives', [])
        
        # 提取区域特征
        region_features, region_mask, _ = detector.extract_features(images)
        region_features = region_features.to(device)
        region_mask = region_mask.to(device)
        
        # 前向传播
        output = model(images, captions, region_features, region_mask)
        
        # 计算图像-正样本相似度
        img_features = output['image_features']
        text_features = output['text_features']
        
        # 简单的准确率：图像与正样本的相似度是否大于与硬负样本的相似度
        if hard_negs:
            # 编码硬负样本
            neg_features = model.encode_text(hard_negs[:len(captions)])
            
            for i in range(len(captions)):
                sim_pos = (img_features[i] @ text_features[i]).item()
                sim_neg = (img_features[i] @ neg_features[i]).item()
                
                if sim_pos > sim_neg:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'binding_accuracy': accuracy * 100,
        'correct': correct,
        'total': total
    }


def save_checkpoint(
    model: TheArchitect,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train The Architect')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Debug Mode: {config.get('debug', False)}")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    set_seed(config['seed'])
    
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
    model = build_model(config, device)
    
    # 构建检测器
    print("\nBuilding detector...")
    detector = build_detector(config, device)
    
    # 构建 DataLoader
    print("\nBuilding dataloaders...")
    train_loader = build_dataloaders(config)
    
    # 构建优化器
    print("\nBuilding optimizer...")
    optimizer = build_optimizer(model, config)
    
    # 构建学习率调度器
    num_training_steps = len(train_loader) * config['training']['num_epochs'] // config['training']['gradient_accumulation_steps']
    scheduler = build_scheduler(optimizer, config, num_training_steps)
    
    # 构建损失函数
    loss_fn = StructuralLoss(
        temperature=config['loss']['temperature'],
        contrastive_weight=config['loss']['contrastive_weight'],
        structural_weight=config['loss']['structural_weight'],
        hard_negative_ratio=config['loss']['hard_negative_ratio']
    )
    
    # 混合精度训练
    scaler = GradScaler(enabled=config['training']['amp'])
    
    # 可视化工具
    visualizer = TrainingVisualizer(
        save_dir=config.get('visualization', {}).get('save_dir', 'outputs/visualizations')
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 训练循环
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 60)
        
        # 训练
        train_metrics = train_one_epoch(
            model, detector, train_loader, optimizer, scheduler,
            loss_fn, scaler, epoch, config, device, visualizer
        )
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
        print(f"  Structural Loss: {train_metrics['structural_loss']:.4f}")
        print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
        
        # 更新可视化
        visualizer.update(train_metrics, epoch)
        
        # 保存检查点
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_path = f"{config['experiment_name']}_epoch_{epoch + 1}.pt"
            save_path = os.path.join('outputs/checkpoints', save_path)
            save_checkpoint(model, optimizer, epoch, train_metrics, save_path)
        
        # 保存最佳模型
        if train_metrics['train_loss'] < best_loss:
            best_loss = train_metrics['train_loss']
            save_path = f"{config['experiment_name']}_best.pt"
            save_path = os.path.join('outputs/checkpoints', save_path)
            save_checkpoint(model, optimizer, epoch, train_metrics, save_path)
    
    # 生成最终可视化
    print("\nGenerating visualizations...")
    visualizer.plot_training_curves()
    
    # 保存实验报告
    create_experiment_report(visualizer.history)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
