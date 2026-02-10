#!/usr/bin/env python3
"""
使用 ModelScope 下载 ARO 数据集（国内友好）
"""

import os
import json
from modelscope.msdatasets import MsDataset
from tqdm import tqdm


def download_aro_modelscope():
    """从 ModelScope 下载 ARO 数据集"""
    
    print("尝试从 ModelScope 下载 ARO 数据集...")
    print("-" * 60)
    
    output_dir = "data/aro"
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试不同的数据集名称格式
    dataset_names = [
        "kakaobrain/aro",
        "aro", 
        "kakaobrain/ARO",
        "pzc163/aro"
    ]
    
    for ds_name in dataset_names:
        try:
            print(f"\n尝试: {ds_name}")
            dataset = MsDataset.load(ds_name, split='train')
            print(f"[OK] 成功加载数据集: {ds_name}")
            
            # 处理并保存数据
            for split in ['coco_order', 'flickr30k_order', 'visual_genome_relation']:
                if hasattr(dataset, split) or split in dataset:
                    print(f"\n保存 {split}...")
                    data = []
                    split_data = dataset[split] if split in dataset else dataset
                    for item in tqdm(split_data, desc=f"Processing {split}"):
                        data.append({
                            'image': item.get('image', ''),
                            'caption_0': item.get('caption_0', ''),
                            'caption_1': item.get('caption_1', '')
                        })
                    
                    output_path = os.path.join(output_dir, f"{split}.json")
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"[OK] 保存了 {len(data)} 条样本到 {output_path}")
            
            return True
            
        except Exception as e:
            print(f"  失败: {e}")
            continue
    
    print("\n[ERROR] ModelScope 上未找到 ARO 数据集")
    return False


if __name__ == "__main__":
    download_aro_modelscope()
