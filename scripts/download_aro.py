#!/usr/bin/env python3
"""
下载 ARO 数据集
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm


def download_aro():
    """从 Hugging Face 下载 ARO 数据集"""
    
    print("Downloading ARO dataset from Hugging Face...")
    print("This may take a few minutes...\n")
    
    try:
        # 下载 ARO 数据集
        dataset = load_dataset("kakaobrain/aro", trust_remote_code=True)
        
        output_dir = "data/aro"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各个子集
        for split in ['coco_order', 'flickr30k_order', 'visual_genome_relation']:
            if split in dataset:
                print(f"Saving {split}...")
                data = []
                for item in tqdm(dataset[split], desc=f"Processing {split}"):
                    data.append({
                        'image': item['image'],
                        'caption_0': item['caption_0'],
                        'caption_1': item['caption_1']
                    })
                
                output_path = os.path.join(output_dir, f"{split}.json")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"[OK] Saved {len(data)} samples to {output_path}\n")
        
        print("="*60)
        print("ARO dataset download complete!")
        print("="*60)
        
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        print("\nTrying alternative method...")
        download_aro_manual()


def download_aro_manual():
    """手动下载 ARO 数据集"""
    import requests
    
    output_dir = "data/aro"
    os.makedirs(output_dir, exist_ok=True)
    
    # ARO 数据集的 GitHub 原始文件链接
    base_url = "https://raw.githubusercontent.com/kakaobrain/coyo-dataset/main/aro"
    
    files = {
        'coco_order.json': f'{base_url}/coco_order.json',
        'flickr30k_order.json': f'{base_url}/flickr30k_order.json',
        'visual_genome_relation.json': f'{base_url}/visual_genome_relation.json'
    }
    
    for filename, url in files.items():
        output_path = os.path.join(output_dir, filename)
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"[OK] Saved to {output_path}\n")
            else:
                print(f"[ERROR] Failed to download {filename}: HTTP {response.status_code}\n")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}\n")


if __name__ == "__main__":
    download_aro()
