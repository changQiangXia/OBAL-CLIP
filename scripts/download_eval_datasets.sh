#!/bin/bash
# 下载属性绑定评估数据集

set -e

echo "========================================"
echo "下载属性绑定评估数据集"
echo "========================================"

# 创建数据目录
mkdir -p data/aro
mkdir -p data/sugarcrepe
mkdir -p data/winoground

# 安装依赖
echo "安装依赖..."
pip install -q datasets huggingface-hub requests tqdm

echo ""
echo "========================================"
echo "1. 下载 ARO 数据集"
echo "========================================"

cd data/aro

# 下载 COCO Order
if [ ! -f "coco_order.json" ]; then
    echo "下载 COCO Order..."
    wget -q --show-progress https://huggingface.co/datasets/kakaobrain/aro/resolve/main/coco_order.json || \
    curl -L -o coco_order.json https://huggingface.co/datasets/kakaobrain/aro/resolve/main/coco_order.json
fi

# 下载 Flickr30k Order
if [ ! -f "flickr30k_order.json" ]; then
    echo "下载 Flickr30k Order..."
    wget -q --show-progress https://huggingface.co/datasets/kakaobrain/aro/resolve/main/flickr30k_order.json || \
    curl -L -o flickr30k_order.json https://huggingface.co/datasets/kakaobrain/aro/resolve/main/flickr30k_order.json
fi

# 下载 Visual Genome Relation
if [ ! -f "visual_genome_relation.json" ]; then
    echo "下载 Visual Genome Relation..."
    wget -q --show-progress https://huggingface.co/datasets/kakaobrain/aro/resolve/main/visual_genome_relation.json || \
    curl -L -o visual_genome_relation.json https://huggingface.co/datasets/kakaobrain/aro/resolve/main/visual_genome_relation.json
fi

cd ../..

echo ""
echo "========================================"
echo "2. 下载 SugarCrepe 数据集"
echo "========================================"

cd data/sugarcrepe

BASE_URL="https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"

for file in replace_att.json replace_obj.json replace_rel.json swap_att.json swap_obj.json; do
    if [ ! -f "$file" ]; then
        echo "下载 $file..."
        wget -q --show-progress "$BASE_URL/$file" || curl -L -o "$file" "$BASE_URL/$file"
    fi
done

cd ../..

echo ""
echo "========================================"
echo "数据集下载完成！"
echo "========================================"

echo ""
echo "数据结构："
echo "  data/aro/"
echo "    ├── coco_order.json"
echo "    ├── flickr30k_order.json"
echo "    └── visual_genome_relation.json"
echo ""
echo "  data/sugarcrepe/"
echo "    ├── replace_att.json"
echo "    ├── replace_obj.json"
echo "    ├── replace_rel.json"
echo "    ├── swap_att.json"
echo "    └── swap_obj.json"
echo ""

echo "⚠️  Winoground 需要手动申请："
echo "    https://huggingface.co/datasets/facebook/winoground"
echo ""
