"""
模型下载器
支持 Hugging Face 和 ModelScope (国内镜像)
优先使用本地缓存，避免重复下载
"""

import os
from typing import Optional
import torch


class ModelDownloader:
    """
    模型下载管理器
    优先检查本地缓存，不存在时才下载
    优先使用 ModelScope (国内快)，失败时回退到 Hugging Face
    """
    
    # ModelScope 上的 CLIP 模型映射
    MODELSCOPE_MODELS = {
        "ViT-B-32": "AI-ModelScope/clip-vit-base-patch32",
        "ViT-B-16": "AI-ModelScope/clip-vit-base-patch16",
        "ViT-L-14": "AI-ModelScope/clip-vit-large-patch14",
        "ViT-L-14-336": "AI-ModelScope/clip-vit-large-patch14-336",
        "RN50": "AI-ModelScope/clip-rn50",
    }
    
    @classmethod
    def check_local_cache(cls, model_name: str, cache_dir: str = "./models") -> Optional[str]:
        """
        检查本地是否已有模型缓存
        
        Args:
            model_name: CLIP 模型名称
            cache_dir: 缓存目录
        
        Returns:
            本地路径，如果没有找到则返回 None
        """
        # 检查 open_clip 默认缓存位置
        import open_clip
        try:
            # 尝试获取 open_clip 的缓存路径
            cache_path = os.path.expanduser("~/.cache/open_clip")
            if os.path.exists(cache_path):
                # 检查是否有对应的模型文件
                model_files = []
                for root, dirs, files in os.walk(cache_path):
                    for file in files:
                        if file.endswith('.pt') or file.endswith('.bin') or file.endswith('.safetensors'):
                            if model_name.lower().replace('-', '_') in file.lower():
                                model_files.append(os.path.join(root, file))
                
                if model_files:
                    return os.path.dirname(model_files[0])
        except Exception:
            pass
        
        # 检查项目本地 models 目录
        modelscope_dir = os.path.join(cache_dir, "AI-ModelScope")
        if os.path.exists(modelscope_dir):
            model_id = cls.MODELSCOPE_MODELS.get(model_name, "").replace("AI-ModelScope/", "")
            if model_id:
                model_path = os.path.join(modelscope_dir, model_id)
                if os.path.exists(model_path):
                    # 检查是否有模型文件
                    for file in os.listdir(model_path):
                        if file.endswith(('.pt', '.bin', '.safetensors')):
                            return model_path
        
        return None
    
    @classmethod
    def download_clip_model(
        cls,
        model_name: str,
        cache_dir: str = "./models",
        use_modelscope: bool = False  # Disabled by default (ModelScope IDs may be outdated)
    ) -> str:
        """
        下载 CLIP 模型（优先使用本地缓存）
        
        Note: ModelScope support is disabled by default as model IDs may be outdated.
        Use Hugging Face with mirror (hf-mirror.com) instead.
        
        Args:
            model_name: CLIP 模型名称，如 "ViT-B-32"
            cache_dir: 本地缓存目录
            use_modelscope: 是否使用 ModelScope (不推荐)
        
        Returns:
            model_path: 模型本地路径
        """
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. 首先检查本地缓存
        local_path = cls.check_local_cache(model_name, cache_dir)
        if local_path:
            print(f"[OK] Found local model: {local_path}")
            return local_path
        
        # 2. 本地没有，使用 Hugging Face (with mirror)
        print("[INFO] Downloading from Hugging Face (using hf-mirror.com if in China)...")
        return cls._download_from_huggingface(model_name, cache_dir)
    
    @classmethod
    def _download_from_modelscope(cls, model_name: str, cache_dir: str) -> str:
        """从 ModelScope 下载"""
        try:
            from modelscope import snapshot_download
        except ImportError:
            raise ImportError("Please install modelscope: pip install modelscope")
        
        if model_name not in cls.MODELSCOPE_MODELS:
            available = list(cls.MODELSCOPE_MODELS.keys())
            raise ValueError(f"Model {model_name} not available on ModelScope. Available: {available}")
        
        model_id = cls.MODELSCOPE_MODELS[model_name]
        
        print(f"[INFO] Downloading {model_name} from ModelScope...")
        print(f"   Model ID: {model_id}")
        
        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision="master"
        )
        
        print(f"[OK] Model downloaded to: {model_dir}")
        return model_dir
    
    @classmethod
    def _download_from_huggingface(cls, model_name: str, cache_dir: str) -> str:
        """从 Hugging Face 下载 (使用 open_clip)"""
        print(f"[INFO] Downloading {model_name} from Hugging Face...")
        
        # 设置缓存目录
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
        
        # 尝试使用 HF 镜像
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 使用 open_clip 下载
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='openai'
        )
        
        # 返回缓存目录
        return os.path.join(cache_dir, 'open_clip')
    
    @classmethod
    def get_openai_clip_path(cls, model_name: str, cache_dir: str = "./models") -> str:
        """
        获取 OpenAI CLIP 模型的本地路径
        用于 open_clip 加载
        """
        # 先尝试下载（或检查本地）
        model_dir = cls.download_clip_model(model_name, cache_dir)
        return model_dir


def setup_modelscope_cache(cache_dir: str = "./models"):
    """
    设置 ModelScope 缓存目录
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['MODELSCOPE_CACHE'] = cache_dir
    print(f"[OK] ModelScope cache set to: {cache_dir}")


if __name__ == "__main__":
    print("Testing Model Downloader...")
    
    # 设置缓存目录
    setup_modelscope_cache("./models")
    
    # 测试检查本地缓存
    print("\nChecking local cache...")
    local_path = ModelDownloader.check_local_cache("ViT-B-32")
    if local_path:
        print(f"[OK] Found local model at: {local_path}")
    else:
        print("[INFO] No local model found, will download")
    
    # 测试下载（或检查缓存）
    try:
        model_path = ModelDownloader.download_clip_model(
            "ViT-B-32",
            use_modelscope=True
        )
        print(f"\n[OK] Model ready at: {model_path}")
    except Exception as e:
        print(f"\n[FAIL] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try: pip install modelscope -U")
        print("  3. Or use VPN and set use_modelscope=False")
