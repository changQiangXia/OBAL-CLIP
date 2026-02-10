"""
Simplified model loader for CLIP
Uses open_clip's built-in caching mechanism
"""

import os
import torch
import open_clip
from typing import Tuple, Optional


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
    cache_dir: Optional[str] = None
) -> Tuple[torch.nn.Module, callable, callable]:
    """
    Load CLIP model with proper caching
    
    Args:
        model_name: CLIP model architecture (e.g., "ViT-B-32")
        pretrained: Pretrained weights source (e.g., "openai")
        device: Device to load model on
        cache_dir: Optional custom cache directory
    
    Returns:
        model: CLIP model
n        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
    
    Raises:
        RuntimeError: If model cannot be loaded
    """
    # Set cache directory if provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TORCH_HOME'] = cache_dir
        os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    
    # Handle QuickGELU for ViT-B-32 with openai weights
    force_quick_gelu = (model_name == "ViT-B-32" and pretrained == "openai")
    
    print(f"[INFO] Loading CLIP: {model_name} ({pretrained})")
    if force_quick_gelu:
        print("[INFO] Using QuickGELU activation")
    
    try:
        # Load model using open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=force_quick_gelu
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        print(f"[OK] Model loaded successfully on {device}")
        return model, preprocess, tokenizer
        
    except RuntimeError as e:
        if "offline" in str(e).lower() or "connection" in str(e).lower():
            print("\n" + "="*60)
            print("[ERROR] Cannot download model - Network issue or offline mode")
            print("="*60)
            print("\nSolutions:")
            print("1. Connect to internet and retry")
            print("2. Download model manually from:")
            print("   https://huggingface.co/openai")
            print("3. Place downloaded model in cache directory:")
            cache_path = cache_dir or os.path.expanduser("~/.cache")
            print(f"   {cache_path}")
            print("\n" + "="*60)
        raise RuntimeError(f"Failed to load CLIP model: {e}") from e


def get_cache_info() -> dict:
    """Get information about model cache locations"""
    home = os.path.expanduser("~")
    
    return {
        'torch_home': os.environ.get('TORCH_HOME', os.path.join(home, '.cache', 'torch')),
        'hf_home': os.environ.get('HF_HOME', os.path.join(home, '.cache', 'huggingface')),
        'open_clip': os.path.join(home, '.cache', 'open_clip'),
    }


def print_cache_info():
    """Print cache directory information"""
    info = get_cache_info()
    
    print("\n" + "="*60)
    print("Model Cache Information")
    print("="*60)
    
    for name, path in info.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{exists} {name}: {path}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test loading
    print_cache_info()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    try:
        model, preprocess, tokenizer = load_clip_model(device=device)
        print("\n[OK] Test successful!")
    except Exception as e:
        print(f"\n[FAIL] {e}")
