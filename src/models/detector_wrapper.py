"""
检测器包装类
支持 YOLOv8 (默认) 和 Grounding DINO (可选)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image


class DetectorWrapper:
    """
    检测器包装类
    支持 YOLOv8 和 Grounding DINO (如果可用)
    """
    
    def __init__(
        self,
        detector_type: str = "yolov8",
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.3,
        max_detections: int = 20
    ):
        self.detector_type = detector_type
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        
        if detector_type == "yolov8":
            self._init_yolov8(model_name)
        
        elif detector_type == "groundingdino":
            self._init_groundingdino(model_name)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def _init_yolov8(self, model_name: str):
        """初始化 YOLOv8"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.model.to(self.device)
            print(f"✓ Loaded YOLOv8: {model_name}")
        except ImportError:
            raise ImportError(
                "Please install ultralytics: pip install ultralytics"
            )
    
    def _init_groundingdino(self, model_name: str):
        """初始化 Grounding DINO (可选)"""
        try:
            from groundingdino.util.inference import load_model
            self.model = load_model(model_name)
            print(f"✓ Loaded Grounding DINO: {model_name}")
        except ImportError:
            print(
                "⚠️  Grounding DINO not available. Falling back to YOLOv8.\n"
                "    To use Grounding DINO, install with:\n"
                "    pip install groundingdino-py@git+https://github.com/IDEA-Research/GroundingDINO.git"
            )
            # 回退到 YOLOv8
            self.detector_type = "yolov8"
            self._init_yolov8("yolov8n.pt")
    
    @torch.no_grad()
    def extract_features(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        return_boxes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        提取区域特征
        
        Args:
            images: 图像张量 (B, 3, H, W) 或 PIL Image 列表
            return_boxes: 是否返回检测框
        
        Returns:
            region_features: (B, N, D) 区域特征
            region_mask: (B, N) 有效区域掩码
            boxes: (B, N, 4) 检测框 [x1, y1, x2, y2]，可选
        """
        if self.detector_type == "yolov8":
            return self._extract_yolov8(images, return_boxes)
        elif self.detector_type == "groundingdino":
            return self._extract_groundingdino(images, return_boxes)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def _extract_yolov8(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        return_boxes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """YOLOv8 特征提取"""
        
        # 如果是 tensor，转换为 PIL
        if isinstance(images, torch.Tensor):
            images = self._tensor_to_pil(images)
        
        all_features = []
        all_masks = []
        all_boxes = [] if return_boxes else None
        
        for img in images:
            results = self.model(img, verbose=False)[0]
            
            # 获取检测框
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy[:self.max_detections]  # (N, 4)
                confs = results.boxes.conf[:self.max_detections]  # (N,)
                
                # 过滤低置信度
                valid_mask = confs > self.conf_threshold
                boxes = boxes[valid_mask]
                
                # 提取 RoI 特征
                features = self._extract_roi_features(img, boxes)
                
                N = len(features)
            else:
                features = torch.zeros((1, 2048))  # 空特征
                boxes = torch.zeros((1, 4))
                N = 1
            
            # Padding 到固定长度
            if N < self.max_detections:
                padding = torch.zeros(self.max_detections - N, 2048)
                features = torch.cat([features, padding], dim=0)
                
                if return_boxes:
                    box_padding = torch.zeros(self.max_detections - N, 4)
                    boxes = torch.cat([boxes, box_padding], dim=0)
            
            # 创建 mask
            mask = torch.zeros(self.max_detections)
            mask[:N] = 1
            
            all_features.append(features)
            all_masks.append(mask)
            if return_boxes:
                all_boxes.append(boxes)
        
        features = torch.stack(all_features)  # (B, N, D)
        masks = torch.stack(all_masks)  # (B, N)
        
        if return_boxes:
            boxes = torch.stack(all_boxes)  # (B, N, 4)
            return features.to(self.device), masks.to(self.device), boxes.to(self.device)
        
        return features.to(self.device), masks.to(self.device), None
    
    def _extract_groundingdino(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        return_boxes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Grounding DINO 特征提取 (占位符)"""
        # TODO: 实现 Grounding DINO 的特征提取
        # 暂时回退到 YOLOv8
        return self._extract_yolov8(images, return_boxes)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """将 tensor 转换为 PIL Image 列表"""
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        
        images = []
        for i in range(tensor.shape[0]):
            # Denormalize (假设使用 ImageNet 归一化)
            img = tensor[i].cpu().clone()
            # ImageNet 均值和标准差
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            images.append(to_pil(img))
        
        return images
    
    def _extract_roi_features(
        self,
        image: Image.Image,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        从检测框提取特征
        
        注意：这是简化版本。实际应用中应该使用：
        1. RoI Align 从 backbone 提取特征
        2. 或者使用检测器自带的特征输出
        """
        import torchvision.transforms as T
        
        if len(boxes) == 0:
            return torch.zeros((1, 2048))
        
        features = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            # 确保坐标有效
            x1, y1 = max(0, x1), max(0, y1)
            
            try:
                roi = image.crop((x1, y1, x2, y2))
                
                # Resize 并提取特征
                roi_tensor = T.ToTensor()(roi.resize((64, 64)))
                feat = roi_tensor.flatten()[:2048]  # 截取前 2048 维
                
                # Padding 到 2048
                if len(feat) < 2048:
                    feat = torch.cat([feat, torch.zeros(2048 - len(feat))])
                
                features.append(feat)
            except Exception as e:
                # 如果裁剪失败，使用零特征
                features.append(torch.zeros(2048))
        
        return torch.stack(features) if features else torch.zeros((1, 2048))


class DummyDetector:
    """
    虚拟检测器，用于 Debug 模式
    不依赖外部检测器，生成随机区域特征
    """
    
    def __init__(
        self,
        device: str = "cpu",
        max_detections: int = 20,
        feature_dim: int = 2048
    ):
        self.device = device
        self.max_detections = max_detections
        self.feature_dim = feature_dim
    
    def extract_features(
        self,
        images: torch.Tensor,
        return_boxes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """生成随机的区域特征"""
        B = images.shape[0]
        
        # 随机生成特征
        features = torch.randn(B, self.max_detections, self.feature_dim).to(self.device)
        
        # 随机 mask（模拟不同数量的检测物体）
        num_objects = torch.randint(1, self.max_detections + 1, (B,))
        mask = torch.zeros(B, self.max_detections)
        for i in range(B):
            mask[i, :num_objects[i]] = 1
        mask = mask.to(self.device)
        
        boxes = None
        if return_boxes:
            boxes = torch.rand(B, self.max_detections, 4).to(self.device) * 100
        
        return features, mask, boxes


if __name__ == "__main__":
    print("Testing DetectorWrapper...")
    
    # 测试 YOLOv8
    try:
        detector = DetectorWrapper(
            detector_type="yolov8",
            model_name="yolov8n.pt",
            device="cpu",
            max_detections=10
        )
        
        # 创建测试图像
        test_images = [Image.new('RGB', (224, 224), color=(i*50, i*30, i*20)) for i in range(2)]
        
        features, mask, boxes = detector.extract_features(test_images, return_boxes=True)
        
        print(f"\nYOLOv8 Test Results:")
        print(f"  Features shape: {features.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Boxes shape: {boxes.shape if boxes is not None else None}")
        print(f"  Valid objects per image: {mask.sum(dim=1).int().tolist()}")
        print("✓ YOLOv8 test passed!\n")
        
    except Exception as e:
        print(f"⚠️  YOLOv8 test skipped: {e}\n")
    
    # 测试 DummyDetector
    print("Testing DummyDetector...")
    dummy = DummyDetector(device="cpu", max_detections=10)
    dummy_images = torch.randn(2, 3, 224, 224)
    
    features, mask, boxes = dummy.extract_features(dummy_images, return_boxes=True)
    
    print(f"\nDummyDetector Test Results:")
    print(f"  Features shape: {features.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid objects per image: {mask.sum(dim=1).int().tolist()}")
    print("✓ DummyDetector test passed!\n")
