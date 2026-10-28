"""YOLOv8-based Person Detector implementation."""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from src.detection.detector import Detection, PersonDetector

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLOv8PersonDetector(PersonDetector):
    """Person detector using Ultralytics YOLOv8/v11."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8s.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        device: str = "cuda",
        half: bool = True,
        img_size: int = 640,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
            try:
                import onnxruntime as ort
                has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            except ImportError:
                has_cuda = False

        if device == "cuda" and not has_cuda:
            device = "cpu"
        self.device = device
        self.half = half and (device != "cpu")
        self.img_size = img_size
        self.model = None

        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                # Log warning, allow mock fallback
                print(f"[Warning] Could not initialize YOLO from {model_path}: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a frame."""
        if self.model is None or not ULTRALYTICS_AVAILABLE:
            return []

        predict_kwargs = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": [0],
            "device": self.device,
            "imgsz": self.img_size,
            "verbose": False,
        }
        if self.half and self.device != "cpu":
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(coords, confs, classes):
            if cls_id == 0:  # double-check person class
                detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=float(conf),
                        class_id=0,
                    )
                )

        return detections
