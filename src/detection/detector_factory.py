"""Model Factory and Registry for Person Detectors. Allows swapping models without altering pipeline code."""

from typing import Callable, Dict, Any
from src.detection.detector import PersonDetector
from src.detection.person_detector import YOLOv8PersonDetector

_DETECTOR_REGISTRY: Dict[str, Callable[..., PersonDetector]] = {}


def register_detector(name: str) -> Callable:
    """Decorator to register a custom PersonDetector subclass."""
    def decorator(cls: Callable[..., PersonDetector]) -> Callable[..., PersonDetector]:
        _DETECTOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


# Register built-in detectors
register_detector("yolov8")(YOLOv8PersonDetector)
register_detector("yolov11")(YOLOv8PersonDetector)


class RTDETRPersonDetector(PersonDetector):
    """RT-DETR implementation via Ultralytics."""

    def __init__(self, model_path: str = "rtdetr-l.pt", conf_threshold: float = 0.40, device: str = "cuda", **kwargs) -> None:
        self.conf_threshold = conf_threshold
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if device == "cuda" and not has_cuda:
            device = "cpu"
        self.device = device
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(model_path)
        except Exception:
            self.model = None

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, classes=[0], device=self.device, verbose=False)
        from src.detection.detector import Detection
        detections = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                coords = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                detections.append(Detection(bbox=tuple(coords), confidence=conf, class_id=0))
        return detections

register_detector("rtdetr")(RTDETRPersonDetector)


def create_detector(architecture: str = "yolov8", **kwargs: Any) -> PersonDetector:
    """Instantiate a PersonDetector matching the architecture name from configs/model.yaml."""
    arch_key = architecture.lower()
    if arch_key not in _DETECTOR_REGISTRY:
        available = list(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector architecture '{architecture}'. Available: {available}")

    return _DETECTOR_REGISTRY[arch_key](**kwargs)
