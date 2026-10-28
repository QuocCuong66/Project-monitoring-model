"""Model Factory and Registry for Human Pose Estimators."""

from typing import Callable, Dict, Any
from src.pose.pose_estimator import PoseEstimator, YOLOv8PoseEstimator
from src.pose.rtmo_estimator import RTMOPoseEstimator

_POSE_REGISTRY: Dict[str, Callable[..., PoseEstimator]] = {}


def register_pose_estimator(name: str) -> Callable:
    """Decorator to register a custom PoseEstimator subclass."""
    def decorator(cls: Callable[..., PoseEstimator]) -> Callable[..., PoseEstimator]:
        _POSE_REGISTRY[name.lower()] = cls
        return cls
    return decorator


register_pose_estimator("yolov8_pose")(YOLOv8PoseEstimator)
register_pose_estimator("yolov11_pose")(YOLOv8PoseEstimator)
register_pose_estimator("rtmo_s")(RTMOPoseEstimator)
register_pose_estimator("rtmo_pose")(RTMOPoseEstimator)


def create_pose_estimator(architecture: str = "yolov8_pose", **kwargs: Any) -> PoseEstimator:
    """Instantiate a PoseEstimator matching the architecture name from configs/model.yaml."""
    arch_key = architecture.lower()
    if arch_key not in _POSE_REGISTRY:
        available = list(_POSE_REGISTRY.keys())
        raise ValueError(f"Unknown pose architecture '{architecture}'. Available: {available}")

    return _POSE_REGISTRY[arch_key](**kwargs)
