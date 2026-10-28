"""RTMO-s One-Stage Multi-Person Pose Estimation using ONNX Runtime.

High-throughput, real-time pose estimation extracting simultaneously:
- Person Bounding Boxes (dets)
- 17 COCO Human Skeletal Keypoints (keypoints)
Optimized with GPU/CPU provider management and vectorized unscaling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

try:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    cudart = None
    TRT_AVAILABLE = False

from src.detection.detector import Detection
from src.tracking.tracker import Track
from src.tracking.track_manager import bbox_iou_matrix
from src.pose.pose_estimator import (
    PoseEstimator,
    PoseResult,
    SinglePassDetectionItem,
)
from src.pose.keypoints import normalize_keypoints, compute_torso_angle
from src.utils.config import resolve_model_path
from src.utils.ort_providers import dml_provider


logger = logging.getLogger(__name__)


class RTMOPoseEstimator(PoseEstimator):
    """One-Stage Multi-Person Pose Estimator using RTMO-s ONNX Runtime or TensorRT engine.
    
    Extracts both detection bounding boxes and 17 COCO keypoints in a single forward pass.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pose/rtmo-s.onnx",
        conf_threshold: float = 0.30,
        img_size: Tuple[int, int] = (640, 640),
        device: str = "cuda",
        gpu_mem_limit_gb: float = 2.0,
        half: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RTMO-s with TensorRT engine or ONNX Runtime InferenceSession.

        Args:
            model_path: Path to the rtmo-s.onnx or rtmo-s_int8.engine model file.
            conf_threshold: Minimum confidence score to retain detections.
            img_size: Input tensor dimensions (width, height), default (640, 640).
            device: 'cuda' or 'cpu'.
            gpu_mem_limit_gb: GPU memory limit in GB for CUDAExecutionProvider.
        """
        resolved = resolve_model_path(model_path)
        self.model_path = resolved if resolved is not None else Path(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device.lower()
        self.gpu_mem_limit_gb = gpu_mem_limit_gb
        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: str = "None"
        self.input_name: str = "input"
        self.output_names: List[str] = ["dets", "keypoints"]

        # Reusable letterbox buffers (see preprocess)
        self._letterbox_key: Optional[Tuple[int, int]] = None
        self._letterbox_geom: Tuple[float, int, int, int, int] = (1.0, 0, 0, 0, 0)
        self._canvas: Optional[np.ndarray] = None
        self._tensor: Optional[np.ndarray] = None

        # TensorRT attributes
        self.is_tensorrt: bool = False
        self.trt_engine = None
        self.trt_context = None
        self.trt_stream = None
        self.d_in = None
        self.d_dets = None
        self.d_kps = None
        self.h_dets = None
        self.h_kps = None

        if self.model_path.suffix.lower() == ".engine":
            self._initialize_tensorrt_engine()
        else:
            self._initialize_session()

    def _initialize_tensorrt_engine(self) -> None:
        """Initialize TensorRT CUDA engine and execution context."""
        if not TRT_AVAILABLE:
            logger.error("[RTMO-s] TensorRT or cuda.bindings is not installed.")
            return

        if not self.model_path.exists():
            logger.error(f"[RTMO-s] Engine file not found at: {self.model_path.resolve()}")
            return

        try:
            logger_trt = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, "rb") as f, trt.Runtime(logger_trt) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.trt_engine is None:
                logger.error("[RTMO-s] Failed to deserialize TensorRT CUDA engine.")
                return

            self.trt_context = self.trt_engine.create_execution_context()
            tw, th = self.img_size
            input_size = 1 * 3 * th * tw * 4
            dets_size = 1 * 100 * 5 * 4
            kps_size = 1 * 100 * 17 * 3 * 4

            _, self.d_in = cudart.cudaMalloc(input_size)
            _, self.d_dets = cudart.cudaMalloc(dets_size)
            _, self.d_kps = cudart.cudaMalloc(kps_size)
            _, self.trt_stream = cudart.cudaStreamCreate()

            self.trt_context.set_tensor_address("input", int(self.d_in))
            self.trt_context.set_tensor_address("dets", int(self.d_dets))
            self.trt_context.set_tensor_address("keypoints", int(self.d_kps))

            self.h_dets = np.empty((1, 100, 5), dtype=np.float32)
            self.h_kps = np.empty((1, 100, 17, 3), dtype=np.float32)
            self.is_tensorrt = True
            self.active_provider = "TensorRT"
            logger.info(
                f"[RTMO-s] Initialized TensorRT engine with provider: 'TensorRT' "
                f"(Model: {self.model_path.name})"
            )
            print(f"[RTMO-s] Active Provider: TensorRT | Model: {self.model_path.name}")
        except Exception as e:
            logger.error(f"[RTMO-s] Failed to initialize TensorRT engine: {e}")

    def _initialize_session(self) -> None:
        """Configure ExecutionProviders and build the ONNX Runtime session."""
        if not ORT_AVAILABLE:
            logger.error("onnxruntime is not installed. RTMOPoseEstimator cannot initialize.")
            return

        if not self.model_path.exists():
            logger.error(f"Model file not found at: {self.model_path.resolve()}")
            return

        available_providers = ort.get_available_providers()
        providers = []

        if self.device == "cuda":
            if "DmlExecutionProvider" in available_providers:
                providers.append(dml_provider())
            elif "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": int(self.gpu_mem_limit_gb * 1024 * 1024 * 1024),
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))

        # Always add CPUExecutionProvider as fallback
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Single-frame inference: pinning the dynamic batch dim lets ORT/DirectML
        # pre-plan static shapes (~15% faster per frame on DirectML).
        sess_options.add_free_dimension_override_by_name("batch", 1)
        if providers and providers[0][0] == "DmlExecutionProvider":
            # DirectML does not support memory-pattern optimization (ORT requirement)
            sess_options.enable_mem_pattern = False

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
            self.active_provider = self.session.get_providers()[0]
            logger.info(
                f"[RTMO-s] Initialized session with provider: '{self.active_provider}' "
                f"(Model: {self.model_path.name})"
            )
            print(f"[RTMO-s] Active Provider: {self.active_provider} | Model: {self.model_path.name}")
            
            inputs = self.session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name
            outputs = self.session.get_outputs()
            if outputs:
                self.output_names = [o.name for o in outputs]
        except Exception as e:
            logger.error(f"[RTMO-s] Failed to initialize ONNX Runtime session: {e}")
            self.session = None

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Letterbox resize input image to target dimensions while maintaining aspect ratio.

        Args:
            image: BGR image of shape (H, W, 3).

        Returns:
            tensor: Preprocessed tensor of shape (1, 3, target_h, target_w) float32 BGR [0..255].
            scale_ratio: Rescaling factor applied to the original image.
            pad_w: Horizontal padding offset (left).
            pad_h: Vertical padding offset (top).
        """
        orig_h, orig_w = image.shape[:2]
        tw, th = self.img_size

        # Letterbox geometry only changes when the input resolution changes; cache it
        # together with the preallocated canvas/tensor so every frame avoids ~4 MB of
        # fresh allocations (copyMakeBorder + transpose + astype ≈ 4.4 ms -> ~0.9 ms).
        if self._letterbox_key != (orig_w, orig_h):
            scale_ratio = min(tw / orig_w, th / orig_h)
            new_unpad_w = int(round(orig_w * scale_ratio))
            new_unpad_h = int(round(orig_h * scale_ratio))
            pad_h_top = int(round((th - new_unpad_h) / 2.0 - 0.1))
            pad_w_left = int(round((tw - new_unpad_w) / 2.0 - 0.1))
            self._letterbox_key = (orig_w, orig_h)
            self._letterbox_geom = (scale_ratio, new_unpad_w, new_unpad_h, pad_w_left, pad_h_top)
            self._canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
            self._tensor = np.empty((1, 3, th, tw), dtype=np.float32)

        scale_ratio, new_unpad_w, new_unpad_h, pad_w_left, pad_h_top = self._letterbox_geom
        roi = self._canvas[pad_h_top:pad_h_top + new_unpad_h, pad_w_left:pad_w_left + new_unpad_w]
        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            roi[...] = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            roi[...] = image

        # HWC uint8 -> NCHW float32 written into the reused buffer
        np.copyto(self._tensor[0], self._canvas.transpose(2, 0, 1), casting="unsafe")

        return self._tensor, scale_ratio, float(pad_w_left), float(pad_h_top)

    def postprocess(
        self,
        dets_raw: np.ndarray,
        kps_raw: np.ndarray,
        scale_ratio: float,
        pad_w: float,
        pad_h: float,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[List[Detection], List[SinglePassDetectionItem]]:
        """Vectorized unscaling of Bounding Boxes and 17 Keypoints back to original resolution.

        Args:
            dets_raw: Raw output dets of shape (1, N, 5) -> [x1, y1, x2, y2, score].
            kps_raw: Raw output keypoints of shape (1, N, 17, 3) -> [x, y, conf].
            scale_ratio: Scale factor from preprocessing.
            pad_w: Left padding offset.
            pad_h: Top padding offset.
            orig_w: Width of original input frame.
            orig_h: Height of original input frame.

        Returns:
            detections: List[Detection] for ByteTrack.
            items: List[SinglePassDetectionItem] containing unscaled boxes and skeletons.
        """
        if dets_raw.size == 0 or kps_raw.size == 0:
            return [], []

        d = dets_raw[0]  # shape: (N, 5)
        k = kps_raw[0]   # shape: (N, 17, 3)

        # Vectorized confidence filtering
        mask = d[:, 4] >= self.conf_threshold
        if not np.any(mask):
            return [], []

        valid_dets = d[mask].copy()
        valid_kps = k[mask].copy()

        # Vectorized unscaling: subtract pad, divide by scale_ratio
        valid_dets[:, [0, 2]] = (valid_dets[:, [0, 2]] - pad_w) / scale_ratio
        valid_dets[:, [1, 3]] = (valid_dets[:, [1, 3]] - pad_h) / scale_ratio

        # Clip bounding boxes to image boundaries
        valid_dets[:, [0, 2]] = np.clip(valid_dets[:, [0, 2]], 0.0, float(orig_w))
        valid_dets[:, [1, 3]] = np.clip(valid_dets[:, [1, 3]], 0.0, float(orig_h))

        # Vectorized keypoints unscaling
        valid_kps[:, :, 0] = (valid_kps[:, :, 0] - pad_w) / scale_ratio
        valid_kps[:, :, 1] = (valid_kps[:, :, 1] - pad_h) / scale_ratio
        valid_kps[:, :, 0] = np.clip(valid_kps[:, :, 0], 0.0, float(orig_w))
        valid_kps[:, :, 1] = np.clip(valid_kps[:, :, 1], 0.0, float(orig_h))

        detections: List[Detection] = []
        items: List[SinglePassDetectionItem] = []

        for i in range(len(valid_dets)):
            b = valid_dets[i]
            conf = float(b[4])
            bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            kp = valid_kps[i]

            detections.append(Detection(bbox=bbox, confidence=conf, class_id=0))
            items.append(SinglePassDetectionItem(bbox=bbox, confidence=conf, keypoints=kp, class_id=0))

        return detections, items

    def detect_and_estimate_single_pass(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Detection], List[SinglePassDetectionItem]]:
        """Single-pass perception: Run RTMO-s ONCE to get both Detections and 17 Keypoints.

        Args:
            frame: Input image (H, W, 3) in BGR.

        Returns:
            detections: List[Detection] for ByteTrack
            items: List[SinglePassDetectionItem] with unscaled coordinates.
        """
        if not self.is_tensorrt and self.session is None:
            return [], []
        if frame is None or frame.size == 0:
            return [], []

        orig_h, orig_w = frame.shape[:2]
        tensor, scale_ratio, pad_w, pad_h = self.preprocess(frame)

        if self.is_tensorrt:
            try:
                tw, th = self.img_size
                input_size = 1 * 3 * th * tw * 4
                dets_size = 1 * 100 * 5 * 4
                kps_size = 1 * 100 * 17 * 3 * 4

                cudart.cudaMemcpyAsync(
                    self.d_in, tensor.ctypes.data, input_size,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream
                )
                self.trt_context.execute_async_v3(self.trt_stream)
                cudart.cudaMemcpyAsync(
                    self.h_dets.ctypes.data, self.d_dets, dets_size,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream
                )
                cudart.cudaMemcpyAsync(
                    self.h_kps.ctypes.data, self.d_kps, kps_size,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream
                )
                cudart.cudaStreamSynchronize(self.trt_stream)
                dets_raw = self.h_dets
                kps_raw = self.h_kps
            except Exception as e:
                logger.error(f"[RTMO-s TensorRT] Inference failed: {e}")
                return [], []
        else:
            try:
                outputs = self.session.run(None, {self.input_name: tensor})
                dets_raw = outputs[0]
                kps_raw = outputs[1]
            except Exception as e:
                logger.error(f"[RTMO-s] Inference failed: {e}")
                return [], []

        return self.postprocess(
            dets_raw=dets_raw,
            kps_raw=kps_raw,
            scale_ratio=scale_ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_w=orig_w,
            orig_h=orig_h,
        )

    def estimate(self, frame: np.ndarray, tracks: List[Track]) -> List[PoseResult]:
        """Estimate human poses for active tracks in the given frame using RTMO-s.

        Args:
            frame: Input image (H, W, 3) in BGR.
            tracks: List of confirmed person tracks.

        Returns:
            List of PoseResult objects matching tracks.
        """
        if not tracks or (not self.is_tensorrt and self.session is None):
            return []

        detections, single_pass_items = self.detect_and_estimate_single_pass(frame)
        if not single_pass_items:
            return []

        matched_kp_map = self.match_tracks_to_keypoints(tracks, single_pass_items, iou_threshold=0.70)
        pose_results: List[PoseResult] = []

        for track in tracks:
            tid = track.track_id
            raw_kp = matched_kp_map.get(tid)
            if raw_kp is not None:
                norm_kp = normalize_keypoints(raw_kp, bbox_fallback=track.bbox)
                torso_angle = compute_torso_angle(raw_kp)
                mean_conf = float(np.mean(raw_kp[:, 2]))
                pose_results.append(
                    PoseResult(
                        track_id=tid,
                        keypoints=raw_kp,
                        normalized_keypoints=norm_kp,
                        torso_angle=torso_angle,
                        confidence=mean_conf,
                    )
                )

        return pose_results

    @staticmethod
    def match_tracks_to_keypoints(
        active_tracks: List[Track],
        items: List[SinglePassDetectionItem],
        iou_threshold: float = 0.70,
    ) -> Dict[int, np.ndarray]:
        """Strict IoU Matching (>= 0.70) between ByteTrack tracks and RTMO-s detection bboxes."""
        if not active_tracks or not items:
            return {}

        matched: Dict[int, np.ndarray] = {}
        used_item_indices = set()

        iou_matrix = bbox_iou_matrix(
            np.array([t.bbox for t in active_tracks], dtype=np.float32),
            np.array([it.bbox for it in items], dtype=np.float32),
        )

        flat_indices = np.argsort(-iou_matrix, axis=None)
        for flat_idx in flat_indices:
            t_idx, i_idx = np.unravel_index(flat_idx, iou_matrix.shape)
            best_iou = float(iou_matrix[t_idx, i_idx])
            if best_iou < iou_threshold:
                break

            tid = active_tracks[t_idx].track_id
            if tid not in matched and i_idx not in used_item_indices:
                matched[tid] = items[i_idx].keypoints
                used_item_indices.add(i_idx)

        return matched

    def __del__(self) -> None:
        """Release CUDA device memory if allocated for TensorRT engine."""
        if getattr(self, "is_tensorrt", False) and TRT_AVAILABLE and cudart is not None:
            try:
                if self.d_in is not None:
                    cudart.cudaFree(self.d_in)
                    self.d_in = None
                if self.d_dets is not None:
                    cudart.cudaFree(self.d_dets)
                    self.d_dets = None
                if self.d_kps is not None:
                    cudart.cudaFree(self.d_kps)
                    self.d_kps = None
                if self.trt_stream is not None:
                    cudart.cudaStreamDestroy(self.trt_stream)
                    self.trt_stream = None
            except Exception:
                pass

