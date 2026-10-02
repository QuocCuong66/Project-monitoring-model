"""Multi-Camera Stream Manager for multi-room / multi-angle surveillance scalability."""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from src.camera.stream import CameraStream, FramePacket
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    source_type: str  # "webcam", "rtsp", "video"
    source_uri: str | int
    enabled: bool = True


class MultiCameraManager:
    """Orchestrates multiple concurrent camera feeds (RTSP, Webcams, Videos)."""

    def __init__(self, camera_configs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.streams: Dict[str, CameraStream] = {}
        self.configs: Dict[str, CameraConfig] = {}
        self._lock = threading.Lock()

        if camera_configs:
            for cfg in camera_configs:
                self.add_camera(
                    camera_id=str(cfg.get("id", f"cam_{len(self.streams)+1}")),
                    name=str(cfg.get("name", "Camera")),
                    source_type=str(cfg.get("type", "webcam")),
                    source_uri=cfg.get("uri", 0),
                    enabled=bool(cfg.get("enabled", True)),
                )

    def add_camera(
        self,
        camera_id: str,
        name: str,
        source_type: str,
        source_uri: str | int,
        enabled: bool = True,
    ) -> bool:
        """Register and start a new camera source stream."""
        with self._lock:
            if camera_id in self.streams:
                return False

            cam_cfg = CameraConfig(
                camera_id=camera_id,
                name=name,
                source_type=source_type,
                source_uri=source_uri,
                enabled=enabled,
            )
            self.configs[camera_id] = cam_cfg

            if not enabled:
                return True

            stream = self._create_stream(source_type, source_uri)
            if stream.start():
                self.streams[camera_id] = stream
                return True
            return False

    def _create_stream(self, source_type: str, source_uri: str | int) -> CameraStream:
        if source_type.lower() == "webcam":
            dev_idx = int(source_uri) if str(source_uri).isdigit() else 0
            return WebcamStream(device_index=dev_idx, max_queue_size=1)
        elif source_type.lower() == "rtsp":
            stream = RTSPStream.from_config(str(source_uri), max_queue_size=1)
            stream.allow_offline_start = True
            return stream
        elif source_type.lower() == "video":
            return VideoFileStream(video_path=str(source_uri), loop=True, max_queue_size=1)
        else:
            raise ValueError(f"Unsupported camera type: {source_type}")

    def read_all(self, timeout: float = 0.5) -> Dict[str, FramePacket]:
        """Fetch latest frame packet from each active camera stream."""
        results: Dict[str, FramePacket] = {}
        with self._lock:
            cam_items = list(self.streams.items())

        for cam_id, stream in cam_items:
            packet = stream.read(timeout=timeout)
            if packet is not None:
                packet.source_name = self.configs[cam_id].name
                results[cam_id] = packet

        return results

    def create_grid_display(
        self,
        frames_dict: Dict[str, np.ndarray],
        cell_size: Tuple[int, int] = (640, 360),
    ) -> np.ndarray:
        """Compose multiple camera frames into a consolidated grid layout."""
        if not frames_dict:
            return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

        n = len(frames_dict)
        # Determine grid cols & rows
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))

        grid_h = rows * cell_size[1]
        grid_w = cols * cell_size[0]
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        idx = 0
        for cam_id, frame in frames_dict.items():
            r = idx // cols
            c = idx % cols
            y1 = r * cell_size[1]
            y2 = y1 + cell_size[1]
            x1 = c * cell_size[0]
            x2 = x1 + cell_size[0]

            resized = cv2.resize(frame, cell_size, interpolation=cv2.INTER_AREA)

            # Draw camera label banner
            name = self.configs.get(cam_id, CameraConfig(cam_id, cam_id, "unknown", 0)).name
            cv2.putText(resized, f"CAM: {name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

            canvas[y1:y2, x1:x2] = resized
            idx += 1

        return canvas

    def stop_all(self) -> None:
        """Gracefully terminate all running camera streams."""
        with self._lock:
            for stream in self.streams.values():
                stream.release()
            self.streams.clear()
