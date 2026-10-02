"""Decoupled Multi-Threaded Real-Time Video Intelligence Pipeline.

Separates Render Thread (30-60 FPS smooth video) from AI Inference Thread.
Perception + multi-object tracking run on every fresh camera frame (pose_stride=1 -> ~30 FPS),
while temporal reasoning / risk evaluation is sampled at a fixed rate (temporal_hz, default 15 Hz)
so window sizes, hold-on timers and kinematic thresholds keep their calibrated meaning.
"""

import time
import logging
import traceback
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import cv2

from src.camera.stream import CameraStream, FramePacket
from src.detection.detector import Detection
from src.tracking.tracker import Track
from src.tracking.track_manager import ByteTrackManager
from src.pose.pose_estimator import YOLOv8PoseEstimator, PoseResult, translate_keypoints
from src.pose.one_euro_filter import KeypointOneEuroFilter
from src.pose.keypoints import normalize_keypoints, compute_torso_angle
from src.temporal.sequence_buffer import SequenceBuffer, TrackSnapshot
from src.temporal.sequence_features import build_model_sequence
from src.temporal.temporal_features import TemporalFeatureExtractor, KinematicFeatures
from src.temporal.gated_trigger import KinematicGatedTrigger
from src.action.action_classifier import ActionClassifier, ActionPrediction
from src.action.behavior_classifier import BehaviorSequenceAnalyzer
from src.anomaly.anomaly_score import ReconstructionAnomalyScorer, AnomalyResult
from src.risk.risk_engine import RiskEngine, RiskAssessment
from src.risk.threshold_manager import ThresholdManager
from src.alerts.alert_manager import AlertManager, AlertEvent
from src.detection.motion_gater import MotionGater, MotionGateResult
from src.utils.visualizer import PipelineVisualizer
from src.utils.privacy import anonymize_alert_snapshot, anonymize_people
from src.utils.profiler import LatencyProfiler, ResourceMonitor
from src.utils.config import load_config, resolve_model_path

logger = logging.getLogger(__name__)



@dataclass
class InterpolatedTrackState:
    """Represents an interpolated person track state for high-FPS rendering."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[np.ndarray]
    torso_angle: float
    action: str
    risk_score: float
    state: str


class SharedRenderState:
    """Thread-safe state container shared between AI Worker and Render Thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_ai_timestamp: float = 0.0
        self.tracks: Dict[int, Track] = {}
        self.poses: Dict[int, PoseResult] = {}
        self.assessments: Dict[int, RiskAssessment] = {}
        self.latest_alerts: List[AlertEvent] = []
        self.ai_fps: float = 0.0

    def update_from_ai(
        self,
        tracks: List[Track],
        poses: Dict[int, PoseResult],
        assessments: List[RiskAssessment],
        alerts: List[AlertEvent],
        ai_fps: float,
        timestamp: float,
    ) -> None:
        with self.lock:
            self.tracks = {t.track_id: t for t in tracks}
            self.poses = poses
            self.assessments = {a.track_id: a for a in assessments}
            if alerts:
                self.latest_alerts.extend(alerts)
                self.latest_alerts = self.latest_alerts[-10:]  # Keep last 10 alerts
            self.ai_fps = ai_fps
            self.last_ai_timestamp = timestamp

    def update(
        self,
        active_tracks: Optional[List[Track]] = None,
        poses_dict: Optional[Dict[int, PoseResult]] = None,
        assessments: Optional[Dict[int, RiskAssessment]] = None,
        ai_latency_ms: float = 0.0,
        ai_fps: float = 0.0,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update telemetry and state safely without erasing existing tracks on throttled frames."""
        with self.lock:
            if active_tracks:
                self.tracks = {t.track_id: t for t in active_tracks}
            if poses_dict:
                self.poses = poses_dict
            if assessments:
                self.assessments = assessments
            if ai_fps > 0:
                self.ai_fps = ai_fps


    def get_render_data(
        self,
        curr_time: float,
    ) -> Tuple[List[Track], List[PoseResult], List[RiskAssessment], float]:
        """Retrieve tracks and poses with linear Kalman motion extrapolation."""
        with self.lock:
            tracks_copy = list(self.tracks.values())
            poses_copy = list(self.poses.values())
            assessments_copy = list(self.assessments.values())
            ai_fps = self.ai_fps

        return tracks_copy, poses_copy, assessments_copy, ai_fps


class DecoupledPipeline:
    """High-Throughput Decoupled Pipeline orchestrating asynchronous AI Inference and smooth 60 FPS Rendering."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        img_size: int = 480,
        device: str = "cuda",
        half: bool = True,
        ai_stride: Optional[int] = None,  # Perception every N camera frames (None -> inference.yaml scheduling.pose_stride)
        anomaly_interval: int = 5,  # Run heavy Autoencoder every 5 temporal ticks
        action_classifier: Optional[ActionClassifier] = None,
        anomaly_detector: Optional[ReconstructionAnomalyScorer] = None,
    ) -> None:
        self.anomaly_interval = anomaly_interval

        # Load configs
        model_cfg = load_config("model.yaml")
        infer_cfg = load_config("inference.yaml")
        sched_cfg = infer_cfg.get("scheduling", {})
        self.ai_stride = max(1, int(ai_stride if ai_stride is not None else sched_cfg.get("pose_stride", 1)))
        # Temporal/risk reasoning cadence. The temporal models, sequence window (30 frames ~ 2s),
        # gate hold-on timers and kinematic features are calibrated for 15 Hz sampling.
        self.temporal_hz = float(sched_cfg.get("temporal_hz", 15.0))
        self.gated_ai = bool(sched_cfg.get("enable_gated_ai", True))
        # Live-view face blur (toggled from Settings via /api/camera/blur); alert snapshots are always anonymized
        self.blur_faces = bool(infer_cfg.get("privacy", {}).get("blur_faces", False))
        self._temporal_period = 1.0 / max(1e-3, self.temporal_hz)

        # GPU = an onnxruntime GPU provider (DirectML in the app build) or a working torch CUDA.
        # torch may be absent or partial in the packaged app, so any failure there means "no".
        has_cuda = False
        try:
            import onnxruntime as ort
            gpu_providers = {"CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"}
            has_cuda = bool(gpu_providers.intersection(ort.get_available_providers()))
        except ImportError:
            pass
        if not has_cuda:
            try:
                import torch
                has_cuda = bool(torch.cuda.is_available())
            except (ImportError, AttributeError):
                pass

        if device == "cuda" and not has_cuda:
            device = "cpu"
        self.device = device
        self.half = half and (device != "cpu")

        # 1. Single-Pass Pose & Detection Model
        pose_cfg = model_cfg.get("pose", {})
        arch = pose_cfg.get("architecture", "yolov8_pose")
        model_str = str(model_name)
        if model_str.endswith(".engine") or model_str.endswith(".onnx") or arch in ["rtmo_s", "rtmo_pose"]:
            from src.pose.rtmo_estimator import RTMOPoseEstimator
            actual_path = resolve_model_path(model_name)
            if actual_path is None or not actual_path.exists():
                actual_path = resolve_model_path("models/pose/rtmo-s.onnx")
            self.pose_estimator = RTMOPoseEstimator(
                model_path=str(actual_path or model_name),
                conf_threshold=pose_cfg.get("conf_threshold", 0.35),
                device=self.device,
                img_size=(640, 640),
            )
        else:
            self.pose_estimator = YOLOv8PoseEstimator(
                model_path=str(resolve_model_path(model_name) or model_name),
                conf_threshold=pose_cfg.get("conf_threshold", 0.35),
                device=self.device,
                half=self.half,
                img_size=img_size,
            )

        # 2. ByteTrack with Fall Protection
        trk_cfg = model_cfg.get("tracker", {})
        self.tracker = ByteTrackManager(
            track_high_thresh=trk_cfg.get("track_high_thresh", 0.50),
            track_low_thresh=trk_cfg.get("track_low_thresh", 0.15),
            new_track_thresh=trk_cfg.get("new_track_thresh", 0.60),
            match_thresh=trk_cfg.get("match_thresh", 0.70),
            max_lost_frames=trk_cfg.get("track_buffer_frames", 60),
        )

        # 3. One Euro Filter per track ID
        self.keypoint_filters: Dict[int, KeypointOneEuroFilter] = {}

        # 4. Temporal Sequence Buffer & Kinematics
        temp_cfg = model_cfg.get("temporal", {})
        window_size = temp_cfg.get("window_size", 30)
        self.window_size = window_size
        self.sequence_buffer = SequenceBuffer(window_size=window_size)
        self.feature_extractor = TemporalFeatureExtractor(fps=self.temporal_hz)

        # 5. Dual-Stream AI Models (ST-Transformer / TCN & Periodic Autoencoder)
        # Device for these tiny (~2 MB) sequence models is configurable
        # (inference.yaml runtime.temporal_device); falls back to CPU without a GPU.
        temporal_device = infer_cfg.get("runtime", {}).get("temporal_device", "cuda")
        if temporal_device == "cuda" and self.device == "cpu":
            temporal_device = "cpu"
        self.action_classifier = action_classifier or ActionClassifier(
            weights_path=temp_cfg.get("weights_path", None),
            device=temporal_device,
            architecture=temp_cfg.get("architecture", "tcn"),
        )
        anom_cfg = model_cfg.get("anomaly", {})
        self.anomaly_detector = anomaly_detector or ReconstructionAnomalyScorer(
            weights_path=anom_cfg.get("weights_path", None),
            threshold=anom_cfg.get("threshold", 0.22),
            steepness=anom_cfg.get("steepness", 15.0),
            device=temporal_device,
        )
        # Last autoencoder result per track, reused between the periodic evaluations
        self._last_anomaly: Dict[int, AnomalyResult] = {}
        self.behavior_analyzer = BehaviorSequenceAnalyzer()

        # 6. Risk Engine & State Machine
        self.thresholds = ThresholdManager()
        self.risk_engine = RiskEngine(self.thresholds)

        # 6.2. Tier-0 Motion Gating
        motion_cfg = model_cfg.get("motion_gating", {})
        enable_t0 = infer_cfg.get("scheduling", {}).get("enable_tier0_motion_gating", True) and motion_cfg.get("enabled", True)
        self.motion_gater = MotionGater(
            enabled=enable_t0,
            method=motion_cfg.get("method", "mog2"),
            min_motion_ratio=motion_cfg.get("min_motion_ratio", 0.001),
            history=motion_cfg.get("history", 100),
            var_threshold=motion_cfg.get("var_threshold", 16.0),
            detect_shadows=motion_cfg.get("detect_shadows", False),
            cooldown_frames=motion_cfg.get("cooldown_frames", 45),
            idle_fps=motion_cfg.get("idle_fps", 12.0),
            active_fps=motion_cfg.get("active_fps", 30.0),
            idle_stride=motion_cfg.get("idle_stride", 2),
            downsample_size=(motion_cfg.get("downsample_width", 320), motion_cfg.get("downsample_height", 240)),
            periodic_check_interval=motion_cfg.get("periodic_check_interval", 45),
        )

        # 6.5. Tier 1 Kinematic Gated Trigger (Hold-on timer 45 frames ~ 3.0s)

        self.gated_trigger = KinematicGatedTrigger(
            hold_on_frames=45,
            ground_proximity_thresh=0.22,
            area_change_thresh=0.32,
        )
        self.iou_match_thresh = model_cfg.get("perception", {}).get("iou_match_thresh", 0.45)
        self._prev_raw_keypoints: Dict[int, np.ndarray] = {}
        self._prev_track_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        # 7. Alert Manager
        self.alert_manager = AlertManager(self.thresholds)

        # 8. Visualizer & Telemetry
        self.visualizer = PipelineVisualizer(infer_cfg.get("display", {}))
        self.render_profiler = LatencyProfiler()
        self.ai_profiler = LatencyProfiler()
        self.profiler = self.render_profiler  # Standard API compatibility
        self.resource_monitor = ResourceMonitor()

        # Shared state & worker threading
        self.shared_state = SharedRenderState()
        self._running = False
        self._ai_worker_thread: Optional[threading.Thread] = None
        self._latest_camera_packet: Optional[FramePacket] = None
        self._packet_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._ai_frame_count = 0
        self._last_processed_packet: Optional[FramePacket] = None
        self._last_temporal_ts: float = 0.0
        self._temporal_tick_count = 0
        self._last_assessments: Dict[int, RiskAssessment] = {}
        # (keypoints, bbox) of everyone in the last perceived frame, for anonymizing clip frames
        self._last_people: List[Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]] = []

    def start(self) -> None:
        """Start background AI inference thread."""
        self._running = True
        self._ai_worker_thread = threading.Thread(
            target=self._ai_worker_loop,
            daemon=True,
            name="AIInferenceWorker",
        )
        self._ai_worker_thread.start()

    def stop(self) -> None:
        """Stop background worker gracefully."""
        self._running = False
        self._new_frame_event.set()
        if self._ai_worker_thread and self._ai_worker_thread.is_alive():
            self._ai_worker_thread.join(timeout=1.5)

    def submit_frame(self, packet: FramePacket) -> None:
        """Called by Camera ingestion to supply the freshest frame for AI."""
        with self._packet_lock:
            self._latest_camera_packet = packet
        self._new_frame_event.set()

    def _ai_worker_loop(self) -> None:
        """Dedicated background loop running Single-Pass AI Inference at 15-20 FPS."""
        while self._running:
            try:
                self._run_ai_step()
            except Exception as e:
                logger.error(f"[DecoupledPipeline Worker Error]: {traceback.format_exc()}")
                time.sleep(0.05)

    def _run_ai_step(self) -> None:
        """Execute a single perception, tracking, and risk evaluation cycle."""
        # Wait for fresh frame
        self._new_frame_event.wait(timeout=0.1)
        self._new_frame_event.clear()

        with self._packet_lock:
            packet = self._latest_camera_packet

        if packet is None or packet.frame is None:
            return
        # The wait above times out when the camera is slower than the AI; never feed the
        # same frame twice (duplicate timestamps corrupt tracker velocity & kinematics).
        if packet is self._last_processed_packet:
            return
        self._last_processed_packet = packet

        self._ai_frame_count += 1
        # Subsample frames: e.g. process 1 out of every ai_stride frames
        if self._ai_frame_count % self.ai_stride != 0:
            return

        # 0. Tier-0 Motion Gating
        has_active = bool(self.tracker._tracks) if hasattr(self.tracker, "_tracks") else (bool(self.tracker.tracks) if hasattr(self.tracker, "tracks") else False)
        gate_res = self.motion_gater.evaluate(packet.frame, has_active_tracks=has_active)
        if not gate_res.should_run_ai or gate_res.is_throttled_frame:
            self.shared_state.update(
                ai_latency_ms=0.0,
                ai_fps=self.ai_profiler.fps,
                telemetry={
                    "tier0_status": gate_res.status,
                    "tier0_motion_ratio": gate_res.motion_ratio,
                    "tier0_cooldown": gate_res.cooldown_remaining,
                    "tier0_rationale": gate_res.rationale,
                    "ai_bypassed": True,
                }
            )
            # Static scene: keep the fall-clip pre-roll going, masking heads at their last known spots
            self.alert_manager.record_frame(packet.frame, packet.timestamp, self._last_people)
            time.sleep(1.0 / self.motion_gater.idle_fps)
            return

        t_start = time.perf_counter()
        self.ai_profiler.start_frame()

        # 1. Single-Pass Perception (YOLO-Pose extracts both BBoxes & Skeletons)
        detections, single_pass_items = self.pose_estimator.detect_and_estimate_single_pass(packet.frame)


        # 2. ByteTrack with Fall Protection
        active_tracks = self.tracker.update(detections, packet.timestamp)
        active_track_ids = {t.track_id for t in active_tracks}

        # Fixed-rate temporal tick (0.9 tolerance absorbs camera timestamp jitter)
        temporal_tick = (packet.timestamp - self._last_temporal_ts) >= self._temporal_period * 0.9
        if temporal_tick:
            self._last_temporal_ts = packet.timestamp
            self._temporal_tick_count += 1
            self.sequence_buffer.update_activity(active_track_ids)
            self.risk_engine.cleanup_inactive(active_track_ids)
            self.gated_trigger.cleanup_inactive(active_track_ids)
            for dead in [t for t in self._last_anomaly if t not in active_track_ids]:
                self._last_anomaly.pop(dead, None)

        # Cleanup translation offset buffers for dead tracks
        dead_tids = [tid for tid in self._prev_raw_keypoints if tid not in active_track_ids]
        for tid in dead_tids:
            self._prev_raw_keypoints.pop(tid, None)
            self._prev_track_bboxes.pop(tid, None)
            self.keypoint_filters.pop(tid, None)

        # 2.5. Map detections to keypoints using IoU Matching (>= 0.70)
        matched_kp_map = self.pose_estimator.match_tracks_to_keypoints(
            active_tracks,
            single_pass_items,
            iou_threshold=self.iou_match_thresh,
        )

        poses_dict: Dict[int, PoseResult] = {}
        for track in active_tracks:
            tid = track.track_id
            curr_bbox = track.bbox

            if tid in matched_kp_map:
                # Direct match with original YOLO-Pose detection bbox (IoU >= 0.70)
                raw_kp = matched_kp_map[tid]
            elif tid in self._prev_raw_keypoints and tid in self._prev_track_bboxes:
                # Translation offset compensation for skipped frames / partial occlusions:
                # p_k(t) = p_k(t-1) + (c_bbox(t) - c_bbox(t-1))
                raw_kp = translate_keypoints(
                    self._prev_raw_keypoints[tid],
                    self._prev_track_bboxes[tid],
                    curr_bbox,
                )
            else:
                raw_kp = None

            if raw_kp is not None:
                # Update translation offset state
                self._prev_raw_keypoints[tid] = raw_kp.copy()
                self._prev_track_bboxes[tid] = curr_bbox

                # 3. Apply One Euro Filter to remove jitter without phase lag
                if tid not in self.keypoint_filters:
                    self.keypoint_filters[tid] = KeypointOneEuroFilter()
                filtered_kp = self.keypoint_filters[tid].filter(raw_kp, packet.timestamp)

                norm_kp = normalize_keypoints(filtered_kp, bbox_fallback=track.bbox)
                torso_angle = compute_torso_angle(filtered_kp)
                mean_conf = float(np.mean(filtered_kp[:, 2]))

                pose_res = PoseResult(
                    track_id=tid,
                    keypoints=filtered_kp,
                    normalized_keypoints=norm_kp,
                    torso_angle=torso_angle,
                    confidence=mean_conf,
                )
                poses_dict[tid] = pose_res

                # Record snapshot into SequenceBuffer at the temporal sampling rate only
                if temporal_tick:
                    snapshot = TrackSnapshot(
                        timestamp=packet.timestamp,
                        bbox=track.bbox,
                        keypoints=filtered_kp,
                        normalized_keypoints=norm_kp,
                        torso_angle=torso_angle,
                    )
                    self.sequence_buffer.add_snapshot(tid, snapshot)

        # 3.5. Fall-clip pre-roll: heads pixelated from this exact frame's raw detections + tracks
        self._last_people = [(it.keypoints, it.bbox) for it in single_pass_items] + [
            (poses_dict[t.track_id].keypoints if t.track_id in poses_dict else None, t.bbox) for t in active_tracks
        ]
        self.alert_manager.record_frame(packet.frame, packet.timestamp, self._last_people)

        # 4. Cascade / Gated Temporal Reasoning & Risk Evaluation
        assessments: List[RiskAssessment] = []
        alerts_raised: List[AlertEvent] = []

        # Minimum snapshots needed for the Tier-1 kinematic gate to produce a
        # meaningful velocity/acceleration reading (2 for Vy, 3 for Ay). Waiting
        # for the full window (sequence_buffer.is_ready(), ~1s of history) before
        # running ANY gating meant a brand-new or just-reset track_id (which is
        # exactly what happens when a sudden fall breaks IoU tracking continuity)
        # went completely unmonitored during the critical first second of a fall.
        MIN_SNAPSHOTS_FOR_GATING = 3

        if not temporal_tick:
            # Between temporal ticks only perception/tracking is refreshed; keep the
            # last risk assessment of every still-active track for rendering.
            assessments = [a for tid, a in self._last_assessments.items() if tid in active_track_ids]

        for track in (active_tracks if temporal_tick else []):
            tid = track.track_id
            snapshots_hist = self.sequence_buffer.get_snapshots(tid)
            if len(snapshots_hist) >= MIN_SNAPSHOTS_FOR_GATING:
                seq = build_model_sequence(
                    np.stack([s.keypoints for s in snapshots_hist]), self.window_size
                )
                kinematics = self.feature_extractor.extract(snapshots_hist)

                # Retrieve previous assessment state for hysteresis
                prev_ass = self._last_assessments.get(tid)
                curr_state = prev_ass.state if prev_ass else "NORMAL"

                # Tier 1: Biomechanical Gating Filter
                gate_res = self.gated_trigger.evaluate(
                    track_id=tid,
                    snapshots=snapshots_hist,
                    kinematics=kinematics,
                    current_state=curr_state,
                )

                # Upright warning signs (staggering, clutching head / chest) never trip the
                # kinematic gate, so with enable_gated_ai off the tiny model runs every tick.
                if gate_res.should_run_ai or not self.gated_ai:
                    # Tier 2: Deep AI Inference (Gate OPEN: Hard fall, slow slide, Z-axis fall, or hold-on timer)
                    action_pred = self.action_classifier.predict(seq, kinematics=kinematics)
                    if self._temporal_tick_count % self.anomaly_interval == 0 or tid not in self._last_anomaly:
                        self._last_anomaly[tid] = self.anomaly_detector.score(seq)
                    anomaly_res = self._last_anomaly[tid]
                else:
                    # Gate CLOSED: Normal walking/standing/sitting (~0 ms compute)
                    action_pred = gate_res.heuristic_prediction
                    anomaly_res = AnomalyResult(0.05, False, 0.05, 0.35)

                self.behavior_analyzer.update(tid, action_pred.primary_action, packet.timestamp)

                assessment = self.risk_engine.assess(
                    track_id=tid,
                    action_pred=action_pred,
                    anomaly_res=anomaly_res,
                    kinematics=kinematics,
                    timestamp=packet.timestamp,
                )
                assessments.append(assessment)

                # Cooldown-deduplicated alert check. The evidence snapshot is always
                # anonymized (every visible head pixelated) before it is saved or sent.
                snapshot = None
                if assessment.should_alert:
                    snapshot = anonymize_alert_snapshot(
                        packet.frame,
                        detections=[(it.keypoints, it.bbox) for it in single_pass_items],
                        tracks=[(poses_dict[t.track_id].keypoints if t.track_id in poses_dict else None, t.bbox)
                                for t in active_tracks],
                        alerted_bbox=track.bbox,
                    )
                alert = self.alert_manager.process_assessment(
                    assessment=assessment,
                    frame=snapshot,
                    timestamp=packet.timestamp,
                )
                if alert is not None:
                    alerts_raised.append(alert)

        if temporal_tick:
            self._last_assessments = {a.track_id: a for a in assessments}

        # Update shared render state atomically
        ai_fps = self.ai_profiler.fps
        self.shared_state.update_from_ai(
            tracks=active_tracks,
            poses=poses_dict,
            assessments=assessments,
            alerts=alerts_raised,
            ai_fps=ai_fps,
            timestamp=packet.timestamp,
        )


    def render_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[AlertEvent]]:
        """Render Thread (30 - 60 FPS): Instantly renders overlay without blocking on neural inference."""
        self.render_profiler.start_frame()
        tracks, poses, assessments, ai_fps = self.shared_state.get_render_data(timestamp)

        pose_dict = {p.track_id: p for p in poses}
        if self.blur_faces:
            annotated = anonymize_people(
                frame, [(pose_dict[t.track_id].keypoints if t.track_id in pose_dict else None, t.bbox) for t in tracks]
            )
        else:
            annotated = frame.copy()
        risk_dict = {a.track_id: a for a in assessments}

        has_emergency = False
        # Draw persons
        for track in tracks:
            tid = track.track_id
            pose = pose_dict.get(tid)
            ass = risk_dict.get(tid)

            state = ass.state if ass else "NORMAL"
            action = ass.evidence.get("primary_action", "detected") if ass else "detected"
            risk_score = ass.risk_score if ass else 0.05
            evidence = ass.evidence if ass else None

            if state in ["HIGH_RISK", "ALERT_SENT"] or risk_score >= 0.7:
                has_emergency = True

            # Draw skeleton
            if pose is not None and self.visualizer.show_skeleton:
                color = self.visualizer.get_state_color(state)
                self.visualizer.draw_skeleton(annotated, pose.keypoints, color=color)

            # Draw person card
            self.visualizer.draw_person_card(
                frame=annotated,
                track_id=tid,
                bbox=track.bbox,
                state=state,
                action=action,
                risk_score=risk_score,
                evidence=evidence,
            )

        # Draw emergency red border around video frame when fall/emergency is detected
        if has_emergency:
            self.visualizer.draw_emergency_frame_border(annotated)

        # Draw HUD
        if self.visualizer.show_fps_meter:
            self.visualizer.draw_hud(annotated, fps=self.render_profiler.fps, active_tracks=len(tracks))

        # Append telemetry HUD
        cv2.putText(
            annotated,
            f"Display FPS: {self.render_profiler.fps:.1f} | AI Rate: {ai_fps:.1f} FPS",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 242, 254),
            2,
            cv2.LINE_AA,
        )

        # Hand each alert to the caller exactly once (previously the last 10 alerts were
        # re-returned and re-printed on every rendered frame, stalling the render loop).
        with self.shared_state.lock:
            alerts = self.shared_state.latest_alerts
            self.shared_state.latest_alerts = []
        return annotated, alerts
