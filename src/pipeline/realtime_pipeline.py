"""Master real-time video intelligence pipeline orchestrating all modules."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.detection.detector import PersonDetector
from src.detection.person_detector import YOLOv8PersonDetector
from src.detection.detector_factory import create_detector
from src.tracking.tracker import Tracker, Track
from src.tracking.track_manager import ByteTrackManager
from src.pose.pose_estimator import PoseEstimator, PoseResult, YOLOv8PoseEstimator, translate_keypoints
from src.pose.pose_factory import create_pose_estimator
from src.pose.keypoints import normalize_keypoints, compute_torso_angle
from src.temporal.sequence_buffer import SequenceBuffer, TrackSnapshot
from src.temporal.sequence_features import build_model_sequence
from src.temporal.temporal_features import TemporalFeatureExtractor
from src.temporal.gated_trigger import KinematicGatedTrigger
from src.action.action_classifier import ActionClassifier
from src.action.behavior_classifier import BehaviorSequenceAnalyzer
from src.anomaly.anomaly_detector import AnomalyDetector
from src.anomaly.anomaly_score import ReconstructionAnomalyScorer, AnomalyResult
from src.risk.risk_engine import RiskEngine, display_action as _display_action, RiskAssessment
from src.risk.threshold_manager import ThresholdManager
from src.alerts.alert_manager import AlertManager, AlertEvent
from src.detection.motion_gater import MotionGater, MotionGateResult
from src.utils.visualizer import PipelineVisualizer
from src.utils.privacy import FaceBlurrer, anonymize_alert_snapshot
from src.utils.profiler import LatencyProfiler, ResourceMonitor
from src.utils.config import load_config


@dataclass
class PipelineFrameResult:
    """Comprehensive output produced after processing a single video frame."""
    frame_idx: int
    timestamp: float
    annotated_frame: np.ndarray
    tracks: List[Track]
    poses: List[PoseResult]
    assessments: List[RiskAssessment]
    alerts: List[AlertEvent]
    telemetry: Dict[str, Any]


class RealtimePipeline:
    """End-to-End multi-stage video understanding pipeline for elderly monitoring."""

    def __init__(
        self,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[Tracker] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        action_classifier: Optional[ActionClassifier] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        risk_engine: Optional[RiskEngine] = None,
        alert_manager: Optional[AlertManager] = None,
        visualizer: Optional[PipelineVisualizer] = None,
        motion_gater: Optional[MotionGater] = None,
        blur_faces: bool = False,
        single_pass: bool = True,
    ) -> None:
        # Load configs

        try:
            model_cfg = load_config("model.yaml")
        except Exception:
            model_cfg = {}

        try:
            infer_cfg = load_config("inference.yaml")
        except Exception:
            infer_cfg = {}

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

        dev = infer_cfg.get("runtime", {}).get("device", "cuda")
        if dev == "cuda" and not has_cuda:
            dev = "cpu"
        self.device = dev
        self.use_fp16 = infer_cfg.get("runtime", {}).get("use_fp16", True) and (dev != "cpu")
        # Real-time stream: no face blurring
        self.blur_faces = infer_cfg.get("privacy", {}).get("blur_faces", False) and blur_faces
        self.single_pass = single_pass and (detector is None)

        # 1. Detection (Only instantiated when 2-pass mode is explicitly requested)
        if not self.single_pass:
            det_cfg = model_cfg.get("detector", {})
            det_arch = det_cfg.get("architecture", "yolov8")
            self.detector = detector or create_detector(
                architecture=det_arch,
                model_path=det_cfg.get("model_name", "yolov8n.pt"),
                conf_threshold=det_cfg.get("conf_threshold", 0.40),
                device=self.device,
                half=self.use_fp16,
            )
        else:
            self.detector = None

        # 2. Tracking (ByteTrack)
        trk_cfg = model_cfg.get("tracker", {})
        self.tracker = tracker or ByteTrackManager(
            track_high_thresh=trk_cfg.get("track_high_thresh", 0.50),
            track_low_thresh=trk_cfg.get("track_low_thresh", 0.15),
            new_track_thresh=trk_cfg.get("new_track_thresh", 0.60),
            match_thresh=trk_cfg.get("match_thresh", 0.70),
        )

        # 3. Pose (Single-pass / Pluggable Architecture)
        pose_cfg = model_cfg.get("pose", {})
        pose_arch = pose_cfg.get("architecture", "yolov8_pose")
        self.pose_estimator = pose_estimator or create_pose_estimator(
            architecture=pose_arch,
            model_path=pose_cfg.get("model_name", "yolov8n-pose.pt"),
            conf_threshold=pose_cfg.get("conf_threshold", 0.35),
            device=self.device,
            half=self.use_fp16,
        )

        # 4. Temporal Sequence Buffer & Kinematics
        temp_cfg = model_cfg.get("temporal", {})
        window_size = temp_cfg.get("window_size", 30)
        self.window_size = window_size
        self.sequence_buffer = SequenceBuffer(window_size=window_size)
        self.feature_extractor = TemporalFeatureExtractor(fps=15.0)

        # 5. Dual-Stream AI Models
        self.action_classifier = action_classifier or ActionClassifier(
            weights_path=temp_cfg.get("weights_path", None),
            device=self.device,
            architecture=temp_cfg.get("architecture", "tcn"),
        )
        anom_cfg = model_cfg.get("anomaly", {})
        self.anomaly_detector = anomaly_detector or ReconstructionAnomalyScorer(
            weights_path=anom_cfg.get("weights_path", None),
            threshold=anom_cfg.get("threshold", 0.22),
            steepness=anom_cfg.get("steepness", 15.0),
            device=self.device,
        )
        self.behavior_analyzer = BehaviorSequenceAnalyzer()

        # 6. Risk Engine & State Machine
        self.thresholds = ThresholdManager()
        self.risk_engine = risk_engine or RiskEngine(self.thresholds)

        # 6.5. Tier-0 Motion Gater & Tier-1 Kinematic Gated Trigger
        motion_cfg = model_cfg.get("motion_gating", {})
        enable_t0 = infer_cfg.get("scheduling", {}).get("enable_tier0_motion_gating", True) and motion_cfg.get("enabled", True)
        self.motion_gater = motion_gater or MotionGater(
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
        self.gated_trigger = KinematicGatedTrigger(hold_on_frames=45)
        self.iou_match_thresh = model_cfg.get("perception", {}).get("iou_match_thresh", 0.45)
        self._prev_raw_keypoints: Dict[int, np.ndarray] = {}
        self._prev_track_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        # 7. Alert Manager
        self.alert_manager = alert_manager or AlertManager(self.thresholds)

        # 8. Visualizer & Utilities
        self.visualizer = visualizer or PipelineVisualizer(infer_cfg.get("display", {}))
        self.face_blurrer = FaceBlurrer()
        self.profiler = LatencyProfiler()
        self.resource_monitor = ResourceMonitor()

        self._frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> PipelineFrameResult:
        """Execute end-to-end multi-stage pipeline on a single frame."""
        self.profiler.start_frame()
        self._frame_count += 1
        annotated_frame = frame.copy()

        # STAGE 0: Tier-0 Motion Gating (Background Subtraction / Pixel Differencing)
        self.profiler.start("tier0_motion")
        has_active = bool(self.tracker.tracks) if hasattr(self.tracker, "tracks") else False
        motion_gate_res = self.motion_gater.evaluate(frame, has_active_tracks=has_active)
        self.profiler.stop("tier0_motion")

        # Case 1: Throttled idle frame (reducing FPS to 10-15 FPS when room is static)
        if motion_gate_res.is_throttled_frame:
            fps = self.profiler.fps
            telemetry = {
                **self.profiler.get_summary(),
                **self.resource_monitor.get_telemetry(),
                "active_tracks_count": 0,
                "tier0_status": motion_gate_res.status,
                "tier0_motion_ratio": motion_gate_res.motion_ratio,
                "tier0_cooldown": motion_gate_res.cooldown_remaining,
                "tier0_rationale": motion_gate_res.rationale,
                "ai_bypassed": True,
            }
            if self.visualizer:
                annotated_frame = self.visualizer.draw_hud(
                    frame=annotated_frame,
                    fps=fps,
                    active_tracks=0,
                    system_status="TIER-0 IDLE",
                    telemetry=telemetry,
                )
            return PipelineFrameResult(
                frame_idx=self._frame_count,
                timestamp=timestamp,
                annotated_frame=annotated_frame,
                tracks=[],
                poses=[],
                assessments=[],
                alerts=[],
                telemetry=telemetry,
            )

        # Case 2: Static room (Gate CLOSED -> Completely bypass AI inference, 0% GPU load)
        if not motion_gate_res.should_run_ai:
            fps = self.profiler.fps
            telemetry = {
                **self.profiler.get_summary(),
                **self.resource_monitor.get_telemetry(),
                "active_tracks_count": 0,
                "tier0_status": motion_gate_res.status,
                "tier0_motion_ratio": motion_gate_res.motion_ratio,
                "tier0_cooldown": motion_gate_res.cooldown_remaining,
                "tier0_rationale": motion_gate_res.rationale,
                "ai_bypassed": True,
            }

            if self.visualizer:
                annotated_frame = self.visualizer.draw_hud(
                    frame=annotated_frame,
                    fps=fps,
                    active_tracks=0,
                    system_status="TIER-0 IDLE",
                    telemetry=telemetry,
                )
            return PipelineFrameResult(
                frame_idx=self._frame_count,
                timestamp=timestamp,
                annotated_frame=annotated_frame,
                tracks=[],
                poses=[],
                assessments=[],
                alerts=[],
                telemetry=telemetry,
            )

        poses: List[PoseResult] = []

        if self.single_pass and hasattr(self.pose_estimator, "detect_and_estimate_single_pass"):
            # STAGE 1 & 3: Single-Pass Perception (YOLO-Pose extracts both BBoxes & Keypoints)
            self.profiler.start("perception")
            detections, single_pass_items = self.pose_estimator.detect_and_estimate_single_pass(frame)
            self.profiler.stop("perception")

            # STAGE 2: Multi-Object Tracking
            self.profiler.start("tracking")
            active_tracks = self.tracker.update(detections, timestamp)
            active_track_ids = {t.track_id for t in active_tracks}
            self.sequence_buffer.update_activity(active_track_ids)
            self.risk_engine.cleanup_inactive(active_track_ids)
            self.gated_trigger.cleanup_inactive(active_track_ids)
            self.profiler.stop("tracking")

            # Match keypoints with IoU >= 0.70
            self.profiler.start("pose")
            matched_kp_map = self.pose_estimator.match_tracks_to_keypoints(
                active_tracks,
                single_pass_items,
                iou_threshold=self.iou_match_thresh,
            )

            for track in active_tracks:
                tid = track.track_id
                curr_bbox = track.bbox

                if tid in matched_kp_map:
                    raw_kp = matched_kp_map[tid]
                elif tid in self._prev_raw_keypoints and tid in self._prev_track_bboxes:
                    raw_kp = translate_keypoints(
                        self._prev_raw_keypoints[tid],
                        self._prev_track_bboxes[tid],
                        curr_bbox,
                    )
                else:
                    raw_kp = None

                if raw_kp is not None:
                    self._prev_raw_keypoints[tid] = raw_kp.copy()
                    self._prev_track_bboxes[tid] = curr_bbox

                    norm_kp = normalize_keypoints(raw_kp, bbox_fallback=track.bbox)
                    torso_angle = compute_torso_angle(raw_kp)
                    mean_conf = float(np.mean(raw_kp[:, 2]))
                    poses.append(
                        PoseResult(
                            track_id=tid,
                            keypoints=raw_kp,
                            normalized_keypoints=norm_kp,
                            torso_angle=torso_angle,
                            confidence=mean_conf,
                        )
                    )
            pose_dict = {p.track_id: p for p in poses}
            self.profiler.stop("pose")

        else:
            # Legacy 2-Pass Mode Fallback
            self.profiler.start("detection")
            detections = self.detector.detect(frame) if self.detector else []
            self.profiler.stop("detection")

            self.profiler.start("tracking")
            active_tracks = self.tracker.update(detections, timestamp)
            active_track_ids = {t.track_id for t in active_tracks}
            self.sequence_buffer.update_activity(active_track_ids)
            self.risk_engine.cleanup_inactive(active_track_ids)
            self.gated_trigger.cleanup_inactive(active_track_ids)
            self.profiler.stop("tracking")

            self.profiler.start("pose")
            poses = self.pose_estimator.estimate(frame, active_tracks)
            pose_dict = {p.track_id: p for p in poses}
            self.profiler.stop("pose")

        # Fall-clip pre-roll (heads pixelated from this frame's detections + tracks)
        self.alert_manager.record_frame(
            frame,
            timestamp,
            [(None, d.bbox) for d in detections]
            + [(pose_dict[t.track_id].keypoints if t.track_id in pose_dict else None, t.bbox) for t in active_tracks],
        )

        # STAGE 4 & 5: Temporal Sequence Buffer & Cascade / Gated AI Reasoning
        self.profiler.start("temporal_reasoning")
        assessments: List[RiskAssessment] = []
        alerts_raised: List[AlertEvent] = []

        for track in active_tracks:
            tid = track.track_id
            pose = pose_dict.get(tid)

            # Record snapshot
            if pose is not None:
                snapshot = TrackSnapshot(
                    timestamp=timestamp,
                    bbox=track.bbox,
                    keypoints=pose.keypoints,
                    normalized_keypoints=pose.normalized_keypoints,
                    torso_angle=pose.torso_angle,
                )
                self.sequence_buffer.add_snapshot(tid, snapshot)

            # Retrieve temporal sequence. Only 3 snapshots are needed for the
            # Tier-1 kinematic gate to compute Vy/Ay (velocity/acceleration) —
            # waiting for the full sequence_buffer.is_ready() window (~1s) left
            # brand-new or just-reset track_ids completely unmonitored during
            # the critical first second of a fall.
            snapshots_hist = self.sequence_buffer.get_snapshots(tid)
            if len(snapshots_hist) >= 3:
                seq = build_model_sequence(
                    np.stack([s.keypoints for s in snapshots_hist]), self.window_size
                )
                kinematics = self.feature_extractor.extract(snapshots_hist)

                # Previous state for hysteresis
                prev_ass = self.risk_engine.assessments.get(tid)
                curr_state = prev_ass.state if prev_ass else "NORMAL"

                # Tier 1 Kinematic Trigger evaluation
                gate_res = self.gated_trigger.evaluate(
                    track_id=tid,
                    snapshots=snapshots_hist,
                    kinematics=kinematics,
                    current_state=curr_state,
                )

                if gate_res.should_run_ai:
                    # Tier 2: Deep AI inference (Supervised Action + Unsupervised Anomaly)
                    action_pred = self.action_classifier.predict(seq, kinematics=kinematics)
                    anomaly_res = self.anomaly_detector.score(seq)
                else:
                    # Gate CLOSED: Heuristic fast path (~0 ms)
                    action_pred = gate_res.heuristic_prediction
                    anomaly_res = AnomalyResult(0.05, False, 0.05, 0.35)

                self.behavior_analyzer.update(tid, action_pred.primary_action, timestamp)

                # Multi-Signal Risk Assessment
                assessment = self.risk_engine.assess(
                    track_id=tid,
                    action_pred=action_pred,
                    anomaly_res=anomaly_res,
                    kinematics=kinematics,
                    timestamp=timestamp,
                    snapshot=snapshots_hist[-1],
                    frame_size=(frame.shape[1], frame.shape[0]),
                )
                assessments.append(assessment)

                # STAGE 6: Alert Dispatching
                # If high-risk confirmed, prepare the snapshot with every visible head pixelated
                # (always, independent of the live-view blur toggle)
                snapshot_frame = None
                if assessment.should_alert:
                    snapshot_frame = anonymize_alert_snapshot(
                        frame,
                        detections=[(None, d.bbox) for d in detections],
                        tracks=[(pose_dict[t.track_id].keypoints if t.track_id in pose_dict else None, t.bbox)
                                for t in active_tracks],
                        alerted_bbox=track.bbox,
                    )

                alerts_raised.extend(self.alert_manager.process(
                    assessment=assessment,
                    frame=snapshot_frame,
                    timestamp=timestamp,
                ))

                # Render Visual Overlays
                if pose is not None and self.visualizer.show_skeleton:
                    color = self.visualizer.get_state_color(assessment.state)
                    self.visualizer.draw_skeleton(annotated_frame, pose.keypoints, color=color)

                self.visualizer.draw_person_card(
                    frame=annotated_frame,
                    track_id=tid,
                    bbox=track.bbox,
                    state=assessment.state,
                    action=_display_action(assessment.evidence),
                    risk_score=assessment.risk_score,
                    evidence=assessment.evidence,
                )
            else:
                # Buffer warming up, draw minimal bounding box
                self.visualizer.draw_person_card(
                    frame=annotated_frame,
                    track_id=tid,
                    bbox=track.bbox,
                    state="NORMAL",
                    action="calibrating",
                    risk_score=0.0,
                )

        self.profiler.stop("temporal_reasoning")

        # STAGE 7: Privacy Face Blurring on Live Output (if enabled)
        if self.blur_faces:
            for p in poses:
                self.face_blurrer.blur_keypoints_face(annotated_frame, p.keypoints)

        # STAGE 8: HUD & Telemetry
        telemetry = {
            **self.profiler.get_summary(),
            **self.resource_monitor.get_telemetry(),
            "active_tracks_count": len(active_tracks),
            "tier0_status": motion_gate_res.status,
            "tier0_motion_ratio": motion_gate_res.motion_ratio,
            "tier0_cooldown": motion_gate_res.cooldown_remaining,
            "tier0_rationale": motion_gate_res.rationale,
            "ai_bypassed": False,
        }
        self.visualizer.draw_hud(
            frame=annotated_frame,
            fps=self.profiler.fps,
            active_tracks=len(active_tracks),
            system_status=f"ONLINE ({motion_gate_res.status})",
            telemetry=telemetry,
        )


        return PipelineFrameResult(
            frame_idx=self._frame_count,
            timestamp=timestamp,
            annotated_frame=annotated_frame,
            tracks=active_tracks,
            poses=poses,
            assessments=assessments,
            alerts=alerts_raised,
            telemetry=telemetry,
        )
