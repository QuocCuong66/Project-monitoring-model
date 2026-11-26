"""Deep autoencoder and reconstruction error scoring for unsupervised anomaly detection."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

from src.anomaly.anomaly_detector import AnomalyDetector, AnomalyResult
from src.utils.config import resolve_model_path
from src.utils.ort_providers import dml_provider



if TORCH_AVAILABLE:
    class PoseSequenceAutoencoder(nn.Module):
        """Conv1D temporal autoencoder for skeletal sequence reconstruction."""

        def __init__(self, input_dim: int = 51, latent_dim: int = 64) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder: compresses (B, 51, T) -> (B, latent_dim)
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),  # T -> T/2

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),  # T/2 -> 4

                nn.Flatten(),
                nn.Linear(128 * 4, latent_dim),
                nn.ReLU(),
            )

            # Decoder: reconstructs (B, latent_dim) -> (B, 51, T)
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_dim, 128 * 4),
                nn.ReLU(),
            )

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 4 -> 8
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),   # 8 -> 16
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, input_dim, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Tensor of shape (B, T, 17, 3) or (B, T, 51) or (B, 51, T).

            Returns:
                Reconstructed sequence tensor matching input shape.
            """
            orig_shape = x.shape
            if x.dim() == 4:
                b, t, k, c = x.shape
                x = x.view(b, t, k * c).transpose(1, 2)  # (B, 51, T)
            elif x.dim() == 3:
                if x.shape[1] != self.input_dim and x.shape[2] == self.input_dim:
                    x = x.transpose(1, 2)  # (B, T, 51) -> (B, 51, T)

            target_len = x.shape[2]

            z = self.encoder(x)  # (B, latent_dim)
            dec_fc = self.decoder_fc(z).view(-1, 128, 4)  # (B, 128, 4)
            reconstructed = self.decoder_conv(dec_fc)       # (B, 51, 32)

            # Interpolate to exact original length T
            if reconstructed.shape[2] != target_len:
                reconstructed = nn.functional.interpolate(reconstructed, size=target_len, mode="linear", align_corners=False)

            if len(orig_shape) == 4:
                # Reshape back to (B, T, 17, 3)
                reconstructed = reconstructed.transpose(1, 2).view(orig_shape)
            elif len(orig_shape) == 3 and orig_shape[2] == self.input_dim:
                # Reshape back to (B, T, 51)
                reconstructed = reconstructed.transpose(1, 2)

            return reconstructed
else:
    class PoseSequenceAutoencoder:
        pass


class ReconstructionAnomalyScorer(AnomalyDetector):
    """Computes anomaly score based on autoencoder reconstruction discrepancy."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.22,
        device: str = "cuda",
        steepness: float = 15.0,
    ) -> None:
        self.threshold = threshold
        self.steepness = steepness
        self.has_weights = False
        self.ort_session = None
        self.is_onnx = False
        self.model = None

        resolved_weights = None
        if weights_path:
            p = resolve_model_path(weights_path)
            if p and p.exists():
                resolved_weights = p
            else:
                p_onnx = resolve_model_path(str(weights_path).replace(".pt", ".onnx"))
                if p_onnx and p_onnx.exists():
                    resolved_weights = p_onnx

        # 1. Try ONNX Runtime first
        if resolved_weights and str(resolved_weights).endswith(".onnx") and ORT_AVAILABLE:
            try:
                providers = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
                available = ort.get_available_providers()
                actual_providers = [dml_provider() if p == "DmlExecutionProvider" else p for p in providers if p in available]
                sess_options = ort.SessionOptions()
                if actual_providers[0] == "CPUExecutionProvider":
                    # Tiny sequence model: a single non-spinning thread avoids starving the
                    # camera, render and RTMO threads (default pool spins on every core).
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                self.ort_session = ort.InferenceSession(str(resolved_weights), sess_options=sess_options, providers=actual_providers)
                self.is_onnx = True
                self.has_weights = True
            except Exception as e:
                print(f"[Warning] Failed to load ONNX anomaly weights from {resolved_weights}: {e}")

        # 2. If not ONNX, fallback to PyTorch if available
        if not self.has_weights and TORCH_AVAILABLE:
            dev_str = device if torch.cuda.is_available() and device == "cuda" else "cpu"
            self.device = torch.device(dev_str)
            self.model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64)
            if resolved_weights and Path(resolved_weights).exists():
                try:
                    ckpt = torch.load(resolved_weights, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        self.model.load_state_dict(ckpt)
                    self.has_weights = True
                except Exception as e:
                    print(f"[Warning] Failed to load anomaly weights from {resolved_weights}: {e}")

            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()

    def score(self, sequence: np.ndarray) -> AnomalyResult:
        """Compute reconstruction error and map to [0, 1] anomaly score."""
        if self.has_weights:
            if self.is_onnx and self.ort_session is not None:
                inp = sequence[np.newaxis, ...].astype(np.float32)
                input_name = self.ort_session.get_inputs()[0].name
                reconstructed = self.ort_session.run(None, {input_name: inp})[0]
                rec_np = reconstructed[0]
                error = float(np.mean(np.abs(sequence - rec_np)))
            elif TORCH_AVAILABLE and self.model is not None:
                with torch.no_grad():
                    seq_tensor = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
                    if seq_tensor.dim() in (2, 3):
                        tensor_seq = seq_tensor.unsqueeze(0)
                    else:
                        tensor_seq = seq_tensor
                    reconstructed = self.model(tensor_seq)
                    rec_np = reconstructed.cpu().numpy()[0]
                    error = float(np.mean(np.abs(sequence - rec_np)))
            else:
                error = self._heuristic_anomaly_error(sequence)
        else:
            # Heuristic calculation based on extreme body deformation & sudden acceleration
            error = self._heuristic_anomaly_error(sequence)

        # Sigmoid scaling around baseline threshold
        # Sigmoid calibrated so 0.5 sits at the p95 normal error (see training/train_anomaly.py)
        anomaly_score = float(1.0 / (1.0 + np.exp(-self.steepness * (error - self.threshold))))
        is_anomaly = anomaly_score > 0.65

        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            reconstruction_error=error,
            baseline_threshold=self.threshold,
        )

    def _heuristic_anomaly_error(self, sequence: np.ndarray) -> float:
        """Physical anomaly heuristic based on abnormal kinematic variance."""
        pts = sequence[:, :, :2]  # (T, 17, 2)
        # Check acceleration spikes
        if len(pts) >= 3:
            vel = np.diff(pts, axis=0)
            acc = np.diff(vel, axis=0)
            max_acc = float(np.max(np.abs(acc)))
            # Check horizontal distortion
            last_frame = pts[-1]
            height = max(0.1, np.max(last_frame[:, 1]) - np.min(last_frame[:, 1]))
            width = max(0.1, np.max(last_frame[:, 0]) - np.min(last_frame[:, 0]))
            ar = width / height
            error = 0.10 + 0.15 * min(1.0, max_acc / 0.5) + (0.12 if ar > 1.2 else 0.0)
            return float(min(1.0, error))
        return 0.10
