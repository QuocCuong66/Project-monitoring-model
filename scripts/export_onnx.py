"""Export PyTorch models (YOLO-Pose, TCN/ST-Transformer, Autoencoder) to ONNX format with FP16/FP32."""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import SpatialTemporalTransformer, TCNSequenceClassifier
from src.anomaly.anomaly_score import PoseSequenceAutoencoder


def export_pose_model(model_name: str = "yolov8n-pose.pt", half: bool = True) -> None:
    """Export YOLOv8-Pose to ONNX via Ultralytics."""
    print(f"[Info] Exporting YOLO-Pose model ({model_name}) to ONNX (half={half})...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        exported_path = model.export(format="onnx", half=half, dynamic=True)
        print(f"[Success] YOLO-Pose exported successfully to: {exported_path}")
    except Exception as e:
        print(f"[Warning] Failed to export YOLO-Pose to ONNX: {e}")


def export_action_model(weights_path: str, output_path: str, architecture: str = "tcn") -> None:
    print(f"[Info] Exporting Action Model ({architecture}) to ONNX: {output_path}...")
    if architecture == "tcn":
        model = TCNSequenceClassifier(input_dim=51, num_classes=10)
    else:
        model = SpatialTemporalTransformer(input_dim=51, num_classes=10)

    if Path(weights_path).exists():
        try:
            ckpt = torch.load(weights_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[Notice] Using initialized architecture weights ({e})")

    model.eval()
    dummy_input = torch.randn(1, 30, 17, 3, dtype=torch.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["skeletal_sequence"],
        output_names=["action_logits"],
        dynamic_axes={
            "skeletal_sequence": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )
    print(f"[Success] Action model ({architecture}) exported successfully to {output_path}")


def export_anomaly_model(weights_path: str, output_path: str) -> None:
    print(f"[Info] Exporting Anomaly Model to ONNX: {output_path}...")
    model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64)
    if Path(weights_path).exists():
        try:
            ckpt = torch.load(weights_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[Notice] Using initialized architecture weights ({e})")

    model.eval()
    dummy_input = torch.randn(1, 30, 17, 3, dtype=torch.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["skeletal_sequence"],
        output_names=["reconstructed_sequence"],
        dynamic_axes={
            "skeletal_sequence": {0: "batch_size"},
            "reconstructed_sequence": {0: "batch_size"},
        },
    )
    print(f"[Success] Anomaly model exported successfully to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--model-type", type=str, choices=["pose", "action", "anomaly", "all"], default="all")
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--action-arch", type=str, choices=["tcn", "st_transformer"], default="tcn")
    parser.add_argument("--action-weights", type=str, default="models/checkpoints/best_tcn.pt")
    parser.add_argument("--anomaly-weights", type=str, default="models/checkpoints/best_pose_autoencoder.pt")
    parser.add_argument("--half", action="store_true", default=True, help="Export FP16 precision")
    args = parser.parse_args()

    if args.model_type in ["pose", "all"]:
        export_pose_model(args.pose_model, half=args.half)
    if args.model_type in ["action", "all"]:
        export_action_model(args.action_weights, f"models/temporal/{args.action_arch}.onnx", architecture=args.action_arch)
    if args.model_type in ["anomaly", "all"]:
        export_anomaly_model(args.anomaly_weights, "models/anomaly/pose_autoencoder.onnx")


if __name__ == "__main__":
    main()
