"""Export PyTorch YOLO checkpoint (.pt) to TensorRT Engine (.engine) with metadata for reference_app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Redirect ultralytics config to local project to avoid permission issues
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parents[1] / ".ultralytics")

import tensorrt as trt
import torch

# Compatibility patch for TensorRT 10/11 where flags/methods were deprecated or removed
if not hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
    setattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", 0)
if not hasattr(trt.Builder, "platform_has_fast_fp16"):
    setattr(trt.Builder, "platform_has_fast_fp16", True)
if not hasattr(trt.Builder, "platform_has_fast_int8"):
    setattr(trt.Builder, "platform_has_fast_int8", True)
if not hasattr(trt.BuilderFlag, "FP16"):
    setattr(trt.BuilderFlag, "FP16", getattr(trt.BuilderFlag, "TF32", 0))
if not hasattr(trt.BuilderFlag, "INT8"):
    setattr(trt.BuilderFlag, "INT8", getattr(trt.BuilderFlag, "TF32", 0))

from ultralytics import YOLO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_tensorrt(
    checkpoint: Path,
    device: str = "0",
    imgsz: int = 640,
    max_batch: int = 1,
    half: bool = True,
):
    print(f"\n{'='*60}")
    print(f"EXPORT YOLO KE TENSORRT ENGINE")
    print(f"{'='*60}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Device     : GPU {device}")
    print(f"Image Size : {imgsz}x{imgsz}")
    print(f"Precision  : {'FP16 (Half)' if half else 'FP32'}")
    print(f"Max Batch  : {max_batch}")
    print(f"{'='*60}\n")

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")

    model = YOLO(str(checkpoint))
    exported_path = model.export(
        format="engine",
        device=int(device) if device.isdigit() else 0,
        imgsz=imgsz,
        half=half,
        batch=max_batch,
        # max_batch > 1 butuh profil dinamis (1..max_batch); engine statis menolak batch yang lebih kecil
        dynamic=max_batch > 1,
    )
    engine_path = Path(exported_path).resolve()
    print(f"\nTensorRT engine berhasil dibuat: {engine_path}")

    # Generate metadata json matching reference_app requirements
    cuda_device = torch.device(f"cuda:{device}" if device.isdigit() else device)
    metadata = {
        "source_sha256": _sha256(checkpoint),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(cuda_device)),
        "tensorrt_version": trt.__version__,
        "imgsz": imgsz,
        "max_batch": max_batch,
        "half": half,
    }

    metadata_path = checkpoint.with_suffix(".engine.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata engine tersimpan di: {metadata_path}")
    print(f"{'='*60}\n")


def main():
    root = Path(__file__).resolve().parents[1]
    default_weights = root / "weights" / "model_yolo_baru.pt"

    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT with metadata")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Path ke model .pt")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size (default: 1)")
    parser.add_argument("--fp32", action="store_true", help="Gunakan FP32 (default: FP16)")

    args = parser.parse_args()
    export_tensorrt(
        checkpoint=Path(args.weights).resolve(),
        device=args.device,
        imgsz=args.imgsz,
        max_batch=args.batch,
        half=not args.fp32,
    )


if __name__ == "__main__":
    main()
