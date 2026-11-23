"""Choose an exported YOLO engine only when it matches this PyTorch checkpoint and GPU."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_yolo_path(
    checkpoint: str | Path, *, device: str, imgsz: int, max_batch: int = 1
) -> tuple[Path, str]:
    """Return (model path, backend); keep the checkpoint as the safe fallback."""
    source = Path(checkpoint).expanduser().resolve()
    if not str(device).startswith("cuda") or not torch.cuda.is_available() or source.suffix.lower() != ".pt":
        return source, "pytorch"
    engine = source.with_suffix(".engine")
    metadata_path = source.with_suffix(".engine.json")
    if not engine.is_file() or not metadata_path.is_file():
        return source, "pytorch"
    try:
        import tensorrt as trt

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != _sha256(source):
            raise ValueError("checkpoint checksum changed")
        if metadata.get("gpu_compute_capability") != list(torch.cuda.get_device_capability(torch.device(device))):
            raise ValueError("GPU architecture differs")
        if metadata.get("tensorrt_version") != trt.__version__:
            raise ValueError("TensorRT version differs")
        if int(metadata.get("imgsz", -1)) != int(imgsz):
            raise ValueError("image size differs")
        if int(metadata.get("max_batch", 0)) < int(max_batch):
            raise ValueError("engine batch profile is too small")
    except Exception as exc:
        logger.warning("YOLO TensorRT unavailable (%s); using PyTorch checkpoint", exc)
        return source, "pytorch"
    return engine, "tensorrt"


def load_yolo(
    checkpoint: str | Path, *, device: str, imgsz: int, max_batch: int = 1
):
    """Load TRT when valid, or the original PyTorch model on the requested device."""
    from ultralytics import YOLO

    selected, backend = select_yolo_path(checkpoint, device=device, imgsz=imgsz, max_batch=max_batch)
    if backend == "tensorrt":
        try:
            model = YOLO(str(selected), task="detect")
            logger.info("YOLO TensorRT selected: %s", selected)
            return model, backend
        except Exception as exc:
            logger.warning("YOLO TensorRT load failed (%s); using PyTorch checkpoint", exc)
    model = YOLO(str(checkpoint))
    model.to(torch.device(device))
    return model, "pytorch"

