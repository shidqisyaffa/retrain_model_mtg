"""Script retraining / fine-tuning YOLO untuk deteksi objek (person, weapon, knife, car, bicycle)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Redirect ultralytics config to local project to avoid permission issues
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parents[1] / ".ultralytics")

from ultralytics import YOLO


def run_training(
    weights: Path,
    data: Path,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    patience: int = 25,
    freeze: int = 0,
    project: Path | None = None,
    name: str = "fine_tune_yolo",
):
    print(f"\n{'='*65}")
    print(f"MEMULAI TRAINING / FINE-TUNING YOLO")
    print(f"{'='*65}")
    print(f"Base Weights : {weights}")
    print(f"Data Config  : {data}")
    print(f"Epochs       : {epochs}")
    print(f"Batch Size   : {batch}")
    print(f"Image Size   : {imgsz}")
    print(f"Device       : {device}")
    print(f"Optimizer    : {optimizer} (lr0={lr0})")
    print(f"Early Stop   : patience={patience}")
    if freeze > 0:
        print(f"Freeze Layers: {freeze} layer awal di-freeze")
    print(f"{'='*65}\n")

    if not weights.is_file():
        raise FileNotFoundError(f"File weights tidak ditemukan: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"File data config tidak ditemukan: {data}")

    model = YOLO(str(weights))

    train_args = {
        "data": str(data),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "cos_lr": True,          # Cosine learning rate scheduler
        "close_mosaic": 10,      # Matikan augmentasi mosaic 10 epoch terakhir untuk lokalisasi presisi
        "project": str(project) if project else "yolo/runs",
        "name": name,
        "save": True,
        "save_period": 10,
        "exist_ok": True,
    }

    if freeze > 0:
        train_args["freeze"] = freeze

    results = model.train(**train_args)
    print("\nTraining selesai!")
    save_dir = getattr(results, "save_dir", Path("yolo/runs") / name)
    print(f"Model terbaik disimpan di: {save_dir / 'weights' / 'best.pt'}")
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    default_weights = root / "weights" / "model_yolo_baru.pt"
    default_data = root / "configs" / "data.yaml"
    default_project = root / "runs"

    parser = argparse.ArgumentParser(description="Retrain/Fine-tune YOLO model")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Pretrained weights (.pt)")
    parser.add_argument("--data", type=str, default=str(default_data), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index atau cpu (default: 0)")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--freeze", type=int, default=0, help="Jumlah layer awal yang di-freeze (default: 0)")
    parser.add_argument("--name", type=str, default="fine_tune_yolo", help="Nama folder eksperimen run")

    args = parser.parse_args()

    run_training(
        weights=Path(args.weights).resolve(),
        data=Path(args.data).resolve(),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        freeze=args.freeze,
        project=default_project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
