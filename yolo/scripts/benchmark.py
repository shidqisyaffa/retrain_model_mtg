"""Benchmark speed (latency & FPS) and evaluate accuracy for YOLO models."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Redirect ultralytics config to local project to avoid permission issues
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parents[1] / ".ultralytics")

import torch
from ultralytics import YOLO


def benchmark_speed(model_path: Path, device: str = "0", imgsz: int = 640, iterations: int = 100, warmup: int = 15):
    print(f"\n{'='*60}")
    print(f"BENCHMARK KECEPATAN: {model_path.name}")
    print(f"{'='*60}")
    print(f"Model path  : {model_path}")
    print(f"Device      : {device}")
    print(f"Image Size  : {imgsz}x{imgsz}")
    print(f"Iterations  : {iterations} (Warmup: {warmup})")

    # Load model
    is_engine = model_path.suffix.lower() == ".engine"
    if is_engine:
        model = YOLO(str(model_path), task="detect")
    else:
        model = YOLO(str(model_path))

    # Determine torch device
    if device.isdigit():
        target_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    if not is_engine and target_device.startswith("cuda"):
        model.to(torch.device(target_device))

    # Synthetic input tensor (1, 3, imgsz, imgsz) in [0, 1]
    dummy_input = torch.rand(1, 3, imgsz, imgsz)
    if target_device.startswith("cuda"):
        dummy_input = dummy_input.to(target_device)

    # Warmup
    print("Melakukan warmup GPU...")
    for _ in range(warmup):
        _ = model(dummy_input, verbose=False)
    if target_device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timing
    print(f"Mengukur {iterations} iterasi inferensi...")
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(dummy_input, verbose=False)
        if target_device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    fps = 1000.0 / avg_ms

    print("\n--- HASIL BENCHMARK KECEPATAN ---")
    print(f"Rata-rata Latensi : {avg_ms:.2f} ms")
    print(f"Throughput        : {fps:.1f} FPS")
    print(f"Min / Max Latensi : {min_ms:.2f} ms / {max_ms:.2f} ms")
    print(f"{'='*60}\n")
    return {"avg_ms": avg_ms, "fps": fps}


def evaluate_accuracy(model_path: Path, data_yaml: Path, device: str = "0", imgsz: int = 640):
    print(f"\n{'='*60}")
    print(f"EVALUASI AKURASI (VAL): {model_path.name}")
    print(f"{'='*60}")
    if not data_yaml.is_file():
        print(f"ERROR: data.yaml tidak ditemukan di {data_yaml}")
        return

    model = YOLO(str(model_path))
    try:
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=True)
        print("\n--- HASIL EVALUASI AKURASI ---")
        print(f"mAP50       : {metrics.box.map50:.4f}")
        print(f"mAP50-95    : {metrics.box.map:.4f}")
        print(f"Precision   : {metrics.box.mp:.4f}")
        print(f"Recall      : {metrics.box.mr:.4f}")
    except Exception as e:
        print(f"Gagal melakukan evaluasi validasi: {e}")
        print("Pastikan folder dataset val/ memiliki gambar dan file label .txt.")
    print(f"{'='*60}\n")


def main():
    root = Path(__file__).resolve().parents[1]
    default_weights = root / "weights" / "model_yolo_baru.pt"
    default_data = root / "configs" / "data.yaml"

    parser = argparse.ArgumentParser(description="Benchmark Kecepatan & Evaluasi YOLO")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Path ke model .pt atau .engine")
    parser.add_argument("--data", type=str, default=str(default_data), help="Path ke data.yaml")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID (default: 0) atau cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi gambar (default: 640)")
    parser.add_argument("--iterations", type=int, default=100, help="Jumlah iterasi benchmark kecepatan")
    parser.add_argument("--mode", choices=["speed", "val", "all"], default="speed", help="Mode: speed, val, atau all")
    args = parser.parse_args()

    model_path = Path(args.weights).resolve()
    if not model_path.is_file():
        print(f"File model tidak ditemukan: {model_path}")
        sys.exit(1)

    if args.mode in ("speed", "all"):
        benchmark_speed(model_path, device=args.device, imgsz=args.imgsz, iterations=args.iterations)

    if args.mode in ("val", "all"):
        evaluate_accuracy(model_path, Path(args.data).resolve(), device=args.device, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
