"""Generate the local-workstation notebook for fine-tuning model_yolo_baru.pt (far-range / small objects).

Run:  python yolo/scripts/generate_local_notebook.py
Output: yolo/notebooks/yolo_train_local.ipynb

Design notes (why the notebook looks like this):
- Fine-tunes the existing checkpoint; never trains from scratch.
- New classes are appended after the old ones and the detection head is *expanded*
  (old class weights copied), instead of letting Ultralytics drop the mismatched head.
- New datasets only label some classes, so the old model pseudo-labels the classes a
  dataset does not annotate (anti catastrophic forgetting). Old training data can be
  added as "replay" sources when it is found.
- Classes are remapped by *name* from each dataset's data.yaml, never by numeric id.
- TensorRT engines are GPU-architecture specific: build them on the production GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}


def _source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def add_md(text: str) -> None:
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": _source(text)})


def add_code(code: str) -> None:
    notebook["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(code)}
    )


# =====================================================================
add_md(r'''
# Fine-Tuning YOLO Jarak Jauh (Drone / CCTV Tinggi) dari `model_yolo_baru.pt`

**Tujuan:** model sekarang sudah oke untuk jarak dekat–menengah, tapi sering *miss* objek kecil dari jauh.
Notebook ini **melanjutkan training dari bobot yang sudah ada** (bukan dari nol) di resolusi lebih tinggi
(`imgsz=1024`), sekaligus menambah kelas `face`, tanpa membuat model "lupa" kemampuan lamanya.

**Target hardware training:** RTX 5090 (32 GB VRAM), RAM 64 GB.

### Kelas (single detector)
| id | kelas | catatan |
|---|---|---|
| 0 | person | lama |
| 1 | knife | lama |
| 2 | weapon | lama (senjata api) |
| 3 | bicycle | lama |
| 4 | car | lama |
| 5 | face | **baru** |

Urutan id 0–4 **tidak boleh berubah** (dipakai aplikasi). Kelas baru (mis. `license_plate` nanti) selalu ditambah di belakang.

### Strategi anti *catastrophic forgetting*
1. **Head diperluas, bukan di-reset**: bobot klasifikasi 5 kelas lama disalin utuh, hanya baris kelas baru yang diinisialisasi.
2. **Pseudo-label dari model lama**: dataset baru hanya melabeli sebagian kelas (mis. dataset pisau tidak melabeli orang).
   Kelas lama yang tidak dianotasi diisi dari prediksi model lama, supaya model tidak diajari "orang = background".
3. **Replay data lama** (opsional): dataset `merged_rgb_thermal` belum ditemukan; slotnya tersedia di konfigurasi.
4. **Training 2 tahap**: tahap 1 backbone dibekukan (head/neck beradaptasi), tahap 2 seluruh jaringan dengan LR kecil.
5. **Evaluasi lama vs baru per dataset** (hanya label asli) untuk mendeteksi regresi sebelum deploy.

### Alur
0. Konfigurasi → 1. Cek GPU → 2. Bobot dasar → 3. Ambil dataset dari Drive → 4. Ekstrak & remap →
5. Pseudo-label → 6. `data.yaml` → 7. Perluas head → 8–9. Training 2 tahap → 10. Evaluasi →
11. Bandingkan lama vs baru → 12. Uji visual → 13. Simpan & panduan deploy → 14–15. (opsional) TensorRT & benchmark
''')

# =====================================================================
add_md(r'''
### Langkah 0: Konfigurasi
**Hanya cell ini yang perlu disesuaikan.** Cell lain membaca nilai dari sini.
''')
add_code(r'''
import os
from pathlib import Path

# ---------------- Lokasi proyek (folder Retrain_YOLO_TCCLIP) ----------------
PROJECT_ROOT = Path.cwd().resolve()
for _ in range(3):
    if (PROJECT_ROOT / "yolo").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
YOLO_DIR = PROJECT_ROOT / "yolo"
assert YOLO_DIR.is_dir(), f"Folder yolo/ tidak ditemukan dari {Path.cwd()} — jalankan notebook dari dalam proyek"

WEIGHTS_DIR = YOLO_DIR / "weights"
CONFIGS_DIR = YOLO_DIR / "configs"
DATASET_DIR = YOLO_DIR / "dataset"
DOWNLOAD_DIR = YOLO_DIR / "_downloads"   # salinan lokal zip dari Google Drive
RUNS_DIR = YOLO_DIR / "runs"
for _p in (WEIGHTS_DIR, CONFIGS_DIR, DATASET_DIR, DOWNLOAD_DIR, RUNS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Harus di-set SEBELUM ultralytics di-import; folder harus sudah ada (kalau tidak, Ultralytics memakai cwd)
(YOLO_DIR / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_DIR / ".ultralytics"))

# ---------------- Kelas target ----------------
# Urutan kelas lama WAJIB sama dengan model dasar. Kelas baru selalu ditambah di BELAKANG
# (nanti: tambahkan "license_plate" di akhir + sumber datanya).
TARGET_NAMES = ["person", "knife", "weapon", "bicycle", "car", "face"]

BASE_CHECKPOINT = WEIGHTS_DIR / "model_yolo_baru.pt"
HF_REPO_ID = "mtg-ai-team/drone-violence-models"
HF_FILENAME = "YOLO/model_yolo_baru.pt"   # token dari env HF_TOKEN atau `huggingface-cli login` (JANGAN ditulis di notebook)

# ---------------- Sumber dataset ----------------
# zip_path    : zip lokal ATAU di folder Google Drive for Desktop, mis. Path(r"G:\My Drive\dataset\x.zip").
#               Zip selalu disalin dulu ke yolo/_downloads (SSD lokal) — jangan training langsung dari Drive.
# gdrive_id   : (opsional) file ID Google Drive jika tidak pakai Drive for Desktop -> diunduh via gdown.
# class_map   : {nama kelas di dataset: nama kelas target}, dicocokkan dengan NAMA di data.yaml dalam zip
#               (bukan nomor id). Kelas yang tidak disebut dibuang. Huruf besar/kecil, spasi, '-' diabaikan.
# pseudo_label: isi kelas lama yang tidak dianotasi dataset ini dengan prediksi model lama.
DATASET_SRC_DIR = Path(r"D:\dataset")
DATASET_SOURCES = [
    {
        "prefix": "visdrone",
        "zip_path": DATASET_SRC_DIR / "VisDrone.v3-validation-set.yolo26.zip",
        "gdrive_id": None,
        # van dipetakan ke car: kalau dibuang, van yang terlihat seperti mobil diajarkan sebagai background
        "class_map": {"pedestrian": "person", "people": "person", "bicycle": "bicycle", "car": "car", "van": "car"},
        "pseudo_label": True,
    },
    {
        "prefix": "knife",
        "zip_path": DATASET_SRC_DIR / "Knife detection.v1i.yolo26.zip",
        "gdrive_id": None,
        "class_map": {"knife": "knife"},
        "pseudo_label": True,
    },
    {
        "prefix": "face",
        "zip_path": DATASET_SRC_DIR / "face-detection.v1i.yolo26.zip",
        "gdrive_id": None,
        "class_map": {"face": "face"},
        "pseudo_label": True,
    },
    # Sangat disarankan: dataset senjata api agar kelas weapon punya contoh nyata, mis.
    # {"prefix": "gun", "zip_path": DATASET_SRC_DIR / "gun.zip", "gdrive_id": None,
    #  "class_map": {"gun": "weapon", "pistol": "weapon", "rifle": "weapon"}, "pseudo_label": True},
]

# Data training lama (replay). merged_rgb_thermal belum ditemukan -> kosong.
# Jika ketemu (folder format YOLO berisi data.yaml), tambahkan misalnya:
# {"prefix": "replay", "dir": Path(r"E:\merged_rgb_thermal"), "class_map": None, "pseudo_label": False}
# class_map None = nama kelas dipakai apa adanya (yang tidak ada di TARGET_NAMES dibuang).
REPLAY_SOURCES = []

FORCE_REBUILD_DATASET = False   # True = ekstrak ulang semua sumber walaupun tidak berubah

# ---------------- Pseudo-label dari model lama ----------------
PSEUDO_LABEL_ENABLED = True
PSEUDO_IMGSZ = 640               # resolusi asli model lama (terkalibrasi di sini)
PSEUDO_CONF = {"person": 0.50, "knife": 0.65, "weapon": 0.50, "bicycle": 0.50, "car": 0.50}
PSEUDO_CONF_DEFAULT = 0.50       # untuk kelas model dasar yang tidak tercantum di atas
PSEUDO_SPLITS = ("train", "val")  # split test dibiarkan murni label asli
FORCE_PSEUDO_LABEL = False

# ---------------- Training (RTX 5090 32 GB / RAM 64 GB) ----------------
DEVICE = 0
IMGSZ = 1024
BATCH = 24          # YOLO26s @1024 kira-kira 18–22 GB. Naikkan ke 32 jika VRAM longgar, turunkan ke 16 jika OOM.
WORKERS = 8
CACHE = False       # cache="ram" di Windows diduplikasi ke setiap worker -> boros RAM. Dataset di NVMe sudah cukup cepat.
EXP_NAME = "ft_far_range_1024"
FORCE_RETRAIN = False   # True = timpa run yang sudah ada (default: berhenti agar tidak menimpa tanpa sengaja)

# Tahap 1: backbone dibekukan -> head/neck beradaptasi ke 1024 + kelas baru tanpa merusak fitur lama
STAGE1_EPOCHS = 10
STAGE1_FREEZE = 10
STAGE1_LR0 = 0.001
# Tahap 2: seluruh jaringan, LR kecil
STAGE2_EPOCHS = 60
STAGE2_LR0 = 0.0003
PATIENCE = 20

# ---------------- Deploy ----------------
APP_MAX_BATCH = 16                         # = cctv.yolo_batch_size di config aplikasi
FINAL_MODEL_NAME = "model_yolo_far1024.pt"  # nama file untuk workspace/models/YOLO/

print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset dir  : {DATASET_DIR}")
print(f"Kelas target : {TARGET_NAMES}")
''')

# =====================================================================
add_md(r'''
### Langkah 1: Verifikasi GPU, CUDA, dan Library
RTX 50xx (Blackwell, compute capability 12.x) **wajib** PyTorch build CUDA ≥ 12.8:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
''')
add_code(r'''
import sys
import torch

print(f"Python     : {sys.version.split()[0]}")
print(f"PyTorch    : {torch.__version__} (CUDA build {torch.version.cuda})")
assert torch.cuda.is_available(), "GPU CUDA tidak terdeteksi — cek driver NVIDIA & PyTorch build CUDA"

_cap = torch.cuda.get_device_capability(DEVICE)
_props = torch.cuda.get_device_properties(DEVICE)
print(f"GPU        : {_props.name} | VRAM {_props.total_memory / 2**30:.1f} GB | compute capability {_cap}")

_arch = f"sm_{_cap[0]}{_cap[1]}"
_archs = torch.cuda.get_arch_list()
if _arch not in _archs and f"compute_{_cap[0]}{_cap[1]}" not in _archs:
    raise RuntimeError(
        f"PyTorch ini tidak punya kernel untuk {_arch} (tersedia: {_archs}). "
        "Untuk RTX 50xx install build cu128: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")
try:
    import tensorrt as trt
    print(f"TensorRT   : {trt.__version__}")
except ImportError:
    print("TensorRT   : belum terpasang (hanya perlu untuk Langkah 14)")
''')

# =====================================================================
add_md(r'''
### Langkah 2: Bobot Dasar (`model_yolo_baru.pt`)
Dipakai sebagai titik awal fine-tuning **dan** sebagai "guru" pseudo-label. Jika belum ada, diunduh dari Hugging Face
(token dibaca dari environment variable `HF_TOKEN` atau login `huggingface-cli`).
''')
add_code(r'''
import shutil

if not BASE_CHECKPOINT.is_file():
    from huggingface_hub import hf_hub_download

    print(f"{BASE_CHECKPOINT.name} belum ada, mengunduh dari Hugging Face ({HF_REPO_ID})...")
    _downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=os.environ.get("HF_TOKEN"))
    shutil.copy2(_downloaded, BASE_CHECKPOINT)

_ckpt = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
_base_model = _ckpt.get("ema") or _ckpt["model"]
BASE_NAMES = [_base_model.names[i] for i in range(len(_base_model.names))]
BASE_IMGSZ = int((_ckpt.get("train_args") or {}).get("imgsz") or 640)
del _ckpt, _base_model

print(f"Checkpoint : {BASE_CHECKPOINT} ({BASE_CHECKPOINT.stat().st_size / 2**20:.1f} MB)")
print(f"Kelas lama : {BASE_NAMES} | dilatih di imgsz={BASE_IMGSZ}")
if TARGET_NAMES[: len(BASE_NAMES)] != BASE_NAMES:
    raise ValueError(
        f"TARGET_NAMES harus diawali kelas model dasar dengan urutan sama.\n  model : {BASE_NAMES}\n  target: {TARGET_NAMES}"
    )
print(f"Kelas baru : {TARGET_NAMES[len(BASE_NAMES):] or '(tidak ada)'}")
''')

# =====================================================================
add_md(r'''
### Langkah 3: Ambil Dataset dari Google Drive ke SSD Lokal
Zip disalin sekali ke `yolo/_downloads/` (dilewati jika ukurannya sama). Membaca zip secara acak langsung dari
folder Drive sangat lambat, jadi semua proses berikutnya bekerja dari salinan lokal.
''')
add_code(r'''
import zipfile


def fetch_zip(src: dict) -> Path:
    zp = Path(src["zip_path"]) if src.get("zip_path") else None
    local = DOWNLOAD_DIR / (zp.name if zp is not None else f"{src['prefix']}.zip")
    if zp is not None and zp.is_file():
        if zp.resolve() == local.resolve():
            return local
        if local.is_file() and local.stat().st_size == zp.stat().st_size:
            print(f"[cache] {local.name}")
            return local
        print(f"Menyalin {zp} -> {local} ({zp.stat().st_size / 2**30:.2f} GB)...")
        tmp = local.with_name(local.name + ".part")
        shutil.copyfile(zp, tmp)
        tmp.replace(local)
        return local
    if local.is_file():
        print(f"[cache] {local.name} (sumber asli tidak ditemukan, memakai salinan lokal)")
        return local
    if src.get("gdrive_id"):
        import gdown  # pip install gdown

        gdown.download(id=src["gdrive_id"], output=str(local), quiet=False)
        return local
    raise FileNotFoundError(
        f"Dataset '{src['prefix']}' tidak ditemukan di {zp} dan gdrive_id kosong. "
        "Perbaiki zip_path / gdrive_id di Langkah 0."
    )


LOCAL_ZIPS = {}
for _src in DATASET_SOURCES:
    _local = fetch_zip(_src)
    if not zipfile.is_zipfile(_local):
        raise ValueError(f"{_local} bukan file zip yang valid (unduhan rusak?)")
    LOCAL_ZIPS[_src["prefix"]] = _local
for _src in REPLAY_SOURCES:
    if not Path(_src["dir"]).is_dir():
        raise FileNotFoundError(f"Folder replay '{_src['prefix']}' tidak ditemukan: {_src['dir']}")
print("Semua sumber dataset tersedia di lokal.")
''')

# =====================================================================
add_md(r'''
### Langkah 4: Ekstraksi & Remap Kelas (berdasarkan nama)
- Nama kelas dibaca dari `data.yaml` di dalam zip lalu dipetakan lewat `class_map` → tidak tergantung urutan id Roboflow.
  Jika nama di `class_map` tidak ada di dataset, proses **berhenti** (bukan diam-diam menghasilkan label kosong).
- Label poligon/OBB dikonversi ke bounding box; koordinat di-clip ke [0, 1].
- Jika dataset tidak punya split validasi, ~10% train dipindah ke val secara deterministik.
- Label asli disimpan di `dataset/labels_gt/`; folder `labels/` final disusun di Langkah 5.
- Idempotent: sumber yang tidak berubah dilewati; sumber yang dihapus dari konfigurasi ikut dibersihkan.
''')
add_code(r'''
import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath

import yaml
from tqdm.auto import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLITS = ("train", "val", "test")
GT_DIR = DATASET_DIR / "labels_gt"
PSEUDO_DIR = DATASET_DIR / "labels_pseudo"
MANIFEST_PATH = DATASET_DIR / "prepare_manifest.json"


def norm(name) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


NAME_TO_ID = {norm(n): i for i, n in enumerate(TARGET_NAMES)}


class ZipSource:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.names = [n for n in self.z.namelist() if not n.endswith("/")]

    def read(self, name):
        return self.z.read(name)

    def close(self):
        self.z.close()


class DirSource:
    def __init__(self, root):
        self.root = Path(root)
        self.names = [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def read(self, name):
        return (self.root / name).read_bytes()

    def close(self):
        pass


def read_source_names(source) -> list:
    yamls = sorted(
        (n for n in source.names if PurePosixPath(n).name.lower() in ("data.yaml", "data.yml")),
        key=lambda n: n.count("/"),
    )
    if not yamls:
        raise FileNotFoundError("data.yaml tidak ada di dalam sumber; kelas tidak bisa dipetakan berdasarkan nama")
    names = yaml.safe_load(source.read(yamls[0]))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def build_id_map(src: dict, src_names: list):
    prefix = src["prefix"]
    normalized = [norm(n) for n in src_names]
    cmap = src.get("class_map")
    if cmap is None:
        cmap = {n: n for n in normalized if n in NAME_TO_ID}
    cmap = {norm(k): norm(v) for k, v in cmap.items()}
    bad_targets = sorted({v for v in cmap.values() if v not in NAME_TO_ID})
    if bad_targets:
        raise ValueError(f"[{prefix}] kelas target {bad_targets} tidak ada di TARGET_NAMES {TARGET_NAMES}")
    missing = sorted(k for k in cmap if k not in normalized)
    if missing:
        raise ValueError(f"[{prefix}] kelas {missing} tidak ada di data.yaml dataset. Kelas tersedia: {src_names}")
    id_map = {i: NAME_TO_ID[cmap[n]] for i, n in enumerate(normalized) if n in cmap}
    dropped = [src_names[i] for i in range(len(src_names)) if i not in id_map]
    return id_map, dropped


def split_of(name: str):
    for part in PurePosixPath(name).parts[:-1]:
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return None


def label_name_for(img_name: str) -> str:
    parts = list(PurePosixPath(img_name).parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return str(PurePosixPath(*parts).with_suffix(".txt"))


def convert_line(line: str, id_map: dict):
    vals = line.split()
    if len(vals) < 5:
        return None
    try:
        cls = int(float(vals[0]))
        coords = [float(v) for v in vals[1:]]
    except ValueError:
        return None
    if cls not in id_map:
        return None
    if len(coords) == 4:
        xc, yc, w, h = coords
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    elif len(coords) >= 6 and len(coords) % 2 == 0:  # poligon / OBB -> bbox
        xs, ys = coords[0::2], coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
        return None
    return id_map[cls], ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def format_rows(rows) -> str:
    return "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, (x, y, w, h) in rows)


def clear_prefix(prefix: str) -> None:
    for root in (DATASET_DIR / "images", DATASET_DIR / "labels", GT_DIR, PSEUDO_DIR):
        for split in SPLITS:
            d = root / split
            if d.is_dir():
                for f in d.glob(f"{prefix}__*"):
                    f.unlink()


manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
manifest.setdefault("sources", {})
manifest.setdefault("pseudo", {})

ALL_SOURCES = DATASET_SOURCES + REPLAY_SOURCES
_prefixes = [s["prefix"] for s in ALL_SOURCES]
assert len(set(_prefixes)) == len(_prefixes), f"prefix sumber harus unik: {_prefixes}"
assert all("__" not in p for p in _prefixes), "prefix tidak boleh mengandung '__'"

for _old in sorted(set(manifest["sources"]) - set(_prefixes)):
    print(f"[hapus] sumber '{_old}' tidak ada lagi di konfigurasi -> file-nya dibersihkan")
    clear_prefix(_old)
    manifest["sources"].pop(_old, None)
    manifest["pseudo"].pop(_old, None)

for split in SPLITS:
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (GT_DIR / split).mkdir(parents=True, exist_ok=True)

for src in ALL_SOURCES:
    prefix = src["prefix"]
    if src.get("dir"):
        source = DirSource(src["dir"])
        fingerprint = {"kind": "dir", "path": str(src["dir"]), "files": len(source.names)}
    else:
        zpath = LOCAL_ZIPS[prefix]
        source = ZipSource(zpath)
        fingerprint = {"kind": "zip", "path": str(zpath), "size": zpath.stat().st_size}
    fingerprint["class_map"] = src.get("class_map")
    fingerprint["target_names"] = TARGET_NAMES

    prev = manifest["sources"].get(prefix)
    if prev and prev.get("fingerprint") == fingerprint and not FORCE_REBUILD_DATASET:
        print(f"[lewat] {prefix}: tidak berubah ({prev['images']} gambar)")
        source.close()
        continue

    try:
        src_names = read_source_names(source)
        id_map, dropped = build_id_map(src, src_names)
        print(f"\n[{prefix}] kelas dataset: {src_names}")
        print(f"  dipetakan: { {src_names[k]: TARGET_NAMES[v] for k, v in id_map.items()} }")
        if dropped:
            print(f"  dibuang  : {dropped}")
        clear_prefix(prefix)

        name_set = set(source.names)
        images = sorted(n for n in source.names if PurePosixPath(n).suffix.lower() in IMG_EXTS)
        if not images:
            raise ValueError(f"[{prefix}] tidak ada gambar di dalam sumber")
        split_map = {n: split_of(n) or "train" for n in images}
        if "val" not in split_map.values():
            for n in images:
                if split_map[n] == "train" and int(hashlib.md5(n.encode()).hexdigest(), 16) % 10 == 0:
                    split_map[n] = "val"
            print("  tidak ada split val -> 10% train dipindah ke val")

        split_counts, box_counts, no_label = Counter(), Counter(), 0
        used_stems = set()
        for n in tqdm(images, desc=f"Ekstrak {prefix}"):
            split = split_map[n]
            stem = f"{prefix}__{PurePosixPath(n).stem}"
            if (split, stem) in used_stems:  # nama sama di subfolder berbeda
                stem = f"{stem}_{hashlib.md5(n.encode()).hexdigest()[:8]}"
            used_stems.add((split, stem))

            rows = []
            label_name = label_name_for(n)
            if label_name in name_set:
                for line in source.read(label_name).decode("utf-8", errors="ignore").splitlines():
                    row = convert_line(line, id_map)
                    if row:
                        rows.append(row)
            else:
                no_label += 1

            (DATASET_DIR / "images" / split / f"{stem}{PurePosixPath(n).suffix.lower()}").write_bytes(source.read(n))
            (GT_DIR / split / f"{stem}.txt").write_text(format_rows(rows), encoding="utf-8")
            split_counts[split] += 1
            box_counts.update(TARGET_NAMES[c] for c, _ in rows)
    finally:
        source.close()

    if no_label:
        print(f"  {no_label} gambar tanpa file label (dianggap gambar background)")
    manifest["sources"][prefix] = {
        "fingerprint": fingerprint,
        "images": len(images),
        "splits": dict(split_counts),
        "annotated": sorted(set(id_map.values())),
        "pseudo_label": bool(src.get("pseudo_label", True)),
        "dropped_classes": dropped,
        "gt_boxes": dict(box_counts),
    }
    manifest["pseudo"].pop(prefix, None)  # pseudo-label harus dibuat ulang untuk sumber ini
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nRingkasan sumber:")
for prefix, info in manifest["sources"].items():
    print(f"  {prefix:<10} gambar={info['images']:>7,}  split={info['splits']}  "
          f"anotasi={[TARGET_NAMES[c] for c in info['annotated']]}")
    print(f"  {'':<10} box asli={info['gt_boxes']}")
''')

# =====================================================================
add_md(r'''
### Langkah 5: Pseudo-Label dari Model Lama + Susun Label Final
Untuk setiap sumber, kelas **lama** yang tidak dianotasi dataset itu (mis. `person`/`car` di dataset pisau & wajah)
diprediksi oleh `model_yolo_baru.pt` dengan ambang `PSEUDO_CONF`, lalu digabung dengan label asli ke `dataset/labels/`.
Hasil prediksi di-cache di `labels_pseudo/`, jadi menjalankan ulang cell ini hanya menyusun ulang label (cepat).

> Keterbatasan: kelas yang tidak dikenal model lama (`face`) tidak bisa di-pseudo-label. Wajah di dataset VisDrone/pisau
> tetap tidak berlabel — dampaknya kecil untuk drone (wajah nyaris tak terlihat), dan bisa diperbaiki di putaran
> berikutnya dengan model hasil notebook ini sebagai guru.
''')
add_code(r'''
from ultralytics import YOLO

BASE_IDS = range(len(BASE_NAMES))
_conf_by_id = {NAME_TO_ID[norm(k)]: v for k, v in PSEUDO_CONF.items() if norm(k) in NAME_TO_ID}


def pseudo_targets(info: dict) -> dict:
    if not (PSEUDO_LABEL_ENABLED and info["pseudo_label"]):
        return {}
    return {c: _conf_by_id.get(c, PSEUDO_CONF_DEFAULT) for c in BASE_IDS if c not in info["annotated"]}


teacher = None
pseudo_counts = Counter()
for prefix, info in manifest["sources"].items():
    want = pseudo_targets(info)
    if not want:
        continue
    key = {"want": {str(k): v for k, v in want.items()}, "imgsz": PSEUDO_IMGSZ, "splits": list(PSEUDO_SPLITS),
           "teacher": str(BASE_CHECKPOINT), "teacher_size": BASE_CHECKPOINT.stat().st_size}
    if manifest["pseudo"].get(prefix, {}).get("key") == key and not FORCE_PSEUDO_LABEL:
        print(f"[lewat] pseudo-label {prefix}: sudah ada")
        pseudo_counts.update(manifest["pseudo"][prefix]["boxes"])
        continue

    if teacher is None:
        teacher = YOLO(str(BASE_CHECKPOINT))
    print(f"\n[{prefix}] pseudo-label kelas: {[TARGET_NAMES[c] for c in want]}")
    boxes = Counter()
    for split in PSEUDO_SPLITS:
        out_dir = PSEUDO_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob(f"{prefix}__*.txt"):
            f.unlink()
        images = sorted(p for p in (DATASET_DIR / "images" / split).glob(f"{prefix}__*") if p.suffix.lower() in IMG_EXTS)
        if not images:
            continue
        # Sumber berupa file .txt daftar path: streaming & path asli terjaga.
        # (List Python di-load semua ke RAM dan nama file-nya diganti "image0.jpg", dst.)
        list_file = PSEUDO_DIR / f"_{prefix}_{split}.txt"
        list_file.write_text("\n".join(str(p) for p in images), encoding="utf-8")
        results = teacher.predict(str(list_file), imgsz=PSEUDO_IMGSZ, conf=min(want.values()), device=DEVICE,
                                  half=True, batch=32, stream=True, verbose=False)
        written = 0
        for r in tqdm(results, total=len(images), desc=f"{prefix}/{split}"):
            stem = Path(r.path).stem
            assert stem.startswith(f"{prefix}__"), f"path hasil prediksi tidak dikenali: {r.path}"
            rows = []
            if r.boxes is not None and len(r.boxes):
                for c, s, xywh in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                    if c in want and s >= want[c]:
                        rows.append((c, tuple(xywh)))
                        boxes[TARGET_NAMES[c]] += 1
            (out_dir / f"{stem}.txt").write_text(format_rows(rows), encoding="utf-8")
            written += 1
        list_file.unlink()
        assert written == len(images), f"{prefix}/{split}: {written} dari {len(images)} gambar diproses"
    manifest["pseudo"][prefix] = {"key": key, "boxes": dict(boxes)}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pseudo_counts.update(boxes)
    print(f"  box pseudo: {dict(boxes)}")

if teacher is not None:
    del teacher
    torch.cuda.empty_cache()

# ---- Susun labels/ final = label asli (+ pseudo untuk split PSEUDO_SPLITS) ----
final_counts = {s: Counter() for s in SPLITS}
for split in SPLITS:
    lbl_dir = DATASET_DIR / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    gt_files = {f.name: f for f in (GT_DIR / split).glob("*.txt")}
    for stale in lbl_dir.glob("*.txt"):
        if stale.name not in gt_files:
            stale.unlink()
    for name, gt in gt_files.items():
        text = gt.read_text(encoding="utf-8")
        prefix = name.split("__", 1)[0]
        info = manifest["sources"].get(prefix)
        pseudo = PSEUDO_DIR / split / name
        if split in PSEUDO_SPLITS and info and pseudo_targets(info) and pseudo.is_file():
            text += pseudo.read_text(encoding="utf-8")
        (lbl_dir / name).write_text(text, encoding="utf-8")
        final_counts[split].update(TARGET_NAMES[int(l.split()[0])] for l in text.splitlines() if l.strip())
for cache in (DATASET_DIR / "labels").glob("*.cache"):
    cache.unlink()

print("\nJumlah box final per kelas:")
print(f"  {'kelas':<10}" + "".join(f"{s:>12}" for s in SPLITS))
for name in TARGET_NAMES:
    print(f"  {name:<10}" + "".join(f"{final_counts[s][name]:>12,}" for s in SPLITS))

LOW_DATA = [n for n in TARGET_NAMES if final_counts["train"][n] < 300]
if LOW_DATA:
    print(f"\n[PERINGATAN] Kelas dengan < 300 box di train: {LOW_DATA}")
    print("  Kelas ini berisiko 'dilupakan' model karena hampir tidak muncul sebagai positif saat fine-tuning.")
    print("  Solusi: tambahkan dataset untuk kelas tersebut di DATASET_SOURCES (mis. dataset senjata api untuk 'weapon')")
    print("  atau data lama di REPLAY_SOURCES. Tahap 1 (backbone beku) + LR kecil hanya mengurangi, bukan menghilangkan risiko.")
''')

# =====================================================================
add_md(r'''
### Langkah 6: Buat `data_local.yaml`
''')
add_code(r'''
DATA_YAML = CONFIGS_DIR / "data_local.yaml"
for split in ("train", "val"):
    assert any((DATASET_DIR / "images" / split).iterdir()), f"images/{split} kosong — cek Langkah 4"

_data_cfg = {
    "path": DATASET_DIR.as_posix(),
    "train": "images/train",
    "val": "images/val",
    "names": {i: n for i, n in enumerate(TARGET_NAMES)},
}
if any((DATASET_DIR / "images" / "test").iterdir()):
    _data_cfg["test"] = "images/test"
DATA_YAML.write_text(yaml.safe_dump(_data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(DATA_YAML.read_text(encoding="utf-8"))
''')

# =====================================================================
add_md(r'''
### Langkah 7: Perluas Head Model Lama (5 → 6 kelas) Tanpa Membuang Bobot
Jika jumlah kelas berubah, Ultralytics secara default **membuang** lapisan klasifikasi yang ukurannya tidak cocok
(kemampuan klasifikasi lama hilang). Di sini lapisan tersebut diperluas: baris kelas lama disalin persis, hanya baris
kelas baru yang diinisialisasi. Hasilnya diverifikasi: output kelas lama harus **identik** dengan model lama.
''')
add_code(r'''
import copy
import datetime

from ultralytics.nn.tasks import DetectionModel


def expand_checkpoint(src_pt: Path, dst_pt: Path, names: list) -> Path:
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    old = (ckpt.get("ema") or ckpt["model"]).float().eval()
    old_nc, new_nc = len(old.names), len(names)
    if new_nc == old_nc:
        print("Jumlah kelas sama -> checkpoint dasar dipakai langsung")
        return src_pt
    assert new_nc > old_nc, "kelas hanya boleh ditambah, bukan dikurangi"

    cfg = copy.deepcopy(old.yaml)
    cfg["nc"] = new_nc
    new = DetectionModel(cfg, nc=new_nc, verbose=False).float().eval()
    old_sd, new_sd = old.state_dict(), new.state_dict()
    expanded = []
    for k, v_new in new_sd.items():
        v_old = old_sd[k]
        if v_old.shape == v_new.shape:
            new_sd[k] = v_old.clone()
        elif v_old.shape[0] == old_nc and v_new.shape[0] == new_nc and v_old.shape[1:] == v_new.shape[1:]:
            t = v_new.clone()
            t[:old_nc] = v_old
            new_sd[k] = t
            expanded.append(k)
        else:
            raise ValueError(f"Bentuk tensor tidak terduga {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
    new.load_state_dict(new_sd)
    new.names = {i: n for i, n in enumerate(names)}
    new.args = getattr(old, "args", {})
    print(f"Tensor diperluas ({len(expanded)}): {expanded}")

    # Verifikasi: logit kelas lama di setiap cabang head harus identik
    head_old, head_new = old.model[-1], new.model[-1]
    captured = {}

    def hook(tag):
        return lambda m, i, o: captured.__setitem__(tag, o.detach())

    handles = []
    for attr in ("cv3", "one2one_cv3"):
        for idx, (mo, mn) in enumerate(zip(getattr(head_old, attr, []), getattr(head_new, attr, []))):
            handles += [mo.register_forward_hook(hook(("old", attr, idx))), mn.register_forward_hook(hook(("new", attr, idx)))]
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        old(x)
        new(x)
    for h in handles:
        h.remove()
    compared = 0
    for (who, attr, idx), out_old in captured.items():
        if who != "old" or ("new", attr, idx) not in captured:
            continue
        out_new = captured[("new", attr, idx)]
        assert torch.allclose(out_new[:, :old_nc], out_old, atol=1e-5), f"output {attr}[{idx}] berubah!"
        compared += 1
    assert compared > 0, "verifikasi head tidak berjalan"
    print(f"Verifikasi OK: {compared} cabang head menghasilkan logit kelas lama yang identik")

    torch.save(
        {
            "date": datetime.datetime.now().isoformat(),
            "version": ckpt.get("version"),
            "epoch": -1,
            "best_fitness": None,
            "model": new.half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "train_args": ckpt.get("train_args", {}),
        },
        dst_pt,
    )
    return dst_pt


INIT_CHECKPOINT = expand_checkpoint(BASE_CHECKPOINT, WEIGHTS_DIR / f"{BASE_CHECKPOINT.stem}_{len(TARGET_NAMES)}cls_init.pt", TARGET_NAMES)
print(f"Checkpoint awal fine-tuning: {INIT_CHECKPOINT}")
print(f"Kelas: {YOLO(str(INIT_CHECKPOINT)).names}")
''')

# =====================================================================
add_md(r'''
### Langkah 8: Training Tahap 1 — Backbone Dibekukan
`freeze=10` membekukan backbone sehingga fitur lama terlindungi, sementara neck/head menyesuaikan diri ke resolusi
1024 dan belajar kelas baru. `warmup_bias_lr=0` penting: default Ultralytics (0.1) terlalu besar untuk head yang
sudah terlatih bila memakai AdamW.
''')
add_code(r'''
COMMON_TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    device=DEVICE,
    cache=CACHE,
    optimizer="AdamW",
    cos_lr=True,
    warmup_bias_lr=0.0,
    mosaic=1.0,
    scale=0.7,          # variasi skala lebar -> mensimulasikan objek jauh/kecil
    fliplr=0.5,
    project=str(RUNS_DIR),
    exist_ok=True,
    plots=True,
)


def stage_dir(stage: str) -> Path:
    return RUNS_DIR / f"{EXP_NAME}_{stage}"


def guard_existing(stage: str) -> bool:
    last = stage_dir(stage) / "weights" / "last.pt"
    if last.is_file() and not FORCE_RETRAIN:
        print(f"[lewat] {stage_dir(stage).name} sudah ada. Pakai cell Resume jika terputus, "
              "atau set FORCE_RETRAIN=True untuk menimpa.")
        return False
    return True


STAGE1_BEST = stage_dir("s1") / "weights" / "best.pt"
if STAGE1_EPOCHS > 0 and guard_existing("s1"):
    YOLO(str(INIT_CHECKPOINT)).train(
        **COMMON_TRAIN_ARGS,
        name=stage_dir("s1").name,
        epochs=STAGE1_EPOCHS,
        freeze=STAGE1_FREEZE,
        lr0=STAGE1_LR0,
        lrf=0.1,
        warmup_epochs=1,
        close_mosaic=0,
        patience=STAGE1_EPOCHS,
    )
STAGE2_START = STAGE1_BEST if STAGE1_EPOCHS > 0 else INIT_CHECKPOINT
print(f"Titik awal tahap 2: {STAGE2_START}")
''')

# =====================================================================
add_md(r'''
### Langkah 9: Training Tahap 2 — Fine-Tune Seluruh Jaringan
''')
add_code(r'''
FINAL_BEST = stage_dir("s2") / "weights" / "best.pt"
if guard_existing("s2"):
    assert Path(STAGE2_START).is_file(), f"{STAGE2_START} tidak ada — tahap 1 belum selesai?"
    YOLO(str(STAGE2_START)).train(
        **COMMON_TRAIN_ARGS,
        name=stage_dir("s2").name,
        epochs=STAGE2_EPOCHS,
        lr0=STAGE2_LR0,
        lrf=0.05,
        warmup_epochs=1,
        close_mosaic=10,    # 10 epoch terakhir tanpa mosaic untuk lokalisasi presisi
        patience=PATIENCE,
    )
print(f"Model final: {FINAL_BEST}")
''')

# =====================================================================
add_md(r'''
### Langkah 9B: Resume (jika training terputus)
Isi `RESUME_STAGE` dengan `"s1"` atau `"s2"`, lalu jalankan cell tahap berikutnya seperti biasa.
''')
add_code(r'''
RESUME_STAGE = "s2"
_last = stage_dir(RESUME_STAGE) / "weights" / "last.pt"
if _last.is_file():
    YOLO(str(_last)).train(resume=True)
else:
    print(f"{_last} belum ada.")
''')

# =====================================================================
add_md(r'''
### Langkah 10: Evaluasi Per Kelas (model final)
Val berisi label asli + pseudo-label; split test (jika ada) hanya label asli.
''')
add_code(r'''
eval_model = YOLO(str(FINAL_BEST))


def per_class_table(metrics, names) -> None:
    box = metrics.box
    print(f"  {'kelas':<10}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for j, c in enumerate(box.ap_class_index):
        print(f"  {names[int(c)]:<10}{box.p[j]:>8.3f}{box.r[j]:>8.3f}{box.ap50[j]:>8.3f}{box.ap[j]:>10.3f}")
    print(f"  {'SEMUA':<10}{box.mp:>8.3f}{box.mr:>8.3f}{box.map50:>8.3f}{box.map:>10.3f}")


for split in ("val", "test"):
    if split not in _data_cfg:
        continue
    m = eval_model.val(data=str(DATA_YAML), split=split, imgsz=IMGSZ, batch=BATCH, device=DEVICE, workers=WORKERS,
                       project=str(RUNS_DIR / "eval"), name=f"{EXP_NAME}_{split}", exist_ok=True, verbose=False)
    print(f"\n=== {split.upper()} @ imgsz {IMGSZ} ===")
    per_class_table(m, TARGET_NAMES)
''')

# =====================================================================
add_md(r'''
### Langkah 11: Bandingkan Model Lama vs Baru (cek regresi & peningkatan jarak jauh)
Per sumber dataset, hanya memakai **label asli** (tanpa pseudo-label, supaya tidak bias ke model lama).
Kolom: model lama @640 (kondisi produksi sekarang), model lama @IMGSZ (efek resolusi saja), model baru @IMGSZ.
Baris VisDrone adalah indikator utama peningkatan deteksi jarak jauh.

> `weapon` tidak punya label asli di dataset mana pun, jadi tidak bisa diukur di sini — uji manual dengan footage nyata.
''')
add_code(r'''
EVAL_DIR = DATASET_DIR / "_eval"


def build_eval_set(prefix: str, split: str, keep_ids: set, tag: str) -> Path:
    root = EVAL_DIR / f"{prefix}_{split}_{tag}"
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for img in (DATASET_DIR / "images" / split).glob(f"{prefix}__*"):
        dst = root / "images" / img.name
        try:
            os.link(img, dst)  # hardlink: tanpa menyalin data
        except OSError:
            shutil.copy2(img, dst)
        gt = GT_DIR / split / f"{img.stem}.txt"
        lines = [l for l in gt.read_text(encoding="utf-8").splitlines() if l.strip() and int(l.split()[0]) in keep_ids]
        (root / "labels" / f"{img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def eval_yaml(root: Path, names: list) -> Path:
    path = root / "data.yaml"
    cfg = {"path": root.as_posix(), "train": "images", "val": "images", "names": {i: n for i, n in enumerate(names)}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def class_scores(weights, data, imgsz, tag) -> dict:
    m = YOLO(str(weights)).val(data=str(data), imgsz=imgsz, batch=16, device=DEVICE, workers=WORKERS, plots=False, verbose=False,
                               project=str(RUNS_DIR / "eval"), name=tag, exist_ok=True)
    return {int(c): (m.box.ap50[j], m.box.ap[j], m.box.r[j]) for j, c in enumerate(m.box.ap_class_index)}


regressions = []
for prefix, info in manifest["sources"].items():
    split = "test" if info["splits"].get("test") else "val"
    annotated = set(info["annotated"])
    if not info["splits"].get(split):
        continue
    root_new = build_eval_set(prefix, split, annotated, "new")
    new_scores = class_scores(FINAL_BEST, eval_yaml(root_new, TARGET_NAMES), IMGSZ, f"cmp_{prefix}_new")

    old_ids = annotated & set(BASE_IDS)
    old640, old_hi = {}, {}
    if old_ids:
        root_old = build_eval_set(prefix, split, old_ids, "old")  # tanpa label kelas baru (model lama tidak mengenalnya)
        y_old = eval_yaml(root_old, BASE_NAMES)
        old640 = class_scores(BASE_CHECKPOINT, y_old, BASE_IMGSZ, f"cmp_{prefix}_old{BASE_IMGSZ}")
        old_hi = class_scores(BASE_CHECKPOINT, y_old, IMGSZ, f"cmp_{prefix}_old{IMGSZ}")

    print(f"\n=== {prefix} ({split}, label asli: {[TARGET_NAMES[c] for c in sorted(annotated)]}) — mAP50-95 / recall ===")
    print(f"  {'kelas':<10}{'lama@' + str(BASE_IMGSZ):>16}{'lama@' + str(IMGSZ):>16}{'baru@' + str(IMGSZ):>16}{'selisih':>10}")
    fmt = lambda s: f"{s[1]:.3f} / {s[2]:.3f}" if s else "-"
    for c in sorted(annotated):
        o, oh, n = old640.get(c), old_hi.get(c), new_scores.get(c)
        delta = f"{n[1] - o[1]:+.3f}" if (o and n) else "baru"
        print(f"  {TARGET_NAMES[c]:<10}{fmt(o):>16}{fmt(oh):>16}{fmt(n):>16}{delta:>10}")
        if o and n and n[1] < o[1] - 0.02:
            regressions.append(f"{prefix}/{TARGET_NAMES[c]}: {o[1]:.3f} -> {n[1]:.3f}")

print("\n" + ("[PERINGATAN] Regresi terdeteksi:\n  " + "\n  ".join(regressions) if regressions
              else "Tidak ada regresi > 0.02 mAP50-95 pada kelas yang punya label asli."))
''')

# =====================================================================
add_md(r'''
### Langkah 12: Uji Visual — Model Lama vs Baru
''')
add_code(r'''
import cv2
import matplotlib.pyplot as plt

_old_model = YOLO(str(BASE_CHECKPOINT))
_first = DATASET_SOURCES[0]["prefix"]
_samples = sorted((DATASET_DIR / "images" / "val").glob(f"{_first}__*"))[:3]
for img_p in _samples:
    r_old = _old_model.predict(str(img_p), imgsz=BASE_IMGSZ, conf=0.30, device=DEVICE, verbose=False)[0]
    r_new = eval_model.predict(str(img_p), imgsz=IMGSZ, conf=0.30, device=DEVICE, verbose=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, r, title in ((axes[0], r_old, f"LAMA @{BASE_IMGSZ}: {len(r_old.boxes)} objek"),
                         (axes[1], r_new, f"BARU @{IMGSZ}: {len(r_new.boxes)} objek")):
        ax.imshow(cv2.cvtColor(r.plot(line_width=1, font_size=8), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(img_p.name)
    plt.tight_layout()
    plt.show()
''')

# =====================================================================
add_md(r'''
### Langkah 13: Simpan Model Final & Panduan Deploy
''')
add_code(r'''
FINAL_EXPORT = WEIGHTS_DIR / FINAL_MODEL_NAME
shutil.copy2(FINAL_BEST, FINAL_EXPORT)
print(f"Model final disalin ke: {FINAL_EXPORT}\n")
print(f"""LANGKAH DEPLOY (lakukan bersamaan dalam satu perubahan):
1. Salin {FINAL_EXPORT.name} ke  Drone_Violence_Konser/workspace/models/YOLO/
2. Di Drone_Violence_Konser/app/config/config.yaml:
     detection.yolo_model_path: "workspace/models/YOLO/{FINAL_EXPORT.name}"
     detection.yolo_imgsz: {IMGSZ}
   (jangan ubah yolo_imgsz sebelum model baru dipasang — model lama dilatih di {BASE_IMGSZ})
3. Bangun engine TensorRT DI MESIN PRODUKSI (engine terikat arsitektur GPU & versi TensorRT):
     python yolo/scripts/export_trt.py --weights <path>/{FINAL_EXPORT.name} --imgsz {IMGSZ} --batch {APP_MAX_BATCH}
   File .engine dan .engine.json harus berada di sebelah file .pt.
4. Catatan beban: inferensi @{IMGSZ} kira-kira {(IMGSZ / 640) ** 2:.1f}x lebih berat daripada @640 — cek FPS di GPU produksi.""")
''')

# =====================================================================
add_md(r'''
### Langkah 14 (opsional): Export TensorRT di Mesin Ini
Engine hanya valid untuk GPU dengan **compute capability yang sama** dan versi TensorRT yang sama
(dicek oleh `reference_app/detection/yolo_backend.py`). Engine dari RTX 5090 (12.x) tidak bisa dipakai di
GPU produksi RTX 30xx (8.6). Jalankan cell ini hanya untuk benchmark di sini, atau jika mesin ini = mesin produksi.
''')
add_code(r'''
import subprocess

print(f"GPU ini: compute capability {torch.cuda.get_device_capability(DEVICE)}")
subprocess.run(
    [sys.executable, str(YOLO_DIR / "scripts" / "export_trt.py"), "--weights", str(FINAL_EXPORT),
     "--device", str(DEVICE), "--imgsz", str(IMGSZ), "--batch", str(APP_MAX_BATCH)],
    check=True,
    env={**os.environ},
)
''')

# =====================================================================
add_md(r'''
### Langkah 15 (opsional): Benchmark Kecepatan
''')
add_code(r'''
import time

_engine = FINAL_EXPORT.with_suffix(".engine")
_bench_path = _engine if _engine.is_file() else FINAL_EXPORT
bench_model = YOLO(str(_bench_path), task="detect")
print(f"Benchmark: {_bench_path.name}")

for bs in (1, APP_MAX_BATCH):
    x = torch.rand(bs, 3, IMGSZ, IMGSZ, device=f"cuda:{DEVICE}")
    for _ in range(10):
        bench_model(x, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        bench_model(x, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    avg = sum(lat) / len(lat)
    print(f"  batch={bs:<3} rata-rata {avg:7.2f} ms/batch | {bs * 1000 / avg:7.1f} gambar/detik | min {min(lat):.2f} ms")
''')

out_path = Path(__file__).resolve().parents[1] / "notebooks" / "yolo_train_local.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Notebook lokal berhasil dibuat:", out_path)
