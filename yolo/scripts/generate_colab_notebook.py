"""Generate complete Google Colab notebook for CCTV & Drone multi-dataset YOLO training.
Includes Colab Free Tier optimizations and seamless Resume Training cell.
"""
from __future__ import annotations

import json
from pathlib import Path

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

def add_md(text: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# ==================== CELL 1: HEADER ====================
add_md("""# Pipeline Training YOLO Multi-Dataset (CCTV & Drone Surveillance)
### *Dioptimalkan untuk Google Colab Free Tier (Anti-Lost Progress & Resume Support)*

Notebook ini menggabungkan dan menyelaraskan 3 dataset dari Google Drive Anda secara otomatis:
1. 📁 **VisDrone** (`visdrone/VisDrone.v3-validation-set.yolo26.zip`) $\\rightarrow$ `person` (0), `bicycle` (3), `car` (4)
2. 📁 **Knife Detection** (`knife_detection/Knife detection.v1i.yolo26.zip`) $\\rightarrow$ `knife` (1)
3. 📁 **Face Detection** (`face_detection/face-detection.v1i.yolo26.zip`) $\\rightarrow$ `face` (5)

### Target 6 Kelas Model:
- **0: person** (orang/pejalan kaki jarak dekat & jauh)
- **1: knife** (pisau/senjata tajam)
- **2: weapon** (senjata api / threats)
- **3: bicycle** (sepeda)
- **4: car** (mobil)
- **5: face** (wajah untuk CCTV)
""")

# ==================== CELL 2: GPU CHECK ====================
add_md("""### Langkah 1: Cek Ketersediaan GPU
Pastikan runtime Colab diatur ke GPU (**Runtime > Change runtime type > T4 GPU**).""")
add_code("!nvidia-smi")

# ==================== CELL 3: MOUNT GOOGLE DRIVE ====================
add_md("""### Langkah 2: Mount Google Drive
Menghubungkan Google Drive agar ketiga folder dataset terbaca dan checkpoint tersimpan langsung di Drive.""")
add_code("""from google.colab import drive
drive.mount('/content/drive')""")

# ==================== CELL 4: INSTALL DEPENDENCIES ====================
add_md("""### Langkah 3: Install Ultralytics & Dependencies""")
add_code("!pip install -q ultralytics huggingface_hub tqdm")

# ==================== CELL 5: DOWNLOAD BASE CHECKPOINT ====================
add_md("""### Langkah 4: Unduh Base Checkpoint (`model_yolo_baru.pt`)
Mengambil bobot dasar dari repository Hugging Face private Anda.""")
add_code("""import os
from huggingface_hub import hf_hub_download

os.makedirs('/content/weights', exist_ok=True)

# Token disimpan di Colab Secrets (ikon kunci di sidebar) dengan nama HF_TOKEN — jangan ditulis di notebook
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
REPO_ID = 'mtg-ai-team/drone-violence-models'

print('Mengunduh model_yolo_baru.pt dari Hugging Face...')
base_model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename='YOLO/model_yolo_baru.pt',
    local_dir='/content/weights',
    token=HF_TOKEN
)
print('Base weights berhasil diunduh ke:', base_model_path)""")

# ==================== CELL 6: UNIVERSAL DATASET MERGER ====================
add_md("""### Langkah 5: Ekstrak & Re-Mapping Otomatis Ketiga Dataset
Cell ini memproses ketiga dataset sekaligus dengan aman tanpa bentrok nama file atau class ID:
- **VisDrone**: Pedestrian(7) & People(8) $\\rightarrow$ 0, Bicycle(1) $\\rightarrow$ 3, Car(3) $\\rightarrow$ 4
- **Knife**: knife(0) $\\rightarrow$ 1
- **Face**: Face(0) & face(1) $\\rightarrow$ 5""")
add_code("""import os
import zipfile
from tqdm import tqdm

OUT_DIR = '/content/dataset'

# Siapkan folder tujuan standar YOLO
for split in ['train', 'val', 'test']:
    os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

# Konfigurasi 3 dataset Anda
DATASETS_CONFIG = [
    {
        "name": "VisDrone",
        "zip_path": "/content/drive/MyDrive/visdrone/VisDrone.v3-validation-set.yolo26.zip",
        "prefix": "visdrone",
        "class_map": {
            7: 0,  # Pedestrian -> person (0)
            8: 0,  # People     -> person (0)
            1: 3,  # Bicycle    -> bicycle (3)
            3: 4,  # Car        -> car (4)
        }
    },
    {
        "name": "Knife Detection",
        "zip_path": "/content/drive/MyDrive/knife_detection/Knife detection.v1i.yolo26.zip",
        "prefix": "knife",
        "class_map": {
            0: 1   # knife -> knife (1)
        }
    },
    {
        "name": "Face Detection",
        "zip_path": "/content/drive/MyDrive/face_detection/face-detection.v1i.yolo26.zip",
        "prefix": "face",
        "class_map": {
            0: 5,  # Face -> face (5)
            1: 5   # face -> face (5)
        }
    }
]

total_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

for cfg in DATASETS_CONFIG:
    zip_path = cfg["zip_path"]
    prefix = cfg["prefix"]
    cmap = cfg["class_map"]
    dname = cfg["name"]
    
    if not os.path.exists(zip_path):
        print(f"[PERINGATAN] File {zip_path} tidak ditemukan, dilewati!")
        continue
        
    print(f"\\n{'='*60}")
    print(f"Memproses {dname} ({zip_path})...")
    print(f"{'='*60}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        imgs = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Jumlah gambar dalam zip: {len(imgs):,}")
        
        for img_path in tqdm(imgs, desc=f"Extracting {prefix}"):
            # Mapping split folder
            if img_path.startswith('train/'):
                target_split = 'train'
            elif img_path.startswith('valid/') or img_path.startswith('val/'):
                target_split = 'val'
            elif img_path.startswith('test/'):
                target_split = 'test'
            else:
                target_split = 'train'
                
            filename = os.path.basename(img_path)
            base_name, _ = os.path.splitext(filename)
            
            # Cari file label
            label_dir = os.path.dirname(img_path).replace('images', 'labels')
            label_path = f"{label_dir}/{base_name}.txt"
            
            new_lines = []
            if label_path in all_files:
                raw_labels = z.read(label_path).decode('utf-8', errors='ignore').splitlines()
                for line in raw_labels:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        orig_cls = int(parts[0])
                        if orig_cls in cmap:
                            new_cls = cmap[orig_cls]
                            total_stats[new_cls] += 1
                            new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                    except ValueError:
                        continue
            
            # Simpan dengan prefix unik
            out_img = f"{OUT_DIR}/images/{target_split}/{prefix}_{filename}"
            out_lbl = f"{OUT_DIR}/labels/{target_split}/{prefix}_{base_name}.txt"
            
            with open(out_img, 'wb') as f_img:
                f_img.write(z.read(img_path))
            with open(out_lbl, 'w') as f_lbl:
                f_lbl.write("\\n".join(new_lines))

print(f"\\n{'='*60}")
print("PENGGABUNGAN DATASET SELESAI!")
print(f"{'='*60}")
print("Statistik total objek per kelas yang siap dilatih:")
class_names = {0: 'person', 1: 'knife', 2: 'weapon', 3: 'bicycle', 4: 'car', 5: 'face'}
for cid, cname in class_names.items():
    print(f"  [{cid}] {cname:<10}: {total_stats[cid]:,} bounding box")""")

# ==================== CELL 7: DATA.YAML ====================
add_md("""### Langkah 6: Buat File `data.yaml` Target 6 Kelas""")
add_code('''data_yaml_content = """
path: /content/dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: knife
  2: weapon
  3: bicycle
  4: car
  5: face
"""

with open('/content/data.yaml', 'w') as f:
    f.write(data_yaml_content.strip())

print("File /content/data.yaml 6-kelas berhasil dibuat!")''')

# ==================== CELL 8: TRAINING ====================
add_md("""### Langkah 7: Jalankan Training / Fine-Tuning
Strategi untuk Colab Free Tier:
- `epochs=60`: Karena ini fine-tuning (bukan dari nol), 50-60 epoch sudah mencapai mAP optimal.
- `batch=8` & `imgsz=1024`: Hanya menggunakan ~7.5 GB VRAM dari 16 GB yang tersedia.
- Estimasi waktu: ~1.5 - 2 jam (Selesai jauh sebelum batas waktu timeout Colab Free!).
- Hasil checkpoint disimpan langsung ke Google Drive.""")
add_code("""from ultralytics import YOLO

# 1. Load pretrained base checkpoint
base_checkpoint = '/content/weights/YOLO/model_yolo_baru.pt'
model = YOLO(base_checkpoint)

# 2. Lokasi penyimpanan di Google Drive
drive_save_dir = '/content/drive/MyDrive/YOLO_Retrain_CCTV_Drone'

# 3. Jalankan training
results = model.train(
    data='/content/data.yaml',
    epochs=60,               # 60 epoch sangat optimal untuk fine-tuning di Colab Free
    imgsz=1024,              # Resolusi tinggi (optimal drone & small objects)
    batch=8,                 # Sesuaikan VRAM T4 GPU (~7.5 GB VRAM)
    optimizer='AdamW',       # Optimizer AdamW
    lr0=0.0008,              # Learning rate fine-tuning
    cos_lr=True,             # Cosine learning rate schedule
    scale=0.7,               # Augmentasi skala jarak
    mosaic=1.0,              # Mosaic augmentasi
    close_mosaic=10,         # Matikan mosaic 10 epoch terakhir
    box=8.5,                 # Bobot penalti lokasi bounding box
    patience=20,             # Early stopping jika stagnan 20 epoch berturut-turut
    device=0,                # GPU 0
    save=True,
    save_period=10,          # Simpan checkpoint per 10 epoch
    project=drive_save_dir,
    name='exp_cctv_drone_6cls',
    exist_ok=True
)

print("\\n--- TRAINING SELESAI ---")
print(f"Model terbaik tersimpan di: {drive_save_dir}/exp_cctv_drone_6cls/weights/best.pt")""")

# ==================== CELL 8B: RESUME TRAINING ====================
add_md("""### Langkah 7B: Jalankan Cell Ini HANYA JIKA Training Terputus di Tengah Jalan (Resume)
Jika sesi Google Colab Free Anda terputus (misal karena jaringan atau limit waktu):
1. Sambungkan kembali (*Reconnect*) Colab ke GPU.
2. Jalankan Langkah 2 (Mount Google Drive).
3. Jalankan cell di bawah ini untuk **melanjutkan training dari epoch terakhir tanpa mengulang dari nol!**""")
add_code("""# JALANKAN HANYA JIKA INGIN MELANJUTKAN TRAINING YANG TERPUTUS
from ultralytics import YOLO

last_checkpoint = '/content/drive/MyDrive/YOLO_Retrain_CCTV_Drone/exp_cctv_drone_6cls/weights/last.pt'

if os.path.exists(last_checkpoint):
    print(f"Melanjutkan training dari checkpoint terakhir: {last_checkpoint}")
    model = YOLO(last_checkpoint)
    model.train(resume=True)
else:
    print("Checkpoint last.pt belum ditemukan di Google Drive.")""")

# ==================== CELL 9: EVALUASI ====================
add_md("""### Langkah 8: Evaluasi Akurasi pada Dataset Validasi (Semua Kelas)""")
add_code("""best_model_path = f"{drive_save_dir}/exp_cctv_drone_6cls/weights/best.pt"
eval_model = YOLO(best_model_path)

metrics = eval_model.val(data='/content/data.yaml', imgsz=1024, device=0)

print("\\n--- HASIL EVALUASI MODEL BARU ---")
print(f"mAP@50     : {metrics.box.map50:.4f}")
print(f"mAP@50-95  : {metrics.box.map:.4f}")
print(f"Precision  : {metrics.box.mp:.4f}")
print(f"Recall     : {metrics.box.mr:.4f}")""")

# ==================== CELL 10: VISUAL TEST ====================
add_md("""### Langkah 9: Uji Coba Visual Prediksi pada Gambar Uji""")
add_code("""import glob
import matplotlib.pyplot as plt
import cv2

val_images = glob.glob('/content/dataset/images/val/*.jpg')[:4]

for img_file in val_images:
    preds = eval_model.predict(source=img_file, imgsz=1024, conf=0.30, device=0)
    plot_bgr = preds[0].plot()
    plot_rgb = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(plot_rgb)
    plt.axis('off')
    plt.title(f"Deteksi: {os.path.basename(img_file)}")
    plt.show()""")

# ==================== CELL 11: EXPORT ====================
add_md("""### Langkah 10: Export Model ke Format ONNX
Mengekspor model terbaik `best.pt` ke ONNX agar bisa diunduh ke komputer lokal dan dikonversi ke TensorRT.""")
add_code("""onnx_file = eval_model.export(format='onnx', imgsz=1024, dynamic=False)
print("Model ONNX berhasil diekspor ke:", onnx_file)
print("Unduh file ONNX ini ke komputer Anda untuk dideploy dengan TensorRT!")""")

out_path = Path(__file__).resolve().parents[1] / "notebooks" / "yolo_retrain_jarak_jauh_colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook berhasil diperbarui dengan fitur Resume Colab Free:", out_path.resolve())
