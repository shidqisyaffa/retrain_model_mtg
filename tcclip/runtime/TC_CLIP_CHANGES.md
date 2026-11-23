# Perubahan runtime TC-CLIP

Folder `tc-clip/` adalah runtime inference minimum dari TC-CLIP upstream. Folder
dataset, split dataset, dokumentasi gambar, workspace, checkpoint, hasil CSV,
dan `.git` upstream tidak disalin.

Basis upstream yang dipakai:

```text
https://github.com/naver-ai/tc-clip
commit 8728fe4ca2b27f00e7235ed8f81f59cb933c8ed9
```

Patch yang dipertahankan:

- `torch.load(..., weights_only=True)` untuk bobot backbone CLIP.
- `torch.load(..., weights_only=False)` hanya untuk checkpoint TC-CLIP tepercaya
  yang memuat metadata selain tensor.
- Import `trunc_normal_` menggunakan `timm.layers`, sesuai timm baru.
- Distributed backend memakai NCCL jika tersedia dan Gloo sebagai fallback.
- Tensor sinkronisasi distributed mengikuti device backend, tidak memaksa CUDA.
- Cache backbone dapat diarahkan melalui environment `TC_CLIP_CACHE`.
- Preview mengarahkan cache ke `models/tcclip/` agar paket portable.

Konfigurasi TC-CLIP untuk preview dibuat langsung oleh
`tsm/preview_tsm_action_gabungan.py`, sehingga Hydra/MMCV/decord tidak diperlukan
untuk inference ini.
