# Vendored TC-CLIP Runtime

Runtime TC-CLIP di folder `tc-clip/` **di-vendor** (disalin sebagai file biasa) ke
repo ini agar `git clone` langsung lengkap - tanpa submodule, tanpa token GitHub.

| | |
|---|---|
| Repo asal | `https://github.com/mshalmank11/ujicoba_yolo-tsm-tcclip.git` |
| Commit saat di-vendor | `9201d94e196b78bf744a74ba4606cabd28b7a225` |
| Tanggal vendor | 2026-10-05 |
| Isi yang diambil | `tc-clip/` (clip, tome, trainers, utils, LICENSE, NOTICE), `TC_CLIP_CHANGES.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE` |
| Tidak diambil | `tsm/`, `samples/`, `outputs/`, `data/`, `scripts/`, `models/`, `requirements*`, `.github/` |

## Update runtime

Jalankan dari checkout fork lokal:

```powershell
.\development\setup\sync_tcclip_runtime.ps1 -Source <path-checkout-fork>
```

Script menyalin `tc-clip/` + file notices, lalu memperbarui catatan commit di file ini.

## Catatan

- Jangan mengedit file runtime di sini secara langsung kecuali patch darurat;
  patch tersebut harus dicatat di `TC_CLIP_CHANGES.md` dan disinkronkan kembali
  ke repo asal.
- `app/config/config.yaml` tetap menunjuk `workspace/third_party/tc_clip/tc-clip`.
- Bobot/engine (checkpoint, backbone, TensorRT `.plan`) **tidak** ada di sini -
  diunduh ke `workspace/models/TC_CLIP` lewat `setup_models.ps1` (HF privat).
