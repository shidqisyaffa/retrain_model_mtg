# Third-party notices

## Ultralytics

Preview menggunakan package `ultralytics` untuk YOLO dan TrackTrack. Versi yang
diuji adalah 8.4.115. Ultralytics menyediakan jalur lisensi AGPL-3.0 dan lisensi
enterprise. File `LICENSE` pada root paket berisi AGPL-3.0.

Sumber: <https://github.com/ultralytics/ultralytics>

## TC-CLIP

Runtime pada folder `tc-clip/` berasal dari project TC-CLIP milik NAVER Cloud
Corp. dan mempertahankan file `tc-clip/LICENSE` serta `tc-clip/NOTICE`.
TC-CLIP dilisensikan dengan Creative Commons Attribution-NonCommercial 4.0
International. Jangan gunakan backend TC-CLIP untuk produk komersial tanpa
memastikan izin yang sesuai.

Sumber: <https://github.com/naver-ai/tc-clip>

Runtime ini berisi patch kompatibilitas lokal untuk Python 3.12 dan PyTorch
baru. Detailnya ada di `TC_CLIP_CHANGES.md`.

## Model weights

Bobot YOLO, ReID, TSM, TC-CLIP, dan backbone CLIP tidak otomatis memperoleh
lisensi baru hanya karena dipakai oleh project ini. Pengguna bertanggung jawab
memastikan hak distribusi setiap checkpoint sebelum mengunggahnya ke Release,
Git LFS, atau layanan model lain.

