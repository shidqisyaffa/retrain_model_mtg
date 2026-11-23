"""Install the private model bundle on Windows, Linux, or macOS.

Run uv sync first, then uv run python -m development.setup.setup_models.
Use --verify to check an existing installation without network access.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("model-setup")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "workspace" / "models"
RUNTIME_ROOT = PROJECT_ROOT / "workspace" / "third_party" / "tc_clip" / "tc-clip"
DEFAULT_REPO_ID = "mtg-ai-team/drone-violence-models"
BACKBONE_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CHECKPOINT_SHA256 = "1c6c01bc65894ca9b32f96e3fce813e98be7f287d723dba1860a2ea2aad9fcb1"
BACKBONE_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    + BACKBONE_SHA256
    + "/ViT-B-16.pt"
)
REQUIRED_FILES = (
    "YOLO/yolo26s.pt",
    "TSM/tsm_mobilenetv2_autosplit_best_use.pth",
    "TC_CLIP/ViT-B-16.pt",
    "TC_CLIP/zero_shot_k400_llm_tc_clip.pth",
    "TRANSLATION/opus-mt-id-en-ct2-int8/config.json",
    "TRANSLATION/opus-mt-id-en-ct2-int8/model.bin",
    "TRANSLATION/opus-mt-id-en-ct2-int8/shared_vocabulary.json",
    "TRANSLATION/opus-mt-id-en-ct2-int8/source.spm",
    "TRANSLATION/opus-mt-id-en-ct2-int8/target.spm",
    "TRANSLATION/opus-mt-id-en-ct2-int8/tokenizer_config.json",
    "TRANSLATION/opus-mt-id-en-ct2-int8/vocab.json",
)
REQUIRED_RUNTIME_DIRS = ("clip", "trainers", "utils")


def _dotenv_value(name: str) -> str:
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.is_file():
        return ""
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == name:
            return value.strip().strip("'\"")
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_backbone_if_missing() -> None:
    target = MODEL_ROOT / "TC_CLIP" / "ViT-B-16.pt"
    if target.is_file() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(BACKBONE_URL, timeout=120) as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        if _sha256(temporary) != BACKBONE_SHA256:
            raise RuntimeError("CLIP backbone checksum mismatch")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_runtime() -> None:
    if all((RUNTIME_ROOT / directory).is_dir() for directory in REQUIRED_RUNTIME_DIRS):
        return
    raise RuntimeError(
        "TC-CLIP runtime tidak ditemukan di workspace/third_party/tc_clip/tc-clip. "
        "Runtime bersifat vendored — clone/pull ulang repository utama untuk memulihkannya."
    )


def _verify_installation() -> bool:
    missing = []
    for relative in REQUIRED_FILES:
        path = MODEL_ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    for directory in REQUIRED_RUNTIME_DIRS:
        path = RUNTIME_ROOT / directory
        if not path.is_dir():
            missing.append(str(path))
    if missing:
        for path in missing:
            logger.error("[Models] Missing or empty: %s", path)
        return False
    for relative, expected in (
        ("TC_CLIP/ViT-B-16.pt", BACKBONE_SHA256),
        ("TC_CLIP/zero_shot_k400_llm_tc_clip.pth", CHECKPOINT_SHA256),
    ):
        path = MODEL_ROOT / relative
        if _sha256(path) != expected:
            logger.error("[Models] SHA-256 mismatch: %s", path)
            return False
    logger.info("[Models] YOLO, TSM, TC-CLIP, translator, and runtime verified.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify local assets without downloading")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.verify:
        repo_id = os.environ.get("HF_MODEL_REPO_ID") or _dotenv_value("HF_MODEL_REPO_ID") or DEFAULT_REPO_ID
        revision = os.environ.get("HF_MODEL_REVISION") or _dotenv_value("HF_MODEL_REVISION") or "main"
        token = os.environ.get("HF_TOKEN") or _dotenv_value("HF_TOKEN") or None
        try:
            from huggingface_hub import snapshot_download

            logger.info("[Models] Downloading %s@%s to %s", repo_id, revision, MODEL_ROOT)
            snapshot_download(repo_id=repo_id, revision=revision, local_dir=MODEL_ROOT, token=token)
            _download_backbone_if_missing()
            _ensure_runtime()
        except Exception as exc:
            logger.error("[Models] Setup failed: %s", exc)
            logger.error("[Models] Check repository access/HF_TOKEN and retry.")
            return 1

    if not _verify_installation():
        return 1
    from app.tools.check_models import main as check_configured_backend

    return check_configured_backend()


if __name__ == "__main__":
    raise SystemExit(main())
