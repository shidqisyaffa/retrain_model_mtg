"""
Object Detection Service — provides swappable detection backends.

Backends:
    - "dummy"    : Simulated detections for pipeline testing (no model needed)
    - "yolo"     : YOLO-only detection (person/weapon bounding boxes)
    - "yolo_tsm" : Full violence pipeline — YOLO tracking + TSM action recognition
    - "grid_tsm" : TSM-Grid "Lazy Detection" — area-level TSM + triggered YOLO

Usage:
    detector = create_detector(backend="yolo_tsm", ...)
    detections = detector.detect(frame)
"""

import base64
import json
import logging
import math
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from app.core.config.config import settings
from app.services.detection.action_queue import KeyedLatestActionWorker
from app.services.detection.weapon_threat import weapon_threat_level
from app.services.detection.yolo_scheduler import YoloTrackProvider

logger = logging.getLogger(__name__)


def compose_detection_label(
    action: str | None,
    has_hug: bool,
    has_weapon: bool,
    tsm_classes: list[str],
    weapon_label: str | None = None,
) -> str:
    if has_weapon:
        return (weapon_label or "KNIFE").upper()

    action_value = str(action or "").strip()
    action_lower = action_value.lower()
    tsm_class_set = {str(c).lower() for c in tsm_classes}

    parts: list[str] = []
    if (
        action_value
        and action_lower in tsm_class_set
        and action_lower not in {"normal", "tracking"}
    ):
        parts.append(action_value.upper())
    if has_hug:
        parts.append("HUG")
    if parts:
        return " | ".join(parts)
    if action_lower in {"", "normal", "tracking"}:
        return "person"
    return action_value


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _resolve_ai_cpu_threads() -> int:
    raw = str(settings.AI_CPU_THREADS).strip().lower()
    if raw in {"", "auto"}:
        return max(1, _available_cpu_count() - 1)
    return max(1, int(raw))


def resolve_yolo_device(device: str) -> str:
    """Map a friendly device value to one accepted by Ultralytics."""
    normalized = (device or "auto").strip().lower()
    if normalized != "auto":
        if normalized.startswith("cuda") or normalized.isdecimal():
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return normalized

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_overlap(box_a, box_b) -> bool:
    """Axis-aligned overlap test — non-zero IoU area."""
    return (
        max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
        max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    ) > 0


def _bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU for two boxes in [x1, y1, x2, y2] format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


class StableTrackIdAssigner:
    """
    Assign stable logical IDs over unstable detector track IDs.

    When a raw ID disappears and comes back with a new value, this tries to
    reconnect it to the previous stable ID by IoU matching.
    """

    def __init__(self, max_age_frames: int = 30, iou_threshold: float = 0.35):
        self._max_age_frames = max_age_frames
        self._iou_threshold = iou_threshold
        self._next_stable_id = 1

        self._raw_to_stable: dict[int, int] = {}
        self._stable_last_seen: dict[int, int] = {}
        self._stable_last_box: dict[int, np.ndarray] = {}

    def assign(
        self,
        raw_track_id: int,
        box_xyxy: np.ndarray,
        frame_index: int,
        occupied_stable_ids: set[int],
    ) -> int:
        raw_track_id = int(raw_track_id)
        box = np.array(box_xyxy, dtype=float)

        if raw_track_id in self._raw_to_stable:
            stable_id = self._raw_to_stable[raw_track_id]
        else:
            stable_id = self._match_existing(box, frame_index, occupied_stable_ids)
            if stable_id is None:
                stable_id = self._next_stable_id
                self._next_stable_id += 1
            self._raw_to_stable[raw_track_id] = stable_id

        self._stable_last_seen[stable_id] = frame_index
        self._stable_last_box[stable_id] = box
        return stable_id

    def forget_stable_ids(self, stable_ids: set[int]) -> None:
        if not stable_ids:
            return

        for stable_id in stable_ids:
            self._stable_last_seen.pop(stable_id, None)
            self._stable_last_box.pop(stable_id, None)

        for raw_id, stable_id in list(self._raw_to_stable.items()):
            if stable_id in stable_ids:
                self._raw_to_stable.pop(raw_id, None)

    def _match_existing(
        self,
        box_xyxy: np.ndarray,
        frame_index: int,
        occupied_stable_ids: set[int],
    ) -> int | None:
        best_id: int | None = None
        best_iou = 0.0

        for stable_id, last_box in self._stable_last_box.items():
            if stable_id in occupied_stable_ids:
                continue

            last_seen = self._stable_last_seen.get(stable_id, -10**9)
            if frame_index - last_seen > self._max_age_frames:
                continue

            iou = _bbox_iou_xyxy(box_xyxy, last_box)
            if iou > self._iou_threshold and iou > best_iou:
                best_iou = iou
                best_id = stable_id

        return best_id


# ── Detection Data ────────────────────────────────────────


@dataclass
class Detection:
    """A single detected object in a frame."""

    x: int  # top-left x
    y: int  # top-left y
    w: int  # width
    h: int  # height
    confidence: float  # 0.0 - 1.0
    label: str = "person"

    # ── Extended fields (populated by ViolenceDetector) ──
    track_id: int | None = None
    action: str | None = None
    has_weapon: bool = False
    weapon_label: str | None = None
    has_hug: bool = False
    is_alert: bool = False
    threat_level: str | None = None  # "medium" | "high" | None (weapon always high)
    roi_type: str = "individual"
    member_ids: tuple[int, ...] = ()
    # Presentation-only grid identity/layout at inference time; bbox keeps overlap.
    grid_cell: dict[str, int] | None = None

    @property
    def cx(self) -> int:
        """Center x of bounding box."""
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        """Center y of bounding box."""
        return self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict:
        d = {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "cx": self.cx,
            "cy": self.cy,
            "confidence": round(self.confidence, 3),
            "label": self.label,
        }
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.action is not None:
            d["action"] = self.action
        d["has_weapon"] = self.has_weapon
        if self.weapon_label:
            d["weapon_label"] = self.weapon_label
        d["has_hug"] = self.has_hug
        d["is_alert"] = self.is_alert
        # Weapons use detection.weapon_threat_level; threat_level from class config otherwise
        d["threat_level"] = weapon_threat_level(self.threat_level) if self.has_weapon else self.threat_level
        d["roi_type"] = self.roi_type
        d["member_ids"] = list(self.member_ids)
        if self.roi_type == "region" and self.grid_cell is not None:
            d["grid_cell"] = dict(self.grid_cell)
        return d


@dataclass(frozen=True, slots=True)
class ActionDetectionBatch:
    """Action-only updates produced asynchronously for a source frame."""

    source_frame_id: int
    detected_at: float
    detections: tuple[Detection, ...]


# ── Abstract Base ─────────────────────────────────────────


class BaseDetector(ABC):
    """
    Abstract detector interface.

    All detection backends (dummy, YOLO, etc.) must implement this.
    The only method to implement is `detect(frame) -> list[Detection]`.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.

        :param frame: BGR numpy array (H, W, 3) from OpenCV.
        :return: List of Detection objects found in the frame.
        """
        ...

    @abstractmethod
    def get_info(self) -> dict:
        """Return metadata about this detector backend."""
        ...

    def set_feature_flags(
        self,
        *,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
    ) -> None:
        """Runtime AI feature toggles. Subclasses may override."""
        return None

    def detect_with_context(
        self,
        frame: np.ndarray,
        *,
        frame_id: int | None = None,
        captured_at: float | None = None,
        include_actions: bool = False,
    ) -> list[Detection]:
        """Run detection with optional stream-frame identity."""
        del frame_id, captured_at, include_actions
        return self.detect(frame)

    def drain_action_updates(self) -> list[ActionDetectionBatch]:
        """Return completed asynchronous action updates, if supported."""
        return []

    def close(self) -> None:
        """Release optional detector-owned background workers."""
        return None


# ── Dummy Detector (for testing without models) ──────────


class DummyDetector(BaseDetector):
    """
    Simulates a moving person detection for testing the tracking pipeline.

    The dummy bounding box follows a figure-8 pattern across the frame,
    simulating a person walking around. This allows full PID + drone
    tracking testing without needing the real YOLO model.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        bbox_width: int = 120,
        bbox_height: int = 240,
    ):
        self._fw = frame_width
        self._fh = frame_height
        self._bw = bbox_width
        self._bh = bbox_height
        self._start_time = time.time()

        logger.info(
            "DummyDetector initialized (frame=%dx%d, bbox=%dx%d)",
            frame_width, frame_height, bbox_width, bbox_height,
        )

    def detect(self, frame: np.ndarray | None = None) -> list[Detection]:
        """
        Generate a single fake detection that moves in a figure-8 pattern.

        The frame parameter is accepted but ignored — detection is purely
        time-based to simulate movement.
        """
        t = time.time() - self._start_time
        speed = 0.3  # cycles per second

        # Figure-8 parametric curve (Lissajous with 2:1 frequency ratio)
        # x oscillates at 1x speed, y at 2x speed
        norm_x = math.sin(2 * math.pi * speed * t)
        norm_y = math.sin(2 * math.pi * speed * t * 2) * 0.6

        # Map normalized [-1, 1] to frame coordinates with padding
        pad_x = self._bw
        pad_y = self._bh
        cx = int((norm_x + 1) / 2 * (self._fw - 2 * pad_x) + pad_x)
        cy = int((norm_y + 1) / 2 * (self._fh - 2 * pad_y) + pad_y)

        # Top-left corner
        x = max(0, cx - self._bw // 2)
        y = max(0, cy - self._bh // 2)

        # Vary bbox size slightly to simulate distance changes
        scale = 1.0 + 0.3 * math.sin(2 * math.pi * speed * t * 0.5)
        w = int(self._bw * scale)
        h = int(self._bh * scale)

        # Clamp to frame
        w = min(w, self._fw - x)
        h = min(h, self._fh - y)

        return [
            Detection(
                x=x, y=y, w=w, h=h,
                confidence=0.85 + 0.1 * math.sin(t),
                label="FIGHT",
                action="fight",
                is_alert=True,
            )
        ]

    def get_info(self) -> dict:
        return {
            "backend": "dummy",
            "model": None,
            "description": "Simulated figure-8 moving target for testing",
            "frame_size": f"{self._fw}x{self._fh}",
        }


# ── Online Detector (vLLM / Vision AI Spatial Understanding) ──

GENERALIZED_BEHAVIOR_CATALOG = """\
Kamu adalah sistem AI keamanan vision real-time dan detektor objek presisi. Pindai SELURUH area spasial gambar untuk mendeteksi orang, objek, dan aktivitas tidak wajar.

INSTRUKSI DETEKSI Bounding Box 2D (SANGAT PENTING):
1. Untuk SETIAP anomali/perilaku yang terdeteksi, `box_2d` WAJIB mengurung SELURUH TUBUSH MANUSIA/SUBJEK ORANG (dari kepala hingga kaki/tangan) atau SELURUH OBJEK UTAMA yang terlibat. JANGAN HANYA mengurung bagian kecil seperti tangan/pisau saja.
2. Bertindaklah sangat presisi seperti detektor objek profesional. Sertakan seluruh proporsi tubuh subjek.
3. Gunakan koordinat [ymin, xmin, ymax, xmax] ter-skala 0 hingga 1000.

## KAMUS ANOMALI KEAMANAN TERGENERALISASI
1. Penyusupan: Orang memasuki area terlarang/steril; memanjat, melompati, atau menerobos pembatas/pagar/barikade; mengendap-endap/merayap/bersembunyi.
2. Perilaku Mencurigakan: Orang berlari secara cepat, tergesa-gesa, panik, atau mencurigakan; mondar-mandir atau berhenti terlalu lama; bergerak melawan arus; berkumpul secara mendadak; menutupi wajah; membawa barang besar/terselubung mencurigakan; mengarahkan/mengancam dengan benda berbahaya/senjata.
3. Barang Mencurigakan: Tas, koper, kardus, atau paket ditinggalkan tanpa pemilik (unattended item); benda asing misterius di area pengamanan.
4. Kendaraan: Kendaraan melaju kecepatan tinggi/menerobos barikade; melaju melawan arah; parkir/berhenti di area terlarang; tanpa pelat nomor.
5. Senjata & Benda Berbahaya: Pistol, senapan, pisau, golok, atau bahan peledak.
6. Perimeter & Infrastruktur: Pembatas/pagar/barikade dirusak, dipotong, atau dibuka secara paksa; pelanggaran garis perimeter.
7. Keselamatan & Darurat: Nyala api, asap tebal, ledakan, atau percikan api.
8. Drone / Objek Udara: Drone atau UAV melayang/terbang di area vital/acara.
9. Sabotase Kamera: Lensa kamera tertutup, diputar, dihalangi, atau dirusak.
10. Pos & Personel Jaga: Pos jaga kosong tanpa petugas; petugas tertidur atau terdistraksi HP berlebihan.
11. Pergerakan Massa: Kerumunan massa meningkat mendadak; massa merangsek/berdesakan ekstrem/melempar benda.
12. Anomali Umum: Pergerakan janggal jam sensitif; objek/orang menetap secara aneh.

JIKA KONDISI AMAN ATAU NORMAL, KEMBALIKAN anomalies_detected SEBAGAI ARRAY KOSONG [].

Output WAJIB berupa JSON valid dengan format:
{
  "anomalies_detected": [
    {
      "kategori": "Perilaku Mencurigakan",
      "aktivitas": "Orang berlari secara tergesa-gesa dan mencurigakan",
      "tingkat_prioritas": "Tinggi",
      "box_2d": [ymin, xmin, ymax, xmax],
      "confidence_score": "High"
    }
  ]
}
Catatan: box_2d adalah koordinat [ymin, xmin, ymax, xmax] ter-skala 0-1000.
"""


class AnomalyItem(BaseModel):
    """Single detected anomaly matched 1:1 with camera-genai schema."""
    kategori: str
    aktivitas: str
    tingkat_prioritas: str
    box_2d: list[int]
    confidence_score: str


class GeminiAnomalyOutput(BaseModel):
    """Guaranteed structured JSON response_schema for Gemini Vision API."""
    anomalies_detected: list[AnomalyItem]


class OnlineDetector(BaseDetector):
    """
    Real-time Online AI Provider Detector using vLLM / Vision-Language Model.

    Executes asynchronous background vision inference to prevent video stream stuttering.
    Converts 0-1000 normalized spatial bounding boxes to pixel-space Detections.
    """

    def __init__(self):
        self._last_inference_ts: float = 0.0
        self._last_detections: list[Detection] = []
        self._is_processing: bool = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OnlineAIWorker")
        logger.info("OnlineDetector initialized (vLLM / Vision AI real-time engine ready)")

    def detect(self, frame: np.ndarray | None = None) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        now = time.time()
        sampling_interval = getattr(settings, "ONLINE_AI_SAMPLING_INTERVAL_MS", 500) / 1000.0

        if not self._is_processing and (now - self._last_inference_ts >= sampling_interval):
            self._is_processing = True
            self._last_inference_ts = now
            frame_copy = frame.copy()
            h, w = frame.shape[:2]
            self._executor.submit(self._async_vlm_inference, frame_copy, w, h)

        return self._last_detections

    def _async_vlm_inference(self, frame: np.ndarray, img_w: int, img_h: int) -> None:
        try:
            import cv2
            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not success:
                self._is_processing = False
                return

            jpeg_bytes = buffer.tobytes()
            anomalies = []

            # Option A: Gemini Vision SDK (if API key available)
            api_key = getattr(settings, "GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            if api_key:
                try:
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=api_key)
                    model_name = getattr(settings, "GEMINI_SPATIAL_MODEL", "gemini-3.1-flash-lite")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                            "Analisis frame CCTV/Drone ini. Deteksi semua anomali spasial dan kembalikan bounding box [ymin, xmin, ymax, xmax] ter-skala 0-1000 mengurung SELURUH SUBJEK ORANG.",
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=GENERALIZED_BEHAVIOR_CATALOG,
                            response_mime_type="application/json",
                            response_schema=GeminiAnomalyOutput,
                            temperature=0.1,
                        ),
                    )
                    if response and response.text:
                        parsed = json.loads(response.text)
                        anomalies = parsed.get("anomalies_detected", [])
                except Exception as exc:
                    logger.debug("Gemini Vision call failed: %s", exc)

            # Option B: vLLM / OpenAI Compatible Endpoint (fallback)
            if not anomalies and not api_key:
                b64_img = base64.b64encode(jpeg_bytes).decode("utf-8")
                api_base = getattr(settings, "VLLM_API_BASE", "http://localhost:8080/v1").rstrip("/")
                model_name = getattr(settings, "VLLM_MODEL_NAME", "qwen2-vl-7b-instruct")

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": GENERALIZED_BEHAVIOR_CATALOG},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analisis frame CCTV/Drone ini untuk mendeteksi anomali spasial."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            ],
                        },
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                }

                req = urllib.request.Request(
                    f"{api_base}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(req, timeout=3.0) as resp:
                        if resp.status == 200:
                            res_body = json.loads(resp.read().decode("utf-8"))
                            content_str = res_body.get("choices", [{}])[0].get("message", {}).get("content", "")
                            parsed = json.loads(content_str)
                            anomalies = parsed.get("anomalies_detected", [])
                except Exception:
                    pass

            new_detections = []
            if anomalies:
                for idx, item in enumerate(anomalies):
                    box = item.get("box_2d", [])
                    if len(box) == 4:
                        ymin, xmin, ymax, xmax = box
                        x1 = int(max(0, min(1000, xmin)) / 1000.0 * img_w)
                        y1 = int(max(0, min(1000, ymin)) / 1000.0 * img_h)
                        x2 = int(max(0, min(1000, xmax)) / 1000.0 * img_w)
                        y2 = int(max(0, min(1000, ymax)) / 1000.0 * img_h)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)

                        aktivitas = item.get("aktivitas", "Anomali terdeteksi")
                        prioritas = item.get("tingkat_prioritas", "Tinggi")
                        conf_str = str(item.get("confidence_score", "High")).capitalize()
                        conf = 0.9 if conf_str == "High" else (0.7 if conf_str == "Medium" else 0.5)

                        new_detections.append(
                            Detection(
                                x=x1,
                                y=y1,
                                w=bw,
                                h=bh,
                                confidence=conf,
                                label=aktivitas,
                                action=item.get("kategori", aktivitas),
                                track_id=1 + idx,
                                is_alert=(prioritas in ("Tinggi", "Sangat Tinggi")),
                            )
                        )
            self._last_detections = new_detections

        except Exception:
            pass
        finally:
            self._is_processing = False

    def get_info(self) -> dict:
        return {
            "backend": "online",
            "mode": "online",
            "available": True,
            "status": "active",
            "loaded": True,
        }


# ── YOLO-only Detector ───────────────────────────────────


class YOLODetector(BaseDetector):
    """
    YOLO-based detector (bounding boxes only, no action recognition).

    To use:
    1. Set detection.backend: "yolo" in config.yaml
    2. Set detection.model_path to your trained model file
    3. Install ultralytics: pip install ultralytics
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._device = device
        self._model = None

        self._load_model()

    def _load_model(self):
        """Load the YOLO model from the specified path."""
        try:
            from app.services.detection.yolo_backend import load_yolo

            runtime_device = resolve_yolo_device(self._device)
            load_device = "cuda:0" if runtime_device == "0" else runtime_device
            self._model, self._model_backend = load_yolo(
                self._model_path,
                device=load_device,
                imgsz=settings.DETECTION_YOLO_IMGSZ,
            )
            logger.info(
                "YOLO model loaded from '%s' (device=%s, engine=%s, conf=%.2f)",
                self._model_path, runtime_device, self._model_backend, self._conf_threshold,
            )
        except ImportError:
            logger.error(
                "ultralytics not installed. Run: pip install ultralytics"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load YOLO model '%s': %s", self._model_path, exc)
            raise

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO inference on a frame and return Detection objects."""
        if self._model is None:
            logger.warning("YOLO model not loaded, returning empty detections")
            return []

        runtime_device = resolve_yolo_device(self._device)
        try:
            results = self._model.predict(
                source=frame,
                conf=self._conf_threshold,
                device=runtime_device,
                verbose=False,
            )
        except Exception as exc:
            if self._model_backend != "tensorrt":
                raise
            logger.warning("YOLO-only TensorRT failed; retrying with PyTorch: %s", exc)
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            self._model_backend = "pytorch"
            results = self._model.predict(
                source=frame,
                conf=self._conf_threshold,
                device=runtime_device,
                verbose=False,
            )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, f"class_{cls_id}")
                if str(label).strip().lower() == "weapon":
                    label = "GUN"

                detections.append(Detection(
                    x=int(x1),
                    y=int(y1),
                    w=int(x2 - x1),
                    h=int(y2 - y1),
                    confidence=conf,
                    label=label,
                ))

        return detections

    def get_info(self) -> dict:
        return {
            "backend": "yolo",
            "model": self._model_path,
            "confidence_threshold": self._conf_threshold,
            "device": resolve_yolo_device(self._device),
            "inference_engine": getattr(self, "_model_backend", "pytorch"),
            "loaded": self._model is not None,
        }


# ── Violence Detector (YOLO + TSM) ───────────────────────


class ViolenceDetector(BaseDetector):
    """
    Full violence detection pipeline: YOLO tracking + TSM action recognition.

    Flow per frame:
        1. YOLO.track()  → person bounding boxes with persistent IDs (ByteTrack)
        2. YOLO.predict() → weapon bounding boxes
        3. For each person crop → resize 224×224, normalize, push to temporal buffer
        4. Every `inference_interval` frames → batch TSM inference on full buffers
        5. Majority-vote smoothing over last 5 predictions per track
        6. Weapon overlap check per person
        7. Return Detection list with action labels and alert flags
    """

    # YOLO class indices — match the trained model
    # (`models/YOLO/best.pt`, same as workspace/experiments/cctv_testing_script):
    #   0 = HUG     (interaction, amber tone in UI)
    #   1 = KNIFE   (red alert)
    #   2 = PERSON  (subject tracked for TSM action recognition)
    #   3 = WEAPON  (displayed as GUN, red alert)
    CLASS_HUG = 0
    CLASS_KNIFE = 1
    CLASS_PERSON = 2
    CLASS_WEAPON = 3

    def __init__(
        self,
        yolo_model_path: str,
        tsm_model_path: str,
        tsm_classes: list[str] | None = None,
        alert_classes: list[str] | None = None,
        num_segments: int = 16,
        inference_interval: int = 10,
        batch_size: int = 4,
        conf_person: float = 0.3,
        conf_weapon: float = 0.25,
        device: str = "auto",
        contrast_alpha: float = 1.2,
        contrast_beta: float = 10.0,
        yolo_imgsz: int = 480,
        yolo_half: bool = True,
        tsm_compile: bool = True,
        grid_violence_threshold: float = 0.70,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
    ):
        self._yolo_path = yolo_model_path
        self._tsm_path = tsm_model_path
        # Contrast preprocessing — matches workspace/experiments/cctv_testing_script.
        # Set α=1.0 + β=0 to disable.
        self._contrast_alpha = float(contrast_alpha)
        self._contrast_beta = float(contrast_beta)

        # Binary classifier from the validated POC: index 0 = "normal",
        # index 1 = "violence". `alert_classes` is the subset that flips
        # the bbox red and raises an alarm — defaults to {"violence"}.
        self._tsm_classes = tsm_classes or ["normal", "violence"]
        # dict {name: threat_level} — "high" is default for legacy callers.
        # Names are stored casefold-normalized.
        self._alert_classes: dict[str, str] = {
            c.strip().casefold(): "high"
            for c in (alert_classes or ["violence"])
        }

        self._num_segments = num_segments
        self._inference_interval = inference_interval
        self._batch_size = batch_size
        self._conf_person = conf_person
        self._conf_weapon = conf_weapon
        self._device_str = device
        self._violence_threshold = grid_violence_threshold
        self._action_recognition_enabled = bool(action_recognition_enabled)
        self._object_detection_enabled = bool(object_detection_enabled)

        # Performance knobs — fp16 for YOLO is roughly 2× faster on CUDA
        # and free accuracy on detection-class models. Auto-disabled on
        # CPU / unsupported MPS during `_load_models()`. `yolo_imgsz`
        # controls YOLO's internal letterbox — 480 is ~35% faster than
        # 640 with negligible miss rate on close-range subjects.
        self._yolo_imgsz = int(yolo_imgsz)
        self._yolo_half = bool(yolo_half)
        self._tsm_compile = bool(tsm_compile)

        # Models (loaded lazily in _load_models)
        self._yolo = None
        self._tsm = None
        self._device = None

        # Per-track temporal state — kept minimal to mirror the POC
        # exactly. The POC never expires tracks, but in long-running
        # production we'd leak memory; track ids that haven't appeared
        # for `_MAX_AGE` frames are pruned in a small janitor pass.
        self._buffers: dict[int, deque] = {}       # track_id → deque of crop tensors
        self._actions: dict[int, str] = {}          # track_id → current action label
        self._history: dict[int, deque] = {}        # track_id → deque of recent predictions
        self._violence_scores: dict[int, deque] = {}       # track_id → deque of recent raw violence probabilities (timestamp, score)
        self._violence_start_times: dict[int, float] = {}  # track_id → time.time() when continuous violence started
        self._violence_last_seen: dict[int, float] = {}    # track_id → time.time() when violence was last seen
        self._last_seen: dict[int, int] = {}        # track_id → frame_count last seen
        self._frame_count: int = 0
        self._MAX_AGE = 60

        self._load_models()

    def set_feature_flags(
        self,
        *,
        action_recognition_enabled: bool = True,
        object_detection_enabled: bool = True,
    ) -> None:
        self._action_recognition_enabled = bool(action_recognition_enabled)
        self._object_detection_enabled = bool(object_detection_enabled)

    def _load_models(self):
        """Load YOLO and TSM models with explicit GPU prioritization."""
        import torch
        from ultralytics import YOLO

        cpu_threads = _resolve_ai_cpu_threads()
        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
            logger.info("ViolenceDetector: torch CPU threads=%d", cpu_threads)
        except RuntimeError as exc:
            logger.info("ViolenceDetector: torch thread config already locked: %s", exc)

        # Device selection — CUDA > MPS (Apple Silicon) > CPU
        if self._device_str == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device_str)

        logger.info(
            "ViolenceDetector: using compute device → %s",
            self._device,
        )

        # YOLO — Ultralytics auto-handles device per inference call, but
        # we also try to .to(device) so the model weights live on the
        # right device (MPS / CUDA) instead of bouncing CPU → GPU each
        # call. Some Ultralytics versions reject .to() — best-effort.
        self._yolo = YOLO(self._yolo_path)
        try:
            self._yolo.to(self._device)
            logger.info("ViolenceDetector: YOLO moved to %s", self._device)
        except Exception as exc:
            logger.warning(
                "ViolenceDetector: YOLO .to(%s) skipped — Ultralytics will "
                "set device per-call instead (%s)",
                self._device, exc,
            )

        # fp16 is only safe on CUDA. MPS half-precision support is
        # incomplete in Ultralytics → silently fall back. CPU half is
        # always slower than fp32 in PyTorch.
        if self._yolo_half and self._device.type != "cuda":
            logger.info(
                "ViolenceDetector: yolo_half=True ignored on device=%s "
                "(only CUDA gives a real fp16 win)", self._device.type,
            )
            self._yolo_half = False

        logger.info(
            "ViolenceDetector: YOLO loaded from '%s' (imgsz=%d, half=%s)",
            self._yolo_path, self._yolo_imgsz, self._yolo_half,
        )

        # TSM — TSN is built on MobileNetV2 + TemporalShift modules.
        # The `app/tsm_ops/` package (ported from the validated POC at
        # workspace/experiments/cctv_testing_script/ops/) ships to Ray actor
        # subprocesses via runtime_env working_dir.
        from app.tsm_ops.models import TSN

        self._tsm = TSN(
            num_class=len(self._tsm_classes),
            num_segments=self._num_segments,
        )

        state_dict = torch.load(self._tsm_path, map_location=self._device, weights_only=False)
        missing, unexpected = self._tsm.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("TSM missing keys: %s", missing)
        if unexpected:
            logger.warning("TSM unexpected keys: %s", unexpected)

        self._tsm.to(self._device).eval()

        # `torch.compile` fuses TSM kernels — ~1.3-1.8× speedup at the
        # cost of a one-time ~10s warm-up on first call. CUDA + recent
        # PyTorch only; on CPU/MPS the inductor backend either silently
        # falls back to eager or raises, so we wrap defensively and
        # never let compile failures take the detector down.
        if self._tsm_compile and self._device.type == "cuda":
            try:
                self._tsm = torch.compile(self._tsm, mode="reduce-overhead")
                logger.info("ViolenceDetector: TSM torch.compile() enabled")
            except Exception as exc:
                logger.warning(
                    "ViolenceDetector: torch.compile(TSM) failed, "
                    "running eager: %s", exc,
                )
                self._tsm_compile = False
        elif self._tsm_compile:
            logger.info(
                "ViolenceDetector: tsm_compile=True ignored on device=%s "
                "(only CUDA inductor is reliably faster)", self._device.type,
            )
            self._tsm_compile = False

        logger.info(
            "ViolenceDetector: TSM loaded from '%s' (%d classes, %d segments, device=%s)",
            self._tsm_path, len(self._tsm_classes), self._num_segments, self._device,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run the violence detection pipeline on a single BGR frame.

        Implementation mirrors `workspace/experiments/cctv_testing_script/surveilance_inference_pt.py`
        step-for-step. Any divergence (EMA smoothing, padded crops, area
        filters, geometric validators) was historically responsible for
        TSM never firing in production, so we keep this method honest and
        boring on purpose.
        """
        import cv2
        import torch
        import torchvision.transforms.functional as F

        if frame is None or self._yolo is None:
            return []
        if not self._action_recognition_enabled and not self._object_detection_enabled:
            return []

        self._frame_count += 1

        # 0. Contrast adjust — matches POC training preprocessing.
        if self._contrast_alpha != 1.0 or self._contrast_beta != 0.0:
            frame = cv2.convertScaleAbs(
                frame, alpha=self._contrast_alpha, beta=self._contrast_beta,
            )

        h, w = frame.shape[:2]

        # 1. YOLO — ONE combined track() over all three classes. Tracking
        #    the object classes too is essentially free (ByteTrack just
        #    sees them as extra tracks; we never look at their ids) and
        #    halves the YOLO forward-pass cost vs. running track() and
        #    predict() back-to-back. The `conf` floor is the lower of the
        #    two thresholds; per-class filtering is done in Python after.
        device_str = str(self._device)
        yolo_classes = [self.CLASS_PERSON]
        if self._object_detection_enabled:
            yolo_classes.extend([self.CLASS_HUG, self.CLASS_KNIFE, self.CLASS_WEAPON])
        yolo_conf = (
            min(self._conf_person, self._conf_weapon)
            if self._object_detection_enabled else self._conf_person
        )
        results = self._yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=yolo_classes,
            conf=yolo_conf,
            imgsz=self._yolo_imgsz,
            half=self._yolo_half,
            device=device_str,
            verbose=False,
        )
        person_results = results

        boxes_result = results[0].boxes
        all_boxes_np = None
        all_cls_np = None
        all_conf_np = None
        if boxes_result is not None and len(boxes_result) > 0:
            all_boxes_np = boxes_result.xyxy.cpu().numpy()
            all_cls_np = boxes_result.cls.cpu().numpy().astype(int)
            if boxes_result.conf is not None:
                all_conf_np = boxes_result.conf.cpu().numpy()

        hug_boxes: list = []
        knife_boxes: list = []
        gun_boxes: list = []
        if self._object_detection_enabled and all_boxes_np is not None:
            for i, (box, cls) in enumerate(zip(all_boxes_np, all_cls_np)):
                cf = float(all_conf_np[i]) if all_conf_np is not None else 1.0
                if cf < self._conf_weapon:
                    continue
                if cls == self.CLASS_HUG:
                    hug_boxes.append(box)
                elif cls == self.CLASS_KNIFE:
                    knife_boxes.append(box)
                elif cls == self.CLASS_WEAPON:
                    gun_boxes.append(box)

        # 2. Prep full-frame tensor for cropping inside the TSM loop.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        ).to(self._device)

        detections: list[Detection] = []
        tracks_to_infer: list[int] = []

        # 3. Per-person tracking + crop buffering.
        if (
            all_boxes_np is not None
            and person_results[0].boxes is not None
            and person_results[0].boxes.id is not None
        ):
            all_ids_np = person_results[0].boxes.id.cpu().numpy().astype(int)
            person_mask = (all_cls_np == self.CLASS_PERSON)
            if all_conf_np is not None:
                person_mask = person_mask & (all_conf_np >= self._conf_person)

            boxes = all_boxes_np[person_mask]
            ids = all_ids_np[person_mask]
            confs = all_conf_np[person_mask] if all_conf_np is not None else None

            for idx, (box, tid_arr) in enumerate(zip(boxes, ids)):
                tid = int(tid_arr)

                # Raw bbox, clamped to frame — no EMA, no padding.
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop, resize, normalize → push into the temporal buffer.
                crop = frame_tensor[:, y1:y2, x1:x2]
                if crop.numel() > 0:
                    crop = F.resize(crop, [224, 224], antialias=True)
                    crop = F.normalize(
                        crop,
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    )

                    if tid not in self._buffers:
                        self._buffers[tid] = deque(maxlen=self._num_segments)
                        self._history[tid] = deque(maxlen=5)

                    self._buffers[tid].append(crop)

                    if (
                        self._action_recognition_enabled
                        and self._tsm is not None
                        and
                        len(self._buffers[tid]) == self._num_segments
                        and self._frame_count % self._inference_interval == 0
                    ):
                        tracks_to_infer.append(tid)

                self._last_seen[tid] = self._frame_count

                # Object overlap — has_hug colours the box amber, has_knife
                # turns it red and raises an alert.
                person_box = [x1, y1, x2, y2]
                has_hug = (
                    self._object_detection_enabled
                    and any(_has_overlap(person_box, hb) for hb in hug_boxes)
                )
                has_knife = (
                    self._object_detection_enabled
                    and any(_has_overlap(person_box, kb) for kb in knife_boxes)
                )
                has_gun = (
                    self._object_detection_enabled
                    and any(_has_overlap(person_box, gb) for gb in gun_boxes)
                )
                weapon_label = "GUN" if has_gun else ("KNIFE" if has_knife else None)

                current_action = (
                    self._actions.get(tid, "normal")
                    if self._action_recognition_enabled else None
                )

                action_name = (current_action or "").strip().casefold()
                is_alert = (
                    (self._action_recognition_enabled and action_name in self._alert_classes)
                    or has_knife
                    or has_gun
                )
                # Resolve per-class threat level; weapon always high
                threat_level: str | None = None
                if has_knife or has_gun:
                    threat_level = "high"
                elif self._action_recognition_enabled and action_name in self._alert_classes:
                    threat_level = self._alert_classes[action_name]

                label = compose_detection_label(
                    current_action,
                    has_hug,
                    has_knife or has_gun,
                    self._tsm_classes,
                    weapon_label,
                )

                detections.append(Detection(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    confidence=float(confs[idx]) if confs is not None else 0.0,
                    label=label,
                    track_id=tid,
                    action=current_action,
                    has_weapon=has_knife or has_gun,
                    weapon_label=weapon_label,
                    has_hug=has_hug,
                    is_alert=is_alert,
                    threat_level=threat_level,
                ))

        # 4. Batch TSM inference — `torch.stack` so the input is 5-D
        #    `(B, S, C, H, W)` which is what `TSN.forward` expects. The
        #    earlier prod code used `torch.cat` (4-D) and silently crashed.
        if self._action_recognition_enabled and tracks_to_infer and self._tsm is not None:
            with torch.inference_mode():
                for i in range(0, len(tracks_to_infer), self._batch_size):
                    sub = tracks_to_infer[i: i + self._batch_size]
                    batch = torch.stack([
                        torch.stack(list(self._buffers[tid])) for tid in sub
                    ]).to(self._device)

                    outputs = self._tsm(batch)  # (B, num_classes)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                    for j, tid in enumerate(sub):
                        pred_class_idx = preds[j]
                        pred_label = self._tsm_classes[pred_class_idx]

                        # Find the raw probability of the violence class
                        violence_idx = self._tsm_classes.index("violence") if "violence" in self._tsm_classes else 1
                        if violence_idx >= probs.shape[1]:
                            violence_idx = probs.shape[1] - 1
                        violence_prob = float(probs[j, violence_idx])

                        # 1. Score Averaging: Add to sliding window
                        if tid not in self._violence_scores:
                            self._violence_scores[tid] = deque(maxlen=30)
                        self._violence_scores[tid].append((time.time(), violence_prob))

                        # Prune scores older than 4.0 seconds
                        now = time.time()
                        while self._violence_scores[tid] and now - self._violence_scores[tid][0][0] > 4.0:
                            self._violence_scores[tid].popleft()

                        # Calculate average score
                        if self._violence_scores[tid]:
                            avg_violence_prob = sum(score for ts, score in self._violence_scores[tid]) / len(self._violence_scores[tid])
                        else:
                            avg_violence_prob = 0.0

                        # 2. Hysteresis (Dual-Threshold)
                        trigger_thresh = self._violence_threshold  # e.g. 0.70
                        maint_thresh = max(0.4, trigger_thresh - 0.15)  # e.g. 0.55

                        is_currently_violent = tid in self._violence_start_times
                        if is_currently_violent:
                            is_violence_signal = avg_violence_prob >= maint_thresh
                        else:
                            is_violence_signal = avg_violence_prob >= trigger_thresh

                        # 3. Grace Period continuous validation
                        if is_violence_signal:
                            if tid not in self._violence_start_times:
                                self._violence_start_times[tid] = now
                            self._violence_last_seen[tid] = now
                        else:
                            if tid in self._violence_start_times:
                                # If normal, check if grace period (1.5 seconds) has expired
                                if now - self._violence_last_seen.get(tid, now) > 1.5:
                                    self._violence_start_times.pop(tid, None)
                                    self._violence_last_seen.pop(tid, None)

                        # Determine final action label
                        if tid in self._violence_start_times:
                            elapsed = now - self._violence_start_times[tid]
                            # Only escalate to "violence" after 3 seconds
                            if elapsed >= 3.0:
                                self._actions[tid] = "violence"
                            else:
                                self._actions[tid] = "normal" if "normal" in self._tsm_classes else self._tsm_classes[0]
                        else:
                            self._actions[tid] = "normal" if "normal" in self._tsm_classes else self._tsm_classes[0]

                        # Keep self._history updated for backward compatibility
                        self._history[tid].append(self._actions[tid])

            # Refresh labels on freshly-classified detections.
            for det in detections:
                tid = det.track_id
                if tid is None or tid not in self._actions:
                    continue
                det.action = self._actions[tid]

                det.label = compose_detection_label(
                    det.action,
                    det.has_hug,
                    det.has_weapon,
                    self._tsm_classes,
                    det.weapon_label,
                )
                det.is_alert = (
                    (det.action is not None and det.action.strip().casefold() in self._alert_classes)
                    or (self._object_detection_enabled and det.has_weapon)
                )
                if det.has_weapon:
                    det.threat_level = "high"
                elif det.action is not None and det.action.strip().casefold() in self._alert_classes:
                    det.threat_level = self._alert_classes[det.action.strip().casefold()]
                else:
                    det.threat_level = None

        # 5. Stale-track janitor — keeps memory bounded over long runs.
        #    The POC script never expires tracks, but a 24/7 surveillance
        #    actor cannot afford that.
        inactive = [
            tid for tid, last_f in self._last_seen.items()
            if self._frame_count - last_f > self._MAX_AGE
        ]
        for tid in inactive:
            self._buffers.pop(tid, None)
            self._actions.pop(tid, None)
            self._history.pop(tid, None)
            self._violence_scores.pop(tid, None)
            self._violence_start_times.pop(tid, None)
            self._violence_last_seen.pop(tid, None)
            self._last_seen.pop(tid, None)

        return detections

    def get_info(self) -> dict:
        return {
            "backend": "yolo_tsm",
            "yolo_model": self._yolo_path,
            "tsm_model": self._tsm_path,
            "tsm_classes": self._tsm_classes,
            "alert_classes": [
                {"name": name, "threat_level": level}
                for name, level in self._alert_classes.items()
            ],
            "num_segments": self._num_segments,
            "inference_interval": self._inference_interval,
            "device": str(self._device),
            "yolo_imgsz": self._yolo_imgsz,
            "yolo_half": self._yolo_half,
            "tsm_compile": self._tsm_compile,
            "loaded": self._yolo is not None and self._tsm is not None,
            "active_tracks": len(self._buffers),
            "action_recognition_enabled": self._action_recognition_enabled,
            "object_detection_enabled": self._object_detection_enabled,
        }

    def set_alert_classes_with_levels(self, alert_dict: dict[str, str]) -> None:
        """Update alert classes from a {name: threat_level} dict (preferred API)."""
        self._alert_classes = {
            k.strip().casefold(): v.strip().lower()
            for k, v in alert_dict.items()
            if k.strip()
        }

    def set_alert_classes(self, classes: set[str]) -> None:
        """Backward-compat: accept a plain set of class names (all treated as high)."""
        self._alert_classes = {
            c.strip().casefold(): "high"
            for c in classes
            if c.strip()
        }


# ── Factory ───────────────────────────────────────────────


def create_detector(
    backend: str = "dummy",
    model_path: str = "",
    confidence_threshold: float = 0.5,
    frame_width: int = 1280,
    frame_height: int = 720,
    ai_mode: str = "offline",
    # ── ViolenceDetector / GridViolenceDetector params ──
    tsm_model_path: str = "",
    tsm_classes: list[str] | None = None,
    alert_classes: list[str] | None = None,
    num_segments: int = 16,
    inference_interval: int = 10,
    batch_size: int = 4,
    conf_person: float = 0.3,
    conf_weapon: float = 0.25,
    contrast_alpha: float = 1.2,
    contrast_beta: float = 10.0,
    yolo_imgsz: int = 480,
    yolo_half: bool = True,
    tsm_compile: bool = True,
    tcclip_runtime_root: str = "",
    tcclip_checkpoint_path: str = "",
    tcclip_cache_dir: str = "",
    tcclip_normal_prompt: str = "",
    tcclip_fight_prompt: str = "",
    tcclip_prompt_classes: list[dict[str, object]] | None = None,
    tcclip_fight_threshold: float = 0.70,
    tcclip_num_frames: int = 16,
    tcclip_temporal_stride: int = 2,
    tcclip_target_fps: float = 10.0,
    tcclip_batch_size: int = 1,
    tcclip_inference_interval_sec: float = 0.8,
    tcclip_max_rois_per_job: int = 2,
    tcclip_max_result_age_sec: float = 2.0,
    tcclip_action_ttl_sec: float = 2.0,
    tcclip_action_scheduler: KeyedLatestActionWorker[
        str,
        Callable[[], object],
        object,
    ]
    | None = None,
    yolo_scheduler: YoloTrackProvider | None = None,
    yolo_stream_id: str | None = None,
    # ── GridViolenceDetector-specific params ──
    grid_tsm_interval: int = 15,
    grid_tsm_segments: int = 8,
    grid_violence_threshold: float = 0.60,
    action_recognition_enabled: bool = True,
    object_detection_enabled: bool = True,
    tcclip_alerts_threshold: float | None = None,
) -> BaseDetector:
    """
    Factory function to create the appropriate detector backend.
    """
    if str(ai_mode).strip().lower() == "online":
        return OnlineDetector()

    if backend == "dummy":
        return DummyDetector(frame_width=frame_width, frame_height=frame_height)

    elif backend == "yolo":
        if not model_path:
            raise ValueError(
                "detection.model_path must be set in config.yaml when using 'yolo' backend"
            )
        return YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

    elif backend == "yolo_tsm":
        if not model_path:
            raise ValueError(
                "detection.yolo_model_path must be set in config.yaml when using 'yolo_tsm' backend"
            )
        if not tsm_model_path:
            raise ValueError(
                "detection.tsm_model_path must be set in config.yaml when using 'yolo_tsm' backend"
            )
        from .dynamic_roi_tsm import DynamicRoiTsmDetector

        return DynamicRoiTsmDetector(
            yolo_model_path=model_path,
            tsm_model_path=tsm_model_path,
            tsm_classes=tsm_classes,
            alert_classes=alert_classes,
            num_segments=num_segments,
            inference_interval=inference_interval,
            batch_size=batch_size,
            conf_person=conf_person,
            conf_weapon=conf_weapon,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            yolo_imgsz=yolo_imgsz,
            yolo_half=yolo_half,
            tsm_compile=tsm_compile,
            grid_violence_threshold=grid_violence_threshold,
            action_recognition_enabled=action_recognition_enabled,
            object_detection_enabled=object_detection_enabled,
            yolo_scheduler=yolo_scheduler,
            yolo_stream_id=yolo_stream_id,
        )

    elif backend == "yolo_tcclip":
        if not model_path:
            raise ValueError(
                "detection.yolo_model_path must be set in config.yaml when using 'yolo_tcclip' backend"
            )
        if not tcclip_runtime_root:
            raise ValueError(
                "detection.tcclip_runtime_root must be set in config.yaml when using 'yolo_tcclip' backend"
            )
        if not tcclip_checkpoint_path:
            raise ValueError(
                "detection.tcclip_checkpoint_path must be set in config.yaml when using 'yolo_tcclip' backend"
            )
        if not tcclip_cache_dir:
            raise ValueError(
                "detection.tcclip_cache_dir must be set in config.yaml when using 'yolo_tcclip' backend"
            )
        from .dynamic_roi_tcclip import DynamicRoiTcClipDetector
        from .tcclip_classifier import TcClipPrompts

        if tcclip_prompt_classes is None:
            if not str(tcclip_normal_prompt).strip():
                raise ValueError("detection.tcclip_normal_prompt must not be empty")
            if not str(tcclip_fight_prompt).strip():
                raise ValueError("detection.tcclip_fight_prompt must not be empty")
            prompts = TcClipPrompts(tcclip_normal_prompt, tcclip_fight_prompt)
        else:
            prompts = TcClipPrompts(classes=tcclip_prompt_classes)

        return DynamicRoiTcClipDetector(
            yolo_model_path=model_path,
            tcclip_runtime_root=tcclip_runtime_root,
            tcclip_checkpoint_path=tcclip_checkpoint_path,
            tcclip_cache_dir=tcclip_cache_dir,
            normal_prompt=prompts.normal,
            fight_prompt=prompts.fight,
            prompt_classes=prompts.to_payload(),
            # Keep the old keyword/positional slot compatible for external callers.
            alerts_threshold=(
                tcclip_alerts_threshold if tcclip_alerts_threshold is not None else tcclip_fight_threshold
            ),
            num_frames=tcclip_num_frames,
            temporal_stride=tcclip_temporal_stride,
            target_fps=tcclip_target_fps,
            batch_size=tcclip_batch_size,
            inference_interval_sec=tcclip_inference_interval_sec,
            max_rois_per_job=tcclip_max_rois_per_job,
            max_result_age_sec=tcclip_max_result_age_sec,
            action_ttl_sec=tcclip_action_ttl_sec,
            conf_person=conf_person,
            conf_weapon=conf_weapon,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            yolo_imgsz=yolo_imgsz,
            yolo_half=yolo_half,
            action_recognition_enabled=action_recognition_enabled,
            object_detection_enabled=object_detection_enabled,
            action_scheduler=tcclip_action_scheduler,
            yolo_scheduler=yolo_scheduler,
            yolo_stream_id=yolo_stream_id,
        )

    elif backend == "grid_tsm":
        if not model_path:
            raise ValueError(
                "detection.yolo_model_path must be set in config.yaml when using 'grid_tsm' backend"
            )
        if not tsm_model_path:
            raise ValueError(
                "detection.tsm_model_path must be set in config.yaml when using 'grid_tsm' backend"
            )

        from .grid_detector import GridViolenceDetector

        return GridViolenceDetector(
            yolo_model_path=model_path,
            tsm_model_path=tsm_model_path,
            tsm_classes=tsm_classes,
            alert_classes=alert_classes,
            num_segments=num_segments,
            inference_interval=inference_interval,
            batch_size=batch_size,
            conf_person=conf_person,
            conf_weapon=conf_weapon,
            contrast_alpha=contrast_alpha,
            contrast_beta=contrast_beta,
            yolo_imgsz=yolo_imgsz,
            yolo_half=yolo_half,
            tsm_compile=tsm_compile,
            grid_tsm_interval=grid_tsm_interval,
            grid_tsm_segments=grid_tsm_segments,
            grid_violence_threshold=grid_violence_threshold,
            action_recognition_enabled=action_recognition_enabled,
            object_detection_enabled=object_detection_enabled,
        )

    else:
        raise ValueError(
            f"Unknown detection backend '{backend}'. "
            f"Use 'dummy', 'yolo', 'yolo_tsm', 'yolo_tcclip', or 'grid_tsm'."
        )
