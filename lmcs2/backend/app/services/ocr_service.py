"""
PaddleOCR-powered OCR service for packaged-commodity label photographs.

Why PaddleOCR (over Tesseract):
- PaddlePaddle's PP-OCRv4 detector + recognizer meaningfully outperforms
  Tesseract on real-world "in the wild" text: curved/glossy packaging,
  small declaration blocks, mixed font sizes, and busy backgrounds.
- Built-in text-direction classifier (`use_angle_cls`) removes the need to
  brute-force multiple Tesseract PSM modes.
- A single detector call returns tight quadrilateral boxes per text line,
  which we use directly for font-size (mm) measurement instead of the
  looser word-bounding-boxes Tesseract produces.

Strategy:
1. Load the original image (from disk, bytes, or an in-memory frame from
   the live camera).
2. Downscale very large photos to a sane working size, upscale very small /
   tightly-cropped ones — PP-OCRv4 has a sweet spot around 960-2200 px on
   the long side.
3. Run the detector+recognizer on:
     - the full frame (a couple of lighting-robust variants), and, in
       "thorough" mode only,
     - targeted crops of the bottom/right declaration block (where MRP,
       net quantity, batch no. and MFD are usually printed small), tried
       at 0/90/270 degree rotation.
4. Merge duplicate lines across passes, keeping the highest-confidence /
   most legally-relevant reading of each line.
5. Preserve bounding boxes + confidence for downstream font-size and
   rule-engine analysis.

Two entry points:
- `run_ocr(...)`            -> thorough, multi-pass pipeline used for the
                                final, persisted compliance scan.
- `run_quick_ocr(...)`      -> single-pass, full-frame-only pipeline used
                                for the live camera's real-time on-screen
                                feedback (must return in well under a
                                second on CPU).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.config import settings

ASSUMED_DPI_FALLBACK = 300
MM_PER_INCH = 25.4


class OCRUnavailableError(RuntimeError):
    """Raised when the PaddleOCR engine cannot be loaded/used by this server."""


class OCRExtractionError(RuntimeError):
    """Raised when a submitted image contains no usable OCR text."""


# ---------------------------------------------------------
# ENGINE (lazy singleton — PaddleOCR model load is expensive,
# ~1-3s CPU / <1s GPU, so we pay that cost once per process, not per request)
# ---------------------------------------------------------

_engine_lock = threading.Lock()
_engines: dict[str, "PaddleOCR"] = {}  # noqa: F821 (forward ref, imported lazily)


def get_ocr_engine(lang: Optional[str] = None):
    """Return a process-wide singleton PaddleOCR engine for the given language."""
    lang = lang or settings.OCR_LANG

    if lang in _engines:
        return _engines[lang]

    with _engine_lock:
        if lang in _engines:  # re-check after acquiring the lock
            return _engines[lang]

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover
            raise OCRUnavailableError(
                "PaddleOCR is not installed. Run `pip install -r requirements.txt` "
                "(paddlepaddle + paddleocr) before starting the backend."
            ) from exc

        try:
            engine = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=settings.OCR_USE_GPU,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.8,
                drop_score=0.35,
                show_log=False,
            )
        except Exception as exc:
            raise OCRUnavailableError(
                "PaddleOCR failed to initialize. If this is the first run, it needs a one-time "
                "internet connection to download detection/recognition/classification models "
                "(~15 MB), or bake them into the image at build time — see docs/DEPLOYMENT.md."
            ) from exc

        _engines[lang] = engine
        return engine


def ensure_ocr_available() -> None:
    """Fail early (e.g. at app startup) rather than silently evaluating an empty result."""
    get_ocr_engine()


def warm_up_ocr() -> None:
    """Load the model and run one throwaway inference so the first real request is fast."""
    try:
        engine = get_ocr_engine()
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        engine.ocr(blank, cls=True)
    except Exception:
        # Non-fatal: the first real request will surface a clearer OCRUnavailableError.
        pass


@dataclass
class TextLine:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float  # 0-100
    block_num: int
    line_num: int
    angle_deg: float = 0.0


@dataclass
class OcrResult:
    full_text: str
    lines: list[TextLine] = field(default_factory=list)
    image_width_px: int = 0
    image_height_px: int = 0
    skew_angle_deg: float = 0.0


# ---------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image at {path}. File may be corrupt or unsupported.")

    return image


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image bytes. File may be corrupt or unsupported.")

    return image


def _resize_to_working_size(image: np.ndarray, min_side: int = 700, max_side: int = 2000) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    shortest = min(h, w)

    if longest > max_side:
        scale = max_side / float(longest)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if shortest < min_side:
        scale = min_side / float(shortest)
        # Cap the upscale so we don't blow up a tiny thumbnail into noise.
        scale = min(scale, 3.0)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return image


# ---------------------------------------------------------
# BASIC IMAGE PREPARATION
# ---------------------------------------------------------

def upscale(image: np.ndarray, scale: float = 2.5) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """PaddleOCR expects a 3-channel BGR array."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    """Local-contrast enhancement — helps glare/uneven-lighting frames without
    destroying character strokes the way binarization can."""
    if len(image.shape) == 2:
        g = image
    else:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------
# IMAGE CROPS (targeted declaration-block passes, thorough mode only)
# ---------------------------------------------------------

def make_crops(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Targeted OCR views for small, dense declaration text.

    The bottom-right declaration panel on many packaged products
    (MRP, Net Quantity, Batch No., MFD, Unit Sale Price) is often printed
    much smaller than the brand name, so it benefits from its own
    higher-zoom OCR pass rather than relying on the full-frame pass alone.
    """
    h, w = image.shape[:2]

    crops = [
        ("bottom_right", image[int(h * 0.50):h, int(w * 0.40):w]),
        ("bottom_declarations", image[int(h * 0.62):h, int(w * 0.25):w]),
    ]

    return [(name, crop) for name, crop in crops if crop.size > 0]


def make_rotated_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Small declaration blocks are sometimes printed sideways along a package edge."""
    return [
        ("normal", image),
        ("rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]


# ---------------------------------------------------------
# OCR (single pass)
# ---------------------------------------------------------

def run_single_ocr(image: np.ndarray) -> list[TextLine]:
    """Run PaddleOCR detection + angle classification + recognition on one image array."""
    if image is None or image.size == 0:
        return []

    engine = get_ocr_engine()
    image = to_bgr(image)

    try:
        raw = engine.ocr(image, cls=True)
    except Exception as exc:
        raise OCRUnavailableError(f"PaddleOCR inference failed: {exc}") from exc

    if not raw or not raw[0]:
        return []

    results: list[TextLine] = []

    for line_idx, item in enumerate(raw[0]):
        if not item or len(item) < 2:
            continue

        box, (text, conf) = item
        text = (text or "").strip()

        if not text:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = int(round(min(xs))), int(round(max(xs)))
        y0, y1 = int(round(min(ys))), int(round(max(ys)))

        # Skew of this specific line from its quad box, useful for font-height
        # normalization on angled labels (curved bottles, wrapped pouches).
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        line_angle = float(np.degrees(np.arctan2(dy, dx))) if (dx or dy) else 0.0

        results.append(
            TextLine(
                text=text,
                x=x0,
                y=y0,
                width=max(1, x1 - x0),
                height=max(1, y1 - y0),
                confidence=round(float(conf) * 100, 1),
                block_num=0,
                line_num=line_idx,
                angle_deg=round(line_angle, 1),
            )
        )

    return results


# ---------------------------------------------------------
# TEXT NORMALIZATION / IMPORTANCE SCORING
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text only for duplicate detection. We keep the original OCR text elsewhere."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9₹%./:@\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


IMPORTANT_TERMS = [
    "mrp", "maximum retail price", "net quantity", "net qty", "quantity",
    "manufactured by", "marketed by", "packed by", "imported by",
    "date", "month", "year", "mfg", "mfd", "batch", "lot",
    "unit sale price", "price", "consumer care", "customer care",
    "ingredients", "allergen", "fssai", "license", "lic",
    "contact", "address", "email", "phone",
]


def importance_score(text: str) -> float:
    normalized = normalize_text(text)
    score = 0.0

    for term in IMPORTANT_TERMS:
        if term in normalized:
            score += 25.0

    if re.search(r"\d", text):
        score += 8.0

    if re.search(r"(₹|rs\.?|kg|g\b|gm|mg|ml|l\b|%|pcs)", normalized):
        score += 12.0

    return score


# ---------------------------------------------------------
# MERGE OCR RESULTS
# ---------------------------------------------------------

def merge_lines(candidates: list[TextLine]) -> list[TextLine]:
    groups: dict[str, TextLine] = {}

    for line in candidates:
        text = line.text.strip()
        if len(text) < 2:
            continue

        normalized = normalize_text(text)
        if not normalized:
            continue

        score = line.confidence + importance_score(text)

        if normalized not in groups:
            groups[normalized] = line
        else:
            existing = groups[normalized]
            existing_score = existing.confidence + importance_score(existing.text)
            if score > existing_score:
                groups[normalized] = line

    result = list(groups.values())

    # Remove very weak garbage unless it looks like an important declaration.
    # PaddleOCR's recognizer is already fairly conservative (drop_score=0.35
    # filters low-confidence reads at the engine level), so this is a lighter
    # secondary pass than the old Tesseract heuristic needed.
    filtered = [
        line for line in result
        if line.confidence >= 40 or importance_score(line.text) >= 25
    ]

    filtered.sort(key=lambda x: (-importance_score(x.text), -x.confidence, x.y, x.x))
    return filtered


# ---------------------------------------------------------
# MAIN OCR — THOROUGH (final, persisted compliance scan)
# ---------------------------------------------------------

def run_ocr(image_path: str, fast: bool = False) -> OcrResult:
    """
    OCR pipeline for packaged-commodity labels.

    fast=False (default): thorough multi-pass pipeline — full frame under a
        couple of lighting conditions, plus targeted/rotated declaration-block
        crops. Used once per submitted scan image; a few hundred ms to ~2s on
        CPU depending on image size and how many crops fire.
    fast=True: a single full-frame pass, no crops. Same code path as
        `run_quick_ocr` — kept here so callers that already have a file path
        (rather than an in-memory frame) can opt into the fast path too.
    """
    ensure_ocr_available()
    original = load_image(image_path)
    return _run_ocr_on_array(original, fast=fast)


def run_quick_ocr(image: np.ndarray) -> OcrResult:
    """Single-pass OCR on an in-memory frame — used for the live camera's
    real-time on-screen feedback loop. Must stay fast (no crops, no rotation
    passes, no denoise)."""
    ensure_ocr_available()
    return _run_ocr_on_array(image, fast=True)


def _run_ocr_on_array(original: np.ndarray, fast: bool) -> OcrResult:
    original = _resize_to_working_size(original)
    h, w = original.shape[:2]

    all_candidates: list[TextLine] = []
    ocr_pass_texts: list[str] = []

    def run_pass(img: np.ndarray) -> None:
        candidates = run_single_ocr(img)
        all_candidates.extend(candidates)
        pass_text = "\n".join(line.text for line in candidates)
        if pass_text.strip():
            ocr_pass_texts.append(pass_text)

    # -----------------------------------------------------
    # 1. FULL FRAME
    # -----------------------------------------------------
    run_pass(original)

    if not fast:
        # A CLAHE-enhanced second pass materially helps on glossy/glared
        # packaging without the cost of a full preprocessing-variant sweep
        # (PaddleOCR's own detector already normalizes contrast reasonably
        # well, unlike Tesseract, so we don't need 6+ binarization variants).
        try:
            run_pass(clahe_enhance(original))
        except Exception:
            pass

    # -----------------------------------------------------
    # 2. TARGETED DECLARATION CROPS (thorough mode only)
    # -----------------------------------------------------
    if not fast:
        for crop_name, crop in make_crops(original):
            for rotation_name, rotated in make_rotated_variants(crop):
                zoomed = upscale(rotated, 2.0)
                try:
                    run_pass(zoomed)
                except Exception as exc:
                    print(f"Targeted OCR failed (crop={crop_name}, rotation={rotation_name}): {exc}")

    # -----------------------------------------------------
    # 3. MERGE RESULTS
    # -----------------------------------------------------
    merged = merge_lines(all_candidates)
    full_text = "\n\n".join(ocr_pass_texts).strip()

    if not full_text:
        raise OCRExtractionError(
            "No readable text could be extracted from this image. Retake the photo with less glare, "
            "better focus, and a closer view of the label."
        )

    return OcrResult(
        full_text=full_text,
        lines=merged,
        image_width_px=w,
        image_height_px=h,
        skew_angle_deg=merged[0].angle_deg if merged else 0.0,
    )


# ---------------------------------------------------------
# FONT SIZE ANALYSIS
# ---------------------------------------------------------

def px_to_mm(height_px: int, calibration_mm_per_px: Optional[float]) -> tuple[float, str]:
    if calibration_mm_per_px:
        return round(height_px * calibration_mm_per_px, 2), "calibrated"

    mm_per_px = MM_PER_INCH / ASSUMED_DPI_FALLBACK
    return round(height_px * mm_per_px, 2), "estimated"


def analyze_font_sizes(ocr_result: OcrResult, calibration_mm_per_px: Optional[float] = None) -> list[dict]:
    measurements = []

    for line in ocr_result.lines:
        height_mm, confidence = px_to_mm(line.height, calibration_mm_per_px)
        measurements.append(
            {
                "text": line.text,
                "bbox": {"x": line.x, "y": line.y, "width": line.width, "height": line.height},
                "ocr_confidence_pct": line.confidence,
                "estimated_height_mm": height_mm,
                "measurement_confidence": confidence,
            }
        )

    return measurements
