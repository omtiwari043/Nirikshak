"""
Unit tests for the OCR service's pure/helper logic — text normalization,
importance scoring, and line merging. These deliberately avoid loading the
actual PaddleOCR model (that needs downloaded weights + real inference and
is exercised instead by an integration/smoke test against a real image;
see docs/DEPLOYMENT.md).

Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ocr_service import (
    TextLine, normalize_text, importance_score, merge_lines, px_to_mm,
)


def test_normalize_text_strips_noise_but_keeps_useful_symbols():
    assert normalize_text("MRP: Rs. 185.00/-") == "mrp: rs. 185.00/-"
    assert normalize_text("  Net   Qty  ") == "net qty"


def test_importance_score_favors_legal_declarations():
    assert importance_score("MRP Rs. 185.00") > importance_score("SunGold Sunflower Oil")
    assert importance_score("blue plastic box shape") == 0.0


def test_merge_lines_deduplicates_and_prefers_higher_quality_reading():
    candidates = [
        TextLine(text="MRP Rs 185.00", x=10, y=10, width=100, height=12, confidence=52.0, block_num=0, line_num=0),
        TextLine(text="MRP Rs 185.00", x=10, y=10, width=100, height=12, confidence=91.0, block_num=0, line_num=0),
        TextLine(text="xz", x=200, y=200, width=10, height=10, confidence=20.0, block_num=0, line_num=1),
    ]
    merged = merge_lines(candidates)
    texts = [line.text for line in merged]

    assert texts.count("MRP Rs 185.00") == 1
    kept = next(line for line in merged if line.text == "MRP Rs 185.00")
    assert kept.confidence == 91.0
    # Low-confidence, non-declaration garbage below 2 chars is dropped entirely.
    assert "xz" not in texts or len(merged) == 1


def test_merge_lines_drops_low_confidence_non_declaration_noise():
    candidates = [
        TextLine(text="asdkj qpx", x=0, y=0, width=5, height=5, confidence=15.0, block_num=0, line_num=0),
    ]
    assert merge_lines(candidates) == []


def test_px_to_mm_uses_calibration_when_available():
    calibrated_mm, source = px_to_mm(40, calibration_mm_per_px=0.1)
    assert calibrated_mm == 4.0
    assert source == "calibrated"

    estimated_mm, source = px_to_mm(40, calibration_mm_per_px=None)
    assert source == "estimated"
    assert estimated_mm > 0
