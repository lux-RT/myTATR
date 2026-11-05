"""Normalize legacy RapidOCR line output into TATR-compatible tokens."""
import math


def normalize_lines(lines, scale=1.0):
    sx, sy = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    if not all(math.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError("OCR scale must be positive and finite")
    tokens = []
    for index, (polygon, text, confidence) in enumerate(lines or []):
        if not text.strip():
            continue
        xs, ys = zip(*polygon)
        bbox = [min(xs) / sx, min(ys) / sy, max(xs) / sx, max(ys) / sy]
        if not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("OCR returned an invalid polygon")
        tokens.append({"text": text, "bbox": bbox, "confidence": float(confidence),
                       "block_num": 0, "line_num": index, "span_num": 0})
    return tokens
