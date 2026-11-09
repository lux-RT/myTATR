"""Normalize RapidOCR outputs into JSON-safe, original-coordinate tokens."""
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
        bbox = [float(min(xs)) / sx, float(min(ys)) / sy, float(max(xs)) / sx, float(max(ys)) / sy]
        if not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("OCR returned an invalid polygon")
        tokens.append({"text": text, "bbox": bbox, "confidence": float(confidence),
                       "block_num": 0, "line_num": index, "span_num": 0})
    return tokens


def normalize_result(result, scale=1.0):
    """Adapt RapidOCR 3.x result objects without NumPy truth-value checks."""
    fields = (result.boxes, result.txts, result.scores)
    if all(field is None for field in fields):
        return []
    if any(field is None for field in fields):
        raise ValueError("RapidOCR returned an incomplete result")
    boxes, texts, scores = fields
    if len(boxes) != len(texts) or len(texts) != len(scores):
        raise ValueError("RapidOCR result arrays have different lengths")
    return normalize_lines(list(zip(boxes, texts, scores)), scale)
