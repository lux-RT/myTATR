"""Run OCR in its own environment and exchange only UTF-8 JSON with TATR."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from tatr_ocr.ocr_adapter import normalize_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.scale) or not 0 < args.scale <= 4 or not 1 <= args.threads <= 64:
        parser.error("scale must be in (0,4] and threads must be in [1, 64]")
    import numpy as np
    import cv2
    from PIL import Image, ImageOps
    from rapidocr_onnxruntime import RapidOCR

    cv2.setNumThreads(args.threads)
    engine = RapidOCR(intra_op_num_threads=args.threads, inter_op_num_threads=1)
    records = []
    for path in args.images:
        start = perf_counter()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        original_size = image.size
        # Account for integer rounding in cv2.resize, including fractional scales.
        scaled = np.asarray(image)
        if args.scale != 1:
            scaled = cv2.resize(scaled, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        lines, timing = engine(scaled[:, :, ::-1].copy(), use_cls=False)
        factors = (scaled.shape[1] / original_size[0], scaled.shape[0] / original_size[1])
        tokens = normalize_lines(lines, factors)
        records.append({"image": str(path.resolve()), "image_size": list(original_size),
                        "tokens": tokens, "seconds": round(perf_counter() - start, 4),
                        "model_stage_seconds": timing})
        print(f"OCR {path.stem}: {len(tokens)} text boxes", flush=True)
    data = {"engine": "RapidOCR-ONNXRuntime", "version": version("rapidocr-onnxruntime"),
            "onnxruntime_version": version("onnxruntime"), "scale": args.scale,
            "device": "cpu", "images": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
