"""Run OCR in its own environment and exchange only UTF-8 JSON with TATR."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from tatr_ocr.ocr_adapter import normalize_result

ROOT = Path(__file__).resolve().parent


def model_metadata(engine):
    """Record the pinned package's actual model route and cached weight digest."""
    from rapidocr.inference_engine.base import FileInfo, InferSession

    models = {}
    for task, loaded in (("Det", engine.text_det), ("Rec", engine.text_rec)):
        cfg = engine.cfg[task]
        info = InferSession.get_model_url(FileInfo(
            cfg.engine_type, cfg.ocr_version, cfg.task_type, cfg.lang_type, cfg.model_type))
        path = Path(engine.cfg.Global.model_root_dir) / Path(info["model_dir"]).name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest is not None and digest.lower() != info["SHA256"].lower():
            raise ValueError(f"OCR weight checksum mismatch: {path.name}")
        models[task.lower()] = {"file": path.name, "sha256": digest,
                               "source": info["model_dir"], "loaded": loaded is not None}
    return models


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
    from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion

    cv2.setNumThreads(args.threads)
    engine = RapidOCR(params={
        "Global.model_root_dir": str(ROOT / "models" / "ocr-v6"),
        "Global.use_det": True, "Global.use_cls": False, "Global.use_rec": True,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Det.ocr_version": OCRVersion.PPOCRV6, "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": ModelType.SMALL, "Rec.model_type": ModelType.SMALL,
        "Det.lang_type": "ch", "Rec.lang_type": "ch",
        "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
    })
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
        result = engine(scaled[:, :, ::-1].copy(), use_cls=False)
        factors = (scaled.shape[1] / original_size[0], scaled.shape[0] / original_size[1])
        tokens = normalize_result(result, factors)
        records.append({"image": str(path.resolve()), "image_size": list(original_size),
                        "tokens": tokens, "seconds": round(perf_counter() - start, 4),
                        "model_stage_seconds": [float(t) if t is not None else None
                                                for t in (result.elapse_list or [])]})
        print(f"OCR {path.stem}: {len(tokens)} text boxes", flush=True)
    data = {"engine": "RapidOCR-ONNXRuntime", "version": version("rapidocr"),
            "model_family": "PP-OCRv6", "model_tier": "small", "orientation_classifier": False,
            "models": model_metadata(engine),
            "onnxruntime_version": version("onnxruntime"), "scale": args.scale,
            "device": "cpu", "images": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
