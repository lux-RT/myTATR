"""Recognize structure and text in cropped tables using local TATR + OCR."""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from tatr_ocr.inputs import default_model_dir, validate_stems

ROOT = Path(__file__).resolve().parent
LOCAL_DATA = ROOT.parent / "PubTables-1M复现"
DEFAULT_SAMPLES = ["PMC1262774_table_0", "PMC1064078_table_0",
                   "PMC1064082_table_1", "PMC1064879_table_0"]


def venv_python(name):
    return ROOT / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def draw_overlay(image, table, tokens, path):
    from PIL import ImageDraw
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for cell in table["cells"]:
        merged = len(cell["row_nums"]) * len(cell["column_nums"]) > 1
        draw.rectangle(cell["bbox"], outline="#bb2588" if merged else "#176fba", width=2)
    for token in tokens:
        draw.rectangle(token["bbox"], outline="#16a070", width=1)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, help="One or more cropped table images")
    parser.add_argument("--image-dir", type=Path, default=LOCAL_DATA / "datasets/PubTables-1M-structure/"
                        "PubTables-1M-Structure_Images_Test/PubTables-1M-Structure_Images_Test")
    parser.add_argument("--model-dir", type=Path, default=default_model_dir(ROOT))
    parser.add_argument("--ocr-python", type=Path, default=venv_python(".venv-ocr"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--ocr-scale", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or not 1 <= args.threads <= 64 or not 0 < args.ocr_scale <= 4:
        parser.error("threshold must be in (0,1], threads in [1,64], OCR scale in (0,4]")
    images = [p.resolve() for p in (args.images or [args.image_dir / (s + ".jpg") for s in DEFAULT_SAMPLES])]
    for path in images + [args.ocr_python, args.model_dir / "model.safetensors"]:
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    try:
        validate_stems(images)
    except ValueError as error:
        parser.error(str(error))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a new --out to preserve earlier results")
    args.out.mkdir(parents=True, exist_ok=True)

    # OCR is isolated from the established torch/Transformers environment.
    subprocess.run([str(args.ocr_python.resolve()), str(ROOT / "ocr_worker.py"),
                    "--images", *map(str, images), "--output", str((args.out / "ocr.json").resolve()),
                    "--scale", str(args.ocr_scale), "--threads", str(args.threads)],
                   check=True, cwd=ROOT)
    ocr = json.loads((args.out / "ocr.json").read_text(encoding="utf-8"))
    by_path = {r["image"]: r for r in ocr["images"]}

    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    from tatr_ocr.fusion import fuse
    from tatr_ocr.export import html_document, write_csv

    torch.set_num_threads(args.threads)
    start = perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True, use_fast=False)
    model = TableTransformerForObjectDetection.from_pretrained(
        args.model_dir, local_files_only=True, use_pretrained_backbone=False).eval()
    load_seconds = perf_counter() - start
    results = []
    for path in images:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        record = by_path[str(path)]
        if list(image.size) != record["image_size"]:
            raise ValueError("TATR and OCR image coordinate systems do not match")
        start = perf_counter()
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        if not torch.isfinite(outputs.logits).all() or not torch.isfinite(outputs.pred_boxes).all():
            raise ValueError("TATR returned non-finite predictions")
        prediction = processor.post_process_object_detection(
            outputs, threshold=args.threshold, target_sizes=[(image.height, image.width)])[0]
        objects = [{"label": model.config.id2label[int(label)], "score": float(score),
                    "bbox": [float(v) for v in bbox]} for label, score, bbox in
                   zip(prediction["labels"], prediction["scores"], prediction["boxes"])]
        tatr_seconds = perf_counter() - start
        start = perf_counter()
        table = fuse(objects, record["tokens"], args.threshold)
        fusion_seconds = perf_counter() - start
        result = {"image": path.name, "size": list(image.size), "structure_threshold": args.threshold,
                  "objects": objects, "tokens": record["tokens"], "table": table,
                  "seconds": {"ocr": record["seconds"], "tatr": round(tatr_seconds, 4),
                              "fusion": round(fusion_seconds, 4)}}
        (args.out / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.out / f"{path.stem}.html").write_text(html_document(path.stem, table), encoding="utf-8")
        write_csv(args.out / f"{path.stem}.csv", table)
        draw_overlay(image, table, record["tokens"], args.out / f"{path.stem}_overlay.png")
        results.append({"image": path.name, "rows": table["rows"], "columns": table["columns"],
                        "cells": len(table["cells"]), "populated_cells": sum(bool(c["text"]) for c in table["cells"]),
                        "ocr_tokens": len(record["tokens"]), "assigned_tokens": table["assigned_tokens"],
                        "unassigned_tokens": len(table["unassigned_tokens"]),
                        "ambiguous_tokens": len(table["ambiguous_token_indices"]), "seconds": result["seconds"]})
        print(f"TATR+OCR {path.stem}: {table['rows']} rows x {table['columns']} columns, "
              f"{table['assigned_tokens']}/{len(record['tokens'])} text boxes assigned", flush=True)
    summary = {"device": "cpu", "tatr": "microsoft/table-transformer-structure-recognition",
               "torch": torch.__version__, "transformers": version("transformers"),
               "ocr_engine": ocr["engine"], "ocr_version": ocr["version"],
               "onnxruntime_version": ocr["onnxruntime_version"], "ocr_scale": args.ocr_scale,
               "model_load_seconds": round(load_seconds, 4), "images": results,
               "limitations": "Cropped-table smoke test; token assignment rate is NOT OCR accuracy or GriTS."}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Outputs: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
