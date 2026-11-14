# Benchmark PNG Implementation Plan

> Execute inline with superpowers:executing-plans; complete one independent review before publishing the existing feature branch.

**Goal:** Export the same cell/text overlays as the ten-image runner for both thousand-image benchmark variants, including existing results without rerunning inference or scoring.

**Spec:** User requests the missing PNG feature after being shown that saved cell coordinates and original images allow backfilling both sets. Blue ordinary cells, purple merged cells, green words/OCR tokens; original image coordinates and dimensions. Error results must be visibly marked. Default OCR remains PP-OCRv6 small.

**Architecture:** Add a Pillow drawing helper, call a saved-result renderer after paired exports in evaluate_one, and expose a separate `render` stage for completed experiments. The render stage reads manifest, saved per-table cells and token files, and writes only `*_overlay.png`. It must not modify any saved predictions, metrics or configuration fingerprints. The existing benchmark binds stage caches to the complete runner source hash, so historical experiments use render rather than re-entering predict/evaluate with changed code.

**Files:** tatr_ocr/visualization.py; run_benchmark.py; tests/test_benchmark_png.py; CI Pillow dependency; README and benchmark usage/report notes.

- [x] Add failing drawing/backfill tests: correct original-coordinate pixels and merged/text colors; preserve input/export/metric bytes; missing exports fail clearly; auto export integration and failed-table marking.
- [x] Implement drawing and render stage; new evaluations export PNG automatically. Keep the ten-image code and model/fusion/scoring unchanged.
- [x] Run the full unit suite, then render all existing 2,000 PNGs. Verify decodability, source dimensions and byte-identical original evaluation artifacts; visually inspect examples.
- [x] Update commands/color legend/cache guidance and independently review. 38 tests pass; 13,110 existing non-PNG artifacts have identical SHA-256 before/after rendering.

Publication step: commit and push feature/ppocr-v6, then verify the remote commit and GitHub CI. Existing authorization requires updating this original code branch; main is not part of this change.

Large local PNG collections remain under ignored outputs; the previously published ZIPs remain historical HTML/CSV/JSON archives, with their recorded hashes unchanged.
