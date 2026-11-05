# TATR OCR Integration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly requested implementation, real-image tests and upload to lux-RT/myTATR; execute inline without another approval cycle.

**Goal:** Turn locally available cropped table images into tables containing recognized text, and publish reproducible code and sample results.

**Architecture:** Keep the local Transformers TATR checkpoint unchanged. Run RapidOCR/ONNX Runtime in a separate environment and exchange UTF-8 JSON with the TATR process. Adapt the unchanged Microsoft postprocess module to reconstruct cells and assign OCR tokens. Export JSON, merged-cell HTML, CSV and annotated PNG.

**Tech Stack:** Python 3.12; torch 2.8 CPU; transformers 4.57.1; RapidOCR-ONNXRuntime 1.4.4; ONNX Runtime 1.23.2; Pillow; PyMuPDF geometry for the official postprocessor.

**Spec:** User requests in this chat, including the explicit destination https://github.com/lux-RT/myTATR.git.

## Global Constraints

- Do not change the existing model weights, download program or dataset.
- Keep virtual environments, model weights and datasets out of Git.
- Use predicted TATR structures and real OCR; ground truth must not feed inference.
- Default tests use the four previously selected PubTables-1M test images.
- Upload completed and verified code to the repository's original branch, as requested by the user.
- CPU support is required; camera photographs and full-page table detection are outside this initial cropped-table workflow.

## Review Focus

- Missing text must not delete blank rows or columns.
- Tokens outside the table must not be silently counted as assigned.
- Merged cells must occupy their full rectangular grid span without duplicate HTML cells.
- Unicode file paths must work across the two Windows processes.
- Empty OCR, unreadable images and absent structure must have explicit output or errors.

### Task 1: Cell fusion and exports

- [x] Write tests for two-column text assignment, merged headers, empty OCR, outside tokens and HTML escaping.
- [x] Run tests and observe failure before implementation.
- [x] Vendor unmodified pinned Microsoft postprocess.py and MIT license with commit/hash provenance.
- [x] Implement `fuse(objects, tokens, threshold)` in `tatr_ocr/fusion.py` and HTML/CSV exports in `tatr_ocr/export.py`.
- [x] Run all unit tests.

### Task 2: Real OCR and TATR CLI

- [x] Create isolated `.venv-ocr`, install pinned OCR dependencies, retain dependency lock.
- [x] Add `ocr_worker.py` returning original-image-coordinate OCR boxes, confidence and ordering fields.
- [x] Add `run_table_ocr.py` loading local TATR weights, calling the OCR worker and writing all artifacts.
- [x] Run four real dataset samples without reading XML during inference; inspect rendered results and characterize errors.

### Task 3: Reproduction and publication

- [x] Add setup/reproduction instructions, exact file-change explanation and sample report.
- [x] Check complete tests, dependency consistency, outputs and a fresh offline run.
- [x] Obtain independent code review, address important findings.
- [x] Commit and push the verified implementation and curated sample artifacts to the remote's original branch; verify remote commit.

## Execution ledger

- Initial state: empty remote cloned to `D:/Projects/Codex/小毕设/myTATR`; no existing application code or tests.
- Ruling: use legacy RapidOCR-ONNXRuntime 1.4.4, which bundles small PP-OCR models, to keep Windows CPU setup and offline reruns simple. The OCR adapter can later be replaced without retraining TATR.
- Ruling: use two environments locally so the working `.venv-download` retains its existing dependency set except the additive official-postprocess geometry dependency. Both environments and their versions are documented.

- Task 1: complete — official postprocessor adapted; geometry and export tests pass.
- Task 2: complete — final four-image offline run exited 0; output accounting/grid/CSV checks pass.
- Task 3: verification complete — 14 tests and both pip checks pass; independent review issues fixed and reviewed. Implementation published to original main branch; remote verified at 3f3ffaa7b22bef01f1f3edcc2e75a67731050d7a.
- Ruling: include only four attributed small sample images and derived artifacts to permit reproduction without the full dataset.

- Task 3: complete — original main branch pushed and remote commit independently verified.
