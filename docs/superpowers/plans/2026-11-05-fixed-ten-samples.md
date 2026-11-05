# Fixed Ten-Sample Runner Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. User explicitly requests removal of the selector, ten diverse images from their downloaded test split, automatic inference on launch, and update of the existing GitHub branch.

**Goal:** Running `run.bat` or `run_table_ocr.py` immediately processes ten bundled sample images.

**Architecture:** Store original copied dataset images and provenance in `sample_images/`. Remove the terminal selector and its CLI flag/tests. Discover sorted supported image files in the bundled directory by default; preserve explicit `--images` and optional `--image-dir`. Keep inference/models unchanged.

**Tech Stack:** Existing Python 3.12 / CPU TATR + OCR environments; standard-library input discovery; Windows launcher.

**Spec:** This small design implements the user's explicit replacement of the interactive menu. Chosen approach: fixed ten images in the project, rather than further modifying the interactive selector.

## Global Constraints

- The ten images come from the downloaded PubTables-1M test set, from ten different papers, selected by structural variation and visual review.
- Copy original bytes and retain hashes, original filenames, article links and CDLA-Permissive-2.0 license.
- No stdin prompt, file dialog or image-selection step remains in the run path.
- The default path is relative to the script, independent of terminal working directory.
- Keep user-owned IDE settings private and keep models/dataset archives outside Git.
- Preserve original branch `main`; push and verify it after completion.

## Tasks

- [x] Select and visually inspect ten diverse samples; copy them with manifest and preview.
- [x] Write failing tests for automatic ten-image defaults, directory discovery, explicit inputs, and missing/empty input directories.
- [x] Implement input resolution and remove interactive code/tests/flags; simplify `run.bat`.
- [x] Update usage instructions and sample provenance documentation.
- [x] Run full tests and actual launcher on all ten images; verify artifacts and record real outcomes.
- [ ] Obtain independent review; fix important findings; commit/push/verify original main branch.

## Verification ledger

- 18 unit tests pass; the removed menu tests are intentionally replaced by fixed-default/directory tests.
- Actual `run.bat --out outputs/ten-sample-test` exited 0 without stdin, processing all ten images.
- Source hashes and bytes match the downloaded dataset; all 40 per-image artifacts verified, with correct CSV dimensions, complete grid coverage and token accounting.
- All ten predicted row/column counts match independently read annotations. This is not an OCR or merged-cell accuracy evaluation.
- Independent reviewer: no important issues. Original main publication pending.
