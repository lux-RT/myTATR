# Terminal Image Selection Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. User explicitly requested terminal selection and previously authorized updating the existing GitHub branch.

**Goal:** Let the user launch `run.bat` in PyCharm Terminal, choose images interactively, and run the existing TATR + OCR pipeline.

**Architecture:** Add a standard-library-only image selector. Keep explicit `--images` unchanged; enable selection with `--interactive` or with no image arguments in a real terminal. Add a Windows launcher that selects the installed TATR environment and forces interactive mode. OCR, TATR inference, and exports are unchanged.

**Tech Stack:** Python 3.12 standard library; existing model environments; Windows CMD launcher.

**Spec:** This document records the small design for the user's explicit request. Terminal menu (selected approach) avoids editing IDE configurations for each test; file-dialog UI is unnecessary for terminal use; existing command-line arguments remain useful for automation.

## Global Constraints

- Work in the user's current `main` checkout and push the update to `origin/main`.
- Preserve user-owned `.idea/` settings; ignore them in Git rather than publishing them.
- Never interpret pasted paths as shell commands.
- Handle quoted Windows paths, spaces, Chinese filenames, missing images, folder selection, invalid numbers and cancellation.
- Folder listings are non-recursive and paginated; Enter selects the displayed page, `all` explicitly selects the whole folder.
- Cancel before creating output folders or importing/running ML models.

## Review Focus

- Explicit `--images` must bypass prompts.
- Duplicate and reserved stems must be rejected before OCR starts, including folder selection.
- No image selection should cause an implicit full-dataset inference.
- EOF and Ctrl+C cancel cleanly.
- Launcher must work from CMD/PowerShell and with Chinese/space-containing repository paths.

### Task 1: Selector

- [x] Write and run failing tests for quoted paths, numeric/range selection, paging, folder filtering, invalid input/retry and cancellation.
- [x] Implement `parse_selection`, `parse_paths`, `choose_from_list`, `select_images` in `tatr_ocr/interactive.py`.
- [x] Run complete test suite.

### Task 2: CLI and launcher

- [x] Add `--interactive`, TTY auto-selection and cancellation to `run_table_ocr.py`.
- [x] Add `run.bat`, prefer repo `.venv-tatr`, fall back to sibling `.venv-download`; force UTF-8 and preserve exit code.
- [x] Update README and ignore `.idea/`.
- [x] Test actual launcher cancellation and one real-image interactive inference.
- [x] Obtain code review, fix important findings, run final checks, commit and push original branch.

## Verification ledger

- Baseline: 14 tests passed; clean tracked main branch; user `.idea/` settings kept private.
- Selector and CLI: 28 tests passed, including cancellation before model access/output creation and explicit-image bypass.
- Actual Windows launcher: cancellation exits 0 without output directory; UTF-8/Chinese path launch verified.
- Actual selected-image inference: chose menu 1, image 2; exited 0; recovered 9 x 6 grid, 48/49 OCR tokens assigned, all four artifact types verified.
- Independent review: no important findings. Implementation pushed to original main branch and remote verified at b87a6d69adfa2c2eeb2f4702484737aa5fe8dfbc.
