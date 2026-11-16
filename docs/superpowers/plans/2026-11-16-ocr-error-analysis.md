# Existing OCR benchmark error analysis

**Goal:** Diagnose the original 1000-table PP-OCRv6 small experiment and give every one of its 927 AccCon failures an auditable classification, without modifying predictions or published scores.

**Spec:** User explicitly asks to analyze all 927 non-exact tables before implementing improvements. Classification is relative to the experiment's reference data, not an assertion that official text is visually infallible. Keep current inference, fusion, evaluation and model files unchanged.

**Architecture:** Add a standard-library diagnostic helper and a read-only analysis entry point. Reconstruct reference cells from the same XML/official words, verify cached input fingerprints, and reproduce existing fusion from cached structure/OCR outputs for token provenance. Separate topology errors from directly aligned content differences. Text-difference categories are whitespace, narrowly specified dash/quote variants, missing content, extra content and other content changes; mixed categories remain explicit. Geometry/elsewhere-text indicators are candidates, not automatically proven OCR or assignment causes. All raw comparison evidence is retained.

**Outputs:** A Markdown report, all-927 classification list, machine-readable records and a local HTML inspection table in docs/benchmark-1000/error-analysis-20261116. Large per-position details and audit caches remain in a separate ignored output directory if needed. No new OCR or TATR inference; no score recalculation or normalization replacing the official metric.

- [x] Add failing tests for text categories, mixed categories, merged/overlapping cells, incomplete references and conservative relocation evidence; implement helpers.
- [x] Process every manifest sample, reconstruct reference cells and validate the original 73/927 split, source fingerprints and unchanged saved exports. Distinguish grid-position counts from unique cells.
- [x] Classify all 927, visually inspect 14 original images, document verified examples and unresolved causes separately; generate complete readable outputs. All 6003 source hashes unchanged; prior 798-table diagnostic counts reproduced exactly.
- [x] Independent review, 55-test suite and artifact consistency checks. Read-only output/cache path guard added with a red/green test. Browser preview was rejected for file://; offline HTML validation used, with no bypass or browser-interaction claim.

Publication step: update the current feature/ppocr-v6 branch with analysis code/reports and verify GitHub CI; do not alter the original experimental result files or published ZIP assets.

Primary categories are observations, not automatic causal labels: structure mismatch takes precedence; topology-aligned text is partitioned into whitespace-only, symbol-variant-only, missing-only, extra-only, other-content-only and mixed textual differences. A separate unresolved-alignment category is permitted rather than forcing an invalid comparison.
