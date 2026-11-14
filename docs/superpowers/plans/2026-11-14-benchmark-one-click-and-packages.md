# One-command benchmark and PNG packages

**Goal:** One `run_benchmark.bat` invocation runs the paired 1000-image evaluation and PNG export; publish both complete result ZIPs with PNGs to GitHub.

**Spec:** User explicitly requested both a single-script paired test and new downloadable packages. Preserve PP-OCRv6 small, seed 42 and count 1000 defaults, existing measurements and old archives. Update the current feature/ppocr-v6 branch.

**Architecture:** Add an `all` runner stage calling prepare/words/predict/ocr/evaluate/report sequentially, stopping on failure. Its default output is a new timestamped directory, with explicit --out for resumption. A Windows BAT reuses the existing TATR interpreter discovery and forwards options. Tests exercise stage order, fail-fast and default output isolation without expensive inference. New ZIPs extend the published originals with 1000 PNGs each and updated source/instructions; compare all other payload bytes. If over the 100 MiB Git file limit, use repository Release attachments and commit download links/checksums rather than large Git blobs.

- [x] Observe orchestration tests fail, implement all stage and BAT, run the suite.
- [x] Build and verify two 4007-entry ZIPs: 1000 each HTML/CSV/JSON/PNG plus 7 metadata files; CRC, unchanged prediction/metric payloads, local PNG byte equality.
- [x] Publish verified attachments and SHA-256 file; update usage/download documentation and a package verification summary.
- [x] Independent code review, local checks, commit/push current branch and verify GitHub CI plus remote assets.

Completion evidence: 43 tests passed locally and on GitHub CI at f6abe6f. Release benchmark-1000-png-20261114 is public and points to f6abe6f; both ZIPs and the two small metadata assets match GitHub server SHA-256/size, and all four unauthenticated download URLs were verified. The release holds the large files; the feature branch contains the launcher, documentation and checksums.
