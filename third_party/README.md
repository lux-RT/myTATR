# Third-party code and models

- `tatr/postprocess.py` is an **unmodified** copy from [Microsoft Table Transformer](https://github.com/microsoft/table-transformer), pinned to commit `16d124f616109746b7785f03085100f1f6247575`. Its MIT license is in `tatr/LICENSE`; download URLs and SHA-256 digests are in `tatr/provenance.json`.
- Local adapters are in `tatr_ocr/`; they are separate from Microsoft's source. The adapter builds structure without OCR tokens, then assigns OCR tokens in a second pass to preserve blank rows/columns.
- TATR weights are downloaded separately from `microsoft/table-transformer-structure-recognition`, revision `f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343`. No weights are committed.
- RapidOCR-ONNXRuntime 1.4.4 bundles PP-OCRv4 detection/recognition and a direction classifier. The direction classifier is disabled for the upright dataset samples. We use its CPU ONNX engine, rather than the PaddleOCR Python package. The adapter accepts line boxes, so another OCR backend can replace it later.
- PyMuPDF is used by the upstream postprocessor for rectangle geometry. It has its own AGPL/commercial licensing terms; the upstream MIT license does not replace dependency licenses.
- Sample results are derived from [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m), distributed under CDLA-Permissive-2.0. Dataset terms and sample attribution are included with the examples.
