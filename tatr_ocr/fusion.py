"""Adapt Transformers objects and OCR tokens to Microsoft's cell postprocessor."""
from copy import deepcopy
import math

from third_party.tatr import postprocess as pp

LABELS = ("table", "table column", "table row", "table column header",
          "table projected row header", "table spanning cell")


def _valid_box(box):
    return (len(box) == 4 and all(math.isfinite(x) for x in box)
            and box[2] > box[0] and box[3] > box[1])


def _cell_text(tokens):
    """Use geometry for line order, preserving line breaks within a cell."""
    lines = []
    for token in sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0])):
        box = token["bbox"]
        cy = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        line = next((line for line in lines if abs(line[0] - cy) <= .5 * min(line[1], height)), None)
        if line is None:
            line = [cy, height, []]
            lines.append(line)
        line[2].append(token)
    return "\n".join(" ".join(t["text"] for t in sorted(line[2], key=lambda t: t["bbox"][0]))
                     for line in sorted(lines, key=lambda line: line[0]))


def fuse(objects, tokens, threshold=0.5):
    """Reconstruct one cropped table; never use ground-truth annotations.

    Build structure without OCR first: missing OCR must not prune blank rows.
    Then assign tokens uniquely with >=50% token-area coverage by a cell.
    Unassigned and geometrically ambiguous tokens remain available for audit.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    filtered = [deepcopy(o) for o in objects if o["label"] in LABELS
                and o["score"] >= threshold and _valid_box(o["bbox"])]
    tables = [o for o in filtered if o["label"] == "table"]
    if not tables:
        raise ValueError("No valid table detected; use a cropped table image or lower the threshold.")
    if len(tables) != 1:
        raise ValueError("Multiple tables detected; supply one cropped table image at a time.")
    table = tables[0]
    table["page_num"] = 0
    inside = [o for o in filtered if pp.iob(o["bbox"], table["bbox"]) >= .5]
    for o in inside:
        o["label"] = LABELS.index(o["label"])
    thresholds = {label: threshold for label in LABELS}
    structure, raw_cells, _ = pp.objects_to_cells(table, inside, [], LABELS, thresholds)
    if not raw_cells:
        raise ValueError("Detected table has no usable rows or columns.")
    for token in tokens:
        if not _valid_box(token["bbox"]):
            raise ValueError("OCR token has an invalid bbox")
    assignments, token_cells, scores = pp.slot_into_containers(
        raw_cells, tokens, overlap_threshold=.5, unique_assignment=True, forced_assignment=False)
    cells = []
    for raw, indices in zip(raw_cells, assignments):
        spans = [tokens[i] for i in indices]
        cells.append({"bbox": raw["bbox"], "row_nums": sorted(raw["row_nums"]),
                      "column_nums": sorted(raw["column_nums"]), "text": _cell_text(spans),
                      "column header": raw["header"], "projected row header": raw["subheader"],
                      "token_indices": indices,
                      "ocr_confidence": sum(t["confidence"] for t in spans) / len(spans) if spans else None})
    cells.sort(key=lambda c: (min(c["row_nums"]), min(c["column_nums"])))
    ambiguous = [i for i, token in enumerate(tokens)
                 if sum(pp.iob(token["bbox"], c["bbox"]) >= .2 for c in cells) > 1]
    return {"rows": len(structure["rows"]), "columns": len(structure["columns"]),
            "bbox": table["bbox"], "cells": cells,
            "assigned_tokens": sum(bool(c) for c in token_cells),
            "unassigned_tokens": [t for t, c in zip(tokens, token_cells) if not c],
            "ambiguous_token_indices": ambiguous,
            "mean_token_cell_overlap": sum(scores) / len(scores) if scores else None}
