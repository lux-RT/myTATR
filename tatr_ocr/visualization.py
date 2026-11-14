"""Draw saved cell/token geometry without running models or changing results."""
import math
from pathlib import Path

from PIL import ImageDraw


def draw_cells(image, cells, tokens, path, error=None):
    """Keep original image coordinates; purple marks spanning cells."""
    canvas = image.convert('RGB')
    draw = ImageDraw.Draw(canvas)

    def rectangle(box, color, width):
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] < box[0] or box[3] < box[1]):
            raise ValueError('Cannot render an invalid bounding box')
        draw.rectangle(box, outline=color, width=width)

    for cell in cells:
        merged = len(cell['row_nums']) * len(cell['column_nums']) > 1
        rectangle(cell['bbox'], '#bb2588' if merged else '#176fba', 2)
    for token in tokens:
        rectangle(token['bbox'], '#16a070', 1)
    if error:
        draw.rectangle([0, 0, canvas.width - 1, min(20, canvas.height - 1)], fill='#b40000')
        draw.text((3, 3), 'FAILED: see JSON error', fill='white')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    canvas.save(temporary, format='PNG')
    temporary.replace(path)
