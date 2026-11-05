"""Human-readable exports from a cell grid, retaining merge information."""
import csv
from html import escape
import io


def table_html(table):
    anchors = {(min(c["row_nums"]), min(c["column_nums"])): c for c in table["cells"]}
    covered = {(r, col) for c in table["cells"] for r in c["row_nums"] for col in c["column_nums"]}
    parts = ['<table>']
    for row in range(table["rows"]):
        parts.append('<tr>')
        for col in range(table["columns"]):
            cell = anchors.get((row, col))
            if cell is None:
                if (row, col) not in covered:
                    parts.append('<td></td>')
                continue
            tag = 'th' if cell["column header"] else 'td'
            attrs = f' rowspan="{len(cell["row_nums"])}" colspan="{len(cell["column_nums"])}"'
            parts.append(f'<{tag}{attrs}>{escape(cell["text"]).replace(chr(10), "<br>")}</{tag}>')
        parts.append('</tr>')
    parts.append('</table>')
    return '\n'.join(parts)


def table_csv(table):
    """CSV has no merges: write merged-cell text at its top-left anchor only."""
    grid = [[""] * table["columns"] for _ in range(table["rows"])]
    for cell in table["cells"]:
        grid[min(cell["row_nums"])][min(cell["column_nums"])] = cell["text"]
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(grid)
    return stream.getvalue()


def write_csv(path, table):
    # csv.writer already emits CRLF. Disable Windows' extra newline translation.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        stream.write(table_csv(table))


def html_document(title, table):
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<title>{escape(title)}</title><style>'
            'body{font:15px system-ui;margin:32px;color:#183047}table{border-collapse:collapse}'
            'td,th{border:1px solid #aab9c5;padding:7px 12px;vertical-align:top}'
            'th{background:#e8eff5}h1{font-size:21px}</style>'
            f'<h1>{escape(title)}</h1>{table_html(table)}</html>')
