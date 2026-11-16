"""Read-only observations about saved benchmark errors, never new scores."""
from collections import Counter


CATEGORY_LABELS = {
    'exact': '整表内容完全一致',
    'structure_mismatch': '结构拓扑不一致（可伴文字问题）',
    'whitespace_only': '仅空格或换行差异',
    'symbol_variant_only': '仅横线或引号字形差异（可含空白）',
    'missing_only': '仅整格文本缺失（可含空白差异）',
    'extra_only': '仅参考空单元格出现文本（可含空白差异）',
    'content_only': '其他文字内容差异（可含空白）',
    'mixed_text': '多种文字差异混合',
    'unresolved_alignment': '无法直接对齐，需复核',
}

# Diagnostic grouping only. A mathematical minus and a range hyphen are not
# asserted to have identical meaning. Never apply this map to published scores.
SYMBOL_MAP = str.maketrans({c: '-' for c in '‐‑‒–—−'} | {
    '‘': "'", '’': "'", '“': '"', '”': '"'})


def no_whitespace(text):
    return ''.join(c for c in text if not c.isspace())


def text_difference(reference, prediction):
    if reference == prediction:
        return 'equal'
    ref, pred = no_whitespace(reference), no_whitespace(prediction)
    if ref == pred:
        return 'whitespace'
    if not pred:
        return 'missing'
    if not ref:
        return 'extra'
    if ref.translate(SYMBOL_MAP) == pred.translate(SYMBOL_MAP):
        return 'symbol_variant'
    return 'content'


def text_flags(reference, prediction):
    ref, pred = no_whitespace(reference), no_whitespace(prediction)
    flags = []
    if ''.join(c for c in ref if c.isdigit()) != ''.join(c for c in pred if c.isdigit()):
        flags.append('digits_changed')
    if ''.join(c for c in ref if c.isalpha()) != ''.join(c for c in pred if c.isalpha()):
        flags.append('letters_changed')
    if ''.join(c for c in ref if not c.isalnum()) != ''.join(c for c in pred if not c.isalnum()):
        flags.append('punctuation_changed')
    if ref and not pred and not any(c.isalnum() for c in ref):
        flags.append('missing_punctuation_only')
    return flags


def primary_category(topology_exact, counts, comparable=True):
    if not topology_exact:
        return 'structure_mismatch'
    if not comparable:
        return 'unresolved_alignment'
    kinds = {key for key, value in counts.items() if value and key != 'equal'}
    if not kinds:
        return 'exact'
    if kinds == {'whitespace'}:
        return 'whitespace_only'
    kinds.discard('whitespace')
    if len(kinds) > 1:
        return 'mixed_text'
    return {'symbol_variant': 'symbol_variant_only', 'missing': 'missing_only',
            'extra': 'extra_only', 'content': 'content_only'}[next(iter(kinds))]


def span_key(cell):
    return tuple(sorted(cell['row_nums'])), tuple(sorted(cell['column_nums']))


def text_grid(cells):
    """Match official grid projection, including last-cell-wins overlaps/0 holes."""
    if not cells:
        return [[]], [[]], 0
    rows = max(max(c['row_nums']) for c in cells) + 1
    columns = max(max(c['column_nums']) for c in cells) + 1
    grid = [[0.0] * columns for _ in range(rows)]
    owners = [[None] * columns for _ in range(rows)]
    overlaps = 0
    for index, cell in enumerate(cells):
        for row in cell['row_nums']:
            for column in cell['column_nums']:
                overlaps += owners[row][column] is not None
                grid[row][column] = cell['cell_text']
                owners[row][column] = index
    return grid, owners, overlaps


def compare_grids(reference, prediction):
    ref, ref_owners, ref_overlaps = text_grid(reference)
    pred, pred_owners, pred_overlaps = text_grid(prediction)
    shape_ref = [len(ref), len(ref[0])]
    shape_pred = [len(pred), len(pred[0])]
    result = {'reference_shape': shape_ref, 'predicted_shape': shape_pred,
              'reference_overlap_writes': ref_overlaps, 'predicted_overlap_writes': pred_overlaps,
              'comparable': shape_ref == shape_pred, 'counts': {}, 'differences': []}
    if not result['comparable']:
        return result
    counts = Counter()
    for r, (ref_row, pred_row) in enumerate(zip(ref, pred)):
        for c, (expected, actual) in enumerate(zip(ref_row, pred_row)):
            if not isinstance(expected, str) or not isinstance(actual, str):
                result['comparable'] = False
                continue
            kind = text_difference(expected, actual)
            if kind == 'equal':
                continue
            counts[kind] += 1
            result['differences'].append({'row': r + 1, 'column': c + 1, 'kind': kind,
                'reference_cell': ref_owners[r][c], 'predicted_cell': pred_owners[r][c],
                'reference': expected, 'prediction': actual, 'flags': text_flags(expected, actual)})
    result['counts'] = dict(counts)
    return result


def relocation_candidates(reference, prediction):
    """Unique full-cell text found elsewhere is evidence to inspect, not proof."""
    ref_counts = Counter(no_whitespace(c['cell_text']) for c in reference)
    pred_counts = Counter(no_whitespace(c['cell_text']) for c in prediction)
    spans = Counter(span_key(c) for c in prediction)
    pred_by_span = {span_key(c): c for c in prediction}
    candidates = []
    for i, cell in enumerate(reference):
        text = no_whitespace(cell['cell_text'])
        key = span_key(cell)
        if len(text) < 3 or ref_counts[text] != 1 or pred_counts[text] != 1 or spans[key] != 1:
            continue
        if no_whitespace(pred_by_span[key]['cell_text']) == text:
            continue
        other = next((j for j, p in enumerate(prediction)
                      if span_key(p) != key and no_whitespace(p['cell_text']) == text), None)
        if other is not None:
            candidates.append({'reference_cell': i, 'found_in_predicted_cell': other,
                               'text': cell['cell_text'], 'evidence': 'unique_full_text_elsewhere'})
    return candidates
