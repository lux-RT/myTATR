"""Classify saved OCR benchmark errors; do not rerun models or change scores."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

from tatr_ocr.error_analysis import (CATEGORY_LABELS, compare_grids, no_whitespace,
    primary_category, relocation_candidates, span_key, text_grid)

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze(source, cache):
    from tatr_ocr.benchmark import ground_truth, current_cells
    from tatr_ocr.fusion import fuse
    from third_party.tatr.postprocess import iob

    hashes = {}

    def load(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f'Saved input differs from evaluated fingerprint: {path.name}')
        hashes[str(path)] = digest
        return json.loads(data.decode('utf-8-sig'))

    manifest = load(source / 'manifest.json')
    original_summary = load(source / 'current-summary.json')
    config = load(source / 'evaluation-config.json')
    if file_sha(ROOT / 'tatr_ocr/fusion.py') != config['fusion_sha256']:
        raise ValueError('Fusion source differs from the saved experiment; use its original source snapshot')
    records = []
    edit_patterns = Counter()
    legacy_counts = Counter()
    all_grid_counts = Counter()
    unique_pair_counts = Counter()
    for index, sample in enumerate(manifest['samples'], 1):
        name = sample['id']
        evaluated = load(source / 'evaluated' / (name + '.json'))
        fingerprint = evaluated['input_fingerprint']
        words = load(source / 'words' / (name + '.json'), fingerprint['words'])
        ocr = load(source / 'ocr' / (name + '.json'), fingerprint['ocr'])
        prediction = load(source / 'predictions' / (name + '.json'), fingerprint['prediction'])
        xml = Path(sample['xml']).resolve()
        digest = file_sha(xml)
        if digest != fingerprint['xml']:
            raise ValueError('XML changed: ' + name)
        hashes[str(xml)] = digest
        if hashes[str((source / 'evaluation-config.json').resolve())] != fingerprint['configuration']:
            raise ValueError('Evaluation configuration changed: ' + name)
        saved = load(source / 'current' / (name + '.json'))
        if saved['id'] != name or saved['metrics'] != evaluated['current']['metrics']:
            raise ValueError('Export/metric ID or score mismatch: ' + name)
        reference, stats = ground_truth(xml, words)
        for key in ['num_rows', 'num_columns', 'num_cells', 'num_spanning_cells']:
            if stats[key] != evaluated['ground_truth_statistics'][key]:
                raise ValueError('Reference reconstruction differs: ' + name + ' ' + key)
        if evaluated['current']['error'] or ocr['error'] or prediction['current']['error']:
            raise ValueError('This diagnostic expects the original zero-failure experiment')
        fused = fuse(prediction['current']['objects'], ocr['tokens'], .5)
        if current_cells(fused['cells']) != saved['cells']:
            raise ValueError('Reconstructed fusion differs from original export: ' + name)
        metrics = evaluated['current']['metrics']
        topology_exact = metrics['grits_top'] == 1
        comparison = compare_grids(reference, saved['cells'])
        counts = comparison['counts'] if topology_exact else {}
        category = primary_category(topology_exact, counts, comparison['comparable'])
        if metrics['acc_con'] == 1:
            if category != 'exact':
                raise ValueError('Published exact result has unexpected direct-grid differences: ' + name)
        elif category == 'exact':
            category = 'unresolved_alignment'
        ref_keys = Counter(span_key(c) for c in reference)
        pred_keys = Counter(span_key(c) for c in saved['cells'])
        structure = {'reference_rows': stats['num_rows'], 'reference_columns': stats['num_columns'],
            'predicted_rows': saved['rows'], 'predicted_columns': saved['columns'],
            'reference_cells': len(reference), 'predicted_cells': len(saved['cells']),
            'reference_spanning_cells': stats['num_spanning_cells'],
            'predicted_spanning_cells': sum(len(c['row_nums']) * len(c['column_nums']) > 1 for c in saved['cells']),
            'predicted_overlap_writes': comparison['predicted_overlap_writes'],
            'reference_overlap_writes': comparison['reference_overlap_writes'],
            'missing_span_instances': sum((ref_keys - pred_keys).values()),
            'extra_span_instances': sum((pred_keys - ref_keys).values())}
        if not topology_exact:
            structure['subtype'] = ('row_or_column_count' if
                (stats['num_rows'], stats['num_columns']) != (saved['rows'], saved['columns'])
                else 'merged_or_cell_partition')
            # Different topology cannot safely be diagnosed by equal grid indices.
            comparison['differences'] = []
            comparison['counts'] = {}
            comparison['comparable'] = False
        token_owners = {i: ci for ci, cell in enumerate(fused['cells']) for i in cell['token_indices']}
        issues = []
        seen_pairs = set()
        ref_text_counts = Counter(no_whitespace(c['cell_text']) for c in reference)
        ocr_by_text = defaultdict(list)
        for ti, token in enumerate(ocr['tokens']):
            ocr_by_text[no_whitespace(token['text'])].append(ti)
        flags = set()
        for diff in comparison['differences']:
            pair = diff['reference_cell'], diff['predicted_cell']
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            flags.update(diff['flags'])
            unique_pair_counts[diff['kind']] += 1
            ref_cell = reference[pair[0]]
            if diff['kind'] == 'missing':
                nearby = [ti for ti, token in enumerate(ocr['tokens']) if iob(token['bbox'], ref_cell['bbox']) >= .2]
                issues.append({'kind': 'missing_with_nearby_ocr' if nearby else 'missing_without_nearby_ocr',
                    'row': diff['row'], 'column': diff['column'], 'reference': diff['reference'],
                    'reference_bbox': ref_cell['bbox'], 'token_indices': nearby,
                    'nearby_texts': [ocr['tokens'][ti]['text'] for ti in nearby],
                    'assigned_cell_indices': [token_owners.get(ti) for ti in nearby]})
            normalized = no_whitespace(diff['reference'])
            candidates = ocr_by_text[normalized]
            if (diff['kind'] not in ['whitespace'] and len(normalized) >= 3
                    and ref_text_counts[normalized] == 1 and len(candidates) == 1):
                ti = candidates[0]
                owner = token_owners.get(ti)
                if owner is not None and span_key(fused['cells'][owner]) != span_key(ref_cell):
                    issues.append({'kind': 'correct_full_text_assigned_elsewhere_candidate',
                        'row': diff['row'], 'column': diff['column'], 'text': diff['reference'],
                        'token_index': ti, 'assigned_cell_index': owner,
                        'token_overlap_with_reference': iob(ocr['tokens'][ti]['bbox'], ref_cell['bbox'])})
            a, b = no_whitespace(diff['reference']), no_whitespace(diff['prediction'])
            if a and b:
                for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if tag != 'equal' and max(j - i, l - k) <= 12:
                        edit_patterns[(a[i:j], b[k:l])] += 1
        elsewhere = relocation_candidates(reference, saved['cells']) if topology_exact else []
        record = {'index': sample['index'], 'id': name, 'table_type': evaluated['category'],
            'primary_category': category, 'primary_label': CATEGORY_LABELS[category], 'metrics': metrics,
            'structure': structure, 'text_comparison': comparison, 'text_flags': sorted(flags),
            'ocr': {'tokens': len(ocr['tokens']), 'unassigned': len(fused['unassigned_tokens']),
                    'ambiguous': len(fused['ambiguous_token_indices'])},
            'candidate_issues': issues, 'unique_text_elsewhere_candidates': elsewhere}
        records.append(record)
        if topology_exact:
            all_grid_counts.update(counts)
        if topology_exact and evaluated['official']['metrics']['grits_top'] == 1 and evaluated['official']['metrics']['acc_con'] == 1:
            legacy_counts['tables'] += 1
            legacy_counts['grid_positions'] += comparison['reference_shape'][0] * comparison['reference_shape'][1]
            legacy_counts.update({'mismatching_grid_positions': sum(counts.values()),
                'whitespace_only_mismatching_positions': counts.get('whitespace', 0),
                'missing_content_positions': counts.get('missing', 0), 'extra_content_positions': counts.get('extra', 0)})
            if not counts: legacy_counts['strict_exact_tables'] += 1
            if set(counts) <= {'whitespace'}: legacy_counts['equal_after_removing_all_whitespace_tables'] += 1
        if index % 25 == 0:
            print(f'Analyzed {index}/{manifest["count"]}', flush=True)
    if len(records) != manifest['count'] or len({r['id'] for r in records}) != manifest['count']:
        raise ValueError('Manifest coverage mismatch')
    exact = sum(r['metrics']['acc_con'] == 1 for r in records)
    if exact != original_summary['all']['exact_tables']:
        raise ValueError('Original exact-table count mismatch')
    if abs(sum(r['metrics']['grits_con'] for r in records) / len(records) - original_summary['all']['grits_con']) > 1e-12:
        raise ValueError('Original mean content score mismatch')
    print('Checking all source hashes again...', flush=True)
    for path, before in hashes.items():
        if file_sha(path) != before:
            raise ValueError('Source changed during diagnosis: ' + Path(path).name)
    result = {'source_experiment': 'outputs/benchmark-1000-seed42', 'tables': len(records),
        'exact_tables': exact, 'inexact_tables': len(records) - exact,
        'primary_counts': dict(Counter(r['primary_category'] for r in records if r['metrics']['acc_con'] != 1)),
        'topology_exact_tables': sum(r['metrics']['grits_top'] == 1 for r in records),
        'aligned_grid_difference_counts': dict(all_grid_counts),
        'unique_owner_pair_difference_counts': dict(unique_pair_counts),
        'legacy_subset_crosscheck': dict(legacy_counts),
        'common_nonwhitespace_edit_fragments': [{'reference': a, 'prediction': b, 'count': n}
            for (a, b), n in edit_patterns.most_common(50)],
        'source_files_hash_checked': len(hashes), 'source_files_unchanged': True,
        'normalization_is_diagnostic_only': True, 'records': records}
    save_json(cache / 'source-audit.json', hashes)
    save_json(cache / 'analysis-data.json', result)
    return result


def export_analysis(data, out):
    out.mkdir(parents=True, exist_ok=True)
    for record in data['records']:
        record['primary_label'] = CATEGORY_LABELS[record['primary_category']]
    incorrect = [r for r in data['records'] if r['metrics']['acc_con'] != 1]
    data['primary_counts'] = dict(Counter(r['primary_category'] for r in incorrect).most_common())
    aligned = [r for r in data['records'] if r['metrics']['grits_top'] == 1 and r['text_comparison']['comparable']]
    aligned_inexact = [r for r in aligned if r['metrics']['acc_con'] != 1]
    secondary = Counter()
    issue_pairs = Counter()
    clean_bad_cell_counts = []
    collapsed_equal = removed_equal = 0
    for r in aligned:
        diffs = r['text_comparison']['differences']
        collapsed_equal += all(re.sub(r'\s+', ' ', d['reference']).strip() == re.sub(r'\s+', ' ', d['prediction']).strip() for d in diffs)
        removed_equal += all(no_whitespace(d['reference']) == no_whitespace(d['prediction']) for d in diffs)
        if r['metrics']['acc_con'] == 1:
            continue
        secondary.update('has_' + kind for kind in r['text_comparison']['counts'])
        secondary.update(r['text_flags'])
        secondary.update({'has_ambiguous_ocr_boxes': int(r['ocr']['ambiguous'] > 0),
                          'unique_reference_text_elsewhere_candidate': int(bool(r['unique_text_elsewhere_candidates']))})
        for kind in {i['kind'] for i in r['candidate_issues']}:
            secondary[kind] += 1
        issue_pairs.update(i['kind'] for i in r['candidate_issues'])
        strong = [i for i in r['candidate_issues'] if i['kind'] == 'correct_full_text_assigned_elsewhere_candidate' and i['token_overlap_with_reference'] >= .5]
        secondary['full_text_elsewhere_and_reference_overlap_ge_50_percent'] += bool(strong)
        s = r['structure']
        if not (s['reference_overlap_writes'] or s['predicted_overlap_writes'] or
                s['missing_span_instances'] or s['extra_span_instances']):
            clean_bad_cell_counts.append(len({(d['reference_cell'], d['predicted_cell']) for d in diffs}))
    data['diagnostic_observations'] = {
        'aligned_topology_tables': len(aligned), 'aligned_inexact_tables': len(aligned_inexact),
        'secondary_table_counts_within_aligned_inexact': dict(secondary),
        'candidate_owner_pair_counts': dict(issue_pairs),
        'structure_mismatch_subtypes': dict(Counter(r['structure']['subtype'] for r in incorrect if r['primary_category'] == 'structure_mismatch')),
        'structure_mismatch_row_count_tables': sum(r['structure']['reference_rows'] != r['structure']['predicted_rows'] for r in incorrect if r['primary_category'] == 'structure_mismatch'),
        'structure_mismatch_column_count_tables': sum(r['structure']['reference_columns'] != r['structure']['predicted_columns'] for r in incorrect if r['primary_category'] == 'structure_mismatch'),
        'clean_aligned_inexact_tables': len(clean_bad_cell_counts),
        'clean_aligned_only_one_bad_cell_tables': sum(n == 1 for n in clean_bad_cell_counts),
        'clean_aligned_at_most_three_bad_cells_tables': sum(n <= 3 for n in clean_bad_cell_counts),
        'clean_aligned_median_bad_cells': statistics.median(clean_bad_cell_counts) if clean_bad_cell_counts else None,
        'diagnostic_equal_after_collapsing_whitespace_tables': collapsed_equal,
        'diagnostic_equal_after_removing_all_whitespace_tables': removed_equal,
        'note': 'Whitespace-normalized equality is not a newly computed AccCon. Secondary counts overlap. Missing/extra means empty/nonempty, not every deleted/inserted character.'}
    data['primary_counts_by_table_type'] = {kind: dict(Counter(r['primary_category'] for r in incorrect
        if r['table_type'] == kind)) for kind in ['simple', 'complex']}
    data['edit_fragment_count_note'] = 'Counts are short non-whitespace edit fragments, not numbers of tables or independently verified OCR errors.'
    save_json(out / 'incorrect-tables.json', incorrect)
    save_json(out / 'summary.json', {k: v for k, v in data.items() if k != 'records'})
    counts = data['primary_counts']
    table = '\n'.join(f'| {CATEGORY_LABELS[k]} | {v} | {100*v/len(incorrect):.2f}% |' for k, v in counts.items())
    findings_link = '[分析结论与代表性原图核查](FINDINGS.md)\n' if (out / 'FINDINGS.md').is_file() else ''
    report = f'''# 原 OCR 千图结果：全部未完全一致表格的错误分类

{findings_link}

分析原 `outputs/benchmark-1000-seed42` 实验：{data['tables']} 张、完全一致 {data['exact_tables']} 张，未完全一致 **{len(incorrect)} 张**。模型预测、OCR、原评分及原导出未修改；参考文本是本实验使用的官方词信息，不宣称逐字绝对正确。

## 主分类（互斥）

| 分类 | 表格数 | 占未完全一致表格 |
|---|---:|---:|
{table}

主分类先区分结构拓扑是否完全匹配，再按文字差异分类。文字类别中“可含空白”表示允许同时存在空格/换行差异；“混合”表示存在两种以上非空白差异。符号形式只比较明确列出的横线、减号和弯引号，不能据此断言数学含义相同。全部正规评价分数保持原样。

“整格文本缺失”专指参考非空、预测为空；“参考空单元格出现文本”专指参考单元格为空、预测非空。非空文本中的部分漏字、多字、错字仍归到其他内容差异，不能把前两项计数解释成所有漏字/多字总量。

## 阅读方式与限制

- [全部 {len(incorrect)} 张逐表分类清单](classification.md)；[本地筛选查看页面](index.html)；[完整逐表证据 JSON](incorrect-tables.json)；[汇总 JSON](summary.json)。
- 文字位置统计使用与官方评分相同的网格投影：合并格会覆盖多个位置，重叠单元格按原顺序最后一次写入。网格位置数、唯一单元格对数和表格数分别统计，不能混称单元格错误率。
- 结构拓扑不一致时，不强行把相同行列下标视为同一个真实单元格；此类表格记录结构跨度差异和框问题，文字根因仍需看图。
- “正确文本出现在其他位置”“附近存在 OCR 框”“跨界歧义框”均为候选证据，不能自动等同于已确认的错配、漏检或识错。
- 本次分析用于发现问题，不能把诊断字符串归一化后的相等数量作为新的 AccCon，也不据此调改本轮正式测试指标。
- 已对 {data['source_files_hash_checked']} 个源文件做前后 SHA-256 检查，均未改变。分析未运行 TATR/OCR 模型，也未重新计算官方 GriTS。

本地页面中的 PNG 链接需要原 `outputs` 目录；GitHub 上可按表格 ID 在已发布的千图含 PNG 结果包中查找。
'''
    (out / 'README.md').write_text(report, encoding='utf-8', newline='\n')
    rows = ['# 全部未完全一致表格的主分类', '',
        '| 原名单序号 | 表格 ID | 主分类 | 次级文字标签 | 结构分数 | 内容分数 | 差异网格位置数 |',
        '|---:|---|---|---|---:|---:|---:|']
    for r in incorrect:
        rows.append(f'| {r["index"]} | {r["id"]} | {r["primary_label"]} | {", ".join(r["text_flags"])} | '
                    f'{r["metrics"]["grits_top"]:.5f} | {r["metrics"]["grits_con"]:.5f} | '
                    f'{sum(r["text_comparison"]["counts"].values()) if r["text_comparison"]["comparable"] else "不可直接对齐"} |')
    (out / 'classification.md').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    html_rows = []
    for r in incorrect:
        differences = sorted(r['text_comparison']['differences'], key=lambda d: d['kind'] == 'whitespace')
        examples = []
        seen = set()
        for d in differences:
            pair = d['reference_cell'], d['predicted_cell']
            if pair not in seen:
                examples.append(f'<tr><td>R{d["row"]} C{d["column"]}</td><td>{html.escape(d["reference"])}</td><td>{html.escape(d["prediction"])}</td><td>{d["kind"]}</td></tr>')
                seen.add(pair)
            if len(examples) == 8: break
        detail = '<table><tr><th>位置</th><th>参考文本</th><th>当前文本</th><th>差异类型</th></tr>' + ''.join(examples) + '</table>' if examples else '结构不一致，未强行逐格配对。'
        png = '../../../outputs/benchmark-1000-seed42/current/' + r['id'] + '_overlay.png'
        html_rows.append(f'<tr data-category="{r["primary_category"]}"><td>{r["index"]}</td><td>{r["id"]}</td><td>{r["primary_label"]}</td><td>{r["metrics"]["grits_top"]:.5f}</td><td>{r["metrics"]["grits_con"]:.5f}</td><td><details><summary>差异示例与结构图</summary><a href="{png}">当前 PNG</a><p>{html.escape(str(r["structure"]))}</p>{detail}</details></td></tr>')
    options = ''.join(f'<option value="{k}">{CATEGORY_LABELS[k]} ({v})</option>' for k, v in counts.items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OCR 未完全一致表格分类</title>
<style>body{font:16px/1.6 system-ui;margin:24px;color:#203448}table{border-collapse:collapse;width:100%}th,td{border:1px solid #becad3;padding:8px;vertical-align:top}th{background:#e9eff4}td table td{white-space:pre-wrap}input,select{font:inherit;padding:6px;margin:8px}summary{cursor:pointer}</style>
<h1>''' + str(len(incorrect)) + ''' 张未完全一致表格：逐表检查</h1><p>分类是相对参考文本的观察，不等于所有问题的视觉根因已确认。完整差异见 incorrect-tables.json。分数为原始分数。</p>
<input id="search" placeholder="搜索表格 ID"><select id="category"><option value="">全部分类</option>''' + options + '''</select><span id="count"></span>
<table id="results"><thead><tr><th>名单序号</th><th>表格 ID</th><th>主分类</th><th>GriTS Top</th><th>GriTS Con</th><th>检查</th></tr></thead><tbody>''' + ''.join(html_rows) + '''</tbody></table>
<script>const search=document.getElementById('search'),category=document.getElementById('category');function filter(){let n=0;document.querySelectorAll('#results > tbody > tr').forEach(r=>{const show=r.cells[1].textContent.toLowerCase().includes(search.value.toLowerCase())&&(!category.value||r.dataset.category===category.value);r.hidden=!show;if(show)n++;});document.getElementById('count').textContent=n+' 张';}search.oninput=filter;category.onchange=filter;filter();</script></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/benchmark-1000-seed42')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/benchmark-1000/error-analysis-20261116')
    parser.add_argument('--cache', type=Path, default=ROOT / 'outputs/error-analysis-1000-20261116')
    args = parser.parse_args()
    source = args.source.resolve()
    for target in [args.out.resolve(), args.cache.resolve()]:
        if target == source or source in target.parents:
            parser.error('Analysis report/cache must stay outside the original experiment directory')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose a new analysis output directory to preserve prior reports')
    data = analyze(source, args.cache.resolve())
    export_analysis(data, args.out.resolve())
    print(json.dumps(data['primary_counts'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
