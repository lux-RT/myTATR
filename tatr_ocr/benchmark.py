"""Frozen sampling and adapters to the unmodified Microsoft evaluation functions."""
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stat
import xml.etree.ElementTree as ET

import numpy as np
from third_party.tatr import grits, postprocess

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('acc_con', 'grits_top', 'grits_con', 'grits_loc')
LABELS = ['table', 'table column', 'table row', 'table column header',
          'table projected row header', 'table spanning cell', 'no object']
CLASS_MAP = {name:i for i,name in enumerate(LABELS)}
THRESHOLDS = {name:(10 if name=='no object' else .5) for name in LABELS}

# PyMuPDF 1.26 no longer accepts numpy integer scalars in Rect(). Convert only
# the scalar types, then execute the original upstream IoU calculation.
_upstream_iou = grits.iou
def _numpy_safe_iou(a, b):
    return _upstream_iou([float(v) for v in a], [float(v) for v in b])
grits.iou = _numpy_safe_iou


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def sample_ids(population, count, seed):
    population=list(population)
    if len(population)!=len(set(population)) or not 0<count<=len(population):
        raise ValueError('Sampling requires unique IDs and a valid sample count')
    return random.Random(seed).sample(sorted(population),count)


def check_config(actual, expected):
    if actual!=expected:raise ValueError('Cache configuration mismatch; choose a new output directory')


@lru_cache(maxsize=1)
def official_functions():
    """Load only pure evaluator functions, avoiding DETR training dependencies.

    The complete upstream file remains byte-identical and hash-verified. Its
    function ASTs execute unchanged with their original numerical dependencies.
    """
    provenance=json.loads((ROOT/'third_party/tatr/provenance.json').read_text(encoding='utf-8-sig'))
    for name in ['eval.py','grits.py','postprocess.py']:
        if digest(ROOT/'third_party/tatr'/name)!=provenance['files'][name]['sha256']:
            raise ValueError(f'Official source hash mismatch: {name}')
    file=ROOT/'third_party/tatr/eval.py'
    tree=ast.parse(file.read_text(encoding='utf-8'))
    names={'objects_to_cells','compute_metrics','compute_statistics'}
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in functions}!=names:raise ValueError('Official evaluation API changed')
    namespace={'np':np,'grits':grits,'postprocess':postprocess,'stat':stat,
               'grits_top':grits.grits_top,'grits_loc':grits.grits_loc,'grits_con':grits.grits_con}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(file),'exec'),namespace)
    return namespace


def read_objects(xml_path):
    root=ET.parse(xml_path).getroot()
    boxes=[];labels=[]
    for obj in root.findall('object'):
        label=CLASS_MAP[obj.findtext('name')]
        box=[float(obj.findtext('bndbox/'+key)) for key in ['xmin','ymin','xmax','ymax']]
        if not all(math.isfinite(x) for x in box) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError(f'Invalid annotation box: {xml_path}')
        boxes.append(box);labels.append(label)
    return boxes,labels


def official_cells(boxes, labels, scores, words):
    return official_functions()['objects_to_cells'](
        deepcopy(boxes),list(labels),list(scores),deepcopy(words),LABELS,THRESHOLDS,CLASS_MAP)


def ground_truth(xml_path, words):
    boxes,labels=read_objects(xml_path)
    structure,cells,_=official_cells(boxes,labels,[1.0]*len(labels),words)
    if not cells:raise ValueError(f'Empty ground truth: {xml_path}')
    stats=official_functions()['compute_statistics'](structure,cells)
    return cells,stats


def current_cells(cells):
    return [dict(row_nums=list(c['row_nums']),column_nums=list(c['column_nums']),
                 bbox=list(c['bbox']),cell_text=c['text'],header=c['column header'],
                 subheader=c['projected row header']) for c in cells]


def official_export_table(structure, cells):
    return {'rows':len(structure['rows']),'columns':len(structure['columns']),
            'cells':[{**deepcopy(c),'text':c['cell_text'],'column header':c['header'],
                      'projected row header':c['subheader']} for c in cells]}


def score_cells(truth, prediction):
    if not truth:raise ValueError('Empty ground truth is not an evaluable reference')
    if not prediction:return {name:0.0 for name in METRICS}
    metrics=official_functions()['compute_metrics']('grits',[],[],[],truth,[],[],[],prediction)
    selected={name:float(metrics[name]) for name in METRICS}
    if not all(math.isfinite(v) and 0<=v<=1 for v in selected.values()):
        raise ValueError(f'Invalid metric values: {selected}')
    return selected


def summarize(records, expected_count):
    if len(records)!=expected_count or len({r['id'] for r in records})!=expected_count:
        raise ValueError('Summary must contain every unique selected table exactly once')
    summaries={}
    for category in ['all','simple','complex']:
        rows=[r for r in records if category=='all' or r['category']==category]
        summary={'num_tables':len(rows),'failures':sum(bool(r.get('error')) for r in rows)}
        if rows:
            values=np.array([[r['metrics'][k] for k in METRICS] for r in rows],dtype=float)
            if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
                raise ValueError('Missing or invalid scores in summary')
            summary.update(dict(zip(METRICS,values.mean(axis=0).tolist())))
            summary['exact_tables']=int(values[:,0].sum())
            # Resample whole articles to account for multiple tables in a paper.
            groups={}
            for r,value in zip(rows,values):groups.setdefault(r['id'].split('_table_')[0],[]).append(value)
            sums=np.array([np.sum(v,axis=0) for v in groups.values()])
            counts=np.array([len(v) for v in groups.values()])
            rng=np.random.default_rng(42)
            weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
            boot=(weights@sums)/(weights@counts)[:,None]
            intervals=np.quantile(boot,[.025,.975],axis=0)
            summary['ci95_article_bootstrap']={k:[float(intervals[0,i]),float(intervals[1,i])] for i,k in enumerate(METRICS)}
            summary['num_articles']=len(groups)
        summaries[category]=summary
    return summaries
