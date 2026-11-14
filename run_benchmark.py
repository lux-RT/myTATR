"""Reproducible paired PubTables-1M benchmark; every stage is resumable."""
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from time import perf_counter, sleep

ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=ROOT.parent/'PubTables-1M复现/datasets/PubTables-1M-structure'
DEFAULT_OUT=ROOT/'outputs/benchmark-1000-seed42'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def emit(message):print(message,flush=True)


def prepare(args):
    from tatr_ocr.benchmark import sample_ids,check_config
    filelist=args.data/'PubTables-1M-Structure_Filelists/test_filelist.txt'
    ids=[Path(line.strip()).stem for line in filelist.read_text().splitlines() if line.strip()]
    if len(ids)!=93834:raise ValueError('Expected the full official 93,834-table test population')
    chosen=sample_ids(ids,args.count,args.seed)
    image_dir=args.data/'PubTables-1M-Structure_Images_Test/PubTables-1M-Structure_Images_Test'
    xml_dir=args.data/'PubTables-1M-Structure_Annotations_Test'
    samples=[]
    for index,name in enumerate(chosen):
        image=image_dir/(name+'.jpg');xml=xml_dir/(name+'.xml')
        if not image.is_file() or not xml.is_file():raise FileNotFoundError(name)
        samples.append({'index':index+1,'id':name,'image':str(image.resolve()),'xml':str(xml.resolve()),
                        'image_sha256':sha(image),'xml_sha256':sha(xml)})
    manifest={'dataset':'bsmock/pubtables-1m','dataset_revision':'35b1c097807e0b07ec5313879b85956b7b3890db',
              'split':'test','population':len(ids),'count':args.count,'seed':args.seed,
              'sampling':'random.Random(seed).sample(sorted(official_test_ids), count), without replacement',
              'filelist_sha256':sha(filelist),'samples':samples}
    target=args.out/'manifest.json'
    if target.exists():check_config(read_json(target),manifest)
    else:write_json(target,manifest)
    emit(f'Frozen {len(samples)} / {len(ids)} test tables; seed={args.seed}; manifest={sha(target)}')


def verify_manifest(out):
    manifest=read_json(out/'manifest.json')
    if len(manifest['samples'])!=manifest['count'] or len({s['id'] for s in manifest['samples']})!=manifest['count']:
        raise ValueError('Manifest sample count/uniqueness mismatch')
    for sample in manifest['samples']:
        for kind in ['image','xml']:
            if sha(sample[kind])!=sample[kind+'_sha256']:raise ValueError(f'Input changed: {sample["id"]} {kind}')
    return manifest


def stage_config(args, stage, extra):
    from tatr_ocr.benchmark import check_config
    config={'stage':stage,'manifest_sha256':sha(args.out/'manifest.json'),**extra}
    path=args.out/(stage+'-config.json')
    if path.exists():check_config(read_json(path),config)
    else:write_json(path,config)
    return config


def words(args):
    manifest=verify_manifest(args.out)
    archive=args.data/'PubTables-1M-Structure_Table_Words.tar.gz'
    expected='dbab74fa5fe5354cca229533efe7e374b58dfb3b352c07efcd84bb69169232f9'
    if not archive.is_file() or archive.stat().st_size!=4174842415:raise ValueError('Official words archive is incomplete')
    emit('Verifying official word archive SHA-256...')
    if sha(archive)!=expected:raise ValueError('Official words archive SHA-256 mismatch')
    output=args.out/'words';output.mkdir(exist_ok=True)
    selected={s['id'] for s in manifest['samples']};found={}
    emit('Extracting only the selected word JSONs...')
    with tarfile.open(archive,'r|gz') as tar:
        for entry in tar:
            # Streaming mode on Python 3.12 still caches TarInfo entries. The
            # archive has about a million files; no random lookup is needed.
            if len(tar.members)>1000:tar.members.clear()
            if not entry.isfile():continue
            name=Path(entry.name).name
            if not name.endswith('_words.json'):continue
            table=name.removesuffix('_words.json')
            if table not in selected:continue
            if table in found:raise ValueError(f'Duplicate words entry: {table}')
            data=tar.extractfile(entry).read()
            tokens=json.loads(data)
            if not isinstance(tokens,list):raise ValueError(f'Invalid word data: {table}')
            # Destination derives only from a selected safe basename, never tar paths.
            destination=output/(table+'.json')
            if destination.exists() and destination.read_bytes()!=data:raise ValueError('Existing word data differs')
            destination.write_bytes(data);found[table]=hashlib.sha256(data).hexdigest()
            if len(found)%100==0:emit(f'Word data {len(found)}/{len(selected)}')
    if set(found)!=selected:raise ValueError(f'Missing word data: {sorted(selected-set(found))}')
    write_json(args.out/'words-provenance.json',{'archive_sha256':expected,'count':len(found),'files':found})
    emit('All selected word data extracted and verified')


def predict(args):
    from importlib.metadata import version
    from tatr_ocr.inputs import default_model_dir
    manifest=verify_manifest(args.out)
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,TableTransformerForObjectDetection
    model_dir=default_model_dir(ROOT)
    config=stage_config(args,'predict',{'model_id':'microsoft/table-transformer-structure-recognition',
        'model_revision':'f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343',
        'files':{n:sha(model_dir/n) for n in ['config.json','model.safetensors','preprocessor_config.json']},
        'torch':torch.__version__,'transformers':version('transformers'),'pillow':version('pillow'),
        'threads':args.threads,'threshold':.5,'source_sha256':sha(__file__),
        'official_resize':'PIL.Image.resize(round(width*1000/max_edge), round(height*1000/max_edge)); default resample',
        'current_resize':'existing local AutoImageProcessor settings'})
    torch.set_num_threads(args.threads)
    processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=False)
    model=TableTransformerForObjectDetection.from_pretrained(model_dir,local_files_only=True,use_pretrained_backbone=False).eval()
    directory=args.out/'predictions';directory.mkdir(exist_ok=True)
    for sample in manifest['samples']:
        destination=directory/(sample['id']+'.json')
        if destination.exists():continue
        with Image.open(sample['image']) as raw:image=raw.convert('RGB')
        result={'id':sample['id'],'image_size':list(image.size)}
        for mode in ['official','current']:
            start=perf_counter()
            try:
                if mode=='official':
                    scale=1000/max(image.size)
                    resized=image.resize((int(round(scale*image.width)),int(round(scale*image.height))))
                    inputs=processor(images=resized,do_resize=False,return_tensors='pt')
                else:inputs=processor(images=image,return_tensors='pt')
                with torch.inference_mode():output=model(**inputs)
                if not torch.isfinite(output.logits).all() or not torch.isfinite(output.pred_boxes).all():
                    raise ValueError('Non-finite model prediction')
                if mode=='official':
                    scores,labels=output.logits.softmax(-1).max(-1)
                    cx,cy,w,h=output.pred_boxes[0].unbind(-1)
                    boxes=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],dim=-1)*torch.tensor([image.width,image.height,image.width,image.height])
                    result[mode]={'boxes':boxes.tolist(),'labels':labels[0].tolist(),'scores':scores[0].tolist()}
                else:
                    detected=processor.post_process_object_detection(output,threshold=.5,target_sizes=[(image.height,image.width)])[0]
                    result[mode]={'objects':[{'label':model.config.id2label[int(label)],'score':float(score),'bbox':box.tolist()}
                                            for label,score,box in zip(detected['labels'],detected['scores'],detected['boxes'])]}
                result[mode]['error']=None
            except Exception as error:result[mode]={'error':f'{type(error).__name__}: {error}'}
            result[mode]['seconds']=perf_counter()-start
        write_json(destination,result)
        if sample['index']%10==0:emit(f'Structure predictions {sample["index"]}/{manifest["count"]}')
    emit('Both structure prediction sets completed')


def ocr(args):
    manifest=verify_manifest(args.out)
    python=ROOT/'.venv-ocr-v6/Scripts/python.exe'
    if os.name!='nt':python=ROOT/'.venv-ocr-v6/bin/python'
    stage_config(args,'ocr',{'worker_sha256':sha(ROOT/'ocr_worker.py'),
        'adapter_sha256':sha(ROOT/'tatr_ocr/ocr_adapter.py'),'scale':2.0,'threads':args.threads,
        'models':{p.name:sha(p) for p in sorted((ROOT/'models/ocr-v6').glob('*.onnx'))},'source_sha256':sha(__file__)})
    output=args.out/'ocr';output.mkdir(exist_ok=True)
    batches=args.out/'ocr-batches';batches.mkdir(exist_ok=True)
    pending=[s for s in manifest['samples'] if not (output/(s['id']+'.json')).exists()]
    def run_batch(samples,tag):
        result=batches/(tag+'.json');log=batches/(tag+'.log')
        command=[str(python),'-u','-X','utf8',str(ROOT/'ocr_worker.py'),'--images',*[s['image'] for s in samples],
                 '--output',str(result),'--scale','2.0','--threads',str(args.threads)]
        with log.open('w',encoding='utf-8') as stream:
            try:
                completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
                failed=completed.returncode!=0
            except subprocess.TimeoutExpired:
                failed=True
                stream.write('\nOCR timeout after 1800 seconds.\n')
        if failed:
            if len(samples)>1:
                for sample in samples:run_batch([sample],sample['id'])
            else:write_json(output/(samples[0]['id']+'.json'),{'id':samples[0]['id'],'error':log.read_text(encoding='utf-8')[-5000:]})
            return
        data=read_json(result)
        metadata={k:v for k,v in data.items() if k!='images'}
        meta_path=args.out/'ocr-metadata.json'
        if meta_path.exists():
            from tatr_ocr.benchmark import check_config
            check_config(read_json(meta_path),metadata)
        else:write_json(meta_path,metadata)
        records={Path(r['image']).stem:r for r in data['images']}
        if set(records)!={s['id'] for s in samples}:raise ValueError('OCR batch IDs do not match selected inputs')
        for sample in samples:write_json(output/(sample['id']+'.json'),{'id':sample['id'],'error':None,**records[sample['id']]})
    for offset in range(0,len(pending),25):
        subset=pending[offset:offset+25]
        run_batch(subset,f'batch-{subset[0]["index"]:04}-{subset[-1]["index"]:04}')
        done=len(list(output.glob('*.json')))
        emit(f'OCR outputs {done}/{manifest["count"]}')
    emit('OCR inference completed')


def evaluation_fingerprint(sample,out):
    out=Path(out);name=sample['id']
    paths={'xml':Path(sample['xml']),'words':out/'words'/(name+'.json'),
           'prediction':out/'predictions'/(name+'.json'),'ocr':out/'ocr'/(name+'.json'),
           'configuration':out/'evaluation-config.json'}
    return {key:sha(path) for key,path in paths.items()}


def evaluation_is_current(sample,out):
    record=Path(out)/'evaluated'/(sample['id']+'.json')
    if not record.is_file():return False
    try:return read_json(record).get('input_fingerprint')==evaluation_fingerprint(sample,out)
    except FileNotFoundError:return False


def evaluation_inputs_ready(sample,out):
    return all((Path(out)/folder/(sample['id']+'.json')).is_file() for folder in ['predictions','ocr'])


def render_saved_sample(sample,out):
    """Render saved exports only; no model execution or metric/cache writes."""
    from PIL import Image
    from tatr_ocr.visualization import draw_cells
    out=Path(out);name=sample['id']
    records={}
    for mode in ['official','current']:
        path=out/mode/(name+'.json')
        if not path.is_file():raise FileNotFoundError(f'Missing saved export: {path}; run evaluate first')
        records[mode]=read_json(path)
        if records[mode]['id']!=name:raise ValueError(f'Export ID mismatch: {path}')
    words_data=read_json(out/'words'/(name+'.json'))
    ocr_data=read_json(out/'ocr'/(name+'.json'))
    # Benchmark prediction uses raw RGB orientation, not EXIF-transposed input.
    with Image.open(sample['image']) as source:
        for mode,record in records.items():
            tokens=words_data if mode=='official' else ocr_data.get('tokens',[])
            draw_cells(source,record['cells'],tokens,out/mode/(name+'_overlay.png'),error=record.get('error'))


def render(args):
    manifest=verify_manifest(args.out)
    for index,sample in enumerate(manifest['samples'],1):
        render_saved_sample(sample,args.out)
        if index%25==0 or index==manifest['count']:
            emit(f'PNG overlays {index}/{manifest["count"]} per system')
    emit(f'PNG overlays saved in {args.out / "official"} and {args.out / "current"}')


def evaluate_one(sample,out):
    from tatr_ocr.benchmark import ground_truth,official_cells,current_cells,score_cells,official_export_table
    from tatr_ocr.fusion import fuse
    from tatr_ocr.export import html_document,write_csv
    out=Path(out);name=sample['id']
    words_data=read_json(out/'words'/(name+'.json'))
    truth,statistics=ground_truth(sample['xml'],words_data)
    category='complex' if statistics['num_spanning_cells']>0 else 'simple'
    prediction=read_json(out/'predictions'/(name+'.json'))
    ocr_data=read_json(out/'ocr'/(name+'.json'))
    fingerprint=evaluation_fingerprint(sample,out)
    result={'id':name,'category':category,'ground_truth_statistics':statistics,'input_fingerprint':fingerprint}
    for mode in ['official','current']:
        source=prediction[mode];error=source.get('error');cells=[];table=None
        start=perf_counter()
        if mode=='current':error=error or ocr_data.get('error')
        if not error:
            try:
                if mode=='official':
                    structure,cells,confidence=official_cells(source['boxes'],source['labels'],source['scores'],words_data)
                    if not cells:raise ValueError('No usable predicted table')
                    table=official_export_table(structure,cells)
                else:
                    if prediction['image_size']!=ocr_data['image_size']:raise ValueError('OCR image coordinates differ')
                    table=fuse(source['objects'],ocr_data['tokens'],.5)
                    cells=current_cells(table['cells'])
            except Exception as caught:error=f'{type(caught).__name__}: {caught}';cells=[];table=None
        post_seconds=perf_counter()-start
        # Metric implementation errors are fatal, not silently scored as model failures.
        metrics=score_cells(truth,cells)
        result[mode]={'id':name,'category':category,'metrics':metrics,'error':error,
                      'seconds':{'structure':source['seconds'],'postprocess':post_seconds,
                                 'ocr':ocr_data.get('seconds',0) if mode=='current' else 0},
                      'rows':table['rows'] if table else 0,'columns':table['columns'] if table else 0}
        folder=out/mode;folder.mkdir(exist_ok=True)
        write_json(folder/(name+'.json'),{'id':name,'cells':cells,**result[mode]})
        if table:
            (folder/(name+'.html')).write_text(html_document(name,table),encoding='utf-8')
            write_csv(folder/(name+'.csv'),table)
    if evaluation_fingerprint(sample,out)!=fingerprint:raise ValueError('Inputs changed during evaluation')
    render_saved_sample(sample,out)
    write_json(out/'evaluated'/(name+'.json'),result)
    return name


def evaluate(args):
    manifest=verify_manifest(args.out)
    words_meta=read_json(args.out/'words-provenance.json')
    if set(words_meta['files'])!={s['id'] for s in manifest['samples']}:raise ValueError('Words IDs differ')
    for sample in manifest['samples']:
        name=sample['id']
        if sha(args.out/'words'/(name+'.json'))!=words_meta['files'][name]:raise ValueError('Word checksum mismatch')
        for folder in ['predictions','ocr']:
            if not args.wait_for_inputs and not (args.out/folder/(name+'.json')).exists():
                raise ValueError(f'Incomplete {folder}: {name}')
    stage_config(args,'evaluation',{'source_sha256':sha(__file__),
        'helper_sha256':sha(ROOT/'tatr_ocr/benchmark.py'),'fusion_sha256':sha(ROOT/'tatr_ocr/fusion.py'),
        'official_sources':read_json(ROOT/'third_party/tatr/provenance.json'),
        'predict_config_sha256':sha(args.out/'predict-config.json'),'ocr_config_sha256':sha(args.out/'ocr-config.json'),
        'word_provenance_sha256':sha(args.out/'words-provenance.json'),
        'content_normalization':'none; exact source text including whitespace',
        'iou_compatibility':'cast numpy scalar coordinates to Python float before unmodified upstream IoU'})
    (args.out/'evaluated').mkdir(exist_ok=True)
    pending=[s for s in manifest['samples'] if not evaluation_is_current(s,args.out)]
    done=manifest['count']-len(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        while pending or futures:
            available=[s for s in pending if evaluation_inputs_ready(s,args.out)]
            for sample in available[:max(0,args.workers*2-len(futures))]:
                futures[pool.submit(evaluate_one,sample,str(args.out))]=sample['id']
                pending.remove(sample)
            if not futures:
                if not args.wait_for_inputs:raise ValueError('Evaluation inputs disappeared')
                sleep(5);continue
            completed,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for future in completed:
                future.result();futures.pop(future);done+=1
                if done%10==0:emit(f'Paired evaluation {done}/{manifest["count"]}')
    emit('Paired evaluation completed')


def report(args):
    from tatr_ocr.benchmark import summarize,METRICS
    import csv
    manifest=verify_manifest(args.out)
    if any(not evaluation_is_current(s,args.out) for s in manifest['samples']):
        raise ValueError('Evaluation results are incomplete or stale; run evaluate again')
    records=[read_json(args.out/'evaluated'/(s['id']+'.json')) for s in manifest['samples']]
    result={'sample_count':manifest['count'],'seed':manifest['seed'],'population':manifest['population'],
            'manifest_sha256':sha(args.out/'manifest.json'),'created_utc':datetime.now(timezone.utc).isoformat(),
            'evaluation':'strict official GriTS and AccCon; fixed threshold 0.5; no test-set tuning',
            'confidence_intervals':'2000 bootstrap replicates, clustering by source article, seed 42',
            'limitations':['Microsoft Hugging Face checkpoint; no retraining or original DETR runtime numerical equivalence claim.',
                          'The official protocol and current system differ in preprocessing, token source and postprocessing; the gap is not an isolated OCR ablation.',
                          'A 1000-table test subset, not the full test set; table crops, not full-page detection.',
                          'Exact content scoring preserves whitespace; runtime reflects this CPU and concurrent stages, not paper hardware.']}
    result['paper_reference']={'source':'https://arxiv.org/pdf/2110.00061#page=8','row':'Table 4, Canonical / DETR / All',
                               'acc_con':.8138,'grits_top':.9845,'grits_con':.9846,'grits_loc':.9781,
                               'scope':'Full official test set; different published experimental version, contextual reference only'}
    for mode in ['official','current']:
        rows=[r[mode] for r in records]
        result[mode]=summarize(rows,manifest['count'])
        result[mode]['mean_seconds']={k:sum(r['seconds'][k] for r in rows)/len(rows) for k in ['structure','postprocess','ocr']}
        result[mode]['mean_seconds']['total']=sum(result[mode]['mean_seconds'].values())
        write_json(args.out/f'{mode}-summary.json',result[mode])
        with (args.out/f'{mode}-metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=['id','category','rows','columns',*METRICS,'structure_seconds','postprocess_seconds','ocr_seconds','error']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow({k:r[k] for k in ['id','category','rows','columns','error']}|r['metrics']|
                                          {k+'_seconds':v for k,v in r['seconds'].items()})
    write_json(args.out/'comparison-summary.json',result)
    rows=''
    for cat,label in [('all','全部表格'),('simple','简单表格'),('complex','复杂表格')]:
        for mode,title in [('official','官方评价条件基线'),('current','TATR + PP-OCRv6 当前系统')]:
            r=result[mode][cat]
            if not r['num_tables']:continue
            rows+=f'<tr><td>{label}</td><td>{title}</td><td>{r["num_tables"]}</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in METRICS)+f'<td>{r["failures"]}</td></tr>'
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1000张表格双版本测试结果</title>
<style>body{{max-width:1200px;margin:48px auto;color:#203448;font:18px/1.8 SimSun,serif}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bac7d1;padding:8px}}th{{background:#e9eff4}}a{{color:#245e88}}</style>
<h1>PubTables-1M：1000 张测试表格双版本评估</h1>
<p>从官方 93,834 张测试表格中固定随机抽取 {manifest['count']} 张，随机种子 {manifest['seed']}。两套系统采用相同图片和标准答案，使用官方 GriTS 与 AccCon 评价。所有失败样本计入分母。</p>
<table><tr><th>分组</th><th>流程</th><th>样本数</th><th>AccCon</th><th>GriTS Top</th><th>GriTS Con</th><th>GriTS Loc</th><th>运行失败</th></tr>{rows}</table>
<p>AccCon 是整表内容完全正确的比例；GriTS 是平均相似度分数，不是整表成功率。简单表与复杂表按官方标准答案是否包含跨行或跨列单元格划分。</p>
<p>论文表 4 中 Canonical / DETR / All 的参考值依次为：AccCon 81.38%、GriTS Top 98.45%、GriTS Con 98.46%、GriTS Loc 97.81%。论文是全测试集结果，权重、代码及样本范围需结合本次记录解释，不能仅凭数值接近认定完成全部论文实验。</p>
<p>官方评价条件基线采用微软 Hugging Face 发布权重、官方最长边 1000 像素预处理、官方词信息及后处理。当前系统保留原有 AutoImageProcessor、PP-OCRv6 small（放大 2 倍）及文字融合流程。因此差值同时包含预处理、OCR 和后处理的影响。</p>
<p>位置指标还受到边界生成规则影响：官方后处理依据词框调整行列和单元格边界，当前融合流程先在空词信息上构造结构框，随后填入 OCR 文字。因此 GriTS Loc 的差距不能单独解释为 TATR 定位能力的差距。</p>
<p>本实验属于预训练模型的推理与评估复现，未重新训练。结果来自测试子集，不能等同于全测试集结果。严格内容比较保留空格和换行。95% 置信区间、版本及详细指标见 JSON 文件。</p>
<ul><li><a href="official-metrics.csv">官方基线逐表结果 CSV</a></li><li><a href="current-metrics.csv">当前系统逐表结果 CSV</a></li><li><a href="comparison-summary.json">完整统计与置信区间 JSON</a></li><li><a href="manifest.json">固定测试名单</a></li></ul>
<p>参考：<a href="https://github.com/microsoft/table-transformer">Microsoft Table Transformer</a>，<a href="https://arxiv.org/pdf/2110.00061">PubTables-1M 论文</a>。</p></html>'''
    (args.out/'测试结果.html').write_text(document,encoding='utf-8')
    emit(json.dumps({mode:result[mode]['all'] for mode in ['official','current']},ensure_ascii=False))


def run_all(args):
    """Run both benchmark variants, failing immediately if any stage fails."""
    emit(f'Paired benchmark output: {args.out}')
    stages=['prepare','words','predict','ocr','evaluate','report']
    for index,stage in enumerate(stages,1):
        emit(f'[{index}/{len(stages)}] Starting {stage}')
        globals()[stage](args)
    emit(f'Completed both systems. Report: {args.out / "测试结果.html"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','words','predict','ocr','evaluate','report','render','all'])
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path,help='Output directory; all defaults to a new timestamped run')
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--wait-for-inputs',action='store_true',help='Evaluate completed images while remaining OCR/predictions are still running')
    args=parser.parse_args()
    if not 1<=args.threads<=64 or not 1<=args.workers<=16:parser.error('Invalid worker/thread count')
    if args.out is None:
        args.out=ROOT/'outputs'/datetime.now().strftime('benchmark-%Y%m%d-%H%M%S-%f') if args.stage=='all' else DEFAULT_OUT
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    (run_all if args.stage=='all' else globals()[args.stage])(args)


if __name__=='__main__':main()
