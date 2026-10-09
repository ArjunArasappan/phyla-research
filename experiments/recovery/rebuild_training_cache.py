"""Rebuild lost LTX samples from preserved windows/stats, after evaluation.

Never modifies the historical datasets. New hashes explicitly identify a
reconstructed debugging dataset, not a bit-identical original training cache.
"""
from pathlib import Path
import argparse, fcntl, json, os, platform, shutil, sys, time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from flomo.config import atomic_json, digest, encoder_signature, file_hash, load_config
from flomo.data import PreparedDataset, read_jsonl, torch_load, write_jsonl
from flomo.geometry import PercentileStats, render_flow
from flomo.model import FrozenEncoders
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/workspace/phyla-research-runs'))
parser.add_argument('--validate-inputs', action='store_true')
args = parser.parse_args()
root = args.root
data = root/'exp03-recovery-root/data/exp03/pushcube'
cache = data/'debug-cache/ltx'

def status(phase, **fields):
    record = dict(phase=phase, updated_unix=time.time(), pid=os.getpid(), **fields)
    atomic_json(root/'training-cache-status.json', record)
    print(json.dumps(record), flush=True)

def inputs(n):
    config = load_config(REPO/f'experiments/03-flow-vs-rgb-data-efficiency-ood/configs/N{n:03d}_A_seed0.yaml')
    src = data/f'pilot_N{n:03d}_prepared_v2'
    meta = json.loads((src/'dataset.json').read_text())
    rows = read_jsonl(src/'windows.jsonl')
    stats = json.loads((src/'stats.json').read_text())
    assert digest({k:v for k,v in meta.items() if k!='dataset_id'}) == meta['dataset_id']
    assert digest(rows) == meta['manifest_hash']
    assert digest({k:v for k,v in stats.items() if k!='stats_id'}) == meta['stats_id']
    assert encoder_signature(config) == meta['encoder_signature']
    for row in rows:
        assert (data/'raw'/row['path']).is_file(), row['path']
        assert (src/'tracks'/f"{row['window_id']}.npz").is_file(), row['window_id']
    return config, src, meta, rows, stats

if args.validate_inputs:
    for n in (16,64):
        config,src,meta,rows,stats = inputs(n)
        print(json.dumps(dict(N=n, windows=len(rows), original_dataset_id=meta['dataset_id'], inputs_verified=True)))
    sys.exit(0)

root.mkdir(parents=True, exist_ok=True)
lock = open(root/'training-cache.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
status('queued_after_evaluation', requested_windows=398, output=str(cache))
while True:
    evaluation = json.loads((root/'evaluation-status.json').read_text())
    if evaluation['phase'] == 'completed':
        assert evaluation['completed_cells'] == 36
        break
    try:
        os.kill(evaluation['pid'], 0)
    except ProcessLookupError:
        raise RuntimeError('Evaluation stopped before completion; cache remains queued. Inspect evaluation.log.')
    time.sleep(30)

encoders = None
text_cache = {}
try:
    for n in (16,64):
        config,src,old,rows,stats = inputs(n)
        out = cache/f'N{n:03d}'
        out.mkdir(parents=True, exist_ok=True)
        if (out/'dataset.json').exists():
            for split in sorted({r['split'] for r in rows}):
                PreparedDataset(out,split=split).validate()
            continue
        recipe = dict(original_dataset_id=old['dataset_id'], torch=torch.__version__,
                      numpy=np.__version__, python=platform.python_version(),
                      encoder_signature=encoder_signature(config))
        stamp = out/'rebuilding.json'
        if stamp.exists() and json.loads(stamp.read_text()) != recipe:
            raise ValueError('Incomplete cache has a different rebuild recipe; choose a new directory')
        atomic_json(stamp,recipe)
        if encoders is None:
            status('loading_encoders', N=n)
            encoders = FrozenEncoders(config.model, 'cuda')
        else:
            assert encoder_signature(config) == encoder_signature(previous_config)
        previous_config = config
        episodes = {e['episode_id']:e for e in old['episodes']}
        ledger_path = out/'sample-hashes.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        (out/'samples').mkdir(exist_ok=True)
        (out/'tracks').mkdir(exist_ok=True)
        for index,row in enumerate(rows):
            wid = row['window_id']
            target = out/'samples'/f'{wid}.pt'
            status('encoding', N=n, completed_windows=index, requested_windows=len(rows), window_id=wid)
            if not (target.exists() and wid in ledger and file_hash(target)==ledger[wid]):
                with np.load(data/'raw'/row['path'],allow_pickle=False) as ep, np.load(src/'tracks'/f'{wid}.npz',allow_pickle=False) as tracks:
                    start,h = row['start'],config.model.horizon
                    meta = episodes[row['episode_id']]
                    rgb = ep['rgb'][start:start+h+int(config.model.future_only)]
                    sl = slice(1,None) if config.model.future_only else slice(None)
                    rendered = render_flow(tracks['flow'][sl],tracks['valid'][sl],
                        PercentileStats(**stats['flow'][row['source']]),config.data.grid_size,config.model.image_size)
                    obs = encoders.encode_observations(rgb[0])
                    future = rgb[1:,0]
                    if not config.model.future_only:
                        future = np.concatenate([future,future[-1:]],axis=0)
                    video = encoders.encode_video(torch.from_numpy(future.copy()).permute(0,3,1,2).float()/255)
                    flow = encoders.encode_video(rendered)
                    if meta['instruction'] not in text_cache:
                        text_cache[meta['instruction']] = tuple(t.cpu() for t in encoders.encode_text(meta['instruction']))
                    text,text_valid = text_cache[meta['instruction']]
                    action = torch.zeros(h,config.model.action_dim)
                    if row['has_actions']:
                        action = torch.from_numpy(PercentileStats(**stats['action']).normalize(ep['actions'][start:start+h],signed=True))
                    views = obs.shape[0]
                    assert meta['camera_names'] == config.eval.camera_names[:views]
                    observation = torch.zeros(config.model.views,*obs.shape[1:])
                    observation[:views] = obs.cpu()
                    sample = dict(obs=observation,view_valid=torch.arange(config.model.views)<views,
                        text=text,text_valid=text_valid,flow=flow.cpu(),video=video.cpu(),action=action,
                        has_action=torch.tensor(row['has_actions']),source=row['source'],window_id=wid,episode_id=row['episode_id'])
                    for value in sample.values():
                        if isinstance(value,torch.Tensor) and value.is_floating_point():
                            assert torch.isfinite(value).all()
                    temporary = target.with_suffix('.tmp')
                    torch.save(sample,temporary)
                    temporary.replace(target)
                ledger[wid] = file_hash(target)
                atomic_json(ledger_path,ledger)
            row['sample_sha256'] = ledger[wid]
            with (src/'tracks'/f'{wid}.npz').open('rb') as inf, (out/'tracks'/f'{wid}.npz').open('wb') as outf:
                shutil.copyfileobj(inf,outf,256*1024)
        atomic_json(out/'stats.json',stats)
        write_jsonl(out/'windows.jsonl',rows)
        write_jsonl(out/'rejected.jsonl',read_jsonl(src/'rejected.jsonl'))
        metadata = {k:v for k,v in old.items() if k!='dataset_id'}
        metadata.update(manifest_hash=digest(rows),reconstruction=recipe,
                        cache_purpose='debugging; regenerated latents, not historical bit-identical samples')
        metadata['preprocess'] = {**old['preprocess'],'raw':str(data/'raw'),'prepared':str(out)}
        metadata['dataset_id'] = digest(metadata)
        atomic_json(out/'dataset.json',metadata)
        for split in sorted({r['split'] for r in rows}):
            PreparedDataset(out,split=split).validate()
    status('completed', requested_windows=398, output=str(cache), subsets=[16,64], validated=True)
except Exception as error:
    status('failed',error=repr(error),output=str(cache))
    raise
