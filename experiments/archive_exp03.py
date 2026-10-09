"""Create verifiable compact Experiment 3 backups; never silently drop scientific files.

core: raw demonstrations, canonical flow-label arrays, subset/preprocess manifests,
      all learned-only policy exports, audits, config/metrics, and failed provenance.
eval: completed scientific evaluation arrays/videos, summaries, forecasts/reports.
Full optimizer checkpoints and replaceable frozen model weights remain on the node.
"""
from pathlib import Path
import argparse,json,hashlib,tarfile,time
ROOT=Path('/mnt/nvme/scratch/phyla-ubuntu');REPO=ROOT/'worktrees/exp03'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['core','eval'],default='core');p.add_argument('--output');a=p.parse_args()
 files=set()
 def add(path):
  if path.is_file() and not path.name.endswith(('.tmp','.part','.lock')):files.add(path)
 def tree(path):
  if path.exists():
   for f in path.rglob('*'):
    if f.is_file():add(f)
 manifest=ROOT/'runs/exp03/pilot/manifest.json'
 if a.phase=='core':
  tree(ROOT/'data/exp03/pushcube/raw')
  for n in (16,64):
   prep=ROOT/f'data/exp03/pushcube/pilot_N{n:03d}_prepared_v2'
   for name in ('dataset.json','stats.json','windows.jsonl','rejected.jsonl','preparing.json'):add(prep/name)
   tree(prep/'tracks')
   for name in ('episodes.jsonl','subset.json'):add(ROOT/f'data/exp03/pushcube/subsets/N{n:03d}_subset0'/name)
  add(manifest);tree(ROOT/'control/exp03')
  missing=[]
  for r in json.loads(manifest.read_text())['runs']:
   run=ROOT/'runs/exp03/pilot'/r['run'];bundle=run/'policy'
   if not (bundle/'bundle.json').is_file():missing.append(r['run']);continue
   tree(bundle)
   for name in ('config.json','metrics.jsonl','lora_audit.json','latest.json'):add(run/name)
  if missing:raise RuntimeError('Refusing incomplete adapter backup: '+repr(missing))
  tree(ROOT/'runs/exp03/ood_sanity')
 elif a.phase=='eval':
  for r in json.loads(manifest.read_text())['runs']:
   run=ROOT/'runs/exp03/pilot'/r['run']
   # Only completed cells, so a scientific artifact cannot be copied mid-write.
   for condition in ('id','cube_color','camera30'):
    cell=run/('eval_'+condition)
    if (cell/'summary.json').is_file():tree(cell)
   for failed in run.glob('eval_*failed*'):tree(failed)
  tree(ROOT/'runs/exp03/diagnostic_repair_validation')
  tree(ROOT/'runs/exp03/bf16_fixed_validation')
  tree(REPO/'experiments/03-flow-vs-rgb-data-efficiency-ood/results/ai')
 else:raise AssertionError
 # Scientific backups may never contain pretrained base weights/full optimizers.
 assert all(not f.name.startswith('step_') for f in files)
 assert all(f.name!='weights.pt' or f.parent.name=='policy' for f in files)
 records=[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(files)]
 backup=ROOT/'backups';backup.mkdir(exist_ok=True)
 ledger=backup/f'exp03-{a.phase}-manifest.json';ledger.write_text(json.dumps({'phase':a.phase,'files':records,'input_bytes':sum(x['bytes'] for x in records),'omissions':['frozen base checkpoints','full optimizer checkpoints','regenerable prepared latent caches'],'created_unix':time.time()},indent=2));files.add(ledger)
 out=Path(a.output) if a.output else backup/f'exp03-{a.phase}.tar.gz';temp=out.with_suffix(out.suffix+'.part')
 with tarfile.open(temp,'w:gz',compresslevel=6) as archive:
  for f in sorted(files):archive.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
 temp.replace(out)
 result={'archive':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'files':len(records),'input_bytes':sum(x['bytes'] for x in records),'manifest':str(ledger)}
 out.with_suffix(out.suffix+'.sha256.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
