from pathlib import Path
import os,sys,time,json,fcntl,traceback,gc,shutil,subprocess
REPO=Path('/root/phyla-research');sys.path.insert(0,str(REPO))
ROOT=Path('/mnt/nvme/scratch/phyla-ubuntu');PERSIST=Path('/workspace/phyla-research-runs')
lock=open('/root/phyla-exp03-recovery.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**kwargs):
 v={'phase':phase,'updated_unix':time.time(),'pid':os.getpid(),**kwargs}
 p=PERSIST/'evaluation-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
 print(json.dumps(v),flush=True)
status('waiting_for_pinned_assets')
log=PERSIST/'restore-download.log';deadline=time.time()+3600
while 'DOWNLOADS COMPLETE' not in log.read_text():
 if 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Asset restoration failed; inspect restore-download.log')
 if time.time()>deadline:raise TimeoutError('Model asset download exceeded one hour')
 time.sleep(10)
from flomo.model import convert_ltx
rev='8984fa25007f376c1a299016d0957a37a2f797bb';ar='e58e28c39631af4d1468ee57a853764e11c1d37e';base=ROOT/'cache/checkpoints';converted=base/'ltx096-full'/rev
if not (converted/'flomo_ltx.json').exists():
 status('converting_pinned_ltx')
 # The volume does not support chmod/copystat; copy bytes without metadata changes.
 def volume_copytree(src,dst,*args,**kwargs):
  src=Path(src);dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
  for p in src.rglob('*'):
   q=dst/p.relative_to(src)
   if p.is_dir():q.mkdir(parents=True,exist_ok=True)
   elif p.is_file():
    q.parent.mkdir(parents=True,exist_ok=True)
    with p.open('rb') as inf,q.open('wb') as outf:shutil.copyfileobj(inf,outf,256*1024)
  return str(dst)
 original=shutil.copytree;shutil.copytree=volume_copytree
 try:convert_ltx(base/'ltx096'/rev/'ltxv-2b-0.9.6-dev-04-25.safetensors',base/'ltx095-config'/ar,rev,converted)
 finally:shutil.copytree=original
from flomo.config import load_config
from flomo.policy import BundlePolicy
from flomo.sim import evaluate
from flomo.lora_audit import frozen_digest
import torch
rows=json.loads((ROOT/'runs/exp03/pilot/manifest.json').read_text())['runs']
results=[]
for row in rows:
 name=row['run'];cfg=load_config(REPO/row['config']);cfg.eval.render_backend='cpu';out=Path(cfg.train.output)
 status('loading_policy',run=name,completed_cells=len(results))
 policy=BundlePolicy(out/'policy',cfg.eval.device,cfg.eval.sampler,cfg.eval.sampling_steps,cfg.eval.shift)
 expected=json.loads((out/'lora_audit.json').read_text())['frozen_base_sha256']
 actual=frozen_digest(policy.model)
 if actual!=expected:raise RuntimeError(f'Frozen base mismatch: {actual} != {expected}')
 for condition in ('id','cube_color','camera30'):
  cfg.eval.ood_condition=condition;cfg.eval.output=str(out/('eval_'+condition));target=Path(cfg.eval.output)
  if (target/'summary.json').exists():summary=json.loads((target/'summary.json').read_text())
  else:
   if target.exists():target.rename(out/('eval_'+condition+'_incomplete_'+str(int(time.time()))))
   status('evaluating',run=name,condition=condition,completed_cells=len(results),requested_cells=36)
   summary=evaluate(cfg,policy_instance=policy)
  results.append({'run':name,'condition':condition,'summary':summary})
  (PERSIST/'completed-evaluation-cells.json').write_text(json.dumps(results,indent=2))
  status('cell_completed',run=name,condition=condition,completed_cells=len(results),requested_cells=36)
 del policy;gc.collect();torch.cuda.empty_cache()
status('auditing',completed_cells=len(results))
subprocess.run([sys.executable,str(REPO/'experiments/check_eval_contracts.py')],check=True,cwd=REPO)
subprocess.run([sys.executable,str(REPO/'experiments/recovery/report_evaluations.py')],check=True,cwd=REPO)
status('completed',completed_cells=len(results),requested_cells=36,rollouts=180,results_root=str(ROOT/'runs/exp03/pilot'))
