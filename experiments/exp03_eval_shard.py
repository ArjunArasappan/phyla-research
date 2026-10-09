"""Matched paired evaluation shards; retain failed-attempt provenance."""
from __future__ import annotations
import argparse,datetime,json,os,pathlib,sys,time,gc,traceback
REPO=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from flomo.config import load_config,atomic_json
from flomo.policy import BundlePolicy
from flomo.sim import evaluate
import torch
ROOT=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu');CONTROL=ROOT/'control/exp03';CUTOFF=datetime.datetime.fromisoformat('2026-10-09T04:45:00+00:00').timestamp()
def state(name,value,**kw):atomic_json(CONTROL/'jobs'/(name+'.json'),{'job':name,'state':value,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'updated':time.time(),**kw})
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--shards',type=int,default=5);args=parser.parse_args()
 rows=[r for i,r in enumerate(json.loads((ROOT/'runs/exp03/pilot/manifest.json').read_text())['runs']) if i%args.shards==args.shard];done=set()
 while len(done)<len(rows):
  if time.time()>CUTOFF:break
  ready=[r for r in rows if r['run'] not in done and (pathlib.Path(load_config(REPO/r['config']).train.output)/'policy/bundle.json').exists()]
  if not ready:time.sleep(5);continue
  for row in ready:
   if time.time()>CUTOFF:break
   name=row['run'];cfg=load_config(REPO/row['config']);out=pathlib.Path(cfg.train.output);job=CONTROL/'jobs'/('eval_'+name+'.json')
   if job.exists() and json.loads(job.read_text()).get('state')=='failed':
    failures=CONTROL/'failed_attempts';failures.mkdir(exist_ok=True);job.rename(failures/(job.stem+'_bf16_'+str(int(time.time()))+'.json'))
   state('eval_'+name,'running',phase='paired_ID_OOD_retry',log=str(ROOT/'runs/exp03'/('evaluation_shard'+str(args.shard)+'.log')))
   failed=[]
   try:
    policy=BundlePolicy(out/'policy',cfg.eval.device,cfg.eval.sampler,cfg.eval.sampling_steps,cfg.eval.shift)
    for condition in ('id','cube_color','camera30'):
     if time.time()>CUTOFF:break
     cfg.eval.ood_condition=condition;cfg.eval.output=str(out/('eval_'+condition));target=pathlib.Path(cfg.eval.output)
     if (target/'summary.json').exists():continue
     if target.exists():target.rename(out/('eval_'+condition+'_failed_before_retry_'+str(int(time.time()))))
     try:
      result=evaluate(cfg,policy_instance=policy);print(json.dumps({'run':name,'condition':condition,'summary':result}),flush=True)
     except Exception as exc:
      failed.append({'condition':condition,'error':repr(exc)});traceback.print_exc()
    del policy;gc.collect();torch.cuda.empty_cache()
    completed=[c for c in ('id','cube_color','camera30') if (out/('eval_'+c)/'summary.json').exists()]
    state('eval_'+name,'completed' if len(completed)==3 else 'failed',completed_conditions=completed,failures=failed,episodes_per_condition=cfg.eval.episodes)
   except Exception as exc:state('eval_'+name,'failed',error=repr(exc));traceback.print_exc()
   done.add(name)
 state('eval_shard'+str(args.shard),'completed' if len(done)==len(rows) else 'cutoff',completed=len(done),requested=len(rows))
if __name__=='__main__':main()
