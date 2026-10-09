"""Bounded monitored pilot queues. No OOD checkpoint selection or future inputs."""
from __future__ import annotations
import argparse,datetime,json,os,pathlib,subprocess,sys,time,shutil
REPO=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from flomo.config import load_config,atomic_json
from flomo.train import export_bundle
ROOT=pathlib.Path(os.environ.get("PHYLA_RUN_ROOT","/mnt/nvme/scratch/phyla-ubuntu"));CONTROL=ROOT/"control/exp03"
CUTOFF=datetime.datetime.fromisoformat("2026-10-09T04:40:00+00:00").timestamp()
MANIFEST=ROOT/"runs/exp03/pilot/manifest.json"
def state(name,phase,**kw):atomic_json(CONTROL/"jobs"/(name+".json"),{"job":name,"state":phase,"pid":os.getpid(),"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"updated":time.time(),**kw})
def train_queue(args):
 for row in json.loads(MANIFEST.read_text())["runs"]:
  if row["arm"]!=args.arm or (args.n and row["N"]!=args.n):continue
  if time.time()>CUTOFF:state("queue_"+args.arm+str(args.n),"cutoff");break
  name=row["run"];cfg=load_config(REPO/row["config"]);out=pathlib.Path(cfg.train.output);log=ROOT/"runs/exp03/pilot"/(name+".log")
  if (out/"policy/bundle.json").exists():continue
  state(name,"running",log=str(log),config=row["config"],phase="LoRA_SFT")
  if not (out/f"step_{cfg.train.steps:06d}.pt").exists():
   with open(log,"a") as f:code=subprocess.call([sys.executable,"-m","flomo.cli","train","--config",row["config"]],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
   if code:state(name,"failed",returncode=code,log=str(log));continue
  audit=json.loads((out/"lora_audit.json").read_text())
  if not audit.get("verified_after_training"):state(name,"failed",error="Unverified frozen base");continue
  bundle=export_bundle(out/f"step_{cfg.train.steps:06d}.pt",out/"policy");shutil.copy2(out/"lora_audit.json",out/"policy/lora_audit.json")
  state(name,"completed",bundle=str(out/"policy"),weights_bytes=(out/"policy/weights.pt").stat().st_size,base_verified=True,steps=cfg.train.steps,log=str(log))
def eval_queue(args):
 from flomo.policy import BundlePolicy
 from flomo.sim import evaluate
 rows=json.loads(MANIFEST.read_text())["runs"];done=set()
 while len(done)<len(rows):
  if time.time()>CUTOFF:state("eval_queue","cutoff",completed=len(done));break
  ready=[r for r in rows if r["run"] not in done and (pathlib.Path(load_config(REPO/r["config"]).train.output)/"policy/bundle.json").exists()]
  if not ready:time.sleep(5);continue
  for row in ready:
   if time.time()>CUTOFF:break
   name=row["run"];cfg=load_config(REPO/row["config"]);out=pathlib.Path(cfg.train.output)
   state("eval_"+name,"running",phase="paired_ID_OOD",log=str(ROOT/"runs/exp03/evaluation.log"))
   try:
    policy=BundlePolicy(out/"policy",cfg.eval.device,cfg.eval.sampler,cfg.eval.sampling_steps,cfg.eval.shift)
    for condition in ("id","cube_color","camera30"):
     cfg.eval.ood_condition=condition;cfg.eval.output=str(out/("eval_"+condition))
     if (pathlib.Path(cfg.eval.output)/"summary.json").exists():continue
     result=evaluate(cfg,policy_instance=policy);print(json.dumps({"run":name,"condition":condition,"summary":result}),flush=True)
    del policy
    import torch,gc
    gc.collect();torch.cuda.empty_cache()
    state("eval_"+name,"completed",conditions=["id","cube_color","camera30"],episodes_per_condition=cfg.eval.episodes)
   except Exception as exc:state("eval_"+name,"failed",error=repr(exc));print(json.dumps({"run":name,"error":repr(exc)}),flush=True)
   done.add(name)
 state("eval_queue","completed" if len(done)==len(rows) else "cutoff",completed=len(done),requested=len(rows))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["train","eval"]);p.add_argument("--arm",choices=["A","R","F-GT"]);p.add_argument("--n",type=int,default=0);a=p.parse_args();(train_queue if a.stage=="train" else eval_queue)(a)
if __name__=="__main__":main()
