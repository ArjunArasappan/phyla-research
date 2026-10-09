"""Experiment 3: genuine robot demonstrations, nested subsets and matched LoRA runs.

Run from repository root. Large artifacts stay outside Git. No toy data fallback.
"""
from __future__ import annotations
import argparse,json,os,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flomo.config import Config,atomic_json,digest
from flomo.data import save_episode,read_jsonl,write_jsonl
ROOT=Path(os.environ.get("PHYLA_RUN_ROOT","/mnt/nvme/scratch/phyla-ubuntu"))

def status(state,**kw):
 atomic_json(ROOT/"control/exp03/jobs/collection.json",{"experiment":3,"state":state,"gpus":[4],"timestamp":time.time(),"log":str(ROOT/"runs/exp03/collection.log"),**kw})

def collect(args):
 import gymnasium as gym
 import mani_skill.envs
 from mani_skill.examples.motionplanning.panda.solutions.push_cube import solve
 from flomo.sim import ManiSkillAdapter
 cfg=Config();cfg.model.image_size=args.size;cfg.model.action_dim=8
 cfg.eval.backend="maniskill";cfg.eval.env_id="PushCube-v1";cfg.eval.robot="panda"
 cfg.eval.control_mode="pd_joint_pos";cfg.eval.camera_names=["base_camera"]
 cfg.eval.embodiment="panda_native_joint_position";cfg.eval.instruction="push the cube to the target"
 cfg.eval.horizon=400;cfg.eval.sim_backend="physx_cpu"
 adapter=ManiSkillAdapter(cfg,collect_geometry=True)
 class Capture(gym.Wrapper):
  def reset(self,**kwargs):
   result=self.env.reset(**kwargs);adapter.obs=result[0];self.images=[adapter.observation()[0]];self.geom=[adapter.geometry()[0]];self.actions=[];return result
  def step(self,action):
   result=self.env.step(action);adapter.obs=result[0]
   self.actions.append(np.asarray(action.detach().cpu() if hasattr(action,"detach") else action).reshape(-1).astype(np.float32))
   self.images.append(adapter.observation()[0]);self.geom.append(adapter.geometry()[0]);return result
 env=Capture(adapter.env);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 existing=read_jsonl(out/"episodes.jsonl") if (out/"episodes.jsonl").exists() else []
 count=len(existing);attempts=[]
 try:
  for seed in range(args.start_seed,args.start_seed+args.max_attempts):
   if count>=args.count:break
   if any(x.get("collection_seed")==seed for x in existing):continue
   status("collecting",completed=count,requested=args.count,seed=seed)
   try:
    res=solve(env,seed=seed,debug=False,vis=False)
    success=res!=-1 and bool(res[-1]["success"].item())
    attempts.append({"seed":seed,"success":success})
    if not success:continue
    # Append genuine hold actions to preserve complete future windows at endpoint.
    for _ in range(16):env.step(env.actions[-1][None])
    for g in env.geom:
     if not np.array_equal(g["body_ids"],env.geom[0]["body_ids"]):raise ValueError("Segmentation body identities changed within episode")
    arrays={"rgb":np.stack(env.images),"actions":np.stack(env.actions),"times":np.arange(len(env.images))/20.}
    arrays["action_times"]=arrays["times"][:-1]
    arrays.update({k:np.stack([g[k] for g in env.geom]) for k in env.geom[0] if k!="body_ids"});arrays["body_ids"]=env.geom[0]["body_ids"]
    meta={"source":"pushcube_expert","instruction":cfg.eval.instruction,"camera_names":cfg.eval.camera_names,"embodiment":cfg.eval.embodiment,"control_mode":cfg.eval.control_mode,"control_hz":20,"split_group":f"pushcube_seed_{seed}","collection":"official_maniskill_motionplanning","collection_seed":seed,"success_at_end":success,"action_dim":adapter.action_dim,"geometry":"rigid_link_material_points; visibility evaluated separately"}
    row=save_episode(out,f"pushcube_{seed:06d}",arrays,meta);existing.append(row);count+=1
    print(json.dumps({"saved":row["episode_id"],"frames":len(env.images),"count":count}),flush=True)
   except Exception as exc:
    attempts.append({"seed":seed,"success":False,"error":repr(exc)});print(json.dumps(attempts[-1]),flush=True)
   write_jsonl(out/"collection_attempts.jsonl",attempts)
 finally:env.close()
 atomic_json(out/"collection.json",{"requested":args.count,"saved":count,"attempts":len(attempts),"controller":cfg.eval.control_mode,"action_dim":adapter.action_dim})
 if count<args.count:raise RuntimeError(f"Only {count}/{args.count} successful demonstrations")
 status("demonstrations_ready",episodes=count,raw=str(out))

def subsets(args):
 raw=Path(args.raw);rows=read_jsonl(raw/"episodes.jsonl")
 order=np.random.default_rng(args.seed).permutation(len(rows));rows=[rows[i] for i in order]
 if len(rows)<max(args.counts):raise ValueError("Insufficient independent demonstrations")
 for n in args.counts:
  out=Path(args.output)/f"N{n:03d}_subset{args.seed}";out.mkdir(parents=True,exist_ok=True)
  selected=rows[:n]
  for row in selected:
   dst=out/row["path"]
   if not dst.exists():dst.symlink_to((raw/row["path"]).resolve())
  write_jsonl(out/"episodes.jsonl",selected)
  atomic_json(out/"subset.json",{"N":n,"seed":args.seed,"episode_ids":[r["episode_id"] for r in selected],"parent_manifest":digest(read_jsonl(raw/"episodes.jsonl")),"nested":True,"split":"train_only","validation":"none; fixed-step pilot"})
 print(json.dumps({"subsets":args.counts,"output":args.output}),flush=True)

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="stage",required=True)
 q=s.add_parser("collect");q.add_argument("--output",default=str(ROOT/"data/exp03/pushcube/raw"));q.add_argument("--count",type=int,default=64);q.add_argument("--size",type=int,default=256);q.add_argument("--start-seed",type=int,default=10000);q.add_argument("--max-attempts",type=int,default=192)
 q=s.add_parser("subsets");q.add_argument("--raw",required=True);q.add_argument("--output",required=True);q.add_argument("--counts",type=int,nargs="+",default=[16,64]);q.add_argument("--seed",type=int,default=0)
 args=p.parse_args();globals()[args.stage](args)
if __name__=="__main__":main()
