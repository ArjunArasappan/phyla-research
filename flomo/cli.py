"""Thin CLI; all commands call the same library interfaces."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch

from .config import Config, load_config, atomic_json
from .data import prepare, PreparedDataset, save_episode, read_jsonl
from .model import FrozenEncoders
from .train import train, export_bundle
from .sim import collect, evaluate, aggregate
from .policy import paired_comparison


def import_maniskill(config,trajectory,source,limit=None,replay=False):
    """Import standard replayed HDF5 sensor observations; no simulator import."""
    import h5py
    trajectory=Path(trajectory)
    if replay:
        command=[sys.executable,"-m","mani_skill.trajectory.replay_trajectory","--traj-path",str(trajectory),
                 "--save-traj","--use-env-states","--obs-mode","rgb"]
        # Do not automatically choose a converted result filename: version naming differs.
        subprocess.run(command,check=True)
        return {"replayed":True,"next":"Pass the generated trajectory RGB HDF5 to import-maniskill without --replay"}
    metadata=json.loads(trajectory.with_suffix(".json").read_text())
    env_info=metadata["env_info"]; kwargs=env_info.get("env_kwargs",{})
    declared=kwargs.get("control_mode")
    if declared and declared!=config.eval.control_mode: raise ValueError("Recorded and configured controllers differ; convert/replay explicitly first")
    episodes=metadata.get("episodes",[]); by_id={str(e["episode_id"]):e for e in episodes}
    count=0
    with h5py.File(trajectory,"r") as handle:
        for name in sorted(handle):
            if not name.startswith("traj_"): continue
            if limit is not None and count>=limit: break
            group=handle[name]; actions=np.asarray(group["actions"],dtype=np.float32)
            if actions.shape[-1]!=config.model.action_dim: raise ValueError("Recorded action dimensions differ")
            cameras=[]
            for camera in config.eval.camera_names:
                key=f"obs/sensor_data/{camera}/rgb"
                if key not in group: raise ValueError(f"Missing {key}; replay with RGB observations and matching cameras")
                cameras.append(np.asarray(group[key],dtype=np.uint8))
            rgb=np.stack(cameras,axis=1)
            arrays={"rgb":rgb,"actions":actions,"times":np.arange(len(rgb))/config.eval.control_hz,
                    "action_times":np.arange(len(actions))/config.eval.control_hz}
            ep_meta=by_id.get(name.removeprefix("traj_"),{})
            label=ep_meta.get("instruction",config.eval.instruction)
            meta={"source":source,"instruction":label,"camera_names":config.eval.camera_names,"embodiment":config.eval.embodiment,
                  "control_mode":config.eval.control_mode,"control_hz":config.eval.control_hz,"env_id":env_info.get("env_id"),
                  "split_group":name,"imported_from":str(trajectory.resolve()),"replay_validation":"not_verified_by_import",
                  "collection":"maniskill_demonstration"}
            save_episode(config.data.raw,f"{source}_{count:06d}",arrays,meta); count+=1
    return {"episodes":count,"raw":config.data.raw,"flow_note":"RGB-only imports use tracker preprocessing or externally supplied posed geometry"}


def import_video(config,path,instruction,source,episode_id):
    try: import imageio.v3 as iio
    except ImportError as exc: raise RuntimeError("Install .[video] to decode human video") from exc
    frames=iio.imread(path,plugin="FFMPEG")
    meta_video=iio.immeta(path,plugin="FFMPEG"); fps=float(meta_video["fps"])
    if frames.ndim!=4 or frames.shape[-1]<3: raise ValueError("Unsupported video layout")
    arrays={"rgb":frames[...,:3].astype(np.uint8)[:,None],"times":np.arange(len(frames))/fps}
    meta={"source":source,"instruction":instruction,"camera_names":config.eval.camera_names[:1],"split_group":episode_id,
          "collection":"human_video","source_fps":fps,"imported_from":str(Path(path).resolve())}
    save_episode(config.data.raw,episode_id,arrays,meta)
    return {"episode":episode_id,"frames":len(frames),"fps":fps}


def smoke(workdir):
    root=Path(workdir).resolve()
    if root.exists() and (root/"report.json").exists(): raise FileExistsError("Smoke run already exists; choose a new workdir")
    root.mkdir(parents=True,exist_ok=True)
    config=Config(); config.data.raw=str(root/"raw"); config.data.prepared=str(root/"prepared")
    config.eval.horizon=32; config.train.steps=4; config.train.save_every=2; config.train.validate_every=2
    config.train.output=str(root/"midtrain"); config.eval.output=str(root/"expert_eval"); config.eval.episodes=3
    config.data.grid_size=8
    collection=collect(config,count=24,human_fraction=.25)
    metadata=prepare(config,FrozenEncoders(config.model))
    expert=evaluate(config,baseline="expert")
    checkpoint=train(config)
    export_bundle(checkpoint,root/"midtrain_bundle")
    config.train.stage="sft"; config.train.init=str(root/"midtrain_bundle"); config.train.steps=2
    config.train.lr=5.6e-5; config.train.losses["action"]=1.; config.train.output=str(root/"sft")
    checkpoint=train(config); export_bundle(checkpoint,root/"policy")
    config.eval.output=str(root/"policy_eval")
    policy=evaluate(config,bundle=str(root/"policy"))
    config.train.stage="probe"; config.train.probe="flow"; config.model.targets=["action"]
    config.train.init=str(root/"midtrain_bundle"); config.train.output=str(root/"probe"); config.train.steps=2
    config.train.losses={"action":1.,"flow":0.,"video":0.}
    probe=train(config)
    report={"collection":collection,"dataset":metadata["dataset_id"],"expert":expert,"policy":policy,"probe_checkpoint":probe,
            "verification":"CPU integration only; tiny model and toy environment; no claim of Wan/ManiSkill performance"}
    atomic_json(root/"report.json",report); return report


def doctor(config):
    packages={}
    for name in ["torch","numpy","h5py","diffusers","transformers","mani_skill"]:
        try:
            module=importlib.import_module(name); packages[name]=getattr(module,"__version__","installed")
        except ImportError: packages[name]="not installed"
    return {"platform":platform.platform(),"python":sys.version,"packages":packages,"cuda":torch.cuda.is_available(),
            "model_backend":config.model.backend,"gpu_checks":"not run" if not torch.cuda.is_available() else "available, integration still needs verification"}


def attention_audit(config,bundle,output,samples=4):
    from .policy import BundlePolicy
    from .data import collate
    from .model import noisy_targets
    from .train import move_batch,autocast
    policy=BundlePolicy(bundle,config.eval.device)
    if policy.config.model.architecture!="joint": raise ValueError("Attention audit requires a joint model")
    dataset=PreparedDataset(config.data.prepared,"val",labeled_only=True)
    batch=move_batch(collate([dataset[i] for i in range(min(samples,len(dataset)))]),policy.device)
    torch.manual_seed(config.train.seed)
    noisy,_,t=noisy_targets(batch,policy.config.model.targets)
    policy.model.audit=True
    with torch.no_grad(),autocast(policy.device,"bf16" if policy.config.model.backend in {"wan","ltx"} else "fp32"):
        policy.model(batch,noisy,t)
    result={"bundle":policy.bundle_id,"noise_levels":t.cpu().tolist(),"layers":policy.model.attention_log,
            "interpretation":"Descriptive attention mass; not evidence of a causal mechanism"}
    atomic_json(output,result); return result


def parser():
    p=argparse.ArgumentParser(description="Compact FloMo research pipeline")
    p.add_argument("--threads",type=int,default=1,help="CPU Torch threads (default 1 for small laptop checks)")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ["doctor","prepare","validate-data","train","probe","collect","evaluate","import-maniskill","import-video","attention"]:
        cmd=sub.add_parser(name); cmd.add_argument("--config",required=True)
        if name=="collect":
            cmd.add_argument("--episodes",type=int,default=20); cmd.add_argument("--teacher"); cmd.add_argument("--source",default="sim"); cmd.add_argument("--action-free-fraction",type=float,default=0.)
        if name=="evaluate":
            group=cmd.add_mutually_exclusive_group(required=True); group.add_argument("--bundle"); group.add_argument("--baseline",choices=["random","expert"])
        if name=="probe": cmd.add_argument("--signal",choices=["flow","video"],required=True)
        if name=="import-maniskill":
            cmd.add_argument("--trajectory",required=True); cmd.add_argument("--source",required=True); cmd.add_argument("--limit",type=int); cmd.add_argument("--replay",action="store_true")
        if name=="import-video":
            cmd.add_argument("--video",required=True); cmd.add_argument("--instruction",required=True); cmd.add_argument("--source",required=True); cmd.add_argument("--episode-id",required=True)
        if name=="attention":
            cmd.add_argument("--bundle",required=True); cmd.add_argument("--output",required=True); cmd.add_argument("--samples",type=int,default=4)
    cmd=sub.add_parser("export"); cmd.add_argument("--checkpoint",required=True); cmd.add_argument("--output",required=True)
    cmd=sub.add_parser("aggregate"); cmd.add_argument("--episodes",required=True); cmd.add_argument("--output")
    cmd=sub.add_parser("smoke"); cmd.add_argument("--workdir",default="runs/smoke")
    cmd=sub.add_parser("compare"); cmd.add_argument("--left",required=True); cmd.add_argument("--right",required=True); cmd.add_argument("--output")
    cmd=sub.add_parser("convert-ltx")
    for name in ("weights","assets","revision","output"): cmd.add_argument(f"--{name}",required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv); torch.set_num_threads(args.threads)
    config=load_config(args.config) if hasattr(args,"config") else None
    if args.command=="doctor": result=doctor(config)
    elif args.command=="convert-ltx":
        from .model import convert_ltx
        result=convert_ltx(args.weights,args.assets,args.revision,args.output)
    elif args.command=="prepare": result=prepare(config,FrozenEncoders(config.model,config.train.device))
    elif args.command=="validate-data": result=PreparedDataset(config.data.prepared).validate()
    elif args.command=="train": result={"checkpoint":train(config)}
    elif args.command=="probe":
        config.train.stage="probe"; config.train.probe=args.signal; config.model.targets=["action"]
        config.train.losses={"action":1.,"flow":0.,"video":0.}; result={"checkpoint":train(config)}
    elif args.command=="export": result=export_bundle(args.checkpoint,args.output)
    elif args.command=="collect":
        if not 0<=args.action_free_fraction<1: raise ValueError("action-free-fraction must lie in [0,1)")
        result=collect(config,args.episodes,args.teacher,args.source,args.action_free_fraction)
    elif args.command=="evaluate": result=evaluate(config,args.bundle,args.baseline)
    elif args.command=="import-maniskill": result=import_maniskill(config,args.trajectory,args.source,args.limit,args.replay)
    elif args.command=="import-video": result=import_video(config,args.video,args.instruction,args.source,args.episode_id)
    elif args.command=="attention": result=attention_audit(config,args.bundle,args.output,args.samples)
    elif args.command=="aggregate":
        result=aggregate(read_jsonl(args.episodes))
        if args.output: atomic_json(args.output,result)
    elif args.command=="smoke": result=smoke(args.workdir)
    elif args.command=="compare":
        result=paired_comparison(read_jsonl(args.left),read_jsonl(args.right))
        if args.output: atomic_json(args.output,result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=="__main__": main()
