"""One trainer for mixed mid-training, robot SFT, and privileged probes."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import tempfile
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import Config, atomic_json, digest, encoder_signature, file_hash, model_signature, action_contract, from_dict
from .data import PreparedDataset, MixtureSampler, collate, read_jsonl, torch_load
from .model import build_model, modality_loss, noisy_targets


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix=".tmp"); os.close(fd)
    try:
        torch.save(value,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def move_batch(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def autocast(device,precision):
    return torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16) if precision=="bf16" else nullcontext()


def model_state(model):
    state=model.state_dict()
    if model.config.backend=="tiny": return {k:v.detach().cpu() for k,v in state.items()}
    trainable={k for k,p in model.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu() for k,v in state.items() if k in trainable}


def apply_state(model,state):
    expected=set(model_state(model))
    if set(state)!=expected: raise ValueError(f"Weight key mismatch. Missing={expected-set(state)}, unexpected={set(state)-expected}")
    model.load_state_dict(state,strict=model.config.backend=="tiny")


def rng_state():
    return {"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"]); random.setstate(state["python"])
    if state["cuda"] is not None: torch.cuda.set_rng_state_all(state["cuda"])


def resume_signature(config):
    t=asdict(config.train)
    for k in ["output","steps","save_every","validate_every","validation_batches","resume","init"]: t.pop(k)
    return digest({"model":model_signature(config),"train":t,"weights":config.data.source_weights,"sources":config.data.sources})


def forward_loss(model,batch,config):
    if config.model.architecture=="bc":
        return modality_loss(model(batch),{"action":batch["action"]},batch["has_action"],{"action":1.})
    known={config.train.probe:batch[config.train.probe]} if config.train.probe else {}
    targets=[name for name in config.model.targets if name not in known]
    noisy,vel,t=noisy_targets(batch,targets)
    prediction=model(batch,noisy,t,known)
    return modality_loss(prediction,vel,batch["has_action"],config.train.losses)


@torch.no_grad()
def validate(model,dataset,config,device):
    model.eval(); total=0; metrics={}
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.train.seed+991)
        for start in range(0,min(len(dataset),config.train.batch_size*config.train.validation_batches),config.train.batch_size):
            batch=move_batch(collate([dataset[i] for i in range(start,min(start+config.train.batch_size,len(dataset)))]),device)
            with autocast(device,config.train.precision): loss,terms=forward_loss(model,batch,config)
            n=len(batch["obs"]); total+=n
            for k,value in {"loss":loss.detach(),**terms}.items(): metrics[k]=metrics.get(k,0)+float(value)*n
    model.train()
    return {k:v/total for k,v in metrics.items()}


def train(config: Config):
    config.validate()
    if config.train.workers: raise ValueError("Stateless sampler currently loads in-process; set workers=0")
    world=int(os.environ.get("WORLD_SIZE",1)); rank=int(os.environ.get("RANK",0)); local=int(os.environ.get("LOCAL_RANK",0))
    device=torch.device(f"cuda:{local}" if world>1 and config.train.device.startswith("cuda") else config.train.device)
    if device.type=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use configs/smoke.yaml on this laptop")
        torch.cuda.set_device(device)
    if config.model.backend in {"wan","ltx"} and config.model.architecture=="joint" and device.type!="cuda":
        raise RuntimeError(f"{config.model.backend} training requires CUDA")
    if world>1: dist.init_process_group("nccl" if device.type=="cuda" else "gloo")
    try:
        random.seed(config.train.seed); np.random.seed(config.train.seed); torch.manual_seed(config.train.seed)
        dataset=PreparedDataset(config.data.prepared,"train",config.data.sources,
                                labeled_only=config.train.stage in {"sft","probe"} or config.model.architecture=="bc",label_fraction=config.train.label_fraction,seed=config.train.seed)
        if dataset.metadata["encoder_signature"]!=encoder_signature(config): raise ValueError("Dataset encoder/action signature mismatch")
        if dataset.metadata["action_contract"]!=action_contract(config): raise ValueError("Dataset/controller/camera contract mismatch")
        stats=json.loads((Path(config.data.prepared)/"stats.json").read_text())
        if stats["stats_id"]!=dataset.metadata["stats_id"] or digest({"flow":stats["flow"],"action":stats["action"]})!=stats["stats_id"]:
            raise ValueError("Normalization artifact checksum mismatch")
        if "action" in config.model.targets and not any(r["has_actions"] for r in dataset.rows): raise ValueError("Action training has no labeled data")
        if config.train.stage=="probe" and config.model.targets!=["action"]:
            raise ValueError("Privileged probes use model.targets: [action] only")
        model=build_model(config.model).to(device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.train.lr,
                                    betas=tuple(config.train.betas),weight_decay=config.train.weight_decay)
        step=0
        if config.train.init and config.train.resume: raise ValueError("Use either init (fresh optimizer) or resume")
        if config.train.init:
            initial=load_artifact(config.train.init)
            if initial["encoder_signature"]!=encoder_signature(config) or initial["stats_id"]!=stats["stats_id"]:
                raise ValueError("Stage handoff cannot silently change encoders or normalization")
            if action_contract(config)!=action_contract(from_dict(initial["config"])):
                raise ValueError("Stage handoff changes the controller/camera contract")
            apply_state(model,initial["weights"])
        if config.train.resume:
            initial=load_artifact(config.train.resume)
            if initial["dataset_id"]!=dataset.metadata["dataset_id"] or initial["resume_signature"]!=resume_signature(config) or initial["world_size"]!=world:
                raise ValueError("Exact resume requires same dataset, objective, batch, seed, and world size")
            apply_state(model,initial["weights"]); optimizer.load_state_dict(initial["optimizer"]); step=initial["step"]
            restore_rng(initial["rng_by_rank"][rank])
        else:
            torch.manual_seed(config.train.seed+rank*100003)
        wrapped=DistributedDataParallel(model,device_ids=[local] if device.type=="cuda" else None,find_unused_parameters=True) if world>1 else model
        sampler=MixtureSampler(dataset,config.data.source_weights,config.train.batch_size,config.train.seed,rank,world)
        val_rows=[r for r in read_jsonl(Path(config.data.prepared)/"windows.jsonl") if r["split"]=="val" and
                  (not config.data.sources or r["source"] in config.data.sources) and (config.train.stage=="midtrain" or r["has_actions"])]
        validation=PreparedDataset(config.data.prepared,"val",config.data.sources,labeled_only=config.train.stage!="midtrain") if val_rows else None
        output=Path(config.train.output)
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
            if (output/"config.json").exists() and not config.train.resume:
                raise FileExistsError("Run already exists; use resume or choose a new output")
            atomic_json(output/"config.json",asdict(config))
            print(json.dumps({"parameters":model.parameter_counts(),"global_batch":world*config.train.batch_size*config.train.accumulation,"world_size":world,"dataset_id":dataset.metadata["dataset_id"]}),flush=True)
        model.train()
        for update in range(step,config.train.steps):
            begin=time.monotonic(); optimizer.zero_grad(set_to_none=True); logs={}; sources={}
            for micro in range(config.train.accumulation):
                batch=move_batch(collate([dataset[i] for i in sampler.indices(update*config.train.accumulation+micro)]),device)
                for s in batch["source"]: sources[s]=sources.get(s,0)+1
                context=wrapped.no_sync() if world>1 and micro<config.train.accumulation-1 else nullcontext()
                with context,autocast(device,config.train.precision):
                    loss,terms=forward_loss(wrapped,batch,config)
                    if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at step {update}")
                    (loss/config.train.accumulation).backward()
                for k,value in {"loss":loss.detach(),**terms}.items(): logs[k]=logs.get(k,0)+float(value)/config.train.accumulation
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.grad_clip,error_if_nonfinite=True)
            optimizer.step(); logs.update(step=update+1,grad_norm=float(norm),seconds=time.monotonic()-begin)
            if world>1:
                for k in ["loss","flow","action","video"]:
                    if k in logs:
                        tensor=torch.tensor(logs[k],device=device); dist.all_reduce(tensor); logs[k]=float(tensor/world)
            if rank==0:
                logs["source_counts_rank0"]=sources
                if validation is not None and (update+1)%config.train.validate_every==0:
                    logs["validation"]=validate(model,validation,config,device)
                with open(output/"metrics.jsonl","a") as f: f.write(json.dumps(logs,allow_nan=False)+"\n")
                print(json.dumps(logs),flush=True)
            if (update+1)%config.train.save_every==0 or update+1==config.train.steps:
                state=rng_state(); states=[None]*world
                if world>1: dist.all_gather_object(states,state)
                else: states=[state]
                if rank==0:
                    payload={"schema":1,"kind":"checkpoint","config":asdict(config),"weights":model_state(model),"optimizer":optimizer.state_dict(),
                             "step":update+1,"rng_by_rank":states,"world_size":world,"resume_signature":resume_signature(config),
                             "dataset_id":dataset.metadata["dataset_id"],"encoder_signature":encoder_signature(config),"model_signature":model_signature(config),
                             "stats_id":stats["stats_id"],"stats":stats,"probe":config.train.probe,"parameters":model.parameter_counts()}
                    path=output/f"step_{update+1:06d}.pt"
                    atomic_torch(path,payload); atomic_json(path.with_suffix(".json"),{"sha256":file_hash(path),"step":update+1})
                    atomic_json(output/"latest.json",{"checkpoint":path.name,"step":update+1})
        return str(output/f"step_{config.train.steps:06d}.pt")
    finally:
        if world>1 and dist.is_initialized(): dist.destroy_process_group()


def load_artifact(path):
    path=Path(path)
    if path.is_dir():
        header=json.loads((path/"bundle.json").read_text())
        identity={k:v for k,v in header.items() if k!="bundle_id"}
        if digest(identity)!=header["bundle_id"]: raise ValueError("Bundle metadata checksum mismatch")
        if file_hash(path/"weights.pt")!=header["weights_sha256"]: raise ValueError("Bundle checksum mismatch")
        header["weights"]=torch_load(path/"weights.pt")
        return header
    checksum=path.with_suffix(".json")
    if not checksum.exists() or file_hash(path)!=json.loads(checksum.read_text())["sha256"]:
        raise ValueError("Checkpoint missing completion marker or checksum mismatch")
    return torch_load(path)


def export_bundle(checkpoint_path,output):
    state=load_artifact(checkpoint_path); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    atomic_torch(output/"weights.pt",state["weights"])
    header={k:state[k] for k in ["config","dataset_id","encoder_signature","model_signature","stats_id","stats","probe","parameters","step"]}
    header.update(schema=1,kind="policy_bundle",weights_sha256=file_hash(output/"weights.pt"))
    header["bundle_id"]=digest(header)
    atomic_json(output/"bundle.json",header)
    return header
