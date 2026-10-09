"""Canonical NPZ episodes, fixed manifests, preprocessing, and mixed sampling.

No simulator or optimizer dependency. Arrays never use object/pickle dtypes.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import tempfile
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, atomic_json, digest, file_hash, encoder_signature, action_contract
from .geometry import PercentileStats, canonical_flow, oracle_tracks, query_grid, render_flow


def atomic_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent,suffix=".npz")
    try:
        with os.fdopen(fd,"wb") as f: np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def read_jsonl(path):
    with open(path) as f: return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w") as f:
            for row in rows: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def validate_episode(arrays, meta):
    rgb=arrays["rgb"]
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be uint8 [T,V,H,W,3]")
    times=arrays["times"]
    if times.shape != (len(rgb),) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError("Observation timestamps must be finite and strictly increasing")
    if len(meta["camera_names"]) != rgb.shape[1]: raise ValueError("Camera names mismatch")
    if not meta.get("instruction") or not meta.get("source"): raise ValueError("Instruction/source required")
    if "actions" in arrays:
        a=arrays["actions"]
        if a.ndim != 2 or len(a)!=len(rgb)-1 or not np.isfinite(a).all():
            raise ValueError("Robot episodes require N actions and N+1 observations")
        if not meta.get("embodiment") or not meta.get("control_mode"):
            raise ValueError("Action labels need embodiment/controller semantics")
        if "action_times" in arrays and not np.allclose(arrays["action_times"],times[:-1]):
            raise ValueError("Actions must be aligned with pre-action observations")
    for name,value in arrays.items():
        if value.dtype.hasobject: raise ValueError(f"Object/pickle arrays forbidden: {name}")


def save_episode(root, episode_id, arrays, meta):
    root=Path(root)
    if not episode_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in episode_id):
        raise ValueError("Episode ID must be path-safe")
    validate_episode(arrays,meta)
    path=root/f"{episode_id}.npz"
    if path.exists(): raise FileExistsError(path)
    atomic_npz(path,arrays)
    row={**meta,"episode_id":episode_id,"path":path.name,"sha256":file_hash(path),"frames":len(arrays["rgb"]),"has_actions":"actions" in arrays}
    atomic_json(root/f"{episode_id}.json",row)
    index=root/"episodes.jsonl"
    rows=read_jsonl(index) if index.exists() else []
    if any(r["episode_id"]==episode_id for r in rows): raise ValueError("Duplicate episode ID")
    write_jsonl(index,rows+[row])
    return row


def split_for(group, seed, val_fraction, test_fraction):
    number=int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16],16)/2**64
    if number<test_fraction: return "test"
    if number<test_fraction+val_fraction: return "val"
    return "train"


def make_windows(config: Config):
    root=Path(config.data.raw); rows=read_jsonl(root/"episodes.jsonl"); windows=[]
    seen=set()
    for row in rows:
        if row["episode_id"] in seen: raise ValueError("Duplicate raw episode")
        seen.add(row["episode_id"])
        group=f'{row["source"]}:{row.get("split_group",row["episode_id"])}'
        split=split_for(group,config.train.seed,config.data.val_fraction,config.data.test_fraction)
        if config.data.sources and row["source"] not in config.data.sources: continue
        with np.load(root/row["path"],allow_pickle=False) as ep:
            validate_episode(ep,row)
            if file_hash(root/row["path"])!=row["sha256"]: raise ValueError("Raw episode checksum changed")
            if config.model.backend == "ltx" and not np.allclose(np.diff(ep["times"]),1/config.model.video_fps,atol=1e-5):
                raise ValueError("LTX clips must match video_fps; resample human footage before ingestion")
            if ep["rgb"].shape[1] > config.model.views: raise ValueError("Too many cameras for model")
            if row["has_actions"] and ep["actions"].shape[-1]!=config.model.action_dim: raise ValueError("Action dimensions mismatch")
            if row["has_actions"]:
                for key in ["embodiment","control_mode","control_hz"]:
                    if row.get(key)!=getattr(config.eval,key): raise ValueError(f"Recorded action {key} mismatch in {row['episode_id']}")
                if not np.allclose(np.diff(ep["times"]),1/config.eval.control_hz,atol=1e-5):
                    raise ValueError("Robot observations must match the declared controller rate")
            max_start=len(ep["rgb"])-config.model.horizon-(1 if row["has_actions"] else 0)
            for start in range(0,max_start+1,config.data.stride):
                if config.data.provider=="precomputed" and start!=int(row.get("tracks_anchor",0)): continue
                identity={"episode_sha":row["sha256"],"metadata_hash":digest(row),"start":start,"horizon":config.model.horizon,"split":split,"source":row["source"]}
                windows.append({**identity,"window_id":digest(identity)[:24],"episode_id":row["episode_id"],"path":row["path"],"has_actions":row["has_actions"]})
    if not windows or not any(w["split"]=="train" for w in windows): raise ValueError("No train windows; add data or adjust split")
    return windows,{r["episode_id"]:r for r in rows}


class SpatialTracker:
    """Adapter to official SpaTrackerV2 offline predictor (optional GPU dependency)."""
    def __init__(self, config, device):
        if not str(device).startswith("cuda"): raise RuntimeError("SpatialTracker preprocessing requires a CUDA machine")
        root=Path(config.tracker_upstream).resolve()
        if not (root/"models/SpaTrackV2/models/predictor.py").is_file():
            raise RuntimeError("Set data.tracker_upstream to a pinned SpaTrackerV2 checkout")
        if not config.tracker_revision: raise ValueError("Pin data.tracker_revision")
        sys.path.insert(0,str(root))
        from models.SpaTrackV2.models.predictor import Predictor
        self.predictor=Predictor.from_pretrained(config.tracker_weights)
        self.predictor.eval(); self.predictor.to(device)
        self.front=None; self.device=device; self.config=config
        if config.tracker_mode=="monocular":
            from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
            from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
            self.front=VGGT4Track.from_pretrained(config.tracker_front).eval().to(device)
            self.preprocess_image=preprocess_image
        elif config.tracker_mode!="rgbd": raise ValueError("tracker_mode must be monocular or rgbd")

    @torch.no_grad()
    def __call__(self, rgb, episode, start, horizon, grid):
        video=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float()
        uncertainty=None
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            if self.front is not None:
                video=self.preprocess_image(video)[None]
                pred=self.front(video.to(self.device)/255.)
                depth=pred["points_map"][...,2].squeeze(0).cpu().numpy()
                intrinsic=pred["intrs"].squeeze(0).cpu().numpy()
                extrinsic=pred["poses_pred"].squeeze(0).cpu().numpy()
                uncertainty=pred["unc_metric"].squeeze(0).cpu().numpy()>.5
                video=video.squeeze(0).cpu()
            else:
                sl=slice(start,start+horizon)
                depth=episode["depth"][sl]
                intrinsic=episode["intrinsics"][sl]
                extrinsic=np.linalg.inv(episode["camera_to_world"][sl])
            uv=query_grid(video.shape[-2],video.shape[-1],grid)
            queries=np.concatenate([np.zeros((len(uv),1)),uv],axis=1).astype(np.float32)
            result=self.predictor(video,depth=depth,intrs=intrinsic,extrs=extrinsic,queries=queries,
                                  fps=1,full_point=False,iters_track=4,query_no_BA=True,fixed_cam=False,
                                  stage=1,unc_metric=uncertainty,support_frame=horizon-1,replace_ratio=.2)
        poses,_,_,_,tracks,_,visible,confidence,_=result
        def cpu(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
        poses,tracks=cpu(poses),cpu(tracks)[...,:3]
        valid=(cpu(visible)>.5)&(cpu(confidence)>.5)&np.isfinite(tracks).all(-1)
        tracks=np.nan_to_num(tracks)
        flow=canonical_flow(tracks,poses,"camera")
        return flow,valid


def extract_flow(config, episode, window, tracker=None):
    start=window["start"]; h=config.model.horizon; sl=slice(start,start+h)
    grid=config.data.grid_size
    if config.data.provider=="oracle":
        required={"depth","segmentation","intrinsics","camera_to_world","body_poses","body_ids"}
        if required-set(episode.files): raise ValueError(f"Oracle data missing: {required-set(episode.files)}")
        points,valid=oracle_tracks(episode["depth"][start],episode["segmentation"][start],episode["intrinsics"][start],
                                   episode["camera_to_world"][sl],episode["body_poses"][sl],episode["body_ids"],grid)
        return canonical_flow(points,episode["camera_to_world"][sl]),valid
    if config.data.provider=="tracker":
        return tracker(episode["rgb"][sl,0],episode,start,h,grid)
    points=episode["tracks"][sl]
    valid=episode["track_valid"][sl]
    if points.shape!=(h,grid*grid,3): raise ValueError("Precomputed tracks require matching initial query grid")
    return canonical_flow(points,episode["camera_to_world"][sl],"world"),valid.astype(bool)


def prepare(config: Config, encoders):
    config.validate(); out=Path(config.data.prepared)
    if (out/"dataset.json").exists(): raise FileExistsError("Prepared datasets are immutable; choose a new output directory")
    out.mkdir(parents=True,exist_ok=True)
    windows,episodes=make_windows(config)
    preparation={"encoder":encoder_signature(config),"data":asdict(config.data),"seed":config.train.seed,"windows":digest(windows)}
    stamp=out/"preparing.json"
    if stamp.exists() and json.loads(stamp.read_text())!=preparation:
        raise ValueError("Incomplete output belongs to a different preprocessing recipe")
    atomic_json(stamp,preparation)
    tracker=SpatialTracker(config.data,config.train.device) if config.data.provider=="tracker" else None
    accepted=[]; rejected=[]; flow_values={}; action_values=[]
    rng=np.random.default_rng(config.train.seed)
    for window in windows:
        with np.load(Path(config.data.raw)/window["path"],allow_pickle=False) as ep:
            cached=out/"tracks"/f'{window["window_id"]}.npz'
            if cached.exists():
                with np.load(cached,allow_pickle=False) as stored: flow,valid=stored["flow"],stored["valid"]
            else: flow,valid=extract_flow(config,ep,window,tracker)
            if valid.mean()<config.data.min_valid_fraction:
                rejected.append({"window_id":window["window_id"],"reason":"insufficient_valid_tracks","valid_fraction":float(valid.mean())}); continue
            if not np.isfinite(flow).all() or not np.allclose(flow[0],0): raise ValueError("Bad cumulative flow")
            if not cached.exists(): atomic_npz(cached,{"flow":flow,"valid":valid})
            accepted.append({**window,"valid_fraction":float(valid.mean())})
            if window["split"]=="train":
                values=flow[valid]
                limit=max(1,config.data.max_stat_points//max(1,len(windows)))
                values=values[rng.choice(len(values),min(limit,len(values)),replace=False)]
                flow_values.setdefault(window["source"],[]).append(values)
                if window["has_actions"]:
                    action_values.append(ep["actions"][window["start"]:window["start"]+config.model.horizon])
    if not accepted: raise ValueError("All windows rejected")
    flow_stats={source:asdict(PercentileStats.fit(np.concatenate(v))) for source,v in flow_values.items()}
    missing={w["source"] for w in accepted}-flow_stats.keys()
    if missing: raise ValueError(f"Sources lack training flow statistics: {missing}")
    action_stats=asdict(PercentileStats.fit(np.concatenate(action_values),(2,98))) if action_values else None
    stats={"flow":flow_stats,"action":action_stats}; stats_id=digest(stats)
    atomic_json(out/"stats.json",{**stats,"stats_id":stats_id})
    for window in accepted:
        with np.load(Path(config.data.raw)/window["path"],allow_pickle=False) as ep, np.load(out/"tracks"/f'{window["window_id"]}.npz',allow_pickle=False) as tracks:
            start=window["start"]; h=config.model.horizon; meta=episodes[window["episode_id"]]
            rgb=ep["rgb"][start:start+h]
            stats_f=PercentileStats(**flow_stats[window["source"]])
            rendered=render_flow(tracks["flow"],tracks["valid"],stats_f,config.data.grid_size,config.model.image_size)
            obs=encoders.encode_observations(rgb[0])
            # Future RGB excludes the current conditioning image. Repeat the
            # final valid future frame to retain the shared temporal grid.
            future=rgb[1:,0]
            future=np.concatenate([future,future[-1:]],axis=0)
            video=encoders.encode_video(torch.from_numpy(future.copy()).permute(0,3,1,2).float()/255)
            flow=encoders.encode_video(rendered)
            text,text_valid=encoders.encode_text(meta["instruction"])
            action=torch.zeros(h,config.model.action_dim)
            if window["has_actions"]:
                action=torch.from_numpy(PercentileStats(**action_stats).normalize(ep["actions"][start:start+h],signed=True))
            views=obs.shape[0]
            if meta["camera_names"] != config.eval.camera_names[:views]:
                raise ValueError(f"Camera order mismatch: {meta['camera_names']}")
            observation=torch.zeros(config.model.views,*obs.shape[1:]); observation[:views]=obs.cpu()
            sample={"obs":observation,"view_valid":torch.arange(config.model.views)<views,
                    "text":text.cpu(),"text_valid":text_valid.cpu(),"flow":flow.cpu(),"video":video.cpu(),
                    "action":action,"has_action":torch.tensor(window["has_actions"]),"source":window["source"],"window_id":window["window_id"],"episode_id":window["episode_id"]}
            path=out/"samples"/f'{window["window_id"]}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(sample,path); window["sample_sha256"]=file_hash(path)
    write_jsonl(out/"windows.jsonl",accepted); write_jsonl(out/"rejected.jsonl",rejected)
    metadata={"schema":1,"encoder_signature":encoder_signature(config),"action_contract":action_contract(config),"stats_id":stats_id,
              "preprocess":asdict(config.data),"model":asdict(config.model),"episodes":list(episodes.values()),
              "manifest_hash":digest(accepted),"windows":len(accepted),"rejected":len(rejected),
              "augmentation":"deterministic_resize_no_random_crop","temporal_policy":f"repeat_last_to_1_mod_{config.model.temporal_stride}"}
    metadata["video_window"]="frames_1_to_horizon_minus_1_then_repeat_last"
    metadata["dataset_id"]=digest(metadata)
    atomic_json(out/"dataset.json",metadata)
    return metadata


def torch_load(path):
    # These checkpoints/samples are generated by this repository. Do not load untrusted pickles.
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")


class PreparedDataset(Dataset):
    def __init__(self, root, split="train", sources=(), labeled_only=False, label_fraction=1., seed=0):
        self.root=Path(root); self.metadata=json.loads((self.root/"dataset.json").read_text())
        if digest({k:v for k,v in self.metadata.items() if k!="dataset_id"})!=self.metadata["dataset_id"]:
            raise ValueError("Dataset metadata checksum mismatch")
        all_rows=read_jsonl(self.root/"windows.jsonl")
        if digest(all_rows)!=self.metadata["manifest_hash"]: raise ValueError("Manifest checksum mismatch")
        self.rows=[r for r in all_rows if r["split"]==split and (not sources or r["source"] in sources) and (not labeled_only or r["has_actions"])]
        if label_fraction<1:
            episodes=sorted({r["episode_id"] for r in self.rows})
            rng=np.random.default_rng(seed); rng.shuffle(episodes)
            keep=set(episodes[:max(1,int(len(episodes)*label_fraction))])
            self.rows=[r for r in self.rows if r["episode_id"] in keep]
        if not self.rows: raise ValueError(f"No {split} examples after source/label filtering")

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]; path=self.root/"samples"/f'{row["window_id"]}.pt'
        sample=torch_load(path)
        return sample

    def validate(self):
        for row in self.rows:
            path=self.root/"samples"/f'{row["window_id"]}.pt'
            if file_hash(path)!=row["sample_sha256"]: raise ValueError(f"Sample checksum mismatch: {path}")
            for key,value in torch_load(path).items():
                if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {key} in {path}")
        return {"split_windows":len(self.rows),"dataset_id":self.metadata["dataset_id"]}


def collate(samples):
    return {k:torch.stack([s[k] for s in samples]) if isinstance(samples[0][k],torch.Tensor) else [s[k] for s in samples] for k in samples[0]}


class MixtureSampler:
    """Stateless global-batch sampling; exact resume for a fixed world size."""
    def __init__(self, dataset, weights, batch_size, seed=0, rank=0, world_size=1):
        self.groups={}
        for i,row in enumerate(dataset.rows): self.groups.setdefault(row["source"],[]).append(i)
        self.sources=sorted(self.groups)
        if weights and set(weights)!=set(self.sources):
            raise ValueError(f"Weights must exactly cover selected sources: {self.sources}")
        weight=np.array([weights.get(s,len(self.groups[s])) for s in self.sources],dtype=float)
        self.probs=weight/weight.sum(); self.batch_size=batch_size; self.seed=seed; self.rank=rank; self.world_size=world_size

    def indices(self, microstep):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,microstep]))
        sources=rng.choice(self.sources,size=self.batch_size*self.world_size,p=self.probs)
        result=[int(rng.choice(self.groups[source])) for source in sources]
        return result[self.rank*self.batch_size:(self.rank+1)*self.batch_size]
