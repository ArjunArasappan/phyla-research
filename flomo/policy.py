"""A single bundle/runtime interface, with reset-safe chunk execution."""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
import torch

from .config import Config, from_dict
from .geometry import PercentileStats, rotation_6d_to_matrix, matrix_to_axis_angle
from .model import FrozenEncoders, build_model, sample_joint
from .train import apply_state, autocast, load_artifact


@dataclass
class ActionChunk:
    actions: torch.Tensor
    episode_ids: list[str]
    observation_steps: list[int]
    bundle_id: str
    flow: torch.Tensor | None = None
    video: torch.Tensor | None = None


class NativeActionCodec:
    def __init__(self, stats, dimensions, low=None, high=None):
        if stats is None: raise ValueError("Bundle has no action normalization")
        self.stats=PercentileStats(**stats); self.dimensions=dimensions
        if len(self.stats.low)!=dimensions: raise ValueError("Action stats dimensions mismatch")
        self.low=low; self.high=high

    def decode(self, normalized):
        if normalized.shape[-1]!=self.dimensions or not torch.isfinite(normalized).all():
            raise ValueError("Invalid model actions")
        action=self.stats.denormalize(normalized,signed=True)
        if self.low is not None:
            action=torch.maximum(torch.minimum(action,action.new_tensor(self.high)),action.new_tensor(self.low))
        return action


def pose6d_to_delta(target,current_position,current_rotation,translation_scale,rotation_scale):
    """Standalone conversion for a root-frame delta / right-composed body rotation.

    This is not enabled for native ManiSkill control without verified controller
    conventions. target [..,10] = absolute p + first-two-column R6D + gripper.
    """
    desired=rotation_6d_to_matrix(target[...,3:9])
    delta_position=(target[...,:3]-current_position)/translation_scale
    delta_rotation=matrix_to_axis_angle(current_rotation.transpose(-1,-2)@desired)/rotation_scale
    return torch.cat([delta_position,delta_rotation,target[...,9:10]],-1).clamp(-1,1)


class BundlePolicy:
    def __init__(self,path,device="cpu",sampler=None,sampling_steps=None,shift=None):
        artifact=load_artifact(path)
        if artifact.get("probe"): raise ValueError("Privileged future-conditioned probes cannot be deployed as policies")
        self.config=from_dict(artifact["config"]); self.device=torch.device(device)
        if self.config.model.backend in {"wan","ltx"} and self.device.type!="cuda":
            raise RuntimeError(f"{self.config.model.backend} runtime requires CUDA")
        self.bundle_id=artifact.get("bundle_id",f'checkpoint-step-{artifact["step"]}')
        self.model=build_model(self.config.model).to(self.device).eval(); apply_state(self.model,artifact["weights"])
        self.encoders=FrozenEncoders(self.config.model,device)
        self.stats=artifact["stats"]
        e=self.config.eval; self.sampler=sampler or e.sampler; self.steps=sampling_steps or e.sampling_steps; self.shift=shift or e.shift
        self.episode_ids=[]

    def assert_compatible(self,config: Config):
        old=self.config.eval; new=config.eval
        for name in ["backend","embodiment","robot","control_mode","control_hz","camera_names","action_space"]:
            if getattr(old,name)!=getattr(new,name): raise ValueError(f"Bundle/environment {name} mismatch")
        if old.backend=="maniskill" and self.config.model.action_dim!=config.model.action_dim:
            raise ValueError("Bundle/environment action dimensions mismatch")

    def reset(self,episode_ids): self.episode_ids=list(episode_ids)

    @torch.no_grad()
    def predict(self,rgb,instructions,episode_ids,observation_steps,seed=0):
        if list(episode_ids)!=self.episode_ids: raise ValueError("Policy must reset to current episode IDs")
        c=self.config.model
        rgb=np.asarray(rgb)
        if rgb.shape[1]!=c.views: raise ValueError("Runtime camera count mismatch")
        obs=torch.stack([self.encoders.encode_observations(images) for images in rgb])
        encoded=[self.encoders.encode_text(s) for s in instructions]
        batch={"obs":obs.to(self.device),"view_valid":torch.ones(len(rgb),c.views,dtype=torch.bool,device=self.device),
               "text":torch.stack([x[0] for x in encoded]),"text_valid":torch.stack([x[1] for x in encoded]),
               "has_action":torch.ones(len(rgb),dtype=torch.bool,device=self.device)}
        precision="bf16" if c.backend in {"wan","ltx"} else "fp32"
        with autocast(self.device,precision):
            result=sample_joint(self.model,batch,self.steps,self.shift,self.sampler,seed)
        if "action" not in result: raise ValueError("Bundle does not predict executable actions")
        return ActionChunk(result["action"].cpu(),list(episode_ids),list(observation_steps),self.bundle_id,result.get("flow"),result.get("video"))


class ChunkExecutor:
    def __init__(self,execute_steps):
        self.execute_steps=execute_steps; self.clear()

    def clear(self): self.chunk=None; self.cursor=0; self.episode_ids=[]

    def reset(self,episode_ids): self.clear(); self.episode_ids=list(episode_ids)

    def needs_chunk(self): return self.chunk is None or self.cursor>=min(self.execute_steps,self.chunk.actions.shape[1])

    def install(self,chunk,current_steps):
        if chunk.episode_ids!=self.episode_ids or list(current_steps)!=chunk.observation_steps:
            raise ValueError("Stale chunk: episode or observation step mismatch")
        if chunk.actions.ndim!=3 or len(chunk.actions)!=len(self.episode_ids) or chunk.actions.shape[1]<self.execute_steps:
            raise ValueError("Bad action chunk shape")
        if not torch.isfinite(chunk.actions).all(): raise ValueError("Nonfinite action chunk")
        self.chunk=chunk; self.cursor=0

    def pop(self):
        if self.needs_chunk(): raise RuntimeError("No current action chunk")
        value=self.chunk.actions[:,self.cursor]; self.cursor+=1
        return value


def wilson(successes,trials):
    if trials<1: raise ValueError("No evaluation trials")
    z=1.959963984540054; p=successes/trials; scale=1+z*z/trials
    mid=(p+z*z/(2*trials))/scale
    half=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/scale
    return [max(0.,mid-half),min(1.,mid+half)]


def paired_comparison(left,right,seed=0,bootstrap=2000):
    """Task-stratified paired bootstrap for matched initializations, one seed/run."""
    left={r["episode_id"]:r for r in left}; right={r["episode_id"]:r for r in right}
    if set(left)!=set(right): raise ValueError("Paired runs must contain the same episode IDs")
    groups={}
    for key,a in left.items():
        b=right[key]
        if any(a.get(k)!=b.get(k) for k in ["seed","task","instruction"]):
            raise ValueError("Paired episode initializations/instructions differ")
        groups.setdefault(a["task"],[]).append(float(b["success_at_end"])-float(a["success_at_end"]))
    if not groups: raise ValueError("Empty comparison")
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(bootstrap): estimates.append(np.mean([rng.choice(values,len(values),replace=True).mean() for values in groups.values()]))
    return {"right_minus_left_macro_success":float(np.mean([np.mean(v) for v in groups.values()])),
            "paired_bootstrap95":np.percentile(estimates,[2.5,97.5]).tolist(),"episodes":len(left),
            "limitation":"One pair of runs; this interval does not measure training-seed variance"}
