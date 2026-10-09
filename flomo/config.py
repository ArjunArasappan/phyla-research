"""Small strict configuration and immutable artifact utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import os
import tempfile
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backend: str = "tiny"
    architecture: str = "joint"
    checkpoint: str = ""
    upstream: str = ""
    upstream_commit: str = ""
    checkpoint_revision: str = ""
    image_size: int = 32
    latent_channels: int = 6
    spatial_stride: int = 4
    text_dim: int = 32
    text_length: int = 16
    width: int = 64
    layers: int = 2
    heads: int = 4
    horizon: int = 16
    action_dim: int = 2
    views: int = 1
    targets: list[str] = field(default_factory=lambda: ["flow", "action"])
    lora_rank: int = 64
    lora_alpha: float = 128.0
    gradient_checkpointing: bool = False
    video_fps: float = 20.0

    @property
    def temporal_stride(self):
        return 8 if self.backend == "ltx" else 4


@dataclass
class DataConfig:
    raw: str = "data/raw"
    prepared: str = "data/prepared"
    stride: int = 16
    grid_size: int = 8
    val_fraction: float = 0.2
    test_fraction: float = 0.1
    provider: str = "oracle"
    tracker_upstream: str = ""
    tracker_revision: str = ""
    tracker_front: str = "Yuxihenry/SpatialTrackerV2_Front"
    tracker_weights: str = "Yuxihenry/SpatialTrackerV2-Offline"
    tracker_mode: str = "monocular"
    min_valid_fraction: float = 0.5
    max_stat_points: int = 200000
    source_weights: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    stage: str = "midtrain"
    output: str = "runs/midtrain"
    device: str = "cpu"
    precision: str = "fp32"
    seed: int = 0
    steps: int = 100
    batch_size: int = 4
    accumulation: int = 1
    lr: float = 2.8e-4
    betas: list[float] = field(default_factory=lambda: [0.95, 0.999])
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    losses: dict[str, float] = field(default_factory=lambda: {"flow": 1., "action": .1, "video": 0.})
    save_every: int = 50
    validate_every: int = 25
    validation_batches: int = 4
    init: str = ""
    resume: str = ""
    probe: str = ""
    label_fraction: float = 1.0
    workers: int = 0


@dataclass
class EvalConfig:
    backend: str = "toy"
    env_id: str = "toy_point"
    robot: str = "point"
    control_mode: str = "xy_delta"
    control_hz: int = 20
    camera_names: list[str] = field(default_factory=lambda: ["front"])
    embodiment: str = "toy_point"
    episodes: int = 10
    num_envs: int = 1
    horizon: int = 96
    execute_steps: int = 16
    sampler: str = "euler"
    sampling_steps: int = 10
    shift: float = 5.
    device: str = "cpu"
    seed: int = 1000
    output: str = "runs/eval"
    instruction: str = "move the point to the target"
    record: bool = False
    ledger: str = ""
    action_space: str = "native"
    flow_source: str = ""
    sim_backend: str = "physx_cpu"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self):
        m, d, t, e = self.model, self.data, self.train, self.eval
        if m.backend not in {"tiny", "wan", "ltx"} or m.architecture not in {"joint","bc"} or d.provider not in {"oracle", "tracker", "precomputed"}:
            raise ValueError("Unknown model backend or flow provider")
        if not m.targets or not set(m.targets) <= {"flow", "action", "video"}:
            raise ValueError("targets must be a nonempty subset of flow/action/video")
        if m.architecture=="bc" and m.targets!=["action"]:
            raise ValueError("BC predicts action only")
        if m.horizon < 2 or m.views < 1 or m.action_dim < 1 or m.image_size % m.spatial_stride:
            raise ValueError("Invalid horizon, views, actions or image/latent dimensions")
        if (m.backend != "ltx" and (m.image_size // m.spatial_stride) % 2) or m.width % m.heads:
            raise ValueError("Latent spatial size must be even; width divisible by heads")
        if m.backend == "wan" and (m.latent_channels != 48 or m.spatial_stride != 16 or m.text_dim != 4096):
            raise ValueError("Released Wan TI2V-5B requires C=48, stride=16, text_dim=4096")
        if m.backend == "ltx":
            if (m.latent_channels, m.spatial_stride, m.text_dim) != (128, 32, 4096):
                raise ValueError("LTX 2B requires C=128, stride=32, text_dim=4096")
            if not 0 < m.video_fps < 1000:
                raise ValueError("LTX video_fps must be a finite positive rate below 1000")
            if m.video_fps != e.control_hz:
                raise ValueError("LTX video_fps must match the native controller rate; resample human videos to this rate")
            if e.sampler != "euler":
                raise ValueError("LTX currently uses the joint Euler flow solver")
        if d.stride < 1 or not 0 <= d.val_fraction + d.test_fraction < 1 or min(d.val_fraction,d.test_fraction) < 0:
            raise ValueError("Invalid stride or split fractions")
        if not 0 < d.min_valid_fraction <= 1 or d.max_stat_points < 1:
            raise ValueError("Invalid flow quality/statistics budget")
        if t.stage not in {"midtrain", "sft", "probe"} or t.probe not in {"", "flow", "video"}:
            raise ValueError("Unknown training stage/probe")
        if t.stage == "probe" and (not t.probe or "action" not in m.targets):
            raise ValueError("Probe requires a future signal and action output")
        if not 0 < t.label_fraction <= 1 or min(t.batch_size,t.accumulation,t.steps,t.save_every,t.validate_every) < 1:
            raise ValueError("Invalid batch, steps, intervals or label_fraction")
        if t.precision not in {"fp32", "bf16"} or t.lr <= 0:
            raise ValueError("Invalid precision or learning rate")
        if set(t.losses) - {"flow","action","video"} or any(v < 0 for v in t.losses.values()):
            raise ValueError("Invalid modality losses")
        if not any(t.losses.get(k,0) for k in m.targets):
            raise ValueError("At least one enabled target must have a positive loss")
        if e.backend not in {"toy", "maniskill"} or e.sampler not in {"euler", "unipc"}:
            raise ValueError("Unknown simulation backend or sampler")
        if not 1 <= e.execute_steps <= m.horizon or min(e.episodes,e.num_envs,e.horizon,e.sampling_steps) < 1 or e.shift <= 0:
            raise ValueError("Invalid evaluation horizon/batch/solver settings")
        if len(e.camera_names) != m.views:
            raise ValueError("camera_names must match model.views")
        if d.source_weights and any(v <= 0 for v in d.source_weights.values()):
            raise ValueError("Source weights must be positive")
        return self


def from_dict(value: dict) -> Config:
    allowed = {f.name for f in fields(Config)}
    if set(value) - allowed:
        raise ValueError(f"Unknown config sections: {set(value)-allowed}")
    classes = {"model":ModelConfig,"data":DataConfig,"train":TrainConfig,"eval":EvalConfig}
    result = {}
    for name, cls in classes.items():
        values = value.get(name,{})
        if not isinstance(values,dict): raise ValueError(f"{name} must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown: raise ValueError(f"Unknown {name} settings: {unknown}")
        result[name] = cls(**values)
    return Config(**result).validate()


def load_config(path: str | Path) -> Config:
    with open(path) as f: value = yaml.safe_load(f)
    return from_dict(value or {})


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def model_signature(config: Config) -> str:
    m = asdict(config.model)
    # Paths are deployment locations; revision and dimensions carry compatibility.
    m.pop("upstream"); m.pop("checkpoint"); m.pop("gradient_checkpointing")
    if config.model.backend != "ltx": m.pop("video_fps")  # Keep existing Wan/tiny identities stable.
    return digest(m)


def encoder_signature(config: Config) -> str:
    m=asdict(config.model)
    keys=("backend","upstream_commit","checkpoint_revision","image_size","latent_channels",
          "spatial_stride","text_dim","text_length","horizon","action_dim","views")
    return digest({k:m[k] for k in keys})


def action_contract(config: Config) -> dict:
    e=asdict(config.eval)
    return {k:e[k] for k in ["backend","embodiment","robot","control_mode","control_hz","camera_names","action_space"]}
