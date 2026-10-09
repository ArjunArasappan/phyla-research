"""Strict trainable allowlist and exact frozen-base integrity for real LoRA SFT."""
from __future__ import annotations
import hashlib
import torch
from .model import LoRALinear

def frozen_digest(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:continue
  h.update(name.encode());h.update(str(tuple(p.shape)).encode())
  # Chunking bounds temporary CPU allocations while checking every byte.
  flat=p.detach().reshape(-1)
  for start in range(0,flat.numel(),4_000_000):
   value=flat[start:start+4_000_000].cpu().contiguous().view(torch.uint8)
   h.update(value.numpy().tobytes())
 return h.hexdigest()

def assert_lora_only(model,optimizer=None):
 if model.config.backend not in {"wan","ltx"}:return None
 allowed=set();adapters=0
 for module in model.modules():
  if isinstance(module,LoRALinear):
   allowed.update(id(p) for part in (module.a,module.b) for p in part.parameters());adapters+=1
 for name in ("action_in","action_out"):
  allowed.update(id(p) for p in getattr(model,name).parameters())
 allowed.add(id(model.action_position))
 trainable={id(p) for p in model.parameters() if p.requires_grad}
 unexpected=[n for n,p in model.named_parameters() if p.requires_grad and id(p) not in allowed]
 if not adapters or unexpected:raise ValueError(f"LoRA-only SFT violation: adapters={adapters}, unexpected={unexpected}")
 if allowed!=trainable:raise ValueError("Registered LoRA/action interfaces are unexpectedly frozen")
 if optimizer is not None:
  params=[p for group in optimizer.param_groups for p in group["params"]]
  if {id(p) for p in params}!=trainable or len(params)!=len(trainable):raise ValueError("Optimizer differs from LoRA/action allowlist")
 return {"adapters":adapters,"trainable_names":[n for n,p in model.named_parameters() if p.requires_grad],"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"frozen_base_sha256":frozen_digest(model)}

def assert_base_unchanged(model,audit):
 if audit is None:return
 after=frozen_digest(model)
 if after!=audit["frozen_base_sha256"]:raise ValueError("Frozen pretrained base changed during LoRA SFT")
 audit["verified_after_training"]=True
