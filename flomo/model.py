"""Frozen codecs and one joint motion/action transformer.

The tiny model is a CPU integration backend, not a pretrained Wan substitute.
Wan adapters reuse official weights/blocks but explicitly pack mixed modalities.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import sys
import json
import re
import shutil
from importlib.metadata import version

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, atomic_json, file_hash


LTX_DIFFUSERS_VERSION = "0.33.1"
LTX_VARIANT = "ltxv-2b-0.9.6-dev-04-25"
LTX_CONVERTER_VERSION = 2


def require_ltx_dependencies():
    try:
        installed = version("diffusers")
    except Exception as exc:
        raise RuntimeError("Install the ltx extra: pip install '.[ltx]'") from exc
    if installed != LTX_DIFFUSERS_VERSION:
        raise RuntimeError(f"LTX adapter requires diffusers=={LTX_DIFFUSERS_VERSION}, found {installed}")


def ltx_checkpoint(config):
    """Only local, explicitly converted non-distilled 2B assets are loaded."""
    require_ltx_dependencies()
    root = Path(config.checkpoint)
    marker = root / "flomo_ltx.json"
    if not config.checkpoint_revision or not marker.is_file():
        raise ValueError("Pin checkpoint_revision and run convert-ltx before loading LTX")
    info = json.loads(marker.read_text())
    if (info.get("variant"), info.get("revision"), info.get("diffusers_version"), info.get("converter_version")) != (
        LTX_VARIANT, config.checkpoint_revision, LTX_DIFFUSERS_VERSION, LTX_CONVERTER_VERSION
    ):
        raise ValueError("LTX checkpoint variant/revision/converter mismatch")
    return root


def convert_ltx_state(source, component):
    """Backport the v0.9.5/0.9.6 VAE encoder mapping absent in Diffusers 0.33.1.

    Decoder/statistics and transformer renames use the pinned upstream helpers.
    Only structural renames occur: no tensor data or shapes are synthesized.
    """
    from diffusers.loaders.single_file_utils import (
        convert_ltx_transformer_checkpoint_to_diffusers, convert_ltx_vae_checkpoint_to_diffusers
    )
    if component == "transformer":
        return convert_ltx_transformer_checkpoint_to_diffusers(dict(source))
    if component != "vae": raise ValueError("Unknown LTX component")
    prefix = "vae.encoder.down_blocks."
    result = convert_ltx_vae_checkpoint_to_diffusers({k:v for k,v in source.items() if not k.startswith(prefix)})
    blocks = ("down_blocks.0", "down_blocks.0.downsamplers.0", "down_blocks.1",
              "down_blocks.1.downsamplers.0", "down_blocks.2", "down_blocks.2.downsamplers.0",
              "down_blocks.3", "down_blocks.3.downsamplers.0", "mid_block")
    for key,value in source.items():
        if not key.startswith(prefix): continue
        match = re.fullmatch(r"vae\.encoder\.down_blocks\.(\d+)\.(.+)",key)
        if match is None or int(match[1]) >= len(blocks): raise ValueError(f"Unknown LTX encoder block: {key}")
        suffix = match[2].replace("res_blocks","resnets").replace("conv_shortcut","conv_shortcut.conv")
        renamed = f"encoder.{blocks[int(match[1])]}.{suffix}"
        if renamed in result: raise ValueError(f"Duplicate converted tensor: {renamed}")
        result[renamed] = value
    return result


def strict_component_state(model, state):
    expected = model.state_dict()
    if expected.keys() != state.keys():
        raise ValueError(f"LTX conversion key mismatch: missing={expected.keys()-state.keys()}, extra={state.keys()-expected.keys()}")
    wrong = [k for k in expected if expected[k].shape != state[k].shape]
    if wrong: raise ValueError(f"LTX conversion shape mismatch: {wrong}")
    model.load_state_dict(state, strict=True, assign=True)


def convert_ltx(weights, assets, revision, output):
    """Convert the official single-file checkpoint once, without network fetches.

    assets is a pinned local Diffusers config/tokenizer/T5 snapshot for the 2B
    family (official 0.9.5 configs); the 0.9.6 file supplies DiT and VAE weights.
    Conversion is CPU work and should be done on the GPU host with enough RAM.
    """
    require_ltx_dependencies()
    weights, assets, output = Path(weights), Path(assets), Path(output)
    if weights.name != LTX_VARIANT + ".safetensors" or not weights.is_file():
        raise ValueError(f"Use the official non-distilled {LTX_VARIANT}.safetensors file")
    if not revision or not assets.is_dir(): raise ValueError("Local assets and a pinned revision are required")
    if output.exists(): raise FileExistsError(output)
    from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from safetensors.torch import load_file
    for folder in ("text_encoder","tokenizer"):
        if not (assets/folder).is_dir(): raise ValueError(f"Missing local LTX assets: {folder}")
    tc = json.loads((assets/"transformer/config.json").read_text())
    vc = json.loads((assets/"vae/config.json").read_text())
    text_config = json.loads((assets/"text_encoder/config.json").read_text())
    if text_config.get("d_model") != 4096: raise ValueError("LTX T5 assets require d_model=4096")
    if not any(p.suffix in {".safetensors",".bin"} for p in (assets/"text_encoder").iterdir()):
        raise ValueError("Local LTX assets are missing T5 weights")
    if not (assets/"tokenizer/tokenizer_config.json").is_file(): raise ValueError("Local LTX assets are missing tokenizer configuration")
    if (tc["in_channels"], tc["num_layers"], tc["num_attention_heads"], vc["latent_channels"]) != (128,28,32,128):
        raise ValueError("Assets must describe the non-distilled LTX 2B architecture")
    source = load_file(str(weights), device="cpu")
    # Write in a sibling temporary directory. A failed conversion leaves no
    # partially initialized model or misleading completion marker at output.
    output.parent.mkdir(parents=True,exist_ok=True)
    import tempfile
    temporary = Path(tempfile.mkdtemp(prefix=output.name+"-",dir=output.parent))
    try:
        for name,cls,cfg in (("transformer",LTXVideoTransformer3DModel,tc),("vae",AutoencoderKLLTXVideo,vc)):
            with torch.device("meta"): model = cls.from_config(cfg)
            state = convert_ltx_state(source,name)
            strict_component_state(model,state)
            model.to(dtype=torch.bfloat16).save_pretrained(temporary/name,safe_serialization=True)
            del model, state
        del source
        # T5/tokenizer are already in the local Diffusers format. Copy them
        # without allocating a second large text encoder during conversion.
        for folder in ("text_encoder","tokenizer"):
            shutil.copytree(assets/folder,temporary/folder)
        info = {"variant": LTX_VARIANT, "revision": revision, "diffusers_version": LTX_DIFFUSERS_VERSION,
                "converter_version": LTX_CONVERTER_VERSION, "source_sha256": file_hash(weights),
                "assets": str(assets.resolve()), "asset_config_sha256": {
                    name:file_hash(assets/name/"config.json") for name in ("transformer","vae","text_encoder")}}
        atomic_json(temporary/"flomo_ltx.json",info)
        temporary.rename(output)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return info


def add_wan_path(config: ModelConfig):
    root=Path(config.upstream).resolve()
    if not (root/"wan/modules/model.py").is_file():
        raise RuntimeError("Set model.upstream to a pinned official Wan2.2 checkout")
    if not config.upstream_commit or not config.checkpoint_revision:
        raise ValueError("Pin upstream_commit and checkpoint_revision before using Wan")
    sys.path.insert(0,str(root))


def pad_video(video, temporal_stride=4):
    """Pad to the codec's first-frame-plus-groups layout without truncation."""
    if len(video) < 1 or temporal_stride < 1: raise ValueError("Empty video or invalid temporal stride")
    extra=(-(len(video)-1))%temporal_stride
    return torch.cat([video,video[-1:].expand(extra,-1,-1,-1)],0) if extra else video


class FrozenEncoders:
    def __init__(self, config: ModelConfig, device="cpu"):
        self.config=config; self.device=torch.device(device); self.vae=None; self.text_model=None; self.tokenizer=None
        if config.backend=="wan":
            if self.device.type!="cuda": raise RuntimeError("Wan codecs require the CUDA backend; use tiny for CPU checks")
            add_wan_path(config)
            from wan.modules.vae2_2 import Wan2_2_VAE
            from wan.modules.t5 import T5EncoderModel
            root=Path(config.checkpoint)
            self.vae=Wan2_2_VAE(vae_pth=str(root/"Wan2.2_VAE.pth"),device=device,dtype=torch.bfloat16)
            self.text_model=T5EncoderModel(text_len=config.text_length,device=device,dtype=torch.bfloat16,
                checkpoint_path=str(root/"models_t5_umt5-xxl-enc-bf16.pth"),tokenizer_path=str(root/"google/umt5-xxl"))
        elif config.backend == "ltx":
            if self.device.type != "cuda": raise RuntimeError("LTX codecs require CUDA; use tiny for CPU smoke")
            root = ltx_checkpoint(config)
            from diffusers import AutoencoderKLLTXVideo
            from transformers import T5EncoderModel, T5TokenizerFast
            self.vae = AutoencoderKLLTXVideo.from_pretrained(root, subfolder="vae", local_files_only=True,
                torch_dtype=torch.bfloat16).to(self.device).eval().requires_grad_(False)
            self.text_model = T5EncoderModel.from_pretrained(root, subfolder="text_encoder", local_files_only=True,
                torch_dtype=torch.bfloat16).to(self.device).eval().requires_grad_(False)
            self.tokenizer = T5TokenizerFast.from_pretrained(root, subfolder="tokenizer", local_files_only=True)
            if (self.vae.config.latent_channels, self.vae.spatial_compression_ratio,
                self.vae.temporal_compression_ratio, self.text_model.config.d_model) != (128, 32, 8, config.text_dim):
                raise ValueError("LTX checkpoint codecs disagree with configuration")

    @torch.no_grad()
    def encode_video(self, video):
        video=video.to(self.device).float()
        if video.ndim!=4 or video.shape[1]!=3: raise ValueError("Expected [T,3,H,W] video")
        video=F.interpolate(video,size=(self.config.image_size,)*2,mode="bilinear",align_corners=False)
        video=pad_video(video,self.config.temporal_stride)
        if self.config.backend == "ltx":
            dtype = next(self.vae.parameters()).dtype
            latents = self.vae.encode((video*2-1).permute(1,0,2,3)[None].to(dtype)).latent_dist.mode()
            result = self._ltx_scale(latents)[0]
        elif self.vae is not None:
            result=self.vae.encode([(video*2-1).permute(1,0,2,3)])[0]
        else:
            pooled=F.avg_pool2d(video*2-1,self.config.spatial_stride)
            chunks=[pooled[:1]]+[pooled[i:i+4].mean(0,keepdim=True) for i in range(1,len(pooled),4)]
            latent=torch.cat(chunks,0).permute(1,0,2,3)
            result=latent.repeat(math.ceil(self.config.latent_channels/3),1,1,1)[:self.config.latent_channels]
        expected_t=1+(len(video)-1)//self.config.temporal_stride
        if result.shape[:2]!=(self.config.latent_channels,expected_t):
            raise ValueError(f"VAE shape mismatch: {result.shape}")
        return result.float()

    @torch.no_grad()
    def encode_observations(self, rgb):
        rgb=np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=4: raise ValueError("Expected uint8 [V,H,W,3]")
        return torch.stack([self.encode_video(torch.from_numpy(image.copy()).permute(2,0,1)[None].float()/255) for image in rgb])

    @torch.no_grad()
    def encode_text(self, instruction):
        c=self.config
        if self.config.backend == "ltx":
            tokens = self.tokenizer(instruction, padding="max_length", max_length=c.text_length,
                                    truncation=True, add_special_tokens=True, return_tensors="pt").to(self.device)
            hidden = self.text_model(**tokens).last_hidden_state[0].float()
            mask = tokens.attention_mask[0].bool()
            return hidden * mask[:,None], mask
        elif self.text_model is not None:
            hidden=self.text_model([instruction],self.device)[0].float()
        else:
            words=instruction.split()[:c.text_length] or [""]
            embeddings=[]
            for word in words:
                seed=int(hashlib.sha256(word.encode()).hexdigest()[:8],16)
                embeddings.append(torch.randn(c.text_dim,generator=torch.Generator().manual_seed(seed)))
            hidden=torch.stack(embeddings).to(self.device)
        result=torch.zeros(c.text_length,c.text_dim,device=self.device)
        result[:len(hidden)]=hidden
        return result,torch.arange(c.text_length,device=self.device)<len(hidden)

    @torch.no_grad()
    def decode_video(self, latent, frames):
        if self.config.backend == "ltx":
            dtype = next(self.vae.parameters()).dtype
            value = self._ltx_scale(latent[None].to(self.device), inverse=True).to(dtype)
            timestep = torch.zeros(1,device=self.device,dtype=dtype) if self.vae.config.timestep_conditioning else None
            result = (self.vae.decode(value, timestep).sample[0].permute(1,0,2,3)+1)/2
        elif self.vae is not None:
            result=(self.vae.decode([latent.to(self.device)])[0].permute(1,0,2,3)+1)/2
        else:
            x=latent[:3].permute(1,0,2,3)
            x=torch.cat([x[:1],x[1:].repeat_interleave(4,0)],0)
            result=(F.interpolate(x,size=(self.config.image_size,)*2,mode="bilinear",align_corners=False)+1)/2
        if len(result)<frames: raise ValueError("Decoded sequence shorter than requested")
        return result[:frames].clamp(0,1)

    def _ltx_scale(self, latents, inverse=False):
        mean = self.vae.latents_mean.reshape(1,-1,1,1,1).to(latents)
        std = self.vae.latents_std.reshape(1,-1,1,1,1).to(latents)
        if not torch.isfinite(std).all() or (std <= 0).any(): raise ValueError("Invalid LTX latent scales")
        scale = self.vae.config.scaling_factor
        return latents*std/scale+mean if inverse else (latents-mean)*scale/std


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, alpha):
        super().__init__(); self.base=base.requires_grad_(False); self.scale=alpha/rank
        self.a=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.b=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5)); nn.init.zeros_(self.b.weight)

    def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale


def inject_lora(trunk, rank, alpha):
    if rank<1: raise ValueError("LoRA rank must be positive")
    count=0
    for block in trunk.blocks:
        for attn in [block.self_attn,block.cross_attn]:
            for name in ["q","k","v","o"]:
                setattr(attn,name,LoRALinear(getattr(attn,name),rank,alpha)); count+=1
        for i in [0,2]: block.ffn[i]=LoRALinear(block.ffn[i],rank,alpha); count+=1
    return count


@dataclass
class Span:
    name: str
    start: int
    stop: int
    grid: tuple[int,int,int] | None


def positional_grid(grid, stream, device):
    t,h,w=grid
    tt,yy,xx=torch.meshgrid(torch.arange(t,device=device)+stream*32,torch.arange(h,device=device),torch.arange(w,device=device),indexing="ij")
    return torch.stack([tt,yy,xx],-1).reshape(-1,3)


def rotary(x, positions, frequencies):
    """Wan-compatible 3-axis RoPE over explicit mixed-stream coordinates."""
    half=x.shape[-1]//2; side=half//3; chunks=[half-2*side,side,side]
    parts=frequencies.split(chunks,-1)
    mult=torch.cat([part[positions[:,i]] for i,part in enumerate(parts)],-1)[None,:,None,:]
    xc=torch.view_as_complex(x.float().reshape(*x.shape[:-1],half,2))
    return torch.view_as_real(xc*mult).flatten(-2).to(x.dtype)


def attention(q,k,v,valid):
    mask=valid[:,None,None,:]
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask,dropout_p=0.).transpose(1,2)


@torch.no_grad()
def action_attention_mass(q,k,valid,spans):
    """Debug path materializes action-query rows only, not the full attention map."""
    query=next((s for s in spans if s.name=="action"),None)
    if query is None: return {}
    scores=torch.einsum("bqhd,bkhd->bhqk",q[:,query.start:query.stop].float(),k.float())/math.sqrt(q.shape[-1])
    weights=scores.masked_fill(~valid[:,None,None,:],-torch.inf).softmax(-1)
    result={}
    for span in spans:
        mass=weights[...,span.start:span.stop].sum(-1).mean()
        result[span.name]={"mass":float(mass),"per_token_mass":float(mass)/(span.stop-span.start),"tokens":span.stop-span.start}
    return result


class TinyBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__(); self.heads=heads
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width); self.norm3=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3); self.out=nn.Linear(width,width)
        self.cross_q=nn.Linear(width,width); self.cross_kv=nn.Linear(width,width*2); self.cross_o=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
        self.audit_spans=None; self.attention_mass=None

    def forward(self,x,valid,text,text_valid):
        b,n,d=x.shape; q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,d//self.heads).unbind(2)
        if self.audit_spans is not None: self.attention_mass=action_attention_mass(q,k,valid,self.audit_spans)
        x=x+self.out(attention(q,k,v,valid).flatten(2))
        q=self.cross_q(self.norm2(x)).reshape(b,n,self.heads,d//self.heads)
        k,v=self.cross_kv(text).reshape(b,text.shape[1],2,self.heads,d//self.heads).unbind(2)
        x=x+self.cross_o(attention(q,k,v,text_valid).flatten(2))
        return x+self.ff(self.norm3(x))


class JointModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__(); self.config=config; c=config; self.trunk=None
        self.audit=False; self.attention_log=[]
        if c.backend=="wan":
            add_wan_path(c)
            from wan.modules.model import WanModel
            self.trunk=WanModel.from_pretrained(c.checkpoint,torch_dtype=torch.bfloat16).requires_grad_(False)
            if self.trunk.in_dim!=c.latent_channels or self.trunk.dim!=c.width:
                raise ValueError("Checkpoint dimensions disagree with config")
            inject_lora(self.trunk,c.lora_rank,c.lora_alpha)
        else:
            self.patch=nn.Conv3d(c.latent_channels,c.width,(1,2,2),stride=(1,2,2))
            self.visual_out=nn.Linear(c.width,c.latent_channels*4)
            self.blocks=nn.ModuleList([TinyBlock(c.width,c.heads) for _ in range(c.layers)])
            self.time=nn.Sequential(nn.Linear(1,c.width),nn.SiLU(),nn.Linear(c.width,c.width))
            self.text_projection=nn.Linear(c.text_dim,c.width)
            self.position_projection=nn.Linear(3,c.width)
            self.final_norm=nn.LayerNorm(c.width)
        self.action_in=nn.Linear(c.action_dim,c.width)
        self.action_out=nn.Linear(c.width,c.action_dim)
        self.action_position=nn.Parameter(torch.randn(c.horizon,c.width)*.02)

    def parameter_counts(self):
        return {"total":sum(p.numel() for p in self.parameters()),"trainable":sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def _pack(self,batch,noisy,t,known):
        b=batch["obs"].shape[0]; device=t.device; tokens=[]; masks=[]; coords=[]; times=[]; spans=[]; cursor=0
        patch=self.trunk.patch_embedding if self.trunk is not None else self.patch
        def add_visual(name,x,valid,clean=False):
            nonlocal cursor
            encoded=patch(x); grid=tuple(encoded.shape[2:]); seq=encoded.flatten(2).transpose(1,2); length=seq.shape[1]
            tokens.append(seq); masks.append(valid[:,None].expand(b,length)); coords.append(positional_grid(grid,len(spans),device))
            times.append((torch.zeros_like(t) if clean else t)[:,None].expand(b,length))
            spans.append(Span(name,cursor,cursor+length,grid)); cursor+=length
        for view in range(batch["obs"].shape[1]):
            add_visual(f"obs_{view}",batch["obs"][:,view],batch["view_valid"][:,view],True)
        enabled=list(self.config.targets)
        for name in known:
            if name not in enabled: enabled.append(name)
        for name in enabled:
            x=known.get(name,noisy.get(name))
            if x is None: raise ValueError(f"Missing target {name}")
            if name=="action":
                seq=self.action_in(x)+self.action_position[None]
                length=x.shape[1]; tokens.append(seq); masks.append(batch["has_action"][:,None].expand(b,length))
                coords.append(torch.stack([torch.arange(length,device=device)+len(spans)*32,torch.zeros(length,device=device),torch.zeros(length,device=device)],-1).long())
                times.append(t[:,None].expand(b,length)); spans.append(Span(name,cursor,cursor+length,None)); cursor+=length
            else: add_visual(name,x,torch.ones(b,dtype=torch.bool,device=device),name in known)
        return torch.cat(tokens,1),torch.cat(masks,1),torch.cat(coords,0),torch.cat(times,1),spans

    def forward(self,batch,noisy,t,known=None):
        known=known or {}; x,valid,positions,token_t,spans=self._pack(batch,noisy,t,known)
        self.attention_log=[]
        if not valid.any(-1).all(): raise ValueError("Every example needs a valid conditioning token")
        if self.trunk is None:
            x=x+self.time(token_t[...,None])+self.position_projection(positions.float()[None]/32.)
            text=self.text_projection(batch["text"])
            for block in self.blocks:
                block.audit_spans=spans if self.audit else None
                if self.config.gradient_checkpointing and self.training:
                    x=checkpoint(block,x,valid,text,batch["text_valid"],use_reentrant=False)
                else: x=block(x,valid,text,batch["text_valid"])
                if self.audit: self.attention_log.append(block.attention_mass)
            x=self.final_norm(x); e=None
        else:
            trunk=self.trunk; frequency_dim=trunk.freq_dim
            frequency=10000**(-torch.arange(frequency_dim//2,device=x.device).float()/(frequency_dim//2))
            args=token_t.flatten()[:,None]*1000*frequency[None]
            sinusoid=torch.cat([args.cos(),args.sin()],-1).reshape(*token_t.shape,frequency_dim)
            e=trunk.time_embedding(sinusoid).float()
            modulation=trunk.time_projection(e).reshape(*token_t.shape,6,trunk.dim).float()
            text=trunk.text_embedding(batch["text"])
            frequencies=trunk.freqs.to(x.device)
            for block in trunk.blocks:
                if self.config.gradient_checkpointing and self.training:
                    def run(value,mod,ctx,block=block):
                        return self._wan_block(block,value,mod,valid,positions,frequencies,ctx,batch["text_valid"])
                    x=checkpoint(run,x,modulation,text,use_reentrant=False)
                else: x=self._wan_block(block,x,modulation,valid,positions,frequencies,text,batch["text_valid"],spans,self.attention_log if self.audit else None)
        result={}
        for span in spans:
            if span.name.startswith("obs_") or span.name in known: continue
            values=x[:,span.start:span.stop]
            if span.name=="action": result[span.name]=self.action_out(values).float()
            else:
                pixels=self.trunk.head(values,e[:,span.start:span.stop]) if self.trunk is not None else self.visual_out(values)
                f,h,w=span.grid; c=self.config.latent_channels
                pixels=pixels.reshape(len(x),f,h,w,1,2,2,c).permute(0,7,1,4,2,5,3,6)
                result[span.name]=pixels.reshape(len(x),c,f,h*2,w*2).float()
        return result

    @staticmethod
    def _wan_block(block,x,mod,valid,positions,frequencies,text,text_valid,spans=None,audit=None):
        e=(block.modulation[None]+mod).unbind(2)
        inp=block.norm1(x).float()*(1+e[1])+e[0]
        attn=block.self_attn; b,n,_=inp.shape; heads=attn.num_heads; hd=attn.head_dim
        q=attn.norm_q(attn.q(inp)).reshape(b,n,heads,hd)
        k=attn.norm_k(attn.k(inp)).reshape(b,n,heads,hd)
        v=attn.v(inp).reshape(b,n,heads,hd)
        q,k=rotary(q,positions,frequencies),rotary(k,positions,frequencies)
        if audit is not None: audit.append(action_attention_mass(q,k,valid,spans))
        out=attention(q,k,v,valid)
        x=x+attn.o(out.flatten(2))*e[2]
        attn=block.cross_attn; inp=block.norm3(x)
        q=attn.norm_q(attn.q(inp)).reshape(b,n,heads,hd)
        k=attn.norm_k(attn.k(text)).reshape(b,text.shape[1],heads,hd)
        v=attn.v(text).reshape(b,text.shape[1],heads,hd)
        x=x+attn.o(attention(q,k,v,text_valid).flatten(2))
        return x+block.ffn(block.norm2(x).float()*(1+e[4])+e[3])*e[5]


def inject_ltx_lora(trunk, rank, alpha):
    count = 0
    for block in trunk.transformer_blocks:
        for attn in (block.attn1, block.attn2):
            for name in ("to_q", "to_k", "to_v"):
                setattr(attn, name, LoRALinear(getattr(attn,name), rank, alpha)); count += 1
            attn.to_out[0] = LoRALinear(attn.to_out[0], rank, alpha); count += 1
        block.ff.net[0].proj = LoRALinear(block.ff.net[0].proj, rank, alpha)
        block.ff.net[2] = LoRALinear(block.ff.net[2], rank, alpha); count += 2
    return count


class LTXJointModel(nn.Module):
    """LTX 2B blocks over clean image prefixes and jointly noisy targets.

    LTX uses linear per-voxel projections (patch=1), its own learned timestep
    modulation, and interleaved RoPE across the full hidden width before heads.
    The upstream trunk is frozen except LoRA; action projections are new.
    """
    def __init__(self, config, trunk=None):
        super().__init__(); self.config=config; self.audit=False; self.attention_log=[]
        if trunk is None:
            root = ltx_checkpoint(config)
            from diffusers import LTXVideoTransformer3DModel
            trunk = LTXVideoTransformer3DModel.from_pretrained(root, subfolder="transformer",
                local_files_only=True, torch_dtype=torch.bfloat16)
        c, tc = config, trunk.config
        actual = (tc.in_channels,tc.out_channels,tc.num_attention_heads*tc.attention_head_dim,
                  tc.caption_channels,tc.num_layers,tc.num_attention_heads,tc.patch_size,tc.patch_size_t)
        expected = (c.latent_channels,c.latent_channels,c.width,c.text_dim,c.layers,c.heads,1,1)
        if actual != expected: raise ValueError(f"LTX checkpoint dimensions {actual} disagree with {expected}")
        self.trunk = trunk.requires_grad_(False)
        inject_ltx_lora(trunk,c.lora_rank,c.lora_alpha)
        self.action_in = nn.Linear(c.action_dim,c.width)
        self.action_out = nn.Linear(c.width,c.action_dim)
        self.action_position = nn.Parameter(torch.randn(c.horizon,c.width)*.02)

    def parameter_counts(self):
        return JointModel.parameter_counts(self)

    def _pack(self,batch,noisy,t,known):
        b=len(t); c=self.config; tokens=[]; masks=[]; coords=[]; times=[]; spans=[]; cursor=0
        def add(name,value,valid,clean=False):
            nonlocal cursor
            stream=len(spans)
            if name == "action":
                if value.shape[1:] != (c.horizon,c.action_dim): raise ValueError("LTX action shape mismatch")
                seq=self.action_in(value)+self.action_position[None]; grid=None
                xyz=torch.zeros(c.horizon,3,device=t.device)
                xyz[:,0]=(torch.arange(c.horizon,device=t.device)+stream*32)/c.video_fps
            else:
                if value.ndim != 5 or value.shape[1] != c.latent_channels: raise ValueError("LTX visual shape mismatch")
                grid=tuple(value.shape[2:]); seq=self.trunk.proj_in(value.flatten(2).transpose(1,2))
                xyz=positional_grid(grid,0,t.device).float()
                # v0.9.6 uses causal temporal positioning: frame 0 is a
                # singleton; later latent cells begin at frames 1,9,... .
                frame=(xyz[:,0]*c.temporal_stride+1-c.temporal_stride).clamp_min(0)
                # RGB targets start at observation frame 1; cumulative flow
                # targets include frame 0. Keep both aligned to action time.
                xyz[:,0]=(frame+(1 if name=="video" else 0)+stream*32)/c.video_fps
                xyz[:,1:]*=c.spatial_stride
            n=seq.shape[1]; tokens.append(seq); masks.append(valid[:,None].expand(b,n)); coords.append(xyz)
            times.append((torch.zeros_like(t) if clean else t)[:,None].expand(b,n))
            spans.append(Span(name,cursor,cursor+n,grid)); cursor+=n
        for view in range(batch["obs"].shape[1]):
            add(f"obs_{view}",batch["obs"][:,view],batch["view_valid"][:,view],True)
        enabled=list(c.targets)+[name for name in known if name not in c.targets]
        for name in enabled:
            value=known.get(name,noisy.get(name))
            if value is None: raise ValueError(f"Missing LTX target {name}")
            valid=batch["has_action"] if name=="action" else torch.ones(b,dtype=torch.bool,device=t.device)
            add(name,value,valid,name in known)
        return torch.cat(tokens,1),torch.cat(masks,1),torch.cat(coords,0),torch.cat(times,1),spans

    @staticmethod
    def _attention(attn,x,context,mask,rope=None,spans=None,audit=None):
        context=x if context is None else context
        q=attn.norm_q(attn.to_q(x)); k=attn.norm_k(attn.to_k(context)); v=attn.to_v(context)
        if rope is not None:
            # Match Diffusers' LTX interleaved full-width rotary operation.
            cos,sin=rope
            def rotate(value):
                pairs=value.float().reshape(*value.shape[:-1],-1,2)
                orth=torch.stack([-pairs[...,1],pairs[...,0]],-1).flatten(-2)
                return (value.float()*cos+orth*sin).to(value.dtype)
            q,k=rotate(q),rotate(k)
        b,n,d=q.shape; heads=attn.heads
        q=q.reshape(b,n,heads,d//heads); k=k.reshape(b,k.shape[1],heads,d//heads); v=v.reshape(b,v.shape[1],heads,d//heads)
        if audit is not None: audit.append(action_attention_mass(q,k,mask,spans))
        return attn.to_out[1](attn.to_out[0](attention(q,k,v,mask).flatten(2)))

    @classmethod
    def _block(cls,block,x,mod,valid,rope,text,text_valid,spans=None,audit=None):
        shift,scale,gate,ff_shift,ff_scale,ff_gate=(block.scale_shift_table[None,None]+mod.reshape(*mod.shape[:2],6,-1)).unbind(2)
        value=block.norm1(x)*(1+scale)+shift
        x=x+cls._attention(block.attn1,value,None,valid,rope,spans,audit)*gate
        x=x+cls._attention(block.attn2,x,text,text_valid)
        return x+block.ff(block.norm2(x)*(1+ff_scale)+ff_shift)*ff_gate

    def forward(self,batch,noisy,t,known=None):
        known=known or {}; x,valid,coords,token_t,spans=self._pack(batch,noisy,t,known)
        if not valid.any(-1).all(): raise ValueError("Every example needs a valid conditioning token")
        trunk=self.trunk; self.attention_log=[]
        rope=trunk.rope(x,video_coords=coords.T[None].expand(len(x),-1,-1))
        mod,embedding=trunk.time_embed((token_t*1000).flatten(),batch_size=len(x),hidden_dtype=x.dtype)
        mod=mod.reshape(len(x),x.shape[1],-1); embedding=embedding.reshape(len(x),x.shape[1],-1)
        text=trunk.caption_projection(batch["text"])
        for block in trunk.transformer_blocks:
            if self.config.gradient_checkpointing and self.training:
                def run(value,modulation,context,block=block):
                    return self._block(block,value,modulation,valid,rope,context,batch["text_valid"])
                x=checkpoint(run,x,mod,text,use_reentrant=False)
            else:
                x=self._block(block,x,mod,valid,rope,text,batch["text_valid"],spans,self.attention_log if self.audit else None)
        shift,scale=(trunk.scale_shift_table[None,None]+embedding[:,:,None]).unbind(2)
        x=trunk.norm_out(x)*(1+scale)+shift
        result={}
        for span in spans:
            if span.name.startswith("obs_") or span.name in known: continue
            value=x[:,span.start:span.stop]
            if span.name=="action": result[span.name]=self.action_out(value).float()
            else: result[span.name]=trunk.proj_out(value).transpose(1,2).reshape(len(x),self.config.latent_channels,*span.grid).float()
        return result


class BehaviorCloning(nn.Module):
    """Compact direct action-chunk baseline with the same frozen input codecs."""
    deterministic=True

    def __init__(self,config):
        super().__init__(); self.config=config
        spatial=config.image_size//config.spatial_stride
        inputs=config.views*config.latent_channels*spatial*spatial+config.text_dim
        self.net=nn.Sequential(nn.Linear(inputs,config.width),nn.GELU(),nn.Linear(config.width,config.width),nn.GELU(),
                               nn.Linear(config.width,config.horizon*config.action_dim))

    def parameter_counts(self):
        n=sum(p.numel() for p in self.parameters()); return {"total":n,"trainable":n}

    def forward(self,batch,noisy=None,t=None,known=None):
        obs=batch["obs"]*batch["view_valid"][:,:,None,None,None,None]
        mask=batch["text_valid"][...,None]
        text=(batch["text"]*mask).sum(1)/mask.sum(1).clamp_min(1)
        x=torch.cat([obs.flatten(1),text],-1)
        return {"action":self.net(x).reshape(len(x),self.config.horizon,self.config.action_dim)}


def build_model(config):
    if config.architecture=="bc": return BehaviorCloning(config)
    return LTXJointModel(config) if config.backend=="ltx" else JointModel(config)


def noisy_targets(batch, targets, generator=None):
    b=len(batch["obs"]); device=batch["obs"].device
    t=torch.rand(b,device=device,generator=generator)
    noisy={}; velocity={}
    for name in targets:
        clean=batch[name]; eps=torch.randn(clean.shape,device=device,generator=generator)
        expanded=t.reshape(b,*([1]*(clean.ndim-1)))
        noisy[name]=(1-expanded)*clean+expanded*eps; velocity[name]=eps-clean
    return noisy,velocity,t


def modality_loss(prediction, target, has_action, weights):
    terms={}; loss=None
    for name,pred in prediction.items():
        if name not in target: continue
        per_example=(pred.float()-target[name].float()).square().flatten(1).mean(1)
        if name=="action": per_example=per_example*has_action.float()
        mean=per_example.mean(); terms[name]=mean.detach()
        contribution=mean*weights.get(name,0.)
        loss=contribution if loss is None else loss+contribution
    if loss is None: raise ValueError("No supervised targets")
    return loss,terms


@torch.no_grad()
def sample_joint(model,batch,steps=10,shift=5.,solver="euler",seed=0,known=None):
    if getattr(model,"deterministic",False): return model(batch)
    device=batch["obs"].device; known=known or {}
    gen=torch.Generator(device=device).manual_seed(seed)
    c=model.config; b=len(batch["obs"]); obs=batch["obs"]
    frames=1+math.ceil((c.horizon-1)/c.temporal_stride)
    shapes={"action":(b,c.horizon,c.action_dim),"flow":(b,c.latent_channels,frames,obs.shape[-2],obs.shape[-1]),
            "video":(b,c.latent_channels,frames,obs.shape[-2],obs.shape[-1])}
    state={name:torch.randn(shapes[name],generator=gen,device=device) for name in c.targets if name not in known}
    if solver=="unipc":
        if c.backend=="ltx": raise ValueError("LTX uses Euler; Wan UniPC must not load a different upstream")
        add_wan_path(c)
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        schedules={name:FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False) for name in state}
        for sched in schedules.values(): sched.set_timesteps(steps,device=device,shift=shift)
        reference=next(iter(schedules.values()))
        for i,step in enumerate(reference.timesteps):
            # Use the same sigma used by the solver, not its rounded integer timestep.
            t=torch.full((b,),float(reference.sigmas[i]),device=device)
            pred=model(batch,state,t,known)
            state={name:schedules[name].step(pred[name],step,x,return_dict=False)[0] for name,x in state.items()}
    elif solver=="euler":
        schedule=torch.linspace(1,0,steps+1,device=device); schedule=shift*schedule/(1+(shift-1)*schedule)
        for i in range(steps):
            pred=model(batch,state,schedule[i].expand(b),known)
            state={name:x+(schedule[i+1]-schedule[i])*pred[name] for name,x in state.items()}
    else: raise ValueError("Unknown solver")
    return state
