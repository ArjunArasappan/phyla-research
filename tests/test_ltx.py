"""CPU checks against real, small Diffusers LTX components; no weights fetched."""
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flomo.config import Config, ModelConfig, from_dict, load_config, model_signature
from flomo.model import LTXJointModel, FrozenEncoders, pad_video, sample_joint, ltx_checkpoint, convert_ltx
from flomo.train import model_state, apply_state


def fixture_model():
    pytest.importorskip("diffusers")
    from diffusers import LTXVideoTransformer3DModel
    c = ModelConfig(backend="ltx",latent_channels=4,spatial_stride=32,image_size=64,
                    text_dim=8,text_length=5,width=24,layers=2,heads=2,lora_rank=2,lora_alpha=4)
    trunk = LTXVideoTransformer3DModel(in_channels=4,out_channels=4,num_attention_heads=2,
        attention_head_dim=12,cross_attention_dim=24,caption_channels=8,num_layers=2)
    return c, trunk


def fixture_batch(c):
    return {"obs":torch.randn(2,1,4,1,2,2),"view_valid":torch.ones(2,1,dtype=torch.bool),
            "text":torch.randn(2,5,8),"text_valid":torch.ones(2,5,dtype=torch.bool),
            "flow":torch.randn(2,4,3,2,2),"action":torch.randn(2,16,2),
            "has_action":torch.tensor([True,False])}


def test_ltx_config_and_temporal_padding():
    c=from_dict({"model":{"backend":"ltx","latent_channels":128,"spatial_stride":32,
                          "text_dim":4096,"image_size":96}})
    assert c.model.temporal_stride==8
    movie=torch.randn(16,3,4,4); padded=pad_video(movie,8)
    assert len(padded)==17 and torch.equal(padded[:16],movie) and torch.equal(padded[-1],movie[-1])
    assert len(pad_video(movie[:9],8))==9
    with pytest.raises(ValueError,match="LTX 2B requires"): from_dict({"model":{"backend":"ltx"}})
    bad=asdict(c); bad["model"]["video_fps"]=30
    with pytest.raises(ValueError,match="controller rate"): from_dict(bad)
    bad=asdict(c); bad["eval"]["sampler"]="unipc"
    with pytest.raises(ValueError,match="Euler"): from_dict(bad)


def test_ltx_pretrained_visual_forward_matches_upstream():
    torch.manual_seed(7); c,trunk=fixture_model(); c.targets=["flow"]
    upstream=copy.deepcopy(trunk).eval(); model=LTXJointModel(c,trunk).eval()
    b=fixture_batch(c); t=torch.tensor([.2,.7]); noisy={"flow":b["flow"]}
    _,_,coords,times,spans=model._pack(b,noisy,t,{})
    raw=torch.cat([b["obs"][:,0].flatten(2).transpose(1,2),b["flow"].flatten(2).transpose(1,2)],1)
    with torch.no_grad():
        expected=upstream(raw,b["text"],times*1000,b["text_valid"],video_coords=coords.T[None].expand(2,-1,-1)).sample
        actual=model(b,noisy,t)["flow"]
    span=spans[-1]
    expected=expected[:,span.start:span.stop].transpose(1,2).reshape_as(actual)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-5)


def test_ltx_masking_conditioning_gradients_and_state():
    torch.manual_seed(3); c,trunk=fixture_model(); model=LTXJointModel(c,trunk)
    b=fixture_batch(c); noisy={"flow":b["flow"],"action":b["action"]}; t=torch.tensor([.3,.6])
    model.audit=True; first=model(b,noisy,t)
    assert first["flow"].shape==(2,4,3,2,2) and first["action"].shape==(2,16,2)
    for layer in model.attention_log: assert abs(sum(x["mass"] for x in layer.values())-1)<1e-5
    changed={**noisy,"action":noisy["action"].clone()}; changed["action"][1]+=100
    assert torch.allclose(first["flow"][1],model(b,changed,t)["flow"][1],atol=1e-6)
    image_changed={**b,"obs":b["obs"]+2}
    assert not torch.allclose(first["action"],model(image_changed,noisy,t)["action"])
    text_changed={**b,"text":b["text"]+2}
    assert not torch.allclose(first["action"],model(text_changed,noisy,t)["action"])
    first["action"][0].square().mean().backward()
    assert model.action_in.weight.grad is not None
    assert model.trunk.transformer_blocks[0].attn1.to_q.b.weight.grad is not None
    assert all(p.grad is None for n,p in model.trunk.named_parameters() if not p.requires_grad)
    weights=model_state(model)
    assert "action_in.weight" in weights and not any("base.weight" in n for n in weights)
    apply_state(model,weights)


def test_ltx_probe_and_solver_shapes():
    c,trunk=fixture_model(); model=LTXJointModel(c,trunk).eval(); b=fixture_batch(c)
    result=sample_joint(model,b,steps=2,shift=1,seed=1)
    assert result["flow"].shape==(2,4,3,2,2) and result["action"].shape==(2,16,2)
    assert torch.equal(result["action"],sample_joint(model,b,steps=2,shift=1,seed=1)["action"])
    with pytest.raises(ValueError,match="LTX uses Euler"): sample_joint(model,b,solver="unipc")
    model.config.targets=["action"]
    result=model(b,{"action":b["action"]},torch.tensor([.2,.8]),known={"flow":b["flow"]})
    assert set(result)=={"action"}
    _,_,_,times,spans=model._pack(b,{"action":b["action"]},torch.tensor([.2,.8]),{"flow":b["flow"]})
    flow=next(s for s in spans if s.name=="flow")
    assert not times[:,flow.start:flow.stop].any()


def test_ltx_checkpointing_gradient_equivalence():
    c,trunk=fixture_model(); plain=LTXJointModel(c,trunk); checked=copy.deepcopy(plain)
    checked.config.gradient_checkpointing=True; b=fixture_batch(c)
    for model in (plain,checked):
        y=model(b,{"flow":b["flow"],"action":b["action"]},torch.tensor([.2,.8]))
        (y["flow"].square().mean()+y["action"][0].square().mean()).backward()
    for a,b in zip(plain.parameters(),checked.parameters()):
        if a.grad is not None: assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("timestep_conditioning", [False, True])
def test_ltx_real_small_codec_and_scale_roundtrip(timestep_conditioning):
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLLTXVideo
    c,_=fixture_model()
    encoder=FrozenEncoders.__new__(FrozenEncoders); encoder.config=c; encoder.device=torch.device("cpu")
    encoder.vae=AutoencoderKLLTXVideo(latent_channels=4,block_out_channels=(8,8,8,8),
        decoder_block_out_channels=(8,8,8,8),layers_per_block=(1,1,1,1,1),decoder_layers_per_block=(1,1,1,1,1),
        timestep_conditioning=timestep_conditioning)
    encoder.vae.eval().requires_grad_(False)
    encoder.vae.latents_mean.copy_(torch.tensor([.2,.3,.4,.5]))
    encoder.vae.latents_std.copy_(torch.tensor([1.,2.,3.,4.]))
    x=torch.randn(1,4,3,2,2)
    assert torch.allclose(encoder._ltx_scale(encoder._ltx_scale(x),inverse=True),x,atol=1e-6)
    z=encoder.encode_video(torch.rand(16,3,64,64))
    assert z.shape==(4,3,2,2)
    assert encoder.decode_video(z,16).shape==(16,3,64,64)
    assert encoder.encode_observations(torch.randint(0,256,(1,64,64,3),dtype=torch.uint8).numpy()).shape==(1,4,1,2,2)


def test_ltx_offline_checkpoint_contract(tmp_path):
    pytest.importorskip("diffusers")
    c=ModelConfig(backend="ltx",checkpoint=str(tmp_path),checkpoint_revision="abc")
    with pytest.raises(ValueError,match="convert-ltx"): ltx_checkpoint(c)
    import json
    info={"variant":"ltxv-2b-0.9.6-dev-04-25","revision":"abc","diffusers_version":"0.33.1","converter_version":2}
    (tmp_path/"flomo_ltx.json").write_text(json.dumps(info))
    assert ltx_checkpoint(c)==tmp_path
    c.checkpoint_revision="different"
    with pytest.raises(ValueError,match="mismatch"): ltx_checkpoint(c)
    with pytest.raises(ValueError,match="non-distilled"): convert_ltx("distilled.safetensors",tmp_path,"abc",tmp_path/"out")


def test_legacy_model_signatures_stay_compatible():
    from flomo.config import digest
    c=Config(); old=asdict(c.model)
    for name in ("upstream","checkpoint","gradient_checkpointing","video_fps"): old.pop(name)
    assert model_signature(c)==digest(old)


def test_ltx_real_t5_encoding_masks_padding():
    pytest.importorskip("transformers")
    from transformers import T5Config, T5EncoderModel, BatchEncoding
    c,_=fixture_model()
    encoder=FrozenEncoders.__new__(FrozenEncoders); encoder.config=c; encoder.device=torch.device("cpu")
    encoder.text_model=T5EncoderModel(T5Config(d_model=8,d_ff=16,d_kv=4,num_layers=1,num_heads=2,vocab_size=20)).eval()
    class Tokenizer:
        def __call__(self,instruction,**kwargs):
            assert kwargs["max_length"]==5 and kwargs["padding"]=="max_length"
            return BatchEncoding({"input_ids":torch.tensor([[1,2,0,0,0]]),"attention_mask":torch.tensor([[1,1,0,0,0]])})
    encoder.tokenizer=Tokenizer()
    hidden,mask=encoder.encode_text("move cube")
    assert hidden.shape==(5,8) and mask.tolist()==[True,True,False,False,False]
    assert not hidden[2:].any()


def test_ltx_gpu_templates_and_cpu_guard():
    from pathlib import Path
    from flomo.train import train
    root=Path(__file__).resolve().parents[1]
    for stage in ("midtrain","sft"):
        c=load_config(root/f"configs/ltx_{stage}.yaml")
        assert c.model.backend=="ltx" and c.model.width==2048 and c.model.temporal_stride==8
        assert c.eval.sampler=="euler" and c.train.stage==stage
        c.train.device="cpu"
        with pytest.raises(RuntimeError,match="ltx training requires CUDA"): train(c)


def test_ltx_human_video_timing_is_checked(tmp_path):
    import numpy as np
    from flomo.data import save_episode, make_windows
    c=from_dict({"model":{"backend":"ltx","latent_channels":128,"spatial_stride":32,
                          "text_dim":4096,"image_size":64},
                 "data":{"val_fraction":0,"test_fraction":0}})
    arrays={"rgb":np.zeros((17,1,8,8,3),dtype=np.uint8),"times":np.arange(17)/30}
    meta={"source":"human","instruction":"move cube","camera_names":["front"]}
    c.data.raw=str(tmp_path/"bad"); save_episode(c.data.raw,"human",arrays,meta)
    with pytest.raises(ValueError,match="resample human footage"): make_windows(c)
    c.data.raw=str(tmp_path/"good"); arrays["times"]=np.arange(17)/20
    save_episode(c.data.raw,"human",arrays,meta)
    windows,_=make_windows(c)
    assert len(windows)==1 and not windows[0]["has_actions"]


def test_actual_096_checkpoint_shapes_convert_strictly_on_meta():
    """944 real checkpoint tensor schemas, without downloading tensor data."""
    pytest.importorskip("diffusers")
    import json
    from pathlib import Path
    from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from flomo.model import convert_ltx_state, strict_component_state
    root=Path(__file__).parent/"fixtures"
    header=json.loads((root/"ltx_096_shapes.json").read_text())
    source={k:torch.empty(v["shape"],device="meta") for k,v in header["tensors"].items()}
    assert len(source)==944
    for name,cls in (("transformer",LTXVideoTransformer3DModel),("vae",AutoencoderKLLTXVideo)):
        cfg=json.loads((root/f"ltx_095_{name}.json").read_text())
        with torch.device("meta"): model=cls.from_config(cfg)
        state=convert_ltx_state(source,name)
        strict_component_state(model,state)
        assert all(p.device.type=="meta" for p in model.parameters())
    # A broken rename must fail rather than initialize random frozen weights.
    wrong=dict(state); wrong.pop(next(iter(wrong)))
    with pytest.raises(ValueError,match="key mismatch"): strict_component_state(model,wrong)
    wrong=dict(state); key=next(iter(wrong)); wrong[key]=torch.empty(1,device="meta")
    with pytest.raises(ValueError,match="shape mismatch"): strict_component_state(model,wrong)


def test_ltx_causal_rotary_times():
    c,trunk=fixture_model(); model=LTXJointModel(c,trunk); b=fixture_batch(c)
    _,_,coords,_,spans=model._pack(b,{"flow":b["flow"],"action":b["action"]},torch.tensor([.2,.5]),{})
    span=next(s for s in spans if s.name=="flow")
    frame_times=coords[span.start:span.stop,0].reshape(3,2,2)[:,0,0]
    assert torch.allclose(frame_times,torch.tensor([32.,33.,41.])/20)
    _,_,coords,_,spans=model._pack(b,{"flow":b["flow"],"action":b["action"]},torch.tensor([.2,.5]),{"video":b["flow"]})
    span=next(s for s in spans if s.name=="video")
    frame_times=coords[span.start:span.stop,0].reshape(3,2,2)[:,0,0]
    assert torch.allclose(frame_times,torch.tensor([97.,98.,106.])/20)


def test_ltx_bfloat16_autocast_and_lora_gradients():
    c,trunk=fixture_model(); model=LTXJointModel(c,trunk.to(torch.bfloat16)); b=fixture_batch(c)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        out=model(b,{"flow":b["flow"],"action":b["action"]},torch.tensor([.2,.8]))
        loss=out["flow"].square().mean()+out["action"][0].square().mean()
    loss.backward()
    assert all(torch.isfinite(x).all() for x in out.values())
    trainable=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert trainable and all(torch.isfinite(p.grad).all() for p in trainable)


def test_ltx_095_residual_codec_family():
    pytest.importorskip("diffusers")
    import json
    from pathlib import Path
    from diffusers import AutoencoderKLLTXVideo
    cfg=json.loads((Path(__file__).parent/"fixtures/ltx_095_vae.json").read_text())
    cfg.update(latent_channels=4,block_out_channels=[8,16,32,64,128],decoder_block_out_channels=[8,16,32],
               layers_per_block=[1,1,1,1,1],decoder_layers_per_block=[1,1,1,1])
    c,_=fixture_model(); encoder=FrozenEncoders.__new__(FrozenEncoders)
    encoder.config=c; encoder.device=torch.device("cpu"); encoder.vae=AutoencoderKLLTXVideo.from_config(cfg).eval()
    z=encoder.encode_video(torch.rand(16,3,64,64))
    assert z.shape==(4,3,2,2)
    assert encoder.decode_video(z,16).shape==(16,3,64,64)


@pytest.mark.parametrize("fail_save", [False, True])
def test_ltx_conversion_transaction_and_local_assets(tmp_path,monkeypatch,fail_save):
    """Conversion transaction with small tensors; schema conversion tested above."""
    pytest.importorskip("diffusers")
    import json
    import diffusers
    import safetensors.torch
    import flomo.model as module
    root=tmp_path/"assets"; root.mkdir()
    fixtures=Path(__file__).parent/"fixtures"
    for name in ("transformer","vae","text_encoder","tokenizer"): (root/name).mkdir()
    for name in ("transformer","vae"):
        (root/name/"config.json").write_text((fixtures/f"ltx_095_{name}.json").read_text())
    (root/"text_encoder/config.json").write_text(json.dumps({"d_model":4096}))
    (root/"text_encoder/pytorch_model.bin").write_bytes(b"local-test-text-asset")
    (root/"tokenizer/tokenizer_config.json").write_text("{}")
    weights=tmp_path/(module.LTX_VARIANT+".safetensors"); weights.write_bytes(b"local-test-weight-file")
    output=tmp_path/"converted"
    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(2,2))
        @classmethod
        def from_config(cls,cfg): return cls()
        def save_pretrained(self,path,**kwargs):
            if fail_save: raise OSError("simulated write failure")
            path.mkdir(); torch.save(self.state_dict(),path/"weights.pt")
    monkeypatch.setattr(diffusers,"LTXVideoTransformer3DModel",Component)
    monkeypatch.setattr(diffusers,"AutoencoderKLLTXVideo",Component)
    monkeypatch.setattr(safetensors.torch,"load_file",lambda *a,**k:{"weight":torch.ones(2,2)})
    monkeypatch.setattr(module,"convert_ltx_state",lambda source,component:dict(source))
    if fail_save:
        with pytest.raises(OSError,match="simulated write failure"): module.convert_ltx(weights,root,"pinned",output)
        assert not output.exists() and not list(tmp_path.glob("converted-*"))
    else:
        info=module.convert_ltx(weights,root,"pinned",output)
        assert info["converter_version"]==2 and info["revision"]=="pinned"
        assert (output/"text_encoder/pytorch_model.bin").read_bytes()==b"local-test-text-asset"
        assert json.loads((output/"flomo_ltx.json").read_text())==info
        with pytest.raises(FileExistsError): module.convert_ltx(weights,root,"pinned",output)
