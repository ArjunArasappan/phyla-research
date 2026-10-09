import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flomo.config import Config, from_dict
from flomo.model import JointModel, FrozenEncoders, noisy_targets, modality_loss, sample_joint, LoRALinear, inject_lora, rotary, build_model
from flomo.data import PreparedDataset,collate


def batch(config):
    c=config.model; h=c.image_size//c.spatial_stride
    return {"obs":torch.randn(2,c.views,c.latent_channels,1,h,h),"view_valid":torch.ones(2,c.views,dtype=torch.bool),
            "text":torch.randn(2,c.text_length,c.text_dim),"text_valid":torch.ones(2,c.text_length,dtype=torch.bool),
            "flow":torch.randn(2,c.latent_channels,5,h,h),"video":torch.randn(2,c.latent_channels,5,h,h),
            "action":torch.randn(2,c.horizon,c.action_dim),"has_action":torch.tensor([True,False])}


def test_strict_config():
    with pytest.raises(ValueError,match="Unknown model"): from_dict({"model":{"typo":1}})
    with pytest.raises(ValueError,match="requires C=48"): from_dict({"model":{"backend":"wan"}})


def test_encoder_keeps_trailing_motion_frame():
    c=Config().model; encoder=FrozenEncoders(c)
    movie=torch.zeros(16,3,32,32); movie[-1]=1.
    latent=encoder.encode_video(movie)
    assert latent.shape==(6,5,8,8)
    assert latent[:,-1].mean()>latent[:,0].mean()
    assert encoder.decode_video(latent,16).shape==(16,3,32,32)


def test_missing_actions_do_not_affect_flow():
    c=Config(); model=JointModel(c.model).eval(); b=batch(c)
    noisy={"flow":b["flow"].clone(),"action":b["action"].clone()}; t=torch.tensor([.2,.6])
    first=model(b,noisy,t)
    noisy["action"][1]=torch.randn_like(noisy["action"][1])*100
    second=model(b,noisy,t)
    assert torch.allclose(first["flow"][1],second["flow"][1],atol=1e-6)
    assert first["action"].shape==(2,16,2)


def test_action_free_loss_is_zero_and_shared_noise_time():
    c=Config(); b=batch(c)
    noisy,target,t=noisy_targets(b,c.model.targets,torch.Generator().manual_seed(3))
    for name in c.model.targets:
        sigma=t.reshape(2,*([1]*(b[name].ndim-1)))
        assert torch.allclose(noisy[name]-b[name],sigma*target[name],atol=1e-6)
    prediction={"action":torch.zeros_like(b["action"])}
    target={"action":torch.ones_like(b["action"])}
    loss,_=modality_loss(prediction,target,b["has_action"],{"action":1})
    assert loss==.5


def test_sampler_direction_and_repeatability():
    class ConstantModel:
        config=Config().model
        def __call__(self,batch,state,t,known): return {k:torch.ones_like(x)*2 for k,x in state.items()}
    b=batch(Config()); model=ConstantModel(); result=sample_joint(model,b,steps=4,shift=5,seed=9)
    gen=torch.Generator().manual_seed(9)
    # Target insertion order is flow then action.
    flow=torch.randn(result["flow"].shape,generator=gen)
    action=torch.randn(result["action"].shape,generator=gen)
    assert torch.allclose(result["action"],action-2,atol=1e-6)
    assert torch.allclose(result["flow"],flow-2,atol=1e-6)
    assert torch.equal(result["action"],sample_joint(model,b,steps=4,shift=5,seed=9)["action"])


def test_lora_initial_equivalence_and_frozen_base():
    base=nn.Linear(8,8); inputs=torch.randn(3,8)
    expected=base(inputs).detach(); layer=LoRALinear(base,2,4)
    assert torch.equal(layer(inputs),expected)
    layer(inputs).sum().backward()
    assert base.weight.grad is None and layer.b.weight.grad is not None


def test_explicit_mixed_rope_preserves_norm():
    x=torch.randn(2,7,4,12); positions=torch.tensor([[0,0,0],[1,0,0],[32,2,3],[65,0,0],[66,0,0],[67,0,0],[68,0,0]])
    theta=torch.arange(128)[:,None]*torch.arange(1,7)[None]*.01
    frequencies=torch.polar(torch.ones_like(theta),theta)
    rotated=rotary(x,positions,frequencies)
    assert torch.allclose(x.square().sum(-1),rotated.square().sum(-1),atol=1e-5)


def test_clean_future_probe_returns_only_action():
    c=Config(); c.model.targets=["action"]; model=JointModel(c.model); b=batch(c)
    result=model(b,{"action":b["action"]},torch.tensor([.2,.8]),known={"flow":b["flow"]})
    assert set(result)=={"action"}


def test_gradient_accumulation_matches_global_mean():
    torch.manual_seed(8); c=Config(); b=batch(c)
    full=JointModel(c.model); micro=copy.deepcopy(full)
    noisy,velocity,t=noisy_targets(b,c.model.targets)
    loss,_=modality_loss(full(b,noisy,t),velocity,b["has_action"],c.train.losses); loss.backward()
    for i in range(2):
        small={k:v[i:i+1] for k,v in b.items()}
        pred=micro(small,{k:v[i:i+1] for k,v in noisy.items()},t[i:i+1])
        loss,_=modality_loss(pred,{k:v[i:i+1] for k,v in velocity.items()},small["has_action"],c.train.losses)
        (loss/2).backward()
    for a,b in zip(full.parameters(),micro.parameters()):
        if a.grad is not None: assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5)


def test_direct_bc_bypasses_denoising():
    c=Config(); c.model.architecture="bc"; c.model.targets=["action"]
    model=build_model(c.model); b=batch(c)
    expected=model(b)["action"]
    assert torch.equal(sample_joint(model,b,steps=10)["action"],expected)


def test_activation_checkpointing_matches_gradients():
    c=Config(); plain=JointModel(c.model)
    checked=copy.deepcopy(plain); checked.config=copy.deepcopy(c.model); checked.config.gradient_checkpointing=True
    b=batch(c); noisy,velocity,t=noisy_targets(b,c.model.targets)
    for model in [plain,checked]:
        loss,_=modality_loss(model(b,noisy,t),velocity,b["has_action"],c.train.losses); loss.backward()
    for p,q in zip(plain.parameters(),checked.parameters()):
        if p.grad is not None: assert torch.allclose(p.grad,q.grad,atol=1e-6)


def test_wan_adapter_block_schema_and_lora_gradient():
    from flomo.model import Span
    class Attn(nn.Module):
        def __init__(self):
            super().__init__(); self.num_heads=2; self.head_dim=12
            for key in ["q","k","v","o"]: setattr(self,key,nn.Linear(24,24))
            self.norm_q=nn.Identity(); self.norm_k=nn.Identity()
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.self_attn=Attn(); self.cross_attn=Attn()
            self.norm1=nn.LayerNorm(24); self.norm2=nn.LayerNorm(24); self.norm3=nn.Identity()
            self.ffn=nn.Sequential(nn.Linear(24,48),nn.GELU(),nn.Linear(48,24))
            self.modulation=nn.Parameter(torch.randn(1,6,24)*.1)
    trunk=nn.Module(); trunk.blocks=nn.ModuleList([Block()]); trunk.requires_grad_(False)
    assert inject_lora(trunk,2,4)==10
    x=torch.randn(2,7,24,requires_grad=True); text=torch.randn(2,3,24)
    phases=torch.arange(128)[:,None]*torch.arange(1,7)[None]*.01
    freq=torch.polar(torch.ones_like(phases),phases)
    positions=torch.stack([torch.arange(7),torch.zeros(7),torch.zeros(7)],-1).long()
    log=[]
    y=JointModel._wan_block(trunk.blocks[0],x,torch.zeros(2,7,6,24),torch.ones(2,7,dtype=torch.bool),positions,freq,text,
                           torch.ones(2,3,dtype=torch.bool),[Span("obs",0,3,(1,1,3)),Span("action",3,7,None)],log)
    assert y.shape==x.shape and len(log)==1
    y.square().mean().backward()
    assert trunk.blocks[0].self_attn.q.base.weight.grad is None
    assert trunk.blocks[0].self_attn.q.b.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_mass_sums_to_one():
    c=Config(); model=JointModel(c.model).eval(); model.audit=True; b=batch(c)
    model(b,{"action":b["action"],"flow":b["flow"]},torch.tensor([.1,.2]))
    assert len(model.attention_log)==c.model.layers
    for layer in model.attention_log: assert abs(sum(s["mass"] for s in layer.values())-1)<1e-5


def test_fixed_joint_batch_can_be_fit():
    torch.manual_seed(19); c=Config(); model=JointModel(c.model); b=batch(c)
    noisy,velocity,t=noisy_targets(b,c.model.targets)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    def objective(): return modality_loss(model(b,noisy,t),velocity,b["has_action"],c.train.losses)[0]
    initial=float(objective().detach())
    for _ in range(60):
        optimizer.zero_grad(); loss=objective(); loss.backward(); optimizer.step()
    assert float(objective().detach())<initial*.6
