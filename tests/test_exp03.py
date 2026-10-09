import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flomo.config import Config,encoder_signature
from flomo.model import FrozenEncoders,LoRALinear
from flomo.data import prepare,torch_load,read_jsonl
from flomo.sim import collect
from flomo.lora_audit import assert_lora_only,assert_base_unchanged

def test_matched_actual_future_frames(tmp_path):
 c=Config();c.model.future_only=True;c.data.raw=str(tmp_path/"raw");c.data.prepared=str(tmp_path/"prepared");c.data.val_fraction=0;c.data.test_fraction=0;c.data.stride=16;c.eval.horizon=32
 collect(c,count=1)
 class Spy(FrozenEncoders):
  def __init__(self,c):super().__init__(c);self.movies=[]
  def encode_video(self,x):self.movies.append(x.clone());return super().encode_video(x)
 enc=Spy(c.model);meta=prepare(c,enc)
 assert meta["flow_window"]=="frames_1_to_horizon_anchor_0"
 assert meta["video_window"]=="frames_1_to_horizon"
 # For each window: singleton observation,16 actual RGB futures,16 flow futures.
 assert [len(x) for x in enc.movies[:3]]==[1,16,16]
 with np.load(tmp_path/"raw/sim_000000.npz") as ep:
  expected=torch.from_numpy(ep["rgb"][1:17,0].copy()).permute(0,3,1,2).float()/255
 assert torch.equal(enc.movies[1],expected)
 tracks=np.load(next((tmp_path/"prepared/tracks").glob("*.npz")))
 assert len(tracks["flow"])==17 and np.allclose(tracks["flow"][0],0)
 old=copy.deepcopy(c);old.model.future_only=False
 assert encoder_signature(c)!=encoder_signature(old)

def test_lora_optimizer_allowlist_and_base_integrity():
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.config=SimpleNamespace(backend="ltx");self.trunk=LoRALinear(torch.nn.Linear(3,3),2,2);self.action_in=torch.nn.Linear(2,3);self.action_out=torch.nn.Linear(3,2);self.action_position=torch.nn.Parameter(torch.zeros(2,3))
 m=Model();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]);a=assert_lora_only(m,opt)
 loss=m.action_out(m.trunk(m.action_in(torch.ones(1,2)))).square().mean();loss.backward();opt.step();assert_base_unchanged(m,a)
 with torch.no_grad():m.trunk.base.weight.add_(1)
 with pytest.raises(ValueError,match="base changed"):assert_base_unchanged(m,a)
 m=Model();m.trunk.base.weight.requires_grad=True
 with pytest.raises(ValueError,match="violation"):assert_lora_only(m)

def test_action_only_dummy_future_has_no_content_path():
 from flomo.model import JointModel
 c=Config().model;c.auxiliary_null=True
 m=JointModel(c).eval();b={"obs":torch.randn(1,1,6,1,8,8),"view_valid":torch.ones(1,1,dtype=torch.bool),"text":torch.randn(1,16,32),"text_valid":torch.ones(1,16,dtype=torch.bool),"has_action":torch.ones(1,dtype=torch.bool)}
 a=torch.randn(1,16,2);f=torch.randn(1,6,5,8,8);t=torch.tensor([.5])
 first=m(b,{"action":a,"flow":f},t)["action"]
 second=m(b,{"action":a,"flow":f*100},t)["action"]
 assert torch.allclose(first,second,atol=1e-6)

def test_flow_rgb_ltx_layout_and_attention_identical():
 from test_ltx import fixture_model,fixture_batch
 from flomo.model import LTXJointModel
 c,trunk=fixture_model();c.future_only=True
 f=LTXJointModel(c,copy.deepcopy(trunk)).eval()
 rconfig=copy.deepcopy(c);rconfig.targets=["video","action"]
 r=LTXJointModel(rconfig,copy.deepcopy(trunk)).eval();r.load_state_dict(f.state_dict())
 b=fixture_batch(c);t=torch.tensor([.2,.6]);noisy_f={"flow":b["flow"],"action":b["action"]};noisy_r={"video":b["flow"],"action":b["action"]}
 pf=f._pack(b,noisy_f,t,{});pr=r._pack(b,noisy_r,t,{})
 for index in range(4):assert torch.equal(pf[index],pr[index])
 assert [(s.start,s.stop,s.grid) for s in pf[4]]==[(s.start,s.stop,s.grid) for s in pr[4]]
 assert torch.allclose(f(b,noisy_f,t)["action"],r(b,noisy_r,t)["action"],atol=1e-6)


def test_bf16_decoder_has_numpy_safe_public_boundary():
 from flomo.model import FrozenEncoders
 class StubVAE(torch.nn.Module):
  def __init__(self):
   super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16));self.config=SimpleNamespace(timestep_conditioning=False)
  def decode(self,value,timestep):return SimpleNamespace(sample=torch.zeros(1,3,17,32,32,dtype=torch.bfloat16))
 encoder=FrozenEncoders.__new__(FrozenEncoders);encoder.config=SimpleNamespace(backend="ltx");encoder.device=torch.device("cpu");encoder.vae=StubVAE();encoder._ltx_scale=lambda x,inverse=False:x
 result=encoder.decode_video(torch.zeros(4,3,1,1,dtype=torch.bfloat16),16)
 assert result.dtype==torch.float32 and result.numpy().shape==(16,3,32,32)
