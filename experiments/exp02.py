"""Frozen motion-label VAE benchmark. No diffusion or text encoder is loaded.

Scientific artifacts: flow[T,Q,3], grid_size, initial_valid[Q], foreground[Q].
Source flow is cumulative displacement in initial camera coordinates, in meters.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/nvme/scratch/phyla-ubuntu')
LTX_REV='8984fa25007f376c1a299016d0957a37a2f797bb'
CONFIG_REV='e58e28c39631af4d1468ee57a853764e11c1d37e'
WAN_REV='921dbaf3f1674a56f47e83fb80a34bac8a8f203e'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def status(state,**kwargs):
 save_json(ROOT/'control/exp02/status.json',dict(experiment='exp02',state=state,pid=os.getpid(),updated_unix=time.time(),**kwargs))
def load_flow(path):
 with np.load(path,allow_pickle=False) as data:
  key=next((k for k in ['flow','flow_anchor','anchor_flow','displacement'] if k in data),None)
  if key is None:raise ValueError(f'No canonical flow key: {data.files}')
  flow=np.array(data[key],dtype=np.float32)
  grid=int(data['grid_size']) if 'grid_size' in data else int(data['grid_shape'][0]) if 'grid_shape' in data else int(round(np.sqrt(flow.shape[1])))
  if 'query_ids' in data and not np.array_equal(data['query_ids'],np.arange(grid*grid)):raise ValueError('Noncanonical grid query order')
  valid=np.array(data['initial_valid'],bool) if 'initial_valid' in data else np.isfinite(flow).all((0,2))
  fg=np.array(data['foreground'],bool) if 'foreground' in data else valid.copy()
 if flow.shape[1:]!=(grid*grid,3):raise ValueError('Flow must have ordered initial GxG grid')
 return flow,grid,valid,fg

def calibrate(paths,out):
 # Equal clips, finite valid points; foreground and background each contribute
 # equal samples when both exist. Fixed RNG does not inspect test sources.
 rng=np.random.default_rng(9821);samples=[]
 for p in paths:
  d,g,valid,fg=load_flow(p)
  for group in [valid&fg,valid&~fg]:
   values=d[1:,group].reshape(-1,3);values=values[np.isfinite(values).all(1)]
   if len(values):samples.append(values[rng.choice(len(values),10000,replace=len(values)<10000)])
 if not samples:raise ValueError('No calibration points')
 values=np.concatenate(samples);lo,hi=np.percentile(values,[1,99],axis=0)
 # At least 2cm full range, symmetrically enlarge only degenerate axes.
 mid=(lo+hi)/2;half=np.maximum((hi-lo)/2,.01);lo=mid-half;hi=mid+half
 save_json(out,dict(lower_m=lo.tolist(),upper_m=hi.tolist(),minimum_range_m=.02,percentiles=[1,99],sample_seed=9821,sources=[dict(path=str(p),sha256=sha(p)) for p in paths],split='calibration',sampling='10000_samples_per_clip_per_nonempty_foreground_or_background_group'))

def render(d,g,valid,lo,hi,size=256):
 import torch
 import torch.nn.functional as F
 finite=np.isfinite(d).all(-1)&valid[None]
 render_copy=np.where(finite[...,None],d,0)
 clipped=np.clip(render_copy,lo,hi)
 rgb=((clipped-lo)/(hi-lo)).astype(np.float32)
 x=torch.from_numpy(rgb.reshape(len(d),g,g,3)).permute(0,3,1,2)
 image=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
 return image,clipped,finite

def recover(rgb,g,lo,hi):
 import torch
 import torch.nn.functional as F
 # Query coordinates follow initial grid cells. align_corners=False means
 # normalized cell centers (2*(j+.5)/g -1), independent of image resolution.
 center=(torch.arange(g,device=rgb.device,dtype=rgb.dtype)+.5)*2/g-1
 y,x=torch.meshgrid(center,center,indexing='ij')
 samples=F.grid_sample(rgb,torch.stack([x,y],-1)[None].expand(len(rgb),-1,-1,-1),mode='bilinear',align_corners=False)
 values=samples.permute(0,2,3,1).reshape(len(rgb),g*g,3).float().cpu().numpy()
 return values*(hi-lo)+lo

def convert_ltx():
 import torch
 from safetensors import safe_open
 from diffusers import AutoencoderKLLTXVideo
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
 from flomo.model import convert_ltx_state,strict_component_state
 base=ROOT/'cache/checkpoints';source=base/'ltx096'/LTX_REV/'ltxv-2b-0.9.6-dev-04-25.safetensors';out=base/'ltx096-vae'/LTX_REV
 if (out/'vae-only.json').exists():return out
 cfg=base/'ltx095-config'/CONFIG_REV/'vae/config.json'
 with safe_open(str(source),framework='pt',device='cpu') as f:
  state={k:f.get_tensor(k) for k in f.keys() if k.startswith('vae.')}
 with torch.device('meta'):vae=AutoencoderKLLTXVideo.from_config(json.loads(cfg.read_text()))
 mapped=convert_ltx_state(state,'vae');strict_component_state(vae,mapped)
 out.mkdir(parents=True,exist_ok=True);vae.to(dtype=torch.bfloat16).save_pretrained(out,safe_serialization=True)
 save_json(out/'vae-only.json',dict(source_sha256=sha(source),revision=LTX_REV,config_revision=CONFIG_REV,config_sha256=sha(cfg),tensors=len(mapped),converter_version=2,dtype='bfloat16',component='vae_only'))
 return out

class Codec:
 def __init__(self,name):
  import torch
  self.name=name;self.dtype=torch.bfloat16
  if name=='ltx':
   from diffusers import AutoencoderKLLTXVideo
   self.path=convert_ltx();self.vae=AutoencoderKLLTXVideo.from_pretrained(self.path,local_files_only=True,torch_dtype=self.dtype).to('cuda').eval().requires_grad_(False)
   self.signature=json.loads((self.path/'vae-only.json').read_text())
  else:
   path=ROOT/'cache/checkpoints/wan-upstream/wan/modules/vae2_2.py'
   spec=importlib.util.spec_from_file_location('wan_vae_only',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   self.path=ROOT/'cache/checkpoints/wan22-ti2v5b'/WAN_REV/'Wan2.2_VAE.pth'
   self.vae=module.Wan2_2_VAE(vae_pth=str(self.path),device='cuda',dtype=self.dtype)
   self.vae.model.eval().requires_grad_(False)
   self.signature=json.loads(Path(str(self.path)+'.complete.json').read_text());self.signature['upstream_sha']=os.popen(f'git -C {path.parents[2]} rev-parse HEAD').read().strip()
 def roundtrip(self,image):
  import torch
  x=(image.to('cuda',dtype=self.dtype)*2-1).permute(1,0,2,3)
  with torch.inference_mode():
   if self.name=='wan':
    latent=self.vae.encode([x])[0]
    # Public Wan decode clamps internally. Call the same frozen core decoder
    # directly so overshoot and the raw physical reconstruction remain visible.
    with torch.autocast('cuda',dtype=self.dtype):
     native=self.vae.model.decode(latent[None],self.vae.scale)[0]
    raw=(native.permute(1,0,2,3).float()+1)/2
   else:
    z=self.vae.encode(x[None]).latent_dist.mode()
    mean=self.vae.latents_mean.reshape(1,-1,1,1,1).to(z);std=self.vae.latents_std.reshape(1,-1,1,1,1).to(z)
    scale=self.vae.config.scaling_factor
    if not torch.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid native LTX scales')
    latent=((z-mean)*scale/std)[0]
    restored=latent[None]*std/scale+mean
    timestep=torch.zeros(1,device='cuda',dtype=self.dtype) if self.vae.config.timestep_conditioning else None
    raw=(self.vae.decode(restored,timestep).sample[0].permute(1,0,2,3).float()+1)/2
  return latent,raw

def metric(pred,target,mask):
 delta=np.asarray(pred,dtype=np.float64)-np.asarray(target,dtype=np.float64)
 values=delta[mask]
 if not len(values):return dict(count=0)
 e=np.linalg.norm(values,axis=-1)
 return dict(count=len(e),epe_m=float(e.mean()),median_m=float(np.median(e)),p95_m=float(np.percentile(e,95)),axis_mae_m=np.abs(values).mean(0).tolist(),mse_m2=float(np.square(values).sum(-1).mean()))
def run(args,codec_instance=None):
 import torch
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 stats=json.loads(Path(args.stats).read_text());lo=np.array(stats['lower_m']);hi=np.array(stats['upper_m'])
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 signature=dict(implementation_sha256=sha(__file__),source_kind=args.source_kind,source_sha256=sha(args.source),gt_sha256=sha(args.gt or args.source),stats_sha256=sha(args.stats),codec=args.codec,frames=args.frames,render_size=256)
 if (out/'complete.json').exists():
  existing=json.loads((out/'complete.json').read_text())
  if existing['input_signature']==signature:return
  raise ValueError('Output exists with incompatible signature')
 d,g,valid,fg=load_flow(args.source);d=d[:args.frames]
 gt,gg,gv,gf=load_flow(args.gt or args.source);gt=gt[:args.frames]
 if gt.shape!=d.shape or gg!=g:raise ValueError('GT/source query shape mismatch')
 image,clipped,finite=render(d,g,valid,lo,hi)
 bar=recover(image,g,lo,hi)
 mask=finite&gv[None];mask[0]=False
 codec=codec_instance or Codec(args.codec);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 latent,decoded=codec.roundtrip(image);torch.cuda.synchronize();seconds=time.perf_counter()-start
 expected=(48,(len(d)-1)//4+1,16,16) if args.codec=='wan' else (128,(len(d)-1)//8+1,8,8)
 if tuple(latent.shape)!=expected:raise ValueError(f'Latent shape {latent.shape}, expected {expected}')
 if decoded.shape!=image.shape:raise ValueError(f'Decode shape {decoded.shape}, expected {image.shape}')
 recovered_raw=recover(decoded,g,lo,hi);clamped=decoded.clamp(0,1);recovered=recover(clamped,g,lo,hi)
 tensor_cpu=lambda value:value.detach().float().cpu().numpy()
 rgb_raw=tensor_cpu(decoded);rgb=tensor_cpu(image)
 terms=[d-gt,clipped-d,bar-clipped,recovered-bar]
 identity=np.max(np.abs(sum(terms)[mask]-(recovered-gt)[mask]))
 if identity>1e-6:raise ValueError(f'Error vector identity failed {identity}')
 report=dict(input_signature=signature,codec_signature=codec.signature,latent_shape=list(latent.shape),inference_seconds=seconds,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),latent_bytes=latent.numel()*latent.element_size(),source_flow_bytes=d.nbytes,rgb_input_bytes=rgb.nbytes,finite_coverage=float(finite[:,gv].mean()),decoder_overshoot_fraction=float(((rgb_raw<0)|(rgb_raw>1)).mean()),identity_max_error_m=float(identity))
 for label,pred,target in [('tracking',d,gt),('range',clipped,d),('raster',bar,clipped),('codec_self',recovered,bar),('decoded_source',recovered,d),('decoded_gt',recovered,gt),('raw_decoder_gt',recovered_raw,gt)]:
  report[label]=metric(pred,target,mask)
  report[label+'_foreground']=metric(pred,target,mask&fg[None]);report[label+'_background']=metric(pred,target,mask&~fg[None])
 diff=(tensor_cpu(clamped)-rgb)[1:];report['rgb_mse']=float(np.square(diff).mean());report['rgb_psnr']=float(-10*np.log10(max(report['rgb_mse'],1e-30)))
 report['clipping_fraction_axis']=np.mean(((d<lo)|(d>hi))[mask],axis=0).tolist()
 velocity=metric((recovered[1:]-recovered[:-1])*20,(gt[1:]-gt[:-1])*20,mask[1:]&finite[:-1])
 report['velocity_error']=dict(epe_m_per_s=velocity.pop('epe_m'),axis_mae_m_per_s=velocity.pop('axis_mae_m'),remaining_metrics=velocity)
 np.savez_compressed(out/'reconstruction.npz',source_flow=d,gt_flow=gt,clipped_flow=clipped,render_inverse_flow=bar,decoded_flow=recovered,raw_decoded_flow=recovered_raw,rgb_input=rgb,decoded_rgb_raw=rgb_raw,decoded_rgb_clamped=tensor_cpu(clamped),valid=mask,foreground=fg,grid_size=g)
 torch.save(latent.detach().cpu(),out/'latent.pt');save_json(out/'metrics.json',report)
 fig,axs=plt.subplots(2,3,figsize=(12,7));idx=len(d)-1
 axs[0,0].imshow(np.clip(rgb[idx].transpose(1,2,0),0,1));axs[0,0].set_title('Source motion RGB')
 axs[0,1].imshow(tensor_cpu(clamped)[idx].transpose(1,2,0));axs[0,1].set_title(f'{args.codec} decoded')
 error=np.linalg.norm(recovered[idx]-gt[idx],axis=-1).reshape(g,g)
 im=axs[0,2].imshow(error,cmap='magma',vmin=0);fig.colorbar(im,ax=axs[0,2],label='m');axs[0,2].set_title('Decoded vs GT EPE')
 for ax,name,p,t in zip(axs[1],['dx','dy','dz'],range(3),range(3)):
  ax.plot(d[:,valid,p].mean(1),label='source');ax.plot(recovered[:,valid,p].mean(1),label='decoded');ax.set_title(name);ax.set_xlabel('frame');ax.set_ylabel('m');ax.legend()
 fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
 save_json(out/'complete.json',dict(input_signature=signature,metrics_sha256=sha(out/'metrics.json'),completed_unix=time.time()))
 save_json(ROOT/'control/exp02/jobs'/f'{hashlib.sha256(str(out).encode()).hexdigest()[:12]}-{args.codec}.json',dict(state='complete',source_kind=args.source_kind,codec=args.codec,result=str(out),metrics_path=str(out/'metrics.json'),pid=os.getpid(),updated_unix=time.time()))
 print(json.dumps(report,indent=2),flush=True)
def synth(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);g=32;t=17
 y,x=np.meshgrid(np.linspace(-1,1,g),np.linspace(-1,1,g),indexing='ij');fg=(x*x+y*y<.45).reshape(-1);valid=np.ones(g*g,bool)
 for split,amp in [('calibration',.12),('test',.10)]:
  flow=np.zeros((t,g*g,3),np.float32)
  for i in range(t):
   phase=i/(t-1);flow[i,fg,0]=amp*phase;flow[i,fg,1]=.06*np.sin(phase*np.pi);flow[i,fg,2]=.04*phase
  np.savez_compressed(out/(split+'.npz'),flow=flow,grid_size=g,initial_valid=valid,foreground=fg)
 np.savez_compressed(out/'zero.npz',flow=np.zeros_like(flow),grid_size=g,initial_valid=valid,foreground=fg)
 save_json(out/'metadata.json',dict(kind='synthetic_displacement_fixture_NOT_simulator_GT',purpose='GPU codec acceptance only',units='m',grid_size=g,frames=t))
def batch(args):
 jobs=json.loads(Path(args.manifest).read_text());codec=Codec(args.codec);completed=[];failed=[]
 for job in jobs:
  child=argparse.Namespace(codec=args.codec,source=job['source'],gt=job.get('gt'),source_kind=job['source_kind'],stats=args.stats,output=str(Path(args.output)/job['id']/args.codec),frames=args.frames)
  try:
   run(child,codec);completed.append(job['id'])
  except Exception as exc:
   failed.append(dict(id=job['id'],error=repr(exc)));print('FAILED',job['id'],repr(exc),flush=True)
  save_json(ROOT/'control/exp02'/f'{args.codec}-batch.json',dict(codec=args.codec,stage='simulator_codec_pilot',state='running' if len(completed)+len(failed)<len(jobs) else 'complete_with_failures' if failed else 'complete',pid=os.getpid(),jobs_total=len(jobs),completed=completed,failed=failed,updated_unix=time.time()))
 if failed:raise RuntimeError(f'{len(failed)} codec jobs failed')
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 c=sub.add_parser('synth');c.add_argument('--output',required=True)
 c=sub.add_parser('calibrate');c.add_argument('--source',nargs='+',required=True);c.add_argument('--output',required=True)
 c=sub.add_parser('convert-ltx-vae')
 c=sub.add_parser('batch');c.add_argument('--codec',choices=['wan','ltx'],required=True);c.add_argument('--manifest',required=True);c.add_argument('--stats',required=True);c.add_argument('--output',required=True);c.add_argument('--frames',type=int,default=17)
 c=sub.add_parser('run');c.add_argument('--codec',choices=['wan','ltx'],required=True);c.add_argument('--source',required=True);c.add_argument('--gt');c.add_argument('--stats',required=True);c.add_argument('--output',required=True);c.add_argument('--frames',type=int,default=17);c.add_argument('--source-kind',choices=['synthetic_acceptance','simulator_gt','tracker_prediction','zero_motion_control'],required=True)
 args=p.parse_args()
 if args.cmd=='synth':synth(args.output)
 elif args.cmd=='calibrate':calibrate(args.source,args.output)
 elif args.cmd=='convert-ltx-vae':print(convert_ltx())
 elif args.cmd=='batch':batch(args)
 else:run(args)
if __name__=='__main__':main()
