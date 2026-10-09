"""Frozen tracker adapters; native arrays preserved alongside canonical output."""
import argparse,json,time,sys,hashlib,random
from pathlib import Path
import numpy as np
import torch

def metrics(flow,gt,mask):
    finite=np.isfinite(flow).all(-1); eligible=mask.copy(); eligible[0]=False
    good=eligible&finite; err=np.linalg.norm(flow-gt,axis=-1)
    return {'epe_m':float(err[good].mean()) if good.any() else None,'p95_m':float(np.percentile(err[good],95)) if good.any() else None,
            'coverage':float(finite[eligible].mean()),'pck_1cm':float((good&(err<.01)).sum()/eligible.sum()),'eligible':int(eligible.sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--repo',required=True);p.add_argument('--bf16',action='store_true');p.add_argument('--geometry-cache');p.add_argument('--method',choices=['cotracker3','spatrackerv2','delta'],default='cotracker3');a=p.parse_args()
    sys.path.insert(0,a.repo)
    if a.method=='cotracker3':
        from cotracker.predictor import CoTrackerPredictor
        model=CoTrackerPredictor(checkpoint=a.checkpoint).cuda().eval()
    elif a.method=='delta':
        from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
        from densetrack3d.models.predictor.predictor import Predictor3D
        base=DenseTrack3D(stride=4,window_len=16,add_space_attn=True,num_virtual_tracks=64,model_resolution=(384,512),upsample_factor=4)
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);state=state.get('model',state)
        incompatible=base.load_state_dict(state,strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(f'DELTA checkpoint mismatch: {incompatible}')
        model=Predictor3D(base).cuda().eval()
    else:
        import torchvision.io
        def unavailable_write_video(*args,**kwargs): raise RuntimeError("Unused legacy video writer called")
        torchvision.io.write_video=unavailable_write_video
        from models.SpaTrackV2.models.predictor import Predictor
        model=Predictor.from_pretrained(a.checkpoint);model.to('cuda');model.eval();model.spatrack.track_num=256
    for clip in sorted(Path(a.data).glob('*/gt.npz')):
        out=Path(a.output)/clip.parent.name
        if (out/'READY').exists():continue
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        d=np.load(clip); geometry=np.load(Path(a.geometry_cache)/clip.parent.name/'geometry.npz') if a.geometry_cache else d
        valid=d['initial_valid']; ids=np.flatnonzero(valid); uv=d['query_uv'][ids]; t=len(d['rgb']); q=len(valid)
        video=torch.from_numpy(d['rgb'].copy()).permute(0,3,1,2).float().cuda();queries=np.c_[np.zeros(len(ids)),uv].astype(np.float32)
        torch.cuda.reset_peak_memory_stats();start=time.time()
        with torch.inference_mode():
            if a.method=='cotracker3':
                uvhat,vis=model(video[None],queries=torch.from_numpy(queries)[None].cuda());uvhat=uvhat[0].cpu().numpy();vis=vis[0].cpu().numpy()
                depth=geometry['depth']; px=np.rint(uvhat[...,0]-.5).astype(int);py=np.rint(uvhat[...,1]-.5).astype(int)
                inside=(px>=0)&(px<256)&(py>=0)&(py<256)
                z=depth[np.arange(t)[:,None],np.clip(py,0,255),np.clip(px,0,255)]; z[~inside|~np.isfinite(z)|(z<=0)]=np.nan
                ray=np.einsum('tij,tqj->tqi',np.linalg.inv(geometry['intrinsics']),np.concatenate([uvhat,np.ones((*uvhat.shape[:2],1))],-1))
                pc=ray*z[...,None];world=np.einsum('tij,tqj->tqi',geometry['camera_to_world'][:,:3,:3],pc)+geometry['camera_to_world'][:,:3,3][:,None]
                raw={'uv':uvhat,'visibility':vis,'sampled_depth':z,'camera_xyz':pc,'world_xyz':world}
            elif a.method=='delta':
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.bf16):
                    result=model(video[None],torch.from_numpy(d['depth'].copy())[None,:,None].float().cuda(),queries=torch.from_numpy(queries)[None].cuda(),predefined_intrs=torch.from_numpy(d['intrinsics'][0].copy()).float().cuda())
                raw={name:value[0].detach().cpu().float().numpy() for name,value in result.items() if isinstance(value,torch.Tensor)}
                raw.update({f'native_3d_{name}':value[0].detach().cpu().float().numpy() for name,value in result['trajs_3d_dict'].items()})
                pc=raw['native_3d_coords'];world=np.einsum('tij,tqj->tqi',geometry['camera_to_world'][:,:3,:3],pc)+geometry['camera_to_world'][:,:3,3][:,None]
                raw['camera_xyz']=pc;raw['world_xyz']=world
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.bf16):
                    ret=model.forward(video.cpu(),depth=geometry['depth'],intrs=geometry['intrinsics'].astype(np.float32),extrs=geometry['camera_to_world'].astype(np.float32),queries=queries,unc_metric=(geometry['confidence']>.5) if a.geometry_cache else None,
                        fps=1,full_point=False,iters_track=4,query_no_BA=True,fixed_cam=not bool(a.geometry_cache),stage=1,support_frame=t-1,replace_ratio=.2)
                names=['camera_to_world','intrinsics','point_map','depth_confidence','camera_xyz','uv','visibility','confidence','video']
                raw={name:value.detach().cpu().float().numpy() if isinstance(value,torch.Tensor) else value for name,value in zip(names,ret)}
                pc=raw['camera_xyz'][...,:3];world=np.einsum('tij,tqj->tqi',raw['camera_to_world'][:,:3,:3],pc)+raw['camera_to_world'][:,:3,3][:,None]
                raw['world_xyz']=world
        raw['input_queries']=queries;raw['selected_query_ids']=ids
        elapsed=time.time()-start; canonical=np.full((t,q,3),np.nan,np.float32)
        canonical[:,ids]=(world-world[0])@(raw['camera_to_world'][0,:3,:3] if a.method=='spatrackerv2' else geometry['camera_to_world'][0,:3,:3])
        scale=1.0
        if a.geometry_cache:
            native_flow=canonical.copy();gt_pc0=(d['world_xyz'][0,ids]-d['camera_to_world'][0,:3,3])@d['camera_to_world'][0,:3,:3]
            usable=np.isfinite(pc[0,:,2])&(pc[0,:,2]>0)&np.isfinite(gt_pc0[:,2])&(gt_pc0[:,2]>0)
            scale=float(np.median(gt_pc0[usable,2]/pc[0,usable,2]));canonical*=scale;raw['flow_native_arbitrary_scale']=native_flow;raw['initial_gt_depth_scale']=np.array(scale)
        out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'raw.npz',**raw)
        np.savez_compressed(out/'tracks.npz',flow=canonical,finite=np.isfinite(canonical).all(-1),query_ids=d['query_ids'],initial_valid=valid,grid_shape=d['grid_shape'],foreground=d['foreground'],visible=d['visible'],timestamps=d['timestamps'])
        report={'seed':0,'query_count':int(len(ids)),'support_policy':{'cotracker3':'native6x6supportgrid','delta':'native6x6supportgrid+64virtualtracks','spatrackerv2':'track_num256, replace_ratio0.2, support_frame32, iters4'}[a.method],'precision':'bf16_outer_autocast' if a.bf16 else 'float32_outer_native_internal_attention_precision','method':a.method,'regime':'shared_predicted_geometry_initial_GT_scale_diagnostic' if a.geometry_cache else 'gt_depth_gt_camera','initial_gt_depth_scale':scale,'lift_validity':'finite_positive_front_surface_depth' if a.method=='cotracker3' else 'finite_direct3d_model_output','geometry_cache':a.geometry_cache,'elapsed_s':elapsed,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'checkpoint':a.checkpoint,'input_sha256':hashlib.sha256(clip.read_bytes()).hexdigest(),
            'all':metrics(canonical,d['flow'],d['trajectory_valid']),'visible':metrics(canonical,d['flow'],d['visible']),
            'foreground':metrics(canonical,d['flow'],d['trajectory_valid']&d['foreground'][None]),'static_background':metrics(canonical,d['flow'],d['trajectory_valid']&(~d['foreground'])[None])}
        (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'READY').write_text('validated_shape_raw_preserved\n');print(json.dumps({'clip':clip.parent.name,**report}),flush=True)

if __name__=='__main__':main()
