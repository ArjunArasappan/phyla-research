"""Physical error decomposition on immutable GT masks and native camera diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from exp01_track import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();root=Path(a.data)
    for track in sorted((root/'predictions').glob('*/*/tracks.npz')):
        if not (track.parent/'READY').exists():continue
        gtfile=root/'pilot'/track.parent.name/'gt.npz'
        if not gtfile.exists():continue
        d=np.load(gtfile);prediction=np.load(track);flow=prediction['flow'];gt=d['flow'];valid=d['trajectory_valid'];finite=np.isfinite(flow).all(-1)
        if flow.shape!=gt.shape:raise RuntimeError('Trajectory shape changed')
        pc_gt=np.einsum('tqj,tjk->tqk',d['world_xyz']-d['camera_to_world'][:,None,:3,3],d['camera_to_world'][:,:3,:3])
        mask=valid&finite;mask=mask.copy();mask[0]=False;difference=flow-gt
        groups={'all':valid,'visible':d['visible'],'occluded_in_view':d['in_frame']&~d['visible'],'out_of_view_front':valid&~d['in_frame']&(pc_gt[...,2]>0),
                'foreground':valid&d['foreground'][None],'background':valid&~d['foreground'][None],'dynamic_gt':valid&(np.linalg.norm(gt,axis=-1)>.005),
                'static_gt':valid&(np.linalg.norm(gt,axis=-1)<1e-8)}
        summary={name:metrics(flow,gt,eligible) for name,eligible in groups.items() if eligible[1:].any()}
        summary['axis_mae_m']=np.abs(difference[mask]).mean(0).tolist();summary['axis_rmse_m']=np.sqrt((difference[mask]**2).mean(0)).tolist()
        dt=np.diff(d['timestamps']);v=np.diff(flow,axis=0)/dt[:,None,None];vgt=np.diff(gt,axis=0)/dt[:,None,None];vmask=valid[1:]&valid[:-1]&finite[1:]&finite[:-1]
        summary['velocity_epe_m_per_s']=float(np.linalg.norm(v-vgt,axis=-1)[vmask].mean())
        acceleration=np.diff(v,axis=0)/dt[1:,None,None];accgt=np.diff(vgt,axis=0)/dt[1:,None,None];amask=vmask[1:]&vmask[:-1]
        summary['acceleration_epe_m_per_s2']=float(np.linalg.norm(acceleration-accgt,axis=-1)[amask].mean())
        rawfile=track.parent/'raw.npz'
        if rawfile.exists():
            raw=np.load(rawfile);meta=json.loads((track.parent/'metrics.json').read_text());scale=meta.get('initial_gt_depth_scale',1.)
            if 'geometry_cache' in meta and meta['geometry_cache']:
                geometry=np.load(Path(meta['geometry_cache'])/track.parent.name/'geometry.npz');camera=raw['camera_to_world'] if 'camera_to_world' in raw else geometry['camera_to_world']
                predrel=np.linalg.inv(camera[0])@camera;gtrel=np.linalg.inv(d['camera_to_world'][0])@d['camera_to_world']
                delta=np.einsum('tji,tjk->tik',gtrel[:,:3,:3],predrel[:,:3,:3]);angles=np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1))*180/np.pi
                errors=np.linalg.norm(predrel[:,:3,3]*scale-gtrel[:,:3,3],axis=-1)
                summary['camera_initial_GT_scale_diagnostic']={'rotation_mean_deg':float(angles[1:].mean()),'rotation_final_deg':float(angles[-1]),'translation_mean_m':float(errors[1:].mean()),'translation_final_m':float(errors[-1]),'scale':scale}
                depth=geometry['depth']*scale;depvalid=d['depth']>0
                summary['frontend_depth_initial_GT_scale_absrel']=float((np.abs(depth-d['depth'])/np.maximum(d['depth'],1e-8))[depvalid].mean())
                summary['frontend_focal_relative_error']=float(np.abs(geometry['intrinsics'][:,[0,1],[0,1]]/d['intrinsics'][:,[0,1],[0,1]]-1).mean())
        (track.parent/'extended_metrics.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(track.parent/'physical_errors.npz',flow_error=difference,finite=finite,eligible=valid,visible=d['visible'],foreground=d['foreground'],velocity_error=v-vgt)
    print('Physical masks, axis/velocity/acceleration and camera diagnostics saved.')

if __name__=='__main__':main()
