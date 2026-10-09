"""Oracle lifting closure and explicit zero-motion baseline; no learned model."""
import argparse,json
from pathlib import Path
import numpy as np
from exp01_track import metrics
from exp01 import ray_hits

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    for clip in sorted(Path(a.data).glob('*/gt.npz')):
        d=np.load(clip);pc=np.einsum('tqj,tjk->tqk',d['world_xyz']-d['camera_to_world'][:,None,:3,3],d['camera_to_world'][:,:3,:3])
        uvh=np.einsum('tij,tqj->tqi',d['intrinsics'],pc);uv=uvh[...,:2]/uvh[...,2:3]
        # Avoid converting undefined initial rays to integer positions.
        px=np.floor(np.nan_to_num(uv[...,0],nan=-1)).astype(int);py=np.floor(np.nan_to_num(uv[...,1],nan=-1)).astype(int)
        inside=(px>=0)&(px<256)&(py>=0)&(py<256)&d['trajectory_valid'];z=d['depth'][np.arange(len(pc))[:,None],np.clip(py,0,255),np.clip(px,0,255)].copy();z[~inside]=np.nan
        ray=np.einsum('tij,tqj->tqi',np.linalg.inv(d['intrinsics']),np.concatenate([uv,np.ones((*uv.shape[:2],1))],-1))
        lifted=np.einsum('tij,tqj->tqi',d['camera_to_world'][:,:3,:3],ray*z[...,None])+d['camera_to_world'][:,None,:3,3]
        liftedflow=(lifted-lifted[0])@d['camera_to_world'][0,:3,:3]
        exact_world=[]
        half_sizes=[np.array([1.4,1.4,.06]),np.array([.16,.22,.2]),np.array([.13,.22,.18])]
        for t in range(len(pc)):
            cv=d['camera_to_world'][t];direction=ray[t]@cv[:3,:3].T
            hit,hitid,front_z=ray_hits(np.broadcast_to(cv[:3,3],direction.shape),direction,d['body_poses'][t],half_sizes)
            exact_world.append(hit)
        exact_world=np.stack(exact_world);exactflow=(exact_world-exact_world[0])@d['camera_to_world'][0,:3,:3]
        for name,flow in [('gt_uv_exact_front_ray',exactflow),('gt_uv_gt_front_depth',liftedflow),('zero_motion',np.where(d['trajectory_valid'][...,None],0.,np.nan))]:
            out=Path(a.output)/name/clip.parent.name;out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'tracks.npz',flow=flow.astype(np.float32),initial_valid=d['initial_valid'],query_ids=d['query_ids'],grid_shape=d['grid_shape'],foreground=d['foreground'],visible=d['visible'],timestamps=d['timestamps'])
            report={'method':name,'visible':metrics(flow,d['flow'],d['visible']),'all_diagnostic_not_hidden_point_oracle':metrics(flow,d['flow'],d['trajectory_valid']),
                    'note':'GT UV samples visible front-surface rendered axial depth; hidden point depth never provided.'}
            (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'READY').write_text('shape_validated\n')

if __name__=='__main__':main()
