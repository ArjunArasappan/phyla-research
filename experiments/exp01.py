"""Real SAPIEN rigid-scene tracker benchmark. Frozen material query identities."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)

def matrix_pose(matrix):
    import sapien
    from scipy.spatial.transform import Rotation
    xyzw=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    return sapien.Pose(matrix[:3,3],xyzw[[3,0,1,2]])

def camera_pose(angle):
    eye=np.array([2.4*np.cos(angle),2.4*np.sin(angle),1.35])
    forward=np.array([0.,0.,.35])-eye; forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1.]); right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    cv=np.eye(4); cv[:3,:3]=np.stack([right,down,forward],1); cv[:3,3]=eye
    mount=cv.copy(); mount[:3,:3]=np.stack([forward,-right,-down],1)
    return mount,cv

def body_pose(center,angle=0):
    from scipy.spatial.transform import Rotation
    p=np.eye(4); p[:3,3]=center; p[:3,:3]=Rotation.from_euler('z',angle).as_matrix(); return p

def ray_hits(origins,directions,poses,half_sizes):
    """Exact first hit against the same boxes supplied to the actual renderer."""
    count=len(directions); distance=np.full(count,np.inf); ids=np.full(count,-1,int)
    for i,(pose,half) in enumerate(zip(poses,half_sizes)):
        o=(origins-pose[:3,3])@pose[:3,:3]; d=directions@pose[:3,:3]
        with np.errstate(divide='ignore',invalid='ignore'):
            lo=(-half-o)/d; hi=(half-o)/d
        entry=np.max(np.minimum(lo,hi),axis=1); exit=np.min(np.maximum(lo,hi),axis=1)
        hit=(exit>=np.maximum(entry,0))&(entry>0)&(entry<distance)
        distance[hit]=entry[hit]; ids[hit]=i
    points=origins+directions*distance[:,None]
    points[ids<0]=np.nan
    return points,ids,distance

def generate_clip(out,archetype,mode,seed,grid=16,frames=33):
    import sapien
    if (out/"READY").exists(): return
    rng=np.random.default_rng(seed); offset=rng.uniform(-.05,.05,3); offset[2]=0
    if seed==100: offset[:]=0 # preserves first validated static camera pair
    scene=sapien.Scene(); scene.set_ambient_light([.5,.5,.5])
    scene.add_directional_light([1,1,-2],[3,3,3],shadow=True)
    halves=[np.array([1.4,1.4,.06]),np.array([.16,.22,.2]),np.array([.13,.22,.18])]
    colors=[[.35,.4,.5,1],[.8,.2,.1,1],[.1,.65,.3,1]]
    actors=[]
    for i,(half,color) in enumerate(zip(halves,colors)):
        b=scene.create_actor_builder(); mat=sapien.render.RenderMaterial(); mat.base_color=color
        b.add_box_visual(half_size=half,material=mat); actors.append(b.build_kinematic(name=f'box_{i}'))
    camera=scene.add_camera('benchmark',256,256,np.pi/3,.01,20.)
    pixels=np.rint(np.linspace(4,251,grid)).astype(int)
    uu,vv=np.meshgrid(pixels,pixels); uv0=np.stack([uu.ravel(),vv.ravel()],-1).astype(float)+.5
    rgb=[]; depth=[]; cams=[]; body=[]; xyz=[]; visible=[]; inframe=[]; uv_all=[]; analytic=[]
    k=np.asarray(camera.get_intrinsic_matrix()).copy()
    for t in range(frames):
        phase=t/(frames-1); angle=.55*phase if mode=='orbit' else 0.
        mount,cv=camera_pose(angle); camera.set_pose(matrix_pose(mount))
        c1=np.array([0.,-.28,.32])+offset; c2=np.array([-.4,.35,.3])-offset; rot=0.
        if archetype=='object_translation': c1+=np.array([.42*phase,.25*phase,.08*np.sin(phase*np.pi)])
        elif archetype=='rigid_rotation': rot=phase*np.pi*.8
        elif archetype=='articulated_links': c2=c1+np.array([.36*np.cos(phase*np.pi),.36*np.sin(phase*np.pi),.1]); rot=phase*np.pi
        elif archetype=='occlusion_and_return': c1[1]=-.65+1.3*phase; c2=np.array([.5,0,.34])
        elif archetype=='depth_speed_out_of_view_stress': c1+=np.array([-.5*phase,2.2*phase,.12*phase])
        poses=np.stack([body_pose([0,0,-.06]),body_pose(c1,rot),body_pose(c2)])
        for actor,pose in zip(actors,poses): actor.set_pose(matrix_pose(pose))
        scene.update_render(); camera.take_picture()
        image=np.asarray(camera.get_picture('Color')); pos=np.asarray(camera.get_picture('Position'))
        rgb.append(np.clip(image[:,:,:3]*255,0,255).astype(np.uint8)); depth.append(-pos[:,:,2])
        # The renderer's exported extrinsic is authoritative; acceptance checks verify pose agreement.
        extrinsic=np.eye(4); extrinsic[:3]=np.asarray(camera.get_extrinsic_matrix())
        render_cv=np.linalg.inv(extrinsic)
        if not np.allclose(render_cv,cv,atol=1e-5): raise RuntimeError('SAPIEN camera convention mismatch')
        if t==0:
            dirs_cam=np.c_[uv0,np.ones(len(uv0))]@np.linalg.inv(k).T
            dirs_world=dirs_cam@cv[:3,:3].T
            initial,ids,dist=ray_hits(np.broadcast_to(cv[:3,3],dirs_world.shape),dirs_world,poses,halves)
            valid=ids>=0; local=np.full_like(initial,np.nan)
            for i,pose in enumerate(poses):
                sel=ids==i; local[sel]=(initial[sel]-pose[:3,3])@pose[:3,:3]
        world=np.full_like(local,np.nan)
        for i,pose in enumerate(poses):
            sel=ids==i; world[sel]=local[sel]@pose[:3,:3].T+pose[:3,3]
        pc=(world-cv[:3,3])@cv[:3,:3]; projected=pc@k.T; projected=projected[:,:2]/projected[:,2:3]
        inside=valid&(pc[:,2]>0)&(projected[:,0]>=0)&(projected[:,0]<256)&(projected[:,1]>=0)&(projected[:,1]<256)
        # Ray through each material point, rather than nearest-pixel depth, gives exact visibility.
        directions=world-cv[:3,3]; hit,hitid,frac=ray_hits(np.broadcast_to(cv[:3,3],world.shape),directions,poses,halves)
        vis=inside&(hitid==ids)&(np.abs(frac-1)<1e-5)
        xyz.append(world); visible.append(vis); inframe.append(inside); uv_all.append(projected); cams.append(cv); body.append(poses)
        if t==0:
            rr=depth[-1][uv0[:,1].astype(int),uv0[:,0].astype(int)]
            analytic=pc[:,2]; good=valid&(rr>0)
            # SAPIEN raster half-pixel convention can cause finite boundary differences; inspect percentiles.
            depth_delta=np.abs(rr[good]-analytic[good])
    xyz=np.stack(xyz); cams=np.stack(cams); flow=(xyz-xyz[0])@cams[0,:3,:3]
    reproj=np.linalg.norm(np.stack(uv_all)[0,valid]-uv0[valid],axis=-1)
    checks={'initial_reprojection_max_px':float(reproj.max()),'static_background_flow_max_m':float(np.nanmax(np.linalg.norm(flow[:,ids==0],axis=-1))),
            'renderer_depth_median_error_m':float(np.median(depth_delta)),'renderer_depth_p95_error_m':float(np.percentile(depth_delta,95)),
            'valid_queries':int(valid.sum()),'visible_fraction':float(np.stack(visible)[:,valid].mean())}
    if checks['renderer_depth_p95_error_m']>.001 or checks['initial_reprojection_max_px']>.25 or checks['static_background_flow_max_m']>1e-5: raise RuntimeError(checks)
    out.mkdir(parents=True,exist_ok=True); bundle=out/'gt.npz'; tmp=out/'gt.tmp.npz'
    np.savez_compressed(tmp,rgb=np.stack(rgb),depth=np.stack(depth).astype(np.float32),intrinsics=np.broadcast_to(k,(frames,3,3)),camera_to_world=cams,body_poses=np.stack(body),
        query_uv=uv0,query_ids=np.arange(len(uv0)),grid_shape=np.array([grid,grid]),timestamps=np.arange(frames)/20.,
        world_xyz=xyz.astype(np.float32),flow=flow.astype(np.float32),initial_valid=valid,trajectory_valid=np.broadcast_to(valid,(frames,len(valid))),
        visible=np.stack(visible),in_frame=np.stack(inframe),foreground=ids>0,material_ids=ids,local_xyz=local)
    tmp.replace(bundle)
    metadata={'schema_version':1,'simulator':'sapien','sapien_version':sapien.__version__,'archetype':archetype,'camera_mode':mode,'seed':seed,
              'seed_effect':'actor_initial_xy_offsets; deterministic kinematic motion','units':'meters','depth':'axial_camera_z','axes':'right_down_forward',
              'flow':'cumulative_displacement_in_anchor_camera','grid_layout':'row_major_v_then_u','trajectory_source':'exact_actor_local_box_surface',
              'visibility':'exact_first_surface_ray_box_intersection','checks':checks,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),
              'limitations':['plain colored geometry; texture randomization pending','scripted kinematic links rather than native joint articulation']}
    save_json(out/'metadata.json',metadata); (out/'READY').write_text(metadata['sha256']+'\n')
    from PIL import Image
    Image.fromarray(rgb[0]).save(out/'first_frame.png')
    print(json.dumps({'clip':str(out),'checks':checks}),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--grid',type=int,default=16); p.add_argument('--clips',type=int,default=2); p.add_argument('--seed-offset',type=int,default=100)
    args=p.parse_args(); archetypes=['static_scene','object_translation','rigid_rotation','articulated_links','occlusion_and_return','depth_speed_out_of_view_stress']
    root=Path(args.output); jobs=[(a,m) for a in archetypes for m in ['fixed','orbit']][:args.clips]
    for i,(a,m) in enumerate(jobs): generate_clip(root/f'{a}_{m}',a,m,args.seed_offset+i//2,args.grid)

if __name__=='__main__': main()
