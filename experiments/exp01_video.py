"""Synchronized query-ID consistent 2D track previews; science arrays remain lossless."""
from pathlib import Path
import argparse
import cv2,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    methods=['cotracker3_gt_geometry','spatrackerv2_gt_geometry_fp32_math','delta_gt_geometry']
    for gtfile in sorted((root/'pilot').glob('*/gt.npz')):
        d=np.load(gtfile);valid=d['initial_valid'];ids=np.flatnonzero(valid);pc=np.einsum('tqj,tjk->tqk',d['world_xyz']-d['camera_to_world'][:,None,:3,3],d['camera_to_world'][:,:3,:3]);uvh=np.einsum('tij,tqj->tqi',d['intrinsics'],pc);gtuv=uvh[...,:2]/uvh[...,2:3]
        arrays=[('GT material points',gtuv[:,ids])]
        for method in methods:
            f=root/'predictions'/method/gtfile.parent.name/'raw.npz'
            if not f.exists():continue
            r=np.load(f);uv=r['trajs_uv'] if method.startswith('delta') else r['uv'];arrays.append((method.split('_gt')[0],uv[...,:2]))
        width=256*len(arrays);writer=cv2.VideoWriter(str(out/(gtfile.parent.name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),20,(width,286))
        if not writer.isOpened():raise RuntimeError('Preview encoder unavailable')
        chosen=np.unique(np.r_[np.arange(0,len(ids),5),np.flatnonzero(d['foreground'][ids])])
        for t,image in enumerate(d['rgb']):
            panels=[]
            for name,uv in arrays:
                panel=np.zeros((286,256,3),np.uint8);panel[:256]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                for j in chosen:
                    pt=uv[t,j]
                    if not np.isfinite(pt).all() or not (0<=pt[0]<256 and 0<=pt[1]<256):continue
                    color=cv2.cvtColor(np.uint8([[[int(ids[j]%180),220,255]]]),cv2.COLOR_HSV2BGR)[0,0].tolist()
                    cv2.drawMarker(panel,tuple(np.rint(pt-.5).astype(int)),color,cv2.MARKER_CROSS if not d['visible'][t,ids[j]] else cv2.MARKER_TILTED_CROSS,5,1)
                cv2.putText(panel,name,(4,273),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1);panels.append(panel)
            frame=np.concatenate(panels,1);writer.write(frame)
            if t==16:cv2.imwrite(str(out/(gtfile.parent.name+'.png')),frame)
        writer.release()
    print(out)

if __name__=='__main__':main()
