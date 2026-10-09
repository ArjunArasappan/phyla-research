"""Cached RGB-only SpaTracker geometry frontend; no simulator geometry is input."""
from pathlib import Path
import argparse,sys,json,time,hashlib
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--repo',required=True);p.add_argument('--clips',type=int,default=12);a=p.parse_args();sys.path.insert(0,a.repo)
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    model=VGGT4Track.from_pretrained('Yuxihenry/SpatialTrackerV2_Front').cuda().eval()
    for file in sorted(Path(a.data).glob('*/gt.npz'))[:a.clips]:
        out=Path(a.output)/file.parent.name
        if (out/'READY').exists():continue
        d=np.load(file);video=torch.from_numpy(d['rgb'].copy()).permute(0,3,1,2).float().cuda()[None]/255
        start=time.time();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(video)
        raw={k:v.detach().cpu().float().numpy() for k,v in pred.items() if isinstance(v,torch.Tensor)}
        depth=raw['points_map'][...,2];intr=raw['intrs'][0];poses=raw['poses_pred'][0];confidence=raw['unc_metric']
        if depth.shape!=d['depth'].shape or intr.shape!=d['intrinsics'].shape or poses.shape!=d['camera_to_world'].shape:raise RuntimeError('Frontend shape contract failed')
        if not np.isfinite(depth).all() or not np.isfinite(poses).all() or not np.isfinite(intr).all():raise RuntimeError('Frontend produced nonfinite geometry; inspect raw outputs')
        out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'raw.npz',**raw);np.savez_compressed(out/'geometry.npz',depth=depth,intrinsics=intr,camera_to_world=poses,confidence=confidence)
        metadata={'source':'RGB_only','used_gt_geometry':False,'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'model':'Yuxihenry/SpatialTrackerV2_Front','units':'native_unknown_scale','elapsed_s':time.time()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'preprocessing':'native model internal518 resize; native output returns256 coordinate image domain'}
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'READY').write_text('finite_shape_verified_no_GT_inputs\n');print(json.dumps({'clip':file.parent.name,**metadata}),flush=True)

if __name__=='__main__':main()
