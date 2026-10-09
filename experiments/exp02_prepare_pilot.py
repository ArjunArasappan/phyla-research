import sys,pathlib,json,numpy as np
base=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu');sys.path.insert(0,str(base/'worktrees/exp02/experiments'));import exp02
paths=sorted((base/'data/exp01/calibration').glob('*/gt.npz'))
assert len(paths)==12 and all((p.parent/'READY').exists() for p in paths),[(str(p),list(p.parent.glob('*'))) for p in paths]
exp02.calibrate(paths,base/'data/exp02/calibration-stats-v1.json')
jobs=[]
for p in sorted((base/'data/exp01/pilot').glob('*/gt.npz')):
 jobs.append(dict(id=p.parent.name+'/gt',source=str(p),gt=str(p),source_kind='simulator_gt'))
 d,g,v,f=exp02.load_flow(p);zero=base/'data/exp02/zero-controls'/p.parent.name/'tracks.npz';zero.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(zero,flow=np.zeros_like(d),grid_size=g,initial_valid=v,foreground=f)
 jobs.append(dict(id=p.parent.name+'/zero',source=str(zero),gt=str(p),source_kind='zero_motion_control'))
exp02.save_json(base/'data/exp02/gt-zero-manifest-v1.json',jobs)
print('GT jobs',len(jobs),'calibration',len(paths))
