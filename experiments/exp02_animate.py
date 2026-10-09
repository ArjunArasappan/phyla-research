"""Measured RGB and 3D motion animations; no synthetic trajectories drawn."""
import argparse,pathlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--output',required=True);p.add_argument('--clip',nargs='+',default=['object_translation_orbit','occlusion_and_return_orbit','depth_speed_out_of_view_stress_orbit']);a=p.parse_args();root=pathlib.Path(a.run_root);out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
except Exception:font=ImageFont.load_default()
labels={'gt':'GT','cotracker3_gt_geometry':'CoTracker3','delta_gt_geometry':'DELTA','spatrackerv2_gt_geometry_fp32_math':'SpaV2 FP32'}
for clip in a.clip:
 sources=[s for s in ['gt','cotracker3_gt_geometry','delta_gt_geometry','spatrackerv2_gt_geometry_fp32_math'] if (root/clip/s/'wan/complete.json').exists() and (root/clip/s/'ltx/complete.json').exists()]
 arrays={s:{c:dict(np.load(root/clip/s/c/'reconstruction.npz')) for c in ['wan','ltx']} for s in sources}
 if not sources:continue
 frames=len(arrays[sources[0]]['wan']['source_flow']);images=[]
 for t in range(frames):
  canvas=Image.new('RGB',(768,len(sources)*292+28),'white');draw=ImageDraw.Draw(canvas);draw.text((5,5),f'{clip} — frame {t}; shared motion RGB mapping',fill='black',font=font)
  for i,s in enumerate(sources):
   for j,label in enumerate(['source','wan','ltx']):
    array=arrays[s]['wan']['rgb_input'][t] if label=='source' else arrays[s][label]['decoded_rgb_clamped'][t]
    tile=Image.fromarray(np.round(np.clip(array.transpose(1,2,0),0,1)*255).astype(np.uint8));canvas.paste(tile,(j*256,i*292+64));draw.text((j*256+4,i*292+31),f'{labels[s]}: {label}',fill='black',font=font)
  images.append(canvas)
 images[0].save(out/(clip+'-motion-rgb.gif'),save_all=True,append_images=images[1:],duration=100,loop=0)
 # Recover trajectories in the initial camera frame using the common GT
 # initial position as an explicitly diagnostic anchor, not model initial XYZ.
 base=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu');gtpath=base/'data/exp01/pilot'/clip/'gt.npz'
 with np.load(gtpath) as gt:
  world=np.array(gt['world_xyz']);camera=np.array(gt['camera_to_world']);p0=(world[0]-camera[0,:3,3])@camera[0,:3,:3]
 fig=plt.figure(figsize=(12,len(sources)*3.7));axes=[];selected={};limits=[]
 for s in sources:
  d=arrays[s]['wan'];ids=np.flatnonzero(d['foreground']&d['valid'][1:].all(0))
  if len(ids)>6:ids=ids[np.linspace(0,len(ids)-1,6).astype(int)]
  selected[s]=ids
  for c in ['wan','ltx']:
   x=arrays[s][c]
   for key in ['gt_flow','source_flow','decoded_flow']:limits.append((p0[ids][None]+x[key][:,ids]).reshape(-1,3))
  for c in ['wan','ltx']:axes.append((s,c,fig.add_subplot(len(sources),2,len(axes)+1,projection='3d')))
 allpoints=np.concatenate(limits);lo=allpoints.min(0);hi=allpoints.max(0);margin=np.maximum((hi-lo)*.1,.015);images=[]
 for t in range(frames):
  for s,c,ax in axes:
   ax.clear();ids=selected[s];d=arrays[s][c]
   for key,color,label in [('gt_flow','#168b34','GT'),('source_flow','#666666','raw source'),('decoded_flow','#c000d0','decoded')]:
    xyz=p0[ids][None]+d[key][:t+1,ids]
    for i in range(len(ids)):ax.plot(xyz[:,i,0],xyz[:,i,1],xyz[:,i,2],c=color,alpha=.75,label=label if i==0 else None)
    ax.scatter(xyz[-1,:,0],xyz[-1,:,1],xyz[-1,:,2],c=color,s=8)
   ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_zlim(lo[2]-margin[2],hi[2]+margin[2]);ax.set_xlabel('x right (m)');ax.set_ylabel('y down (m)');ax.set_zlabel('z forward (m)');ax.set_title(f'{labels[s]} / {c} / frame {t}',fontsize=10);ax.legend(loc='upper left',fontsize=7)
  fig.suptitle('Recovered 3D paths in anchor camera; common GT initial anchor; foreground IDs fixed');fig.tight_layout(rect=[0,0,1,.97]);fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];images.append(Image.fromarray(frame.copy()))
 images[0].save(out/(clip+'-3d-paths.gif'),save_all=True,append_images=images[1:],duration=100,loop=0);plt.close(fig)
 print('animated',clip,flush=True)
(out/'README.md').write_text('# Measured pilot animations\n\nRGB rows compare GT, CoTracker3, DELTA and corrected SpaTrackerV2 FP32-math motion labels with the two frozen codecs under identical mapping. Three-dimensional trails add the common GT query-frame position to raw/decoded displacement; this is a flow-reconstruction diagnostic and does not score the model initial-position error. Foreground query IDs and axes are held fixed over time and across panels. Display timing is 10fps for readability; source sampling is20Hz.\n\n'+'\n'.join(f'![{f.name}]({f.name})' for f in sorted(out.glob('*.gif')))+'\n')
