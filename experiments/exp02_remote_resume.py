"""Root transports this driver after child network sandbox reset.

Only36 bounded textured followup jobs; no re-download or duplicate GPU work.
"""
from pathlib import Path
import argparse,subprocess,json,time,hashlib,os
B=Path('/mnt/nvme/scratch/phyla-ubuntu');R=B/'worktrees/exp02';PY=B/'environments/core/bin/python';E=R/'experiments/02-motion-vae-reconstruction';P=R/'experiments/exp02.py';REPORT=R/'experiments/exp02_report.py';STATS=B/'data/exp02/calibration-stats-v2-balanced.json';OUT=B/'runs/exp02/textured64-bounded-v1';PRACTICAL=B/'runs/exp02/practical-shared-front-v1'
def run(cmd):subprocess.run(list(map(str,cmd)),check=True)
def atomic(p,d):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial.json');q.write_text(json.dumps(d,indent=2));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def active(name):return subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def archive(name,paths):
 p=B/'backups'/name;p.parent.mkdir(exist_ok=True)
 if Path(str(p)+'.sha256').exists():return
 session='phyla-exp02-archive-'+name.split('.')[0]
 if active(session):return
 cmd=f'tar --exclude=__pycache__ --exclude="*.pyc" -C {B} -czf {p}.partial '+' '.join(str(x) for x in paths)+f' && mv {p}.partial {p} && sha256sum {p} > {p}.sha256'
 run(['tmux','new-session','-d','-s',session,cmd])
def manifest(root,out,expected):
 import numpy as np
 records=[]
 for complete in sorted(root.glob('*/*/*/complete.json')):
  m=json.loads((complete.parent/'metrics.json').read_text());assert m['identity_max_error_m']<=1e-6
  with np.load(complete.parent/'reconstruction.npz') as d:assert np.isfinite(d['decoded_rgb_raw']).all()
  records.append(dict(job=str(complete.parent.relative_to(root)),input_signature=m['input_signature'],codec_signature=m['codec_signature'],files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in complete.parent.iterdir() if p.is_file()}))
 assert len(records)==expected,(root,len(records),expected)
 atomic(out,dict(root=str(root),jobs=expected,file_bytes=sum(f['bytes'] for r in records for f in r['files'].values()),records=records))
def report(root,out,regime):run([PY,REPORT,'--run-root',root,'--stats',STATS,'--output',out,'--regime',regime])
def launch():
 report(PRACTICAL,E/'results/ai/practical-shared-front','shared_front_initial_GT_scale')
 manifest(PRACTICAL,E/'results/ai/practical-shared-front/artifact-manifest.json',48)
 archive('exp02-practical-increment.tar.gz',['runs/exp02/practical-shared-front-v1','data/exp02/shared-front-manifest-v1.json','data/exp02/calibration-stats-v2-balanced.json','worktrees/exp02/experiments/02-motion-vae-reconstruction/results/ai/practical-shared-front'])
 jobs=[]
 for p in sorted((B/'data/exp01/textured64/pilot').glob('*/gt.npz')):
  assert (p.parent/'READY').exists();jobs.append(dict(id=p.parent.name+'/gt',source=str(p),gt=str(p),source_kind='simulator_gt'))
 for method in ['cotracker3_gt_geometry','spatrackerv2_gt_geometry_fp32_math','delta_gt_geometry']:
  for clip in ['object_translation_orbit','occlusion_and_return_orbit']:
   p=B/'data/exp01/textured64/predictions'/method/clip/'tracks.npz';assert (p.parent/'READY').exists()
   jobs.append(dict(id=clip+'/'+method,source=str(p),gt=str(B/'data/exp01/textured64/pilot'/clip/'gt.npz'),source_kind='tracker_prediction'))
 assert len(jobs)==18,len(jobs);jpath=B/'data/exp02/textured64-bounded-manifest-v1.json';atomic(jpath,jobs)
 atomic(B/'data/exp02/textured64-bounded-design.json',dict(gt_clips=12,key_tracker_clips=['object_translation_orbit','occlusion_and_return_orbit'],grid_size=64,bounds_policy='frozen_original_debug_calibration',statistics_path=str(STATS),statistics_sha256=sha(STATS),note='Density/texture/point sampling changes are not isolated by this combined followup; no hidden condition normalization. Broad tracker averaging uses only2clips andisnot comparableto12clipmacro.'))
 for codec,gpu in [('wan',0),('ltx',1)]:
  session='phyla-exp02-'+codec+'-textured';done=sum((OUT/j['id']/codec/'complete.json').exists() for j in jobs)
  if done==18 or active(session):continue
  cmd=f'CUDA_VISIBLE_DEVICES={gpu} {PY} {P} batch --codec {codec} --manifest {jpath} --stats {STATS} --output {OUT} > {B}/runs/exp02/{codec}-textured-bounded.log 2>&1'
  run(['tmux','new-session','-d','-s',session,cmd])
 atomic(B/'control/exp02/status.json',dict(experiment='exp02',stage='bounded_textured_followup',experiment_state='running',completed_primary=168,completed_context=24,completed_practical=48,gpus=[0,1],run_root=str(OUT),updated_unix=time.time()))
 watcher='phyla-exp02-textured-finalize'
 if not active(watcher):run(['tmux','new-session','-d','-s',watcher,f'{PY} {Path(__file__)} --finalize > {B}/runs/exp02/textured-finalize.log 2>&1'])
def finalize():
 # Bounded lease: only wait on our own sessions, never inspect/kill other jobs.
 while active('phyla-exp02-wan-textured') or active('phyla-exp02-ltx-textured'):time.sleep(5)
 report(OUT,E/'results/ai/textured64-bounded','known_geometry')
 manifest(OUT,E/'results/ai/textured64-bounded/artifact-manifest.json',36)
 notes='''# Bounded textured64 followup

Thirty-six real frozen VAE jobs completed: GT on all12 textured64 clips, plus CoTracker3, DELTA and corrected SpaTrackerV2 FP32-math on two key orbit clips (translation and occlusion). The source query grid is64×64. All methods use the ORIGINAL frozen calibration bounds from the debug study; no condition-specific normalization is silently fit. These outputs have their own run namespace and scientific manifest.

Texture, query density and sampled surfaces change together in this followup. GT video-codec comparison can expose sensitivity to the grid/raster representation, but tracker differences cannot be attributed uniquely to texture or density without the paired same-grid controls. Two-clip tracker averages are not the same estimand as the12-clip GT average. Main debug results remain authoritative for their declared setting; no existing arrays are overwritten.

Both VAEs remain frozen. GPUs0/1 were used under a short lease and released after completion. Every decoder is finite and vector decomposition passed at≤1e-6m. Raw/clamped RGB, latents, recovered flows and exact per-file SHA256/byte ledgers are retained. Practical shared-front labels use a separate table with a fixed initial GT-scale diagnostic; they are not merged into supplied-GT-geometry results.
'''
 (E/'ai_notes/textured64-followup-2026-10-08.md').write_text(notes)
 (E/'STATUS.md').write_text('# Experiment 2 status\n\n**GPU pilot and bounded followups complete.**\n\n- 168 debug supplied-geometry source/codec jobs.\n- 24 GT temporal-context jobs.\n- 48 practical shared-front / fixed initial GT-scale diagnostic jobs, separately reported.\n- 36 bounded textured64 jobs (12 GT clips; two key tracker clips).\n\nAll 276 reconstructions retain scientific arrays/latents and per-file checksum ledgers. GPUs 0/1 and 2/3 released. Broader equal-density/appearance comparisons, noise/raster controls and native-metric monocular claims remain pending.\n\n[Debug report](results/ai/pilot-v2/README.md) · [Context control](results/ai/context33/README.md) · [Practical diagnostic](results/ai/practical-shared-front/README.md) · [Textured followup](results/ai/textured64-bounded/README.md) · [Measured animations](results/ai/animations/README.md)\n')
 archive('exp02-textured-increment.tar.gz',['runs/exp02/textured64-bounded-v1','data/exp02/textured64-bounded-manifest-v1.json','data/exp02/textured64-bounded-design.json','data/exp02/calibration-stats-v2-balanced.json','worktrees/exp02/experiments/02-motion-vae-reconstruction/results/ai/textured64-bounded'])
 atomic(B/'control/exp02/status.json',dict(experiment='exp02',stage='gpu_pilot_followups_complete',experiment_state='complete_bounded_pilot_broader_study_pending',completed_debug=168,completed_context=24,completed_practical=48,completed_textured=36,gpus=[],updated_unix=time.time(),reports=[str(E/'results/ai/practical-shared-front/README.md'),str(E/'results/ai/textured64-bounded/README.md')]))
 run([PY,'-m','pytest',R/'experiments/test_exp02.py','-q'])
 run(['git','-C',R,'add','experiments/exp02_report.py','experiments/exp02_remote_resume.py','experiments/02-motion-vae-reconstruction'])
 run(['git','-C',R,'commit','-m','Add practical shared-front and bounded textured64 VAE diagnostics'])
 print('FINALIZED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
 finalize() if args.finalize else launch()
