"""Factual restart-cohort tables and plots, with direct saved-video links."""
from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/workspace/phyla-research-runs/exp03-recovery-root')
OUT=Path('/workspace/phyla-research-runs/results/experiment-03');OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'runs/exp03/pilot/manifest.json').read_text());rows=[];videos=[]
for r in manifest['runs']:
 for c in ('id','cube_color','camera30'):
  p=ROOT/'runs/exp03/pilot'/r['run']/('eval_'+c)
  if not (p/'summary.json').exists():continue
  eps=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()]
  rows.append({'N':r['N'],'arm':r['arm'],'optimizer_seed':r['optimizer_seed'],'condition':c,'successes':sum(x['success_at_end'] for x in eps),'rollouts':len(eps),'artifact':str(p)})
  videos.extend((r['run'],c,str(v)) for v in p.rglob('*.mp4'))
(OUT/'metrics.json').write_text(json.dumps({'cohort':'H100 recovery with CPU renderer','results':rows,'protocol':manifest},indent=2))
if rows:
 with (OUT/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
 for ax,c in zip(axs,('id','cube_color','camera30')):
  for arm,color in [('A','#444444'),('R','#c7671c'),('F-GT','#1473b5')]:
   cells=[r for r in rows if r['condition']==c and r['arm']==arm]
   ns=sorted(set(r['N'] for r in cells))
   ax.plot(ns,[np.mean([r['successes']/r['rollouts'] for r in cells if r['N']==n]) for n in ns],'o-',label=arm,color=color)
   ax.scatter([r['N'] for r in cells],[r['successes']/r['rollouts'] for r in cells],marker='x',color=color)
  ax.set(title=c,xlabel='Demonstrations',ylim=(-.03,1.03));ax.set_xticks([16,64]);ax.grid(alpha=.2)
 axs[0].set_ylabel('Success at final step');axs[-1].legend();fig.tight_layout();fig.savefig(OUT/'success.png',dpi=180);plt.close(fig)
lines=['# Experiment 3: recorded evaluation results','','## Motivation and hypothesis','','Compare action-only, RGB-future-supervised and GT-flow-supervised policies at fixed demonstration count and training budget. Hypothesis: auxiliary supervision changes paired task success and OOD performance.','','## Experimental setup','','12 saved final LoRA policies; N16/N64, A/R/F-GT, two optimizer seeds. PushCube-v1/Panda, native8D control, RGB+instruction, seeds20000–20004, ID/color/camera30, 150 steps, four-action execution, eight Euler sampler steps, shift5. H100 inference, CPU PhysX and CPU Vulkan rendering. No extra training.','','## Measured results','',f'Completed condition groups: {len(rows)}/36. Rollouts: {sum(r["rollouts"] for r in rows)}/180.','','| N | Arm | Optimizer seed | Condition | Successes / rollouts |','|---|---|---|---|---|']
for r in rows:lines.append(f'| {r["N"]} | {r["arm"]} | {r["optimizer_seed"]} | {r["condition"]} | {r["successes"]}/{r["rollouts"]} |')
lines+=['','## Visuals','','![Success measurements](success.png)','','## Recordings','']
lines.extend(f'- [{name} / {c} / {Path(v).name}]({v})' for name,c,v in videos)
lines+=['','## Measurement scope','','One task, one subset seed, two optimizer seeds and five resets per condition. The recovered cohort uses CPU rendering and Torch2.8; the original node used GPU rendering and Torch2.12. Missing cells are omitted. No original partial summaries are merged into this cohort.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'completed_cells':len(rows),'videos':len(videos),'output':str(OUT)}))
