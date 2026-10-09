import pathlib,json,csv,numpy as np
base=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu');primary=base/'runs/exp02/simulator-pilot-v2';long=base/'runs/exp02/simulator-context33-v1';out=base/'worktrees/exp02/experiments/02-motion-vae-reconstruction/results/ai/context33';out.mkdir(parents=True,exist_ok=True);rows=[]
for p in sorted(primary.glob('*/gt/*/reconstruction.npz')):
 clip,_,codec=p.relative_to(primary).parts[:3];q=long/clip/'gt'/codec/'reconstruction.npz'
 if not (q.parent/'complete.json').exists():continue
 with np.load(p) as short,np.load(q) as longer:
  mask=short['valid']&longer['valid'][:17];mask[0]=False;gt=short['gt_flow'];dshort=short['decoded_flow'];dlong=longer['decoded_flow'][:17]
  rows.append(dict(clip=clip,codec=codec,short17_epe_m=float(np.linalg.norm(dshort-gt,axis=-1)[mask].mean()),long33_first17_epe_m=float(np.linalg.norm(dlong-gt,axis=-1)[mask].mean()),shared_first17_reconstruction_difference_m=float(np.linalg.norm(dlong-dshort,axis=-1)[mask].mean())))
with (out/'paired-first17.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Temporal context control: 17 vs33 input frames','','Both codecs receive identical first17 motionRGB frames and the same calibration bounds; the33-frame run additionally includes16 later frames. The table scores only the shared first17 window, excluding the query frame. This isolates the effect of available codec context, unlike comparing full17 vsfull33 averages with different displacement distributions. No temporal realignment or refit of normalization was performed.','','| Codec | Paired clips | 17-frame EPE(mm) | 33-frame first17 EPE(mm) | Decoded output difference(mm) |','|---|---:|---:|---:|---:|']
for codec in ['wan','ltx']:
 group=[r for r in rows if r['codec']==codec];means=[np.mean([r[k] for r in group])*1000 for k in ['short17_epe_m','long33_first17_epe_m','shared_first17_reconstruction_difference_m']];lines.append(f'| {codec} | {len(group)} | '+' | '.join(f'{v:.4f}' for v in means)+' |')
lines += ['', 'All values are clip-macro means, not independent point-level replications. These two native codecs have different causal/temporal designs and compression rates; temporal-context differences do not isolate one architectural component by themselves. Full33 raw/decoded RGB, latents, flows and metrics are retained in runs/exp02/simulator-context33-v1/.']
(out/'README.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
