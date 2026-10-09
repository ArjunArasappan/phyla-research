"""Aggregate paired codec artifacts without inventing missing measurements."""
import argparse,json,csv,pathlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--output',required=True);p.add_argument('--stats',required=True);p.add_argument('--regime',choices=['known_geometry','shared_front_initial_GT_scale'],default='known_geometry');a=p.parse_args();root=pathlib.Path(a.run_root);out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
stats=json.loads(pathlib.Path(a.stats).read_text());lo=np.array(stats['lower_m']);hi=np.array(stats['upper_m']);zero=-lo/(hi-lo)
rows=[]
for m in sorted(root.glob('*/*/*/metrics.json')):
 if not (m.parent/'complete.json').exists():continue
 d=json.loads(m.read_text());clip,source,codec=m.relative_to(root).parts[:3]
 rows.append(dict(clip=clip,source=source,codec=codec,tracking_epe_m=d['tracking']['epe_m'],range_epe_m=d['range']['epe_m'],raster_epe_m=d['raster']['epe_m'],codec_self_epe_m=d['codec_self']['epe_m'],decoded_gt_epe_m=d['decoded_gt']['epe_m'],foreground_decoded_gt_epe_m=d['decoded_gt_foreground'].get('epe_m',float('nan')),background_decoded_gt_epe_m=d['decoded_gt_background'].get('epe_m',float('nan')),finite_coverage=d['finite_coverage'],inference_seconds=d['inference_seconds'],peak_gpu_gb=d['peak_gpu_memory_bytes']/1e9,decoder_overshoot_fraction=d['decoder_overshoot_fraction']))
if not rows:raise RuntimeError('No completed jobs')
with (out/'per-clip.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
frames=json.loads(next(root.glob('*/*/*/metrics.json')).read_text())['input_signature']['frames']
with np.load(next(root.glob('*/*/*/reconstruction.npz'))) as first:grid_size=int(first['grid_size'])
regime_note='Supplied GT depth/cameras are privileged geometry inputs.' if a.regime=='known_geometry' else 'Practical predicted geometry from a shared RGB-only frontend, followed by one fixed median initial GT-depth scale scalar. This is an initial-GT-scale diagnostic, not uncalibrated metric-monocular performance. GT enters scale evaluation only; depth/camera inference uses RGB. This table is separate from supplied-GT-geometry results.'
lines=['# Experiment 2 GPU pilot results','', f'These are measured frozen VAE reconstructions of {frames}-frame motion windows from exact SAPIEN scripted-box/link trajectories. One scene seed per archetype and paired fixed/orbit cameras; this debug-density pilot does not establish a statistically general model ranking. Both VAEs are frozen; no T5, DiT, training, or diffusion generation is loaded.', '', f'All methods/codecs share calibration-only bounds (12 separate scene clips), a {grid_size}×{grid_size} initial query grid upsampled bilinearly to 256×256, and deterministic posterior mode. Query frame is excluded from main physical averages. Errors below are **macro means across clips**; tracker nonfinite coordinates are excluded from conditional EPE and their coverage is reported separately. {regime_note}', '', '| Source | Codec | Clips | Tracking (mm) | Range (mm) | Raster (mm) | Codec self (mm) | Decoded to GT (mm) | Foreground to GT (mm) | Background to GT (mm) | Coverage |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for source in sorted({r['source'] for r in rows}):
 for codec in ['wan','ltx']:
  group=[r for r in rows if r['source']==source and r['codec']==codec]
  if not group:continue
  names=['tracking_epe_m','range_epe_m','raster_epe_m','codec_self_epe_m','decoded_gt_epe_m','foreground_decoded_gt_epe_m','background_decoded_gt_epe_m'];v=[np.nanmean([r[n] for r in group])*1000 for n in names];coverage=np.mean([r['finite_coverage'] for r in group])
  lines.append(f'| {source} | {codec} | {len(group)} | '+' | '.join(f'{x:.3f}' for x in v)+f' | {coverage:.3%} |')
lines += ['', 'Scalar EPE terms are not additive. The decoded-to-GT quantities include tracking, shared range clipping, rasterization and codec loss; codec-self is measured relative to the no-VAE inverse-rendered flow. Foreground/background values are clip-macro means on initial query group masks. Raw decoder overshoot, RGB error, axis errors, native signatures, and exact vector-error identity checks are in each metrics.json. Complete inputs, raw/clamped RGB, reconstructed flows, masks and latents are retained. Preview figures use shared physical error limits.', '', '## Frozen mapping and visual interpretation','',f'Axis lower bounds (m): {lo.tolist()}; upper bounds (m): {hi.tolist()}. Zero-displacement RGB: {zero.tolist()}. Calibration sampling excludes the anchor frame and uses all32 non-anchor frames, whereas the primary codec window uses16 non-anchor frames. High-speed calibration motion broadens the shared range, so ordinary motion can appear nearly uniform in the RGB previews. Physical error maps use meter units and shared Wan/LTX color limits within each source row. No per-source contrast normalization is applied.', '', '## Side-by-side diagnostics','']
for clip in sorted({r['clip'] for r in rows}):
 sources=sorted({r['source'] for r in rows if r['clip']==clip});fig,axes=plt.subplots(len(sources),5,figsize=(17,max(3,len(sources)*3)),squeeze=False)
 for i,source in enumerate(sources):
  bundles={c:root/clip/source/c/'reconstruction.npz' for c in ['wan','ltx']}
  if not all(p.exists() for p in bundles.values()):continue
  with np.load(bundles['wan']) as w,np.load(bundles['ltx']) as l:
   frame=-1;g=int(w['grid_size']);gt=w['gt_flow'][frame];valid=w['valid'][frame]
   ews=np.linalg.norm(w['decoded_flow'][frame]-gt,axis=-1);els=np.linalg.norm(l['decoded_flow'][frame]-gt,axis=-1);vmax=max(float(np.max(ews[valid])),float(np.max(els[valid])),1e-4)
   axes[i,0].imshow(w['rgb_input'][frame].transpose(1,2,0));axes[i,0].set_title(f'{source}: input motionRGB')
   axes[i,1].imshow(w['decoded_rgb_clamped'][frame].transpose(1,2,0));axes[i,1].set_title('Wan reconstructed RGB')
   axes[i,2].imshow(l['decoded_rgb_clamped'][frame].transpose(1,2,0));axes[i,2].set_title('LTX reconstructed RGB')
   for ax,e,name in [(axes[i,3],ews,'Wan vs GT'),(axes[i,4],els,'LTX vs GT')]:
    error=np.where(valid,e,np.nan).reshape(g,g);im=ax.imshow(error,vmin=0,vmax=vmax,cmap='magma');fig.colorbar(im,ax=ax,label='m');ax.set_title(name)
   for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
 fig.tight_layout();fig.savefig(out/(clip+'.png'),dpi=120);plt.close(fig);lines.append(f'![{clip}]({clip}.png)')
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('aggregated',len(rows),'completed jobs',out)
