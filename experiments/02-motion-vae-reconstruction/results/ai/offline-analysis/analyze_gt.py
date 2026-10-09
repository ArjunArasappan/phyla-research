"""Offline paired comparison of actual saved GT-only codec CSVs."""
from pathlib import Path
import csv,json,hashlib,os
os.environ['MPLCONFIGDIR']=str(Path('work/exp02/offline/mpl').resolve())
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai');OUT=Path('outputs/experiment-02-offline-analysis');OUT.mkdir(parents=True,exist_ok=True)
SOURCES={'debug_g16':BASE/'pilot-v2/per-clip.csv','textured_g64':BASE/'textured64-bounded/per-clip.csv'}
METRICS={'overall':'decoded_gt_epe_m','foreground':'foreground_decoded_gt_epe_m'}
LABELS={'articulated_links':'Scripted links','depth_speed_out_of_view_stress':'Depth/speed stress','object_translation':'Translation','occlusion_and_return':'Occlusion/return','rigid_rotation':'Rotation','static_scene':'Static scene'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(rows):
 w=np.array([r['wan_mm'] for r in rows]);l=np.array([r['ltx_mm'] for r in rows]);delta=l-w
 return dict(n_clips=len(rows),wan_mean_mm=float(w.mean()),ltx_mean_mm=float(l.mean()),paired_delta_mean_mm=float(delta.mean()),paired_delta_median_mm=float(np.median(delta)),paired_delta_min_mm=float(delta.min()),paired_delta_max_mm=float(delta.max()),wan_wins=int((delta>1e-9).sum()),ltx_wins=int((delta < -1e-9).sum()),ties=int((np.abs(delta)<=1e-9).sum()),wan_relative_mean_error_reduction_percent=float(100*(l.mean()-w.mean())/l.mean()))
paired=[];summaries={}
for condition,path in SOURCES.items():
 table=list(csv.DictReader(path.open()));gt=[r for r in table if r['source']=='gt'];clips=sorted({r['clip'] for r in gt});assert len(gt)==24 and len(clips)==12
 for clip in clips:
  camera=next(c for c in ['fixed','orbit'] if clip.endswith('_'+c));archetype=clip[:-(len(camera)+1)]
  rows={c:[r for r in gt if r['clip']==clip and r['codec']==c] for c in ['wan','ltx']};assert all(len(v)==1 for v in rows.values());w,l=rows['wan'][0],rows['ltx'][0]
  assert float(w['finite_coverage'])==float(l['finite_coverage'])==1.0
  for metric,key in METRICS.items():
   wm=float(w[key])*1000;lm=float(l[key])*1000;assert np.isfinite([wm,lm]).all()
   paired.append(dict(condition=condition,clip=clip,archetype=archetype,camera=camera,metric=metric,wan_mm=wm,ltx_mm=lm,delta_ltx_minus_wan_mm=lm-wm,winner='Wan' if lm-wm>1e-9 else 'LTX' if lm-wm < -1e-9 else 'tie'))
 assert len({r['archetype'] for r in paired if r['condition']==condition})==6
 summaries[condition]={}
 for metric in METRICS:
  rows=[r for r in paired if r['condition']==condition and r['metric']==metric]
  summaries[condition][metric]=dict(all_clips=stats(rows),camera={c:stats([r for r in rows if r['camera']==c]) for c in ['fixed','orbit']},archetype={a:stats([r for r in rows if r['archetype']==a]) for a in sorted(LABELS)})
with (OUT/'paired-gt-errors.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
camera_equalities={}
for condition in SOURCES:
 for metric in METRICS:
  rows=[r for r in paired if r['condition']==condition and r['metric']==metric]
  by={(r['archetype'],r['camera']):r for r in rows}
  camera_equalities[condition+'/'+metric]=all(by[(a,'fixed')]['wan_mm']==by[(a,'orbit')]['wan_mm'] and by[(a,'fixed')]['ltx_mm']==by[(a,'orbit')]['ltx_mm'] for a in LABELS)
assert all(camera_equalities.values())
evidence=json.loads((OUT/'latent-shape-evidence.json').read_text())
assert evidence['wan']['latent_shape']==[48,5,16,16] and evidence['ltx']['latent_shape']==[128,3,8,8]
summary=dict(camera_sibling_errors_exactly_identical=camera_equalities,source_csvs={k:dict(path=str(v),sha256=digest(v)) for k,v in SOURCES.items()},metric_units='millimeters',delta_definition='LTX minus Wan; positive favors Wan',experimental_unit='six scene archetype families, fixed/orbit siblings, one seed each',metrics=summaries,latent_capacity=dict(wan_shape=[48,5,16,16],ltx_shape=[128,3,8,8],wan_elements=48*5*16*16,ltx_elements=128*3*8*8,wan_to_ltx_ratio=2.5,saved_tensor_payload_bytes={c:evidence[c]['latent_bytes'] for c in evidence},saved_payload_byte_ratio=5.0,shape_evidence_file='latent-shape-evidence.json',interpretation='Native deployed codec comparison, not matched-rate comparison'))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Paired dumbbells: same clip and GT masks for both endpoints, shared axis scales.
fig,axes=plt.subplots(2,2,figsize=(12.5,9),sharey=True)
clips=sorted({r['clip'] for r in paired});ys=np.arange(len(clips));tick=[]
for clip in clips:
 camera='fixed' if clip.endswith('_fixed') else 'orbit';a=clip[:-(len(camera)+1)];tick.append(f"{LABELS[a]} / {'fixed' if camera=='fixed' else 'orbit'}")
for i,condition in enumerate(SOURCES):
 for j,metric in enumerate(METRICS):
  ax=axes[i,j];rows=[r for r in paired if r['condition']==condition and r['metric']==metric];by={r['clip']:r for r in rows};w=np.array([by[c]['wan_mm'] for c in clips]);l=np.array([by[c]['ltx_mm'] for c in clips])
  ax.hlines(ys,w,l,color='#a5aab2',linewidth=1.6,zorder=1);ax.scatter(w,ys,c='#2465b4',s=28,label='Wan',zorder=2);ax.scatter(l,ys,c='#d46d16',marker='s',s=26,label='LTX',zorder=3)
  all_values=[r['wan_mm'] for r in paired if r['metric']==metric]+[r['ltx_mm'] for r in paired if r['metric']==metric];ax.set_xlim(0,max(all_values)*1.1)
  ax.set_yticks(ys);ax.set_yticklabels(tick,fontsize=8);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Decoded motion vs GT EPE (mm)');s=stats(rows)
  ax.set_title(f"{'Debug G16' if i==0 else 'Textured G64'} · {'whole valid grid' if j==0 else 'foreground'}\nWan wins {s['wan_wins']}/12; mean LTX−Wan {s['paired_delta_mean_mm']:.2f} mm",fontsize=10)
axes[0,0].invert_yaxis()
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.948),ncol=2,frameon=False)
fig.suptitle('GT-only paired reconstruction errors at native codec rates',fontsize=14,y=.985)
fig.text(.5,.012,'Wan: 61,440 latent elements; LTX: 24,576 (2.5× capacity difference). Identical camera-sibling GT scores; six archetypes, one seed each.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.965]);fig.savefig(OUT/'paired-gt-codec-errors.png',dpi=180);fig.savefig(OUT/'paired-gt-codec-errors.svg');plt.close(fig)
# Markdown, including camera strata and six archetype-pair means.
lines=['# Experiment 2: offline paired GT codec comparison','', 'This analysis uses the saved per-clip CSVs only. It includes **GT-source flow**, with twelve clips in each condition: six archetypes, each observed by fixed and orbit cameras. Both codec outputs on one clip use identical inputs, calibration bounds and GT-valid/foreground query masks. The query frame is excluded from motion averages. No tracker rows, GPU jobs or new model inference are used.', '', '“Decoded-to-GT” includes range clipping, the raster roundtrip and frozen VAE distortion. The foreground column is an average over initially foreground queries, followed by a clip-macro average; it is not an image-area score. Positive paired difference below means **LTX error minus Wan error**, favoring Wan.', '', '![Actual paired per-clip GT errors](paired-gt-codec-errors.png)', '', '## Descriptive twelve-clip effects','', '| Condition | Metric | Wan mean (mm) | LTX mean (mm) | Mean paired difference (mm) | Median difference (mm) | Empirical difference range (mm) | Wan / LTX wins |', '|---|---|---:|---:|---:|---:|---:|---:|']
for c in SOURCES:
 for m in METRICS:
  s=summaries[c][m]['all_clips'];lines.append(f"| {c} | {m} | {s['wan_mean_mm']:.3f} | {s['ltx_mean_mm']:.3f} | {s['paired_delta_mean_mm']:+.3f} | {s['paired_delta_median_mm']:+.3f} | [{s['paired_delta_min_mm']:+.3f}, {s['paired_delta_max_mm']:+.3f}] | {s['wan_wins']} / {s['ltx_wins']} |")
lines += ['', 'The ranges are observed per-clip extrema, not confidence intervals. Wins count paired clips, not independent seeds. Fixed/orbit siblings have **exactly identical saved GT codec error values within every archetype**, for both metrics and conditions. The twelve clip-level wins therefore repeat six archetype results; Wan favors all six family means in each condition/metric. Canonical anchor-camera GT displacements remove camera motion, so identical scores are consistent with the representation design. These strata do not establish codec robustness to camera motion. Treating twelve clips or millions of queries as independent replications would exaggerate evidence.', '', '## Camera strata','', '| Condition | Metric | Camera | Clips | Wan mean (mm) | LTX mean (mm) | Mean paired difference (mm) | Wan / LTX wins |', '|---|---|---|---:|---:|---:|---:|---:|']
for c in SOURCES:
 for m in METRICS:
  for camera,s in summaries[c][m]['camera'].items():lines.append(f"| {c} | {m} | {camera} | {s['n_clips']} | {s['wan_mean_mm']:.3f} | {s['ltx_mean_mm']:.3f} | {s['paired_delta_mean_mm']:+.3f} | {s['wan_wins']} / {s['ltx_wins']} |")
lines += ['', '## Six archetype pairs','', 'Each number is a mean of its fixed/orbit sibling pair. This keeps camera siblings together as a scene-family descriptive summary. It supplies no between-seed population estimate.', '', '| Condition | Archetype | Whole-grid Wan / LTX (mm) | Whole-grid difference (mm) | Foreground Wan / LTX (mm) | Foreground difference (mm) |', '|---|---|---:|---:|---:|---:|']
for c in SOURCES:
 for a in sorted(LABELS):
  s=summaries[c]['overall']['archetype'][a];f=summaries[c]['foreground']['archetype'][a];lines.append(f"| {c} | {LABELS[a]} | {s['wan_mean_mm']:.3f} / {s['ltx_mean_mm']:.3f} | {s['paired_delta_mean_mm']:+.3f} | {f['wan_mean_mm']:.3f} / {f['ltx_mean_mm']:.3f} | {f['paired_delta_mean_mm']:+.3f} |")
lines += ['', '## Interpretation and latent-rate limitation','', 'These native codecs are **not rate matched**. At 17×256×256 input, the saved runtime contract is Wan [48,5,16,16] = **61,440 latent elements**, versus LTX [128,3,8,8] = **24,576 elements**. Wan carries **2.5× as many latent scalar elements**. The shapes were verified by reading two small saved GT metrics.json members directly from the local archive, without unpacking scientific arrays. Saved tensor payloads are 245,760 bytes for Wan and 49,152 bytes for LTX: **5×**, because the adapter saves Wan latents as float32 and LTX as BF16. Those are uncompressed tensor payloads, not entropy-coded file rates or proof of extra effective numerical precision. Element count and storage dtype both confound a capacity comparison. A lower error at Wan’s deployed rate does not establish architectural superiority at equal capacity. A matched-rate comparison would require an explicit rate–distortion protocol rather than simply changing these pretrained latent shapes.', '', 'Texture, query density and sampled surfaces change between debug G16 and textured G64. The normalization bounds remain frozen, but these are distinct conditions; differences across them do not isolate texture or density. The paired comparison within a condition is the reliable descriptive unit. Three tracker followup rows elsewhere use only two clips and are deliberately excluded here.', '', 'Natural-image VAEs can reconstruct the synthetic motion RGB while producing physical error that varies by channel ranges, object boundaries and static versus moving surfaces. GT-source results establish a representation/codec floor; they do not validate tracker accuracy, action learning, OOD robustness or policy success.', '', '## Reproducibility','', '- [Paired observations](paired-gt-errors.csv): all 48 condition/clip/metric pairs, with both codec values and empirical winner.', '- [Machine-readable summary](summary.json): exact statistics, camera/archetype strata and input CSV SHA256s.', '- [Figure SVG](paired-gt-codec-errors.svg): vector export of the actual-data plot.', '- [Actual latent-shape evidence](latent-shape-evidence.json): saved GT metric members, shapes and tensor payload bytes.', '', 'The source CSVs and latent-shape evidence are local saved artifacts. Only two small JSON members were streamed from the existing archive to verify shapes; no large scientific arrays or archive directory were extracted. The offline CSV analysis runs independently of GPU inference.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({c:{m:summaries[c][m]['all_clips'] for m in METRICS} for c in SOURCES},indent=2))
