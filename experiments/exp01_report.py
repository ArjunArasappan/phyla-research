"""Aggregate debug pilot artifacts without implying a confirmatory ranking."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((root/'predictions').glob('*/*/metrics.json')):
        if not (path.parent/'READY').exists(): continue
        d=json.loads(path.read_text());view=d['visible'];allv=d.get('all',d.get('all_diagnostic_not_hidden_point_oracle'))
        rows.append({'method':path.parent.parent.name,'clip':path.parent.name,'camera':'orbit' if path.parent.name.endswith('orbit') else 'fixed',
                     'visible_epe_m':view['epe_m'],'all_epe_m':allv['epe_m'],'visible_coverage':view['coverage'],'visible_pck_1cm':view['pck_1cm']})
    with (out/'per_clip.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=sorted(set(r['method'] for r in rows));groups=[('GT supplied geometry / controls',[m for m in methods if 'shared_front' not in m]),('RGB-only frontend; initial GT scale diagnostic',[m for m in methods if 'shared_front' in m])];groups=[x for x in groups if x[1]]
    labels={'cotracker3_gt_geometry':'CoTracker3\nGT depth/poses','delta_gt_geometry':'DELTA\nGT depth/poses','gt_uv_exact_front_ray':'GT UV\nexact front ray','gt_uv_gt_front_depth':'GT UV\nsampled depth','spatrackerv2_gt_geometry':'SpaV2\nGT / BF16','spatrackerv2_gt_geometry_fp32_math':'SpaV2\nGT / FP32','zero_motion_xyz':'Zero motion','cotracker3_shared_front_initial_GT_scale':'CoTracker3\nshared / GT scale','spatrackerv2_shared_front_initial_GT_scale':'SpaV2\nshared / GT scale'}
    fig,axes=plt.subplots(len(groups),2,figsize=(13,4*len(groups)),squeeze=False)
    for k,(group,group_methods) in enumerate(groups):
        for ax,cam in zip(axes[k],['fixed','orbit']):
            for i,method in enumerate(group_methods):
                y=[r['visible_epe_m']*100 for r in rows if r['method']==method and r['camera']==cam]
                ax.scatter([i]*len(y),y,s=20,alpha=.7);ax.scatter([i],[np.mean(y)],marker='_',s=200,c='black')
            ax.set_xticks(range(len(group_methods)),[labels.get(m,m) for m in group_methods],fontsize=8);ax.set_ylabel('Visible displacement EPE (cm)');ax.set_title(group+'; '+cam+' camera',fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.suptitle('SAPIEN pilot — individual clips + means; input regimes kept separate');fig.tight_layout();fig.savefig(out/'visible_geometry.png',dpi=150);plt.close(fig)
    gt=np.load(root/'pilot/object_translation_orbit/gt.npz');fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].imshow(gt['rgb'][0]);axes[0].scatter(gt['query_uv'][gt['initial_valid'],0],gt['query_uv'][gt['initial_valid'],1],s=8,c=gt['foreground'][gt['initial_valid']],cmap='coolwarm');axes[0].set_title('Same immutable initial material queries');axes[0].axis('off')
    for method in methods:
        file=root/'predictions'/method/'object_translation_orbit/tracks.npz'
        if not file.exists():continue
        flow=np.load(file)['flow'];e=np.linalg.norm(flow-gt['flow'],axis=-1);mask=gt['visible']&np.isfinite(e);curve=np.array([e[t,mask[t]].mean() for t in range(len(e))]);axes[1].plot(gt['timestamps'],100*curve,label=method)
    axes[1].set_xlabel('Time (s)');axes[1].set_ylabel('Visible flow EPE (cm)');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'translation_orbit_time.png',dpi=150);plt.close(fig)
    clips=sorted((root/'pilot').glob('*/gt.npz'));calibration=list((root/'calibration').glob('*/gt.npz'));first=np.load(clips[0]);meta=json.loads((clips[0].parent/'metadata.json').read_text());grid=int(first['grid_shape'][0]);textured=meta.get('textured',False)
    title='Textured primary-grid pilot' if textured else 'Uniform-color debug pilot'
    lines=['# Experiment 1 executed '+title,'',f'Real SAPIEN rendered scenes: {len(clips)} test clips + {len(calibration)} disjoint-seeded calibration clips, {len(first["rgb"])} frames at 20 Hz, 256×256 RGB. Queries: {grid}×{grid}, {int(first["initial_valid"].sum())} valid material points in the first listed clip. Exact actor-local material GT and time-varying ray visibility.','',
           ('Seeded nonrepeating material textures provide material cues; this study does not establish a texture-only causal effect. ' if textured else 'Uniform colored surfaces produce substantial correspondence ambiguity. ')+'Scripted linked rigid boxes stand in for articulation; no native robot-joint or broad OOD ranking claim. SpaTrackerV2, CoTracker3 depth lifts and independent DELTA frozen3D are measured separately below. Predicted-geometry rows use one GT initial-depth scale and are privileged-scale diagnostics, not raw metric monocular scores.']
    for group,group_methods in groups:
        lines+=['','## '+group,'','| Method / geometry regime | Fixed visible EPE (cm) | Orbit visible EPE (cm) |','|---|---:|---:|']
        for method in group_methods:
            vals=[np.mean([r['visible_epe_m'] for r in rows if r['method']==method and r['camera']==cam])*100 for cam in ['fixed','orbit']]
            lines.append(f'| {method} | {vals[0]:.3f} | {vals[1]:.3f} |')
    lines+=['','EPE excludes frame 0, uses GT visibility, and is conditional on finite predictions. Coverage and per-clip numbers are in `per_clip.csv`. Clip means are macro-averaged separately for each camera mode. GT-UV/exact-front-ray baseline verifies numerical closure; GT-UV/rendered-nearest-depth shows raster/sampling error. Both are visible-only; hidden material depth is never leaked. Raw tracker outputs are retained beside canonical trajectories.','',
            'SpaTracker BF16 diagnostic and corrected FP32 SDPA/math precision control appear as separate variants, not independent architectures. The initial FP32 attempt hit an upstream swallowed attention exception and was quarantined, excluded from all summaries; the tracked fail-closed patch enables supported SDPA math fallback and raises remaining failures. Model rankings must wait for input convention verification, textured scene evaluation and wider independent scene replication.','', '![Visible geometry](visible_geometry.png)','', '![Translation-orbit time curves](translation_orbit_time.png)']
    (out/'README.md').write_text('\n'.join(lines)+'\n');print(out)

if __name__=='__main__':main()
