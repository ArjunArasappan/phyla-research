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
    methods=sorted(set(r['method'] for r in rows));fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,cam in zip(axes,['fixed','orbit']):
        for i,method in enumerate(methods):
            y=[r['visible_epe_m']*100 for r in rows if r['method']==method and r['camera']==cam]
            ax.scatter([i]*len(y),y,s=20,alpha=.7);ax.scatter([i],[np.mean(y)],marker='_',s=200,c='black')
        ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right');ax.set_ylabel('Visible displacement EPE (cm)');ax.set_title(cam+' camera; individual clips + mean');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Debug SAPIEN pilot — plain-color geometry, 16×16 queries; no general ranking claim');fig.tight_layout();fig.savefig(out/'visible_geometry.png',dpi=150);plt.close(fig)
    gt=np.load(root/'pilot/object_translation_orbit/gt.npz');fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].imshow(gt['rgb'][0]);axes[0].scatter(gt['query_uv'][gt['initial_valid'],0],gt['query_uv'][gt['initial_valid'],1],s=8,c=gt['foreground'][gt['initial_valid']],cmap='coolwarm');axes[0].set_title('Same immutable initial material queries');axes[0].axis('off')
    for method in methods:
        file=root/'predictions'/method/'object_translation_orbit/tracks.npz'
        if not file.exists():continue
        flow=np.load(file)['flow'];e=np.linalg.norm(flow-gt['flow'],axis=-1);mask=gt['visible']&np.isfinite(e);curve=np.array([e[t,mask[t]].mean() for t in range(len(e))]);axes[1].plot(gt['timestamps'],100*curve,label=method)
    axes[1].set_xlabel('Time (s)');axes[1].set_ylabel('Visible flow EPE (cm)');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'translation_orbit_time.png',dpi=150);plt.close(fig)
    lines=['# Experiment 1 executed debug pilot','', 'Real SAPIEN rendered scenes: 12 test clips + 12 disjoint-seeded calibration clips, 33 frames at 20 Hz, 256×256 RGB. Queries: 16×16 with approximately 132 valid material points. Exact actor-local material GT and time-varying ray visibility.','',
           'This is an executable pipeline and diagnostic pilot, not the proposed full tracking comparison. Uniform colored surfaces produce substantial correspondence ambiguity; articulated case uses scripted linked rigid boxes, not native robot joints. SpaTrackerV2, CoTracker3 depth lifts and independent DELTA frozen3D all completed. Textures and larger query density remain pending. Predicted-geometry rows use one GTinitialdepth scale and are separate privileged-scale diagnostics.','',
           '| Method / geometry regime (native rows use initial-GT-scale diagnostic) | Fixed camera visible EPE (cm) | Orbit camera visible EPE (cm) |','|---|---:|---:|']
    for method in methods:
        vals=[np.mean([r['visible_epe_m'] for r in rows if r['method']==method and r['camera']==cam])*100 for cam in ['fixed','orbit']]
        lines.append(f'| {method} | {vals[0]:.3f} | {vals[1]:.3f} |')
    lines+=['','EPE excludes frame 0, uses GT visibility, and is conditional on finite predictions. Coverage and per-clip numbers are in `per_clip.csv`. Mean averages six clips per camera. GT-UV/exact-front-ray baseline verifies numerical closure; GT-UV/rendered-nearest-depth shows raster/sampling error. Both are visible-only; hidden material depth is never leaked. Raw tracker outputs are retained beside canonical trajectories.','',
            'SpaTracker BF16 diagnostic and corrected FP32 SDPA/math precision control appear as separate variants, not independent architectures. The initial FP32 attempt hit an upstream swallowed attention exception and was quarantined, excluded from all summaries; the tracked fail-closed patch enables supported SDPA math fallback and raises remaining failures. Model rankings must wait for input convention verification, textured scene evaluation and wider independent scene replication.','', '![Visible geometry](visible_geometry.png)','', '![Translation-orbit time curves](translation_orbit_time.png)']
    (out/'README.md').write_text('\n'.join(lines)+'\n');print(out)

if __name__=='__main__':main()
