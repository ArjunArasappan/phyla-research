"""Paired query-density comparison on exact shared material identities."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from exp01_track import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    methods=['cotracker3_gt_geometry','spatrackerv2_gt_geometry_fp32_math','delta_gt_geometry']
    for file in sorted((root/'textured16_matched/pilot').glob('*/gt.npz')):
        gt=np.load(file);parent=np.load(root/'textured64/pilot'/file.parent.name/'gt.npz');ids=gt['query_ids'];assert np.allclose(gt['flow'],parent['flow'][:,ids],equal_nan=True)
        for method in methods:
            small=np.load(root/'textured16_matched/predictions'/method/file.parent.name/'tracks.npz')['flow'];large=np.load(root/'textured64/predictions'/method/file.parent.name/'tracks.npz')['flow'][:,ids]
            for name,mask in [('visible',gt['visible']),('foreground_visible',gt['visible']&gt['foreground'][None])]:
                if not mask[1:].any():continue
                a16=metrics(small,gt['flow'],mask);a64=metrics(large,gt['flow'],mask)
                rows.append({'clip':file.parent.name,'method':method,'mask':name,'epe_16_m':a16['epe_m'],'epe_64_same_queries_m':a64['epe_m'],'difference_16_minus_64_m':a16['epe_m']-a64['epe_m'],
                    'coverage_16':a16['coverage'],'coverage_64_same_queries':a64['coverage'],'pck1cm_16':a16['pck_1cm'],'pck1cm_64':a64['pck_1cm']})
    with (out/'paired_query_density.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Query-density control','', 'Each 16×16 run uses an exact material-ID subset of the corresponding textured 64×64 scene. The comparison rescored dense predictions on precisely those same queries, with the same GT visibility mask. RGB, poses, geometry, checkpoint, support policy and seed are shared. The number of jointly tracked query points changes; this is an architecture/context sensitivity test.','', '| Method | 16-grid visible EPE (cm) | Dense-grid same-query EPE (cm) | Paired difference (cm) |','|---|---:|---:|---:|']
    for method in methods:
        selected=[r for r in rows if r['method']==method and r['mask']=='visible'];small=np.mean([r['epe_16_m'] for r in selected])*100;large=np.mean([r['epe_64_same_queries_m'] for r in selected])*100
        lines.append(f'| {method} | {small:.3f} | {large:.3f} | {small-large:+.3f} |')
    lines+=['','This controls query density on textured scenes. Comparing these numbers with the older uniform-color debug pilot changes both texture and query positions/density; it does not isolate a texture-only causal effect. One scene seed per archetype remains a case-study limitation. Foreground-visible paired errors and finite coverage are saved in the CSV.'];(out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
