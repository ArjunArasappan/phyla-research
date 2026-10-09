from pathlib import Path
import csv,json,time
from statistics import mean

root=Path('/mnt/nvme/scratch/phyla-ubuntu');work=root/'worktrees/exp01/experiments/01-tracker-benchmark'
for namespace,reportname in [('textured64','textured64-pilot'),('textured16_matched','textured16-matched')]:
    rows=[]
    for file in sorted((root/'data/exp01'/namespace/'predictions').glob('*/*/extended_metrics.json')):
        if not (file.parent/'READY').exists():continue
        d=json.loads(file.read_text());fg=d.get('foreground_visible',{})
        rows.append({'method':file.parent.parent.name,'clip':file.parent.name,'visible_epe_m':d['visible']['epe_m'],'visible_coverage':d['visible']['coverage'],
            'foreground_visible_epe_m':fg.get('epe_m'),'foreground_object_macro_visible_epe_m':d.get('foreground_object_macro_visible_epe_m'),
            'velocity_epe_m_per_s':d['velocity_epe_m_per_s'],'acceleration_epe_m_per_s2':d['acceleration_epe_m_per_s2'],
            'dx_mae_m':d['axis_mae_m'][0],'dy_mae_m':d['axis_mae_m'][1],'dz_mae_m':d['axis_mae_m'][2]})
    out=work/'results/ai'/reportname
    with (out/'foreground_and_physical_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['','## Foreground accuracy and physical diagnostics','',
        'The visible-scene mean is dominated by background points. The table below weights each foreground object equally within each clip and then averages clips. Occluded-point errors and missing-prediction coverage remain separate in extended_metrics.json and the CSV.','',
        '| Method | Visible foreground-object macro EPE (cm) | Visible coverage |','|---|---:|---:|']
    for method in sorted(set(r['method'] for r in rows)):
        selected=[r for r in rows if r['method']==method];values=[r['foreground_object_macro_visible_epe_m'] for r in selected if r['foreground_object_macro_visible_epe_m'] is not None]
        lines.append(f"| {method} | {mean(values)*100:.3f} | {mean(r['visible_coverage'] for r in selected):.4f} |")
    lines+=['','Per-axis MAE/RMSE, velocity (m/s), acceleration (m/s²), visibility/motion strata and per-object scores are retained beside every source bundle. CoTracker lifting rejects nonpositive or missing sensor depth; missing geometry counts as a failure in threshold scores. Older uniform-color debug sources are not merged into this primary leaderboard.']
    p=out/'README.md';s=p.read_text().split('\n## Foreground accuracy')[0]
    if namespace=='textured16_matched':s=s.replace('Textured primary-grid pilot','Textured exact-query-subset control')
    p.write_text(s+'\n'+'\n'.join(lines)+'\n')

status={'experiment':'exp01','state':'complete','stage':'bounded_pilots_complete_archiving','gpus':[0,1],'running_pids':[],
    'debug_test_clips':12,'debug_calibration_clips':12,'primary_textured64_test_clips':12,'primary_textured64_calibration_clips':12,
    'matched_textured16_test_clips':12,'primary_tracker_jobs_completed':36,'matched_density_tracker_jobs_completed':36,
    'native_debug_tracker_jobs_completed':24,'tracker_architectures':['SpaTrackerV2','CoTracker3+depth','DELTA'],'updated_unix':time.time(),
    'limitations':['one seed per archetype: diagnostic pilot','scripted linked boxes rather than native robot joints','official TAPVid3D and broad asset replication pending','texture-only effect not isolated'],
    'data':str(root/'data/exp01'),'reports':str(work/'results/ai')}
(root/'control/exp01/status.json').write_text(json.dumps(status,indent=2))
(work/'STATUS.md').write_text('''# Experiment 1 executed status

The bounded pilots are complete. No tracker GPU jobs are running.

- 12 uniform-color debug clips and 12 separate calibration clips.
- 12 textured primary clips with a 64×64 grid and 12 separate calibration clips.
- 12 exact material-ID subset clips with a 16×16 grid for a paired density control.
- SpaTrackerV2, CoTracker3 plus depth, and independent DELTA completed every primary and matched-control clip.
- RGB-only geometry and native Spa/CoTracker completed the debug matrix; initial-GT-scale diagnostics are separate from GT geometry.

Reports, physical metrics, figures and compact synchronized footage are in `results/ai/`. Raw arrays remain at `/mnt/nvme/scratch/phyla-ubuntu/data/exp01/`. Checkpoint and artifact hashes are in the primary artifact ledger. Invalid silent-attention outputs and the legacy one-channel zero control are excluded; reasons and hashes are preserved in `ai_notes/quarantine-ledger.json`.

One seed per archetype and scripted linked boxes limit general claims. Official TAPVid3D integration and broader scene/asset replication remain future work. The exact-ID matched comparison isolates query density/context on textured scenes; comparison with the older uniform-color debug pilot does not isolate texture causation.

The GPU node expires at 05:11 UTC on 9 October 2026. Root is backing up the scientific archives and integrating scoped commits.
''')
print('Reports and completion status finalized')
