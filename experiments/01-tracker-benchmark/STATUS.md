# Experiment 1 executed status

The bounded pilots are complete. No tracker GPU jobs are running.

- 12 uniform-color debug clips and 12 separate calibration clips.
- 12 textured primary clips with a 64×64 grid and 12 separate calibration clips.
- 12 exact material-ID subset clips with a 16×16 grid for a paired density control.
- SpaTrackerV2, CoTracker3 plus depth, and independent DELTA completed every primary and matched-control clip.
- RGB-only geometry and native Spa/CoTracker completed the debug matrix; initial-GT-scale diagnostics are separate from GT geometry.

Reports, physical metrics, figures and compact synchronized footage are in `results/ai/`. Raw arrays remain at `/mnt/nvme/scratch/phyla-ubuntu/data/exp01/`. Checkpoint and artifact hashes are in the primary artifact ledger. Invalid silent-attention outputs and the legacy one-channel zero control are excluded; reasons and hashes are preserved in `ai_notes/quarantine-ledger.json`.

One seed per archetype and scripted linked boxes limit general claims. Official TAPVid3D integration and broader scene/asset replication remain future work. The exact-ID matched comparison isolates query density/context on textured scenes; comparison with the older uniform-color debug pilot does not isolate texture causation.

The GPU node expires at 05:11 UTC on 9 October 2026. Root is backing up the scientific archives and integrating scoped commits.
