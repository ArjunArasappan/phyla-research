# Experiment 1 live status

24 real SAPIEN GT clips complete: 12 pilot + 12 calibration. GT pixel-center/depth closure validated. Frozen CoTracker3+GTdepth/GTcamera CUDA completed all12pilot clips; ~3.6 GB peak VRAM. SpaTrackerV2 GT-geometry running on GPU0, exact PID36104, tmux phyla-exp01.

Data and raw predictions: `/mnt/nvme/scratch/phyla-ubuntu/data/exp01/`. Logs: `/mnt/nvme/scratch/phyla-ubuntu/runs/exp01/`.

These are debug16×16 query results on plain colored kinematic rigid geometry. Uniform floor makes correspondence ambiguous; no general tracker ranking claim. Native RGB frontends, independent3D baseline and64×64density remain pending.
