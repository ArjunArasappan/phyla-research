# Experiment 2 status

**Debug GPU pilot matrix completed: 168/168 primary jobs and 24/24 temporal-context jobs.** Broader density, scale, native-monocular and optional noise/raster/precision ablations remain pending.

- Seven sources: exact GT,zero, GT-UV/front-depth,CoTracker3 + GT geometry,DELTA + GT geometry,SpaTrackerV2 BF16 andvalidated SpaTrackerV2 FP32-math.
- Twelve test clips and twelve separate calibration clips; 16×16 initial queries,256×256 pseudo-RGB,17-frame primary window.
- Both VAEs frozen; no T5, DiT, training or LoRA in this experiment.
- Physical GPUs 2/3 released to Experiment 3 after all codec jobs finished.
- [Measured primary report](results/ai/pilot-v2/README.md), [temporal context control](results/ai/context33/README.md), [actual RGB and 3D animations](results/ai/animations/README.md).
- [Scientific artifact checksum ledger](results/ai/artifact-manifest.json) and [storage accounting](results/ai/storage-summary.json).

Node root `/mnt/nvme/scratch/phyla-ubuntu`. Scientific arrays/latents remain in `runs/exp02/simulator-pilot-v2/` and `runs/exp02/simulator-context33-v1/`; mapping/manifests in `data/exp02/`. All 192 reconstructions passed finite decoder checks and exact vector error identity checks (≤1e-6m); per-file SHA256/bytes exported. Five meaningful renderer/geometry CPU tests passed.

Runners: `experiments/exp02.py` (`synth`, `calibrate`, `convert-ltx-vae`, `run`, `batch`), `exp02_download_assets.py`, `exp02_prepare_pilot.py`, `exp02_report.py`, `exp02_context_report.py`, `exp02_animate.py`, `exp02_integrity.py`; tests `test_exp02.py`.

With shared core Python and `CUDA_VISIBLE_DEVICES`, use `batch --codec wan|ltx --manifest <JSON jobs> --stats <frozen calibration JSON> --output <run root>`. Identity records source/GT/statistics/code SHA, source kind, codec and shape. Existing outputs reject incompatible identity; `complete.json` is written last.

Scientific limitations: debug density, colored scripted SAPIEN boxes/links, one seed per archetype, privileged GT depth/camera geometry. Self-reconstruction is not label accuracy. Future context is not policy conditioning. No OOD or policy success claim follows from this pilot.
