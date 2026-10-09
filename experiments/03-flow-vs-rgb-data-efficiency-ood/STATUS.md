# Experiment 3 GPU execution status

2026-10-08, America/Los_Angeles. Genuine B300 execution; exploratory pilot, not the proposed main sweep.

- Collected 64 successful independent official ManiSkill PushCube motion-planning demonstrations. Native Panda `pd_joint_pos`, 8 actions including gripper; 20 Hz; 256px external RGB. No conversion to EE actions.
- Real LTX v0.9.6 2B LoRA plumbing runs completed 10 updates for paired RGB and GT-flow arms. Both have 1,947,308,168 total / 23,922,696 trainable parameters; rank16, alpha32. Exact frozen-base SHA checked unchanged; optimizer contains only registered LoRA and new action interfaces.
- Closed-loop smoke evaluation completed on two identical held-out resets: both 0/2 successes. This verifies actual native actions and inference, and says nothing about target efficacy after only ten updates. Steady planning latency approximately 0.22 seconds/chunk with four sampler steps.
- 52 CPU tests pass, including exact paired LTX tokens/coordinates/masks/action-forward equivalence, future target timestamps, null-stream isolation, and strict LoRA integrity.
- Prepared nested N16 and N64 datasets, respectively 80 and 318 windows. Normalizers use selected episodes only. Future RGB and anchor-camera displacement both supervise t+1..t+16, padded identically to 17 by the native VAE. GT depth rays use sampled integer indices plus SAPIEN half-pixel centers; occlusion does not destroy rigid material-point existence.
- ID/yellow-cube/30-degree-camera paired-reset sanity passed: physical object/goal/robot states identical; images change; official expert succeeds in all three conditions.

## Frozen bounded pilot

A / R / F-GT × N16/N64 × optimizer seeds0/1; subset seed0; 12 runs. All share LTX initialization, rank16, effective batch4, 1000 updates, learning rate5.6e-5, task RGB/text input, and native controller. A retains masked dummy future tokens with no content path to actions and zero auxiliary loss. R/F future stream layout is exactly identical. Shared lambda1 is untuned. Fixed final checkpoint; no validation/OOD selection.

Five held-out resets20000..20004 per ID/color/camera condition, 150 control steps, eight Euler sampler steps, four-action execution chunks. Save every reset result, RGB trajectory video, native actions, object/TCP trajectories, and selected auxiliary forecasts. These small evaluation counts are exploratory, not definitive confidence intervals. VAE/text/base DiT frozen in every arm; no future labels are given at policy evaluation.

GPU4=N16 flow,5=N16 RGB,2=N64 flow,3=N64 RGB,7=action-only queues,6=evaluation. Full resumable checkpoints/RNG stay on NVMe. Immutable policy exports contain only learned adapters/action heads plus config/stats/audit for durable backup. Hard new-job cutoff04:40UTC; node expiry05:11UTC on2026-10-09.

## Limitations and explicit deviations

Single PushCube task, RGB+instruction policy without proprioception, native joint-position control, two N values, one subset seed, shortened1000-update budget, and five resets per condition. No midtraining and no predicted-flow policy arm yet. Appearance shift is object color only; camera shift is one azimuth condition. Do not generalize to broad appearance/geometry/dynamics OOD or interpret prediction losses as closed-loop efficacy.

Large artifacts: `/mnt/nvme/scratch/phyla-ubuntu/data/exp03` and `/mnt/nvme/scratch/phyla-ubuntu/runs/exp03`. Monitor JSON: `/mnt/nvme/scratch/phyla-ubuntu/control/exp03/jobs/`.
