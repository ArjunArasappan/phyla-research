# GPU execution status — 2026-10-08

Three experiment workers are active under the coordinating chat, each in an isolated remote worktree on branch `agent/exp01`, `agent/exp02`, or `agent/exp03`.

| Worker | Reserved GPUs | Current phase |
|---|---|---|
| Experiment 1 | 0, 1 | Actual simulator GT generation and tracker setup |
| Experiment 2 | 2, 3 | Real VAE smoke passed; awaiting disjoint simulator calibration/GT bundles |
| Experiment 3 | 4, 5 | Expert collection, LTX conversion, matched timing/LoRA audits |
| Coordinator | 6, 7 | Shared acceptance checks and leased evaluation |

Verified on the node: core Torch2.12/cu130 BF16 CUDA forward/backward; two-GPU NCCL all-reduce; original48 tests passed. Real Wan/LTX VAE smoke passed with expected latent shapes. The initial codec source was an explicitly synthetic acceptance fixture, not a simulator tracking benchmark result.

## Live monitoring

```bash
python3 /home/ubuntu/phyla-research/scripts/experiment_status.py
python3 /home/ubuntu/phyla-research/scripts/experiment_status.py --json
tmux list-sessions
```

Run through SSH using `-o IdentitiesOnly=yes -i ~/.ssh/national_compute`. Workspaces, caches, data, logs, and checkpoints live under `/mnt/nvme/scratch/phyla-ubuntu/`. Read each experiment's `control/exp0N/status.json`, `STATUS.md`, `ai_notes/`, and `results/ai/` for current scientific progress. This static startup snapshot does not replace live status.

All Wan/LTX DiT adaptation uses LoRA plus declared new heads; pretrained base weights, VAEs, and text encoders remain frozen. Experiments1/2 perform frozen inference/reconstruction.
