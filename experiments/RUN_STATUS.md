# GPU execution and preservation status

Original node expiry: **2026-10-09 05:11 UTC** (October 8, 22:11 PDT). The user reports National Compute confirmed that the node and its disk were deleted. Remote-only artifacts are lost. SSH retries and the backup heartbeat have been stopped; no reboot or termination command was issued by the experiment coordinator.

| Experiment | Verified completion | Outstanding |
|---|---|---|
| 1: tracking | 60 validated GT bundles; all three trackers on original, textured64 and matched16 case studies; reports, physical errors and videos | None for declared bounded pilots |
| 2: motion VAEs | 276 frozen Wan/LTX reconstructions, arrays/latents, figures, manifests and reports | Broader proposed ablations |
| 3: policy supervision | 12/12 LoRA runs at 1,000 updates; 25/36 evaluation conditions observed complete before connectivity loss | Final completion unknown; full evaluation recordings and audit lost. Evaluation must be repeated from saved policies. |

Core B300 CUDA/BF16 and two-GPU NCCL checks passed. All **58 integrated tests** passed after the final tracker merge and BF16 evaluation repair. Pretrained LTX base, VAE and text encoder remain frozen; LoRA and declared new heads are the only trainable parameters. Experiments 1/2 perform frozen inference.

## Preserved artifacts

Initial tracker and VAE scientific archives plus all 12 learned policies/64 demonstrations are saved and checksum-verified locally. Textured tracker and practical/textured VAE archives were pushed to [`artifacts/gpu-pilot-2026-10-09`](https://github.com/ArjunArasappan/phyla-research/tree/artifacts/gpu-pilot-2026-10-09) as checksummed 45 MiB chunks; these archives were fully reconstructed and checksum-verified locally. Readable code, reports, figures and compact videos are on `main`.

All 12 final optimizer/RNG checkpoint archives were created on the node, but their upload connection failed before starting. The full evaluation scientific arrays/videos and optimizer/RNG archives were not copied off and are lost following provider-confirmed deletion. Downloadable base weights can be fetched again; prepared caches can be regenerated.

## Rerun evaluations from preserved policies

On a new GPU, restore the twelve learned policy bundles, fetch the recorded frozen model revisions, recreate the simulator environment, and repeat the full paired ID/OOD evaluation with the physical-reset audit. Do not combine unaudited partial summaries with the new run as a complete benchmark. Completed training need not be repeated; exact optimizer/RNG continuation is unavailable. Preserve each completed evaluation group off-node immediately.

To update ordinary code without fetching the large backup branch:

```bash
git config --add remote.origin.fetch '^refs/heads/artifacts/*'
git fetch origin main
git merge --ff-only origin/main
```

The artifact branch contains its own restore instructions and complete per-chunk/archive SHA-256 catalog. Restore large archives on storage with sufficient free space.
