# GPU execution and preservation status

Node expiry: **2026-10-09 05:11 UTC** (October 8, 22:11 PDT). This report records verified milestones; SSH to `204.52.23.47` currently times out before login. Network permission is granted and GitHub remains reachable. No reboot or termination command was issued. The exact cause requires provider-console status.

| Experiment | Verified completion | Outstanding |
|---|---|---|
| 1: tracking | 60 validated GT bundles; all three trackers on original, textured64 and matched16 case studies; reports, physical errors and videos | None for declared bounded pilots |
| 2: motion VAEs | 276 frozen Wan/LTX reconstructions, arrays/latents, figures, manifests and reports | Broader proposed ablations |
| 3: policy supervision | 12/12 LoRA runs at 1,000 updates; 25/36 evaluation conditions observed complete before connectivity loss | Remaining/final condition count, physical-reset audit, full evaluation retrieval and analysis |

Core B300 CUDA/BF16 and two-GPU NCCL checks passed. All **58 integrated tests** passed after the final tracker merge and BF16 evaluation repair. Pretrained LTX base, VAE and text encoder remain frozen; LoRA and declared new heads are the only trainable parameters. Experiments 1/2 perform frozen inference.

## Preserved artifacts

Initial tracker and VAE scientific archives plus all 12 learned policies/64 demonstrations are saved and checksum-verified locally. Textured tracker and practical/textured VAE archives were pushed to [`artifacts/gpu-pilot-2026-10-09`](https://github.com/ArjunArasappan/phyla-research/tree/artifacts/gpu-pilot-2026-10-09) as checksummed 45 MiB chunks; these archives are also being reconstructed locally from GitHub. Readable code, reports, figures and compact videos are on `main`.

All 12 final optimizer/RNG checkpoint archives were created on the node, but their upload connection failed before starting. Evaluation scientific arrays/videos are not yet retrieved. Downloadable base weights and reproducible prepared latent caches remain on the node.

## Recovery without duplicate jobs

Once SSH recovers, inspect registered tmux sessions and `control/exp03/jobs`, then run the actual-artifact audit and report scripts. Do not retrain completed policies or rerun completed evaluation cells. Preserve final optimizer archives and the completed evaluation archive to the artifact branch; verify hashes before expiry. The coordinator has local copies of the cross-policy audit/finalizer and archive publisher.

To update ordinary code without fetching the large backup branch:

```bash
git config --add remote.origin.fetch '^refs/heads/artifacts/*'
git fetch origin main
git merge --ff-only origin/main
```

The artifact branch contains its own restore instructions and complete per-chunk/archive SHA-256 catalog. Restore large archives on storage with sufficient free space.
