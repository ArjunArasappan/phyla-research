# Training dataset cache inventory

Verified on the recovery GPU on October 9, 2026 at 06:21 UTC. All runtime paths below are on persistent `/workspace`. Rebuilt samples are queued, not yet completed.

| Dataset / model | Saved contents | Status |
|---|---|---|
| Shared PushCube demonstrations | 64 successful demonstrations: RGB, native 8D actions, camera/geometry data; approximately 652 MB | Preserved on GPU and in the locally verified `exp03-core.tar.gz` archive |
| N=16 training subset | 16 episode identities, 80 windows, splits, normalization statistics, canonical GT flow tracks, original metadata | Preserved; every referenced raw file and flow-track file exists; metadata/manifest/statistic identities verified |
| N=64 training subset | 64 episode identities, 318 windows, splits, normalization statistics, canonical GT flow tracks, original metadata | Preserved and verified as above |
| LTX policy training tensors | Conditioning-image latents, text embeddings/masks, normalized actions, future RGB latents, future flow latents, validity/source/window metadata | Original `.pt` samples lost with deleted node. Rebuilding 398 samples into a new debugging cache, queued after evaluation |
| Wan policy training tensors | Wan-specific conditioning/text/RGB/flow encodings | Not generated: no Wan policy SFT experiment has run |
| Wan and LTX Experiment 2 VAE outputs | Encoded motion representations, reconstructions and measurement arrays for 276 completed codec jobs | Preserved in Experiment 2 scientific archives; these are codec evaluation artifacts, not policy training caches |
| Tracker comparison inputs / outputs | Simulated RGB/cameras/GT point trajectories and generated tracker labels | Preserved in Experiment 1 scientific archives |

## Persistent paths

Raw demonstrations and original prepared manifests:
`/workspace/phyla-research-runs/exp03-recovery-root/data/exp03/pushcube/`

New LTX debugging cache:
`/workspace/phyla-research-runs/exp03-recovery-root/data/exp03/pushcube/debug-cache/ltx/N016/`
`/workspace/phyla-research-runs/exp03-recovery-root/data/exp03/pushcube/debug-cache/ltx/N064/`

Queue status/log:
`/workspace/phyla-research-runs/training-cache-status.json`
`/workspace/phyla-research-runs/training-cache.log`
Tmux session: `phyla-training-cache`.

Scientific archives:
`/workspace/phyla-research-preserved/archives/`

## Cache contract

A, R and F-GT use the same per-subset cached sample format, which includes all three supervision targets. Seeds and arms do not need duplicate caches. N016/N064 retain their own original normalization statistics. This helper regenerates latents with pinned LTX weights and the recovered software environment, while preserving original window selection, instruction, raw frames, actions and GT flow.

Historical manifests are immutable. New sample SHA-256 hashes, manifest hash and dataset identity distinguish the regenerated cache; the original dataset identity and rebuild environment are recorded. Regenerated tensors are not claimed to match historical tensors bit for bit. The queue waits for all 36 evaluation groups and their audit before loading encoders, uses an exclusive cache-process lock, resumes hash-verified samples, and validates sample checksums and finite tensors on completion. All full sample tensors remain on persistent storage; compact manifests and completion reports can be backed up locally without copying redundant latent payloads.
