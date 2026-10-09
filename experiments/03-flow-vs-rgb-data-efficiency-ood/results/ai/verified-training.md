# Experiment 3: verified training and recovery status

All 12 bounded-pilot policies completed real LTX-Video 2B LoRA SFT and their learned weights are saved in the verified local backup. Final comparative policy-success results are not yet available locally.

## Verified experiment

- One ManiSkill PushCube task; native eight-dimensional Panda joint-position control, including gripper.
- 64 successful expert demonstrations, with 5,470 recorded action intervals at 20 Hz (273.5 seconds, including terminal holds).
- Nested N16/N64 subsets; A (action only), R (action + RGB), F-GT (action + simulator GT displacement); two optimizer seeds.
- All 12 runs: 1,000 updates, effective batch four, LoRA rank 16/alpha 32, 23,922,696 trainable parameters.
- Same future-stream token layout and t+1 through t+16 timestamps for R/F; frozen VAE, text encoder and pretrained transformer. No future targets enter policy evaluation.
- Exact frozen-base SHA remained unchanged in every run: `948c0f46770b22170a0157f6a508ce1d6541a93b75a400935363bc100a51efb9`.

## Measured training diagnostics

These are training action-velocity losses averaged over the final 100 updates. They are not task success, held-out accuracy, or evidence of data efficiency. Target-domain RGB and flow losses are deliberately not compared.

| Demonstrations | Arm | Optimizer seed | Final-100 action loss | Updates | Base unchanged |
|---|---|---|---|---|---|
| 16 | A | 0 | 0.1383 | 1000 | yes |
| 16 | A | 1 | 0.1939 | 1000 | yes |
| 16 | F-GT | 0 | 0.1314 | 1000 | yes |
| 16 | F-GT | 1 | 0.1668 | 1000 | yes |
| 16 | R | 0 | 0.1424 | 1000 | yes |
| 16 | R | 1 | 0.1774 | 1000 | yes |
| 64 | A | 0 | 0.1985 | 1000 | yes |
| 64 | A | 1 | 0.2123 | 1000 | yes |
| 64 | F-GT | 0 | 0.1655 | 1000 | yes |
| 64 | F-GT | 1 | 0.1874 | 1000 | yes |
| 64 | R | 0 | 0.1691 | 1000 | yes |
| 64 | R | 1 | 0.1922 | 1000 | yes |

Sum of logged optimizer-update wall time across the GPU-assigned runs: 1.802 hours. This excludes setup, data collection, encoding, evaluation, checkpoint export and billed idle time.

## Evaluation and limitations

The last observed node status confirmed 25 of 36 requested ID/color/camera condition cells complete before SSH became unreachable. This is a last-observed progress count, not a claim that the remaining jobs failed or completed. Their per-reset scientific records and final success aggregates have not been recovered locally; no success curve or efficacy conclusion is reported here.

The protocol uses five held-out resets per condition, one task, one demonstration-subset seed, a shortened training budget, RGB plus instruction without proprioception, one cube-color intervention and one 30-degree camera shift. Even a complete result remains exploratory. GT flow is privileged training supervision. Predicted-flow, Wan policy SFT and midtraining-transfer arms have not been run.

## Preserved artifacts

The locally verified `exp03-core.tar.gz` scientific archive contains all 12 learned-only policy bundles, their configuration/normalization and exact integrity audits, the 64 raw simulator demonstrations, canonical flow labels and window/subset manifests.

Archive size: 1,134,587,744 bytes. SHA256: `d659cc979a08041553b8f354364933e956789f44eae56aa6f6ce51f746909bb2`.

Full optimizer/RNG checkpoints remain separate from the core archive; their durable transfer has not been confirmed. The pretrained base and regenerable latent caches are deliberately excluded.

## Recovery

First recover the GPU host’s saved evaluation directories and audit them with the supplied contract checker, including physical-reset equality across all policies and conditions. If those records cannot be recovered, the preserved policy bundles permit the exact fixed evaluation matrix to be rerun on another compatible GPU host using the same base revision, simulator version, reset seeds, sampler and controller. Do not infer success from the training losses.
