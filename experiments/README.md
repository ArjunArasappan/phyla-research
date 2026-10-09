# Experiments

1. [3D tracker benchmark](01-tracker-benchmark/README.md)
2. [Motion-label VAE reconstruction](02-motion-vae-reconstruction/README.md)
3. [Flow versus RGB data efficiency and OOD](03-flow-vs-rgb-data-efficiency-ood/README.md)

[Three-agent GPU launch plan](agent-launch-plan.md)

[FloMo reimplementation plan](flomo-reimplementation-plan.md)

Each experiment has AI/user notes and separate AI/user results folders. Preserve personal notes. Keep large arrays, checkpoints, and videos on NVMe run/data storage and link their immutable manifests from result summaries.

The original combined tracker/codec design is archived under Experiment 1; the current numbered protocols take precedence. All pretrained Wan/LTX DiT adaptation uses LoRA plus declared new heads, with the base DiT, VAE, and text encoder frozen.
