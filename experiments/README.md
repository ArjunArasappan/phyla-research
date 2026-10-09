# Experiments

Experiment specifications, notes, and result summaries live here. Each experiment has its own directory; reusable implementation stays in the main package.

## Experiment index

- [Experiment 1: simulator-grounded 3D tracking and motion-label VAE reconstruction](01-tracker-and-vae-benchmark/README.md)
- [Experiment 1 structured protocol](01-tracker-and-vae-benchmark/protocol.yaml) — proposed settings, not an executable configuration yet.
- [Experiment 3: flow versus RGB data efficiency and OOD generalization](03-flow-vs-rgb-data-efficiency-ood/README.md) — tentative design.
- [FloMo reimplementation plan](flomo-reimplementation-plan.md) — broader training and ManiSkill evaluation design.

## Organization

```text
experiments/
  README.md
  flomo-reimplementation-plan.md
  01-tracker-and-vae-benchmark/
    README.md                 # comprehensive experiment writeup
    protocol.yaml             # structured experimental design
    ai_notes/                 # AI planning, observations, and debugging notes
    user_notes/               # your personal notes
    results/
      README.md
      ai/                     # AI-authored results and analysis
      user/                   # your results and analysis
  03-flow-vs-rgb-data-efficiency-ood/
    README.md                 # tentative Experiment 3 design
    ai_notes/
    user_notes/
    results/
      ai/
      user/
```

Keep notes and result summaries as Markdown with descriptive, dated filenames. Link results to the run ID, code/checkpoint revisions, configuration, and artifact location. Large arrays, checkpoints, videos, and generated report assets belong in an ignored data/runs location or on the GPU host; save their paths and hashes in the result writeup.

An experiment plan is not a measured result. Record actual execution status explicitly and preserve failed runs alongside completed ones.
