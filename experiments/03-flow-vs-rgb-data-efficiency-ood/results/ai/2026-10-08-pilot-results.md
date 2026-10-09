# Experiment 3 measured pilot results

2026-10-08 (America/Los_Angeles). This file contains actual completed cells only.

Started training records: 12/12; completed ID/OOD condition cells: 0/36.

1000 updates, effective batch4, frozen LTX2B with rank16 LoRA + action interfaces. N16/N64 nested episodes; two optimizer seeds; five paired resets per condition. Fixed final checkpoint. No future RGB/flow enters runtime observations.

| N | Arm | Optimizer seed | Condition | Success |
|---|---|---|---|---|

Small pilot counts and a single task do not establish data efficiency or broad OOD generalization. A small OOD gap can reflect uniformly unsuccessful policies. Prediction-domain losses are not compared across RGB and flow.

Protocol deviations: native8D joint-position control, RGB+instruction without proprioception, one color shift and one camera30 shift, shortened budget, one subset seed. GT labels are privileged training supervision. VAE/text/base transformer remain frozen.

[Measured cells CSV](pilot_metrics.csv) · [Full metrics and protocol JSON](pilot_metrics.json)
