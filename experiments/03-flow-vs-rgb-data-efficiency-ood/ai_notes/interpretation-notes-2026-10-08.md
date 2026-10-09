# Interpretation notes for the completed Experiment 3 pilot

This is an exploratory comparison on one PushCube task. It uses N=16/64 nested independent demonstrations, one subset seed, two optimizer seeds, 1,000 updates and effective batch four. Every run therefore samples 4,000 training windows, with more repeated exposure at N=16. This is a fixed-update comparison rather than equal epochs. It does not reproduce the proposed 20,000-update, three-task, six-N main study.

The relevant endpoint is actual simulator success under the same physical reset and controller, not training loss. RGB and flow target losses occupy different latent distributions and cannot be ranked numerically against each other. Compare F-GT against R at each N and condition; show both optimizer-seed outcomes and the action-only control. A remains a matched token-budget control with masked dummy future tokens and no auxiliary loss.

OOD claims are limited to one held-out cube color and a 30-degree camera shift. Both preserve the goal marker and physical reset. A smaller OOD drop is not beneficial if ID and OOD performance are both poor; always report absolute success. Five resets per seed/condition and only two trained seeds are too small for strong generalization claims. Lines between N16 and N64 are descriptive, not evidence of a fitted sample-complexity law.

GT flow is privileged training supervision. Neither GT flow nor future RGB is supplied to the deployed policy. This pilot is RGB-plus-instruction policy inference without proprioception, using native eight-dimensional joint-position control. These choices differ from the initial tentative EE-controller/proprioception design and should remain explicit.

If all or most policies are unsuccessful in ID, the pilot does not establish data efficiency or OOD robustness. The immediate next study should first establish a reliable ID policy and inspect action/observation conditioning, inference integration and convergence under a shared development protocol. Adding predicted-flow noise or a much larger OOD matrix would be difficult to interpret before that gate.

If GT flow helps ID or either tested nuisance shift, the result motivates a larger matched study; it does not establish broad object/dynamics generalization, practical tracker-label effectiveness, or action-free midtraining transfer. Those arms have not been run here.

Final learned adapter/action weights, full optimizer and RNG checkpoints, normalization/window manifests and all per-reset records must be linked to the specific model and data revisions. Historical BF16 diagnostic failures should remain in provenance alongside successful repaired evaluations, so the repair cannot be mistaken for a model retraining or an improved success condition.

Forecast galleries show the joint model's first 16-frame prediction from the initial observation. The controller executes four actions before replanning, so the eventual 150-step rollout is not a counterfactual realization of all 16 originally predicted actions. Do not score the later forecast against that changing rollout as though the action sequence stayed fixed. A separate open-loop action-conditioned diagnostic would be needed for that physical prediction claim.
