# Experiment 3: data efficiency and OOD generalization of flow supervision versus RGB video prediction

**Status:** tentative design, 2026-10-08. This document proposes an experiment; it does not report trained models or measured results. Freeze a smaller executable protocol after Experiment 1 validates the labels/codecs and GPU profiling establishes a practical budget.

## 1. Main question

Does supervising a robot policy with future 3D displacement improve demonstration efficiency and generalization more than supervising the same policy with future RGB video?

The primary comparison is **action prediction plus flow prediction versus action prediction plus RGB prediction**, using identical robot demonstrations, observations, backbone initialization, action supervision, future horizon, and training budget. An action-only policy establishes whether either auxiliary target helps at all.

This is motivated by the motion representation in [FloMo](https://flomo-wam.github.io/FloMo%20Paper.pdf), but the study should test the claimed advantage rather than assume it. Geometry-focused supervision may discourage appearance shortcuts; it may also discard task-relevant appearance, depend on privileged labels, suffer from tracker noise, and reconstruct poorly through a natural-video VAE. Each possibility needs a control.

The primary endpoints are **closed-loop simulator task success** as a function of unique labeled demonstrations and success on predefined held-out conditions. Prediction losses and inverse-dynamics probes are secondary diagnostics. Neither low flow reconstruction error nor a probe using true future motion demonstrates that a policy can act well when the future is unknown.

### Hypotheses

- **H1, data efficiency:** flow auxiliary supervision improves success at low demonstration counts relative to RGB auxiliary supervision.
- **H2, nuisance robustness:** flow-supervised policies lose less success under held-out textures, lighting, backgrounds, and camera configurations.
- **H3, broader transfer:** any advantage extends to held-out object geometry, spatial configurations, or combinations of familiar factors.
- **H4, practical labels:** an advantage with exact simulator flow remains useful with predicted tracker labels selected using Experiment 1.

H2 and H3 are separate. Appearance robustness alone is not evidence of general physical reasoning or adaptation to new contact dynamics.

## 2. Primary model arms

| Arm | Supervised outputs | Role |
|---|---|---|
| A: action only | Future action chunk | Establish whether auxiliary prediction helps |
| R: action + RGB | Same action chunk and future RGB video | Direct appearance-prediction baseline |
| F-GT: action + GT flow | Same action chunk and future cumulative 3D displacement | Clean target-type comparison with privileged training labels |
| F-pred: action + predicted flow | Same action chunk and tracker-generated displacement | Practical label-pipeline comparison |
| RF: action + RGB + flow | Actions and both future targets | Optional complementarity study; extra tokens/compute must be reported |

The initial pilot may use A, R, and F-GT. Add F-pred at selected demonstration counts once label generation is reliable. RF is a separate extension because it introduces another target stream and a larger attention workload.

Use one backbone for the primary sweep. LTX-Video 2B is the tentative first choice because it is already an option in this repository; actual feasibility depends on GPU profiling. Repeat a few representative cells with Wan2.2 TI2V-5B if resources permit. Do not compare LTX-flow against Wan-RGB as the causal test. Within each backbone, use the exact same checkpoint revision and frozen VAE for both target types.

### Match the architecture, not only the model name

R and F use one generic future-video stream with identical tensor shape, token positions, attention connectivity, trainable projections, LoRA ranks, and parameter initialization. Only the supervised target content changes. Avoid a smaller RGB head, an extra flow-only network, unequal observation history, or a flow-specific attention path that changes the comparison.

The shared architecture conditions on current RGB observation/history, proprioception, and an identical task/goal specification. It jointly predicts action and future-target latents. All arms receive the same known calibration if calibration is included; no arm receives simulator object poses, segmentation, GT depth, or GT future states as policy inputs. GT geometry is permitted as a training target for F-GT and must be labeled as privileged supervision.

Freeze the VAE and text encoder in all primary arms. Train the same LoRA and action/future projections. Use identical action normalization, optimizer, noise/timestep sampling distribution, sampler, and controller interface. Record trainable parameter counts and actual attention token counts rather than assuming target names imply parity.

A is both a practical lower-cost baseline and an architecture control. For the latter, retain a same-sized future stream of masked dummy tokens and no auxiliary loss, with no content path to action prediction. Verify masking. Report the lower-cost action-only implementation separately if its architecture/compute differs.

Natural-video pretraining favors RGB statistically, while flow supplies geometry derived from simulator privilege. Shared initialization controls the checkpoint, but it does not make the target distributions equally pretrained. A randomly initialized compact model or frozen-backbone head-only experiment is an optional diagnostic of this asymmetry, not a replacement for the primary pretrained-policy test.

## 3. Two training regimes, with separate claims

### 3A. Supervised robot-data efficiency — primary

Every arm starts from the same generic video checkpoint and trains on the same N action-labeled robot demonstrations. There is no method-specific intermediate pretraining in this comparison. R receives RGB targets from those demonstrations; F receives flow targets from the same recorded states. This tests the auxiliary supervision choice under a fixed robot-data budget.

Use proposed nested counts **N = 8, 16, 32, 64, 128, 256 demonstrations per task**. Count complete trajectories, not extracted windows. Report unique trajectories, frames, windows, action-label duration, and successful/failed demonstration composition. Long episodes create more windows; match sampled windows between methods and report that exposure.

Train single-task policies first so N has a clear meaning. Multi-task training can be a later study with fixed task mixture and an explicit per-task and total demonstration budget.

### 3B. Action-free midtraining plus low-data robot SFT — extension

This tests whether a representation learned from action-free video reduces the robot SFT requirement. Start R and F from the same checkpoint, midtrain them on the **same source clips**, using future RGB or future flow respectively, and then SFT with the same robot demonstration subsets.

For the clean transfer test, remove both auxiliary targets during SFT and train identical action-only objectives. This asks whether representation differences learned during midtraining transfer to action learning. A separate continuation branch retains the original RGB/flow auxiliary target during SFT; it measures the combined pipeline, not midtraining alone.

Action-free simulation clips are a controlled first source. Later human/robot videos can test practical transfer, but their flow is predicted, and their diversity/domain differs. Keep unlabeled clip counts, source mixture, clip windows, and compute equal. Include a no-midtraining arm. Do not give flow midtraining extra clips while calling the result a target-type effect.

Report fixed unlabeled data and label-generation cost beside each robot-data curve. “Robot demonstration efficiency” does not mean “total data efficiency” if a method uses a large external corpus or expensive geometry annotations.

## 4. Tasks and observation/controller setup

Select three task families with different relevant motion:

1. **PushCube-v1:** planar object manipulation, useful for spatial and appearance shifts.
2. **PickCube-v1:** grasp/lift/place motion, depth and contact sensitivity.
3. **StackCube-v1:** multi-object interaction and precision, useful for boundary and relational errors.

These are candidate IDs from the [official ManiSkill task list](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html). Confirm them and their success predicates against the pinned installed version. Use a custom object-family task wrapper for geometry holdouts if the standard cube task does not support the necessary asset variation. Do not claim object-category transfer from changing only cube texture.

Use one robot, one fixed action/controller convention, and one camera arrangement in the initial training distribution. A proposed observation setup is 256 × 256 external-camera RGB plus proprioception and the task goal. Fix wrist-camera inclusion across all arms; defer multi-view expansion until the basic comparison works. Avoid full simulator-state observations that make visual OOD shifts irrelevant.

Tentatively use 20 Hz control, a 16-action horizon, and execute the first 4 actions before replanning. Profile latency and controller stability before freezing these values. Control frames and action scaling must match the recorded demonstrations; ManiSkill defines action behavior through its [controller configuration](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html). An imported demonstration must be replayed or converted to the chosen controller, not assumed compatible from its action dimension.

The policy must receive an unambiguous goal in every arm, whether a visual marker, numerical goal, or instruction. Do not move or hide a goal marker under an “appearance-only” intervention in a way that changes the task information available. Use identical instructions across methods; do not introduce unseen language as another hidden OOD factor.

## 5. Demonstration collection, splits, and sample selection

Collect a master pool of successful expert trajectories with fixed recorded controller settings and sufficient state information for exact flow labels. Validate replay success and log filtering/failures. Use the same expert source for all target arms. ManiSkill provides [demonstration/replay tooling](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html), but whether an available dataset contains the needed states and asset distribution must be checked before using it.

Split by whole trajectory and scene/asset family **before** window extraction or rerendering. Sibling renderings of a trajectory remain in the same split. Separate:

- training pool drawn only from declared ID factors;
- an ID validation pool for tuning and checkpoint selection;
- an independent ID evaluation manifest;
- OOD evaluation manifests with held-out factors, never used for tuning;
- optional OOD expert trajectories for diagnostic prediction metrics, not training.

Use three independent dataset-subset seeds. For each seed, create a balanced random trajectory order and take nested prefixes for all N; R and F receive exactly the same IDs. At small N, balance meaningful scene/goal strata without selecting “easy” demonstrations. Save every selected ID. The first pilot can use one subset seed and two optimizer seeds; full comparisons should use three subset seeds and two optimizer seeds, with the same seed structure across arms. These replicate factors are not six independently collected task datasets.

A proposed fixed ID validation pool is 20 trajectories per task. This is extra labeled data: disclose it separately from N and also report N + validation trajectories as a total labeled-data budget. If that overhead overwhelms the lowest-N claim, fix hyperparameters on an independent development task and use fixed-step checkpoint selection without per-task labeled validation tuning.

Fit normalization on the selected training subset only, or use a declared independent calibration source available equally to all arms and count that source in the data budget. Never fit action or flow bounds using held-out/test scenes or the entire 256-demo pool for an N = 8 claim. Cache statistics separately by subset hash. Record how scale/bound choices vary with N.

## 6. Flow and RGB target construction

Reuse Experiment 1's persistent-point oracle, coordinate convention, visibility masks, raw predicted-track storage, and RGB displacement mapping. [Experiment 1](../01-tracker-and-vae-benchmark/README.md) establishes tracker quality and codec distortion before policy training.

For each observation anchor, query a fixed image grid and define cumulative displacement in that anchor camera. Flow pixels retain initial-query identities. Do not substitute a future-frame raster where identities collide or disappear.

R predicts true future RGB; F predicts channel-coded cumulative dx/dy/dz. Both target videos use the same spatial resolution, same number of future observations, same native VAE, and same latent layout. The current image is a separate conditioning input for both.

A tentative consistent target window is **16 actual future frames, t+1 through t+16**, padded by repeating the last frame to 17 for the codecs. Flow is still anchored at t even though its target sequence starts at t+1. RGB follows exactly the same timing. The initial conditioning frame is not included as an easy target in one arm while the other learns only future changes. Crop prediction metrics to the 16 real frames; document latent loss treatment because a temporally compressed cell may include both valid and repeated frames. Identical padding prevents this from being a method-specific advantage.

The repository's existing temporal offsets and target construction must be audited for this protocol. A configuration name alone does not prove these windows match. Add an explicit data-contract test that compares source timestamps, anchor IDs, padded frames, latent shapes, and action intervals across R and F.

For F-GT, GT trajectory existence continues through occlusion; visibility is not the same as label validity. For F-pred, save coverage and tracker confidence. Use the same action windows as R rather than discarding difficult demonstrations only from F. If flow is unavailable, mask that auxiliary contribution and retain the action example; report the effective amount of flow supervision. A matched window-retention RGB control can diagnose coverage effects.

For imperfect tracker confidence, do not treat a thresholded flow mask as a supervision-volume match automatically. Foreground-only or confidence-filtered training is an explicit ablation. Primary flow is the declared all-valid grid; record foreground/background proportions. RGB's default full-frame loss can be dominated by static background, so add a robot/object-region-weighted RGB control with clearly labeled privileged training masks. This tests whether motion geometry helps beyond simply concentrating loss on relevant regions.

## 7. Objectives, optimization, and compute fairness

Use the same flow-matching parameterization for each predicted stream. Let v be the target velocity under the chosen interpolation/noise convention:

```text
L_action = mean_valid ||vhat_action − v_action||²
L_aux    = mean_valid ||vhat_future − v_future||²
L_total  = L_action + lambda_aux L_aux
```

For R, future means RGB-VAE latents; for F, it means flow-RGB-VAE latents. This is diffusion/flow-matching target velocity, distinct from physical scene-flow velocity. Use mean reductions over valid elements rather than sums that depend on token count. Publish scalar losses and gradient norms; equal scalar weights do not guarantee equally influential auxiliary tasks.

Select lambda from a small identical grid, initially {0.1, 1, 10}, with the same ID-only tuning budget for R and F at a development condition such as N = 32. Then freeze it across N and OOD tests. Report per-arm selected values and the tuning budget. Include a shared-lambda sensitivity subset. Do not tune flow on OOD success and RGB on reconstruction loss.

### Two training budgets

The primary data-efficiency curve uses a common maximum optimizer-update budget, effective batch size, window-sampling schedule, learning-rate schedule, and validation/checkpoint rule. A provisional value is 20,000 updates with effective batch 8; it must be profiled and checked for reasonable convergence on a development condition before freezing. At small N this repeats demonstrations more often; disclose effective epochs. Pair sampled window IDs, augmentation draws, and noise seeds across R and F where possible.

A secondary fixed-epoch or equal-GPU-hour sweep checks whether a conclusion depends on repeated exposure or compute. Do not conflate fewer demonstrations with less training compute. Measure optimizer updates, total target tokens, accelerator hours, peak memory, and wall-clock throughput. Identical latent dimensions do not by themselves ensure identical observed runtime.

Use ID validation success as the checkpoint criterion when available, at the same evaluation cadence. Save the fixed final checkpoint too. Never select the best seed/checkpoint on OOD test results. Use equal numbers of hyperparameter trials and include their costs.

### Inference contract

At evaluation, all predictions start from the same current observations and **sampled/noisy future latents**, with no true future RGB or flow provided. Use the same sampler steps, random seeds, action extraction, chunk execution, and replanning schedule. R and F have equal future-stream dimensions at inference; include their real latency.

If a lower-latency action-only inference path is introduced after joint training, test it separately; removing future tokens changes the trained conditioning structure. GT-flow-conditioned inverse dynamics is an optional oracle ceiling and must never be counted as closed-loop policy success for F.

## 8. OOD evaluation matrix

Define disjoint training ranges/assets and held-out ranges/assets explicitly before running the main sweep. Randomly changing a seed within training ranges is ID variation, not OOD.

| Shift | Proposed holdout | What it tests | Control/limitation |
|---|---|---|---|
| Appearance | Disjoint textures/colors, backgrounds, lighting directions/intensities | Nuisance robustness | Preserve geometry, goals, visibility, and dynamics |
| Camera | Held-out azimuth/elevation/distance/FOV bins | Viewpoint generalization | Keep objects visible; supply the same calibration interface to all methods |
| Spatial layout | Held-out reachable start/goal regions and relative placements | Spatial extrapolation | Check expert feasibility; keep task semantics fixed |
| Geometry | Held-out meshes, shape families, or size intervals | Object generalization | Requires an explicit asset-capable task; report grasp/dynamics changes |
| Composition | Familiar individual factors in unseen combinations | Compositional generalization | Exclude those combinations from every training source |
| Dynamics | Held-out friction/mass/contact parameters | Physical adaptation | Harder and partially unobservable; report separately |
| Multi-shift | Appearance + camera, or camera + geometry | Stress beyond isolated factors | Not the primary attribution analysis |

Start with appearance, camera, and spatial-layout shifts. Geometry is important but requires careful task/expert construction; dynamics is a later extension. Cross-robot or unseen-task transfer is outside the initial experiment.

Use predefined severity levels. Example proposals, to be validated against the renderer and reachability constraints: train camera azimuth within ±10° of nominal and evaluate disjoint 20–30° and 40–50° bins; train one set of object sizes and evaluate disjoint larger/smaller intervals; split workspace into documented train and held-out regions with a buffer. Freeze actual numbers only after expert feasibility tests, before policy results. Camera shifts can change numeric anchor-frame flow targets, so transform GT correctly rather than expecting identical displacement channels across views.

### Paired counterfactual render tests

For appearance shifts, rerender identical state/action trajectories with different textures or lighting. Use these pairs for offline observation/representation sensitivity, target errors, and action consistency. The same physics under different render settings provides a clean nuisance intervention.

For closed-loop evaluation, reset to the same initial physical state under paired rendering settings and run each policy independently. Actions cause trajectories to diverge; do not claim the resulting rollouts have identical physics after the first action. Use synchronized reset seeds and record states so failures can be compared.

A held-out appearance that removes a visible goal or changes object identity semantics is not a pure nuisance shift. Reject such cases from the corresponding comparison or report them as a different information shift.

## 9. Evaluation and statistical analysis

Use the task's pinned simulator success predicate, fixed time limit, and equal controller/execution settings. Proposed evaluation sizes are 100 held-out ID episodes and 100 episodes per primary OOD family for each trained checkpoint. A smaller 25-episode smoke evaluation can validate the evaluator, but it is not the final result.

Save one record per model seed, evaluation condition, and reset. Include success, terminal reason, steps/time, object/goal distance, contact/collision or workspace violations where defined, inference latency, checkpoint hash, and sampler seed. Preserve videos for a predetermined subset and all selected failures.

### Primary summaries

1. Success versus log demonstration count, ID and each OOD family separately.
2. Area under the success curve over the predefined log-N grid, normalized by log-N range; report the numerical integration rule.
3. Minimum measured N reaching a predefined 80% ID-success target, or another threshold fixed before results. If a method never reaches it, report “not reached”; do not extrapolate a crossing.
4. Absolute OOD success and the ID-to-OOD success drop in percentage points.
5. Paired flow-minus-RGB differences at each N/condition and a predeclared low-data average, initially N = 8, 16, 32.

OOD retention S_OOD/S_ID is secondary and undefined or unstable when ID success is very low. A small generalization gap can mean uniformly poor performance, so always show absolute success.

Average across tasks equally for a macro summary, while preserving per-task curves. Report all optimizer/subset seeds rather than a best-run curve. Confidence intervals should account for trained-model and scene variation: use a hierarchical paired bootstrap over subset/training seeds and reset families, keeping matched conditions together. With only a few trained seeds, intervals are approximate; show seed-level points and avoid treating every episode as an independent retraining replicate.

Predeclare the primary contrast (F-GT minus R at low N for ID, and appearance/camera OOD) and present remaining families as secondary/exploratory. Avoid cherry-picking one of many tasks or severity levels. No claim of statistical significance is required for the initial pilot.

### Secondary diagnostics

- Held-out action error under teacher-forced observations, alongside closed-loop success.
- Flow forecast physical EPE/velocity error and RGB reconstruction/prediction metrics against their own targets, with codec reconstruction floors from Experiment 1.
- Prediction accuracy on foreground, object boundaries, and occlusion intervals.
- Observation perturbation/action consistency on paired rerenders.
- Sensitivity to tracker label quality, missing coverage, and motion-RGB clipping.
- Failure taxonomy: localization, approach/grasp, contact, placement, camera confusion, and loss of target identity.

RGB and flow auxiliary losses are in different target domains; never rank policy quality by their raw numerical values. A strong prediction model may still fail control. A policy may improve without visibly better forecasts.

## 10. Controls that determine interpretation

**Core controls:** action-only baseline; same episode IDs; same backbone/VAE; same future timestamps and tokens; GT versus predicted flow; ID-only tuning; fixed OOD manifests; no future leakage; comparable inference cost; normalization fitted without test data.

**High-value extensions:**

- Region-weighted RGB to test whether foreground focus explains flow gains.
- RGB + 2D optical-flow supervision, or depth prediction, to distinguish 3D motion from generic extra structure. Match extra-stream capacity or report it.
- Static/zero flow, or flow shuffled across unrelated episodes, as a small sanity ablation. These are intentionally incorrect objectives and not substitute baselines for meaningful flow.
- Flow label-noise sweep using controlled perturbations grounded in Experiment 1, with equal action examples.
- A comparison using identical domain randomization for both arms. If augmentation closes the gap, flow's apparent advantage may be replaceable nuisance robustness.
- Matched small-backbone or from-scratch runs to examine pretrained-RGB prior effects.
- RF jointly supervising RGB and flow to test complementarity, with its additional compute reported.

Texture can encode object category, goal, or affordance. Include at least one later task where appearance is relevant if claiming broad superiority, because a motion-only target might remove useful semantics. Geometry supervision is not automatically the best representation for every task.

## 11. Training-stage and evaluation interfaces

Keep this experiment inside the existing compact package. Extend configurations/manifests and reuse train, export, policy, and simulation evaluation; do not create a second trainer.

```text
immutable episode manifest
  -> subset/split manifest (whole trajectories and scene families)
  -> matched RGB and flow windows
  -> separate VAE target caches (same codec signature)
  -> same joint-policy trainer with target-content selection
  -> immutable exported bundle
  -> paired ManiSkill rollout manifests
  -> result table, curves, failure videos, analysis
```

For 3B, add an explicit midtraining checkpoint identity followed by a robot SFT run identity. Record exact source mixtures and which trainable parameters are inherited. A stage handoff must reject a changed codec/action convention unless deliberately converted. R and F cannot silently reuse each other's content caches, though their shapes match.

Required additions include episode-level subset selection, split/asset holdouts, matched target-window checks, a single generic future-stream comparison configuration, OOD task/render wrappers, paired reset manifests, success-curve aggregation, and resume-safe run bookkeeping. Existing target-selection support is a starting point, not proof that these scientific controls are already implemented.

Runtime observations and training labels must have separate schemas so GT flow/cameras cannot leak into evaluation. Add a test that the policy input bundle contains only allowed observations, and another that inference uses sampled latents rather than cached future labels.

## 12. Artifacts and result visualizations

For every run, save the episode/subset manifests, split and OOD definitions, normalization hashes, target cache identity, code/checkpoint revisions, hyperparameters, stage lineage, training exposure/compute, chosen checkpoint rule, exported policy, and rollout records. Keep raw simulator/label arrays addressable without duplicating them across model seeds.

Large checkpoints/videos belong in ignored run/data storage or on the GPU host. Markdown summaries go in this experiment's `results/ai/` or `results/user/`; planning and debugging observations go in `ai_notes/` or `user_notes/`.

Required plots:

1. ID and OOD success-versus-N curves, with matched axes, seed points, and uncertainty.
2. Flow-minus-RGB paired effect versus N, not just two overlapping curves.
3. OOD severity curves and a task × shift heatmap at fixed low/high N.
4. Success versus total labeled-data budget and versus GPU-hours.
5. GT-flow versus predicted-flow policy performance, annotated with label EPE/coverage.
6. Counterfactual appearance pairs with matched initial states and action differences.
7. Side-by-side rollout failure gallery with synchronized timestamps and explicit success predicates.
8. Optional midtraining/no-midtraining curves with fixed external data budgets.

A result table should contain task, target arm, backend, stage regime, N, extra validation/unlabeled data, subset/optimizer seed, shift/severity, evaluation count, success, ID–OOD gap, steps, latency, GPU-hours, label coverage, and checkpoint/artifact links. Never present tentative expectations as measured bars or curves.

## 13. Practical run plan and cost

**Pilot:** one task, A/R/F-GT, N = 16 and 64, one subset seed, two optimizer seeds: **12 training runs**. Evaluate ID, appearance, and camera shifts. This checks the trainer/evaluator and the direction of an effect; it does not settle generalization.

**Main 3A:** three tasks × six N values × three arms × three subset seeds × two optimizer seeds: **324 training runs**, before hyperparameter tuning. This can be expensive even with LoRA. Profile two representative runs and estimate training and rollout cost before committing to the full matrix.

**Practical-flow extension:** F-pred at N = 16, 64, 256 on all three tasks with the same seed structure: **54 additional runs**. Other ablations and 3B are separate budgets. If resources are constrained, reduce tasks/N cells/replicates explicitly; preserve method pairing and report the reduced scope. Do not replace independent trained seeds with more reset episodes and claim the same uncertainty.

Recommended order:

1. Validate labels/codecs via Experiment 1; freeze the target timing and observation contract.
2. Establish expert replay, ID/OOD manifests, and expert feasibility under each shift.
3. Run a tiny overfit/leakage test and confirm identical R/F token shapes, parameter counts, and action inputs.
4. Run the 12-run pilot and check whether the training budget is reasonable using ID development data.
5. Freeze the main protocol, hyperparameters, subsets, shifts, checkpoint rule, and statistics policy.
6. Execute the selected full sweep with resume-safe jobs and paired evaluations.
7. Add F-pred and the most informative controls before making a practical-label claim.
8. Run 3B only after 3A is interpretable, then selectively replicate with Wan.

No fixed GPU-hour estimate is asserted before hardware/checkpoint profiling. A negative result is useful: the experiment should identify whether geometry labels fail to help, whether tracker/codec distortion erases a GT advantage, or whether simple RGB augmentation provides the same benefit.

## 14. What a defensible conclusion would look like

Evidence for data efficiency requires higher closed-loop success with fewer unique action-labeled trajectories under the specified compute/tuning budget. Evidence for OOD robustness requires higher absolute success on independently defined shifts, not only a smaller ID–OOD gap. Evidence for a deployable flow pipeline requires the benefit to survive predicted labels and realistic inference.

If F-GT improves but F-pred does not, the result identifies a label-quality bottleneck. If flow helps appearance shifts but not geometry/dynamics, narrow the claim to nuisance robustness. If region-weighted or augmented RGB closes the gap, describe that control rather than attributing all benefit to 3D motion. If only 3B helps, the evidence concerns transfer from midtraining rather than auxiliary robot SFT alone.

The tentative experiment is successful when these alternatives can be distinguished with traceable runs, even if flow supervision does not outperform RGB prediction.
