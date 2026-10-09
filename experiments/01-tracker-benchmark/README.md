# Experiment 1: simulator-grounded 3D point tracking

**Status:** proposed, not executed. Revised 2026-10-08 to separate VAE reconstruction into Experiment 2.

## 1. Question and scope

How accurately do SpaTrackerV2, independent 3D trackers, and 2D trackers lifted with depth recover simulator material-point trajectories under object motion, camera motion, and occlusion?

This experiment generates controlled Kubric or ManiSkill clips with GT camera calibration and persistent 3D point trajectories, runs the tracking baselines, and saves their complete raw and canonical outputs. It ends at physical trajectory evaluation. RGB displacement rendering and VAE reconstruction belong to [Experiment 2](../02-motion-vae-reconstruction/README.md).

GT geometry is privileged supervision for labeled input regimes; native monocular pipelines are reported separately. The experiment makes no policy-success or downstream learning claim. All trackers are frozen inference baselines: there is no Wan/DiT fine-tuning here.

## 2. Dataset: a small paired pilot, then a larger benchmark

Start with **12 clips**: six motion archetypes, each rendered with a fixed camera and an orbiting camera. Replay the same object motions and assets across the camera pair, preferably with identical initial camera pose. This directly tests whether camera motion creates false object motion.

| Archetype | Scene behavior | Main diagnostic |
|---|---|---|
| Static scene | Textured objects and background remain fixed | False flow; camera-motion cancellation |
| Translation | One object translates laterally and in depth | Metric displacement, depth scaling, speed |
| Rotation | A rigid object rotates about its own center | Surface-point identity and nonuniform motion |
| Articulation | Two or more rigid links move relative to one another | Link identity, boundaries, local deformation appearance |
| Occlusion and return | A tracked object passes behind an occluder and reappears | Persistence, visibility, reacquisition |
| Depth/speed/view stress | Motion crosses depth discontinuities or exits the view | Fast motion, discontinuities, out-of-view handling |

Pilot defaults are 256 × 256 RGB, 20 Hz, and 33 frames. The sequence includes the query frame; its elapsed duration is 32/20 = 1.6 seconds. Use a 16 × 16 query grid for debugging, then target a **64 × 64 grid, Q = 4,096**, for the main pilot. If profiling requires 32 × 32, freeze that setting for all methods before reading comparative results. Do not mix densities in one ranking.

A proposed expansion is **90 clips**: six archetypes × five independent test seeds × three cameras—fixed, translating, and orbiting—with 65 frames. Report the pilot as a diagnostic case study; it is too small to support strong statistical ranking claims. The expansion supports paired estimates across scene families.

Generate a separate GT-only calibration split, initially 12 clips with disjoint seeds, to fit RGB displacement bounds. Test clips must never determine normalization bounds. Record seeds and asset identities. For stronger generalization claims, hold out asset families and vary textures, lighting, distances, speeds, and object sizes. Kubric-style scenes may resemble tracker training data; newly rendered seeds alone do not prove out-of-distribution evaluation.

### Simulator choice

Use [Kubric](https://github.com/google-research/kubric) for MOVi-style scenes when its renderer/generation environment is available. Its scene annotations are suitable starting points for camera and object-coordinate supervision. Build the exact point oracle described below rather than assuming an exported flow field gives persistent material-point trajectories.

A ManiSkill/SAPIEN generator is a practical alternative for controlled rigid-body and articulated scenes. Render RGB, depth, segmentation, camera calibration, and actor/link poses together. Use scripted deterministic motions first, so the benchmark does not depend on learning a robot policy or obtaining task demonstrations. If the camera follows a sim robot, save its actual render-time pose on every frame. Ground truth should still describe the same surface points, not merely whatever is visible at a pixel later.

Kubric and ManiSkill should be separate dataset strata. Camera, material, renderer, and motion differences can change performance; do not pool them without showing per-source results. Disable motion blur initially so depth, RGB, pose, and surface geometry correspond to a single instant. Add motion blur or sensor noise as named stress tests later.

## 3. Ground truth: persistent material points and explicit visibility

At the initial query frame, cast a ray at each prescribed image location. Save the first visible surface hit and its identity:

- actor/object and rigid-link ID;
- mesh triangle and barycentric coordinate, or equivalent stable local material coordinate;
- initial world point and axial depth;
- query pixel, query time, grid index, and permanent query ID.

For a rigid link, transform the same local point by its recorded link pose at each time. For a mesh, preserve the material point rather than reselecting a nearest surface. Begin with rigid and articulated geometry. Deformable scenes require valid per-vertex or material-coordinate supervision and should be a separate extension.

This oracle continues through occlusion and outside the image. Optical-flow integration is not an acceptable replacement: it can drift, change surface identity, and lose points during occlusion.

**Visibility is a separate time-varying label.** A point is visible when it lies in front of the camera, projects into the image, and is the first intersected surface at that ray, within a declared numerical tolerance. Use material/instance identity and a ray/depth check together. Save separate masks for:

- initial query validity;
- valid material trajectory;
- positive camera depth;
- in-frame projection;
- visible versus occluded while in view;
- out of view versus behind camera.

Do not broadcast the initial validity mask across time and call it visibility. A point may have valid GT geometry while being invisible. An initially empty query slot remains invalid throughout its trajectory.

Use exact raycasting at fractional query coordinates, or choose integer pixel centers and matching depth samples. Combining fractional UV with nearest-pixel depth or segmentation creates an avoidable GT error near boundaries. Prefer integer-centered grid queries for the first implementation.

### Ground-truth acceptance checks

Before running a model, require these checks:

1. Reproject GT points through recorded camera calibration; initial points agree with query locations within 0.25 pixel. Investigate any systematic offset.
2. Static material points have zero world displacement and zero anchor-frame displacement under moving cameras, to numerical precision. Set the numerical tolerance after validating simulator precision, approximately 10⁻⁵ m for a meter-scale analytic fixture.
3. A known 1 cm translation produces a 1 cm recovered displacement with the expected sign on each axis.
4. Camera rotation, camera translation, and object motion can each be enabled independently.
5. Occlusion labels change when the occluder crosses a query ray, while the GT material trajectory remains continuous.
6. Projected visible GT depth agrees with the rendered surface depth under the documented convention.
7. Timestamps, poses, segmentation, and RGB identify the same render step.

Save the check report and annotated RGB overlays. A benchmark built on incorrect camera conventions is unusable regardless of tracker accuracy.

## 4. Coordinates and the motion representation

The canonical convention is meters, right-handed computer-vision camera coordinates: x right, y down, z forward. Save camera transforms as **camera-to-world**, Eₜ = [Rₜ, tₜ]. Save the original simulator convention too. Explicitly convert OpenGL-style axes and world-to-camera matrices.

Depth used for lifting is axial camera z. If a renderer exports distance along a ray, convert using z = range / ||K⁻¹[u,v,1]ᵀ||. Never silently treat ray range as z. Include depth convention, units, near/far handling, and invalid values in metadata.

For initial frame a and world-space point Pᵂₜ, define its coordinates in the fixed query camera:

```text
Pᶜᵃₜ = Eₐ⁻¹ Pᵂₜ
Dₜ = Pᶜᵃₜ − Pᶜᵃₐ = Rₐᵀ(Pᵂₜ − Pᵂₐ)
```

D is **cumulative displacement from the query frame**, not frame-to-frame velocity. Static background has zero D even when the camera moves.

For a tracker that outputs per-frame camera-space points and estimated camera-to-world poses:

```text
P̂ᶜᵃₜ = Êₐ⁻¹ Êₜ P̂ᶜᵗₜ
D̂ₜ = P̂ᶜᵃₜ − P̂ᶜᵃₐ
```

Validate native coordinate conventions against upstream code and fixtures. Do not compute P̂ᶜᵗₜ − P̂ᶜᵃₐ directly across moving camera frames. For world-space outputs, document the world gauge and map it into the common initial camera basis.

If a model resizes or crops input, save the image-coordinate transform A and use K′ = AK. Pixel-center-aware resizing follows u′ = s(u + 0.5) − 0.5 before any crop offset. Bring predictions back into original coordinates before evaluation. Preserve native outputs alongside canonical ones.

### Scale ambiguity

Known-geometry runs can be evaluated directly in meters. Native monocular runs require a separate scale policy. Report unaligned metric error only if their output units support that interpretation; also report native reprojection error.

A declared diagnostic may calibrate one scalar using initial GT depths, for example s = median(Z_GT,a / Z_pred,a) over a fixed valid initial point set. Apply that same scalar to point coordinates and camera translations for the entire clip. Label this **initial-GT-scale-calibrated**, because it uses privileged information. Define the point set before evaluation and record the resulting scalar.

Do not use per-frame scale fitting, per-frame rigid alignment, or whole-trajectory GT alignment for the primary motion scores. Those operations can remove the very errors being studied. A full-trajectory Sim(3) alignment may appear only as a clearly labeled oracle diagnostic.

When reconstructing trajectories from displacement labels, using P_GT,a + D̂ₜ gives an **anchor-corrected motion diagnostic**. It does not measure the model's initial absolute 3D accuracy. Report initial depth/position error separately and preserve native absolute-state scores.

## 5. Trackers, frontends, and fairness

The mandatory comparison includes GT, SpaTrackerV2, and a 2D tracker with depth lifting. Add at least one independent 3D tracker, ideally TAPIP3D and DELTA, subject to checkpoint availability and installation. Include native and matched-geometry regimes separately.

| Method family | Inputs/regime | Purpose |
|---|---|---|
| Simulator GT | Exact material trajectories | Reference and codec floor |
| Zero-motion control | All valid points have D = 0 | Exposes misleading codec-only rankings |
| GT 2D + GT depth + GT camera | Exact projected UV; visible points only | Lifting and coordinate closure |
| GT 2D + predicted depth + GT camera | Exact UV; shared depth | Isolates depth-induced error |
| CoTracker3 + GT depth + GT camera | RGB tracks with exact geometry | Isolates 2D correspondence error on visible points |
| CoTracker3 + shared depth + GT camera | Predicted UV/depth | Depth contribution under fixed camera geometry |
| CoTracker3 + shared depth + shared camera | Predicted UV and frontend geometry | Complete lifted pipeline |
| SpaTrackerV2 native RGB | Its native geometry frontend | Complete native pipeline |
| SpaTrackerV2 + GT geometry | RGB, GT depth/K/poses where supported | Tracking with privileged geometry |
| SpaTrackerV2 + shared geometry | Common cached frontend, if supported | Matched frontend comparison |
| TAPIP3D native / GT geometry | Native RGB pipeline and known-geometry branch | Independent 3D tracking architecture |
| DELTA native / GT depth | Native depth and matched depth branches | Dense/coarse-to-fine alternative |

[SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) provides RGB and RGBD/camera workflows. [TAPIP3D](https://github.com/zbw001/TAPIP3D) represents persistent 3D tracking in a world-space feature cloud and supports known geometry. [DELTA](https://github.com/snap-research/DELTA_densetrack3d) provides dense and sparse 3D tracking workflows. Use sparse query APIs when available so all methods receive the same points. These descriptions motivate the comparison; performance remains to be measured.

Use [CoTracker3](https://github.com/facebookresearch/co-tracker) as the first 2D baseline. Pin a specific checkpoint variant and record its training domain. An optional second baseline is TAPIR/BootsTAPIR from [TAPNet](https://github.com/google-deepmind/tapnet). More baselines are useful after the core benchmark is verified, not at the cost of delaying GT and codec validation.

### Architectural interfaces and error paths

The benchmark should expose each architecture through its native outputs instead of flattening every method into an identical black box.

**2D tracker plus geometry:** the tracker consumes RGB and queries and returns UV trajectories and visibility/confidence. The lifting stage samples a depth field and uses camera calibration:

```text
P_camera,t = z_t(u_hat,v_hat) K_t^-1 [u_hat,v_hat,1]^T
P_world,t = E_t P_camera,t
```

Its errors come from correspondence, depth sampling, intrinsic calibration, and extrinsics. Save each intermediate so GT substitutions can test those paths independently. Bilinear depth sampling can blend foreground/background surfaces; use a declared policy and compare nearest sampling at boundaries. Neither sampling choice provides the hidden material point's depth during occlusion.

**Direct 3D trackers:** their native interface returns 3D trajectories, often using a geometry frontend and sometimes estimated cameras. Retain the direct trajectory output rather than projecting it and rebuilding it through a depth map. An output labeled “3D” still needs a verified coordinate frame and scale. Shared depth/pose inputs make a correspondence-focused comparison possible only where the architecture genuinely supports them. Camera estimation and trajectory estimation may interact; replacing one output with GT is an intervention, not proof of an additive error budget.

### Two comparison groups

**Native pipelines** use each method's intended depth/camera frontend. This measures the practical pipeline a user would obtain, but differences include frontend quality.

**Matched frontend or privileged geometry** uses common cached depth, intrinsics, and camera poses wherever an upstream method actually supports them. These runs isolate correspondence/trajectory estimation more closely. Show GT-input runs in a distinct table; they are not deployable RGB-only baselines.

A convenient shared frontend is the geometry frontend already used by the pinned SpaTrackerV2 checkout, provided its output contract is verified. Cache it once per clip. TAPIP3D's native MegaSAM/MoGe workflow remains a native-pipeline result unless intentionally replaced. Do not silently equate different frontends or pretend a model accepted supplied intrinsics when its internal estimator ignored them.

For the 2D baseline, backproject predicted UV at time t with Dₜ(u,v), then transform by Eₜ. GT depth at an occluded query's projected location belongs to the visible occluder, not the tracked hidden material point. Consequently, **GT 2D + GT depth is an oracle only for visible points**. Continue saving naive lifted outputs through occlusion if useful, but identify this failure mode. Giving a depth lift the hidden point's GT depth would leak the desired answer.

Optionally run a depth × camera factorial using GT/predicted depth and GT/predicted poses. These interventions reveal sensitivities, but their effects need not add linearly. For 3D methods, converting raw predicted camera-space points using GT poses is a diagnostic extrinsics intervention, not a replacement native output.

### Adapter rules

- Identical query IDs, times, pixel locations, and input RGB across methods.
- Save the complete query/support-point set. Joint trackers can change behavior when query count or chunking changes; record chunks and do not assume invariance.
- Preserve native 2D points, 3D points, depth, cameras, confidence, visibility, and uncertainty when available.
- Preserve NaNs and explicit finite masks in raw artifacts. Fill values only for rendering. A missing prediction is not zero error.
- Do not reject difficult points using model confidence before computing the primary GT-conditioned geometry scores.
- If dense output must be sampled, declare the interpolation rule and map predicted pixels to the original query coordinates. Never select predictions by closest GT 3D location.
- Save checkpoint/repository revisions, SHA-256 hashes, preprocessing, precision, random seed, elapsed time, peak GPU memory, and frontend privileges.

## 6. Tracking metrics

Compute errors on the immutable GT query set. Exclude the query frame from primary motion scores; its displacement is identically zero. Show it separately as an initialization check.

| Metric | Definition/use |
|---|---|
| Absolute 3D position EPE | ||P̂ₜ − P_GT,ₜ|| in a documented common frame; meters |
| Cumulative displacement EPE | ||D̂ₜ − D_GT,ₜ||; primary motion fidelity |
| Axis errors | MAE/RMSE for dx, dy, dz; identifies depth-specific failure |
| Error distribution | Mean, median, p90, p95; tails matter at occlusions/boundaries |
| Metric success thresholds | Fractions within 1, 2, 5, and 10 cm; custom physical PCK |
| Depth-normalized error | EPE / initial GT depth; complements metric scale |
| Reprojection EPE | Pixels under GT K/cameras for compatible physical 3D outputs |
| Velocity error | ||(D̂ₜ−D̂ₜ₋₁)/Δt − (D_GT,ₜ−D_GT,ₜ₋₁)/Δt|| |
| Acceleration error | Difference of finite-difference velocities divided by Δt |
| Static false motion | Predicted displacement magnitude for static material points |
| Occlusion recovery | Error at first reappearance and subsequent frames |
| Coverage | Finite prediction fraction on the fixed eligible GT set |
| Visibility quality | Accuracy/F1 and optional calibration when outputs permit |

Report geometry separately for visible points, occluded-but-in-view points, out-of-view points, foreground objects, static background, and motion/depth-boundary points. Average per object as well as per clip. Pixel-weighted background dominance must not hide foreground failure. Direction angles are meaningful only above a predeclared motion threshold; use 5 mm cumulative GT motion initially and show threshold sensitivity.

For missing predictions, publish finite-only EPE **together with coverage**, and count missing predictions as failures in threshold success rates. An all-method common finite intersection is a secondary paired diagnostic, not the primary mask. Occlusion predictions should not remove model errors from the GT-visible denominator.

Include the official TAPVid-3D metrics as secondary benchmark-compatible scores, importing the pinned upstream implementation rather than recreating a similarly named approximation. Its published evaluation includes 3D-AJ, APD, and OA with scale/occlusion handling. Preserve its required coordinate and alignment policy and report that policy beside the scores. Our centimeter thresholds and fixed-scale EPE are additional metrics, not renamed TAPVid-3D scores. See the [official evaluation documentation](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d).

Where estimated cameras exist, also report rotation error, relative pose error, and translation error after initial-gauge correction and the declared fixed scale. Plot camera-motion magnitude against static false flow. Whole-trajectory camera alignment can be a diagnostic but must not erase drift in the primary analysis.

### Aggregation and uncertainty

Compute per-clip and per-object summaries first. Use paired comparisons on the same scenes. Bootstrap independent scene families, keeping fixed/orbit/translation camera siblings together in each bootstrap sample. Do not treat thousands of correlated points as independent experimental replicates. Report sample counts, confidence intervals, and failed clips. Pilot figures should show individual cases; avoid significance claims from one seed per archetype.

## 7. Saved artifact boundary

Export one immutable bundle per clip/method/regime/anchor with query IDs, initial grid coordinates, timestamps, native/canonical 3D trajectories, cumulative anchor-camera displacement, finite masks, GT validity/visibility, units, camera transforms, scale policy, and provenance. Preserve native UV, depth, confidence, and estimated cameras when provided. Never overwrite invalid coordinates with zero in the raw scientific artifacts.

Publish GT as soon as a clip passes oracle validation; publish each method bundle when its own validation succeeds. Use hashes and atomic completion markers, so Experiment 2 can process ready artifacts without waiting for the full tracking matrix. A GT-only reconstruction pilot can begin after the first valid GT clip. Missing tracker outputs remain explicit missing jobs.

GT/canonical flow has shape [T,Q,3], with fixed query ordering. Metadata identifies RGB/depth/camera input hashes, checkpoint/upstream revisions, preprocessing, supplied GT fields, hardware/dtype, runtime, and GPU memory. Experiment 2 consumes this exact contract and never reruns tracking merely to render a different representation.

## 8. Figures and milestones

Deliver paired fixed/orbit camera comparisons, per-method displacement EPE and coverage, visible/occluded and foreground/background breakdowns, per-axis and velocity errors, time curves, and synchronized RGB/3D track overlays. Use identical query IDs, camera views, physical axis limits, and color scales. Include failed methods/clips and their causes.

Start with analytic coordinate/visibility fixtures and two rendered clips. Then execute the 12-clip pilot; expand to 90 clips only after the oracle and adapters are validated. The first debug grid is 16×16; the proposed primary grid is 64×64, with a frozen 32×32 fallback if profiling requires it. Methods never get selectively easier queries to improve their score.

The tracker worker initially owns GPUs 0/1. Independent clip/model jobs run one process per device; no distributed inference is required. Optional legacy tracker builds must pass B300 kernel acceptance before results are treated as valid.

## 9. Relation to Experiments 2 and 3

Experiment 2 evaluates representation and codec distortion using these saved GT/predicted trajectories. Experiment 3 tests policy data efficiency/OOD performance using independently split robot demonstrations and the validated label/codec pipeline. Reuse code and conventions; do not leak this experiment's held-out evaluation scenes into downstream training datasets.
