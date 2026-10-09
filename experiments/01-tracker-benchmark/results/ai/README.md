# Experiment 1: simulator 3D material tracking — measured results

## Motivation and hypotheses

This experiment measures frozen tracker outputs against simulator material-point trajectories, with known camera motion and object poses. The tracked quantities are cumulative 3D displacement, visibility, and physical derivatives. The saved tracks also supply inputs to the separate motion-image VAE reconstruction experiment.

The hypotheses specified for measurement are: (H1) fixed and moving cameras produce different errors on paired object trajectories; (H2) frozen 3D trackers and 2D tracking followed by depth lifting produce different 3D displacement errors; (H3) measurements differ between visible scene points and visible foreground objects; and (H4) changing the number of jointly queried points changes predictions at the same material IDs. No hypothesis verdict or interpretation is included here.

## Experimental setup and data

Data were collected on October 9, 2026 UTC (October 8 Pacific) using actual SAPIEN 3.0.3 rendering. Each clip has 33 RGB frames, 256 × 256 pixels, at 20 Hz; frame 0 to frame 32 spans 1.6 seconds. Axial camera depth, camera intrinsics, camera-to-world transforms, actor poses, material IDs, exact material trajectories, and visibility are saved with every GT bundle. Rigid box surfaces form the scene assets. Object motion and camera paths are scripted; the linked-box condition uses prescribed rigid-link poses rather than a native robot joint simulation.

There are six conditions, each rendered with a fixed camera and an orbiting camera:

1. Static scene.
2. Object translation.
3. Rigid rotation.
4. Scripted articulated links.
5. Occlusion and return.
6. Depth, speed, and out-of-view stress.

The executed data regimes are:

| Regime | Test clips | Separate calibration clips | Query slots | Appearance |
|---|---|---|---|---|
| Primary textured64 | 12 | 12 | 64 × 64 = 4,096 | Seeded surface textures |
| Matched textured16 | 12 | 0 | 16 × 16 = 256, exact dense material-ID subset | Same primary RGB frames |
| Original debug16 | 12 | 12 | 16 × 16 = 256 | Uniform-color surfaces |


This totals 60 saved GT bundles. Calibration seeds are distinct from the test seeds. The primary test has one seed per scene condition, paired between camera modes. Initially invalid grid slots remain invalid; the first listed primary test clip has 2,104 valid material queries. The static-scene primary clip has 2,131. Calibration clips were published for Experiment 2; they are not included in the tracker test averages below.

## Ground-truth construction and coordinate convention

A query originates at frame 0. Its camera ray passes through the renderer pixel center, using `(u + 0.5, v + 0.5)` and the saved intrinsics. Exact ray-box intersection determines the first visible material surface and its actor-local coordinate. Applying each saved actor pose to that same local coordinate produces its trajectory. GT material identity and position continue through occlusion and departure from the image. Visibility is computed separately from in-frame, positive-depth and first-surface ray checks.

Let `P_t` be the material point in world coordinates and `R_0` the initial camera-to-world rotation. The canonical saved target is `D_t = R_0ᵀ(P_t − P_0)`, in meters. Axes are initial-camera x right, y down and z forward. This is a fixed initial-camera basis; it is not a time-varying camera-coordinate position. Static world material points have zero canonical flow even when the camera moves.

Saved GT validation records include initial reprojection errors, static-background flow, renderer-versus-exact depth errors, valid-query counts and GT visibility fractions. Initial reprojection maximum is approximately `2.5914 × 10⁻⁷` pixels and static-background flow maximum is exactly zero. Per-bundle measured values and SHA256 hashes are in the [artifact ledger](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/artifact-ledger.json).

## Frozen methods and supplied inputs

All tracker weights are frozen. There is no tracker fine-tuning, diffusion training, policy training or robot policy evaluation in Experiment 1.

| Method | Saved checkpoint | Geometry supplied for the primary measurement |
|---|---|---|
| CoTracker3 offline | `scaled_offline.pth` | Predicted 2D tracks lifted with rendered GT axial depth, GT intrinsics and GT camera poses |
| SpaTrackerV2 Offline | `SpatialTrackerV2-Offline` | Rendered GT depth, GT intrinsics and GT camera poses |
| DELTA sparse 3D predictor | `densetrack3d.pth` | Rendered GT depth and GT intrinsics; predicted camera-space points transformed with GT camera poses |

CoTracker lifting samples the front-surface depth at its predicted image location. Nonpositive or missing depth creates missing geometry rather than an accepted 3D prediction. Raw tracker arrays and raw visibility outputs are retained separately from canonical flow. DELTA checkpoint loading recorded zero missing and zero unexpected parameter keys.

Three controls are included: zero 3D motion; GT image coordinates lifted using rendered nearest front depth; and GT image coordinates intersected with the exact front ray. The front-depth controls sample the current visible surface; they do not receive the hidden tracked material's depth during occlusion.

The compute node had eight NVIDIA B300 GPUs; Experiment 1 used physical GPUs 0 and 1. The verified CUDA runtime used Torch 2.12 / CUDA 13.0. Tracker and rendering environments were isolated. The corrected SpaTracker run uses a float32 outer inference path with supported SDPA/math fallback; some internal attention computations retain their upstream precision. This is one architecture, not an additional independent tracker.

## Metric definitions

Frame 0 is excluded from displacement-error averages. Visible EPE is the mean Euclidean error `‖D̂_t − D_t‖₂` over GT-visible eligible queries with finite predictions. Its companion **coverage** is the fraction of GT-visible eligible queries with finite geometry. **PCK at 1 cm** is the fraction of all GT-visible eligible queries whose 3D flow error is below 1 cm; missing predictions count as failures. Tracker-reported visibility does not replace GT visibility in this mask.

Each clip is averaged first; the tables then average clip means with equal clip weight. Fixed and orbit columns each contain six clips. The overall columns contain twelve. The visible scene metric gives every eligible visible point equal weight within a clip. Foreground-point EPE restricts to foreground materials. Foreground-object macro EPE instead gives each foreground actor equal weight within a clip before averaging clips.

Axis MAE uses all valid finite trajectory samples after frame 0, including occluded and out-of-view materials. Velocity error uses finite consecutive valid samples and `Δt = 0.05 s`; acceleration error uses finite triples. These derivative masks are not the visible-only EPE mask.

## Primary textured64 measurements

| Method | Fixed EPE cm | Orbit EPE cm | Overall EPE cm | PCK <1 cm % | Visible coverage % |
|---|---|---|---|---|---|
| CoTracker3 + GT geometry | 0.627 | 2.637 | 1.632 | 64.71 | 99.947 |
| DELTA + GT geometry | 0.708 | 5.189 | 2.949 | 51.72 | 100.000 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | 2.379 | 3.336 | 2.857 | 88.87 | 100.000 |
| GT UV + exact front ray | 0.000 | 0.000 | 0.000 | 100.00 | 100.000 |
| GT UV + rendered front depth | 0.036 | 0.512 | 0.274 | 96.06 | 99.960 |
| Zero motion | 1.897 | 1.931 | 1.914 | 95.96 | 100.000 |


Full precision source: [per-clip measurements](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/per_clip.csv).

| Method | Foreground-point EPE cm | Foreground-object macro EPE cm |
|---|---|---|
| CoTracker3 + GT geometry | 1.862 | 3.893 |
| DELTA + GT geometry | 4.767 | 6.401 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | 22.000 | 21.282 |
| GT UV + exact front ray | 0.000 | 0.000 |
| GT UV + rendered front depth | 0.708 | 0.667 |
| Zero motion | 25.141 | 22.289 |


| Method | dx MAE cm | dy MAE cm | dz MAE cm | Velocity EPE m/s | Acceleration EPE m/s² |
|---|---|---|---|---|---|
| CoTracker3 + GT geometry | 1.614 | 0.463 | 2.568 | 0.1455 | 4.5680 |
| DELTA + GT geometry | 1.231 | 0.599 | 2.534 | 0.0633 | 1.2711 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | 2.806 | 0.600 | 1.349 | 0.0658 | 0.9202 |


Foreground and derivative source: [foreground and physical measurements](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/foreground_and_physical_metrics.csv).

### Per-condition visible EPE

All numbers below are centimeters, with the same finite-conditional visible mask.

| Clip | CoTracker3 + GT | DELTA + GT | SpaTrackerV2 + GT |
|---|---|---|---|
| articulated_links_fixed | 1.325 | 0.677 | 3.072 |
| articulated_links_orbit | 3.546 | 3.641 | 4.708 |
| depth_speed_out_of_view_stress_fixed | 0.722 | 1.390 | 4.997 |
| depth_speed_out_of_view_stress_orbit | 2.846 | 4.565 | 6.593 |
| object_translation_fixed | 0.659 | 0.582 | 2.162 |
| object_translation_orbit | 2.838 | 4.729 | 2.541 |
| occlusion_and_return_fixed | 0.498 | 0.452 | 3.206 |
| occlusion_and_return_orbit | 2.641 | 4.588 | 3.913 |
| rigid_rotation_fixed | 0.431 | 0.629 | 0.728 |
| rigid_rotation_orbit | 2.103 | 8.264 | 1.734 |
| static_scene_fixed | 0.127 | 0.519 | 0.106 |
| static_scene_orbit | 1.849 | 5.351 | 0.527 |


### Measured plots and scene videos

![Visible flow error and coverage](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/visible_geometry.png)

![Object-translation orbit error over time](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/translation_orbit_time.png)

![Saved object-translation orbit comparison frame](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/object_translation_orbit.png)

[Open the synchronized interactive 3D material-track comparison](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/experiment-01-interactive/index.html). The standalone viewer uses the saved object-translation/orbit primary clip, all initially valid foreground queries and 250 deterministic background queries (423 total). Four panels show GT, CoTracker3, SpaTrackerV2 and DELTA at the same frame, material IDs and view. A shared frame slider, view rotation, GT visibility filter, foreground filter and cumulative-flow arrows are available. Positions are initial GT camera-space positions plus canonical cumulative flow, in meters. Prediction colors show flow EPE on a fixed 0–20 cm scale. Viewer subset scores are explicitly separate from the full-query benchmark table. Extracted source arrays and their original member hashes are in [the viewer source manifest](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/experiment-01-interactive/source-manifest.json).

The existing videos show the rendered clip and saved comparison panels:

- static scene: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/static_scene_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/static_scene_orbit.mp4).
- object translation: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/object_translation_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/object_translation_orbit.mp4).
- rigid rotation: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/rigid_rotation_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/rigid_rotation_orbit.mp4).
- articulated links: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/articulated_links_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/articulated_links_orbit.mp4).
- occlusion and return: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/occlusion_and_return_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/occlusion_and_return_orbit.mp4).
- depth speed out of view stress: [fixed video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/depth_speed_out_of_view_stress_fixed.mp4), [orbit video](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/videos/depth_speed_out_of_view_stress_orbit.mp4).

## Paired query-density measurements

For this control, the 256-query run and the 4,096-query run use identical RGB, geometry, scene seed and model checkpoint. The dense predictions are rescored only at the exact 256 shared material IDs. GT visibility is shared. Model-specific additional support-query settings are held constant within each model. Each row below averages the same twelve paired test clips. The difference column is `(256-query EPE − 4,096-query EPE on those same points)`.

| Method | Mask | 256-query EPE cm | 4,096-query same-ID EPE cm | Difference cm |
|---|---|---|---|---|
| CoTracker3 + GT geometry | visible | 1.644 | 1.736 | -0.092 |
| CoTracker3 + GT geometry | foreground_visible | 2.363 | 2.279 | 0.083 |
| DELTA + GT geometry | visible | 2.919 | 3.011 | -0.091 |
| DELTA + GT geometry | foreground_visible | 3.579 | 4.081 | -0.502 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | visible | 2.571 | 3.497 | -0.926 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | foreground_visible | 24.347 | 26.207 | -1.860 |


Source with per-clip coverage and PCK: [paired query-density measurements](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/query-density-control/paired_query_density.csv). Dense full-query averages in the primary table and these dense subset averages use different eligible point sets.

## Separate predicted-geometry measurement

This completed measurement uses the original twelve uniform-color debug16 clips, not the textured64 clips. `SpatialTrackerV2_Front` predicts depth, intrinsics and camera poses from RGB. Its unmodified frontend outputs are cached and shared between the CoTracker lifting run and the native SpaTracker pipeline; the latter also applies its native refinement. Frontend inference receives no GT geometry.

The evaluation applies one scalar per clip, measured from initial-frame GT depth divided by predicted depth. That scalar is applied to points and camera translations throughout the whole clip. No per-frame or future-frame GT alignment is applied. These are **initial-GT-scale diagnostic measurements**, rather than scale-unassisted monocular metric scores.

| Method / initial-GT-scale diagnostic | Fixed visible EPE cm | Orbit visible EPE cm | PCK <1 cm % |
|---|---|---|---|
| CoTracker3 + shared predicted frontend | 22.818 | 43.531 | 10.38 |
| SpaTrackerV2 native frontend + refinement | 5.232 | 16.180 | 14.28 |


Source: [debug and predicted-geometry per-clip measurements](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/debug-pilot/per_clip.csv). The separate original debug known-geometry matrix and BF16 precision diagnostic are retained in that CSV. The original debug CoTracker depth lift used its earlier zero-depth handling; the primary textured and matched-query runs use the strict positive-depth mask described above.

## Recorded scope and validation exclusions

The measured primary suite contains primitive rigid box assets, scripted linked poses, two camera modes and one seed per scene condition. It does not contain native robot joint articulation, a multi-seed asset/domain evaluation, official TAPVid3D benchmark scoring, or downstream policy outcomes. No confidence intervals are reported for this single-seed pilot.

An initial SpaTracker float32 attempt encountered an upstream attention exception that was swallowed by the implementation. Those twelve outputs were quarantined and excluded from these measured tables. The corrected patch enables supported SDPA fallback and raises remaining exceptions. Patch SHA256 is `35108abe01a6ff9ae9b34227f3e77730fa47c497bf4a0274515f52bb967131b8`. The old malformed `[T,Q,1]` zero-motion control was also excluded; the accepted `zero_motion_xyz` source is `[T,Q,3]`.

Checkpoint hashes, upstream revisions, environment records, GT validations and scientific-file hashes are retained in the [artifact ledger](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/textured64-pilot/artifact-ledger.json). The interactive source NPZs were copied directly from the locally saved textured incremental archive without rerunning inference.
