# Experiment 2 GPU pilot results

These are measured frozen VAE reconstructions of 17-frame motion windows from exact SAPIEN scripted-box/link trajectories. One scene seed per archetype and paired fixed/orbit cameras; this debug-density pilot does not establish a statistically general model ranking. Both VAEs are frozen; no T5, DiT, training, or diffusion generation is loaded.

All methods/codecs share calibration-only bounds (12 separate scene clips), a 16×16 initial query grid upsampled bilinearly to 256×256, and deterministic posterior mode. Query frame is excluded from main physical averages. Errors below are **macro means across clips**; tracker nonfinite coordinates are excluded from conditional EPE and their coverage is reported separately. Tracker results with GT depth/cameras are privileged labels, not monocular metrics.

| Source | Codec | Clips | Tracking (mm) | Range (mm) | Raster (mm) | Codec self (mm) | Decoded to GT (mm) | Foreground to GT (mm) | Background to GT (mm) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cotracker3_gt_geometry | wan | 12 | 95.076 | 9.900 | 1.626 | 7.609 | 84.922 | 52.340 | 87.336 | 100.000% |
| cotracker3_gt_geometry | ltx | 12 | 95.076 | 9.900 | 1.626 | 13.067 | 81.705 | 56.681 | 83.570 | 100.000% |
| delta_gt_geometry | wan | 12 | 74.220 | 2.385 | 1.228 | 6.135 | 72.651 | 34.541 | 75.397 | 100.000% |
| delta_gt_geometry | ltx | 12 | 74.220 | 2.385 | 1.228 | 9.345 | 70.857 | 37.804 | 73.245 | 100.000% |
| gt | wan | 12 | 0.000 | 0.000 | 0.565 | 4.009 | 4.466 | 13.517 | 3.811 | 100.000% |
| gt | ltx | 12 | 0.000 | 0.000 | 0.565 | 6.243 | 6.664 | 21.793 | 5.599 | 100.000% |
| gt_uv_gt_front_depth | wan | 12 | 20.289 | 10.745 | 1.179 | 4.925 | 12.187 | 37.829 | 9.678 | 97.400% |
| gt_uv_gt_front_depth | ltx | 12 | 20.289 | 10.745 | 1.179 | 8.904 | 14.645 | 47.010 | 11.720 | 97.400% |
| spatrackerv2_gt_geometry_bf16 | wan | 12 | 93.016 | 0.315 | 1.385 | 6.988 | 92.168 | 148.841 | 87.048 | 100.000% |
| spatrackerv2_gt_geometry_bf16 | ltx | 12 | 93.016 | 0.315 | 1.385 | 11.593 | 87.574 | 150.859 | 81.917 | 100.000% |
| spatrackerv2_gt_geometry_fp32_math | wan | 12 | 93.242 | 0.274 | 1.359 | 6.689 | 92.327 | 149.417 | 87.165 | 100.000% |
| spatrackerv2_gt_geometry_fp32_math | ltx | 12 | 93.242 | 0.274 | 1.359 | 11.494 | 87.805 | 151.249 | 82.118 | 100.000% |
| zero | wan | 12 | 13.211 | 0.000 | 0.000 | 3.535 | 16.478 | 179.865 | 3.514 | 100.000% |
| zero | ltx | 12 | 13.211 | 0.000 | 0.000 | 5.223 | 18.383 | 184.655 | 5.167 | 100.000% |

Scalar EPE terms are not additive. The decoded-to-GT quantities include tracking, shared range clipping, rasterization and codec loss; codec-self is measured relative to the no-VAE inverse-rendered flow. Foreground/background values are clip-macro means on initial query group masks. Raw decoder overshoot, RGB error, axis errors, native signatures, and exact vector-error identity checks are in each metrics.json. Complete inputs, raw/clamped RGB, reconstructed flows, masks and latents are retained. Preview figures use shared physical error limits.

## Interpretation limits

The zero-flow control has poor foreground motion fidelity even when its whole-grid average looks favorable, because most initial queries are static surfaces. Native trackers face severe appearance ambiguity on these plain colored surfaces. Tracker-to-GT improvements after encoding can arise from range clipping or smoothing; low codec self-error never establishes accurate supervision. The corrected SpaTrackerV2 FP32-math source uses the fail-closed SDPA patch from Experiment 1 (SHA256 `35108abe01a6ff9ae9b34227f3e77730fa47c497bf4a0274515f52bb967131b8`); the invalid earlier FP32 branch was excluded. This pilot contains matched privileged geometry baselines, not a completed native-monocular benchmark.

## Frozen mapping and visual interpretation

Axis lower bounds (m): [-0.0061800298281013966, -0.27625635266304016, -0.34161990880966187]; upper bounds (m): [1.787500023841858, 0.10148724168539047, 0.6503063440322876]. Zero-displacement RGB: [0.003445447149538596, 0.7313329909394078, 0.3444005114602956]. Calibration sampling excludes the anchor frame and uses all32 non-anchor frames, whereas the primary codec window uses16 non-anchor frames. High-speed calibration motion broadens the shared range, so ordinary motion can appear nearly uniform in the RGB previews. Physical error maps use meter units and shared Wan/LTX color limits within each source row. No per-source contrast normalization is applied.

## Side-by-side diagnostics

![articulated_links_fixed](articulated_links_fixed.png)
![articulated_links_orbit](articulated_links_orbit.png)
![depth_speed_out_of_view_stress_fixed](depth_speed_out_of_view_stress_fixed.png)
![depth_speed_out_of_view_stress_orbit](depth_speed_out_of_view_stress_orbit.png)
![object_translation_fixed](object_translation_fixed.png)
![object_translation_orbit](object_translation_orbit.png)
![occlusion_and_return_fixed](occlusion_and_return_fixed.png)
![occlusion_and_return_orbit](occlusion_and_return_orbit.png)
![rigid_rotation_fixed](rigid_rotation_fixed.png)
![rigid_rotation_orbit](rigid_rotation_orbit.png)
![static_scene_fixed](static_scene_fixed.png)
![static_scene_orbit](static_scene_orbit.png)
