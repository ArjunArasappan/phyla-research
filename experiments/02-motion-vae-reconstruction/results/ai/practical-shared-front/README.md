# Experiment 2 GPU pilot results

These are measured frozen VAE reconstructions of 17-frame motion windows from exact SAPIEN scripted-box/link trajectories. One scene seed per archetype and paired fixed/orbit cameras; this debug-density pilot does not establish a statistically general model ranking. Both VAEs are frozen; no T5, DiT, training, or diffusion generation is loaded.

All methods/codecs share calibration-only bounds (12 separate scene clips), a 16×16 initial query grid upsampled bilinearly to 256×256, and deterministic posterior mode. Query frame is excluded from main physical averages. Errors below are **macro means across clips**; tracker nonfinite coordinates are excluded from conditional EPE and their coverage is reported separately. Practical predicted geometry from a shared RGB-only frontend, followed by one fixed median initial GT-depth scale scalar. This is an initial-GT-scale diagnostic, not uncalibrated metric-monocular performance. GT enters scale evaluation only; depth/camera inference uses RGB. This table is separate from supplied-GT-geometry results.

| Source | Codec | Clips | Tracking (mm) | Range (mm) | Raster (mm) | Codec self (mm) | Decoded to GT (mm) | Foreground to GT (mm) | Background to GT (mm) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cotracker3_shared_front_initial_GT_scale | wan | 12 | 149.076 | 64.673 | 1.543 | 10.063 | 93.616 | 158.109 | 88.672 | 100.000% |
| cotracker3_shared_front_initial_GT_scale | ltx | 12 | 149.076 | 64.673 | 1.543 | 15.379 | 90.971 | 161.743 | 85.540 | 100.000% |
| spatrackerv2_shared_front_initial_GT_scale | wan | 12 | 83.633 | 7.748 | 0.792 | 6.637 | 79.988 | 188.644 | 70.830 | 100.000% |
| spatrackerv2_shared_front_initial_GT_scale | ltx | 12 | 83.633 | 7.748 | 0.792 | 9.826 | 78.787 | 191.854 | 69.276 | 100.000% |

Scalar EPE terms are not additive. The decoded-to-GT quantities include tracking, shared range clipping, rasterization and codec loss; codec-self is measured relative to the no-VAE inverse-rendered flow. Foreground/background values are clip-macro means on initial query group masks. Raw decoder overshoot, RGB error, axis errors, native signatures, and exact vector-error identity checks are in each metrics.json. Complete inputs, raw/clamped RGB, reconstructed flows, masks and latents are retained. Preview figures use shared physical error limits.

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
