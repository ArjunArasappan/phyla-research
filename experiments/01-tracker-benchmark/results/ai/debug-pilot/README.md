# Experiment 1 executed debug pilot

Real SAPIEN rendered scenes: 12 test clips + 12 disjoint-seeded calibration clips, 33 frames at 20 Hz, 256×256 RGB. Queries: 16×16 with approximately 132 valid material points. Exact actor-local material GT and time-varying ray visibility.

This is an executable pipeline and diagnostic pilot, not the proposed full tracking comparison. Uniform colored surfaces produce substantial correspondence ambiguity; articulated case uses scripted linked rigid boxes, not native robot joints. SpaTrackerV2, CoTracker3 depth lifts and independent DELTA frozen3D all completed. Textures and larger query density remain pending. Predicted-geometry rows use one GTinitialdepth scale and are separate privileged-scale diagnostics.

| Method / geometry regime (native rows use initial-GT-scale diagnostic) | Fixed camera visible EPE (cm) | Orbit camera visible EPE (cm) |
|---|---:|---:|
| cotracker3_gt_geometry | 4.127 | 25.099 |
| cotracker3_shared_front_initial_GT_scale | 22.818 | 43.531 |
| delta_gt_geometry | 4.985 | 20.988 |
| gt_uv_exact_front_ray | 0.000 | 0.000 |
| gt_uv_gt_front_depth | 0.063 | 0.920 |
| spatrackerv2_gt_geometry | 6.556 | 22.976 |
| spatrackerv2_gt_geometry_fp32_math | 6.473 | 22.981 |
| spatrackerv2_shared_front_initial_GT_scale | 7.467 | 14.306 |
| zero_motion | 1.739 | 2.004 |

EPE excludes frame 0, uses GT visibility, and is conditional on finite predictions. Coverage and per-clip numbers are in `per_clip.csv`. Mean averages six clips per camera. GT-UV/exact-front-ray baseline verifies numerical closure; GT-UV/rendered-nearest-depth shows raster/sampling error. Both are visible-only; hidden material depth is never leaked. Raw tracker outputs are retained beside canonical trajectories.

SpaTracker BF16 diagnostic and corrected FP32 SDPA/math precision control appear as separate variants, not independent architectures. The initial FP32 attempt hit an upstream swallowed attention exception and was quarantined, excluded from all summaries; the tracked fail-closed patch enables supported SDPA math fallback and raises remaining failures. Model rankings must wait for input convention verification, textured scene evaluation and wider independent scene replication.

![Visible geometry](visible_geometry.png)

![Translation-orbit time curves](translation_orbit_time.png)
