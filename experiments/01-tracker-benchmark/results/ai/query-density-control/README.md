# Query-density control

Each 16×16 run uses an exact material-ID subset of the corresponding textured 64×64 scene. The comparison rescored dense predictions on precisely those same queries, with the same GT visibility mask. RGB, poses, geometry, checkpoint, support policy and seed are shared. The number of jointly tracked query points changes; this is an architecture/context sensitivity test.

| Method | 16-grid visible EPE (cm) | Dense-grid same-query EPE (cm) | Paired difference (cm) |
|---|---:|---:|---:|
| cotracker3_gt_geometry | 1.644 | 1.736 | -0.092 |
| spatrackerv2_gt_geometry_fp32_math | 2.571 | 3.497 | -0.926 |
| delta_gt_geometry | 2.919 | 3.011 | -0.091 |

This controls query density on textured scenes. Comparing these numbers with the older uniform-color debug pilot changes both texture and query positions/density; it does not isolate a texture-only causal effect. One scene seed per archetype remains a case-study limitation. Foreground-visible paired errors and finite coverage are saved in the CSV.
