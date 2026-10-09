# Experiment 2 followup observations — 2026-10-08

The bounded run completed 276 measured frozen VAE reconstructions: 168 debug supplied-geometry cases, 24 temporal-context controls, 48 practical shared-front cases, and 36 textured64 cases. These are pilot measurements, with one scene seed per archetype and paired fixed/orbit cameras. They establish an executable, preserved label-to-codec pipeline and identify failure modes; they do not establish general downstream policy performance.

## Practical predicted geometry is a separate diagnostic

Both practical methods use the same cached RGB-only geometry frontend. Evaluation applies one median initial GT-depth scalar, fixed over the trajectory. GT depth/camera trajectories are not inputs to the frontend. The initial scalar remains privileged information; this is not uncalibrated metric-monocular tracking, and it cannot correct temporal scale drift or relative depth distortion.

| Source | Raw flow EPE | Range loss | Wan decoded-to-GT | LTX decoded-to-GT | Wan/LTX codec self-error |
|---|---:|---:|---:|---:|---:|
| CoTracker3 + shared predicted geometry | 149.076 mm | 64.673 mm | 93.616 mm | 90.971 mm | 10.063 / 15.379 mm |
| SpaTrackerV2 + shared predicted geometry | 83.633 mm | 7.748 mm | 79.988 mm | 78.787 mm | 6.637 / 9.826 mm |

These are twelve-clip macro averages. The large CoTracker reduction after encoding is strongly affected by clipping under the fixed GT calibration range. Scalar EPE terms are not additive: clipping, rasterization, and codec errors can cancel portions of tracker error. LTX has greater self-distortion while obtaining a lower decoded-to-GT average on these erroneous sources. That is evidence against selecting a codec from decoded tracker-to-GT error alone, not evidence that its reconstruction is intrinsically more faithful.

## Textured64 GT retains a small codec floor

All textured cases retain the **original frozen calibration bounds**, with a separate output namespace. Across twelve GT clips, decoded-to-GT error is 4.277 mm for Wan and 6.522 mm for LTX; self-only error is 3.721 and 6.046 mm. Foreground GT error is 12.807 and 17.652 mm, appreciably larger than the whole-grid average. The range floor is 0.100 mm and the raster floor 0.569 mm.

The twelve-clip debug counterparts are 4.466/6.664 mm overall and 13.517/21.793 mm foreground. These modest differences should not be interpreted as a controlled texture effect: texture, initial query density, and the set of sampled material points change together. Texture does not directly enter the pseudo-RGB input; its effect on tracking and the grid's effect on motion rasterization need separate controls.

## Compare trackers on the same subset

The textured tracker reconstruction followup includes only **object translation/orbit** and **occlusion/return/orbit**. Comparing those two-clip tracker averages against all twelve debug clips would change the estimand. The correctly matched raw flow comparison is:

| Tracker with supplied GT geometry | Same two debug clips, G16 | Same two textured clips, G64 |
|---|---:|---:|
| CoTracker3 | 144.912 mm | 38.293 mm |
| DELTA | 97.262 mm | 30.337 mm |
| Corrected SpaTrackerV2 FP32 math | 87.271 mm | 20.824 mm |

The combined appearance/density change substantially improves this subset, consistent with the severe ambiguity of plain colored surfaces. It does not isolate which change causes the improvement. Experiment 1's same-density texture controls are needed for that attribution.

The whole-grid ordering also hides foreground behavior. On the textured subset, SpaTrackerV2 has the lowest raw whole-grid error, but decoded foreground error remains 96.400/98.451 mm for Wan/LTX. DELTA's corresponding foreground error is 26.259/27.986 mm, and CoTracker3's is 40.296/45.859 mm. Static surfaces occupy most valid queries, so whole-grid EPE alone is a poor proxy for useful manipulation motion labels. CoTracker finite coverage is 99.523%; its conditional EPE must be read with that coverage, while the other two sources have 100% finite coverage.

## Interpretation and remaining work

Wan and LTX are compared at their native compression rates, not equal latent capacity. The GT codec comparison favors Wan for this physical mapping, while wrong tracker labels can favor a more smoothing codec under an end-to-end score. Both findings can hold simultaneously.

Future frames change LTX's reconstruction of the shared first seventeen frames by 1.707 mm on average; Wan's shared prefix is identical. This is a codec context diagnostic, not a policy inference claim. No temporal alignment was applied before scoring.

The invalid initial SpaTrackerV2 FP32 attention branch was never consumed. The corrected FP32 math branch uses the fail-closed SDPA patch documented by Experiment 1. Every preserved reconstruction is checked for finite decoder output and the vector error identity, and scientific files have checksum ledgers. The practical and textured archives are separate increments; neither overwrites the authoritative debug runs.

Useful next controls are matched query density and camera trajectory, symmetric versus asymmetric **fixed** bounds, rasterizer controls, and noise/frequency fixtures. A broader native monocular comparison and downstream data-efficiency/OOD policy claims require their own experiments; these results should guide label-quality checks rather than replace them.
