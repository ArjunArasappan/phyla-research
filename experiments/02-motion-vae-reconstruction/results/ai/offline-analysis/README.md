# Experiment 2: offline paired GT codec comparison

This analysis uses the saved per-clip CSVs only. It includes **GT-source flow**, with twelve clips in each condition: six archetypes, each observed by fixed and orbit cameras. Both codec outputs on one clip use identical inputs, calibration bounds and GT-valid/foreground query masks. The query frame is excluded from motion averages. No tracker rows, GPU jobs or new model inference are used.

Wan has lower error in every saved GT clip in both conditions: the mean whole-grid paired advantage is **2.197 mm** for debug G16 and **2.245 mm** for textured G64. This is a descriptive native-rate result: Wan uses **2.5× as many latent elements**, and fixed/orbit scores repeat the same six archetype outcomes exactly.

“Decoded-to-GT” includes range clipping, the raster roundtrip and frozen VAE distortion. The foreground column is an average over initially foreground queries, followed by a clip-macro average; it is not an image-area score. Positive paired difference below means **LTX error minus Wan error**, favoring Wan.

![Actual paired per-clip GT errors](paired-gt-codec-errors.png)

## Descriptive twelve-clip effects

| Condition | Metric | Wan mean (mm) | LTX mean (mm) | Mean paired difference (mm) | Median difference (mm) | Empirical difference range (mm) | Wan / LTX wins |
|---|---|---:|---:|---:|---:|---:|---:|
| debug_g16 | overall | 4.466 | 6.664 | +2.197 | +2.146 | [+1.687, +2.840] | 12 / 0 |
| debug_g16 | foreground | 13.517 | 21.793 | +8.276 | +5.825 | [+2.103, +18.864] | 12 / 0 |
| textured_g64 | overall | 4.277 | 6.522 | +2.245 | +2.284 | [+1.849, +2.508] | 12 / 0 |
| textured_g64 | foreground | 12.807 | 17.652 | +4.845 | +4.997 | [+2.060, +7.085] | 12 / 0 |

The ranges are observed per-clip extrema, not confidence intervals. Wins count paired clips, not independent seeds. Fixed/orbit siblings have **exactly identical saved GT codec error values within every archetype**, for both metrics and conditions. The twelve clip-level wins therefore repeat six archetype results; Wan favors all six family means in each condition/metric. Canonical anchor-camera GT displacements remove camera motion, so identical scores are consistent with the representation design. These strata do not establish codec robustness to camera motion. Treating twelve clips or millions of queries as independent replications would exaggerate evidence.

## Camera strata

| Condition | Metric | Camera | Clips | Wan mean (mm) | LTX mean (mm) | Mean paired difference (mm) | Wan / LTX wins |
|---|---|---|---:|---:|---:|---:|---:|
| debug_g16 | overall | fixed | 6 | 4.466 | 6.664 | +2.197 | 6 / 0 |
| debug_g16 | overall | orbit | 6 | 4.466 | 6.664 | +2.197 | 6 / 0 |
| debug_g16 | foreground | fixed | 6 | 13.517 | 21.793 | +8.276 | 6 / 0 |
| debug_g16 | foreground | orbit | 6 | 13.517 | 21.793 | +8.276 | 6 / 0 |
| textured_g64 | overall | fixed | 6 | 4.277 | 6.522 | +2.245 | 6 / 0 |
| textured_g64 | overall | orbit | 6 | 4.277 | 6.522 | +2.245 | 6 / 0 |
| textured_g64 | foreground | fixed | 6 | 12.807 | 17.652 | +4.845 | 6 / 0 |
| textured_g64 | foreground | orbit | 6 | 12.807 | 17.652 | +4.845 | 6 / 0 |

## Six archetype pairs

Each number is a mean of its fixed/orbit sibling pair. This keeps camera siblings together as a scene-family descriptive summary. It supplies no between-seed population estimate.

| Condition | Archetype | Whole-grid Wan / LTX (mm) | Whole-grid difference (mm) | Foreground Wan / LTX (mm) | Foreground difference (mm) |
|---|---|---:|---:|---:|---:|
| debug_g16 | Scripted links | 5.076 / 7.592 | +2.517 | 23.125 / 37.559 | +14.435 |
| debug_g16 | Depth/speed stress | 5.378 / 8.218 | +2.840 | 29.202 / 48.066 | +18.864 |
| debug_g16 | Translation | 3.993 / 5.985 | +1.991 | 6.499 / 9.106 | +2.607 |
| debug_g16 | Occlusion/return | 4.810 / 6.658 | +1.848 | 10.201 / 13.284 | +3.083 |
| debug_g16 | Rotation | 4.009 / 6.309 | +2.300 | 8.315 / 16.882 | +8.568 |
| debug_g16 | Static scene | 3.533 / 5.220 | +1.687 | 3.760 / 5.863 | +2.103 |
| textured_g64 | Scripted links | 4.996 / 7.309 | +2.314 | 25.233 / 30.288 | +5.055 |
| textured_g64 | Depth/speed stress | 5.175 / 7.455 | +2.281 | 20.410 / 27.496 | +7.085 |
| textured_g64 | Translation | 3.593 / 6.101 | +2.508 | 6.864 / 10.439 | +3.575 |
| textured_g64 | Occlusion/return | 4.356 / 6.588 | +2.231 | 9.625 / 14.564 | +4.939 |
| textured_g64 | Rotation | 4.220 / 6.507 | +2.287 | 11.328 / 17.684 | +6.356 |
| textured_g64 | Static scene | 3.323 / 5.172 | +1.849 | 3.380 / 5.440 | +2.060 |

## Interpretation and latent-rate limitation

These native codecs are **not rate matched**. At 17×256×256 input, the saved runtime contract is Wan [48,5,16,16] = **61,440 latent elements**, versus LTX [128,3,8,8] = **24,576 elements**. Wan carries **2.5× as many latent scalar elements**. The shapes were verified by reading two small saved GT metrics.json members directly from the local archive, without unpacking scientific arrays. Saved tensor payloads are 245,760 bytes for Wan and 49,152 bytes for LTX: **5×**, because the adapter saves Wan latents as float32 and LTX as BF16. Those are uncompressed tensor payloads, not entropy-coded file rates or proof of extra effective numerical precision. Element count and storage dtype both confound a capacity comparison. A lower error at Wan’s deployed rate does not establish architectural superiority at equal capacity. A matched-rate comparison would require an explicit rate–distortion protocol rather than simply changing these pretrained latent shapes.

Texture, query density and sampled surfaces change between debug G16 and textured G64. The normalization bounds remain frozen, but these are distinct conditions; differences across them do not isolate texture or density. The paired comparison within a condition is the reliable descriptive unit. Three tracker followup rows elsewhere use only two clips and are deliberately excluded here.

Natural-image VAEs can reconstruct the synthetic motion RGB while producing physical error that varies by channel ranges, object boundaries and static versus moving surfaces. GT-source results establish a representation/codec floor; they do not validate tracker accuracy, action learning, OOD robustness or policy success.

## Reproducibility

- [Paired observations](paired-gt-errors.csv): all 48 condition/clip/metric pairs, with both codec values and empirical winner.
- [Machine-readable summary](summary.json): exact statistics, camera/archetype strata and input CSV SHA256s.
- [Figure SVG](paired-gt-codec-errors.svg): vector export of the actual-data plot.
- [Actual latent-shape evidence](latent-shape-evidence.json): saved GT metric members, shapes and tensor payload bytes.

The source CSVs and latent-shape evidence are local saved artifacts. Only two small JSON members were streamed from the existing archive to verify shapes; no large scientific arrays or archive directory were extracted. The offline CSV analysis runs independently of GPU inference.
