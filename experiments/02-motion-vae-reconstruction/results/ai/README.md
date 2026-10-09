# Experiment 2: completed motion-label VAE reconstruction study

**Experimental record:** 276 measured frozen VAE reconstruction jobs, completed on 2026-10-08. This writeup contains the experimental setup, motivation, working hypotheses, recorded measurements and visualizations.

## Motivation

FloMo-style supervision represents cumulative 3D point displacement as an RGB video. Experiment 2 measures that representation through the frozen video VAEs associated with Wan and LTX. Its measurements cover the conversion from 3D tracks to RGB, the encoder/decoder roundtrip, and the inverse conversion to 3D displacement. GT and tracker-generated labels are evaluated using the same calibration mapping.

## Working hypotheses

These were working hypotheses for the study; no formal statistical hypothesis tests were performed.

1. Range clipping and grid rasterization produce measurable displacement errors before VAE encoding.
2. The two native frozen codecs produce different physical reconstruction errors on identical motion-label inputs.
3. Reconstruction error relative to a label source and error relative to simulator GT are different measured quantities.
4. Encoding 33 rather than 17 frames can change reconstruction of the shared first 17 frames.

## Experimental setup

### Data and job matrix

Clips came from SAPIEN 3.0.3 simulation with exact rigid-body/material-point trajectories, known camera intrinsics and poses, and stored initial query identities. Six scene archetypes were used: static scene, translating object, rigid rotation, scripted articulated links, occlusion and return, and depth/speed/out-of-view stress. Each archetype had a fixed-camera and orbit-camera clip. Sampling was 20 Hz; the underlying clips contained 33 frames.

| Run group | Label sources and clip selection | Query grid | Codec input frames | Completed jobs |
|---|---|---:|---:|---:|
| Debug supplied-geometry study | Seven sources × twelve clips × two VAEs | 16×16 | 17 | 168 |
| Temporal-context control | GT × twelve clips × two VAEs | 16×16 | 33 | 24 |
| Practical shared-front study | Two sources × twelve clips × two VAEs | 16×16 | 17 | 48 |
| Textured followup | GT on twelve clips; three trackers on two selected clips; two VAEs | 64×64 | 17 | 36 |
| **Total** | | | | **276** |

The debug scenes used plain colored geometry. The textured followup used textured surfaces and a larger initial query grid. The two textured tracker clips were object translation/orbit and occlusion-and-return/orbit. GT results in that followup use all twelve clips; tracker rows use two clips.

The seven debug label sources were exact GT, an independently generated three-channel zero-flow control, GT 2D queries lifted with front-surface GT depth, CoTracker3 with GT geometry, DELTA with GT geometry, and SpaTrackerV2 with GT geometry in BF16 and corrected FP32 math modes. The supplied-geometry regimes used known simulator depth/camera information.

The practical regimes used the same cached predicted depth/intrinsics/camera frontend. A single median initial GT-depth scalar was applied during evaluation and held fixed over time. These are explicitly initial-GT-scale diagnostics. They were stored and reported separately from supplied-GT-geometry runs.

### Motion representation and calibration

The target was cumulative displacement in the initial camera coordinate system, in meters. The three channels were dx, dy and dz at fixed initial query-grid positions. A bilinear rasterizer with `align_corners=False` produced 256×256 images. Invalid query slots were filled with the computed zero-displacement RGB color; validity masks were retained.

One shared channel mapping used the 1st/99th percentiles of twelve separate GT calibration clips. Calibration drew 10,000 samples per nonempty foreground/background group per clip, excluding the query frame and using the remaining 32 frames. The same frozen bounds were used for every source, both VAEs, and the textured followup.

| Channel | Lower bound (m) | Upper bound (m) | Zero-displacement RGB value |
|---|---:|---:|---:|
| dx / R | −0.00618003 | 1.78750002 | 0.00344545 |
| dy / G | −0.27625635 | 0.10148724 | 0.73133299 |
| dz / B | −0.34161991 | 0.65030634 | 0.34440051 |

Float arrays were used for scoring. Display images and GIFs are previews. The inverse sampler recovered displacements at the original grid positions. A no-VAE render/inverse-render control was recorded for every source.

### Frozen codecs and hardware

The codecs were the Wan2.2 TI2V-5B VAE and the LTX-Video 2B v0.9.6 non-distilled VAE. Both remained frozen. Encoding used deterministic posterior mode with native latent normalization and its inverse on decoding. LTX diagnostic decoding used timestep zero and no stochastic decoder noise. Raw pre-clamp and clamped RGB were both saved.

The runs used NVIDIA B300 SXM6 AC GPUs, Torch 2.12.0+cu130 and CUDA 13.0. VAE arithmetic used BF16 autocast. The VAE-only workers loaded the encoder/decoder components.

| Native codec at 17×256×256 input | Latent shape [C,T,H,W] | Latent elements | Saved tensor payload |
|---|---|---:|---:|
| Wan | [48,5,16,16] | 61,440 | 245,760 bytes, float32 |
| LTX | [128,3,8,8] | 24,576 | 49,152 bytes, BF16 |

These shapes and payload sizes were read from actual saved GT metrics. The latent-element ratio was 2.5× and the saved tensor-payload ratio 5×. Runs used native compression rates; no equal latent-element budget was applied. Payload values exclude serialization overhead and are not entropy-coded bitrates.

### Metrics, masks and saved outputs

All reported EPE values are Euclidean displacement errors in millimeters, first averaged over the scored queries/frames of a clip and then averaged across clips. Main motion scores exclude frame zero. Scoring includes initially GT-valid queries with finite source predictions, including coordinates through occlusion when available. Coverage is reported over all 17 frames. Foreground results use the saved initial foreground query mask.

- **Raw flow EPE:** source displacement versus simulator GT.
- **Range EPE:** clipped source versus raw source.
- **Raster EPE:** no-VAE inverse-rendered flow versus clipped source.
- **Codec self-EPE:** decoded displacement versus no-VAE inverse-rendered flow.
- **Decoded-to-GT EPE:** decoded displacement versus simulator GT, including preceding representation steps.

Raw/clipped flow, source RGB, raw/clamped decoded RGB, recovered flow, masks, deterministic latent tensors and per-job metrics were saved. Vector error identities were checked at ≤1e-6 m, and saved decoder outputs were checked for finiteness. Scientific files have SHA256 ledgers. The earlier SpaTrackerV2 FP32 branch with a failed attention kernel was excluded; the corrected FP32 math branch used the recorded fail-closed SDPA patch.

## Recorded results

### Debug supplied-geometry study — 168 jobs

| Label source | Clips | Raw flow EPE (mm) | Range / raster EPE (mm) | Wan / LTX codec self-EPE (mm) | Wan / LTX decoded-to-GT EPE (mm) | Wan / LTX foreground EPE (mm) | Finite coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact simulator GT | 12 | 0.000 | 0.000 / 0.565 | 4.009 / 6.243 | 4.466 / 6.664 | 13.517 / 21.793 | 100.000% |
| Zero-flow control | 12 | 13.211 | 0.000 / 0.000 | 3.535 / 5.223 | 16.478 / 18.383 | 179.865 / 184.655 | 100.000% |
| GT 2D queries + front-surface GT depth | 12 | 20.289 | 10.745 / 1.179 | 4.925 / 8.904 | 12.187 / 14.645 | 37.829 / 47.010 | 97.400% |
| CoTracker3 + GT geometry | 12 | 95.076 | 9.900 / 1.626 | 7.609 / 13.067 | 84.922 / 81.705 | 52.340 / 56.681 | 100.000% |
| DELTA + GT geometry | 12 | 74.220 | 2.385 / 1.228 | 6.135 / 9.345 | 72.651 / 70.857 | 34.541 / 37.804 | 100.000% |
| SpaTrackerV2 + GT geometry, BF16 | 12 | 93.016 | 0.315 / 1.385 | 6.988 / 11.593 | 92.168 / 87.574 | 148.841 / 150.859 | 100.000% |
| SpaTrackerV2 + GT geometry, FP32 math | 12 | 93.242 | 0.274 / 1.359 | 6.689 / 11.494 | 92.327 / 87.805 | 149.417 / 151.249 | 100.000% |

### Practical shared-front study — 48 jobs

All rows below use twelve clips and the single fixed initial GT-depth scale described above.

| Label source | Clips | Raw flow EPE (mm) | Range / raster EPE (mm) | Wan / LTX codec self-EPE (mm) | Wan / LTX decoded-to-GT EPE (mm) | Wan / LTX foreground EPE (mm) | Finite coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoTracker3 + shared predicted geometry | 12 | 149.076 | 64.673 / 1.543 | 10.063 / 15.379 | 93.616 / 90.971 | 158.109 / 161.743 | 100.000% |
| SpaTrackerV2 + shared predicted geometry | 12 | 83.633 | 7.748 / 0.792 | 6.637 / 9.826 | 79.988 / 78.787 | 188.644 / 191.854 | 100.000% |

### Textured64 followup — 36 jobs

GT rows use twelve clips; each tracker row uses the two selected orbit-camera clips. The original calibration bounds were retained.

| Label source | Clips | Raw flow EPE (mm) | Range / raster EPE (mm) | Wan / LTX codec self-EPE (mm) | Wan / LTX decoded-to-GT EPE (mm) | Wan / LTX foreground EPE (mm) | Finite coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact simulator GT | 12 | 0.000 | 0.100 / 0.569 | 3.721 / 6.046 | 4.277 / 6.522 | 12.807 / 17.652 | 100.000% |
| CoTracker3 + GT geometry | 2 | 38.293 | 13.085 / 2.781 | 7.124 / 11.468 | 25.829 / 24.481 | 40.296 / 45.859 | 99.523% |
| DELTA + GT geometry | 2 | 30.337 | 0.170 / 1.508 | 4.644 / 7.890 | 29.894 / 29.319 | 26.259 / 27.986 | 100.000% |
| SpaTrackerV2 + GT geometry, FP32 math | 2 | 20.824 | 0.095 / 2.318 | 4.504 / 7.608 | 21.204 / 21.894 | 96.400 / 98.451 | 100.000% |

### Paired GT codec measurements

Positive paired difference is LTX EPE minus Wan EPE. The ranges below are empirical per-clip extrema.

| Condition | Metric | Mean paired difference (mm) | Median difference (mm) | Observed difference range (mm) | Wan / LTX lower-error counts |
|---|---|---:|---:|---:|---:|
| debug_g16 | overall | 2.197 | 2.146 | 1.687–2.840 | 12 / 0 |
| debug_g16 | foreground | 8.276 | 5.825 | 2.103–18.864 | 12 / 0 |
| textured_g64 | overall | 2.245 | 2.284 | 1.849–2.508 | 12 / 0 |
| textured_g64 | foreground | 4.845 | 4.997 | 2.060–7.085 | 12 / 0 |

Fixed/orbit GT codec scores were exactly equal within each of the six archetype families for both metrics and conditions. The twelve camera-labelled records therefore repeat six archetype results. There was one scene seed per archetype; no population confidence intervals were calculated.

![Measured paired GT errors across clips and foreground masks](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/offline-analysis/paired-gt-codec-errors.png)

### Temporal-context control — 24 jobs

The 17- and 33-frame runs shared the first seventeen inputs. The following table scores only that shared window, excluding the query frame.

| Codec | Clips | 17-frame decoded-to-GT EPE (mm) | 33-frame shared-first17 EPE (mm) | Output displacement difference (mm) |
|---|---:|---:|---:|---:|
| Wan | 12 | 4.4665 | 4.4665 | 0.0000 |
| LTX | 12 | 6.6637 | 6.4488 | 1.7065 |

### Recorded GT encode/decode timing and allocation

These are medians over the twelve GT clips in each 17-frame condition. The timed region covers encoding/decoding with synchronization; model loading and artifact writing are outside it. Allocation is the recorded peak allocated CUDA memory.

| Condition | Codec | Median encode/decode time (s) | Peak allocation (GB) |
|---|---|---:|---:|
| pilot-v2 | wan | 0.0939 | 4.199 |
| pilot-v2 | ltx | 0.0240 | 2.790 |
| textured64-bounded | wan | 0.0933 | 4.199 |
| textured64-bounded | ltx | 0.0234 | 2.790 |

## Visual records and interactive maps

[Open the synchronized GT / Wan / LTX frame browser](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/interactive/gt-wan-ltx-frame-browser.html)

The browser shows all seventeen measured frames of the GT object-translation/orbit clip, with source RGB, saved Wan/LTX reconstruction and physical-error maps. Its slider and playback are synchronized. Clicking either map selects a persistent initial query ID and displays its GT/Wan/LTX dx,dy,dz values. Map limits are fixed across time and codecs. The two source NPZ members were streamed from the verified local archive and checked against the saved SHA256 ledger.

### Motion-RGB sequence

![Measured source and reconstructed motion RGB](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/object_translation_orbit-motion-rgb.gif)

### Recovered 3D paths

The trails use the same GT initial position plus each source/decoded displacement, with fixed query IDs and axis limits. Display playback is 10 fps; source sampling is 20 Hz.

![Measured GT raw-label and decoded 3D paths](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/object_translation_orbit-3d-paths.gif)

### Frame comparison maps

![Debug object-translation/orbit comparisons](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/pilot-v2/object_translation_orbit.png)

![Textured64 object-translation/orbit comparisons](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/textured64-bounded/object_translation_orbit.png)

Additional measured animations: [occlusion motion RGB](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/occlusion_and_return_orbit-motion-rgb.gif) · [occlusion 3D paths](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/occlusion_and_return_orbit-3d-paths.gif) · [stress motion RGB](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/depth_speed_out_of_view_stress_orbit-motion-rgb.gif) · [stress 3D paths](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/animations/depth_speed_out_of_view_stress_orbit-3d-paths.gif).

## Measurement files

- [Debug per-clip CSV](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/pilot-v2/per-clip.csv)
- [Practical per-clip CSV](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/practical-shared-front/per-clip.csv)
- [Textured per-clip CSV](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/textured64-bounded/per-clip.csv)
- [Paired GT observations](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/offline-analysis/paired-gt-errors.csv)
- [Temporal-context paired observations](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/context33/paired-first17.csv)
- [Native latent-shape evidence](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/offline-analysis/latent-shape-evidence.json)
- [Interactive-browser provenance](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/02-motion-vae-reconstruction/results/ai/interactive/frame-browser-provenance.json)

The numerical tables retain the original study groups, clip counts, calibration bounds and scoring masks. The data record contains no formal inferential test results.
