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

![Measured paired GT errors across clips and foreground masks](../offline-analysis/paired-gt-codec-errors.png)

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

[Open the synchronized GT / Wan / LTX frame browser](../interactive/gt-wan-ltx-frame-browser.html)

The browser shows all seventeen measured frames of the GT object-translation/orbit clip, with source RGB, saved Wan/LTX reconstruction and physical-error maps. Its slider and playback are synchronized. Clicking either map selects a persistent initial query ID and displays its GT/Wan/LTX dx,dy,dz values. Map limits are fixed across time and codecs. The two source NPZ members were streamed from the verified local archive and checked against the saved SHA256 ledger.

### Motion-RGB sequence

![Measured source and reconstructed motion RGB](../animations/object_translation_orbit-motion-rgb.gif)

### Recovered 3D paths

The trails use the same GT initial position plus each source/decoded displacement, with fixed query IDs and axis limits. Display playback is 10 fps; source sampling is 20 Hz.

![Measured GT raw-label and decoded 3D paths](../animations/object_translation_orbit-3d-paths.gif)

### Frame comparison maps

![Debug object-translation/orbit comparisons](../pilot-v2/object_translation_orbit.png)

![Textured64 object-translation/orbit comparisons](../textured64-bounded/object_translation_orbit.png)

Additional measured animations: [occlusion motion RGB](../animations/occlusion_and_return_orbit-motion-rgb.gif) · [occlusion 3D paths](../animations/occlusion_and_return_orbit-3d-paths.gif) · [stress motion RGB](../animations/depth_speed_out_of_view_stress_orbit-motion-rgb.gif) · [stress 3D paths](../animations/depth_speed_out_of_view_stress_orbit-3d-paths.gif).

## Measurement files

- [Debug per-clip CSV](../pilot-v2/per-clip.csv)
- [Practical per-clip CSV](../practical-shared-front/per-clip.csv)
- [Textured per-clip CSV](../textured64-bounded/per-clip.csv)
- [Paired GT observations](../offline-analysis/paired-gt-errors.csv)
- [Temporal-context paired observations](../context33/paired-first17.csv)
- [Native latent-shape evidence](../offline-analysis/latent-shape-evidence.json)
- [Interactive-browser provenance](../interactive/frame-browser-provenance.json)

The numerical tables retain the original study groups, clip counts, calibration bounds and scoring masks. The data record contains no formal inferential test results.

## Analysis of the measured pilot

### Motion-codec fidelity under native compression

For exact GT motion, Wan records lower decoded-to-GT EPE than LTX in both the debug study (4.466 versus 6.664 mm) and textured followup (4.277 versus 6.522 mm). Foreground errors are larger than the overall scores: 13.517/21.793 mm in debug and 12.807/17.652 mm in textured scenes. In these measurements, scene-wide averages dilute errors on moving material. The paired tables retain that distinction rather than treating the overall score as object-motion fidelity.

This is a native-codec comparison, not a rate-distortion comparison at equal capacity. At the same 17-frame resolution, Wan has 61,440 latent elements versus LTX's 24,576. Saved tensor payloads also differ in dtype. Wan's smaller measured distortion therefore does not prove superiority at equal bitrate or latent budget. Actual file payloads are not entropy-coded bitrates.

### Separate tracking, representation and codec errors

The experiment retains raw tracker error, range clipping, raster inversion, codec self-error and final decoded-to-GT error. Their vector decomposition is checked, but the scalar norms do not add: clipping or codec bias can sometimes offset part of an existing tracker error. Consequently, a decoded-to-GT score smaller than raw-source error is not evidence that a VAE recovered missing material identity or corrected tracking generally. GT-input runs provide the clearest codec-only diagnostic; tracker-input runs measure the full pipeline.

The shared calibration bounds are highly asymmetric, and zero motion is not middle gray. Physical inversion must use the saved per-channel bounds. Invalid slots and initially stationary background points also affect spatial compression; visual smoothness or RGB error alone is insufficient to judge physical flow fidelity.

### Temporal context, runtime and sampling

Extending GT input from 17 to 33 frames leaves Wan's shared-first17 decoded output unchanged in this control; LTX changes it by 1.7065 mm on average, with shared-window EPE changing from 6.6637 to 6.4488 mm. This is specific to the saved deterministic codec settings, not a general claim about predictive temporal reasoning. Recorded median encode/decode time is approximately 0.094 s for Wan and 0.024 s for LTX, with allocated peaks of 4.199 and 2.790 GB. These B300 measurements exclude loading and artifact I/O.

The twelve paired GT records repeat six archetype results across fixed/orbit cameras; they are not twelve independent motion samples. Tracker rows in the textured followup use only two clips, whereas GT rows use twelve. Neither those rows nor shared-front diagnostics should be pooled into one unqualified average. No population confidence intervals are inferred.

### What remains untested

Codec reconstruction of pseudo-RGB flow does not establish text-conditioned prediction quality, action learning benefits or OOD policy performance. Equal-rate codec sweeps, broader assets/motion amplitudes, multiple calibration/test seeds, clipping sensitivity, object-balanced metrics and downstream controlled LoRA experiments are needed to test those claims. The present report documents the completed codec measurements and their saved visual evidence.

## Complete saved media catalog

Every existing media file from this experiment is listed below. Galleries preserve the original debug/textured and geometry-regime names. Interactive point selection is available in the linked demo; the remaining clips are saved recordings or maps.

### Videos

- [animations/depth_speed_out_of_view_stress_orbit-3d-paths.mp4](../animations/depth_speed_out_of_view_stress_orbit-3d-paths.mp4)
- [animations/depth_speed_out_of_view_stress_orbit-motion-rgb.mp4](../animations/depth_speed_out_of_view_stress_orbit-motion-rgb.mp4)
- [animations/object_translation_orbit-3d-paths.mp4](../animations/object_translation_orbit-3d-paths.mp4)
- [animations/object_translation_orbit-motion-rgb.mp4](../animations/object_translation_orbit-motion-rgb.mp4)
- [animations/occlusion_and_return_orbit-3d-paths.mp4](../animations/occlusion_and_return_orbit-3d-paths.mp4)
- [animations/occlusion_and_return_orbit-motion-rgb.mp4](../animations/occlusion_and_return_orbit-motion-rgb.mp4)
### Animated comparisons


![ animations/depth_speed_out_of_view_stress_orbit-3d-paths.gif ](../animations/depth_speed_out_of_view_stress_orbit-3d-paths.gif)

![ animations/depth_speed_out_of_view_stress_orbit-motion-rgb.gif ](../animations/depth_speed_out_of_view_stress_orbit-motion-rgb.gif)

![ animations/object_translation_orbit-3d-paths.gif ](../animations/object_translation_orbit-3d-paths.gif)

![ animations/object_translation_orbit-motion-rgb.gif ](../animations/object_translation_orbit-motion-rgb.gif)

![ animations/occlusion_and_return_orbit-3d-paths.gif ](../animations/occlusion_and_return_orbit-3d-paths.gif)

![ animations/occlusion_and_return_orbit-motion-rgb.gif ](../animations/occlusion_and_return_orbit-motion-rgb.gif)
### Saved figures and comparison maps


![ offline-analysis/paired-gt-codec-errors.png ](../offline-analysis/paired-gt-codec-errors.png)

![ pilot-v2/articulated_links_fixed.png ](../pilot-v2/articulated_links_fixed.png)

![ pilot-v2/articulated_links_orbit.png ](../pilot-v2/articulated_links_orbit.png)

![ pilot-v2/depth_speed_out_of_view_stress_fixed.png ](../pilot-v2/depth_speed_out_of_view_stress_fixed.png)

![ pilot-v2/depth_speed_out_of_view_stress_orbit.png ](../pilot-v2/depth_speed_out_of_view_stress_orbit.png)

![ pilot-v2/object_translation_fixed.png ](../pilot-v2/object_translation_fixed.png)

![ pilot-v2/object_translation_orbit.png ](../pilot-v2/object_translation_orbit.png)

![ pilot-v2/occlusion_and_return_fixed.png ](../pilot-v2/occlusion_and_return_fixed.png)

![ pilot-v2/occlusion_and_return_orbit.png ](../pilot-v2/occlusion_and_return_orbit.png)

![ pilot-v2/rigid_rotation_fixed.png ](../pilot-v2/rigid_rotation_fixed.png)

![ pilot-v2/rigid_rotation_orbit.png ](../pilot-v2/rigid_rotation_orbit.png)

![ pilot-v2/static_scene_fixed.png ](../pilot-v2/static_scene_fixed.png)

![ pilot-v2/static_scene_orbit.png ](../pilot-v2/static_scene_orbit.png)

![ practical-shared-front/articulated_links_fixed.png ](../practical-shared-front/articulated_links_fixed.png)

![ practical-shared-front/articulated_links_orbit.png ](../practical-shared-front/articulated_links_orbit.png)

![ practical-shared-front/depth_speed_out_of_view_stress_fixed.png ](../practical-shared-front/depth_speed_out_of_view_stress_fixed.png)

![ practical-shared-front/depth_speed_out_of_view_stress_orbit.png ](../practical-shared-front/depth_speed_out_of_view_stress_orbit.png)

![ practical-shared-front/object_translation_fixed.png ](../practical-shared-front/object_translation_fixed.png)

![ practical-shared-front/object_translation_orbit.png ](../practical-shared-front/object_translation_orbit.png)

![ practical-shared-front/occlusion_and_return_fixed.png ](../practical-shared-front/occlusion_and_return_fixed.png)

![ practical-shared-front/occlusion_and_return_orbit.png ](../practical-shared-front/occlusion_and_return_orbit.png)

![ practical-shared-front/rigid_rotation_fixed.png ](../practical-shared-front/rigid_rotation_fixed.png)

![ practical-shared-front/rigid_rotation_orbit.png ](../practical-shared-front/rigid_rotation_orbit.png)

![ practical-shared-front/static_scene_fixed.png ](../practical-shared-front/static_scene_fixed.png)

![ practical-shared-front/static_scene_orbit.png ](../practical-shared-front/static_scene_orbit.png)

![ textured64-bounded/articulated_links_fixed.png ](../textured64-bounded/articulated_links_fixed.png)

![ textured64-bounded/articulated_links_orbit.png ](../textured64-bounded/articulated_links_orbit.png)

![ textured64-bounded/depth_speed_out_of_view_stress_fixed.png ](../textured64-bounded/depth_speed_out_of_view_stress_fixed.png)

![ textured64-bounded/depth_speed_out_of_view_stress_orbit.png ](../textured64-bounded/depth_speed_out_of_view_stress_orbit.png)

![ textured64-bounded/object_translation_fixed.png ](../textured64-bounded/object_translation_fixed.png)

![ textured64-bounded/object_translation_orbit.png ](../textured64-bounded/object_translation_orbit.png)

![ textured64-bounded/occlusion_and_return_fixed.png ](../textured64-bounded/occlusion_and_return_fixed.png)

![ textured64-bounded/occlusion_and_return_orbit.png ](../textured64-bounded/occlusion_and_return_orbit.png)

![ textured64-bounded/rigid_rotation_fixed.png ](../textured64-bounded/rigid_rotation_fixed.png)

![ textured64-bounded/rigid_rotation_orbit.png ](../textured64-bounded/rigid_rotation_orbit.png)

![ textured64-bounded/static_scene_fixed.png ](../textured64-bounded/static_scene_fixed.png)

![ textured64-bounded/static_scene_orbit.png ](../textured64-bounded/static_scene_orbit.png)

Video provenance: Experiment 2 MP4 files are H.264 viewing copies of the original saved GIF animations, with even-dimension padding. They add no new model inference and preserve the original display timing; original GIFs remain included.

## Full measured tables

### context33/paired-first17.csv

[Original CSV](../context33/paired-first17.csv)

| clip | codec | short17_epe_m | long33_first17_epe_m | shared_first17_reconstruction_difference_m |
| --- | --- | --- | --- | --- |
| articulated_links_fixed | ltx | 0.007592324968365321 | 0.007492544229253632 | 0.0020087220030129515 |
| articulated_links_fixed | wan | 0.005075627762798918 | 0.005075627762798918 | 0.0 |
| articulated_links_orbit | ltx | 0.007592324968365321 | 0.007492544229253632 | 0.0020087220030129515 |
| articulated_links_orbit | wan | 0.005075627762798918 | 0.005075627762798918 | 0.0 |
| depth_speed_out_of_view_stress_fixed | ltx | 0.008218398506891962 | 0.007725134431560673 | 0.002231318048116245 |
| depth_speed_out_of_view_stress_fixed | wan | 0.005377903180290042 | 0.005377903180290042 | 0.0 |
| depth_speed_out_of_view_stress_orbit | ltx | 0.008218398506891962 | 0.007725134431560673 | 0.002231318048116245 |
| depth_speed_out_of_view_stress_orbit | wan | 0.005377903180290042 | 0.005377903180290042 | 0.0 |
| object_translation_fixed | ltx | 0.005984605259138936 | 0.0057915417705189315 | 0.0014569191316555962 |
| object_translation_fixed | wan | 0.003993338205927963 | 0.003993338205927963 | 0.0 |
| object_translation_orbit | ltx | 0.005984605259138936 | 0.0057915417705189315 | 0.0014569191316555962 |
| object_translation_orbit | wan | 0.003993338205927963 | 0.003993338205927963 | 0.0 |
| occlusion_and_return_fixed | ltx | 0.006657687634077854 | 0.006188865713330415 | 0.002043957650257567 |
| occlusion_and_return_fixed | wan | 0.0048096571765429835 | 0.0048096571765429835 | 0.0 |
| occlusion_and_return_orbit | ltx | 0.006657687634077854 | 0.006188865713330415 | 0.002043957650257567 |
| occlusion_and_return_orbit | wan | 0.0048096571765429835 | 0.0048096571765429835 | 0.0 |
| rigid_rotation_fixed | ltx | 0.006308936599514592 | 0.006121841337551158 | 0.001575807625761404 |
| rigid_rotation_fixed | wan | 0.004008842195921923 | 0.004008842195921923 | 0.0 |
| rigid_rotation_orbit | ltx | 0.006308936599514592 | 0.006121841337551158 | 0.001575807625761404 |
| rigid_rotation_orbit | wan | 0.004008842195921923 | 0.004008842195921923 | 0.0 |
| static_scene_fixed | ltx | 0.00522016168220535 | 0.005373086282796004 | 0.0009219899125968275 |
| static_scene_fixed | wan | 0.0035334085687964076 | 0.0035334085687964076 | 0.0 |
| static_scene_orbit | ltx | 0.00522016168220535 | 0.005373086282796004 | 0.0009219899125968275 |
| static_scene_orbit | wan | 0.0035334085687964076 | 0.0035334085687964076 | 0.0 |

### offline-analysis/paired-gt-errors.csv

[Original CSV](../offline-analysis/paired-gt-errors.csv)

| condition | clip | archetype | camera | metric | wan_mm | ltx_mm | delta_ltx_minus_wan_mm | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| debug_g16 | articulated_links_fixed | articulated_links | fixed | overall | 5.075627762798918 | 7.592324968365321 | 2.5166972055664028 | Wan |
| debug_g16 | articulated_links_fixed | articulated_links | fixed | foreground | 23.124705193436675 | 37.55928075423868 | 14.434575560802003 | Wan |
| debug_g16 | articulated_links_orbit | articulated_links | orbit | overall | 5.075627762798918 | 7.592324968365321 | 2.5166972055664028 | Wan |
| debug_g16 | articulated_links_orbit | articulated_links | orbit | foreground | 23.124705193436675 | 37.55928075423868 | 14.434575560802003 | Wan |
| debug_g16 | depth_speed_out_of_view_stress_fixed | depth_speed_out_of_view_stress | fixed | overall | 5.377903180290042 | 8.218398506891962 | 2.8404953266019195 | Wan |
| debug_g16 | depth_speed_out_of_view_stress_fixed | depth_speed_out_of_view_stress | fixed | foreground | 29.20246319382074 | 48.06629231267719 | 18.86382911885645 | Wan |
| debug_g16 | depth_speed_out_of_view_stress_orbit | depth_speed_out_of_view_stress | orbit | overall | 5.377903180290042 | 8.218398506891962 | 2.8404953266019195 | Wan |
| debug_g16 | depth_speed_out_of_view_stress_orbit | depth_speed_out_of_view_stress | orbit | foreground | 29.20246319382074 | 48.06629231267719 | 18.86382911885645 | Wan |
| debug_g16 | object_translation_fixed | object_translation | fixed | overall | 3.993338205927963 | 5.984605259138935 | 1.9912670532109722 | Wan |
| debug_g16 | object_translation_fixed | object_translation | fixed | foreground | 6.499096452430848 | 9.105687740950582 | 2.606591288519734 | Wan |
| debug_g16 | object_translation_orbit | object_translation | orbit | overall | 3.993338205927963 | 5.984605259138935 | 1.9912670532109722 | Wan |
| debug_g16 | object_translation_orbit | object_translation | orbit | foreground | 6.499096452430848 | 9.105687740950582 | 2.606591288519734 | Wan |
| debug_g16 | occlusion_and_return_fixed | occlusion_and_return | fixed | overall | 4.809657176542983 | 6.657687634077854 | 1.848030457534871 | Wan |
| debug_g16 | occlusion_and_return_fixed | occlusion_and_return | fixed | foreground | 10.201157146553271 | 13.28406492520345 | 3.0829077786501795 | Wan |
| debug_g16 | occlusion_and_return_orbit | occlusion_and_return | orbit | overall | 4.809657176542983 | 6.657687634077854 | 1.848030457534871 | Wan |
| debug_g16 | occlusion_and_return_orbit | occlusion_and_return | orbit | foreground | 10.201157146553271 | 13.28406492520345 | 3.0829077786501795 | Wan |
| debug_g16 | rigid_rotation_fixed | rigid_rotation | fixed | overall | 4.008842195921923 | 6.308936599514592 | 2.300094403592669 | Wan |
| debug_g16 | rigid_rotation_fixed | rigid_rotation | fixed | foreground | 8.31456214040903 | 16.882071584709458 | 8.567509444300427 | Wan |
| debug_g16 | rigid_rotation_orbit | rigid_rotation | orbit | overall | 4.008842195921923 | 6.308936599514592 | 2.300094403592669 | Wan |
| debug_g16 | rigid_rotation_orbit | rigid_rotation | orbit | foreground | 8.31456214040903 | 16.882071584709458 | 8.567509444300427 | Wan |
| debug_g16 | static_scene_fixed | static_scene | fixed | overall | 3.5334085687964074 | 5.220161682205349 | 1.6867531134089417 | Wan |
| debug_g16 | static_scene_fixed | static_scene | fixed | foreground | 3.760046035165923 | 5.862577378712121 | 2.102531343546198 | Wan |
| debug_g16 | static_scene_orbit | static_scene | orbit | overall | 3.5334085687964074 | 5.220161682205349 | 1.6867531134089417 | Wan |
| debug_g16 | static_scene_orbit | static_scene | orbit | foreground | 3.760046035165923 | 5.862577378712121 | 2.102531343546198 | Wan |
| textured_g64 | articulated_links_fixed | articulated_links | fixed | overall | 4.99556832894862 | 7.309396911185741 | 2.3138285822371207 | Wan |
| textured_g64 | articulated_links_fixed | articulated_links | fixed | foreground | 25.23305527245439 | 30.28777371789441 | 5.05471844544002 | Wan |
| textured_g64 | articulated_links_orbit | articulated_links | orbit | overall | 4.99556832894862 | 7.309396911185741 | 2.3138285822371207 | Wan |
| textured_g64 | articulated_links_orbit | articulated_links | orbit | foreground | 25.23305527245439 | 30.28777371789441 | 5.05471844544002 | Wan |
| textured_g64 | depth_speed_out_of_view_stress_fixed | depth_speed_out_of_view_stress | fixed | overall | 5.174593544075212 | 7.455385179330717 | 2.280791635255505 | Wan |
| textured_g64 | depth_speed_out_of_view_stress_fixed | depth_speed_out_of_view_stress | fixed | foreground | 20.4103452389762 | 27.495588071556146 | 7.085242832579947 | Wan |
| textured_g64 | depth_speed_out_of_view_stress_orbit | depth_speed_out_of_view_stress | orbit | overall | 5.174593544075212 | 7.455385179330717 | 2.280791635255505 | Wan |
| textured_g64 | depth_speed_out_of_view_stress_orbit | depth_speed_out_of_view_stress | orbit | foreground | 20.4103452389762 | 27.495588071556146 | 7.085242832579947 | Wan |
| textured_g64 | object_translation_fixed | object_translation | fixed | overall | 3.5927355750681804 | 6.100691529000126 | 2.5079559539319454 | Wan |
| textured_g64 | object_translation_fixed | object_translation | fixed | foreground | 6.863792028739696 | 10.439012269987797 | 3.5752202412481013 | Wan |
| textured_g64 | object_translation_orbit | object_translation | orbit | overall | 3.5927355750681804 | 6.100691529000126 | 2.5079559539319454 | Wan |
| textured_g64 | object_translation_orbit | object_translation | orbit | foreground | 6.863792028739696 | 10.439012269987797 | 3.5752202412481013 | Wan |
| textured_g64 | occlusion_and_return_fixed | occlusion_and_return | fixed | overall | 4.356430757150332 | 6.5877151628774895 | 2.2312844057271572 | Wan |
| textured_g64 | occlusion_and_return_fixed | occlusion_and_return | fixed | foreground | 9.62545265614128 | 14.564099712315905 | 4.938647056174625 | Wan |
| textured_g64 | occlusion_and_return_orbit | occlusion_and_return | orbit | overall | 4.356430757150332 | 6.5877151628774895 | 2.2312844057271572 | Wan |
| textured_g64 | occlusion_and_return_orbit | occlusion_and_return | orbit | foreground | 9.62545265614128 | 14.564099712315905 | 4.938647056174625 | Wan |
| textured_g64 | rigid_rotation_fixed | rigid_rotation | fixed | overall | 4.220110232867849 | 6.507447352768797 | 2.2873371199009487 | Wan |
| textured_g64 | rigid_rotation_fixed | rigid_rotation | fixed | foreground | 11.328413874786964 | 17.684015556273373 | 6.355601681486409 | Wan |
| textured_g64 | rigid_rotation_orbit | rigid_rotation | orbit | overall | 4.220110232867849 | 6.507447352768797 | 2.2873371199009487 | Wan |
| textured_g64 | rigid_rotation_orbit | rigid_rotation | orbit | foreground | 11.328413874786964 | 17.684015556273373 | 6.355601681486409 | Wan |
| textured_g64 | static_scene_fixed | static_scene | fixed | overall | 3.3229414092717264 | 5.172359449347271 | 1.8494180400755442 | Wan |
| textured_g64 | static_scene_fixed | static_scene | fixed | foreground | 3.3801595304417735 | 5.440020373098421 | 2.0598608426566476 | Wan |
| textured_g64 | static_scene_orbit | static_scene | orbit | overall | 3.3229414092717264 | 5.172359449347271 | 1.8494180400755442 | Wan |
| textured_g64 | static_scene_orbit | static_scene | orbit | foreground | 3.3801595304417735 | 5.440020373098421 | 2.0598608426566476 | Wan |

### pilot-v2/per-clip.csv

[Original CSV](../pilot-v2/per-clip.csv)

| clip | source | codec | tracking_epe_m | range_epe_m | raster_epe_m | codec_self_epe_m | decoded_gt_epe_m | foreground_decoded_gt_epe_m | background_decoded_gt_epe_m | finite_coverage | inference_seconds | peak_gpu_gb | decoder_overshoot_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| articulated_links_fixed | cotracker3_gt_geometry | ltx | 0.06270377234996427 | 0.0009197315713126128 | 0.0015914652172314466 | 0.010835652577395672 | 0.059530297047309444 | 0.07082172320531802 | 0.05880181794034115 | 1.0 | 3.2350877119997676 | 2.789756416 | 0.04827432071461397 |
| articulated_links_fixed | cotracker3_gt_geometry | wan | 0.06270377234996427 | 0.0009197315713126128 | 0.0015914652172314466 | 0.006874189346771908 | 0.060481553610099215 | 0.05575997756538634 | 0.06078617141943552 | 1.0 | 2.745816743999967 | 4.19947776 | 0.0018412272135416667 |
| articulated_links_fixed | delta_gt_geometry | ltx | 0.06964269213245504 | 0.00011571400726538603 | 0.0017528370041809756 | 0.010643207520596991 | 0.06762439512719481 | 0.04593636061100838 | 0.06902362316049716 | 1.0 | 3.1694871560002866 | 2.789756416 | 0.06023780972349877 |
| articulated_links_fixed | delta_gt_geometry | wan | 0.06964269213245504 | 0.00011571400726538603 | 0.0017528370041809756 | 0.006937079464758318 | 0.06884390703306258 | 0.0351986301687592 | 0.071014570056566 | 1.0 | 2.7144370249998246 | 4.19947776 | 0.0025583902994791665 |
| articulated_links_fixed | gt | ltx | 0.0 | 0.0 | 0.0007005657562168579 | 0.007010419006162941 | 0.007592324968365321 | 0.03755928075423868 | 0.005658972982179942 | 1.0 | 0.9533235139997487 | 2.789756416 | 0.07193860820695466 |
| articulated_links_fixed | gt | wan | 0.0 | 0.0 | 0.0007005657562168579 | 0.004510329486682916 | 0.005075627762798918 | 0.023124705193436675 | 0.003911171154370675 | 1.0 | 0.8639630799998486 | 4.19947776 | 0.0012688730277267157 |
| articulated_links_fixed | gt_uv_gt_front_depth | ltx | 0.020291009424510575 | 0.012169902835244386 | 0.0010870675497559373 | 0.00914194857320902 | 0.014635655791380736 | 0.07711354336137713 | 0.010604824335251938 | 1.0 | 0.02319599299971742 | 2.789756416 | 0.07643426633348652 |
| articulated_links_fixed | gt_uv_gt_front_depth | wan | 0.020291009424510575 | 0.012169902835244386 | 0.0010870675497559373 | 0.004881358913904046 | 0.012139055583593893 | 0.06240164753261217 | 0.008896307715915295 | 1.0 | 0.09345030100030272 | 4.19947776 | 0.001677868412990196 |
| articulated_links_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.1032045122302246 | 1.2675755539076459e-05 | 0.0019618854506359526 | 0.013777971838527024 | 0.09583025801891719 | 0.16295095871335444 | 0.0914998902321793 | 1.0 | 0.9456988809997711 | 2.789756416 | 0.04180190142463235 |
| articulated_links_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.1032045122302246 | 1.2675755539076459e-05 | 0.0019618854506359526 | 0.008202381020159587 | 0.10081209036371407 | 0.15524188312210763 | 0.09730049083091448 | 1.0 | 2.341264143999979 | 4.19947776 | 0.001060037051930147 |
| articulated_links_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.10343210225320579 | 5.077311829099375e-06 | 0.0019685654440190913 | 0.013623601171579017 | 0.09603710037467193 | 0.1671954861568266 | 0.09144623677582325 | 1.0 | 0.023518854000030842 | 2.789756416 | 0.04264023724724265 |
| articulated_links_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.10343210225320579 | 5.077311829099375e-06 | 0.0019685654440190913 | 0.007951729564297322 | 0.10120273447490201 | 0.15953850934863972 | 0.09743913609595119 | 1.0 | 0.0926031779999903 | 4.19947776 | 0.000945745729932598 |
| articulated_links_fixed | zero | ltx | 0.0172339825202364 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.022405972004980082 | 0.2892725839727984 | 0.005188771232862769 | 1.0 | 0.023504884000431048 | 2.789756416 | 0.058461507161458336 |
| articulated_links_fixed | zero | wan | 0.0172339825202364 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.020450076643642894 | 0.28296625150469634 | 0.0035135492332523466 | 1.0 | 0.09383115499986161 | 4.19947776 | 0.0013314041436887254 |
| articulated_links_orbit | cotracker3_gt_geometry | ltx | 0.16457290524481377 | 0.002135970369360593 | 0.001916798698301747 | 0.017356308149002936 | 0.153218995282087 | 0.04721849415893753 | 0.1600577372900321 | 1.0 | 0.024312925999765866 | 2.789756416 | 0.04103237974877451 |
| articulated_links_orbit | cotracker3_gt_geometry | wan | 0.16457290524481377 | 0.002135970369360593 | 0.001916798698301747 | 0.010743253425394053 | 0.15972276398474558 | 0.03564996485501885 | 0.16772746070279243 | 1.0 | 0.09375260999968305 | 4.19947776 | 0.0013287114162071078 |
| articulated_links_orbit | delta_gt_geometry | ltx | 0.2436215687736146 | 0.00016464086983240011 | 0.0013364742280626719 | 0.018069554541479682 | 0.23055125300896928 | 0.04497095336479373 | 0.242524175566658 | 1.0 | 0.023666240999773436 | 2.789756416 | 0.04118795955882353 |
| articulated_links_orbit | delta_gt_geometry | wan | 0.2436215687736146 | 0.00016464086983240011 | 0.0013364742280626719 | 0.010411747068599113 | 0.2387103391600772 | 0.04611171629028681 | 0.2511360567645798 | 1.0 | 0.09332255899971642 | 4.19947776 | 0.0009292901731004901 |
| articulated_links_orbit | gt | ltx | 0.0 | 0.0 | 0.0007005657562168579 | 0.007010419006162941 | 0.007592324968365321 | 0.03755928075423868 | 0.005658972982179942 | 1.0 | 0.04776711099930253 | 2.789756416 | 0.07193860820695466 |
| articulated_links_orbit | gt | wan | 0.0 | 0.0 | 0.0007005657562168579 | 0.004510329486682916 | 0.005075627762798918 | 0.023124705193436675 | 0.003911171154370675 | 1.0 | 0.0944873380003628 | 4.19947776 | 0.0012688730277267157 |
| articulated_links_orbit | gt_uv_gt_front_depth | ltx | 0.020350530952211506 | 0.009581305778452329 | 0.0015213207817053056 | 0.01055428318684364 | 0.015403301841364885 | 0.0455945940439929 | 0.013333411427901367 | 0.9478609625668449 | 0.024973517000034917 | 2.789756416 | 0.08637192670036764 |
| articulated_links_orbit | gt_uv_gt_front_depth | wan | 0.020350530952211506 | 0.009581305778452329 | 0.0015213207817053056 | 0.005867434657865768 | 0.012508862237223604 | 0.03015935114537601 | 0.011298759087655577 | 0.9478609625668449 | 0.09366740800032858 | 4.19947776 | 0.001289218079810049 |
| articulated_links_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.18803267796807008 | 2.599245238922198e-06 | 0.0024558857485000372 | 0.01866593385464684 | 0.17467782543699997 | 0.11388873123198046 | 0.17859970248248508 | 1.0 | 0.024491168000167818 | 2.789756416 | 0.037297267539828434 |
| articulated_links_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.18803267796807008 | 2.599245238922198e-06 | 0.0024558857485000372 | 0.011400376704272705 | 0.1833532399929089 | 0.10932215764375403 | 0.18812943885414468 | 1.0 | 0.09380027899987908 | 4.19947776 | 0.0009490368412990196 |
| articulated_links_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.18729926317853382 | 5.721264284202886e-06 | 0.002404993404021493 | 0.019384222614006436 | 0.17327232955628633 | 0.11556032414281328 | 0.1769956847442523 | 1.0 | 0.023382480999316613 | 2.789756416 | 0.03592816521139706 |
| articulated_links_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.18729926317853382 | 5.721264284202886e-06 | 0.002404993404021493 | 0.011676831591244347 | 0.18247765780746417 | 0.11048068978123107 | 0.18712262348657596 | 1.0 | 0.09264253599940275 | 4.19947776 | 0.0009565166398590686 |
| articulated_links_orbit | zero | ltx | 0.0172339825202364 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.022405972004980082 | 0.2892725839727984 | 0.005188771232862769 | 1.0 | 0.023600341999554075 | 2.789756416 | 0.058461507161458336 |
| articulated_links_orbit | zero | wan | 0.0172339825202364 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.020450076643642894 | 0.28296625150469634 | 0.0035135492332523466 | 1.0 | 0.09781820799980778 | 4.19947776 | 0.0013314041436887254 |
| depth_speed_out_of_view_stress_fixed | cotracker3_gt_geometry | ltx | 0.08748093587659113 | 0.016445306187311635 | 0.003116257428110639 | 0.014632827016876529 | 0.07157295543334179 | 0.11132933532599898 | 0.069346598159353 | 1.0 | 0.02401326900053391 | 2.789756416 | 0.07053689395680147 |
| depth_speed_out_of_view_stress_fixed | cotracker3_gt_geometry | wan | 0.08748093587659113 | 0.016445306187311635 | 0.003116257428110639 | 0.008180171652979185 | 0.07132565203828983 | 0.08727824635231454 | 0.07043230675670445 | 1.0 | 0.09350298800018209 | 4.19947776 | 0.008535048540900736 |
| depth_speed_out_of_view_stress_fixed | delta_gt_geometry | ltx | 0.04594534750872666 | 5.074611236164911e-05 | 0.002527599565301456 | 0.009270114881761386 | 0.04647276041298831 | 0.04502304322124141 | 0.04655394457572614 | 1.0 | 0.02408013900003425 | 2.789756416 | 0.08124258003982843 |
| depth_speed_out_of_view_stress_fixed | delta_gt_geometry | wan | 0.04594534750872666 | 5.074611236164911e-05 | 0.002527599565301456 | 0.005482122204068773 | 0.045837081146968085 | 0.03240663137212603 | 0.046589186334359246 | 1.0 | 0.09266185999968002 | 4.19947776 | 0.0032839307598039216 |
| depth_speed_out_of_view_stress_fixed | gt | ltx | 0.0 | 0.0 | 0.0011943326213085825 | 0.007081267558890811 | 0.008218398506891962 | 0.04806629231267719 | 0.00598691645376799 | 1.0 | 0.02416867400006595 | 2.789756416 | 0.06543656891467524 |
| depth_speed_out_of_view_stress_fixed | gt | wan | 0.0 | 0.0 | 0.0011943326213085825 | 0.004394541457455436 | 0.005377903180290042 | 0.029202463193820742 | 0.004043727819532325 | 1.0 | 0.12404324100043596 | 4.19947776 | 0.001304776060814951 |
| depth_speed_out_of_view_stress_fixed | gt_uv_gt_front_depth | ltx | 0.022916719209651062 | 0.013079225078036036 | 0.0016827532766756897 | 0.009634826479818922 | 0.018268820740156996 | 0.09010348063202453 | 0.014246079786212413 | 1.0 | 0.02357029699942359 | 2.789756416 | 0.08261527267156862 |
| depth_speed_out_of_view_stress_fixed | gt_uv_gt_front_depth | wan | 0.022916719209651062 | 0.013079225078036036 | 0.0016827532766756897 | 0.005148583039998089 | 0.014960027054336475 | 0.06813830469448902 | 0.011982043506487933 | 1.0 | 0.09351570000035281 | 4.19947776 | 0.0026203230315563725 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.07168227309959496 | 8.39406588277074e-05 | 0.0011262184870844925 | 0.010026418414259494 | 0.06752830066205605 | 0.41463569347024026 | 0.048090286664797734 | 1.0 | 0.023608735999914643 | 2.789756416 | 0.04112064137178309 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.07168227309959496 | 8.39406588277074e-05 | 0.0011262184870844925 | 0.005813921075181718 | 0.07130870970309071 | 0.41174682691212855 | 0.0522441751393846 | 1.0 | 0.09402108999984193 | 4.19947776 | 0.001021740483302696 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.07313334238931832 | 4.4838284586810253e-05 | 0.001118935262786702 | 0.01023719382727511 | 0.06830122919687003 | 0.4129838079245423 | 0.048999004788120366 | 1.0 | 0.023754162999466644 | 2.789756416 | 0.04391808603324142 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.07313334238931832 | 4.4838284586810253e-05 | 0.001118935262786702 | 0.0057268174399072915 | 0.07286074591686276 | 0.41143501464752746 | 0.053900586867945535 | 1.0 | 0.092919976000303 | 4.19947776 | 0.001014260684742647 |
| depth_speed_out_of_view_stress_fixed | zero | ltx | 0.022731985509740725 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.027940112095298723 | 0.4343528124159038 | 0.005181000877344842 | 1.0 | 0.02364238900008786 | 2.789756416 | 0.058461507161458336 |
| depth_speed_out_of_view_stress_fixed | zero | wan | 0.022731985509740725 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.026039529368592638 | 0.42825351868201356 | 0.0035155459670410654 | 1.0 | 0.09534477600027458 | 4.19947776 | 0.0013314041436887254 |
| depth_speed_out_of_view_stress_orbit | cotracker3_gt_geometry | ltx | 0.18902414029141354 | 0.01032088061869708 | 0.0032432244598846675 | 0.022433560761812958 | 0.16853994487993462 | 0.10368005328633866 | 0.172172098809176 | 1.0 | 0.02392158800012112 | 2.789756416 | 0.044745650945925246 |
| depth_speed_out_of_view_stress_orbit | cotracker3_gt_geometry | wan | 0.18902414029141354 | 0.01032088061869708 | 0.0032432244598846675 | 0.012657958436092194 | 0.17548012693446588 | 0.10322830886526946 | 0.17952622874634092 | 1.0 | 0.0979798109992771 | 4.19947776 | 0.002118278952205882 |
| depth_speed_out_of_view_stress_orbit | delta_gt_geometry | ltx | 0.10762417827331539 | 0.0012723764880458034 | 0.002604905299009426 | 0.013312655802664356 | 0.10287656864101218 | 0.04372831522447928 | 0.10618887083233801 | 1.0 | 0.02362506899953587 | 2.789756416 | 0.04613898782169118 |
| depth_speed_out_of_view_stress_orbit | delta_gt_geometry | wan | 0.10762417827331539 | 0.0012723764880458034 | 0.002604905299009426 | 0.008059581697057467 | 0.10628196764792171 | 0.03412170281028065 | 0.11032294247882962 | 1.0 | 0.09288076699976955 | 4.19947776 | 0.001327813840379902 |
| depth_speed_out_of_view_stress_orbit | gt | ltx | 0.0 | 0.0 | 0.0011943326213085825 | 0.007081267558890811 | 0.008218398506891962 | 0.04806629231267719 | 0.00598691645376799 | 1.0 | 0.024456273000396322 | 2.789756416 | 0.06543656891467524 |
| depth_speed_out_of_view_stress_orbit | gt | wan | 0.0 | 0.0 | 0.0011943326213085825 | 0.004394541457455436 | 0.005377903180290042 | 0.029202463193820742 | 0.004043727819532325 | 1.0 | 0.09542512700045336 | 4.19947776 | 0.001304776060814951 |
| depth_speed_out_of_view_stress_orbit | gt_uv_gt_front_depth | ltx | 0.024736520613137022 | 0.010807996611628156 | 0.0020115642767249515 | 0.01175398208805569 | 0.019290772126309644 | 0.103869262307294 | 0.014260081260526192 | 0.9478609625668449 | 0.02427195499967638 | 2.789756416 | 0.09033203125 |
| depth_speed_out_of_view_stress_orbit | gt_uv_gt_front_depth | wan | 0.024736520613137022 | 0.010807996611628156 | 0.0020115642767249515 | 0.005909451826379717 | 0.015239681504101996 | 0.07448252738395167 | 0.011715943459203876 | 0.9478609625668449 | 0.09352118500009965 | 4.19947776 | 0.001747580135569853 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.29211401016103483 | 1.4014570104226356e-05 | 0.003281964821460048 | 0.019525663107021335 | 0.2785602783460059 | 0.3576760811322735 | 0.2741297933899749 | 1.0 | 0.023397889000079886 | 2.789756416 | 0.04121817794500613 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.29211401016103483 | 1.4014570104226356e-05 | 0.003281964821460048 | 0.012332410549220718 | 0.2859869795978872 | 0.3566643761905169 | 0.2820290453886999 | 1.0 | 0.09377438199953758 | 4.19947776 | 0.000969082701439951 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.2956653589505932 | 2.3410158588155877e-05 | 0.0033087148027555946 | 0.019202664145478124 | 0.2820873526446343 | 0.3547779980661548 | 0.2780166765010291 | 1.0 | 0.023504211000727082 | 2.789756416 | 0.03871334300321691 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.2956653589505932 | 2.3410158588155877e-05 | 0.0033087148027555946 | 0.012073785996277994 | 0.2888548954734522 | 0.3565922977948147 | 0.2850616009434559 | 1.0 | 0.09266235499944742 | 4.19947776 | 0.000990923713235294 |
| depth_speed_out_of_view_stress_orbit | zero | ltx | 0.022731985509740725 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.027940112095298723 | 0.4343528124159038 | 0.005181000877344842 | 1.0 | 0.023517877000813314 | 2.789756416 | 0.058461507161458336 |
| depth_speed_out_of_view_stress_orbit | zero | wan | 0.022731985509740725 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.026039529368592638 | 0.42825351868201356 | 0.0035155459670410654 | 1.0 | 0.0955144930003371 | 4.19947776 | 0.0013314041436887254 |
| object_translation_fixed | cotracker3_gt_geometry | ltx | 0.024578541432768417 | 0.00029839378427158093 | 0.0007221802874799409 | 0.006749878871575823 | 0.02339003965350806 | 0.009600543727928316 | 0.024871555744686052 | 1.0 | 0.024253358000351 | 2.789756416 | 0.06289942124310662 |
| object_translation_fixed | cotracker3_gt_geometry | wan | 0.024578541432768417 | 0.00029839378427158093 | 0.0007221802874799409 | 0.004723902042949752 | 0.02466121132099695 | 0.007568687414182892 | 0.0264975981870183 | 1.0 | 0.09357211200040183 | 4.19947776 | 0.003058040843290441 |
| object_translation_fixed | delta_gt_geometry | ltx | 0.03554083440896503 | 2.459982373584896e-06 | 0.000701787384680945 | 0.0066668144863205984 | 0.036005585739290934 | 0.025144982997783748 | 0.03717242735614708 | 1.0 | 0.024170310000044992 | 2.789756416 | 0.05123542336856618 |
| object_translation_fixed | delta_gt_geometry | wan | 0.03554083440896503 | 2.459982373584896e-06 | 0.000701787384680945 | 0.005477028910376677 | 0.037038229092797664 | 0.021764537473252667 | 0.03867920422547606 | 1.0 | 0.09308302799945523 | 4.19947776 | 0.0013164445465686275 |
| object_translation_fixed | gt | ltx | 0.0 | 0.0 | 0.0003421131543096453 | 0.005762664811289386 | 0.005984605259138936 | 0.009105687740950582 | 0.0056492823478699146 | 1.0 | 0.02381103600055212 | 2.789756416 | 0.07062635234757966 |
| object_translation_fixed | gt | wan | 0.0 | 0.0 | 0.0003421131543096453 | 0.003757550844426741 | 0.003993338205927963 | 0.006499096452430848 | 0.0037241245100226934 | 1.0 | 0.09534884999993665 | 4.19947776 | 0.0012359619140625 |
| object_translation_fixed | gt_uv_gt_front_depth | ltx | 0.00023683579141461805 | 0.0 | 0.00034090335293052154 | 0.005786870271705104 | 0.00603886101734949 | 0.010075846217051475 | 0.00560513533473688 | 1.0 | 0.024613419999695907 | 2.789756416 | 0.07011922200520833 |
| object_translation_fixed | gt_uv_gt_front_depth | wan | 0.00023683579141461805 | 0.0 | 0.00034090335293052154 | 0.0038696874106392497 | 0.004077434594458126 | 0.007337564323581675 | 0.003727172722734109 | 1.0 | 0.09327638300055696 | 4.19947776 | 0.0012874229281556373 |
| object_translation_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.06196585429325639 | 0.0001290000615895156 | 0.0011327356011092283 | 0.008362568924227029 | 0.05952663207075276 | 0.07872457148207981 | 0.05746404353895728 | 1.0 | 0.023799261000021943 | 2.789756416 | 0.041018317727481615 |
| object_translation_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.06196585429325639 | 0.0001290000615895156 | 0.0011327356011092283 | 0.006420738639547982 | 0.0638562208470146 | 0.07819379324492094 | 0.06231582050674366 | 1.0 | 0.09511318799923174 | 4.19947776 | 0.0012937059589460784 |
| object_translation_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.06103544340565321 | 7.3517061971517195e-06 | 0.001103423952230037 | 0.008068433951718126 | 0.05888634903015669 | 0.07775063275073656 | 0.056859607803978675 | 1.0 | 0.02371429300001182 | 2.789756416 | 0.03972042308134191 |
| object_translation_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.06103544340565321 | 7.3517061971517195e-06 | 0.001103423952230037 | 0.006348186835033876 | 0.06309666811373355 | 0.07734116405736167 | 0.06156626772309581 | 1.0 | 0.09262856500026828 | 4.19947776 | 0.0012395522173713235 |
| object_translation_fixed | zero | ltx | 0.009439506485341793 | 0.0 | 1.6804538706085413e-08 | 0.0052381456135744894 | 0.014512160919614214 | 0.1015187042759047 | 0.005164350476376395 | 1.0 | 0.023330965000241122 | 2.789756416 | 0.058461507161458336 |
| object_translation_fixed | zero | wan | 0.009439506485341793 | 0.0 | 1.6804538706085413e-08 | 0.003544437438949087 | 0.012577047901572925 | 0.09698759363100552 | 0.0035081462942785136 | 1.0 | 0.09404255400022521 | 4.19947776 | 0.0013314041436887254 |
| object_translation_orbit | cotracker3_gt_geometry | ltx | 0.15330601116465475 | 0.02318066916036512 | 0.0014073886032756225 | 0.01549999545923518 | 0.12477282282238744 | 0.03274937809895409 | 0.13465963919763232 | 1.0 | 0.02406282299943996 | 2.789756416 | 0.046597649069393385 |
| object_translation_orbit | cotracker3_gt_geometry | wan | 0.15330601116465475 | 0.02318066916036512 | 0.0014073886032756225 | 0.009169328407404437 | 0.1316628879711165 | 0.034574637291012573 | 0.14209385705244995 | 1.0 | 0.09398790699924575 | 4.19947776 | 0.0041940726485906864 |
| object_translation_orbit | delta_gt_geometry | ltx | 0.12564600742030974 | 0.003307223994594131 | 0.0009422755874703446 | 0.008167899383376682 | 0.12200710950686039 | 0.06473065540256928 | 0.12816077812963547 | 1.0 | 0.02353472899994813 | 2.789756416 | 0.041353711894914214 |
| object_translation_orbit | delta_gt_geometry | wan | 0.12564600742030974 | 0.003307223994594131 | 0.0009422755874703446 | 0.007633665775522889 | 0.1248741810593824 | 0.06356120088883098 | 0.1314615260363838 | 1.0 | 0.09269182800017006 | 4.19947776 | 0.003237556008731618 |
| object_translation_orbit | gt | ltx | 0.0 | 0.0 | 0.0003421131543096453 | 0.005762664811289386 | 0.005984605259138936 | 0.009105687740950582 | 0.0056492823478699146 | 1.0 | 0.025012944000081916 | 2.789756416 | 0.07062635234757966 |
| object_translation_orbit | gt | wan | 0.0 | 0.0 | 0.0003421131543096453 | 0.003757550844426741 | 0.003993338205927963 | 0.006499096452430848 | 0.0037241245100226934 | 1.0 | 0.09565973600001598 | 4.19947776 | 0.0012359619140625 |
| object_translation_orbit | gt_uv_gt_front_depth | ltx | 0.042366910224968746 | 0.030476959080800496 | 0.0012622074455136284 | 0.010039432051415258 | 0.015368769698046465 | 0.03222857300995457 | 0.013440875751440151 | 0.9486391571553995 | 0.024902993999603495 | 2.789756416 | 0.08889620911841299 |
| object_translation_orbit | gt_uv_gt_front_depth | wan | 0.042366910224968746 | 0.030476959080800496 | 0.0012622074455136284 | 0.005545279152681023 | 0.013305514268149705 | 0.03117114264013819 | 0.011262605691253825 | 0.9486391571553995 | 0.09339416900002107 | 4.19947776 | 0.002189187442555147 |
| object_translation_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.122396917334068 | 0.0 | 0.0018815583189071615 | 0.011336393519646775 | 0.11765739211972971 | 0.13487458738778274 | 0.11580761080993889 | 1.0 | 0.02316037599939591 | 2.789756416 | 0.03361960018382353 |
| object_translation_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.122396917334068 | 0.0 | 0.0018815583189071615 | 0.008743178881859482 | 0.12329116937358443 | 0.1364993383502657 | 0.12187210989675094 | 1.0 | 0.09375358100078302 | 4.19947776 | 0.0010444790709252451 |
| object_translation_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.12226581947482006 | 0.0 | 0.0018571396581448804 | 0.011004120123461603 | 0.11785756269140805 | 0.1360830981637532 | 0.11589944731008173 | 1.0 | 0.02365638100036449 | 2.789756416 | 0.034142288507199754 |
| object_translation_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.12226581947482006 | 0.0 | 0.0018571396581448804 | 0.008086924050076702 | 0.12293795010158147 | 0.13767303144678927 | 0.12135484218845999 | 1.0 | 0.09311354600049526 | 4.19947776 | 0.0010040881587009803 |
| object_translation_orbit | zero | ltx | 0.009439506485341793 | 0.0 | 1.6804538706085413e-08 | 0.0052381456135744894 | 0.014512160919614214 | 0.1015187042759047 | 0.005164350476376395 | 1.0 | 0.023670145999858505 | 2.789756416 | 0.058461507161458336 |
| object_translation_orbit | zero | wan | 0.009439506485341793 | 0.0 | 1.6804538706085413e-08 | 0.003544437438949087 | 0.012577047901572925 | 0.09698759363100552 | 0.0035081462942785136 | 1.0 | 0.09322529900055088 | 4.19947776 | 0.0013314041436887254 |
| occlusion_and_return_fixed | cotracker3_gt_geometry | ltx | 0.045142274924863876 | 0.016901640563061766 | 0.001658946450892336 | 0.010185319193023887 | 0.029309689386991845 | 0.06137382750461291 | 0.02456977331743047 | 1.0 | 0.024386205000155314 | 2.789756416 | 0.08996282839307598 |
| occlusion_and_return_fixed | cotracker3_gt_geometry | wan | 0.045142274924863876 | 0.016901640563061766 | 0.001658946450892336 | 0.005460514249378126 | 0.02878401962392135 | 0.05937800171788729 | 0.02426143096655247 | 1.0 | 0.09333657299976039 | 4.19947776 | 0.005872838637408088 |
| occlusion_and_return_fixed | delta_gt_geometry | ltx | 0.004600120096438722 | 0.0 | 0.0008932801843787064 | 0.006045374710213917 | 0.008087342847487097 | 0.015525388551684394 | 0.006987805656431843 | 1.0 | 0.02459300999998959 | 2.789756416 | 0.07416459625842524 |
| occlusion_and_return_fixed | delta_gt_geometry | wan | 0.004600120096438722 | 0.0 | 0.0008932801843787064 | 0.0038900910127211627 | 0.006564858753333275 | 0.014878734045659961 | 0.005335851101424111 | 1.0 | 0.09269116899940855 | 4.19947776 | 0.0011764227175245097 |
| occlusion_and_return_fixed | gt | ltx | 0.0 | 0.0 | 0.0008130444806978543 | 0.006272083564337508 | 0.006657687634077854 | 0.01328406492520345 | 0.0056781362084332015 | 1.0 | 0.023258442000042123 | 2.789756416 | 0.06682870902267157 |
| occlusion_and_return_fixed | gt | wan | 0.0 | 0.0 | 0.0008130444806978543 | 0.004091161120825967 | 0.0048096571765429835 | 0.010201157146553272 | 0.004012652833150158 | 1.0 | 0.09307060899936914 | 4.19947776 | 0.0012245926202512254 |
| occlusion_and_return_fixed | gt_uv_gt_front_depth | ltx | 0.027583277286981737 | 0.015212407421716327 | 0.0013676041818107007 | 0.008962682661863938 | 0.018720717078272706 | 0.06739119579584045 | 0.011525950659154 | 1.0 | 0.023987818000023253 | 2.789756416 | 0.08122462852328431 |
| occlusion_and_return_fixed | gt_uv_gt_front_depth | wan | 0.027583277286981737 | 0.015212407421716327 | 0.0013676041818107007 | 0.004900701048125144 | 0.01655062503901683 | 0.06380853962693 | 0.009564672447760098 | 1.0 | 0.09333147199959058 | 4.19947776 | 0.0014427035462622549 |
| occlusion_and_return_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.027698673428313853 | 6.819849416691189e-05 | 0.0002783813433580555 | 0.00730850287498215 | 0.029110192716702836 | 0.1775140144105497 | 0.007172236466308079 | 1.0 | 0.023133717999371584 | 2.789756416 | 0.0630481196384804 |
| occlusion_and_return_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.027698673428313853 | 6.819849416691189e-05 | 0.0002783813433580555 | 0.003467466916753623 | 0.02899389358101887 | 0.17442249807746452 | 0.007495752046761689 | 1.0 | 0.09309951399973215 | 4.19947776 | 0.0013152477787990197 |
| occlusion_and_return_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.027169960405767822 | 2.664906567202486e-06 | 0.00025869793534948206 | 0.007132346641261405 | 0.028797182428981643 | 0.17846785371308038 | 0.006671952760897482 | 1.0 | 0.023676646000239998 | 2.789756416 | 0.056409050436580885 |
| occlusion_and_return_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.027169960405767822 | 2.664906567202486e-06 | 0.00025869793534948206 | 0.003230276488922718 | 0.028510050488801036 | 0.17434564320537163 | 0.006951745478525378 | 1.0 | 0.09284521599965956 | 4.19947776 | 0.0010615330116421568 |
| occlusion_and_return_fixed | zero | ltx | 0.023544034084559164 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.02875184662178123 | 0.1885489084243047 | 0.005129672268364716 | 1.0 | 0.023917411000184075 | 2.789756416 | 0.058461507161458336 |
| occlusion_and_return_fixed | zero | wan | 0.023544034084559164 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.02664805241198978 | 0.18319341554611704 | 0.003506563948684012 | 1.0 | 0.09312719199988351 | 4.19947776 | 0.0013314041436887254 |
| occlusion_and_return_orbit | cotracker3_gt_geometry | ltx | 0.13651750029711796 | 0.03206655386850175 | 0.002740236814010747 | 0.01749328695716149 | 0.0997066481745349 | 0.08356003643147239 | 0.10209353860611803 | 1.0 | 0.024030847999711114 | 2.789756416 | 0.059317794500612746 |
| occlusion_and_return_orbit | cotracker3_gt_geometry | wan | 0.13651750029711796 | 0.03206655386850175 | 0.002740236814010747 | 0.00907849564413831 | 0.10466045566108796 | 0.08361627633104646 | 0.10777133434465933 | 1.0 | 0.09994403600012447 | 4.19947776 | 0.007625205844056373 |
| occlusion_and_return_orbit | delta_gt_geometry | ltx | 0.06887716678987564 | 0.01860188893161566 | 0.001528825957843707 | 0.008858581545262863 | 0.054395217134767415 | 0.04475866904642617 | 0.05581975033043526 | 1.0 | 0.02367978700021922 | 2.789756416 | 0.08220986758961397 |
| occlusion_and_return_orbit | delta_gt_geometry | wan | 0.06887716678987564 | 0.01860188893161566 | 0.001528825957843707 | 0.005844206026016184 | 0.05580098398710282 | 0.044255533933840824 | 0.0575077026906285 | 1.0 | 0.09298544800003583 | 4.19947776 | 0.013218300015318627 |
| occlusion_and_return_orbit | gt | ltx | 0.0 | 0.0 | 0.0008130444806978543 | 0.006272083564337508 | 0.006657687634077854 | 0.01328406492520345 | 0.0056781362084332015 | 1.0 | 0.023091838999789616 | 2.789756416 | 0.06682870902267157 |
| occlusion_and_return_orbit | gt | wan | 0.0 | 0.0 | 0.0008130444806978543 | 0.004091161120825967 | 0.0048096571765429835 | 0.010201157146553272 | 0.004012652833150158 | 1.0 | 0.09301478199995472 | 4.19947776 | 0.0012245926202512254 |
| occlusion_and_return_orbit | gt_uv_gt_front_depth | ltx | 0.04121232785748686 | 0.017172328959568096 | 0.002172809212652437 | 0.011270611516460093 | 0.027304586876814883 | 0.08823895497428846 | 0.017685232191665252 | 0.9478609625668449 | 0.022805163999692013 | 2.789756416 | 0.08707921645220588 |
| occlusion_and_return_orbit | gt_uv_gt_front_depth | wan | 0.04121232785748686 | 0.017172328959568096 | 0.002172809212652437 | 0.006067935891567002 | 0.02538152336860688 | 0.08966739594212342 | 0.01523308614283991 | 0.9478609625668449 | 0.0933228349995261 | 4.19947776 | 0.002159567440257353 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.052097754223666186 | 0.0004579017214851002 | 0.0015833820498386913 | 0.010725694108510198 | 0.04686514878423052 | 0.1318796834353063 | 0.034297782792332365 | 1.0 | 0.023476862000279652 | 2.789756416 | 0.0500787473192402 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.052097754223666186 | 0.0004579017214851002 | 0.0015833820498386913 | 0.00584424812709492 | 0.05167146699264164 | 0.1290846356808671 | 0.04022778118655614 | 1.0 | 0.09343701600027998 | 4.19947776 | 0.0011399213005514705 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.05227617581581014 | 0.0005236969223143674 | 0.001545606572328958 | 0.010640109018144375 | 0.04686798117751776 | 0.13194375208345346 | 0.03429156286968379 | 1.0 | 0.02356467600020551 | 2.789756416 | 0.05023941339231005 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.05227617581581014 | 0.0005236969223143674 | 0.001545606572328958 | 0.005196847307753968 | 0.05170924465433751 | 0.1297779290976919 | 0.04016865651923295 | 1.0 | 0.09268454099947121 | 4.19947776 | 0.0011046166513480392 |
| occlusion_and_return_orbit | zero | ltx | 0.023544034084559164 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.02875184662178123 | 0.1885489084243047 | 0.005129672268364716 | 1.0 | 0.023989669000002323 | 2.789756416 | 0.058461507161458336 |
| occlusion_and_return_orbit | zero | wan | 0.023544034084559164 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.02664805241198978 | 0.18319341554611704 | 0.003506563948684012 | 1.0 | 0.09299879799982591 | 4.19947776 | 0.0013314041436887254 |
| rigid_rotation_fixed | cotracker3_gt_geometry | ltx | 0.015877708197511987 | 0.0017563040960477806 | 0.0005588202803562883 | 0.00631502028810169 | 0.015157155536467617 | 0.07453047304831159 | 0.010290490166644345 | 1.0 | 0.024265906999971776 | 2.789756416 | 0.09689570408241421 |
| rigid_rotation_fixed | cotracker3_gt_geometry | wan | 0.015877708197511987 | 0.0017563040960477806 | 0.0005588202803562883 | 0.0039666083616287885 | 0.013946037030194206 | 0.073527660258021 | 0.009062297421355944 | 1.0 | 0.10069862500040472 | 4.19947776 | 0.006731519512101716 |
| rigid_rotation_fixed | delta_gt_geometry | ltx | 0.012536886738762128 | 2.2977815674631998e-05 | 0.0007296232327619205 | 0.006518240411480888 | 0.01417382255370044 | 0.02932682034668207 | 0.012931773554275718 | 1.0 | 0.023590924999552954 | 2.789756416 | 0.06638650333180147 |
| rigid_rotation_fixed | delta_gt_geometry | wan | 0.012536886738762128 | 2.2977815674631998e-05 | 0.0007296232327619205 | 0.004020175074516509 | 0.013044963327748264 | 0.02689498701697939 | 0.011909715484368663 | 1.0 | 0.09259382400068716 | 4.19947776 | 0.0013071695963541667 |
| rigid_rotation_fixed | gt | ltx | 0.0 | 0.0 | 0.0003394664330362086 | 0.0061086417787484 | 0.006308936599514592 | 0.016882071584709456 | 0.005442286190892062 | 1.0 | 0.02320968499952869 | 2.789756416 | 0.06803355497472427 |
| rigid_rotation_fixed | gt | wan | 0.0 | 0.0 | 0.0003394664330362086 | 0.0037685335310023486 | 0.004008842195921923 | 0.008314562140409031 | 0.0036559143316197006 | 1.0 | 0.09335343899965665 | 4.19947776 | 0.001289218079810049 |
| rigid_rotation_fixed | gt_uv_gt_front_depth | ltx | 0.0012384674107313337 | 0.0 | 0.00043971265815836346 | 0.0062163607248815955 | 0.007211753002925285 | 0.015017565106384606 | 0.006571932338707308 | 1.0 | 0.02403970700015634 | 2.789756416 | 0.07133035098805147 |
| rigid_rotation_fixed | gt_uv_gt_front_depth | wan | 0.0012384674107313337 | 0.0 | 0.00043971265815836346 | 0.0039105209426764515 | 0.005112856495247263 | 0.008419929774055538 | 0.004841784915017078 | 1.0 | 0.0932916079991628 | 4.19947776 | 0.0012709673713235295 |
| rigid_rotation_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.01373673989652808 | 0.00040020211820250773 | 0.0002871388835693386 | 0.007175161026578414 | 0.013033110026889668 | 0.08846096702261347 | 0.0068504987977319794 | 1.0 | 0.023609777999809012 | 2.789756416 | 0.06328926834405638 |
| rigid_rotation_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.01373673989652808 | 0.00040020211820250773 | 0.0002871388835693386 | 0.0036087449013789 | 0.01424993820417812 | 0.08466952602325148 | 0.00847784084195899 | 1.0 | 0.09352757499982545 | 4.19947776 | 0.0017801920572916667 |
| rigid_rotation_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.01320863749639891 | 0.00015153684040397462 | 0.00024245862893443684 | 0.007263414504806365 | 0.01244090229265985 | 0.08821389024276252 | 0.006230001641012091 | 1.0 | 0.023559936999845377 | 2.789756416 | 0.05945961148131127 |
| rigid_rotation_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.01320863749639891 | 0.00015153684040397462 | 0.00024245862893443684 | 0.003175809182953934 | 0.013891824956174796 | 0.08485061141530148 | 0.008075530984115231 | 1.0 | 0.09277898999971512 | 4.19947776 | 0.0012221990847120097 |
| rigid_rotation_fixed | zero | ltx | 0.006318177715559713 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.011464977889073098 | 0.088373811437731 | 0.005160975139183105 | 1.0 | 0.023034345999803918 | 2.789756416 | 0.058461507161458336 |
| rigid_rotation_fixed | zero | wan | 0.006318177715559713 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.009622152173929753 | 0.08402899741497097 | 0.003523230432860801 | 1.0 | 0.09306297300008737 | 4.19947776 | 0.0013314041436887254 |
| rigid_rotation_orbit | cotracker3_gt_geometry | ltx | 0.13089364468189743 | 0.00578379566790597 | 0.0011138857883882059 | 0.014520951106811673 | 0.11843281468377374 | 0.05917026114986749 | 0.12329040103901197 | 1.0 | 0.024103412999465945 | 2.789756416 | 0.04973288143382353 |
| rigid_rotation_orbit | cotracker3_gt_geometry | wan | 0.13089364468189743 | 0.00578379566790597 | 0.0011138857883882059 | 0.008719394677908979 | 0.124998220081999 | 0.05826544267770014 | 0.1304681198692366 | 1.0 | 0.09901003300001321 | 4.19947776 | 0.003411087335324755 |
| rigid_rotation_orbit | delta_gt_geometry | ltx | 0.12087138719229958 | 0.003619665474230494 | 0.0009097094589512808 | 0.011271128132328098 | 0.11460951560506971 | 0.07250433903903074 | 0.1180607595858926 | 1.0 | 0.023845455999435217 | 2.789756416 | 0.04259266572840074 |
| rigid_rotation_orbit | delta_gt_geometry | wan | 0.12087138719229958 | 0.003619665474230494 | 0.0009097094589512808 | 0.007011164892325103 | 0.12001468402331779 | 0.07003149183395509 | 0.124111666989659 | 1.0 | 0.09281622999969841 | 4.19947776 | 0.0033240224800857843 |
| rigid_rotation_orbit | gt | ltx | 0.0 | 0.0 | 0.0003394664330362086 | 0.0061086417787484 | 0.006308936599514592 | 0.016882071584709456 | 0.005442286190892062 | 1.0 | 0.023377007999442867 | 2.789756416 | 0.06803355497472427 |
| rigid_rotation_orbit | gt | wan | 0.0 | 0.0 | 0.0003394664330362086 | 0.0037685335310023486 | 0.004008842195921923 | 0.008314562140409031 | 0.0036559143316197006 | 1.0 | 0.0928000570002041 | 4.19947776 | 0.001289218079810049 |
| rigid_rotation_orbit | gt_uv_gt_front_depth | ltx | 0.021297281739868588 | 0.009949835630905603 | 0.0012456248055716314 | 0.0090487332905312 | 0.014620617807053286 | 0.01789492645595311 | 0.014335119505241859 | 0.9478609625668449 | 0.023782322999977623 | 2.789756416 | 0.0885168337354473 |
| rigid_rotation_orbit | gt_uv_gt_front_depth | wan | 0.021297281739868588 | 0.009949835630905603 | 0.0012456248055716314 | 0.004842921640099037 | 0.01206172916392766 | 0.009008269812460477 | 0.01232797085124905 | 0.9478609625668449 | 0.09292690300026152 | 4.19947776 | 0.0013622209137561275 |
| rigid_rotation_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.1137580013626429 | 0.0008640156167640464 | 0.0013085055470977193 | 0.012661514775062688 | 0.10540275119462104 | 0.11673628375030091 | 0.10447377311628661 | 1.0 | 0.024191267000787775 | 2.789756416 | 0.0401760924096201 |
| rigid_rotation_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.1137580013626429 | 0.0008640156167640464 | 0.0013085055470977193 | 0.007556247991746455 | 0.11252984836855222 | 0.11755778554076386 | 0.11211772237082994 | 1.0 | 0.09569925899995724 | 4.19947776 | 0.0011479994829963235 |
| rigid_rotation_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.11470642687815595 | 0.000927560623632885 | 0.001290102827696061 | 0.012591696862250173 | 0.10650138215909971 | 0.11811613421473073 | 0.10554935330208076 | 1.0 | 0.02395074299965927 | 2.789756416 | 0.040354410807291664 |
| rigid_rotation_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.11470642687815595 | 0.000927560623632885 | 0.001290102827696061 | 0.007083860279002182 | 0.11320684972920364 | 0.11742283956601529 | 0.11286127679176007 | 1.0 | 0.09274638700026117 | 4.19947776 | 0.001190783930759804 |
| rigid_rotation_orbit | zero | ltx | 0.006318177715559713 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.011464977889073098 | 0.088373811437731 | 0.005160975139183105 | 1.0 | 0.023153226999966137 | 2.789756416 | 0.058461507161458336 |
| rigid_rotation_orbit | zero | wan | 0.006318177715559713 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.009622152173929753 | 0.08402899741497097 | 0.003523230432860801 | 1.0 | 0.09309974600000714 | 4.19947776 | 0.0013314041436887254 |
| static_scene_fixed | cotracker3_gt_geometry | ltx | 0.006334662804410377 | 0.0007934248494689889 | 0.00023632855479249263 | 0.005449765244612036 | 0.007562199114450631 | 0.005974979043498018 | 0.007664600409350799 | 1.0 | 0.02406470799996896 | 2.789756416 | 0.08837591433057598 |
| static_scene_fixed | cotracker3_gt_geometry | wan | 0.006334662804410377 | 0.0007934248494689889 | 0.00023632855479249263 | 0.0033242491128740205 | 0.0062363902512360845 | 0.003991997969570797 | 0.006381189753279007 | 1.0 | 0.09753632899992226 | 4.19947776 | 0.005410886278339461 |
| static_scene_fixed | delta_gt_geometry | ltx | 0.0050970326530066215 | 0.0 | 9.827721362433894e-05 | 0.0050531577263152735 | 0.007062748924588693 | 0.007388993756273311 | 0.007041700870931621 | 1.0 | 0.023710657999799878 | 2.789756416 | 0.08430720310585171 |
| static_scene_fixed | delta_gt_geometry | wan | 0.0050970326530066215 | 0.0 | 9.827721362433894e-05 | 0.003314662487217883 | 0.0059032452654989125 | 0.006305763313556671 | 0.005877276359172606 | 1.0 | 0.09303553399968223 | 4.19947776 | 0.001418768190870098 |
| static_scene_fixed | gt | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.00522016168220535 | 0.0058625773787121215 | 0.005178715508237171 | 1.0 | 0.023080191000190098 | 2.789756416 | 0.058461507161458336 |
| static_scene_fixed | gt | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.0035334085687964076 | 0.003760046035165923 | 0.003518786796772568 | 1.0 | 0.09269642799972644 | 4.19947776 | 0.0013314041436887254 |
| static_scene_fixed | gt_uv_gt_front_depth | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.00522016168220535 | 0.0058625773787121215 | 0.005178715508237171 | 1.0 | 0.024393235000388813 | 2.789756416 | 0.058461507161458336 |
| static_scene_fixed | gt_uv_gt_front_depth | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.0035334085687964076 | 0.003760046035165923 | 0.003518786796772568 | 1.0 | 0.09337819200027297 | 4.19947776 | 0.0013314041436887254 |
| static_scene_fixed | spatrackerv2_gt_geometry_bf16 | ltx | 0.003906798832328312 | 0.0003417902204973595 | 0.00018697367410139805 | 0.006092310914051502 | 0.006197052039353081 | 0.006055050909697754 | 0.006206213402556651 | 1.0 | 0.023714556000413722 | 2.789756416 | 0.0712429869408701 |
| static_scene_fixed | spatrackerv2_gt_geometry_bf16 | wan | 0.003906798832328312 | 0.0003417902204973595 | 0.00018697367410139805 | 0.0033439953993304695 | 0.005239886045387169 | 0.003968742031588169 | 0.005321895336600008 | 1.0 | 0.09390487699965888 | 4.19947776 | 0.0018104104434742646 |
| static_scene_fixed | spatrackerv2_gt_geometry_fp32_math | ltx | 0.0028098278818425356 | 3.40160131048778e-05 | 0.00014292901216707597 | 0.0058325228009890534 | 0.005902409439133052 | 0.006359565493976921 | 0.005872915500110867 | 1.0 | 0.023913940000056755 | 2.789756416 | 0.0711511350145527 |
| static_scene_fixed | spatrackerv2_gt_geometry_fp32_math | wan | 0.0028098278818425356 | 3.40160131048778e-05 | 0.00014292901216707597 | 0.003281000192110846 | 0.004717514064809853 | 0.0038053219210351886 | 0.004776365170859831 | 1.0 | 0.0933831000002101 | 4.19947776 | 0.0014789057712928922 |
| static_scene_fixed | zero | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.00522016168220535 | 0.0058625773787121215 | 0.005178715508237171 | 1.0 | 0.02307384899995668 | 2.789756416 | 0.058461507161458336 |
| static_scene_fixed | zero | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.0035334085687964076 | 0.003760046035165923 | 0.003518786796772568 | 1.0 | 0.09265311799936171 | 4.19947776 | 0.0013314041436887254 |
| static_scene_orbit | cotracker3_gt_geometry | ltx | 0.12447739794273374 | 0.008198416841671953 | 0.001205718565876597 | 0.015333535934067267 | 0.1092703031547093 | 0.020159412592239193 | 0.11501939286841706 | 1.0 | 0.0245817929999248 | 2.789756416 | 0.05292466107536765 |
| static_scene_orbit | cotracker3_gt_geometry | wan | 0.12447739794273374 | 0.008198416841671953 | 0.001205718565876597 | 0.008409046423675971 | 0.11710069469329903 | 0.02524654696615164 | 0.12302676874021178 | 1.0 | 0.10322326999994402 | 4.19947776 | 0.0038030287798713237 |
| static_scene_orbit | delta_gt_geometry | ltx | 0.05063344869964048 | 0.0014598693788601931 | 0.0007129688522200673 | 0.008262887776805205 | 0.046417833391238866 | 0.01460724552888621 | 0.04847012938235839 | 1.0 | 0.02345822000006592 | 2.789756416 | 0.061452229817708336 |
| static_scene_orbit | delta_gt_geometry | wan | 0.05063344869964048 | 0.0014598693788601931 | 0.0007129688522200673 | 0.005543592698105565 | 0.04889319201973856 | 0.01896649262596303 | 0.05082394681933698 | 1.0 | 0.09279957699982333 | 4.19947776 | 0.0022639854281556373 |
| static_scene_orbit | gt | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.00522016168220535 | 0.0058625773787121215 | 0.005178715508237171 | 1.0 | 0.024105296000016097 | 2.789756416 | 0.058461507161458336 |
| static_scene_orbit | gt | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.0035334085687964076 | 0.003760046035165923 | 0.003518786796772568 | 1.0 | 0.09287278400006471 | 4.19947776 | 0.0013314041436887254 |
| static_scene_orbit | gt_uv_gt_front_depth | ltx | 0.0212373713628134 | 0.010484878507201638 | 0.0010165312754211106 | 0.009215002960304083 | 0.013651924574007658 | 0.01073203580490961 | 0.01385210977081781 | 0.9478609625668449 | 0.024086487000204215 | 2.789756416 | 0.09376974666819853 |
| static_scene_orbit | gt_uv_gt_front_depth | wan | 0.0212373713628134 | 0.010484878507201638 | 0.0010165312754211106 | 0.004623708999034156 | 0.011376174940860852 | 0.005593794150039435 | 0.011772610260210153 | 0.9478609625668449 | 0.09350728600020375 | 4.19947776 | 0.0019402597464767157 |
| static_scene_orbit | spatrackerv2_gt_geometry_bf16 | ltx | 0.06559286500349147 | 0.0014056310348237414 | 0.0011301010540160812 | 0.013452029647336 | 0.05649848876649449 | 0.026911881919089626 | 0.05840730211148836 | 1.0 | 0.023572721000164165 | 2.789756416 | 0.04015125947840074 |
| static_scene_orbit | spatrackerv2_gt_geometry_bf16 | wan | 0.06559286500349147 | 0.0014056310348237414 | 0.0011301010540160812 | 0.007121807978695613 | 0.06471693650612562 | 0.028724694404774372 | 0.06703901664169667 | 1.0 | 0.09417972400024155 | 4.19947776 | 0.0012416465609681373 |
| static_scene_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.0659057610836188 | 0.0015661803984337232 | 0.0010681689577679112 | 0.01294878194091236 | 0.05670368832148429 | 0.027535996028842206 | 0.05858547492100958 | 1.0 | 0.024566651999521127 | 2.789756416 | 0.04221538468903186 |
| static_scene_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.0659057610836188 | 0.0015661803984337232 | 0.0010681689577679112 | 0.006434389583895673 | 0.0644587224370834 | 0.029740887459531747 | 0.06669858275821576 | 1.0 | 0.0933109070001592 | 4.19947776 | 0.0013765821269914216 |
| static_scene_orbit | zero | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005220160404893241 | 0.00522016168220535 | 0.0058625773787121215 | 0.005178715508237171 | 1.0 | 0.02309483199951501 | 2.789756416 | 0.058461507161458336 |
| static_scene_orbit | zero | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.00353341261636979 | 0.0035334085687964076 | 0.003760046035165923 | 0.003518786796772568 | 1.0 | 0.09304495500055054 | 4.19947776 | 0.0013314041436887254 |

### practical-shared-front/per-clip.csv

[Original CSV](../practical-shared-front/per-clip.csv)

| clip | source | codec | tracking_epe_m | range_epe_m | raster_epe_m | codec_self_epe_m | decoded_gt_epe_m | foreground_decoded_gt_epe_m | background_decoded_gt_epe_m | finite_coverage | inference_seconds | peak_gpu_gb | decoder_overshoot_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| articulated_links_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.43420910789348116 | 0.29962823587148085 | 0.0021712314189736254 | 0.02535036923363881 | 0.18836358280186444 | 0.36251536485301233 | 0.1771279839598549 | 1.0 | 7.845168288999957 | 2.789756416 | 0.10143773696001838 |
| articulated_links_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.43420910789348116 | 0.29962823587148085 | 0.0021712314189736254 | 0.018875700901560436 | 0.18698796941744514 | 0.3527147534423611 | 0.17629591883519252 | 1.0 | 8.141552414000216 | 4.19947776 | 0.023711559819240197 |
| articulated_links_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.05646619058962124 | 0.00011830792518043326 | 0.0010015176136115263 | 0.013877418261107416 | 0.050410393587328114 | 0.19015369638447513 | 0.04139469663267347 | 1.0 | 0.02401467199979379 | 2.789756416 | 0.05196934120327819 |
| articulated_links_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.05646619058962124 | 0.00011830792518043326 | 0.0010015176136115263 | 0.008038369825596687 | 0.0540069298306016 | 0.17997386925676517 | 0.045880030512784595 | 1.0 | 0.10001762400042935 | 4.19947776 | 0.001815496706495098 |
| articulated_links_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.5334573510185144 | 0.3604894728847363 | 0.0023507564848731333 | 0.03349491747474848 | 0.23147334570256384 | 0.3587280796376844 | 0.22326336286803994 | 1.0 | 0.024241269000413013 | 2.789756416 | 0.11554912492340687 |
| articulated_links_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.5334573510185144 | 0.3604894728847363 | 0.0023507564848731333 | 0.020867092042891937 | 0.23491678244747613 | 0.35886159138512486 | 0.22692034316117624 | 1.0 | 0.10041710099994816 | 4.19947776 | 0.020408181583180147 |
| articulated_links_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.06726322316025013 | 0.005232243394747264 | 0.0008731725614860817 | 0.01240643897220303 | 0.0610293960415641 | 0.24623183717518898 | 0.04908085145229798 | 1.0 | 0.024263997999696585 | 2.789756416 | 0.05994220808440564 |
| articulated_links_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.06726322316025013 | 0.005232243394747264 | 0.0008731725614860817 | 0.009132520378339986 | 0.06252843444465642 | 0.2427775928896327 | 0.050899456480464414 | 1.0 | 0.10058177500013699 | 4.19947776 | 0.010036992091758578 |
| depth_speed_out_of_view_stress_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.11906377146347684 | 0.03989917123684835 | 0.0027096276558275455 | 0.015907816708058558 | 0.07693950359008925 | 0.32045746925884394 | 0.06330249751263897 | 1.0 | 0.03877011599979596 | 2.789756416 | 0.08444902008655024 |
| depth_speed_out_of_view_stress_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.11906377146347684 | 0.03989917123684835 | 0.0027096276558275455 | 0.00961291618670253 | 0.07903797588574558 | 0.3052998253800931 | 0.06636731231406211 | 1.0 | 0.09993185499934043 | 4.19947776 | 0.012099920534620098 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.0824459946458639 | 0.0007977443478221844 | 0.0012809171653964742 | 0.010775329511338869 | 0.07670682212090515 | 0.425270235194879 | 0.057187270988762626 | 1.0 | 0.024077901000055135 | 2.789756416 | 0.0472256529564951 |
| depth_speed_out_of_view_stress_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.0824459946458639 | 0.0007977443478221844 | 0.0012809171653964742 | 0.006789266084948118 | 0.08058508344013651 | 0.42121358772150497 | 0.06150988720037989 | 1.0 | 0.1002446249995046 | 4.19947776 | 0.0015210918351715686 |
| depth_speed_out_of_view_stress_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.13962769412582254 | 0.016758750980040305 | 0.0029974245501837235 | 0.02137972275370156 | 0.11609762784448463 | 0.2768993615987951 | 0.10709273075424323 | 1.0 | 0.024135239999850455 | 2.789756416 | 0.05372051164215686 |
| depth_speed_out_of_view_stress_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.13962769412582254 | 0.016758750980040305 | 0.0029974245501837235 | 0.015855495674417183 | 0.12082201355585204 | 0.2647029681612131 | 0.1127646800979518 | 1.0 | 0.09975522400054615 | 4.19947776 | 0.0022065405752144606 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.5166521902104013 | 0.06911450388712681 | 0.0022954976937488623 | 0.019130904026280413 | 0.4903002522265396 | 0.625007708363411 | 0.4827566346828749 | 1.0 | 0.02415149599983124 | 2.789756416 | 0.07321885052849264 |
| depth_speed_out_of_view_stress_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.5166521902104013 | 0.06911450388712681 | 0.0022954976937488623 | 0.013543106432109401 | 0.49361856655680025 | 0.6217635002482903 | 0.4864424502700768 | 1.0 | 0.10042916400016111 | 4.19947776 | 0.03651757333792892 |
| object_translation_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.04072081935988092 | 0.0014959017041577165 | 0.0006472542528054767 | 0.007919731743102239 | 0.038600525267157904 | 0.09555591007992081 | 0.032481351692232964 | 1.0 | 0.02482861899989075 | 2.789756416 | 0.07607374004289216 |
| object_translation_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.04072081935988092 | 0.0014959017041577165 | 0.0006472542528054767 | 0.005532541690434753 | 0.03936025088489338 | 0.09226307332745469 | 0.03367647657288267 | 1.0 | 0.10005154899954505 | 4.19947776 | 0.006191777248008579 |
| object_translation_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.04687403963194269 | 0.0021906037692852845 | 0.000634130118442015 | 0.006263740162115515 | 0.04524533526720963 | 0.1440712046338957 | 0.034627679880706176 | 1.0 | 0.02405685900066601 | 2.789756416 | 0.07765706380208333 |
| object_translation_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.04687403963194269 | 0.0021906037692852845 | 0.000634130118442015 | 0.005143985209376987 | 0.04523554590805829 | 0.1430926824577738 | 0.034721969253956625 | 1.0 | 0.10078034300022409 | 4.19947776 | 0.014918009440104166 |
| object_translation_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.10257853960358762 | 0.004341679532254643 | 0.000939947518662527 | 0.01298080387426958 | 0.09240202612186858 | 0.09888359126453418 | 0.09170565961893758 | 1.0 | 0.024340222999853722 | 2.789756416 | 0.06133434819240196 |
| object_translation_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.10257853960358762 | 0.004341679532254643 | 0.000939947518662527 | 0.009433076944499868 | 0.09691921322286741 | 0.09782882530297583 | 0.0968214863051698 | 1.0 | 0.10080954399927577 | 4.19947776 | 0.0029404584099264708 |
| object_translation_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.051127296471702985 | 0.0044312122787884085 | 0.0006056800265040852 | 0.00919343967080965 | 0.04511549004161835 | 0.11258430673246249 | 0.037866774198800385 | 1.0 | 0.024194973999328795 | 2.789756416 | 0.06313309015012254 |
| object_translation_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.051127296471702985 | 0.0044312122787884085 | 0.0006056800265040852 | 0.007213693935159373 | 0.04797730354917722 | 0.10915316997854078 | 0.04140468980056791 | 1.0 | 0.10043938900071225 | 4.19947776 | 0.007845710305606617 |
| occlusion_and_return_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.058992598370062224 | 0.016298092985918604 | 0.0015924363188316547 | 0.01204551928830775 | 0.03973430973593108 | 0.11986502503539471 | 0.027888899648184286 | 1.0 | 0.024186892999750853 | 2.789756416 | 0.07254686542585784 |
| occlusion_and_return_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.058992598370062224 | 0.016298092985918604 | 0.0015924363188316547 | 0.005894903331802271 | 0.04149062384748037 | 0.11690102206582331 | 0.030342999763029673 | 1.0 | 0.10005037200062361 | 4.19947776 | 0.0028115066827512254 |
| occlusion_and_return_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.033614824326959496 | 0.00022028253464889008 | 0.0003595600172074361 | 0.007906077552518126 | 0.03376187556303127 | 0.18833334261646303 | 0.010912180433393535 | 1.0 | 0.02396470900021086 | 2.789756416 | 0.07463731952742034 |
| occlusion_and_return_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.033614824326959496 | 0.00022028253464889008 | 0.0003595600172074361 | 0.004695980283994082 | 0.03325157102448475 | 0.1833490710023649 | 0.011063244940798115 | 1.0 | 0.09991730800084042 | 4.19947776 | 0.0037784950405943627 |
| occlusion_and_return_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.11735515750224645 | 0.025147317706190872 | 0.0024364132035220656 | 0.015885055195612854 | 0.08870421667546761 | 0.13625038688383487 | 0.08167565238379593 | 1.0 | 0.024167160999240878 | 2.789756416 | 0.07367302389705882 |
| occlusion_and_return_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.11735515750224645 | 0.025147317706190872 | 0.0024364132035220656 | 0.010715224771417889 | 0.09287730501731377 | 0.13660376971612578 | 0.0864133928444459 | 1.0 | 0.10058886000024359 | 4.19947776 | 0.008554496017156863 |
| occlusion_and_return_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.06594389431320179 | 0.005077406723240997 | 0.0010933343839903435 | 0.009548533204066378 | 0.06401077853485353 | 0.16572542795825634 | 0.048974699924437454 | 1.0 | 0.02417520000017248 | 2.789756416 | 0.05491428749234069 |
| occlusion_and_return_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.06594389431320179 | 0.005077406723240997 | 0.0010933343839903435 | 0.006772499870289708 | 0.06336693303813308 | 0.1627649798575907 | 0.048673308725691526 | 1.0 | 0.10060254400013946 | 4.19947776 | 0.004466038124234069 |
| rigid_rotation_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.022666389731417047 | 0.0014063474663998932 | 0.0005204553729234578 | 0.006230172789525797 | 0.02250085680243727 | 0.0792784440452585 | 0.0178469562087634 | 1.0 | 0.024078492000626284 | 2.789756416 | 0.09249369303385417 |
| rigid_rotation_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.022666389731417047 | 0.0014063474663998932 | 0.0005204553729234578 | 0.004244181401062697 | 0.021357037748373366 | 0.07818473976287908 | 0.016699029386528633 | 1.0 | 0.10021311400032573 | 4.19947776 | 0.0057657279220281864 |
| rigid_rotation_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.02297305213550256 | 3.799062489468435e-05 | 0.000250024494347667 | 0.006943289201245395 | 0.023928105493737997 | 0.08954624418982501 | 0.018549569535042342 | 1.0 | 0.024199147999752313 | 2.789756416 | 0.06666834214154412 |
| rigid_rotation_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.02297305213550256 | 3.799062489468435e-05 | 0.000250024494347667 | 0.003993908398952929 | 0.023284829355002434 | 0.08552090293829698 | 0.01818351184817501 | 1.0 | 0.10001192999970954 | 4.19947776 | 0.0014705283969056373 |
| rigid_rotation_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.1080370539253664 | 0.0043321365325511355 | 0.0009105606687579209 | 0.014362308318958987 | 0.09604978567272074 | 0.06557276243793406 | 0.0985479023313098 | 1.0 | 0.024247915000159992 | 2.789756416 | 0.04432349111519608 |
| rigid_rotation_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.1080370539253664 | 0.0043321365325511355 | 0.0009105606687579209 | 0.008465843976158712 | 0.10329681415868337 | 0.06452150051844226 | 0.10647511855542446 | 1.0 | 0.10065471400048409 | 4.19947776 | 0.002465042413449755 |
| rigid_rotation_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.029860831516075277 | 0.002416080489417778 | 0.0004798245105857021 | 0.008308585652338222 | 0.02790095890479972 | 0.09033551242209267 | 0.02278337255092325 | 1.0 | 0.02422187699994538 | 2.789756416 | 0.06105370615042892 |
| rigid_rotation_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.029860831516075277 | 0.002416080489417778 | 0.0004798245105857021 | 0.005254813805231311 | 0.027953526759680766 | 0.08949655170198058 | 0.02290901651850865 | 1.0 | 0.09998109200023464 | 4.19947776 | 0.0070983288334865195 |
| static_scene_fixed | cotracker3_shared_front_initial_GT_scale | ltx | 0.008362451780055525 | 0.0005065617400704826 | 0.0002566394989068284 | 0.005222888195788198 | 0.009425957329763661 | 0.0076093309713426115 | 0.009543159030306954 | 1.0 | 0.024070073000075354 | 2.789756416 | 0.08452591241574754 |
| static_scene_fixed | cotracker3_shared_front_initial_GT_scale | wan | 0.008362451780055525 | 0.0005065617400704826 | 0.0002566394989068284 | 0.003618495175337989 | 0.008011342617416766 | 0.006524119544993646 | 0.008107292493056969 | 1.0 | 0.0997340849999091 | 4.19947776 | 0.004009172028186275 |
| static_scene_fixed | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.008461955040429067 | 2.375251201694041e-05 | 0.00014608747316622017 | 0.005426174158984124 | 0.009931985674826508 | 0.009469955183831738 | 0.009961794093600365 | 1.0 | 0.024095423000289884 | 2.789756416 | 0.08048023897058823 |
| static_scene_fixed | spatrackerv2_shared_front_initial_GT_scale | wan | 0.008461955040429067 | 2.375251201694041e-05 | 0.00014608747316622017 | 0.003671273242382122 | 0.00902460379889785 | 0.00889353392131715 | 0.009033059920032087 | 1.0 | 0.10077344599994831 | 4.19947776 | 0.0012248918121936275 |
| static_scene_orbit | cotracker3_shared_front_initial_GT_scale | ltx | 0.10384575140775301 | 0.005773952652652199 | 0.0009808103537029952 | 0.013766806327106884 | 0.09135905984083821 | 0.0193005484386898 | 0.09600799606033165 | 1.0 | 0.0244172279999475 | 2.789756416 | 0.05162616804534314 |
| static_scene_orbit | cotracker3_shared_front_initial_GT_scale | wan | 0.10384575140775301 | 0.005773952652652199 | 0.0009808103537029952 | 0.007636370198268612 | 0.09831002842412641 | 0.022898697021323088 | 0.10317527561140405 | 1.0 | 0.1004913110000416 | 4.19947776 | 0.0034739176432291665 |
| static_scene_orbit | spatrackerv2_shared_front_initial_GT_scale | ltx | 0.021913879365416864 | 0.0033112538001274297 | 0.0004901089980770962 | 0.008136032577280993 | 0.0171079879118683 | 0.015517087363353948 | 0.017210626656933743 | 1.0 | 0.024260839999442396 | 2.789756416 | 0.05942490521599265 |
| static_scene_orbit | spatrackerv2_shared_front_initial_GT_scale | wan | 0.021913879365416864 | 0.0033112538001274297 | 0.0004901089980770962 | 0.005392987599090005 | 0.019022355698609012 | 0.015724027375115912 | 0.019235151074318245 | 1.0 | 0.10082491499997559 | 4.19947776 | 0.007768518784466912 |

### textured64-bounded/per-clip.csv

[Original CSV](../textured64-bounded/per-clip.csv)

| clip | source | codec | tracking_epe_m | range_epe_m | raster_epe_m | codec_self_epe_m | decoded_gt_epe_m | foreground_decoded_gt_epe_m | background_decoded_gt_epe_m | finite_coverage | inference_seconds | peak_gpu_gb | decoder_overshoot_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| articulated_links_fixed | gt | ltx | 0.0 | 0.00032664241171904534 | 0.000771113041976154 | 0.00661477948232922 | 0.007309396911185741 | 0.03028777371789441 | 0.005671427077713637 | 1.0 | 0.9778490099997725 | 2.789756416 | 0.07022543514476103 |
| articulated_links_fixed | gt | wan | 0.0 | 0.00032664241171904534 | 0.000771113041976154 | 0.004122541017263835 | 0.00499556832894862 | 0.02523305527245439 | 0.0035529776099614474 | 1.0 | 0.901432456999828 | 4.19947776 | 0.0014621510225183824 |
| articulated_links_orbit | gt | ltx | 0.0 | 0.00032664241171904534 | 0.000771113041976154 | 0.00661477948232922 | 0.007309396911185741 | 0.03028777371789441 | 0.005671427077713637 | 1.0 | 0.023361785999441054 | 2.789756416 | 0.07022543514476103 |
| articulated_links_orbit | gt | wan | 0.0 | 0.00032664241171904534 | 0.000771113041976154 | 0.004122541017263835 | 0.00499556832894862 | 0.02523305527245439 | 0.0035529776099614474 | 1.0 | 0.09360097099852283 | 4.19947776 | 0.0014621510225183824 |
| depth_speed_out_of_view_stress_fixed | gt | ltx | 0.0 | 0.0 | 0.0012723814605055024 | 0.006666352129250901 | 0.007455385179330717 | 0.027495588071556144 | 0.005637283738057574 | 1.0 | 0.023388043999148067 | 2.789756416 | 0.06470384784773284 |
| depth_speed_out_of_view_stress_fixed | gt | wan | 0.0 | 0.0 | 0.0012723814605055024 | 0.004140731140243797 | 0.005174593544075212 | 0.0204103452389762 | 0.0037923649177310434 | 1.0 | 0.0934402630009572 | 4.19947776 | 0.0014205633425245097 |
| depth_speed_out_of_view_stress_orbit | gt | ltx | 0.0 | 0.0 | 0.0012723814605055024 | 0.006666352129250901 | 0.007455385179330717 | 0.027495588071556144 | 0.005637283738057574 | 1.0 | 0.023242625000420958 | 2.789756416 | 0.06470384784773284 |
| depth_speed_out_of_view_stress_orbit | gt | wan | 0.0 | 0.0 | 0.0012723814605055024 | 0.004140731140243797 | 0.005174593544075212 | 0.0204103452389762 | 0.0037923649177310434 | 1.0 | 0.09335511300014332 | 4.19947776 | 0.0014205633425245097 |
| object_translation_fixed | gt | ltx | 0.0 | 0.0 | 0.0002964285155076822 | 0.005838732328243025 | 0.006100691529000126 | 0.010439012269987797 | 0.00571461668939577 | 1.0 | 0.024761712000326952 | 2.789756416 | 0.07149011948529412 |
| object_translation_fixed | gt | wan | 0.0 | 0.0 | 0.0002964285155076822 | 0.0033495101340885176 | 0.0035927355750681803 | 0.006863792028739696 | 0.0033016384729667543 | 1.0 | 0.09320167800069612 | 4.19947776 | 0.0012694714116115197 |
| object_translation_orbit | cotracker3_gt_geometry | ltx | 0.028573651761218597 | 0.007530671987902814 | 0.0021738846932625676 | 0.010604961961323427 | 0.019464272210803494 | 0.019916678850036396 | 0.019423836240078034 | 0.9962488538164439 | 0.023709996999969007 | 2.789756416 | 0.06655165728400736 |
| object_translation_orbit | cotracker3_gt_geometry | wan | 0.028573651761218597 | 0.007530671987902814 | 0.0021738846932625676 | 0.006849884467033001 | 0.02138453568061847 | 0.015532568272378287 | 0.021907582785336378 | 0.9962488538164439 | 0.09373045999927854 | 4.19947776 | 0.0030023911420036763 |
| object_translation_orbit | delta_gt_geometry | ltx | 0.028243859986312892 | 0.00028440529131605504 | 0.0012678959865966369 | 0.007835375937886429 | 0.027414294166927044 | 0.023991110670050293 | 0.027718929323799303 | 1.0 | 0.02339656699950865 | 2.789756416 | 0.05358497769224877 |
| object_translation_orbit | delta_gt_geometry | wan | 0.028243859986312892 | 0.00028440529131605504 | 0.0012678959865966369 | 0.004634118014203604 | 0.027689128921250993 | 0.022263609699932768 | 0.028171955477469122 | 1.0 | 0.09348239199971431 | 4.19947776 | 0.0019800522748161763 |
| object_translation_orbit | gt | ltx | 0.0 | 0.0 | 0.0002964285155076822 | 0.005838732328243025 | 0.006100691529000126 | 0.010439012269987797 | 0.00571461668939577 | 1.0 | 0.023565797999253846 | 2.789756416 | 0.07149011948529412 |
| object_translation_orbit | gt | wan | 0.0 | 0.0 | 0.0002964285155076822 | 0.0033495101340885176 | 0.0035927355750681803 | 0.006863792028739696 | 0.0033016384729667543 | 1.0 | 0.09370944100010092 | 4.19947776 | 0.0012694714116115197 |
| object_translation_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.014896925192329002 | 9.523431789640286e-05 | 0.0015680146318251533 | 0.00719272057216557 | 0.016116380020111353 | 0.08266594417324438 | 0.010194016543520817 | 1.0 | 0.024030021999351447 | 2.789756416 | 0.06615433038449754 |
| object_translation_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.014896925192329002 | 9.523431789640286e-05 | 0.0015680146318251533 | 0.004298621967808332 | 0.015563346352787224 | 0.07921264781626813 | 0.009899082385100908 | 1.0 | 0.09312776899969322 | 4.19947776 | 0.0014160754633884803 |
| occlusion_and_return_fixed | gt | ltx | 0.0 | 0.0 | 0.0007330735326922309 | 0.005967963447413464 | 0.006587715162877489 | 0.014564099712315905 | 0.005540572296581685 | 1.0 | 0.023351220001131878 | 2.789756416 | 0.07157389322916667 |
| occlusion_and_return_fixed | gt | wan | 0.0 | 0.0 | 0.0007330735326922309 | 0.0036782151181421053 | 0.0043564307571503326 | 0.00962545265614128 | 0.003664711512712299 | 1.0 | 0.0932831760001136 | 4.19947776 | 0.0012560077742034314 |
| occlusion_and_return_orbit | cotracker3_gt_geometry | ltx | 0.04801178376716934 | 0.018638362145973548 | 0.003387517996495703 | 0.012331335116500215 | 0.0294977046727977 | 0.07180098646434584 | 0.023905213696750804 | 0.9942131580425867 | 0.022971303998929216 | 2.789756416 | 0.06718744016161152 |
| occlusion_and_return_orbit | cotracker3_gt_geometry | wan | 0.04801178376716934 | 0.018638362145973548 | 0.003387517996495703 | 0.007398297390012217 | 0.030274222273798324 | 0.06505978385010137 | 0.025675573398971186 | 0.9942131580425867 | 0.09324036699945282 | 4.19947776 | 0.003073299632352941 |
| occlusion_and_return_orbit | delta_gt_geometry | ltx | 0.032429520296062164 | 5.615946926888605e-05 | 0.0017475859977603638 | 0.007945372257434137 | 0.031224155177767367 | 0.031981188189865124 | 0.03112477158946929 | 1.0 | 0.023341943000559695 | 2.789756416 | 0.05608323041130515 |
| occlusion_and_return_orbit | delta_gt_geometry | wan | 0.032429520296062164 | 5.615946926888605e-05 | 0.0017475859977603638 | 0.004654717096379224 | 0.0320981618535841 | 0.03025398735963854 | 0.0323402657984943 | 1.0 | 0.09358767300000181 | 4.19947776 | 0.0018714455997242646 |
| occlusion_and_return_orbit | gt | ltx | 0.0 | 0.0 | 0.0007330735326922309 | 0.005967963447413464 | 0.006587715162877489 | 0.014564099712315905 | 0.005540572296581685 | 1.0 | 0.023289988001124584 | 2.789756416 | 0.07157389322916667 |
| occlusion_and_return_orbit | gt | wan | 0.0 | 0.0 | 0.0007330735326922309 | 0.0036782151181421053 | 0.0043564307571503326 | 0.00962545265614128 | 0.003664711512712299 | 1.0 | 0.09324039000057383 | 4.19947776 | 0.0012560077742034314 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_fp32_math | ltx | 0.026750818852022774 | 9.507243454797904e-05 | 0.00306874132488098 | 0.00802296442695945 | 0.027671610801943605 | 0.11423666640947128 | 0.016307316629804645 | 1.0 | 0.02353203299935558 | 2.789756416 | 0.06852034026501226 |
| occlusion_and_return_orbit | spatrackerv2_gt_geometry_fp32_math | wan | 0.026750818852022774 | 9.507243454797904e-05 | 0.00306874132488098 | 0.004709719735033148 | 0.026844362543008517 | 0.11358726637524523 | 0.015456720386750536 | 1.0 | 0.09356525399925886 | 4.19947776 | 0.0016042671951593138 |
| rigid_rotation_fixed | gt | ltx | 0.0 | 0.00027037901856320003 | 0.0003380350879574287 | 0.006016916745776242 | 0.006507447352768797 | 0.017684015556273372 | 0.005469500279995924 | 1.0 | 0.02317942499939818 | 2.789756416 | 0.0674884272556679 |
| rigid_rotation_fixed | gt | wan | 0.0 | 0.00027037901856320003 | 0.0003380350879574287 | 0.003709319660563402 | 0.004220110232867848 | 0.011328413874786964 | 0.0035599753128127635 | 1.0 | 0.09305063099964173 | 4.19947776 | 0.0013942344515931373 |
| rigid_rotation_orbit | gt | ltx | 0.0 | 0.00027037901856320003 | 0.0003380350879574287 | 0.006016916745776242 | 0.006507447352768797 | 0.017684015556273372 | 0.005469500279995924 | 1.0 | 0.023221932000524248 | 2.789756416 | 0.0674884272556679 |
| rigid_rotation_orbit | gt | wan | 0.0 | 0.00027037901856320003 | 0.0003380350879574287 | 0.003709319660563402 | 0.004220110232867848 | 0.011328413874786964 | 0.0035599753128127635 | 1.0 | 0.09337269800016657 | 4.19947776 | 0.0013942344515931373 |
| static_scene_fixed | gt | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005172359341195225 | 0.005172359449347271 | 0.005440020373098421 | 0.005146913780850888 | 1.0 | 0.02327040899945132 | 2.789756416 | 0.058461507161458336 |
| static_scene_fixed | gt | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.003322946926125327 | 0.0033229414092717266 | 0.0033801595304417736 | 0.003317501865378377 | 1.0 | 0.09301537099963753 | 4.19947776 | 0.0013314041436887254 |
| static_scene_orbit | gt | ltx | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.005172359341195225 | 0.005172359449347271 | 0.005440020373098421 | 0.005146913780850888 | 1.0 | 0.024040537000473705 | 2.789756416 | 0.058461507161458336 |
| static_scene_orbit | gt | wan | 0.0 | 0.0 | 1.6804538706085413e-08 | 0.003322946926125327 | 0.0033229414092717266 | 0.0033801595304417736 | 0.003317501865378377 | 1.0 | 0.09309256999949866 | 4.19947776 | 0.0013314041436887254 |

