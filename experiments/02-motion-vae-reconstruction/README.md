# Experiment 2: reconstruction of 3D motion labels through frozen video VAEs

**Status:** proposed, not executed. Defined 2026-10-08 by splitting the original tracker/codec study.

## 1. Question

How much physical motion information is lost when GT or tracker-generated 3D displacements are rendered as RGB and encoded/decoded by Wan and LTX VAEs?

This experiment uses [Experiment 1](../01-tracker-benchmark/README.md) trajectory artifacts; it does not estimate tracks itself. Evaluate GT first to establish the representation/codec floor, then every available tracker source and the zero-motion control. A smooth incorrect trajectory can reconstruct well, so self-reconstruction alone is insufficient.

Both VAEs remain frozen. There is no diffusion transformer, T5 loading, denoising generation, or LoRA training in this experiment. Wan/LTX LoRA SFT belongs to policy adaptation in Experiment 3.

## 2. Dependency and streaming execution

The first gate is a validated GT track bundle with correct units, query identities, grid layout, and camera convention. Fit RGB mapping bounds from a separate GT calibration split before the primary test scoring. Once this gate passes, the codec worker can validate encode/decode shapes and GT reconstruction without waiting for every tracker or every clip.

Next, process each completed Experiment 1 method bundle as it arrives. Key jobs by source hash, codec hash, statistics hash, rasterizer, dtype, and window policy. Retain a no-VAE inverse-render control for each source. Reuse cached tracker predictions and do not modify them.

Use GPUs 2/3 initially: one Wan and one LTX process, or independent source jobs when one codec is CPU/I/O-bound. No GPU job starts before its input bundle and model/environment acceptance markers are complete. This is a data dependency, not a requirement to run the entirety of Experiment 1 before any Experiment 2 work.

## 3. Convert tracks to pseudo-RGB motion videos

Arrange D[T,Q,3] on the **initial query grid**, G × G. R, G, and B represent dx, dy, and dz. The grid is attached to initial material-point queries; it is not a scatter plot of where the points project in a future frame. Reprojecting into future pixels would change the representation, lose correspondence at collisions, and make the codec comparison harder to interpret.

Fit channel-wise lower/upper bounds lⱼ,hⱼ using only GT calibration trajectories, with balanced clip/foreground sampling. The primary mapping uses the existing FloMo-style 1st/99th percentile choice:

```text
D_clip,j = clip(D_j, l_j, h_j)
RGB_j = (D_clip,j − l_j) / (h_j − l_j)
D_decoded,j = l_j + RGB_decoded,j (h_j − l_j)
```

Use one frozen mapping for GT and every model, every test clip, and both codecs. Compute the zero-displacement RGB code from those bounds; it is not necessarily gray. Give a degenerate channel a documented physical minimum range rather than dividing by zero. Save the exact bounds, fitting sample IDs, and statistics hash.

Include a symmetric-bound control, with channels centered at zero and calibrated absolute-displacement percentiles. This tests whether asymmetric scaling affects results. Do not make per-method or per-clip normalized images the primary comparison: that would hide scale errors and assign different physical meanings to the same RGB difference.

The primary rasterizer matches the current repository's grid-to-image approach: bilinear upsampling from G × G to 256 × 256 with explicitly recorded pixel-center convention. Save float32 RGB for scientific evaluation; MP4/JPEG files are previews only. A uint8 quantization branch is a separate ablation.

Invalid initial query slots receive the zero-displacement color and an accompanying mask. Predictions that become nonfinite are filled only in the render copy and retain a failure mask. Do not interpret the codec's output at an invalid slot as a recovered track. Score GT-valid points, track finite coverage, and report confidence-filtered training-label rendering as a separate practical regime.

### Measure rendering error before measuring a VAE

Let R turn grid flow into the image and U invert the RGB affine map and sample back at the query grid. The no-VAE roundtrip U(R(D_clip)) may differ from D_clip because bilinear interpolation and sampling are not necessarily exact inverses.

Implement and test U against the rasterizer's actual coordinate convention. Report its physical error as a representation floor. Include an exact integer block-replication rasterizer with center sampling as a control when 256/G is integral. This helps separate the learned codec from avoidable grid resampling loss.

Report channel-wise clipping frequency, clipped displacement magnitude, saturation maps, foreground versus background clipping, and the upper tail of unclipped displacement. Shared quantile bounds are useful for reproducing a training representation but deliberately truncate extremes; that loss must remain visible.

## 4. Frozen VAE comparison: Wan and LTX

Use the VAE associated with the repository's **Wan2.2 TI2V-5B** backend and the **LTX-Video 2B v0.9.6 non-distilled** backend. Pin exact VAE weights and component configs. Do not substitute a Wan2.1 VAE, another Wan2.2 variant, LTX-2, or a different LTX revision under the same model name.

These codecs are components of text/image-conditioned video systems, but this experiment requires only their encoders and decoders. No prompts or image-conditioning branches are needed to reconstruct motion videos. Load **neither T5 nor the diffusion transformer**. Use the same input pseudo-RGB video for both codecs and run deterministic posterior-mode encoding.

The targeted adapters have these nominal shapes at 256 × 256 with 17 input frames:

| Codec | Latent channels | Spatial stride | Temporal stride | Expected latent [C,T,H,W] | Scalar count |
|---|---:|---:|---:|---|---:|
| Wan2.2 target VAE | 48 | 16 | 4 | [48,5,16,16] | 61,440 |
| LTX v0.9.6 target VAE | 128 | 32 | 8 | [128,3,8,8] | 24,576 |

Confirm these shapes with actual pinned weights; reject mismatches. The model families and releases are documented in [Wan2.2](https://github.com/Wan-Video/Wan2.2), [LTX-Video](https://github.com/Lightricks/LTX-Video), and the [official LTX checkpoint collection](https://huggingface.co/Lightricks/LTX-Video).

A 17 × 3 × 256 × 256 input has 3,342,336 scalars. The nominal image-to-latent scalar compression is 54.4× for Wan and 136× for LTX. These are not entropy-coded file compression ratios or equal-capacity comparisons. Report actual bytes, precision, encoding/decoding latency, and compression relative to the original point grid too. At G = 64, the pre-rasterized source contains only 208,896 flow scalars.

### Codec protocol

1. Primary windows have 17 frames, compatible with both temporal layouts. Use the first 17 frames of each clip initially.
2. Run a 33-frame temporal-context control on the same subset. A legacy 16-frame branch may repeat the last frame to 17, then crop scores to the 16 valid frames; padding never counts as another observation.
3. Later windows require fresh queries at the new anchor and recomputed GT, or a verified multi-time-query adapter. Do not reshape frame-zero point IDs into a later-frame image grid.
4. Convert RGB [0,1] to the codec's expected input range, normally [-1,1], using the exact native preprocessing.
5. Apply native latent scaling and inverse scaling exactly once. LTX uses its native channel statistics in the current adapter.
6. Primary encoding is posterior mode. Record tiled/chunked settings; tiling is a separate condition if it changes reconstruction.
7. For the current LTX adapter, primary diagnostic decode uses timestep zero without stochastic decoder noise. Compare the pinned upstream recommended decode setting on a small subset and record any difference explicitly.
8. Save both decoder output before display clamping and the runtime-clamped RGB [0,1] output. Measure overshoot. Invert the affine map without secretly clipping raw decoded physical values.
9. Use BF16 inference initially where supported, with a small FP32 reference subset if feasible. Record actual dtype and differences. Cast metrics to float32/float64.
10. Batch size starts at one; profile before changing it. No training and no generated latent noise are involved.

Wan and LTX have different temporal and spatial compression. Interpret the primary result as a comparison of the deployed codecs at their native rates. Add simple spatial-only, temporal-only, and joint downsample/upsample controls at relevant grids to estimate how much error comes from reduced bandwidth versus learned reconstruction. Those controls are not capacity-matched trained VAEs.

## 5. Error decomposition and reconstruction metrics

For model m, denote raw motion Dₘ, range-clipped motion Dₘᶜ, pre-VAE render/inverse-render motion D̄ₘ, and decoded motion D̃ₘ. Then:

```text
e_tracking       = D_m − D_GT
e_range          = D_m^c − D_m
e_representation = Dbar_m − D_m^c
e_codec          = Dtilde_m − Dbar_m

Dtilde_m − D_GT = e_tracking + e_range + e_representation + e_codec
```

This vector identity should hold numerically on matching masks. If quantization is enabled, split it out as another term. Scalar EPE values do not add. MSE contains cross terms; a codec can partly cancel tracker noise while adding bias. Report those cancellations rather than attributing a signed improvement to inherently better labels.

**Primary physical reconstruction quantities:**

- representation-only EPE against clipped source flow;
- codec-only EPE against D̄ₘ, removing the no-VAE raster floor;
- decoded-versus-raw-source EPE, including range and representation loss;
- decoded-versus-GT EPE, the end-to-end result;
- the same metrics for GT as source, establishing the representation/codec floor;
- axis MAE/RMSE, p95, metric success thresholds, velocity error, and static false motion.

At a sampled query, the affine inverse implies e_D,j = (h_j − l_j) e_RGB,j when comparing decoded versus pre-codec RGB. Thus physical squared error is the sum of channel RGB squared errors weighted by their squared physical ranges. A single unweighted RGB MSE can obscure a much larger depth error when the depth channel has a larger range. Rasterization and clipping still require their separate terms.

**Image diagnostics:** RGB MSE/MAE, PSNR with data range 1, and SSIM. Publish both full-image values and foreground/valid-query-conditioned measurements. Image-space mask definitions differ from point-grid masks; report them explicitly. SSIM is a diagnostic of the image representation, not a measure of physical trajectory correctness. LPIPS/FVD are unnecessary for the primary geometry question and should not determine model selection.

**Motion-specific failure analysis:**

- attenuation of small displacements and high-frequency temporal motion;
- sign flips in channels above a predeclared GT magnitude threshold;
- temporal lag and endpoint drift;
- smoothing across object/motion boundaries;
- depth-channel loss relative to lateral channels;
- off-axis displacement introduced by reconstruction;
- phantom motion in the initially zero frame and static surfaces;
- reconstruction behavior at saturated values and decoder overshoot.

Estimate attenuation by projecting decoded motion onto nonzero source motion and reporting the signed gain, with amplitude-stratified plots. A temporal lag fit may diagnose the codec, but do not shift decoded trajectories before primary scoring. Report t = 0 separately and exclude it from the main motion averages. Average physical errors over queries, not over the larger upsampled RGB image.

Use a fixed common source-valid mask when comparing both codecs on one label set. End-to-end threshold success must still penalize source prediction failures. Missing labels should not become an apparently successful codec result merely because a filled zero image decoded smoothly.

### Optional controlled-label study

Render GT variants with known perturbations: Gaussian coordinate noise at several centimeter amplitudes; temporal jitter; sinusoidal motions spanning amplitudes and frequencies; thin object boundaries; and discontinuous neighboring motions. Evaluate codec response using the same frozen bounds. Keep these synthetic signals in a separate table from actual tracker labels. This identifies noise suppression and bandwidth limits without mistaking them for improved tracking.

## 6. Saved outputs and reports

For every source and representation variant, save raw/clipped flow, RGB float arrays, masks, statistics, and no-VAE recovered flow. For every codec, save deterministic latent codes, raw and clamped decoded RGB, inverse-mapped 3D displacement, physical-error arrays, summary metrics, checkpoint/scaling metadata, actual bytes, timing, and peak GPU memory. Scientific inputs are lossless arrays; video encodings are preview-only.

Build a contact sheet with rows GT and each label baseline, and columns source RGB, no-VAE roundtrip, Wan reconstruction, LTX reconstruction, and physical error maps. Show equal time indices and shared colorbars. Add axis-wise/time-wise errors, static phantom motion, boundary blending, displacement attenuation, clipping/overshoot maps, and decoded-to-GT versus self-error scatter plots.

Primary results distinguish tracker error, clipping loss, rasterization/quantization loss, and codec error. Their vector errors add; scalar EPE does not. A low self-error/high GT-error source must not be called a good label pipeline. Compare compression and cost explicitly because Wan and LTX have different latent rates.

## 7. Milestones and handoff

1. Validate renderer/inverse sampler on CPU and identity controls.
2. Validate real frozen Wan/LTX weights, native scaling, expected shapes, and raw decoder outputs on a GT clip.
3. Complete the GT and zero-motion codec pilot using frozen calibration bounds.
4. Consume the Experiment 1 tracker artifacts incrementally; produce per-source reconstruction and end-to-end tables.
5. Run selected range, rasterizer, precision, temporal-context, and controlled-noise ablations.
6. Publish the tested codec signature and physical fidelity report for Experiment 3.

Experiment 3 can start GT-flow policy pilots once the selected codec and label contract work; it need not wait for every optional Experiment 2 ablation. A large reconstruction error is a scientific result, not a reason to hide that codec or change normalization after inspecting test outcomes. Downstream training choices and any adaptations must be declared.
