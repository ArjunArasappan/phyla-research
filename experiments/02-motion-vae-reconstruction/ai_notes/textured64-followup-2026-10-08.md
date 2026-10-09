# Bounded textured64 followup

Thirty-six real frozen VAE jobs completed: GT on all12 textured64 clips, plus CoTracker3, DELTA and corrected SpaTrackerV2 FP32-math on two key orbit clips (translation and occlusion). The source query grid is64×64. All methods use the ORIGINAL frozen calibration bounds from the debug study; no condition-specific normalization is silently fit. These outputs have their own run namespace and scientific manifest.

Texture, query density and sampled surfaces change together in this followup. GT video-codec comparison can expose sensitivity to the grid/raster representation, but tracker differences cannot be attributed uniquely to texture or density without the paired same-grid controls. Two-clip tracker averages are not the same estimand as the12-clip GT average. Main debug results remain authoritative for their declared setting; no existing arrays are overwritten.

Both VAEs remain frozen. GPUs0/1 were used under a short lease and released after completion. Every decoder is finite and vector decomposition passed at≤1e-6m. Raw/clamped RGB, latents, recovered flows and exact per-file SHA256/byte ledgers are retained. Practical shared-front labels use a separate table with a fixed initial GT-scale diagnostic; they are not merged into supplied-GT-geometry results.
