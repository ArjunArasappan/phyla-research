# Temporal context control: 17 vs33 input frames

Both codecs receive identical first17 motionRGB frames and the same calibration bounds; the33-frame run additionally includes16 later frames. The table scores only the shared first17 window, excluding the query frame. This isolates the effect of available codec context, unlike comparing full17 vsfull33 averages with different displacement distributions. No temporal realignment or refit of normalization was performed.

| Codec | Paired clips | 17-frame EPE(mm) | 33-frame first17 EPE(mm) | Decoded output difference(mm) |
|---|---:|---:|---:|---:|
| wan | 12 | 4.4665 | 4.4665 | 0.0000 |
| ltx | 12 | 6.6637 | 6.4488 | 1.7065 |

All values are clip-macro means, not independent point-level replications. These two native codecs have different causal/temporal designs and compression rates; temporal-context differences do not isolate one architectural component by themselves. Full33 raw/decoded RGB, latents, flows and metrics are retained in runs/exp02/simulator-context33-v1/.
