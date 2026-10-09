# Experiment 1 primary textured64 measurements

[Completed experimental setup, hypotheses, measured results, videos and interactive viewer](/Users/arjunarasappan/Documents/Codex/2026-10-06/re/outputs/gpu-research-checkout/experiments/01-tracker-benchmark/results/ai/README.md)

| Method | Fixed EPE cm | Orbit EPE cm | Overall EPE cm | PCK <1 cm % | Visible coverage % |
|---|---|---|---|---|---|
| CoTracker3 + GT geometry | 0.627 | 2.637 | 1.632 | 64.71 | 99.947 |
| DELTA + GT geometry | 0.708 | 5.189 | 2.949 | 51.72 | 100.000 |
| SpaTrackerV2 + GT geometry (corrected float32 outer / SDPA) | 2.379 | 3.336 | 2.857 | 88.87 | 100.000 |
| GT UV + exact front ray | 0.000 | 0.000 | 0.000 | 100.00 | 100.000 |
| GT UV + rendered front depth | 0.036 | 0.512 | 0.274 | 96.06 | 99.960 |
| Zero motion | 1.897 | 1.931 | 1.914 | 95.96 | 100.000 |

Frame 0 excluded; GT-visible mask; EPE conditional on finite geometry; missing geometry counts as a PCK failure. Values average clip means. Raw CSVs and plots are retained beside this file.
