# Verification

## LTX addition and deeper audit — 2026-10-08

The final suite passed **48 tests in 20.60 seconds**, with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. This includes all original 30 CPU tests and 18 LTX checks using Diffusers 0.33.1 and Transformers 4.49.0. No large model weights were downloaded.

The LTX checks exercise the actual upstream small transformer and VAE components, float32 visual-forward parity before adaptation, image/text conditioning, missing-label masking, timestep-zero future probes, frame padding, posterior-mode encoding and latent scale inversion, both ordinary and timestep-conditioned decoder APIs, the v0.9.5 residual codec architecture, bfloat16 autocast/backward, LoRA-only trunk gradients, activation checkpointing, signed Euler sampling, temporal rate validation and GPU template contracts. Conversion transaction tests use small tensors to check successful asset copying, refusal to overwrite, and cleanup after a failed write.

A second audit read only the 144,648-byte safetensors header from the official non-distilled v0.9.6 file, pinned to revision `8984fa25007f376c1a299016d0957a37a2f797bb`. The sanitized fixture in `tests/fixtures/ltx_096_shapes.json` contains only component configuration and tensor names/shapes/dtypes. It contains no tensor data, encrypted metadata or license text. All **944 tensor schemas** were converted and checked against full-sized Diffusers transformer/VAE models allocated on the meta device. This validates every expected name and shape without allocating the large weights.

That audit found and fixed the missing v0.9.5/v0.9.6 VAE encoder rename mapping in Diffusers 0.33.1. The repository conversion command now uses the upstream transformer/decoder mappings plus the corrected encoder mapping, requires exact state keys/shapes, and marks only a successfully committed output as complete. It also checks local T5/tokenizer assets before conversion. The completion marker has converter version 2; older incomplete conversions are rejected.

The audit also corrected causal temporal positions to frame starts 0, 1, 9, … before converting to seconds. RGB future targets also retain their one-frame offset relative to cumulative flow. An existing version-0.1.0 tiny bundle successfully reloaded and produced a finite action chunk through the installed package. Existing Wan/tiny model signatures remain compatible, and the shared stages/executor/evaluator remain covered by their original tests.

**Not established:** numeric loading or conversion of the actual complete 2B checkpoint, GPU memory/throughput, CUDA tracker behavior, ManiSkill rendering/control, large-scale training or robot success. Header/meta checks and small CPU components do not establish these outcomes. The full conversion and a small GPU training/rollout acceptance run are still required.

The package was rebuilt as version 0.1.1; current source hashes and package checksum are in `verification.json`. The prior CPU integration and distributed-run evidence below is retained as historical evidence, not presented as a GPU LTX result.

## Original CPU pipeline checks — 2026-10-07

Implemented and tested on a GPU-free Intel macOS laptop with Python 3.10.9, PyTorch 2.2.2, NumPy 1.26.4, PyYAML 6.0.3, h5py 3.16.0 and pytest 9.1.1. The dedicated `.venv` contains the CPU dependencies and the built package. `requirements-cpu.txt` records the dependency versions.

## Checks completed

| Check | Result |
| --- | --- |
| CPU tests | 30 passed in 4.74 seconds |
| Full stage handoffs | Collection → preparation → mid-training → SFT → export/reload → evaluation; separate privileged probe training completed |
| Distributed training | Two actual CPU Gloo workers, global batch 8, three optimizer steps, validation each step, checkpoint completed |
| Attention command | Four examples audited from the exported policy; per-layer stream and per-token masses saved |
| Packaging | Wheel built successfully and installed into the local virtual environment |
| Source identity | All 22 source/config/test/build files byte-identical between the destination and the verification checkout; SHA-256 hashes in `verification.json` |

The tests exercise geometry and camera transforms, stable rotation conversion, frame padding, missing action labels, clean-future probes, diffusion solver direction, loss/gradient accumulation, fixed-batch fitting, LoRA gradients on a lightweight Wan-shaped fixture, source sampling, split isolation, native action chunk execution/reset, exact resume, stage/bundle compatibility, BC training, replayed demonstration import, evaluation cohort accounting, oracle flow diagnostics and paired comparisons.

The Wan-shaped fixture checks adapter algebra and gradient paths; it is not a real Wan checkpoint or GPU inference test.

## Artifacts

* `runs/cpu_smoke_final/report.json`: complete CPU integration report.
* `runs/cpu_smoke_final/raw/` and `prepared/`: episodes, track caches, fitted statistics and encoded samples.
* `runs/cpu_smoke_final/midtrain_bundle/` and `policy/`: exported immutable bundles.
* `runs/cpu_smoke_final/probe/step_000002.pt`: privileged inverse-dynamics checkpoint.
* `runs/cpu_smoke_final/attention.json`: descriptive attention audit.
* `runs/cpu_ddp_final/step_000003.pt`: distributed training checkpoint, with per-rank RNG states.
* `runs/cpu_smoke_final/verification-config.yaml`: final dataset and distributed recipe paths.
* `dist/phyla_flomo-0.1.0-py3-none-any.whl`: installable package.

The toy expert succeeded in 3/3 episodes. The tiny policy after four mid-training and two SFT updates succeeded in 1/3. These are interface checks on an analytic point environment, not evidence of robot competence or reproduction of the paper's results. The distributed training loss changed from 1.5121 to 1.2973 across three updates.

## GPU acceptance work still required

1. Fill exact upstream/model revision fields, load the real TI2V-5B VAE/T5/DiT checkpoint, verify token layouts and numerical outputs, and profile memory/throughput with the intended precision and microbatch.
2. Run SpatialTracker on a short real clip with known camera units/frame conventions. Compare tracked cumulative displacement against controlled rigid motion, and inspect rendered motion targets.
3. Instantiate the selected ManiSkill task and controller. Verify RGB/depth/segmentation extraction, CV camera frames, rigid body/link IDs, control frequency and action replay on a known demonstration.
4. Overfit a small robot subset before scale-up; test learned native action replay and sealed-ledger closed-loop evaluation. Check successful and failed trajectories visually.
5. Run the intended source mixtures, label fractions, target ablations and multiple training seeds. CPU tests do not establish transfer, sample efficiency, manipulation success or causal use of motion.

Custom task assets, deformable correspondences, multi-view motion targets, paper-exact stochastic pixel augmentation, real-time asynchronous execution, remote serving and RL/recovery orchestration are not implemented by this core pipeline. Deliberate method adaptations are listed in `README.md`.

## Filesystem constraint during this session

Despite granted access, the automation sandbox could not list the destination root or read its Git metadata. Exact file reads/writes and newly created subdirectories worked. Tests and the wheel build therefore ran from a byte-for-byte source mirror in the Codex workspace; integration artifacts were written directly into this repository. The wheel is installed in this repository's virtual environment and can run from outside the verification mirror. No Git status, diff, commit or branch operation was possible, and no Git metadata was changed.
