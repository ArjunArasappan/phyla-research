# FloMo research implementation

A compact simulation-first implementation of the [FloMo method](https://flomo-wam.github.io/FloMo%20Paper.pdf): jointly flow-match cumulative 3D scene flow and action chunks, starting from Wan or LTX video weights. The repository also includes a small CPU model and analytic toy environment for integration testing. Toy results are not robot benchmarks.

## Organization

The previous plan's many subpackages are collapsed into eight modules:

```text
flomo/
  config.py     strict settings, identities, artifact writes
  geometry.py   coordinate transforms, rigid oracle tracks, scaling, rotations
  data.py       episodes/windows, tracker adapter, preprocessing, mixed sampling
  model.py      frozen encoders, Wan/LTX joint transformers, LoRA, BC, solver
  train.py      mid-training/SFT/probe, DDP, validation, resume, export
  policy.py     bundle runtime, native action codec, chunk execution, comparisons
  sim.py        collection, ManiSkill/toy adapters, rollout evaluation, diagnostics
  cli.py        commands calling these shared functions
configs/        smoke, midtrain, sft, bc
tests/          geometry, model, contracts and complete stage handoffs
```

One trainer handles all stages. One immutable policy bundle is the training/evaluation boundary. No serving platform, database, job service, registry hierarchy or duplicated normalization implementation is needed to run experiments. External GPU libraries are imported lazily. Optional future infrastructure can call these functions without changing formats.

## Laptop setup and CPU smoke

Use Python 3.10+ and PyTorch 2.2+. On Intel macOS, PyTorch 2.2.2 is a suitable CPU test version. GPU Wan work should use the upstream-supported Linux/PyTorch/CUDA environment, not the macOS lock file.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
python -m flomo doctor --config configs/smoke.yaml
python -m pytest
python -m flomo smoke --workdir runs/smoke
```

The smoke command collects 24 toy episodes, including examples with action labels removed; fits statistics only on training splits; extracts/encodes rigid oracle flow; performs short mid-training and robot-only SFT; exports/reloads bundles; evaluates expert and learned policies; and trains a clean-future inverse-dynamics probe. It writes `runs/smoke/report.json` and all intermediate artifacts. A four-update policy is not expected to be competent; the test verifies interface correctness.

Run explicit stages:

```bash
python -m flomo collect --config configs/smoke.yaml --episodes 40 --action-free-fraction 0.25
python -m flomo prepare --config configs/smoke.yaml
python -m flomo validate-data --config configs/smoke.yaml
python -m flomo train --config configs/smoke.yaml
python -m flomo export --checkpoint runs/toy_midtrain/step_000100.pt --output runs/toy_policy
python -m flomo evaluate --config configs/smoke.yaml --bundle runs/toy_policy
```

All paths are relative to the invocation directory unless absolute. Commands refuse to overwrite complete datasets, runs, bundles and evaluation records. Choose a new output path for a new recipe. Interrupted preprocessing can resume with the same configuration and raw manifests; cached tracks are reused. `train.resume` resumes an interrupted stage, while `train.init` starts a new stage with a fresh optimizer.

## GPU setup and actual Wan model

On a Linux NVIDIA machine, install this package's `wan`/`sim` extras and each upstream's own requirements in a compatible environment. Tracker preprocessing may need its own environment; prepared samples are the exchange format. Do not copy the laptop virtual environment to Linux.

Clone [official Wan2.2](https://github.com/Wan-Video/Wan2.2) and [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) into `third_party/`, check out explicit commits, and obtain the TI2V-5B checkpoint/tokenizer at an explicit model revision. Then fill these fields in the GPU configs:

* `model.upstream`, `model.upstream_commit`
* `model.checkpoint`, `model.checkpoint_revision`
* `data.tracker_upstream`, `data.tracker_revision`
* Actual source IDs/weights and robot-only SFT source IDs

The code records these revision strings and rejects missing pins. You are responsible for checking that the checkout/model files match the supplied pins; it does not fetch/check out upstream code automatically. Existing checkpoint paths are expected to contain `config.json`, DiT weight shards, `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl` tokenizer files. No multi-gigabyte weights are downloaded by the CLI.

Wan integration uses original patch embedding, transformer blocks, conditioning projections and visual output head. It adds linear action projections/positions and LoRA to self/cross-attention q/k/v/o and FFN linears. Full attention uses PyTorch SDPA with explicit masks and mixed-stream rotary coordinates; it does not try to run arbitrary action tokens through Wan's original rectangular-video unpatchifier. Clean image-prefix timesteps are zero. Text is masked. Human samples' missing action tokens cannot participate as keys.

On eight GPUs the mid-training template's microbatch 2 × accumulation 32 gives global batch 512. SFT gives 256. Microbatch 2 is a starting value to profile, not a measured fit guarantee.

```bash
python -m flomo prepare --config configs/midtrain.yaml
torchrun --standalone --nproc-per-node=8 -m flomo train --config configs/midtrain.yaml
python -m flomo export --checkpoint runs/midtrain/step_014000.pt --output runs/midtrain_bundle
torchrun --standalone --nproc-per-node=8 -m flomo train --config configs/sft.yaml
python -m flomo export --checkpoint runs/sft/step_001500.pt --output runs/sft_bundle
python -m flomo evaluate --config configs/sft.yaml --bundle runs/sft_bundle
```

Each rank gets a deterministic slice of a global source-weighted sample batch. Losses are means per modality/example, weighted then averaged over the full mixed batch. Missing action labels contribute zero, with the original global-batch denominator. Accumulation and DDP preserve that objective. Checkpoints save rank-specific RNG states, optimizer, step, world size and sampler/config identities. Exact resume requires the same world size, selected data, objective, batch and seeds.

## Lighter text + image → video backbone: LTX-Video 2B

Set `model.backend: ltx` using `configs/ltx_midtrain.yaml` or `configs/ltx_sft.yaml`. This adapter targets **LTX-Video 2B v0.9.6 non-distilled**, specifically the official `ltxv-2b-0.9.6-dev-04-25.safetensors` checkpoint. It uses both image and text conditioning. The current backend does not accept distilled checkpoints or LTX-2.

Install on the GPU host:

```bash
python -m pip install -e '.[ltx,sim]'
```

The LTX extra pins Diffusers 0.33.1 and Transformers 4.49.0. Newer Diffusers releases need newer Torch APIs and are not covered by this adapter's tests. Wan and LTX use the same canonical episodes and flow tracks, but require separate encoded datasets and bundles.

Obtain pinned local snapshots from [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) for the v0.9.6 weight file and [Lightricks/LTX-Video-0.9.5](https://huggingface.co/Lightricks/LTX-Video-0.9.5) for compatible 2B Diffusers component configs, tokenizer and T5 assets. Keep each snapshot revision in your experiment notes. The assets folder needs component configuration JSON files, `text_encoder/` weights and `tokenizer/` files. Conversion obtains the transformer and VAE weights from the v0.9.6 single file rather than loading the v0.9.5 generative weights.

Convert once on the compute host with sufficient system RAM; conversion maps local tensors into components with strict key/shape checks, and copies the already-converted T5/tokenizer assets. It never calls a remote model loader:

```bash
python -m flomo convert-ltx \
  --weights weights/ltx-source/ltxv-2b-0.9.6-dev-04-25.safetensors \
  --assets weights/ltx-2b-assets \
  --revision YOUR_PINNED_V096_SNAPSHOT_COMMIT \
  --output weights/ltx-2b-0.9.6-diffusers
```

Set `model.checkpoint_revision` in both LTX templates to that same revision. The conversion records a source SHA-256, asset config hashes and a completion marker identifying the non-distilled variant and converter version. It includes the v0.9.5/v0.9.6 encoder rename mapping missing from Diffusers 0.33.1, checked against all 944 tensor schemas in the real v0.9.6 checkpoint. A failed conversion removes its temporary output rather than leaving partially initialized frozen weights. Loading rejects a missing marker or mismatched revision. Original large-weight conversion and loading still need acceptance testing on GPU compute.

```bash
python -m flomo prepare --config configs/ltx_midtrain.yaml
torchrun --standalone --nproc-per-node=8 -m flomo train --config configs/ltx_midtrain.yaml
python -m flomo export --checkpoint runs/ltx_midtrain/step_014000.pt --output runs/ltx_midtrain_bundle
torchrun --standalone --nproc-per-node=8 -m flomo train --config configs/ltx_sft.yaml
python -m flomo export --checkpoint runs/ltx_sft/step_001500.pt --output runs/ltx_sft_bundle
python -m flomo evaluate --config configs/ltx_sft.yaml --bundle runs/ltx_sft_bundle
```

LTX's VAE uses 128 channels, spatial compression 32 and temporal compression 8. Sixteen-frame targets are padded to 17, yielding three latent frames, while all sixteen action tokens remain. The adapter reuses LTX's linear voxel projections, text projection, timestep modulation, output normalization and transformer blocks. Attention/FFN LoRA and new action projections are the trainable parameters. Clean image-prefix and privileged future tokens receive timestep zero; action-free examples mask their action tokens as keys. Both image and text influence action prediction in the CPU component tests.

RoPE uses LTX's upstream full-width interleaved rotation, pixel coordinates, and temporal coordinates in seconds. Causal VAE cell positions are frames 0, 1, 9, …, matching the v0.9.6 checkpoint's singleton first frame, rather than frames 0, 8, 16, …. `model.video_fps` must match `eval.control_hz` (20 in the templates). Human footage must be resampled to this declared rate before ingestion; preprocessing rejects mismatched timestamps and does not silently resample it. As with the Wan adapter, separate image/flow/action streams have explicit temporal offsets: this is an adaptation for joint tokens, not an assertion about author code.

Sampling uses the shared Euler flow solver with 30 steps and shift 1 in the templates. Step count and shift need tuning for robot performance; these are experiment defaults, not measured accuracy/latency claims. Wan's upstream UniPC is rejected for LTX. Cached encoding uses the VAE posterior mode, with its per-channel mean/std scaling; diagnostic decoding reverses that scaling and uses decoder timestep zero without stochastic decoder noise.

The shared trainer, loss, source mixture, stage handoff, checkpoint/bundle format, action executor and ManiSkill evaluator remain unchanged. LTX and Wan bundles cannot be initialized from one another. The smaller backbone and stronger compression change the scientific comparison: evaluate fine-motion reconstruction and robot success alongside memory, throughput and matched training exposure.

## Data formats and ingestion

Canonical episodes are `.npz` arrays plus JSON metadata and an `episodes.jsonl` index. Arrays prohibit pickle/object dtypes:

* `rgb`: uint8 `[T,V,H,W,3]`, cameras in declared order.
* `times`: increasing observation timestamps in seconds.
* `actions`: optional float32 `[T-1,A]`, commands applied after each corresponding observation.
* `action_times`: optional timestamps equal to `times[:-1]`.
* Oracle geometry: front-view metric `depth`, `segmentation`, CV `intrinsics`, CV `camera_to_world`, per-body/link `body_poses`, and `body_ids`.
* Precomputed tracks: world-coordinate `[T,Q,3]` `tracks` and `track_valid`, anchored to `tracks_anchor` in metadata. Only that anchor window is used; arbitrary slicing does not create new pixel correspondences.

Metadata carries source, instruction, camera names, embodiment/controller/frequency for labeled examples and `split_group`. Set `split_group` to recording/scene/asset identity when necessary for OOD isolation. Hash-based train/validation/test splits are frozen before overlapping window generation. Do not supply fake zero actions for human clips.

Import replayed ManiSkill demonstrations:

```bash
python -m flomo import-maniskill --config configs/midtrain.yaml \
  --trajectory demos/PickCube-v1/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --source panda_demos
```

The source HDF5 must contain `traj_*/obs/sensor_data/<camera>/rgb` and matching actions, with the companion JSON. Compressed demonstrations lacking images must first be replayed. `--replay` invokes the official tool for observation materialization and tells you to import the resulting file explicitly. Controller conversion is a separate operation: use the official CPU conversion tool, then validate action execution in the target backend. State replay can reproduce images without proving command replay correctness. The importer marks replay validation as unverified.

Human footage:

```bash
python -m pip install '.[video]'
python -m flomo import-video --config configs/midtrain.yaml \
  --video human_clip.mp4 --instruction 'close the cabinet door' \
  --source human --episode-id human_000001
```

This preserves source video timing and records no action labels. For human transfer claims use actual human videos; the toy smoke's labels-removed robot footage is only a pipeline test. Generic producers can call `data.save_episode` directly.

Prepared datasets contain numeric track caches, training-only per-source flow statistics and action statistics, immutable sample tensors, manifests, rejection reports and a completion/identity header. Future RGB latents are retained so prediction-target ablations reuse the same data. Sampling weights must exactly cover selected sources, preventing silent renormalization of missing sources.

## Motion representation and deliberate deviations

Flow is cumulative displacement `p_i - p_0` in the first camera's frame. Query positions are indexed by the first image, not their future pixels. Oracle tracking follows rigid actor/link-local points; static background stays static under camera movement. Occluded oracle points remain geometrically trackable; this is not a claim of tracker visibility. Unknown segmentation IDs are invalid. Deformables need a tracker or material correspondences.

The tracker adapter supports monocular and RGB-D+poses paths, using official predictor outputs; these are different supervision regimes. Oracle collection currently requires one simulator environment because multi-subscene camera/body frame alignment has not been GPU-verified. Evaluation supports vectorized cohorts independently.

Explicit adaptations:

* Released TI2V-5B uses 48 latent channels, despite the paper's 16-channel statement.
* Sixteen valid frames are padded by repeating the last frame to 17 for Wan's first-frame-plus-four temporal codec. Sixteen actions are retained. The final action's post-state is stored in the episode, but the reference flow window includes only the initial frame through frame 15.
* RGB target ablations exclude the current image: frames 1 through 15 are the valid future, with final-frame repetition to the shared temporal grid.
* Initial ManiSkill implementation uses one stock front camera and native Panda commands. Configure additional available cameras consistently for collection/preprocessing/runtime. No bimanual/YAM fidelity is claimed.
* Motion RGB uses 1st/99th percentile rendering; actions use signed 2nd/98th normalization. Invalid tracks are filled with the zero-displacement color after quality rejection, with validity retained separately.
* Preprocessing uses deterministic resize, not the paper's stochastic crop/color-jitter recipe. This makes cached codecs exactly reproducible. Online pixel augmentation is not silently approximated by cropping latents.
* Stream/camera separation uses explicit temporal-coordinate offsets; the paper does not specify mixed-modal rotary positioning.
* No proprioception enters the model. Native robot/controller state belongs to execution or diagnostics.

These are research choices to test, not assertions about unreleased author code.

## Baselines, probes, and analysis

`configs/bc.yaml` trains a compact direct action-chunk baseline with the same frozen observation/text codecs. Set `model.architecture: bc`, `model.targets: [action]`. It uses direct chunk MSE and bypasses denoising at inference; it is not the paper's ambiguous scratch-backbone BC.

Target ablations change `model.targets` and `train.losses`:

* Motion/action: `[flow, action]`, weights 1/0.1 in mid-training.
* Video/action: `[video, action]`, video weight 1.
* Motion/video/action: all three, flow/video weights 1.

Prepared data signatures exclude target/head choices, allowing reuse across these experiments. Model bundles still carry architecture-specific signatures. Removing human data requires explicit source filters and adjusted weights; compare labeled-data exposure and compute separately.

Privileged inverse-dynamics probe:

```bash
python -m flomo probe --config configs/smoke.yaml --signal flow
```

Choose a fresh training output and optional compatible `train.init` first. `--signal video` is the RGB counterpart. `train.label_fraction` chooses a deterministic nested whole-episode subset; use 0.01/0.1/1.0 for the sample-efficiency study. The model receives clean ground-truth future tokens and predicts noisy actions only. Probe bundles are explicitly rejected by deployment. This measures decodability, not future prediction or control success.

```bash
python -m flomo attention --config configs/sft.yaml --bundle runs/sft_bundle \
  --samples 4 --output runs/attention.json
python -m flomo compare --left runs/left/episodes.jsonl --right runs/right/episodes.jsonl \
  --output runs/paired_comparison.json
```

Attention audit materializes action-query rows only, recording stream mass and per-token mass. It is descriptive, not causal evidence. Paired comparisons require matching episode IDs, seeds, tasks and instructions, and bootstrap within task. Replicate training seeds separately; evaluation bootstrap intervals do not measure training-seed variation.

## Evaluation protocol

Native ManiSkill observations come from `reset/step`, not visualization rendering. Policies see RGB/instructions only. Native actions are denormalized then clamped to native controller bounds. Paper-style absolute position/rotation-6D conversion is provided as a tested geometry utility but deliberately not enabled for arbitrary unverified controller conventions.

The evaluator uses fixed horizons and cohort resets, with geometry reconfiguration on reset. It ignores task-success termination and reports both `success_once` and `success_at_end`; there are no partial automatic resets. It executes 16 actions per plan by default and cuts off exactly at the horizon. Change `execute_steps` for a declared replanning ablation. Queue installation checks episode IDs and observation step, and reset clears all pending actions.

Use `eval.ledger` for sealed JSONL episode specifications containing episode ID, seed, task and instruction. CPU/GPU seed equality does not guarantee identical initial states across backends; keep comparisons on the same backend and asset versions. For OOD tasks, encode held-out assets/scenes in the simulator task and split metadata rather than calling new random seeds OOD.

Records include every completed episode, fixed control-step count, success, return, first success and planning timings. Recording saves arrays of actual observations. With `record: true`, oracle geometry and `eval.flow_source` set to the appropriate training renderer source, single-environment runs also compare predicted and realized rigid flow. Invalid actions cause a visible error; they are never silently dropped or replaced. Infrastructure failures need an explicit same-ledger rerun.

Task success is measured under synchronous simulation; planning wall time is reported separately. This does not implement a real-time asynchronous robot scheduler. Custom task assets, deformable simulation, multi-view motion targets, RL/recovery-data orchestration and remote GPU serving require later work beyond the core reference pipeline.

## Verification status

See `VERIFICATION.md` for tests actually run and remaining GPU checks. No full Wan/LTX weight loading, SpatialTracker CUDA run, ManiSkill rendering, robot-controller conversion, large-scale learning or real-time performance has been verified on this GPU-free laptop. Small mocked/unit checks cannot establish those outcomes.
