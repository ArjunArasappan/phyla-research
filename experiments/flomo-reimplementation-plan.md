# FloMo reimplementation plan: training infrastructure and ManiSkill evaluation

Prepared 6 October 2026. This is an implementation specification and execution plan, not a claim that the model or simulator has been implemented or benchmarked.

## 1. Recommended scope and system design

Build a simulation-first research codebase that preserves FloMo's central experiment: initialize a video-generation backbone from pretrained weights, adapt it to jointly predict cumulative 3D scene flow and robot actions, and test whether motion supervision and action-free human video improve robot control and generalization. Start with ManiSkill and one robot/controller; keep interfaces general enough to add bimanual tasks without rewriting training or evaluation.

Keep three distinct tracks in experiment metadata:

| Track | Purpose | What can be concluded |
|---|---|---|
| Method reproduction | Wan 2.2 TI2V-5B, cumulative scene flow, joint action tokens, mixed-data mid-training, robot-only SFT | Whether FloMo's design works in the selected simulation setting |
| Controlled simulator study | Same data, model, seeds and budget; change prediction targets or presence of human video | Evidence for the paper's proposed mechanism in simulation |
| Original-setting reproduction | Bimanual YAM, original camera/action setup, original data and tasks | Needed to reproduce the reported real-robot numbers; not achieved by ManiSkill alone |

Native ManiSkill single-arm actions, simulator-perfect flow, added proprioception, shorter execution chunks, and synthetic action-free robot video are all useful adaptations. Each must be named explicitly in run configuration. They cannot silently stand in for the corresponding paper setting.

The main operational boundary should be:

```text
Human videos        Robot datasets       ManiSkill demos / play
     \                    |                      /
      +------ Source adapters and episode validation ------+
                             |
                  Frozen split + window manifests
                             |
           Scene-flow extraction / simulator oracle
                             |
             Flow renderer + frozen VAE/text encoding
                             |
                 Versioned training sample shards
                             |
    Pretrained Wan --> shared trainer: mid-training --> SFT
                             |
                   Immutable policy bundle
                             |
              Policy runtime / optional GPU server
                             |
            ManiSkill adapter + chunk execution runner
                             |
       Episode records, success metrics, motion diagnostics
                             |
         Optional reviewed recovery data --> new dataset version
```

The trainer should never import ManiSkill. The simulator should never load optimizer state. The policy bundle is their interface. Most work here is getting representations and timing correct; merely connecting a DiT to `env.step()` will not reproduce the method.

## 2. What the paper actually specifies

Source: [FloMo paper](https://flomo-wam.github.io/FloMo%20Paper.pdf), especially §§3.1-3.7 and Appendices A-B. Page references below refer to that PDF.

| Component | Reported recipe | Implementation consequence |
|---|---|---|
| Initial backbone | Pretrained Wan 2.2 TI2V-5B | Load the released video checkpoint; do not start the main model from scratch |
| Transformer | 30 layers, width 3,072, 24 heads, FFN width 14,336 | Wrap the released blocks with mixed-modality token packing |
| Inputs | Current RGB observation and instruction; robot data has front + two wrist views | Separate clean image prefixes for each available view; text cross-attention |
| Main targets | Cumulative 3D scene-flow latents and a 16-step action chunk | Joint denoising in one shared trunk; no required future RGB stream |
| Motion extraction | SpatialTrackerV2, 16 frames, 4,096 initial-frame queries on a 64×64 grid | Offline window-based tracking pipeline |
| Coordinate frame | Camera frame of the first observation | Transform all trajectories into one fixed frame before subtracting |
| Flow definition | `F_i = p_i - p_0`; `F_0 = 0` | Anchor-indexed cumulative displacement, not per-frame optical flow |
| Flow rendering | Clip each channel at dataset-wide 1st/99th percentiles and map to RGB | Versioned per-source statistics; frozen Wan VAE |
| Resolution | 256×256; VAE spatial stride 16; DiT patch size (1,2,2) | 16×16 latent grid and 8×8 patch grid per latent frame |
| Action head | Linear action-to-token map, learned position embedding, linear token-to-action map | No separate large action expert or sequential inverse-dynamics decoder |
| Paper actions | 20D bimanual: per arm 3 position + 6 rotation + 1 gripper | Representation and execution adapter required for ManiSkill |
| Action normalization | 2nd/98th percentile clipping | Train-only, embodiment/controller-specific fitted statistics |
| Adaptation | LoRA rank 64, alpha 128, dropout 0 | All self/cross-attention q/k/v/o and both FFN linears, all 30 blocks |
| Trainable size | Approximately 161M parameters | Verify with a parameter inventory; freeze VAE, text encoder and base weights |
| Noise objective | `x_t=(1-t)x_0+t*epsilon`, target `epsilon-x_0`; shared t across modalities | One noise schedule and solver convention for action and flow |
| Mid-training | Mixed human/robot data; flow weight 1, action weight 0.1 | Human samples have no action supervision |
| Fine-tuning | Target robot demonstration/play data only; action weight 1 | Same architecture/trainer; change manifests, sampler and loss weight |
| Sampling | 10 UniPC steps, shifted noise schedule with shift 5, no CFG | Start at noise and integrate toward clean targets |
| Execution | Execute all 16 predicted actions before replanning | Reference evaluation uses full chunks; shorter chunks are an ablation |
| Reported latency | About 0.9 s per planning cycle on one B200 | Reference hardware result only; measure local deployment latency |

The paper's stage settings (Appendix A, Table 2):

| Setting | Mid-training | Robot-only fine-tuning/SFT |
|---|---:|---:|
| Constant learning rate | 2.8e-4 | 5.6e-5 |
| Global batch | 512 | 256 |
| Approximate optimizer steps | 14,000 | 1,500 |
| Action weight | 0.1 | 1.0 |
| Flow weight | 1.0 | 1.0 |
| Future RGB weight | 0 normally | 0 normally |

Both stages use AdamW with betas (0.95, 0.999), weight decay 1e-6, gradient clipping 1.0 and bf16. The paper states a 1,000-timestep training parameterization, epoch-28 mid-training initialization for SFT, and usually epoch-3 SFT evaluation. Treat the epoch numbers as specific to its data/sampling implementation, not portable milestones.

Actual used training subsets in Appendix B differ from the larger corpora mentioned in the main text: about 81 h EgoVerse, 10 h MolmoAct2-BimanualYAM, 3.7 h in-house robot demonstrations, 0.7 h in-house human data, and 0.4 h robot play. Original in-house data availability is not established. Human video is 70% of the reported mid-training mixture by sampling weight, not necessarily by storage volume.

| Source | Mid-training sampling weight | SFT weight |
|---|---:|---:|
| EgoVerse | 0.65 | 0 |
| MolmoAct2-BimanualYAM | 0.125 | 0 |
| Target robot demonstrations | 0.125 | 0.75 |
| Task-related action-free human video | 0.05 | 0 |
| Target robot play | 0.05 | 0.25 |

For a ManiSkill study, replace the target robot demonstration/play sources with simulator datasets. Preserve and publish the mapping. If real human footage is unavailable, synthetic robot clips with labels removed test action-free learning, but do not establish human-to-robot transfer.

## 3. Resolve these ambiguities before scaling

Maintain an `assumptions.md` and numbered architecture decisions. The project page currently labels code as coming soon; the paper does not supply enough detail for a byte-for-byte reproduction.

### 3.1 VAE channels: paper versus released weights

The paper's §3.3 says 16 latent channels, but the released [TI2V-5B checkpoint configuration](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/config.json) specifies `in_dim=48`, `out_dim=48`. The official [Wan 2.2 VAE wrapper](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py) defaults to `z_dim=48`. The published dimensions/strides otherwise align with TI2V-5B.

Recommendation: use the compatible released 48-channel VAE/backbone pair unless author clarification establishes a different checkpoint. Never insert an arbitrary 16-to-48 projection and call it faithful. Infer dimensions from pinned checkpoint metadata, record the discrepancy, verify encoding and patchification on real tensors, and prohibit incompatible cache reuse.

### 3.2 Sixteen frames and temporal compression

The inspected official VAE encoder computes `1 + floor((T-1)/4)` latent frames and processes the first frame separately followed by blocks of four. At `T=16`, this path processes frames through index 12, leaving three trailing frames outside those blocks. The implementation must test this explicitly with frame-coded inputs; do not guess latent count from a rounded division.

Recommendation for a documented adaptation: retain 16 valid scene-flow frames and 16 actions; repeat the last flow frame to produce 17 VAE input frames, yielding five latent frames. Decode 17 frames for diagnostics, discard the repeated frame, and report the policy. The fifth latent mixes real and padded input and is not cleanly removable with a simple frame mask. Alternative author-approved encoder handling belongs in a separate configuration. This decision affects training/inference token lengths and all cached flow latents.

### 3.3 Action/observation alignment

The paper uses 16 observations and 16 actions even though 16 observations expose only 15 transitions. Canonical robot episodes should have `N+1` observations for `N` actions, with `a_t` applied between `o_t` and `o_(t+1)`.

Proposed reference window: condition at `o_t`; action targets `a_t ... a_(t+15)`; flow frames from `o_t ... o_(t+15)` including zero at the start. Store `o_(t+16)` for final-transition checks and rollout diagnostics. This preserves the paper's shapes while documenting the extra final action. A fully transition-aligned future-flow variant can instead predict displacements through `o_(t+16)`, but changes the target and must be separate.

### 3.4 Representation details not established by the paper

Resolve action position absolute/delta semantics, action reference frame, rotation-6D ordering, gripper meaning, robot control rate, window stride, tracker scale convention, track validity handling, 64→256 upsampling, stream/camera positional encoding, clean-prefix time embeddings, loss reduction, missing-head participation, optimizer reset across stages and UniPC implementation/version. Figure captions describe instantaneous motion; the method explicitly defines cumulative displacement. Follow the method definition and test it.

Using one frozen VAE gives RGB and flow a common encoding space; it does not prove their latent distributions are identical. Measure channel statistics, saturation and reconstruction quality rather than assuming compatibility.

## 4. Repository and package organization

Use one repository with modular Python packages, three container targets and immutable artifacts. Avoid separate training and evaluation repositories that copy normalization code.

```text
flomo/
  pyproject.toml
  locks/                         # pinned environments and upstream commits
  configs/
    model/                       # Wan, LoRA, token layout, target modalities
    data/                        # sources, frozen split manifests, mixtures
    stage/                       # midtrain, sft, optional extensions
    embodiment/                  # robot, action space, codec, camera layout
    sim/                         # task families, assets, physics, randomization
    eval/                        # validation and sealed benchmark specifications
    experiment/                  # full recipes and controlled ablations
  src/flomo/
    contracts/                   # schemas and compatibility validation
    data/
      sources/                   # human, Molmo, ManiSkill, generic episodes
      manifests.py
      windowing.py
      mixture.py
      collate.py
      shards.py
    geometry/
      frames.py                  # named transforms and units
      tracks.py
      canonical_flow.py
      render_flow.py
      stats.py
    preprocess/
      ingest.py
      extract_tracks.py
      encode_latents.py
      encode_text.py
      validate.py
      jobs.py
    models/
      wan_adapter.py
      token_layout.py
      positions.py
      action_heads.py
      lora.py
      flomo.py
    training/
      trainer.py
      objective.py
      stage.py
      distributed.py
      checkpoint.py
      validation.py
    inference/
      policy.py
      sampler.py
      observation_codec.py
      action_codec.py
      bundle.py
      server.py                  # optional; local inference comes first
    sim/
      maniskill_adapter.py
      task_registry.py
      cameras.py
      controllers.py
      collect.py
      replay.py
      oracle_tracks.py
      tasks/                     # custom tasks only when needed
    evaluation/
      runner.py
      chunk_executor.py
      metrics.py
      motion_diagnostics.py
      records.py
      aggregate.py
    baselines/                   # same policy/runtime interface
    cli/                         # thin commands calling libraries
  tests/
    contracts/
    geometry/
    model/
    sim_integration/
    end_to_end/
  docker/
    preprocess.Dockerfile
    train.Dockerfile
    sim.Dockerfile
  scripts/                       # scheduler submissions; no duplicated logic
  docs/
    paper_map.md
    assumptions.md
    decisions/
    runbooks/
  third_party/                   # pinned adapters/patches, not copied giant weights
```

Use PyTorch and typed dataclasses/Pydantic-style validation for contracts; YAML composition with strict unknown-key rejection; Parquet manifests; source HDF5 retained for ManiSkill, and tar/array shards for prepared examples; safetensors for deployable weights where practical. Choose one run tracker and one cluster scheduler. Slurm plus batch preprocessing arrays is sufficient if the lab already has Slurm. Kubernetes/Ray are optional organizational choices, not requirements of the method.

Container separation protects us from tracker, renderer and training CUDA dependency conflicts. Pin Python, PyTorch, CUDA, SAPIEN, ManiSkill, transformer libraries, Wan commit, tracker commit, checkpoint revisions and assets. The simulator container requires a tested NVIDIA/Vulkan setup; a CUDA-only smoke test is insufficient.

## 5. Shared contracts: what every stage reads and writes

### 5.1 EmbodimentSpec and DatasetSpec

`EmbodimentSpec` is a hashed configuration: robot UID, arm ordering, controller class/mode/config, action dimensions/units/bounds, pose reference frame, rotation convention, gripper mapping, physics/control frequencies, camera IDs/intrinsics/extrinsics, and whether proprioception enters the model or only execution.

`DatasetSpec` adds source identity/revision, license/access provenance, embodiment, language annotation policy, FPS, episode schema version, split grouping rules, tracker/flow variant, and normalization IDs. An action-free sample has no action label; it is not represented as a genuine zero-action example.

### 5.2 CanonicalEpisode

Persist raw facts before computing learning targets:

```python
CanonicalEpisode = {
    "schema_version": str,
    "episode_id": str,
    "source_id": str,
    "source_episode_id": str,
    "task_id": str,
    "instruction": str,
    "embodiment_spec_id": str | None,
    "rgb": dict[str, Array],        # camera -> [N+1,H,W,3], uint8
    "observation_times": Array,     # [N+1], seconds
    "actions": Array | None,       # [N,A], source/controller representation
    "action_times": Array | None,  # [N], seconds
    "proprio": dict | None,
    "calibration": dict,
    "sim_state_ref": str | None,   # privileged; offline/evaluator only
    "outcomes": dict,              # success/failure, expert/collector identity
    "provenance": dict,
}
```

Human video does not necessarily have the `N+1`/`N` action structure; its adapter provides timestamped frames and `actions=None`. The episode validator should distinguish these source types. Never fabricate robot embodiment metadata for human clips.

### 5.3 WindowRecord

The window manifest contains episode reference, start time/index, exact sampled frame indices/timestamps, 16 action indices if present, camera coverage, split, instruction revision, temporal spacing, flow-cache key and augmentation policy. Window IDs must change if timing or anchor changes. Cumulative tracks depend on the window's first frame; arbitrary slicing of a single episode-level flow cache is not equivalent to re-anchored tracking.

### 5.4 PreparedTrainingSample

```python
PreparedTrainingSample = {
    "window_id": str,
    "source_id": str,
    "observation_latents": Tensor,  # [V,C,1,16,16]
    "view_valid": Tensor,           # [V], bool
    "text_hidden": Tensor,          # [L,4096], L<=512
    "text_valid": Tensor,           # [L]
    "flow_latents": Tensor,         # [C,Tz,16,16], C from checkpoint
    "action_targets": Tensor | None,# [16,A], normalized
    "action_valid": Tensor | None,  # [16,A] if mixed dimensionality is supported
    "video_latents": Tensor | None, # only for RGB-target experiments
    "target_modalities": set[str],
    "normalization_ids": dict,
    "layout_id": str,
    "provenance": dict,
}
```

Storage may keep raw RGB/flow rather than observation latents when using online augmentation. The collator produces the same model-facing batch. Caches and online encoders must match exactly for the same inputs and frozen encoders.

### 5.5 PolicyBundle

A deployment bundle includes base-weight identifier/hash, LoRA weights, action heads/position embeddings, architecture/token-layout version, VAE/text references, normalization files, image preprocessing, camera layout, action codec, task/embodiment compatibility, temporal spacing, sampler settings and provenance. Evaluation loads this bundle plus an evaluation specification. It should refuse a controller mismatch even when action dimensions happen to agree.

An optimizer checkpoint is larger: bundle material plus optimizer, gradient scaler if used, scheduler/stage state, global step, RNG states, sampler cursor, split hashes and resolved configuration. Write atomically, checksum, and publish a completion marker before downstream jobs discover it.

### 5.6 Runtime protocol

```python
class Policy:
    def reset(self, episode_ids, env_indices=None): ...
    def predict(self, observations, instructions, request_meta) -> "ActionChunk": ...

class ActionChunk:
    actions: Tensor                # [B,16,A] in bundle's model action space
    valid_steps: Tensor            # [B,16]
    flow_latents: Tensor | None     # optional diagnostic output
    episode_ids: list[str]
    observation_steps: Tensor       # protects against stale responses
    bundle_id: str

class ActionCodec:
    def encode_labels(self, source_actions, context) -> Tensor: ...
    def normalize(self, model_actions) -> Tensor: ...
    def denormalize(self, normalized_actions) -> Tensor: ...
    def to_controller(self, model_action, robot_context) -> Tensor: ...

class SimAdapter:
    def reset(self, episode_specs) -> "PolicyObservation": ...
    def step(self, controller_actions) -> "StepResult": ...
    def policy_observation(self) -> "PolicyObservation": ...
    def evaluator_state(self) -> dict: ...
```

Contract tests must verify batch dimension, device, dtype, timing and IDs. Policy-visible RGB and instruction are distinct from privileged simulator state used for metrics/flow labels. Robot proprioception may be used by a pose-to-controller codec; adding it as DiT conditioning is an explicit architecture extension, because FloMo does not specify a proprioceptive input stream.

## 6. Data infrastructure and preprocessing

### 6.1 Storage and lineage

Use object storage or a reliable shared filesystem as the source of record, with local NVMe staging. Suggested logical paths:

```text
raw/<source>/<revision>/...
episodes/<schema>/<source>/<revision>/...
manifests/<dataset_version>/{episodes,windows,splits}.parquet
tracks/<tracker_hash>/<window_id>.*
flow/<flow_definition_hash>/<window_id>.*
stats/<normalization_id>.json
latents/<encoder_hash>/<preprocess_hash>/<window_id>.*
shards/<prepared_dataset_version>/...
checkpoints/<run_id>/<step>/...
bundles/<bundle_id>/...
eval/<suite_version>/<bundle_id>/<run_id>/...
```

Do not put terabytes of data or weights in Git. Version manifests and configs in Git or a content-addressed artifact store. Derived artifact keys include source/window hashes, tracker, frame convention, scale, rendering statistics, augmentation, VAE revision, latent scaling and temporal padding. A new normalization table invalidates flow-latent caches.

Jobs operate on immutable window IDs: claim work, load source, compute, validate, atomically write outputs, emit status. Retry failed windows independently; distinguish missing assets, decode errors, tracker errors and invalid geometry. Quarantine rejected samples with reasons and source-wise rejection rates. An index/manifest with pending/running/done/failed states is enough at first; do not create a large platform before preprocessing works.

### 6.2 Split before windowing

Split at recording/episode level, preferably environment/scene/object-instance group level for the relevant OOD question. Every overlapping window and augmentation from an episode stays in that split. Hold out validation/test initial states and asset families explicitly. Fit statistics on training data only, including during mid-training; do not refit them on validation or evaluation rollouts.

Audit human and robot sources for evaluation-task overlap according to experiment intent. Human demonstrations of a robot-held-out primitive are allowed in the transfer experiment; they must be recorded as such. They are not allowed in a strict all-source unseen-task experiment. Publish both definitions rather than conflating them.

### 6.3 ManiSkill demonstration ingestion

Keep each source `.h5` and paired `.json`. Read environment ID, robot/controller, initial state, seed, actions, available states and timing. ManiSkill's compressed demonstrations can omit images, so replay to materialize the desired sensor views. The [official replay documentation](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) distinguishes controller conversion, state replay and action replay; conversion support is limited and currently not supported in GPU-parallel environments.

Use a two-step path where conversion is needed: CPU controller conversion with supported robots, then GPU replay/rendering in the resulting controller. Verify with the pinned implementation rather than assuming all tasks/controllers convert successfully. State replay is useful for reproducing recorded images, but does not demonstrate that converted actions can execute those trajectories. Independently action-replay a representative validation set in the evaluation backend, retaining success and trajectory-error reports.

For newly collected data, record in the final controller/backend from the start. Use planner experts, teleoperation or a strong teacher policy, and record expert actions separately from achieved TCP poses. Add recovery demonstrations around failures. Play data should contain useful contact and movement diversity rather than mostly static frames. Apply motion filtering only on training data, publish thresholds/rejection rates, and keep validation representative.

### 6.4 Time and language handling

Choose a policy control frequency per embodiment and record it. For an illustrative 20 Hz controller, executing 16 actions spans 0.8 s; the 16-frame flow target including the initial frame reaches 0.75 s. The paper does not establish a transferable control frequency. Do not equate Wan's generation FPS with robot action frequency.

Avoid casually decimating a 30 FPS video and pairing it with every fifth robot action. For native delta commands, skipping intermediate actions changes the executed trajectory. Either collect/replay at the intended frequency or use a controller-aware conversion and validate it. Human clips can have a separate temporal spacing, but track and stratify clip durations; changing motion speed changes cumulative displacement statistics.

Use task descriptions from metadata for single-task demonstrations; custom multi-step tasks need ordered subtask labels and language. Version deterministic templates and any language augmentation. OOD instructions must be annotated before outcome inspection. Cache frozen text embeddings by tokenizer/text-model revision and exact instruction bytes.

### 6.5 Scene-flow extraction

Implement a `TrackProvider` with at least two backends:

1. **Estimated flow:** SpatialTrackerV2 on front-view RGB, with an explicit input mode (monocular or RGB-D+poses). Use the same mode for controlled comparisons. The [official tracker](https://github.com/henry123-boy/SpaTrackerV2) supports both; supplying simulator depth/poses changes supervision quality and must be named.
2. **Simulator oracle:** build anchor-point correspondences from initial depth, segmentation and rigid actor/link transforms. Use this for geometry verification, flow quality diagnostics and an oracle-supervision ablation.

For estimated tracks, store points, visibility/confidence, camera poses, intrinsics, units and whether coordinates are camera-local or world-frame. Never apply an ego-motion transform twice. For local points and camera-to-world poses:

```text
p_i_in_C0 = inverse(T_world_from_C0) @ T_world_from_Ci @ p_i_in_Ci
F_i      = p_i_in_C0 - p_0_in_C0
```

For already world-coordinate tracks, apply only `inverse(T_world_from_C0)`. Translation cancels in displacement, but rotation into the anchor frame remains essential. Monocular geometry may have uncertain scale; examine tracker outputs and static-scene residuals before mixing them with metric simulator labels. Source-specific percentile rendering can hide scale differences without establishing consistent physical scale. Record scale policy and compare native tracker normalization against calibrated variants.

All grid locations refer to pixels in the initial image. Render each point's future displacement back into its original grid cell; do not relocate it to its future projected pixel. The initial 64×64 grid is a correspondence field indexed by the first frame, not a fresh Eulerian flow raster at every timestep.

### 6.6 Simulator oracle construction

For each valid initial query: back-project depth with calibrated intrinsics; convert camera convention and millimeters to the canonical convention/meters; use segmentation to identify actor or articulation link; transform the world point into that body's local coordinates. At each later timestep, transform that local point through its current body pose and into `C0`. Subtract the initial coordinate.

Use link transforms for robot/articulated objects, not a single articulation root transform. Handle background as static world points. Track occlusion separately from physical point existence. A point can remain geometrically trackable after becoming invisible; a tracker-derived label might lose it. Make visibility policy explicit. Deformable objects, strings and changing topology require material correspondences or a tracker; the rigid-body oracle does not solve them automatically.

Geometry acceptance tests: static objects yield zero flow under camera movement; an object translation matches the transformed displacement; a rotation produces different displacements across points; articulated door and robot links follow the correct local transforms; moving wrist cameras do not create background motion. ManiSkill documents mixed GL/CV camera representations and depth units in its [observation specification](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html). Explicit conversions are mandatory.

### 6.7 Flow rendering, normalization and augmentation

Fit train-only `low[c]=p1`, `high[c]=p99` per source/defined dataset. Render `clip((F-low)/(high-low),0,1)` with an epsilon/constant-channel policy. Map x/y/z to R/G/B. Save counts, units, quantile method, clipping frequency and static zero color. Zero flow need not be black or exactly mid-gray. Preserve the numeric displacement array for debugging; MP4 is a visualization, not the canonical geometry store.

Document 64→256 interpolation. Bilinear interpolation is a reasonable initial choice, but can mix motions across boundaries; compare nearest-neighbor on synthetic fixtures. Invalid queries need a documented fill/drop policy. Replacing invalid displacement with zero conflates missing motion with static motion. Keep validity/confidence out of band; reject windows with excessive invalid coverage and record remaining fill behavior. Confidence-aware latent weighting is an optional extension: VAE receptive fields mean a point mask cannot simply be resized into an exact latent loss mask.

The paper applies random resized crop (scale 0.85-1, ratio 0.98-1.02) and RGB color jitter (brightness/contrast/saturation 0.25, hue 0.1) to all sources except EgoVerse. Apply spatial transforms consistently to RGB, anchor-grid flow and calibration. Color-jitter RGB observations/future RGB only; changing flow RGB colors changes the vector labels. This interpretation should be logged because the paper does not spell out augmentation ordering. Crop changes pixel indexing and intrinsics but not already canonical metric displacement values. Re-anchoring the canonical frame would require a separate geometric transform.

A fixed latent cache cannot reproduce arbitrary online pixel crops/color jitter. Initially use either online frozen-VAE encoding after augmentation or a finite, seeded bank of augmented variants. Record which regime is used. Cache only deterministic assets when exact augmentation behavior matters. Encoding target flow before augmentation and cropping latents afterward is not equivalent to encoding cropped flow.

### 6.8 Preprocessing quality gates

Require valid window lengths/timestamps, action alignment, camera naming/calibration, `F0=0`, plausible depth/flow scales, static-background residual reports, visibility rates, clipping/saturation statistics, action distributions/gripper balance and RGB-flow visual overlays. Decode representative flow latents and invert percentile rendering to measure displacement reconstruction error before training. Inspect fast/fine motion and occlusion examples from every source. The frozen VAE may lose small task-relevant motion; detect that before spending a full training budget.

## 7. Model implementation

### 7.1 Preserve the pretrained trunk; replace its sequence plumbing

Use the official Wan code/weights as the compatibility reference. Its standard forward path assumes a patchified video grid and unpatchifies the entire output. FloMo needs a mixed sequence, so wrapping only the standard `generate()` method is insufficient. Implement:

1. Frozen encoders for text and initial RGB observations.
2. Original Wan Conv3d patch embedding independently over each observation and target-flow latent grid; reuse it for future-video ablations.
3. Learned linear action projection and 16 learned action-position embeddings.
4. A `TokenLayout` giving each stream's span, modality, camera, position, target/conditioning status and validity.
5. Shared Wan blocks with mixed-stream position/time handling and full attention over valid tokens.
6. Original visual velocity head applied only to visual target spans, followed by per-stream unpatchification.
7. Linear action velocity head applied only to action spans.

The action input is the noisy normalized action vector, and the output is velocity in that action space. Do not noise its projected token embedding and then compare that output against raw-action velocities. That defines a different stochastic process.

At 256×256, each initial image contributes 64 patch tokens. With the proposed 17-frame VAE padding, scene flow contributes 5×64=320 tokens, and actions contribute 16. A three-view robot sample therefore has approximately 528 self-attention tokens; one-view human flow-only samples have approximately 384. An additional five-latent-frame video stream adds approximately 320 tokens. Text cross-attention has up to 512 tokens. These are planning estimates under the stated temporal handling; derive exact counts from encoder outputs. They show why stream packing and RGB-target compute matching matter.

### 7.2 Positions, cameras and timestep embeddings

Retain Wan's pretrained spatial/temporal rotary positional structure for visual patches. Do not pass one invented rectangular `grid_sizes` for the mixed sequence to unmodified Wan rotary code. The adapter should explicitly build positional coordinates/frequencies for each stream and apply them to Q/K.

The paper does not specify how equal spatial locations in multiple cameras/modalities are distinguished. Initial implementation choice: deterministic stream/camera offsets in temporal coordinates, preserving each stream's spatial grid; a documented action-token rotary convention plus learned action positions. Validate alternative offsets/identity treatment on tiny experiments. Adding learned modality/camera embeddings is another possible extension, with new parameters and a named architecture version. Do not make that undocumented choice look like a published detail.

Use time `t` for all noisy targets and time zero for clean image-prefix tokens. Wan's inspected forward supports per-token timesteps, which is useful for this. Clean prefixes stay unchanged by the solver, though their hidden states evolve inside each DiT forward pass. They remain visible to targets. Compute visual-head conditioning with matching per-token timestep embeddings. Test that noise and updates never touch prefix tensors.

Full attention means all valid observation/target tokens attend to one another; padding and absent modalities remain masked. For human samples, omitting action tokens is a sensible proposed interpretation. Setting action loss to zero while keeping arbitrary action tokens can still change flow prediction through attention. Provide a named alternative if later author code does this differently. Bucket by view/target layout and normalize global sample contributions; do not let human examples inherit fake wrist images or action labels.

### 7.3 LoRA and gradient checks

Inject adapters in both attention types' q/k/v/output projections and both FFN linear layers. Train the action projections/positions from scratch. Keep original patchifier/visual head frozen unless the author implementation establishes otherwise. Print a parameter table by module and compare the total against the reported approximate 161M. Confirm nonzero gradients for action heads and LoRA, and no gradients for base weights/VAE/text encoder.

Freezing the 5B trunk weights does not eliminate backbone activation memory or backward work: gradients to LoRA and inputs still propagate through the blocks. Use activation checkpointing, efficient attention and microbatch accumulation. Start with one GPU and DDP replication if memory fits; add sharding only if profiling warrants it. LoRA optimizer memory is much smaller than full-finetuning optimizer memory, but the frozen base still has to reside or be sharded/offloaded.

### 7.4 Joint flow-matching objective

For each example draw one shared `t`; draw independent Gaussian noise tensors for each enabled target. Use:

```python
noisy[m] = (1 - t) * clean[m] + t * noise[m]
velocity_target[m] = noise[m] - clean[m]
prediction = model(clean_prefix, text, noisy, t, token_layout)
loss = weighted_masked_modality_mse(prediction, velocity_target)
```

Reduce squared error to a per-example mean over valid elements within each modality, then apply modality weights and average over the actual global batch. This is a proposed resolution of the paper's unspecified reduction. Raw sums would make flow dominate by dimensionality and would change the meaning of action weight 0.1. Log mean losses and weighted contributions separately.

For action-free examples, action loss is zero. Keep normalization over the full mixed batch for the reference interpretation and log the labeled fraction. With 30% labeled examples and action weight 0.1, the effective action contribution differs from averaging action error only over labeled examples. A labeled-only normalization may be worthwhile, but is a separate recipe. Distributed bucketing must reproduce the declared objective, not independently average small source-specific microbatches with equal weight.

If multi-embodiment experiments eventually use padded action dimensions, mask invalid dimensions in noise, outputs and loss, and identify embodiment semantics. For the first implementation, support one action spec per training run; this avoids unnecessary complexity and accidental mixing of incompatible action meanings.

### 7.5 Sampler

Create a standalone sampler sharing the training schedule definition. Start at `t=1` Gaussian noise and integrate to `t=0`; the velocity target is noise minus clean, so an incorrect integration direction gives nonsense. Use the Wan-compatible UniPC flow implementation, 10 steps and shift 5, conditional path only. Store exact solver/schedule version. The common shift parameterization `s*t/(1+(s-1)*t)` is a candidate to verify against the pinned implementation, not an excuse to substitute an arbitrary diffusion UniPC scheduler.

Both actions and flow are sampled jointly at every trunk call. RGB-flow decoding is unnecessary for control; return actions directly and decode flow only for sampled diagnostics. There is no online tracker dependency. Do not introduce classifier-free guidance, different per-modality schedules, action-only sampling or a motion-then-action pipeline in the reference recipe.

Tests: constant/analytic velocity fields integrate with the expected sign; seeded sampling is repeatable within the supported environment; prefix tensors remain fixed; action and flow solver grids agree; 10 steps actually means 10 documented model evaluations/solver updates as appropriate; output shapes match the bundle. An Euler sampler is a useful debugging reference, not the headline UniPC result.

## 8. How pretraining, mid-training, SFT and evaluation interface

### 8.1 Pretraining is checkpoint acquisition in the default project

FloMo inherits large-scale video pretraining from Wan. Reproducing internet-scale Wan training is outside the necessary scope. The pretraining boundary emits `BackboneArtifact`: base checkpoint hashes, model config, frozen VAE/text references, latent normalization and upstream revision. A compatibility command encodes a short RGB/flow fixture, patchifies it, loads all blocks and checks dimensions.

If we later train a backbone ourselves, that pipeline must emit the same artifact. It would require a separate large video data/training program and should not be hidden inside FloMo mid-training estimates. Optional human-only flow adaptation is also an added stage; the paper's published mid-training is mixed human and robot supervision from the outset.

### 8.2 Mid-training consumes prepared samples, not simulator environments

Inputs: `BackboneArtifact`, prepared dataset manifests, embodiment/action spec, mixture recipe, loss/sampler configuration. Initialize LoRA and action heads; sample source according to declared weights, then task-balanced/window-balanced according to a deterministic documented policy. Distributed sampling should log realized source/task frequencies, unique windows and oversampling counts.

Main outputs: resumable checkpoints, exportable bundles, validation curves, loss-by-source/time/modality, throughput and data lineage. Run small fixed held-out loss suites frequently. Dispatch simulation evaluation asynchronously every selected checkpoint; do not stall all training ranks while one rank runs a renderer.

For a method-faithful data route, retain EgoVerse and compatible action-free task-related videos, and use ManiSkill demos/play for the action-labeled sources. Mixing Molmo bimanual 20D actions with native Panda actions requires a real multi-embodiment architecture/normalization design; do not simply concatenate them into one action dataset. A simpler first study can use Molmo only as explicitly action-free motion footage, or omit it and report that change. For later bimanual training, add embodiment-specific action projections and codecs or a carefully designed common space as an extension.

The original weights can serve as a reference allocation: 0.65 broad human, 0.125 broad robot, 0.125 target robot, 0.05 related human, 0.05 target play. If no broad robot dataset is compatible, establish a replacement simulator source and predeclare adjusted weights. Do not claim original weights while silently renormalizing absent sources.

### 8.3 SFT changes data and objective, not interfaces

Inputs: selected mid-training checkpoint, target robot demonstration/play manifests, unchanged model/action contract, action weight raised to 1, lower constant LR. All trainable LoRA/head modules continue training. Default proposed behavior is a fresh AdamW optimizer at the stage boundary; retain model weights but record optimizer reset because the paper does not specify it. Continue exact optimizer/RNG state only for an interrupted run of the same stage.

Maintain normalization from the chosen dataset/version across mid-training and SFT unless a migration is deliberate and tested. Re-fitting action statistics changes what the existing action head means; doing it casually can break transfer. If a new embodiment requires a new head or statistics, name that operation and validate separately.

SFT outputs the same `PolicyBundle`. The simulator/evaluator should not care whether it came from mid-training, SFT, a baseline or an optional later stage. Select checkpoints using validation only. Use simulation success alongside action/flow validation losses; low action MSE does not guarantee rollout success.

### 8.4 Evaluation is a separate artifact consumer

Inputs: bundle ID and sealed suite spec. Verify controller, robot, camera, image/timing and action normalization compatibility. Warm up model and rendering; run the predefined episode ledger; emit per-episode records and aggregates. Training configuration is metadata, not executable evaluator dependencies.

Suggested command vocabulary, to be implemented as thin library wrappers:

```bash
flomo backbone verify --spec configs/model/wan_ti2v5b.yaml
flomo data ingest --source configs/data/maniskill_demos.yaml
flomo data split --spec configs/data/splits_v1.yaml
flomo preprocess track --manifest manifests/windows_train.parquet
flomo preprocess fit-stats --manifest manifests/windows_train.parquet
flomo preprocess encode --manifest manifests/windows_train.parquet
flomo data validate --dataset prepared-v1
torchrun --nproc-per-node=8 -m flomo.cli.train --config configs/experiment/midtrain.yaml
torchrun --nproc-per-node=8 -m flomo.cli.train --config configs/experiment/sft.yaml
flomo bundle export --checkpoint checkpoints/run/step
flomo eval run --bundle bundles/model-id --suite configs/eval/validation.yaml
flomo eval aggregate --run eval/run-id
```

These are proposed interfaces, not existing installed commands. Fit train-only statistics before encoding any split with them; hold-out preprocessing uses the frozen training statistics.

### 8.5 Optional post-training feedback loop

The paper does not require RL or reward-based policy optimization. First implement offline mid-training/SFT and evaluation. A later recovery-data loop can consume failures from a designated development suite, obtain corrected teacher/teleop trajectories, review them, version them, and run another SFT. Test episodes and their privileged oracle actions never enter this loop. If adding RL/DAgger, create a new stage with explicit collector, reward/teacher contract and action log-probability/objective assumptions; do not relabel it as the paper's SFT.

## 9. ManiSkill environment and action organization

### 9.1 First robot/controller choice

Start with Panda and a supported native end-effector delta-pose controller, after confirming task-specific support and replayability. A typical single-arm space has six pose coordinates plus one gripper coordinate; derive the actual action shape/config from the environment. Start by predicting native controller-space actions. This is the least ambiguous way to validate the motion/action joint model in simulation, but differs from the paper's 20D bimanual representation.

If replay conversion is unreliable for a task, use the original supported controller or recollect demonstrations. A joint-delta track is acceptable as a separate embodiment recipe. Select one stable controller first instead of maintaining uncontrolled per-task mixtures.

The paper-style alternative predicts 10D per arm (position, rotation-6D, gripper). Implement a codec that denormalizes, reconstructs a valid rotation with a numerically robust 6D-to-SO(3) projection, transforms into the controller's frame, and generates an absolute or delta target according to the spec. A 6D rotation representation is not six rotational degrees of freedom and cannot be sent directly into a six-coordinate EE pose command.

Absolute model targets require current robot/controller context to derive native delta commands. Distinguish delta-from-current-pose from delta-from-previous-target. Clamp using documented native bounds after conversion. For a bimanual model, preserve arm ordering throughout data/model/controller. Never tile one arm's commands into a second arm.

The [ManiSkill controller documentation](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html) contains differing high-level descriptions of rotation semantics across controller sections. Inspect the pinned controller code/config and verify one-axis rotations before implementation; do not infer Euler versus axis-angle semantics from the action vector length.

Codec acceptance tests: encode/decode known poses; noncommuting rotation compositions; unit translations in each frame; zero/hold action; both gripper extremes; native action bounds; per-arm ordering; commanded versus achieved pose under closed-loop PD dynamics. Learned command labels should represent the teacher command, not a retrospective achieved pose mislabeled as its cause.

### 9.2 Cameras and observation adaptation

Single-arm prototype: one front/workspace camera plus one wrist camera if available/configured. A three-camera bimanual setup requires front + left wrist + right wrist. Do not pretend every stock task exposes the paper's camera layout. Add consistent sensor configurations through a wrapper/task subclass and use the same settings for demonstration replay and evaluation.

The adapter maps native `sensor_data` to named policy cameras, scales uint8 RGB correctly, applies the bundle's resize/crop, and removes everything outside the declared policy input. Read observations returned by `reset/step`, not visualization frames from `env.render()`. Save calibration and camera motion for offline flow creation. Use front-view flow only in the reference architecture; multi-view flow is a useful occlusion extension.

Collection can request depth/segmentation for oracle labels while runtime policy sees RGB. Ground-truth object poses, segmentation, task success and future frames are evaluator/label channels, not policy conditioning. Add assertions that diagnostic fields never enter policy requests.

### 9.3 Task registry and benchmark progression

Every task entry needs environment/robot/controller IDs, assets and version, camera specification, maximum control steps, expert/collection method, task language, success criteria, optional ordered subtask predicates, split/randomization definitions and supported backends. Environment IDs should be validated against the pinned registry.

| Tier | Proposed tasks | Why |
|---|---|---|
| Smoke | PushCube-v1, PickCube-v1 | Exercise contacts, gripper, native actions, cameras and chunk resets |
| Initial manipulation suite | Stock stacking, side insertion and diverse object picking tasks after registry verification | Precision, multi-stage progression and object generalization |
| Mechanism study | Appearance/camera perturbations of the same physical tasks | Isolate motion-target benefit without changing dynamics |
| Transfer suite | Held-out objects, motion primitive, and task family with explicit source coverage | Test the human-video transfer hypothesis |
| Paper-like extension | Hinged door/container opening and retrieval, object-to-bowl, cabinet closing | Approximate original task structure with custom assets/predicates |

Do not make string pulling an early requirement: deformable simulation and point correspondence add a major independent project. Likewise, a stock Panda pick-and-place benchmark does not reproduce YAM bimanual precision tasks. Get the end-to-end system working on stock rigid tasks before custom task development.

OOD suite definitions:

* Appearance: held-out textures/materials/lighting/distractors, with physical geometry and goal semantics controlled.
* Object: held-out mesh/category/size combinations, not just an evaluation seed on a seen mesh.
* Motion primitive: exclude a primitive from action-labeled robot training while allowing documented human examples in the transfer setting.
* Task: hold out a task family from robot labels; distinguish human-covered transfer from all-source unseen tasks.
* Camera: fixed predeclared intrinsics/pose perturbations; distinguish viewpoint shift from estimator ego-motion invariance.
* Embodiment: postpone until a single-embodiment result is stable; human-to-robot is already a meaningful transfer question.

A primitive-transfer study needs enough related labeled actions for the robot head to ground the primitive in its embodiment. Failure may expose action grounding rather than bad motion prediction. Diagnose both.

## 10. Simulator evaluation and policy serving

### 10.1 Start local and synchronous

First run one simulator environment and one policy in the same process/GPU if memory permits. At a chunk boundary, snapshot policy-visible RGB/instruction; jointly sample 16 actions/flow; execute one decoded controller action per `env.step`; replan after 16 steps. In offline evaluation, physics advances on simulator steps; model inference wall time need not advance simulated time.

Separate success performance in this synchronous protocol from real-time performance. A model can score well while being too slow for a robot. Report both and keep the reference protocol fixed across baselines.

### 10.2 Batch by compatible task/embodiment/layout

Scale to homogeneous GPU-vectorized cohorts. ManiSkill serves batched Torch tensors and supports GPU physics/rendering, as described in its [quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html). The policy's maximum batch, not simulator FPS alone, determines efficient cohort size. Start with small batches and profile 1/4/8/16/32 environments. Do not assume state-only simulator throughput applies to three RGB cameras plus a 5B policy.

Use one simulator process per assigned GPU with one visible GPU as the initial deployment pattern; the official quickstart documents rendering issues with multiple visible devices in one process. If training occupies an 8-GPU node, give evaluator processes explicit separate GPU resources or place them on another node. Exposing simulator GPUs to training ranks can create memory contention and renderer initialization conflicts.

A simulator cohort should share task, robot, controller and time horizon. Different tasks run as separate jobs; compatible inference requests can later be batched by a serving worker.

### 10.3 Episode ledger, chunks and resets

Use a manifest of episode IDs, initial-state/seed references, asset/randomization identities, instructions and maximum steps. Maintain episode ID, action queue, chunk age and observation step per environment. Clear policy caches, solver state, pending responses and remaining actions whenever an episode resets.

Primary benchmark should follow ManiSkill's standardized evaluation semantics: `ignore_terminations=True`, geometry reconfiguration each reset, fixed horizons and both `success_once`/`success_at_end` reporting. The [official setup](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html) explains these choices. For the first runner, explicitly disable automatic resets and reset full cohorts yourself at the horizon; collect native metrics before reset. If using automatic horizon resets later, correctly read `final_info` and never attach reset observations to the prior episode.

Schematic loop:

```python
for cohort in ledger.cohorts:
    obs = sim.reset(cohort)
    policy.reset(cohort.episode_ids)
    executor.clear_all()
    metrics.start(cohort)
    for step in range(cohort.horizon):
        if executor.needs_chunk():
            request = sim.policy_observation()
            chunk = policy.predict(request.rgb, request.instructions,
                                   request_meta=request.ids_and_steps)
            executor.install(chunk)
        model_action = executor.pop_next()
        command = codec.to_controller(codec.denormalize(model_action),
                                      sim.robot_context())
        result = sim.step(command)
        metrics.update(result.evaluator_metrics)
        recorder.maybe_write(result, chunk_metadata=executor.metadata())
    recorder.finalize(metrics.finish())
```

This is interface pseudocode. Batch readiness and partial active masks need real implementations. Clip execution at the evaluation horizon even if part of a 16-action chunk remains. If a special suite stops early for task failure, declare it rather than inheriting default environment termination behavior. Infrastructure errors should be recorded separately; model-produced invalid commands are policy failures under a declared policy.

### 10.4 Remote inference is optional infrastructure

Introduce a policy server only when model memory/throughput justifies separating simulation and inference. Server owns frozen VAE for fresh observations, text cache, DiT and bundle. Simulator worker owns physics/rendering/controller codecs or uses a shared codec library. Send RGB as uint8 arrays or shared-memory tensors where possible; avoid JPEG unless its effects are an experiment. Text embeddings can be cached; changing scene-image embeddings cannot.

Request: bundle/embodiment/layout IDs, camera tensors, instructions, episode IDs, observation step/time, desired horizon and deterministic per-request noise seed. Response: action chunk, optional flow latents, matching IDs/step and timing. Use bounded queues, request deadlines, batching only among compatible layouts and a clear retry policy. Reject stale responses from old episodes. No silently substituted commands on timeout.

Initial cross-node transport can be simple RPC with typed array serialization. Same-node shared memory avoids copies. Log serialization, encoding, queue, denoising, conversion and execution time separately. Avoid a heavyweight serving platform until local inference is correct.

### 10.5 Async/real-time evaluation

The paper reports overlapping planning and robot motion, but does not specify the full scheduling protocol. Predicting the next chunk before observing the actual chunk-boundary state creates stale conditioning. Implement real-time mode separately: observation timestamps, committed-action prefix, chunk cancellation/age policy and deadlines are explicit. If predicting from an earlier state without predicting intervening executed dynamics, measure that observation age and its cost.

Reference synchronous simulation provides the clean methodological result. Later real-time mode advances physics according to a declared clock during planning or injects measured planning delays; compare synchronous/full-chunk and asynchronous variants. Do not claim paper-equivalent real-time behavior merely because calls occur on another thread.

### 10.6 Motion/action consistency diagnostics

For selected episodes, save the initial query geometry and predicted flow. Execute the sampled action chunk; reconstruct realized simulator flow for those same anchor points/timestamps. Compare displacement error/direction on valid points, with separate robot/object/background regions and occlusion status. This directly tests the paper's observed failure mode: plausible predicted motion but incorrect actions.

Report scene-flow prediction versus teacher futures offline, and prediction-versus-realized-flow online. They answer different questions. Multiple futures can be valid; pointwise teacher error should not be the sole policy metric. Diagnostics use privileged state only after policy prediction and never influence commands in the reference run.

## 11. Evaluation design and experiment matrix

### 11.1 Primary metrics and episode count

Report task-macro-average success, per-task successes/trials, `success_once`, `success_at_end`, failure categories, time/control-steps to success, returns where meaningful, and ordered task progress for custom multi-step tasks. Task progress should be based on predeclared ordered predicates, not arbitrary dense reward. Latch completion if the task definition calls for irreversible progression; do not award later subtasks before earlier ones.

The paper uses only five rollouts per task/condition. Its reported differences motivate hypotheses but are not stable precision targets for a new simulator suite. For development, use a small fixed validation ledger, for example 25 episodes/task. For a main report, target at least 100 episodes/task/condition and three training seeds, increasing episodes based on pilot variance and the minimum effect of interest. At a true success rate near 50%, 100 independent episodes has roughly a ±10 percentage-point normal-approximation interval; correlated episodes and training variation require additional care.

Use Wilson intervals for individual success proportions and paired/bootstrap comparisons over matched episode initializations, with task/training-seed stratification. Keep training-seed variance visible. More evaluation episodes cannot replace training-seed replication. Macro-average tasks rather than allowing a task with more episodes to dominate. Paired initialization and noise-seed policy should be the same across target ablations.

Select all checkpoints/hyperparameters on validation; run sealed test ledgers only after selection. Publish incomplete jobs and exclusions. Rerun infrastructure failures with the same episode specification; log original errors. Count policy invalid actions/timeouts according to the declared protocol rather than quietly dropping hard episodes.

### 11.2 Required experiment groups

| ID | Model/data intervention | Question | Priority |
|---|---|---|---|
| B0 | Expert action replay | Does collection/controller/backend work? | Before learning |
| B1 | Practical compact BC or diffusion/flow policy, same observations/actions | Is the dataset learnable and evaluator functional? | Before large-scale training |
| B2 | Paper-style reduced scratch backbone on robot data | Does pretrained initialization matter in a comparable setup? | Main study if resources allow |
| F1 | Pretrained trunk + flow/action, robot-only training | Motion-target reference without human co-training | Main |
| F2 | Pretrained trunk + flow/action, mixed mid-training then robot SFT | Full adapted FloMo recipe | Main |
| V1 | Pretrained trunk + future video/action | Motion versus RGB prediction target | Main |
| V2 | Pretrained trunk + flow/video/action | Does adding RGB help or interfere? | Main |
| H1 | F2 minus broad human source | Contribution of human video | Main |
| H2 | Matched-compute replacement of human data with robot data | Data type versus extra training exposure | Main control |
| G1 | Estimated flow versus simulator-oracle flow | Cost of tracker/geometry noise | Diagnostic |
| I1 | Ground-truth future flow versus RGB inverse-dynamics probes at 1/10/100% labels | How decodable is the representation? | Mechanism |
| C1 | Execute 16 versus 8 versus 4 actions per replan | Open-loop drift/latency tradeoff | Later |
| O1 | Front-only versus multi-view flow | Occlusion limitation | Later extension |
| P1 | RGB/text only versus added robot proprioception | Ambiguity of current embodiment state | Later extension |

Practical BC and a paper-style scratch backbone answer different questions; do not label them identically. The paper itself describes its scratch BC both as a reduced 8-layer backbone and elsewhere as 20M, so match the declared architecture and report actual parameters rather than copying an ambiguous size label.

Run F1/F2/V1/V2 with the same action representation, camera inputs, splits, optimizer settings and checkpoint selection. Replacing flow with video preserves stream count; adding video increases sequence length and compute. Report both matched-update/sample and matched-compute budgets where feasible. For human removal, preserve total optimization budget and labeled robot exposure in one control, and report the natural mixture change in another. Simply removing a source changes several variables at once.

Optional strong published baselines such as DreamZero/AMPLIFY need their own compatible adapters and fair resource/data settings. They should use the same `Policy`/episode-ledger interface; do not spend the first implementation cycle on several complex baselines before F1 runs.

### 11.3 Inverse-dynamics probe implementation

Reuse the token packer/trunk but provide clean ground-truth future flow or RGB as additional conditioning and predict only noisy actions. Future conditioning tokens remain fixed throughout the action solver. This is a privileged probe, never a deployable policy or primary success result.

Use whole-episode nested subsets for 1%, 10%, 100% robot labels, identical held-out episodes and clean-future inputs, controlled training initialization/budget, and common noise/timestep evaluation draws. Report action flow-matching error and sampled action error in original units; note loss reductions. The paper reports minimum held-out action error, but we should predeclare checkpoint selection and additionally show final/selected results to avoid favoring a noisy minimum. This probe isolates inverse dynamics from future-prediction quality.

### 11.4 Attention and causal diagnostics

For the flow+video ablation, sample action-query attention by layer/head and target stream over a fixed held-out set. Efficient fused attention does not normally return maps; use a small instrumented debug path, not materialized full maps during every training step. Compute stream attention mass and per-token-normalized mass because unequal token counts affect totals. Record noise level and view layout.

Attention concentration alone does not establish a causal role. Add a controlled inference intervention: shuffle/drop motion conditioning in the privileged probe, or perturb predicted flow tokens with a documented coupled-sampler intervention, and measure action impact. Such perturbations are diagnostics and may be off-distribution; describe their limits. Main target/data ablations provide stronger evidence than an attention plot alone.

## 12. Compute, storage and job infrastructure

### 12.1 Resource pools

| Pool | Jobs | Proposed initial resources | Scaling condition |
|---|---|---|---|
| CPU ingest/conversion | Decode/index, HDF5 adapters, controller conversion, manifests, aggregates | CPU workers, ample RAM, shared storage | Decode/IO queue grows |
| Tracker preprocessing | SpatialTrackerV2, quality checks | Independent L40S-class or available GPU workers | Validated throughput and backlog |
| Encoder preprocessing | VAE/text embeddings | Separate GPU workers; cache repeated instructions | Encoding/augmentation becomes training bottleneck |
| Training | LoRA mid-training/SFT | Benchmark on one 80GB-class GPU, then one 8-GPU node | Need throughput or memory sharding |
| Simulation | Collection, replay/rendering, evaluation | One or more NVIDIA renderer-compatible GPUs | Policy or rendering throughput limits |
| Policy inference | DiT/VAE runtime, batching | Local to sim initially; separate GPU later | Model VRAM/latency limits simulator batches |

These are engineering allocations, not measured requirements or procurement recommendations. A 24GB device that can run upstream offloaded video inference is not proof it can train this joint model. The paper does not give training wall-clock throughput or cluster size, so accurate training cost needs a benchmark.

Keep evaluation resources reserved while training is running; otherwise failures appear only after an expensive run finishes. On a small budget, reuse the node in scheduled phases and evaluate selected checkpoints between training windows. On a cluster, issue evaluation jobs when completed checkpoint markers appear, with job dependencies and concurrency limits.

### 12.2 Preprocessing can dominate cost

Appendix B reports approximately 2.4 s per 16-frame tracking window on an L40S, or roughly 72 GPU-hours per hour of 30 FPS footage. Window stride explains the large difference between dense extraction and a nonoverlapping budget. For nonoverlapping windows:

```text
windows/hour ≈ 30 * 3600 / 16 = 6,750
tracking GPU-hours/hour ≈ 6,750 * 2.4 / 3600 = 4.5
```

The arithmetic above shows **4.5**, not 72, GPU-hours for nonoverlapping windows. Approximately **72 GPU-hours** corresponds to stride-one windows: `30*3600*2.4/3600`. Therefore the paper's stated conversion implies dense overlapping extraction or another comparable effective workload. The window stride must be recorded; do not budget using both the 72 multiplier and another 16× overlap factor.

At stride one, 81 h of human footage alone is about 5,832 tracker GPU-hours using the reported rate. Eight perfectly utilized equal-speed GPUs would take about 30 days; sixteen about 15 days, before IO/retries. At stride 16 it is roughly 365 GPU-hours, but supervision windows/coverage differ. These are arithmetic estimates based on the reported tracker speed, not measured local performance. Build and benchmark a task-balanced window manifest before launching all-footage extraction. Tracking a coarser set of anchor windows can reduce cost, but is a declared data-recipe change.

No need to track every possible window to bootstrap the project. First prepare thousands of stratified windows, validate them, and profile tracker throughput. Decide extraction stride, unique-window budget and source coverage together. Reuse caches across all target-ablation runs; prepare future RGB latents separately.

### 12.3 Storage estimates

At 256×256, uncompressed uint8 RGB is 196,608 bytes/frame; at 30 FPS, about 21.2 GB/hour/view in decimal units. Three cameras triple that. Keep raw source videos compressed, but avoid decoding them repeatedly from remote storage during training.

A 16×4,096×3 float32 trajectory/flow tensor is 786,432 bytes, about 0.79 MB/window, excluding validity, camera transforms and source images. A 48-channel, five-frame, 16×16 fp16 flow latent is 122,880 bytes, about 0.123 MB/window. Three single-frame observation latents add about 0.074 MB/window. Unique 512×4,096 bf16 text embeddings are about 4.19 MB/instruction, so duplicate-by-window text storage is wasteful.

At one million windows, flow latents alone are approximately 123 GB and numeric float32 tracks about 786 GB, before replication, observations, manifests and source media. Two caches for different augmentations/normalizations multiply this. Measure actual compression/shard overhead and use retention policies: preserve raw data, manifests/statistics and reconstructible provenance; retain expensive tracks; evict reproducible intermediate RGB visualizations.

### 12.4 Training profiling and estimates

Paper budget implies approximately 7.17M sampled examples in mid-training (`14,000*512`) and 384k in SFT (`1,500*256`). They are sampled exposures, not unique clips. These are useful reference workloads even when epoch definitions differ.

For each candidate batch/layout, measure encoder time, transfer/loader time, forward/backward, optimizer step, peak allocated/reserved VRAM, examples/s, GPU utilization, padding waste and checkpoint overhead. Include the one-view human/three-view robot mixture rather than profiling one easy layout only.

Compute projected wall time as `optimizer_steps * measured_seconds_per_optimizer_step`, then add measured checkpoint/evaluation/IO overhead and a contingency. Compute GPU-hours as wall hours times assigned GPUs. For comparison runs, report trunk token counts and actual total compute; LoRA rank alone does not describe training cost.

Use global batch accumulation: `global_batch = world_size * per_gpu_microbatch * accumulation`, with adjustments for mixed buckets. For example 8 GPUs × 2 samples × 32 accumulation = 512; this is an illustrative arithmetic configuration, not proof that microbatch 2 fits. Use proper DDP `no_sync` and sample-weighted loss accumulation. Resolve frozen/trainable sharding support for the chosen PyTorch/FSDP version before relying on it.

### 12.5 Operations and reproducibility

Each run stores resolved config, Git commit, dirty patch if any, image digest, upstream/model hashes, dataset/split/stats/layout IDs, seeds, world size, batch/accumulation, actual source weights and resource allocation. Save train/validation curves by source, gradient norms, NaN counts, action saturation and flow clipping. Monitor data-loader stalls and per-source rejection rates, not just overall loss.

Checkpoint interrupted jobs safely. Exact distributed resume may depend on world size and sampler implementation; declare whether a restart is exact or statistically resumed. Jobs must not silently overwrite runs/bundles. Use locked completion markers and content-addressed IDs. Minimal dashboards should answer: which data version was used, how far is the run, does action prediction improve, has rollout success changed, and what failed?

## 13. Proposed configuration shape

One composed recipe should be sufficient to reproduce a run. This YAML illustrates design decisions rather than a ready-to-execute implementation:

```yaml
experiment:
  name: flomo_panda_estimatedflow_v1
  track: maniskill_method_adaptation
  seed: 0
model:
  backbone: Wan-AI/Wan2.2-TI2V-5B
  revision: REQUIRED_PINNED_REVISION
  latent_channels: from_checkpoint
  resolution: [256, 256]
  lora: {rank: 64, alpha: 128, dropout: 0.0}
  lora_targets: [self_q, self_k, self_v, self_o,
                 cross_q, cross_k, cross_v, cross_o, ffn_0, ffn_2]
  targets: [scene_flow, action]
  proprio_conditioning: false
  positions: mixed_stream_offsets_v1
  clean_prefix_timestep: 0
data:
  manifest: prepared_dataset_v1
  split: train
  action_steps: 16
  flow_valid_frames: 16
  vae_temporal_policy: repeat_last_to_17
  query_grid: [64, 64]
  flow_frame: first_camera_cv
  flow_type: cumulative_displacement
  flow_provider: spatrackerv2_monocular
  normalization_id: REQUIRED_FROZEN_STATS_ID
  mixture_id: REQUIRED_SOURCE_WEIGHTS_ID
embodiment:
  spec: panda_native_ee_delta_pose_v1
  camera_layout: front_wrist_v1
  control_hz: REQUIRED_VERIFIED_FREQUENCY
stage:
  kind: midtrain
  init: pretrained_backbone
  lr: 0.00028
  global_batch: 512
  optimizer_steps: 14000
  loss: {scene_flow: 1.0, action: 0.1, video: 0.0}
  reduction: per_example_modality_mean_then_global_mean
optimizer:
  name: adamw
  betas: [0.95, 0.999]
  weight_decay: 0.000001
  grad_clip: 1.0
runtime:
  precision: bf16
  activation_checkpointing: true
  microbatch: PROFILE_FIRST
  accumulation: DERIVE_FROM_GLOBAL_BATCH
inference:
  solver: wan_flow_unipc_pinned
  steps: 10
  shift: 5.0
  cfg: false
  execute_steps: 16
evaluation:
  suite: validation_v1
  protocol: synchronous_fixed_horizon
  reconfiguration_freq: 1
  ignore_terminations: true
  auto_reset: false
```

SFT override: initialize from a selected mid-training bundle/checkpoint; use target robot demonstrations/play only, weights 0.75/0.25 if available and appropriate; LR 5.6e-5; global batch 256; approximately 1,500 steps; action weight 1; optimizer reset policy explicit. No changes to bundle/action/flow contracts should be needed.

Required pins/frequencies are intentionally unresolved fields that the validation command must reject before launching training. Do not let placeholders become silent defaults.

## 14. Team organization and implementation ownership

For a small team, assign responsibility by stable contracts rather than by loosely named stages:

| Owner | Owns | Deliverable/interface |
|---|---|---|
| Research/model lead | Paper interpretation, token layout, Wan integration, objective/sampler | Model and training sample contract; model verification |
| Data/geometry engineer | Source adapters, splitting, tracking, canonicalization, caches, statistics | Validated prepared dataset with lineage |
| Simulation/evaluation engineer | ManiSkill tasks/cameras/controllers, experts, rollout runner, metrics | Episode ledger + tested policy/action interface |
| Infrastructure owner, possibly shared | Images, scheduler, storage, distributed runs, export/resume, monitoring | Repeatable jobs and compatible immutable bundles |

One person can hold multiple roles; three engineers/researchers can progress in parallel once the contracts are agreed. With one person, prioritize a vertical slice and reduce source/task breadth. Training and data engineers should jointly review loss normalization and augmentation; model and simulator engineers should jointly review action/timing specs. A single accountable lead approves changes to representations and benchmark definitions.

Define done in terms of artifacts and gates, not code merged: a data adapter is done when sampled episodes validate and replay; a model wrapper is done when base compatibility and gradient tests pass; evaluation is done when expert/baseline rollouts and metrics are correct. Every architecture deviation gets a short decision record with rationale and the affected comparison.

Maintain four operational documents: paper-to-code map, decision log, dataset/model artifact registry, and runbook for collection/preprocessing/training/evaluation failures. Weekly review should prioritize success/failure videos, motion/action mismatch and coverage gaps alongside loss plots. Avoid selecting only attractive motion visualizations; show randomized examples and failures.

## 15. Implementation roadmap with dependencies and acceptance gates

Indicative timing assumes roughly three experienced contributors, accessible GPUs, and no major asset/data-access delays. It is a planning range, not a deadline promise. Full tracker preprocessing can lengthen it substantially.

| Milestone | Indicative duration | Work | Exit gate |
|---|---|---|---|
| M0: settle contracts | 2-4 days | Pin upstreams; inspect weights; decide VAE/time/action/camera semantics; define suite/splits | Approved specs and compatibility report; no unknown shape/timing assumptions |
| M1: simulator/data vertical slice | 1-2 weeks | One stock task; expert replay; camera capture; native codec; canonical episodes/windows; oracle flow | Expert succeeds in evaluation backend; geometry tests and episode alignment pass |
| M2: tiny FloMo loop | 1-2 weeks | Wan wrapper, LoRA, mixed tokens, flow-matching, sampler, bundle, local evaluator | Fit a tiny fixed dataset; seed-repeatable sampling; one-task rollout succeeds above random |
| M3: robust data/training | 1-2 weeks overlapping M2 | Tracker pipeline; source adapters; caching/augmentation; distributed trainer; resume/export | Tracker/oracle audit, data gates, multi-GPU objective equivalence and interruption recovery |
| M4: pilot mid-training/SFT | 1-2 weeks after data ready | Small human/robot mixture; robot-only SFT; validation rollout monitoring | Full stage handoff without manual conversions; learnable baseline; measured scaling budget |
| M5: main controlled study | 2-4+ weeks | F1/F2/V1/V2/H controls; training seeds; ID/OOD ledgers; inverse-dynamics probe | Reproducible tables, confidence intervals, diagnostic videos and compute accounting |
| M6: extensions | separately budgeted | Bimanual/custom tasks; full-source scale; multi-view flow; real-time serving | Named extension results; no regression in reference protocol |

The first useful target is a reproducible single-task end-to-end run in roughly 2-3 weeks if compatibility and GPU setup are straightforward. A convincing multi-task controlled study is more plausibly 8-12+ weeks including overlap, and can be longer if tens of thousands of GPU-hours of tracking are required. Do not begin by recreating every original real-world task.

### M0 backlog

* Retrieve/pin Wan and tracker versions, inspect checkpoint/config and exact solver.
* Resolve 48-channel compatibility and 16/17-frame policy with tensor fixtures.
* Select robot/control rate/native action representation and supported tasks.
* Define policy input whitelist, camera names and full action semantics.
* Draft source licenses/access manifest and original-data substitution table.
* Freeze validation/test grouping and OOD definitions before collection.

### M1 backlog

* Headless NVIDIA/Vulkan renderer smoke test; verify camera RGB versus visualization output.
* Download or generate demonstrations; CPU conversion only where needed; action replay in evaluation backend.
* Persist canonical episode with timestamps and controller metadata.
* Implement frame conversions, rigid/link oracle tracks and cumulative flow renderer.
* Fit train-only statistics; inspect raw/encoded/decoded flow.
* Implement fixed-horizon ledger runner with expert and random `Policy` adapters.

### M2-M3 backlog

* Implement mixed token spans, rotary/time handling and action heads.
* Inject/verify LoRA targets and freeze boundaries.
* Implement masked joint loss and signed solver reference tests.
* Overfit 32-128 fixed windows and run local rollouts; compare with a compact baseline.
* Add estimated tracker samples and source/layout-aware collation.
* Implement weighted distributed accumulation, atomic checkpoints, restart, bundle export and loader validation.
* Add deterministic/offline versus online augmentation equivalence fixtures.

### M4-M5 backlog

* Profile node throughput and approve a measured compute budget.
* Expand task/source coverage only after source-wise flow quality gates.
* Train pilot mid-training and SFT with validation rollouts.
* Launch predeclared target/human controls and nested inverse-dynamics probes.
* Run sealed episode ledgers, aggregate intervals, log failure categories and artifact lineage.
* Package configs, data manifests/statistics, bundles, evaluation records and a reproduction runbook.

## 16. Verification strategy and failure triage

This is scientific infrastructure with several high-impact silent-failure risks. Tests should target those risks, rather than duplicate every line of implementation.

| Gate | Necessary verification |
|---|---|
| Geometry | Static under moving camera; translated/rotated objects; articulation links; mm/meters and GL/CV conversions; anchor indexing |
| Data | Episode-level split isolation; no window boundary crossings; N+1 observations; timestamp alignment; no fabricated action labels |
| Normalization | Train-only stats; zero/constant channel behavior; inverse mapping; action clipping/gripper semantics; incompatible bundle refusal |
| VAE | Loaded dimensions; temporal coverage; frozen gradients; round-trip motion preservation; cache key fidelity |
| Model | Token spans/unpatchification; padding/absent views; clean prefixes; action-space noise; correct LoRA trainable parameters |
| Objective | Shared timestep; masked labels; per-modality mean; global weighted accumulation; single versus multi-GPU gradient agreement |
| Sampler | Integration sign; analytic fixture; schedule/version; seeded repeatability; unchanged prefixes; correct target lengths |
| Simulation | Expert action replay; native controller unit commands; horizon metrics; reset clears chunks; actual camera inputs |
| Stage boundary | Mid-training bundle loads into SFT/eval; same stats/action meanings; documented optimizer reset/resume |
| Evaluation | Matched initializations; successes counted once per episode; final-info handling; no privileged inputs; all attempts accounted for |

Failure triage order:

1. **Expert replay fails:** fix controller/backend/state replay first; no model comparison is meaningful yet.
2. **Compact baseline cannot learn:** inspect labels, alignment, visibility, language coverage and controller difficulty.
3. **FloMo cannot overfit tiny data:** inspect token routing, loss sign/scaling, normalization, temporal truncation and gradient flow.
4. **Motion correct, actions wrong:** inspect action codec and grounding; compare privileged inverse-dynamics probes and realized-flow consistency.
5. **Tracker flow poor, oracle works:** fix estimator mode, camera transforms, scale, visibility and preprocessing quality.
6. **ID strong, OOD weak:** inspect actual held-out source coverage and whether human data contains the relevant primitive; evaluate object grounding separately from motion.
7. **Quality strong, latency poor:** optimize packing, batched observation encoding and model runtime; change sampler/chunk length only in named experiments.

Promote changes only after the corresponding gate passes. Do not keep increasing data or GPUs to compensate for a shape/timing bug.

## 17. Risks, limits and first decisions

The largest risks are unavailable in-house data, uncertain tracker scale/coordinates, costly overlapping extraction, latent loss of small motion, ambiguous action semantics, camera occlusion, and motion/action disagreement. The Wan VAE/time discrepancy is an immediate compatibility issue, not an optional polish item. ManiSkill makes repeated evaluation easier, but does not remove the human-to-robot action grounding problem.

Before implementation, settle these concrete decisions:

1. **Embodiment:** single-arm Panda first, or immediate bimanual fidelity? Recommendation: Panda vertical slice, bimanual later.
2. **Compute:** available training/renderer GPUs and tracker GPU-hour budget? Recommendation: profile one 80GB-class training GPU and a small preprocessing sample before reserving a full node.
3. **Human data:** access to EgoVerse and task-related footage? Recommendation: a real human-data path for transfer claims; synthetic action-free footage only for bootstrap/controls.
4. **Actions:** native controller space or paper-style 10D-per-arm pose targets? Recommendation: native controller space first, with an explicit adaptation label.
5. **Flow:** monocular tracker as the main supervision, RGB-D+poses, or oracle? Recommendation: oracle for correctness checks, estimated flow for the method study, named quality ablations.
6. **Evaluation goal:** mechanism reproduction in stock tasks or original-task analogues? Recommendation: mechanism study first with a sealed ID/OOD suite.

These decisions change scope/budget, but do not prevent building the contracts and one-task vertical slice now. The default assumptions throughout this plan are single-arm native control, synchronous evaluation, pretrained Wan acquisition rather than new backbone pretraining, and eventual real human-video mid-training.

## 18. Completion criteria and source map

A completed simulation reimplementation should provide:

* A runnable pinned repository and container images with tested renderer/training setup.
* Canonical source adapters, split/window manifests, flow statistics and sample lineage.
* Verified estimated and oracle flow paths, plus visual/numeric quality reports.
* Joint Wan flow/action model, reference sampler, trainable-parameter inventory and documented deviations.
* One trainer supporting mixed mid-training and robot-only SFT, reliable resume and bundle export.
* A common `Policy`/`ActionCodec`/`SimAdapter` interface and reset-safe chunk evaluator.
* ID/OOD episode ledgers and per-episode records, model/task-seed variation and intervals.
* Motion/video/human-data controls, at least a learnability baseline, and motion/action mismatch diagnostics.
* Measured preprocessing/training/evaluation compute and policy latency.
* A reproduction runbook and clear distinction between ManiSkill results and the paper's original real-robot results.

Source map:

| Source | Used for |
|---|---|
| [FloMo PDF](https://flomo-wam.github.io/FloMo%20Paper.pdf) | Full method, experiments, Appendix A settings, Appendix B subsets/weights/costs, Appendix E limitations |
| [FloMo project](https://flomo-wam.github.io/) | Architecture figures and current code-availability notice |
| [Official Wan 2.2](https://github.com/Wan-Video/Wan2.2) | Backbone/VAE integration and upstream runtime reference |
| [TI2V-5B checkpoint config](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/config.json) | Actual 48-channel model compatibility |
| [TI2V-5B upstream config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py) | Strides, patch size, transformer dimensions and upstream schedule defaults |
| [Wan 2.2 VAE implementation](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py) | Channels and causal first-frame/four-frame temporal encoding |
| [Wan model implementation](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py) | Standard grid assumptions, per-token timesteps and required mixed-token adaptation |
| [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) | Monocular and RGB-D/pose extraction paths |
| [ManiSkill demonstration setup](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) | Standard demo preparation and benchmark consistency |
| [ManiSkill replay](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) | State/action replay, conversion limitations, backend differences |
| [ManiSkill observations](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html) | Sensor dictionaries, units, calibration and privileged channels |
| [ManiSkill controllers](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html) | Controller semantics and action-space/frame requirements |
| [ManiSkill quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html) | Batched tensors, GPU physics/rendering and device organization |
| [ManiSkill evaluation setup](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html) | Fixed-horizon evaluation, reconfiguration and success metrics |

Paper facts are attributed in §§2-3. Infrastructure, contracts, experiment recommendations and timing estimates are proposed design choices. Upstream URLs are references to inspected behavior; actual implementation must pin revisions instead of tracking `main`/`latest`. The inspected PDF's SHA-256 is `ee305465f304d3c378c4a79ab99187287d7bf3842f05e2d089da2730ee9d79b1`.
