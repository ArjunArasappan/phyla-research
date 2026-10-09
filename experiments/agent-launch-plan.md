# Three-agent GPU experiment launch plan

**Planning date:** 2026-10-08, America/Los_Angeles. **Status:** proposed, no experiment agents or GPU jobs launched by this plan.

## 1. Verified starting state

SSH works as `ubuntu@204.52.23.47` using the local `~/.ssh/national_compute` key with `IdentitiesOnly=yes`. The private key stays on the laptop; it is not copied into the repository or onto the GPU host.

The host has eight NVIDIA B300 SXM6 GPUs, reporting compute capability 10.3 and approximately 269 GiB device memory each. All eight were idle at inspection. It has roughly 2.8 TiB RAM, a 14 TB NVMe mount, Python 3.12.3, driver 580.167.08, and an installed CUDA 13.0.2 toolkit. PyTorch is not installed in system Python. `tmux` and Git are available. The CUDA compiler exists at `/usr/local/cuda/bin/nvcc` but was not on the inspected PATH. No remote Codex CLI was found; the agents can work through SSH and do not require one.

The cloned repository is `/home/ubuntu/phyla-research`, inspected at commit `e94160e` with a clean working tree. It defines:

- Experiment 1: simulator-grounded tracker and motion-label VAE benchmark.
- Experiment 3: flow-versus-RGB data efficiency and OOD policy generalization.
- Experiment 2: frozen Wan/LTX VAE reconstruction of saved 3D displacement labels, split from the original Experiment 1 by user instruction.

`/mnt/nvme` itself is root-owned and not writable by ubuntu. `/mnt/nvme/scratch` is world-writable with a sticky bit. At setup, create an ubuntu-owned experiment workspace below that directory and verify ownership and write access. Do not place model caches on the 104 GB free root filesystem when the NVMe workspace is available.

## 2. Agent model: three workers and one coordinator

Launch three named worker agents under the coordinating chat. Each owns one experiment, its implementation checkout, run manifests, notes, and results. I retain shared-environment setup, GPU leases, integration of shared changes, and the aggregate status view.

The coding agents run through the current agent tools and launch remote Python jobs over SSH. They are distinct from the CUDA processes they supervise. Losing an SSH connection should not destroy a training job: GPU jobs run in persistent tmux sessions with log files and saved checkpoints.

Monitoring is provided by independent experiment status files, terminal sessions, progress notes, and metric logs. This does not depend on exposing a complete child-agent conversation in the app. If separate, independently steerable Codex chats are desired, that is a distinct task/thread creation choice; no such chats are created by this planning document. Official documentation describes [subagent delegation](https://learn.chatgpt.com/docs/agent-configuration/subagents) and [remote engineering workflows](https://developers.openai.com/blog/mastering-codex-remote-for-engineering).

| Worker | Scope | Initial GPU reservation | First useful milestone |
|---|---|---|---|
| `experiment_01` | GT simulator oracle and tracker comparisons | 0, 1 | Two validated GT clips and canonical tracker artifacts |
| `experiment_02` | Frozen motion-RGB rendering and Wan/LTX VAE reconstruction | 2, 3 | Validated GT-through-VAE reconstruction and error decomposition |
| `experiment_03` | Matched flow/RGB/action-only policies, sample efficiency, OOD evaluation | 4, 5 | Matched data contracts and one-task policy pilot |
| Coordinator/shared services | Environment smoke tests, controlled evaluation, setup | 6, 7 | Validated CUDA, codec, and simulator stack |

These are reservations, not instructions to allocate all eight GPUs immediately. While a worker is CPU-bound or waiting on inputs, unused devices remain idle unless the coordinator explicitly grants a temporary lease.

### GPU rules

- Set `CUDA_VISIBLE_DEVICES` per process, ideally with recorded GPU UUIDs. Local `cuda:0` means the first visible GPU, not necessarily physical GPU 0.
- Record physical indices/UUIDs and visible-device mapping in every run.
- Start with one process per GPU. For the large-memory B300s, independent runs often offer a simpler initial schedule than DDP; confirm model fit and throughput first.
- Experiments 1 and 2 use their reserved pairs for independent tracker or codec jobs, not unnecessary distributed inference.
- Experiment 3 can train matched RGB and flow cells concurrently on its pair, then schedule action-only cells and additional seeds. Keep experimental budgets unchanged by concurrency.
- DDP is an optional profiled mode within an experiment's reserved pair. Match effective batch and optimizer updates; do not silently change the comparison.
- GPU 6 is initially available for leased rollout evaluation; GPU 7 for short compatibility/simulator tests. Never let each agent independently assume those devices are free.
- A simple coordinator-owned lease registry plus filesystem locks is enough; no Kubernetes/Ray/Slurm service is needed for this single-node pilot.

## 3. Shared setup before scientific GPU runs

### 3.1 Workspace and pinned source

Proposed layout:

```text
/home/ubuntu/phyla-research/                  # main integration checkout
/mnt/nvme/scratch/phyla-ubuntu/
  worktrees/
    exp01/
    exp02/
    exp03/
  environments/                             # versioned, method-specific envs
  cache/
    checkpoints/                            # immutable, content-hashed
    downloads/                              # coordinator-owned staging
  data/
    exp01/
    exp02/
    exp03/
  runs/
    exp01/<run_id>/
    exp02/<run_id>/
    exp03/<run_id>/
  control/
    gpu_leases.json
    exp01/status.json
    exp02/status.json
    exp03/status.json
    aggregate_status.json
```

Use isolated remote worktrees/checkouts so workers do not race over the same files. Shared library changes have a single owner and are integrated deliberately. A running job uses a pinned code snapshot; an agent must not edit the imported source underneath an active run.

Each run records the actual code commit, dirty-diff hash if applicable, upstream revisions, environment lock, dataset/normalization hashes, checkpoint hashes, and launch command. A proposed config is not a completed run.

### 3.2 B300 compatibility gate

The previous tracker recommendations include older PyTorch/CUDA environments. Do not blindly install those exact old binaries and assume B300 compatibility. Start with a current Blackwell-capable PyTorch CUDA build, pin the tested version, and test actual device kernels. Official [PyTorch release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/) recommends CUDA 13.0+ wheels for newer Blackwell GPUs; that guidance does not guarantee every tracker extension will compile or execute on this particular host.

Acceptance checks include device allocation, BF16 matrix multiplication and backward, convolution, attention, a saved/reloaded checkpoint, and any required compiled point-operations kernels. Verify `torch.cuda.get_device_capability()`, compiled architecture support, and actual numeric results. Run a two-GPU NCCL collective smoke check before DDP. Do not replace failed scientific kernels with fabricated results or CPU stubs and label them GPU acceptance.

Use separate environments for tracker stacks, simulator, and model training/codecs where dependencies conflict. Python 3.12 is the host default, not a requirement for all environments. Provision older Python interpreters where a legacy upstream requires them, but use CUDA-compatible Torch/extension builds and validate any source adaptations. Record deviations from upstream pins.

Prefer standard PyTorch attention first. Treat optional FlashAttention/custom kernels as optimizations requiring compatibility tests. Track installation/build failures per method; one optional tracker should not block GT, CoTracker, SpaTracker, and codec validation indefinitely.

### 3.3 Shared asset/cache rules

One owner downloads each pinned model snapshot and validates its hash. Other agents consume completed cache entries; partial downloads are never treated as ready. Use lock files and atomic completion markers. Do not run three independent multi-gigabyte downloads of identical weights.

Experiment 2 loads only VAEs for reconstruction, not T5 or diffusion transformers. Experiment 3 needs full selected backbone/text components. Both consume the same verified VAE weights when making a within-backbone comparison. Codec-only readiness and policy-stack readiness are separate flags.

No agent changes another agent's environment in place. Create a new environment identity for incompatible upgrades. Export a small reproducible environment manifest after acceptance tests, rather than relying on the mutable system Python.

## 4. Dependencies and launch order

Three agents can work concurrently, but not every expensive run is scientifically independent.

```text
shared CUDA/environment and storage gate
  ├─ Agent 1: oracle validation -> tracker pilot -> tracking benchmark
  ├─ Agent 2: renderer tests + VAE setup
  │               -> first valid GT bundles -> codec pilot
  │               -> tracker bundles incrementally -> reconstruction benchmark
  └─ Agent 3: splits/OOD tooling + matched policy contracts
                  -> validated label/codec + real-policy smoke tests
                  -> overfit test -> paired pilot -> frozen sweep
```

Experiment 2 can start codec setup immediately and GT reconstruction after the first validated GT bundle; it does not wait for the complete tracker matrix. Its primary test scoring waits for frozen GT calibration bounds. Experiment 3 prepares data/evaluation infrastructure in parallel, starts GT-flow pilots after label/codec gates, and starts predicted-flow extensions after a validated tracker pipeline exists.

Reuse code and model components, not held-out test data. Do not use Experiment 1 evaluation scenes as Experiment 3 training data without declaring overlap.

Initial order: provision common storage/runtime; start the three workers with bounded task packets; validate GT, codec, and policy interfaces; run two-clip/GT-codec/one-task pilots; then freeze protocols and expand the ready studies. No agent is launched by this planning document.

## 5. Worker assignments

### Agent 1: tracker benchmark

Read the revised Experiment 1 protocol. Generate exact camera/material-point GT, validate projection/visibility/camera-only zero flow, and publish immutable raw/canonical bundles. Run SpaTrackerV2, CoTracker3 plus depth, GT lifting controls, and independent 3D trackers. Measure geometry, coverage, camera-motion effects, and occlusion behavior. No VAE reconstruction or training belongs to this worker.

Use GPUs 0/1 for independent tracker/clip jobs. Begin with two clips, then the 12-clip pilot; the 90-clip expansion follows validation. Publish each source bundle separately so Agent 2 can consume ready data.

### Agent 2: frozen VAE reconstruction

Read the new Experiment 2 protocol. Implement shared RGB displacement mapping, inverse rendering, clipping/quantization/raster controls, and VAE-only loaders. Load frozen Wan/LTX VAEs; no DiT, T5, or fine-tuning is involved.

Use GPUs 2/3, initially one codec per device. Validate GT reconstruction first, then consume tracker bundles as they arrive. Report codec self-distortion, decoded-versus-GT error, GT codec floors, clipping/rasterization loss, temporal/axis/boundary behavior, latent rate, and costs. Save latent/decoded/recovered-flow artifacts and side-by-side figures. Keep codec validation independent of tracker label accuracy.

### Agent 3: flow versus RGB data efficiency/OOD

Audit matching current observations, future timestamps, padding, target token dimensions, action scaling, trainable parameters, and inference inputs. Construct nested episode subsets and independent ID/OOD manifests. Confirm all rollouts infer the future rather than receiving true future labels.

Begin with one task, three arms (action-only, action+RGB, action+GT flow), N = 16 and 64, one subset seed, and two optimizer seeds: 12 training runs. Use GPUs 4/5 for paired independent cells; lease GPU 6 only for evaluation when available. Record actual runtime and convergence before requesting a larger sweep.

The full tentative design is 324 core training runs plus a 54-run selected predicted-flow extension, excluding tuning. Those numbers are a planning matrix, not an automatic instruction to launch everything. Freeze sample counts, compute exposure, tuning/checkpoint policy, and OOD conditions before scale-up. Deliver success-versus-N curves, paired flow-minus-RGB effects, OOD success/gaps, seed-level data, and failure videos.

## 6. Monitoring each agent and each job

Each experiment has three separate monitoring layers:

1. **Agent progress:** dated Markdown notes and a concise `STATUS.md`, with current stage, completed milestone, next step, unresolved issue, and links to runs. Existing `ai_notes/` is AI-owned; never overwrite `user_notes/` or `results/user/`.
2. **Job health:** an atomic `status.json` and append-only event log per run, including PID/process group, GPU UUID, command, heartbeat timestamp, exit code, current unit/step, progress totals, and artifact/checkpoint locations.
3. **Scientific metrics:** training/validation scalar logs, per-clip or per-rollout results, tables, images, and a small HTML report. Status is not a substitute for quantitative outputs.

Initial persistent sessions: `phyla-exp01`, `phyla-exp02`, and `phyla-exp03`. Each session is created when that worker has an actual workload. Multi-job sessions can expose separate log panes; active-job IDs/PIDs must remain explicit.

Example monitoring commands, once the sessions/files exist:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/national_compute ubuntu@204.52.23.47

tmux attach -t phyla-exp01
tmux attach -t phyla-exp03
nvidia-smi

cat /mnt/nvme/scratch/phyla-ubuntu/control/aggregate_status.json
tail -f /mnt/nvme/scratch/phyla-ubuntu/runs/exp03/<run_id>/stdout.log
```

One tmux attach command is used at a time; detach with Ctrl-b then d. These are planned session names, not currently running sessions. Opening a monitor does not start another training copy.

Have workers update machine-readable progress every 30–60 seconds where practical. A stale heartbeat and a missing process differ from a long-running download/build or an idle agent waiting on inputs. Status states should distinguish `setup`, `running`, `waiting_for_dependency`, `needs_input`, `failed`, and `complete`.

A planned `STATUS.md` per experiment should expose agent and job progress separately. The coordinator produces an aggregate table showing current stage, GPUs, latest metric, elapsed time, last heartbeat, next milestone, and blockers. CSV/JSON and Markdown are sufficient first; a lightweight HTML status page can follow. Do not require a monitoring service, public port, paid tracker, or external account. If TensorBoard is installed later, bind locally and view by SSH forwarding.

For agent-process continuity, persist task packets and notes before compaction or handoff. tmux preserves GPU processes, not an indefinitely running coding-agent turn. When an agent resumes, it checks saved job IDs and existing completion markers before launching anything.

## 7. Failures, stopping, and recovery

- Checkpoint training regularly and save RNG/optimizer/scheduler state needed for the chosen resume mode.
- Resume incomplete units; reuse immutable simulation/track/codec artifacts only when hashes match.
- Stop by a registered run PID/process group or job-specific cancel file, not `killall python` or all-node GPU resets.
- Stopping an agent and stopping its detached GPU jobs are separate operations. User stop requests must explicitly propagate to that agent's registered jobs, then confirm exit and preserve artifacts.
- Finite retry budgets for transient downloads/SSH failures; reproducible numerical/compiler failures become explicit blockers with logs.
- A GPU lease is released only after the registered processes exit. Check for orphaned CUDA jobs before reallocating devices.
- Never silently replace a failed scene, drop a method from a denominator, change a target/window convention, or relax an OOD split to make a table complete.
- Worker commits document tested changes. Coordinator reviews/integrates shared changes. No simultaneous edits or pushes to the same branch, and no automatic merging of incompatible experiment code.

## 8. Mandatory parameter-efficient adaptation

Only Experiment 3 fine-tunes Wan/LTX DiTs. All such adaptation uses LoRA SFT, including any action-free midtraining stage: freeze pretrained base transformer weights, VAEs, and text encoders; train LoRA adapters and declared new action/interface heads only. Matched RGB/flow arms share the LoRA modules/rank and training budget. Check optimizer allowlists, frozen gradients/weights, and adapter/base provenance before running a sweep. Experiments 1 and 2 are frozen inference and reconstruction studies.

## 9. Ready-to-launch task packet template

Every worker receives:

- exact experiment writeup and frozen protocol/commit;
- SSH host, approved identity path, workspace checkout, environment, and GPU reservation;
- owned files plus shared-code change procedure;
- first bounded milestone and acceptance checks;
- data/checkpoint source pins and input-privilege rules;
- artifact/status/log/checkpoint paths;
- stop/resume behavior and retry limits;
- instructions to preserve your personal notes/results;
- requirement to report measured outcomes and failures without fabricating completion.

The first task is a verified pilot, not “run the entire paper.” After the pilot, runtime/quality measurements determine the next batch.

## 10. Remaining launch decisions

All three scientific scopes are now defined. Operational choices to resolve during setup include: tested Torch/runtime pins, whether the selected backbone fits one GPU at useful throughput, exact simulator/render configuration, and measured pilot cost.

This plan reserves all three experiment workers and makes their monitoring concrete, while preserving the dependency gates needed for meaningful comparisons. It does not launch agents, install packages, download weights, or start scientific runs by itself.
