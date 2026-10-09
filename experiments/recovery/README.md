# H100 evaluation recovery

The saved Experiment3 core archive is restored under `/workspace/phyla-research-runs/exp03-recovery-root`. `/mnt/nvme/scratch/phyla-ubuntu` is an alias to that directory, preserving checksummed bundle metadata and checkpoint paths.

`restore_and_download.py` restores archived files without chmod operations and downloads exact pinned LTX0.9.6 weights and0.9.5 T5/config assets. Downloads are finalized only with a byte count and SHA-256 marker.

`run_recovered_eval.py` verifies each saved learned bundle and exact frozen-base parameter digest, then evaluates twelve policies sequentially on one GPU. An exclusive process lock prevents duplicate queues. Completed cells are skipped; incomplete attempts are retained separately. No historical deadline applies.

The H100 does not initialize NVIDIA Vulkan on this node. Mesa CPU Vulkan rendering passed a 256×256 PushCube RGB reset check; inference uses the H100. Set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`. `eval.render_backend` explicitly records this choice.

`check_eval_contracts.py` verifies recorded physical resets, episode seeds, dimensions and forecast types after the full cohort. `report_evaluations.py` writes measured counts, plots and video links without interpretation.

Runtime logs/status, model assets, arrays, videos and reports live on `/workspace/phyla-research-runs`. Original full evaluations/optimizer checkpoints were lost; all learned adapters and demonstrations survived.

Environment: Torch2.8.0+cu128, Diffusers0.33.1, Transformers4.49.0, ManiSkill3.0.1, SAPIEN3.0.3, MPLib0.1.1. Validation:53 repository tests and actual CPU-rendered native8D reset passed.
