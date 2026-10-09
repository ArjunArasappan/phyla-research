#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/nvme/scratch/phyla-ubuntu/cache/pip
export HF_HOME=/mnt/nvme/scratch/phyla-ubuntu/cache/huggingface
PY=/mnt/nvme/scratch/phyla-ubuntu/environments/core/bin/python
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
$PY -m pip install 'numpy>=1.26,<2' PyYAML h5py pytest 'diffusers==0.33.1' 'transformers==4.49.0' safetensors 'accelerate>=0.30,<2' sentencepiece ftfy 'opencv-python==4.10.0.84' 'imageio[ffmpeg]' matplotlib scipy huggingface_hub einops easydict
CUDA_VISIBLE_DEVICES=7 $PY - <<'PY'
import json, torch
from pathlib import Path
assert torch.cuda.is_available()
x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=(x@x.T).float().square().mean()
y.backward()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
report={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'compiled_arches':torch.cuda.get_arch_list(),'loss':y.item(),'bf16_backward_finite':True}
Path('/mnt/nvme/scratch/phyla-ubuntu/control/core-ready.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
