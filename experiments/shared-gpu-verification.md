# Shared GPU acceptance — 2026-10-08

Actual B300 tests and environment checks:

```json
{
  "torch": "2.12.0+cu130",
  "cuda": "13.0",
  "device": "NVIDIA B300 SXM6 AC",
  "capability": [
    10,
    3
  ],
  "compiled_arches": [
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_90",
    "sm_100",
    "sm_120"
  ],
  "loss": 123.91661071777344,
  "bf16_backward_finite": true
}
```

```json
{
  "nccl_two_gpu_all_reduce": true,
  "sum": 3.0,
  "visible_physical_gpus": [
    6,
    7
  ]
}
```

Original repository checks: 48 passed in 6.01 seconds under this core runtime. Real Wan/LTX VAE acceptance, dataset/tracker jobs, and LoRA policy tests have separate per-experiment reports. These checks establish executable infrastructure, not task competence.

Core runtime adds protobuf4.25.8 for the pinned T5 tokenizer. Versioned package freeze is stored in the experiment control archive.
