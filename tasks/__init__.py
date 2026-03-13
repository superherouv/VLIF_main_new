"""
Task definitions for 5 low-level vision benchmarks.

Each task module provides:
  - Task-specific model wrapper (handles task peculiarities)
  - Loss function configuration
  - Evaluation metrics
  - Dataset paths and split conventions

Tasks:
  1. Deraining  : Remove rain streaks (sparse, high-frequency artifacts)
  2. Denoising  : Remove Gaussian/real noise (dense, signal-dependent)
  3. Super-Res  : Upscale low-resolution images (signal reconstruction)
  4. Dehazing   : Remove atmospheric haze (multiplicative + additive)
  5. Low-light  : Enhance underexposed images (non-linear mapping)

SNN Applicability Hypothesis per Task:
  ┌──────────────┬──────────────┬────────────────────────────────────────────┐
  │ Task         │ SNN Suitab.  │ Reason                                     │
  ├──────────────┼──────────────┼────────────────────────────────────────────┤
  │ Deraining    │ HIGH         │ Rain=sparse events; SNN excels at sparse    │
  │              │              │ signal detection; temporal patterns useful  │
  ├──────────────┼──────────────┼────────────────────────────────────────────┤
  │ Denoising    │ MEDIUM       │ Noise is dense; SNN less efficient but      │
  │              │              │ still competitive at moderate noise levels  │
  ├──────────────┼──────────────┼────────────────────────────────────────────┤
  │ Super-Res    │ LOW-MEDIUM   │ Requires dense high-freq reconstruction;    │
  │              │              │ SNN binary repr limits fine detail recovery │
  ├──────────────┼──────────────┼────────────────────────────────────────────┤
  │ Dehazing     │ MEDIUM-HIGH  │ Haze is spatially smooth; SNN can detect   │
  │              │              │ transmission map edges efficiently          │
  ├──────────────┼──────────────┼────────────────────────────────────────────┤
  │ Low-light    │ LOW          │ Requires precise non-linear tone mapping;   │
  │              │              │ binary spikes struggle with smooth gradients│
  └──────────────┴──────────────┴────────────────────────────────────────────┘
"""

from .deraining import DerainingTask
from .denoising import DenoisingTask
from .superresolution import SuperResolutionTask
from .dehazing import DehazingTask
from .lowlight import LowLightTask

TASK_REGISTRY = {
    "deraining": DerainingTask,
    "denoising": DenoisingTask,
    "superresolution": SuperResolutionTask,
    "dehazing": DehazingTask,
    "lowlight": LowLightTask,
}


def build_task(task_name: str, **kwargs):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_name}'. Choose from {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_name](**kwargs)


__all__ = [
    "DerainingTask", "DenoisingTask", "SuperResolutionTask",
    "DehazingTask", "LowLightTask", "TASK_REGISTRY", "build_task",
]
