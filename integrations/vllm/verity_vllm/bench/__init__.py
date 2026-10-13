"""The shared vLLM overhead benchmark.

The public pieces are deliberately small: presets expand deterministic sweep
points, arms describe stock, protocol-option, or experimental installations,
and :mod:`runner` gates and interleaves them before :mod:`result` and
:mod:`table` report one result.
"""

from verity_vllm.bench.arms import Arm
from verity_vllm.bench.presets import Preset, SweepPoint, expand_sweep, get_preset
from verity_vllm.bench.runner import run_benchmark

__all__ = ["Arm", "Preset", "SweepPoint", "expand_sweep", "get_preset", "run_benchmark"]
