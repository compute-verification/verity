"""The shared overhead benchmark that uses the vLLM integration.

The public pieces are deliberately small: presets expand deterministic sweep
points, arms describe stock, protocol-option, or experimental installations,
and :mod:`runner` gates and interleaves them before :mod:`result` and
:mod:`table` report one result.
"""

from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import Preset, SweepPoint, expand_sweep, load_spec
from verity_bench.vllm.runner import run_benchmark

__all__ = ["Arm", "Preset", "SweepPoint", "expand_sweep", "load_spec", "run_benchmark"]
