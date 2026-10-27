"""`run_config --engine-arg key=value` reaches every GPU stage command line and is recorded in run.json."""
from __future__ import annotations

from verity_vllm.pipeline import match as RC
from tests.pipeline.test_run_config_dry_run import dry


def test_engine_arg_reaches_every_gpu_stage():
    c = dry("--phase", "gpu", "--engine-arg", "max_num_batched_tokens=64", "--engine-arg", "enable_prefix_caching=true")
    for st in ("capture", "control", "census", "nonint"):
        assert c[st].count("--engine-arg=max_num_batched_tokens=64") == 1 and "--engine-arg=enable_prefix_caching=true" in c[st], st
    assert RC._engine_arg_dict(["max_num_batched_tokens=64", "enable_prefix_caching=true", "x=abc"]) == {"max_num_batched_tokens": 64, "enable_prefix_caching": True, "x": "abc"}
