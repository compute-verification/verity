"""`verity_vllm.pipeline.tp.capture.parse_engine_args`, the name the TP commands read `--engine-arg` by, is the engine's parser:
a value is JSON where it parses and a string where it does not, and an item without a key or without `=` is refused."""
from __future__ import annotations

import re

import pytest


def test_the_tp_commands_read_engine_args_as_the_engine_does():
    from verity_vllm.pipeline.tp.capture import parse_engine_args

    given = ["max_num_seqs=8", "kv_cache_dtype=fp8", "disable_custom_all_reduce=true", "stop=[1,2", "a=b=c"]
    assert parse_engine_args(given) == {"max_num_seqs": 8, "kv_cache_dtype": "fp8", "disable_custom_all_reduce": True, "stop": "[1,2", "a": "b=c"}
    assert parse_engine_args([]) == {}
    for bad in ("max_num_seqs", "=8"):
        with pytest.raises(ValueError, match=re.escape(f"--engine-arg expects key=value, got {bad!r}")):
            parse_engine_args([bad])
