"""`verity_vllm.pipeline.tp.capture.parse_engine_args`, the name the TP commands read `--engine-arg` by, is the engine's parser:
a value is JSON where it parses and a string where it does not, and an item without a key or without `=` is refused.  And
`verity_vllm.pipeline.tp.capture.rank_line`, what the capture's log says of each rank's probe, a rank that could not read its TP
group or its device included."""
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


@pytest.mark.parametrize("probe, line", [
    ({"pid": 7, "tp": {"rank": 1, "world_size": 2}, "device": {"current": 1, "name": "gpu"}}, (7, 1, 1)),
    ({"pid": 7, "tp_error": "RuntimeError('no TP group')", "device": {"current": 0}}, (7, None, 0)),
    ({"pid": 7, "tp": {"rank": 0}, "device_error": "RuntimeError('no CUDA')"}, (7, 0, None)),
    ({"pid": 7, "tp": None, "device": None}, (7, None, None)), ({"pid": 7, "tp": {}, "device": {}}, (7, None, None)), ({}, (None, None, None))])
def test_the_captures_line_for_a_rank_is_its_pid_its_tp_rank_and_its_device_and_none_for_what_its_probe_could_not_read(probe, line):
    from verity_vllm.pipeline.tp.capture import rank_line

    assert rank_line(probe) == line
