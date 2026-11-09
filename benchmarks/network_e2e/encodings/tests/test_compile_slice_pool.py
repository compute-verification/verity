"""CompileSlice_v4's pool calls (`verity.ml.protocols.certifier.compile_slice_v4.pool_step`) against the replay over
whole traces, on one CacheStep_v3 stream, through `compile_slice_pool.check_trace_v4`: each hit, evictable count and
return they derive is `BlockPool`'s."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CSP = _load("enc_compile_slice_pool", HERE.parent / "compile_slice_pool.py")

V4_PREFIX = {"calls": 239, "frees": 14, "hits": 26, "lookups": 27, "memos": 13, "slots": 3106, "units": 157}
V4_ASYNC_CRUSH = {"calls": 2781, "frees": 106, "hits": 324, "lookups": 1390, "memos": 1161, "slots": 39771, "units": 1507}
V4_SYNC_CRUSH = {"calls": 2765, "frees": 105, "hits": 309, "lookups": 1379, "memos": 1164, "slots": 40105, "units": 1494}


@pytest.mark.parametrize("name, want", [("async-prefix-cache.jsonl", V4_PREFIX), ("async-crush.json.gz", V4_ASYNC_CRUSH),
                                        ("sync-crush.json.gz", V4_SYNC_CRUSH)])
def test_the_stream_s_calls_are_vllm_s(name, want) -> None:
    """Every unit of the trace fits (R 16, W 64, A 16), and CompileSlice_v4's calls on one stream from the empty pool
    derive `BlockPool`'s hits, evictable counts and returns (`check_trace_v4` raises otherwise), most lookups of the
    crush traces the head-of-line memo's."""
    assert CSP.check_trace_v4(HERE / "fixtures" / name, R=16, W=64, A=16) == want
