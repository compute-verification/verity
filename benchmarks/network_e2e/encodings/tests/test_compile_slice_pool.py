"""CompileSlice_v2's pool calls (`verity.ml.protocols.certifier.compile_slice_v2.pool_step`) against the replay over
whole traces, through `compile_slice_pool.check_trace`: at every unit each hit, evictable count and return the calls
derive is `BlockPool`'s, the state's free count is the pool's hashless free count plus its queued copies, the tapped wires
are ReplayStep_v1's rows, and the calls the slice checks rebuild the prover's pool. CompileSlice_v4's calls
(`compile_slice_v4.pool_step`) on one CacheStep_v3 stream, through `check_trace_v4`: each hit, evictable count and
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

PREFIX = {"clock": 896, "compute": 1248, "copies_max": 1, "entries_max": 157, "evicting computes": 28, "free": 14,
          "hits": 26, "late": 1231, "lookup": 13, "shared frees": 13, "take": 14, "units": 157}
ASYNC_CRUSH = {"clock": 27136, "compute": 7447, "copies_max": 1, "entries_max": 158, "evicting computes": 547, "free": 85,
               "hits": 324, "late": 7265, "lookup": 1284, "preempt": 21, "shared frees": 19, "take": 106, "units": 1507}
SYNC_CRUSH = {"clock": 26880, "compute": 7361, "copies_max": 1, "entries_max": 159, "evicting computes": 546, "free": 85,
              "hits": 309, "lookup": 1274, "preempt": 20, "shared frees": 19, "take": 105, "units": 1494}


@pytest.mark.parametrize("name, want", [("async-prefix-cache.jsonl", PREFIX),
                                        pytest.param("async-crush.json.gz", ASYNC_CRUSH, marks=pytest.mark.slow),
                                        pytest.param("sync-crush.json.gz", SYNC_CRUSH, marks=pytest.mark.slow)])
def test_the_slice_s_pool_is_vllm_s(name, want) -> None:
    """Every unit of the trace fits (R 16, W 64, A 16) and its calls hold to `BlockPool` (`check_trace` raises otherwise),
    with evicting computes, frees of a block another slot holds, preemptions (the crush traces) and late caches (the
    async ones), in N = NB − 1 entries of one copy each."""
    assert CSP.check_trace(HERE / "fixtures" / name, R=16, W=64, A=16) == want


V4_PREFIX = {"calls": 239, "frees": 14, "hits": 26, "lookups": 27, "memos": 13, "slots": 3106, "units": 157}
V4_ASYNC_CRUSH = {"calls": 2781, "frees": 106, "hits": 324, "lookups": 1390, "memos": 1161, "slots": 39771, "units": 1507}
V4_SYNC_CRUSH = {"calls": 2765, "frees": 105, "hits": 309, "lookups": 1379, "memos": 1164, "slots": 40105, "units": 1494}


@pytest.mark.parametrize("name, want", [("async-prefix-cache.jsonl", V4_PREFIX), ("async-crush.json.gz", V4_ASYNC_CRUSH),
                                        ("sync-crush.json.gz", V4_SYNC_CRUSH)])
def test_the_stream_s_calls_are_vllm_s(name, want) -> None:
    """Every unit of the trace fits (R 16, W 64, A 16), and CompileSlice_v4's calls on one stream from the empty pool
    derive `BlockPool`'s hits, evictable counts and returns (`check_trace_v4` raises otherwise): CompileSlice_v2's hits, most
    lookups of the crush traces the head-of-line memo's."""
    assert CSP.check_trace_v4(HERE / "fixtures" / name, R=16, W=64, A=16) == want
