"""CompileSlice_v2's pool calls (`verity.ml.protocols.certifier.compile_slice_v2.pool_step`) against the replay over
whole traces, through `compile_slice_pool.check_trace`: at every unit each hit, evictable count and return the calls
derive is `_KV`'s, the state's free count is the pool's hashless free count plus its queued copies, the tapped wires
are ReplayStep_v1's rows, and the calls the slice checks rebuild the prover's pool."""
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
    """Every unit of the trace fits (R 16, W 64, A 16) and its calls hold to `_KV` (`check_trace` raises otherwise),
    with evicting computes, frees of a block another slot holds, preemptions (the crush traces) and late caches (the
    async ones), in N = NB − 1 entries of one copy each."""
    assert CSP.check_trace(HERE / "fixtures" / name, R=16, W=64, A=16) == want
