"""CacheStep_v2's specification (`verity.ml.protocols.certifier.cache_step_v2.reference`) against vLLM's block pool as
the replay models it (`replay._KV`) over whole async traces, through `cache_pool_vectors.drive`: the unit's pool after
every pool operation, the refused witnesses beside each call, the N = NB − 1 bound, and the certifier's difftest
vectors as `cache_pool_vectors` makes them now."""
from __future__ import annotations

import importlib.util
import json
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


V = _load("enc_cache_pool_vectors", HERE.parent / "cache_pool_vectors.py")


@pytest.mark.parametrize("name", ["async-prefix-cache.jsonl", pytest.param("async-crush.json.gz", marks=pytest.mark.slow)])
def test_the_pool_is_vllm_s_over_a_whole_trace(name) -> None:
    """Every operation the replay's pool makes is the unit's call, accepted, and leaves the unit's pool equal to `_KV`'s
    (the same hashes cached with their blocks held by the same slots, the same free-queue order, the same hashless
    count), each eviction removing the blocks vLLM evicts. Beside each call, a made-up eviction, a wrong threshold,
    returns a block either side and a hit or an evictable count a step either side are refused, and so is the hit a
    cache that never evicts would give, wherever eviction shortened vLLM's. One copy an entry is enough."""
    inp = V.replay_input(V.FIXTURES / name)
    c = V.drive(inp, V.statics_for(inp, D=4)).counts
    assert c["steps"] == len(inp.rows) and c["checks"] > 1000
    assert c["copies_max"] == 1
    assert c["evicting computes"] >= 20 and c["evicted"] >= c["evicting computes"]
    assert c["refused the threshold past every queued copy"] == c["evicting computes"]
    assert c["refused the threshold one copy newer"] >= 20 and c["refused a made-up eviction"] >= 20
    assert c["refused the returns plus one"] == c["free"] >= 10 and c["refused the returns less one"] >= 10
    assert c["refused a hit one unit longer"] == c["take"] + c["lookup"]
    assert c["refused the evictable count plus one"] == c["take"] + c["lookup"]
    if name == "async-crush.json.gz":
        assert c["refused the threshold one copy older"] >= 100 and c["refused the evictable count less one"] >= 100
        assert c["refused the hit of a cache that never evicts"] == c["hits shortened by eviction"] >= 500


def test_n_nb_minus_one_entries_hold_every_pool() -> None:
    """vLLM evicts a hash's last block only after its cached descendants, so the unit never orphans an entry and
    N = NB − 1 entries hold every pool: at N = NB − 1 the unit accepts the whole trace, its entries come within a few
    of N, and with one entry fewer than the most the trace holds it refuses one of vLLM's own operations."""
    inp = V.replay_input(V.FIXTURES / "async-prefix-cache.jsonl")
    st = V.statics_for(inp)
    assert st["N"] == inp.engine.num_gpu_blocks - 1
    c = V.drive(inp, st).counts
    assert c["entries dropped"] >= 20 and c["dropped_with_descendant"] == 0
    assert st["N"] - 4 <= c["entries_max"] <= st["N"]
    with pytest.raises(AssertionError, match="refuses vLLM's own operation"):
        V.drive(inp, {**st, "N": c["entries_max"] - 1})


@pytest.mark.slow
def test_the_vectors_are_current() -> None:
    """The certifier's vectors (`cache_step_v2_vectors.json`) are what `cache_pool_vectors` makes from these fixtures
    now."""
    want = json.loads(V.VECTORS.read_text())
    assert want["kind"] == "cache_step_v2_vectors" and want["slices"].keys() == V.SLICES.keys()
    for name, sl in want["slices"].items():
        assert V.slice_of(name, **V.SLICES[name]) == sl, name
