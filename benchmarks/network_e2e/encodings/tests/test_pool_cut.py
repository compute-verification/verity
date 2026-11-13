"""The carried pool cut at every window edge, through `pool_cut.drive`: at each edge of a cut run, vLLM d9105ea80's
``BlockPool`` (its exported row and root), the replay's ``_KV`` and CompileSlice_v6's window, opened on the last edge's
row and closed over the CacheStep_v3 calls the window made, give one row. The synthetic traces' small pool carries a
key's later copies across edges, which the recorded traces never do."""
from __future__ import annotations

import importlib.util
import random
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


PC = _load("enc_pool_cut", HERE.parent / "pool_cut.py")
CPV, RP = PC.CPV, PC.RP

KEYS = ("steps", "windows", "cut", "closed windows", "hit", "take", "alloc", "cache", "free", "max carried keys",
        "max later copies carried", "max M run", "max window slots", "K")


def synthetic(async_scheduling: bool, seed: int):
    """24 requests over three shared prefixes on a 12-block pool of 4-token blocks, admitted over 60 steps."""
    rng = random.Random(seed)
    shared = [tuple(rng.randrange(1, 50) for _ in range(rng.choice([4, 8, 12]))) for _ in range(3)]
    reqs = []
    for _ in range(24):
        p = shared[rng.randrange(3)] + tuple(rng.randrange(50, 99) for _ in range(rng.randrange(0, 9)))
        out = rng.randrange(1, 12)
        reqs.append(RP.RequestIO(prompt_len=len(p), max_tokens=out, output_len=out, eos=False,
                                 tokens=p + tuple(rng.randrange(1, 99) for _ in range(out))))
    e = RP.Engine(max_num_seqs=4, max_num_batched_tokens=32, max_model_len=48, num_gpu_blocks=12, block_size=4,
                  enable_prefix_caching=True, async_scheduling=async_scheduling)
    return CPV.Input(e, reqs, sorted(rng.randrange(0, 60) for _ in range(24)), [])


def _counts(got: dict) -> dict:
    return {k: got.get(k, 0) for k in KEYS}


@pytest.mark.parametrize("async_scheduling, want", [
    (False, {"steps": 65, "windows": 22, "cut": 38, "closed windows": 22, "hit": 84, "take": 54, "alloc": 82,
             "cache": 53, "free": 64, "max carried keys": 11, "max later copies carried": 1, "max M run": 18,
             "max window slots": 265, "K": 9}),
    (True, {"steps": 65, "windows": 22, "cut": 34, "closed windows": 22, "hit": 84, "take": 50, "alloc": 80,
            "cache": 51, "free": 59, "max carried keys": 11, "max later copies carried": 2, "max M run": 16,
            "max window slots": 255, "K": 9}),
])
def test_a_small_pool_carries_a_key_s_later_copies_across_its_edges(async_scheduling, want) -> None:
    """Cut every 3 steps, the pool fills and evicts inside windows, and keys cached twice by requests admitted together
    cross edges with their copies (``cprev``, ``cnext``, ``mlast``) as vLLM's hash map holds them."""
    got = PC.drive(synthetic(async_scheduling, 2), every=3)
    assert _counts(got) == want
    assert len(got["roots"]) == want["closed windows"]


@pytest.mark.parametrize("every, want", [
    (8, {"steps": 155, "windows": 20, "cut": 152, "closed windows": 20, "hit": 166, "take": 165, "alloc": 244,
         "cache": 92, "free": 166, "max carried keys": 157, "max later copies carried": 0, "max M run": 157,
         "max window slots": 2123, "K": 14}),
    (32, {"steps": 155, "windows": 5, "cut": 35, "closed windows": 5, "hit": 49, "take": 48, "alloc": 128,
          "cache": 92, "free": 49, "max carried keys": 157, "max later copies carried": 0, "max M run": 166,
          "max window slots": 2337, "K": 14}),
])
def test_the_prefix_cache_trace_cut_at_every_edge_agrees_at_each_one(every, want) -> None:
    """Each cut preempts the running requests and requeues them by arrival; each continuation looks its own blocks up
    again in the next window, and the three pools agree at every edge."""
    got = PC.drive(CPV.replay_input(HERE / "fixtures" / "async-prefix-cache.jsonl"), every=every)
    assert _counts(got) == want


@pytest.mark.slow
@pytest.mark.parametrize("name, want", [
    ("sync-crush.json.gz", {"steps": 1493, "windows": 24, "cut": 114, "closed windows": 24, "hit": 1469, "take": 149,
                            "alloc": 739, "cache": 613, "free": 218, "max carried keys": 158,
                            "max later copies carried": 0, "max M run": 422, "max window slots": 5729, "K": 57}),
    ("async-crush.json.gz", {"steps": 1507, "windows": 24, "cut": 112, "closed windows": 24, "hit": 1478,
                             "take": 148, "alloc": 732, "cache": 614, "free": 217, "max carried keys": 158,
                             "max later copies carried": 0, "max M run": 421, "max window slots": 5618, "K": 57}),
])
def test_the_crush_traces_cut_every_64_steps_agree_at_each_edge(name, want) -> None:
    """Through the crush traces' own preemptions between edges, sync and async; about 10 s each."""
    got = PC.drive(CPV.replay_input(HERE / "fixtures" / name), every=64)
    assert _counts(got) == want
