"""CacheStep_v2's specification (`verity.protocols.certifier.prover.cache_step_v2.reference`) against vLLM's block pool as
the advice decoder and the replay model it (`verity_vllm.pipeline.advice_code.pool.BlockPool`), through
`cache_pool_vectors.drive`: over whole traces, sync and async, and over a small pool built to reach every edge (a full
pool, evictions of blocks one free queued together and of part of one free's blocks, a preempted request's returns and
its resume's hit on them, hits after eviction began and hits stopped a token short), the unit's pool after every pool
operation, the refused witnesses beside each call, the N = NB − 1 bound, and the certifier's difftest vectors as
`cache_pool_vectors` makes them now. Both sides are this suite's inputs, so a change to either reruns it."""
from __future__ import annotations

import gzip
import importlib.util
import json
import random
import shutil
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
    """Every operation the replay's pool makes is the unit's call, accepted, and leaves the unit's pool equal to `BlockPool`'s
    (the same hashes cached with their blocks held by the same slots, the same free-queue order, the same hashless
    count), each eviction removing the blocks vLLM evicts. Beside each call, a made-up eviction, a wrong threshold,
    returns a block either side and a hit or an evictable count a step either side are refused, and so is the hit a
    cache that never evicts would give, wherever eviction shortened vLLM's. One copy an entry is enough. The crush
    trace's preempted requests resume on blocks their preemption returned, taking them back from the free queue."""
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
        assert c["resumes taking the blocks their preemption returned"] >= 10
        assert c["returned blocks a resume took from the free queue"] >= 100


EDGES = ("allocations a full pool refuses", "preempted frees keeping shared blocks",
         "returned blocks a resume took from the free queue",
         "evictions of blocks one free queued together", "evictions splitting one free's blocks",
         "hits past the first eviction", "hits shortened by eviction",
         "hits stopped a token short with the next block cached", "hits on a hash two blocks hold")


def _edges(asy: bool) -> V.Input:
    """24 requests on a 12-block pool (4 tokens a block, 4 running at most, 32 tokens a step), each prompt one of three
    shared prefixes and a private tail of up to 8 tokens (none for some, which repeats a prompt), each arriving with
    probability 0.4 at each step the scheduler could take it, and replayed at those admission steps."""
    rng = random.Random(0)
    shared = [tuple(rng.randrange(1, 50) for _ in range(rng.choice([4, 8, 12]))) for _ in range(3)]
    reqs = []
    for _ in range(24):
        p = shared[rng.randrange(3)] + tuple(rng.randrange(50, 99) for _ in range(rng.randrange(0, 9)))
        out = rng.randrange(1, 12)
        reqs.append(V.RP.RequestIO(prompt_len=len(p), max_tokens=out, output_len=out, eos=False,
                                   tokens=p + tuple(rng.randrange(1, 99) for _ in range(out))))
    e = V.RP.Engine(max_num_seqs=4, max_num_batched_tokens=32, max_model_len=48, num_gpu_blocks=12, block_size=4,
                    enable_prefix_caching=True, async_scheduling=asy)
    admit = V.RP.replay(e, reqs, decide=lambda i, k, _: rng.random() < 0.4).admitted
    return V.Input(e, reqs, list(admit), [list(s.rows) for s in V.RP.replay(e, reqs, admit=admit).steps])


@pytest.mark.parametrize("asy", [False, True], ids=["sync", "async"])
def test_the_pool_is_vllm_s_at_its_edges(asy) -> None:
    """On a pool far too small for its work, every edge where the unit and `BlockPool` could part is reached, and
    every call there is accepted, with its neighbours refused, and leaves the two pools equal: an allocation the full
    pool refuses (and the preemption it forces), a preempted request returning only the blocks no other request holds,
    and on its resume hitting and taking back blocks its preemption returned to the free queue (under a new slot),
    an eviction taking blocks one free queued together and one splitting a free's blocks at the threshold, a hit after
    eviction began and one eviction shortened, a hit the num_tokens − 1 cap stops with its next block cached (a repeated
    prompt, whose last block is computed again), and a hit on a hash two blocks hold, which takes the older."""
    inp = _edges(asy)
    c = V.drive(inp, V.statics_for(inp, D=4)).counts
    assert c["steps"] == len(inp.rows) and c["preempted frees"] >= 2
    assert all(c[k] >= 1 for k in EDGES), {k: c[k] for k in EDGES}
    assert c["refused the hit of a cache that never evicts"] == c["hits shortened by eviction"]


@pytest.mark.parametrize("arrivals", ["admit", "first"])
def test_the_pool_is_vllm_s_on_a_served_sync_trace(tmp_path, arrivals) -> None:
    """The trace-corpus fixture (served sync, 100 KV blocks of 16 tokens, prefix caching, preemptions), each step's rows
    in Model Runner V2's batch order (sorted stably by n) as served traces keep them: the unit holds `BlockPool` over
    every operation at R = its 256 sequences, through hundreds of evicting computes and preempted requests resuming on
    blocks their preemption returned, with requests arriving at their recorded steps and as the advice encoder replays
    them."""
    src = V.FIXTURES / "corpus-smallkv"
    for f in ("meta.json", "server.log"):
        shutil.copy(src / f, tmp_path / f)
    (tmp_path / "steps.jsonl").write_bytes(gzip.decompress((src / "steps.jsonl.gz").read_bytes()))
    reqs = [json.loads(x) for x in gzip.decompress((src / "requests.jsonl.gz").read_bytes()).decode().splitlines()]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps({**q, "eos": q["finish_reason"] == "stop"}) + "\n"
                                                     for q in reqs))
    inp = V.replay_input(tmp_path)
    inp = V.Input(inp.engine, inp.requests, inp.before, [sorted(s, key=lambda x: x[2]) for s in inp.rows])
    assert not inp.engine.async_scheduling and inp.engine.num_gpu_blocks == 100
    c = V.drive(inp, V.statics_for(inp, R=inp.engine.max_num_seqs, D=4), arrivals=arrivals).counts
    assert c["steps"] == len(inp.rows) and c["preempted frees"] >= 2
    assert c["resumes taking the blocks their preemption returned"] >= 2
    assert c["returned blocks a resume took from the free queue"] >= 50
    assert c["evicting computes"] >= 200 and c["allocations a full pool refuses"] >= 100
    assert c["hits past the first eviction"] >= 100 and c["evictions splitting one free's blocks"] >= 100
    assert c["refused the hit of a cache that never evicts"] == c["hits shortened by eviction"] >= 100


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
