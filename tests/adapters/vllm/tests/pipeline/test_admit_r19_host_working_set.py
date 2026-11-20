"""The Commit host working set on the 188 GB L40S class: the two mechanisms, each pinned by a CPU test.

(c) fork-COW children: `driver._parallel_replay` freezes the parent's heap (gc.freeze) before forking the evaluator pool, so a
    child's cyclic collection never traverses (and copy-on-write duplicates) the inherited C2 heap (the parent's programs + oracle heap,
    74.8 GB modelled; unfrozen, each of two ForkPoolWorkers reaches ~72 GB and grows >= 17 GiB after the fork).  Tests:
    the children run with the frozen heap, the parent is unfrozen after the pool joins, the record names it, the replay record is
    byte-identical serial vs forked (semantics-free).
(a) the RETAIN-windowed pool: `--bounded-staging --retain host --retain-exclude hidden_m1` keeps every family the C2 forms read and drops
    the M1 export stream (84 % of the raw bytes at 1024/128).  Tests: the alias resolves to both tap families; the sampled replay never
    addresses a hidden-M1 name (a reader that refuses them yields the same record as the full reader); oracle_compare classifies the
    family out before any read (the same refusing reader, same verdict).
"""
from __future__ import annotations

import gc
import os
import signal

import numpy as np
import pytest

from verity_vllm.check.replay import driver as SR
from verity_vllm.commit.committer import native_collect as NC

HIDDEN_PREFIXES = ("fa2.m1/", "fa3.m1/")


# ------------------------------------------------------------------------------------------------------ (c) gc.freeze before the fork
@pytest.fixture(autouse=True)
def _heap_not_frozen_by_an_earlier_test():
    """vLLM's `EngineCore.__init__` calls `freeze_gc_heap()` and never unfreezes, so an in-process engine built earlier in the same worker
    leaves a permanent generation these tests would read as their own; it is frozen again afterwards."""
    frozen = gc.get_freeze_count()
    if frozen:
        gc.unfreeze()
    yield
    if frozen:
        gc.freeze()


def _child_probe(vu: dict) -> tuple[dict, dict, float]:
    """The per-VU closure of a forked child: reports the child's gc state (frozen count, pid) as its record."""
    return vu, {"result": True, "frozen": gc.get_freeze_count(), "pid": os.getpid(), "i": vu["i"]}, 0.0


def _killed_child(vu: dict) -> tuple[dict, dict, float]:
    if vu["i"] == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    return vu, {"result": True}, 0.0


# ------------------------------------------------------------------------------------------ the pool fitted after the prewarm
GiB = 1 << 30


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    mx, cur = tmp_path / "memory.max", tmp_path / "memory.current"
    monkeypatch.setattr(SR, "CGROUP_MEMORY", (str(mx), str(cur)))
    return mx, cur


def _pool_in_process(monkeypatch) -> list[int]:
    forked: list[int] = []

    def pool(fn, picked, workers):
        forked.append(workers)
        return [fn(vu) for vu in picked]

    monkeypatch.setattr(SR, "_parallel_replay", pool)
    return forked


class _Stating:
    """A row evaluator that only states each pick's peak: the driver's copy of `ROW_EVALUATORS` (the rows evaluate as on the record)."""

    def __init__(self, peak) -> None:
        self.peak_bytes = peak


# ------------------------------------------------------------------------------------------ (a) the RETAIN-windowed pool by family
def test_retain_exclude_alias_names_both_tap_families():
    assert NC.retain_exclude_families("hidden_m1") == {"fa2_hidden_m1", "fa3_hidden_m1"}
    assert NC.retain_exclude_families("hidden") == {"fa2_hidden_m1", "fa3_hidden_m1"}
    assert NC.retain_exclude_families("fa3_hidden_m1") == {"fa3_hidden_m1"}
    assert NC.retain_exclude_families("fa2_hidden_m1_stream, gemm_qkv") == {"fa2_hidden_m1", "gemm_qkv"}
    assert NC.retain_exclude_families("") == set() and NC.retain_exclude_families(None) == set()
    # the alias names exactly the FAMILY_RULES families of the two taps (value_family of a tap plane name)
    from verity_vllm.commit.committer.native_host import value_family
    assert {value_family("fa2.m1/L0/s0/g32x8x1/S"), value_family("fa3.m1/L3/s0/g8x8x1/P")} == NC.HIDDEN_M1_FAMILIES


def _refusing(committed, seen: list):
    """A committed reader with the hidden-M1 family NOT on host: a read of one of its names is what `--retain-exclude hidden_m1` would
    answer (None) -- and is recorded, so the test can assert the C2 forms never ask."""
    def read(step, name, lo, hi):
        if any(name.startswith(p) or p in name for p in HIDDEN_PREFIXES):
            seen.append((step, name))
            return None
        return committed(step, name, lo, hi)
    return read


def test_oracle_compare_classifies_the_hidden_family_out_before_any_read(tmp_path):
    from tests.check import test_oracle_compare as TOC
    from verity_vllm.check import oracle_compare as OC
    d, vals = TOC.make_match(str(tmp_path))
    bmap, committed = TOC.make_bmap(vals)
    seen: list = []
    full = OC.oracle_compare(OC.MatchOracle(d), bmap, TOC.reader(committed))
    excl = OC.oracle_compare(OC.MatchOracle(d), bmap, _refusing(TOC.reader(committed), seen))
    assert full["result"] is True and full["complete"] and full["not_compared_by_family"] == {"hidden_stream": 1, "weights": 1}
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "timing")}  # noqa: E731
    assert strip(full) == strip(excl)
    assert seen == []


def test_the_opened_reader_answers_none_for_an_excluded_interior_and_verified_bytes_for_the_kept_boundary():
    """The collector-side contract the exclusion rests on: `_HostRanges.holds` routes a read inside the excluded interior to None (opens by
    replay), a read of the kept ranges answers bytes -- opened from the kept leaves and verified against the run root (`OpenedReader`);
    `_excluded_chunk_ranges` excludes the chunk-aligned INTERIOR of the family only."""
    torch = pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.check.oracle_compare import OpenedReader
    from verity_vllm.commit.step_rows import ROW_BYTES as C
    # step stream: [qkv 300 B][fa2.m1 plane 4 chunks + 100 B][o_proj 200 B]
    spec = [("model.layers.0.self_attn.qkv_proj/0", 300, "bfloat16"), ("fa2.m1/L0/s0/g32x8x1/S", 4 * C + 100, "int32"), ("model.layers.0.self_attn.o_proj/0", 200, "bfloat16")]
    used = sum(nb for _name, nb, _dt in spec)
    stream = torch.arange(used, dtype=torch.int64).to(torch.uint8)             # distinguishable bytes
    tensors, off = [], 0
    for i, (name, nb, dt) in enumerate(spec):
        tensors.append((name, bytes(stream[off:off + nb].numpy()), i, dt, (nb // 2,)))
        off += nb
    com, run = commit_steps({0: tensors})                                      # the GPU-tree step block, retain=host
    metas = com._layouts[0]
    assert [m.dev_off for m in metas] == [0, 300, 300 + 4 * C + 100]
    excl = NC._excluded_chunk_ranges(metas, {"fa2_hidden_m1"}, C)
    assert len(excl) == 1 and excl[0][2] == "fa2_hidden_m1"
    lo, hi = excl[0][0], excl[0][1]
    assert lo == C * (-(-300 // C)) and hi == ((300 + 4 * C + 100) // C) * C and lo % C == 0 and hi % C == 0
    assert lo >= 300 and hi <= 300 + 4 * C + 100                              # interior only: the boundary chunks stay on host
    n = -(-used // C) * C
    blk = com._gpu_blocks[0][0]
    ranges = [(0, lo, blk[0:lo]), (hi, n, blk[hi:n])]                         # the collector keeps whole chunk-padded windows outside the interior
    hr = NC._HostRanges(ranges, used, excluded={"fa2_hidden_m1": hi - lo}, chunk=C)
    assert hr.retained_bytes == n - (hi - lo) and hr.numel() == n
    assert hr.holds(0, 300) and hr.holds(metas[2].dev_off, metas[2].dev_off + 200) and not hr.holds(lo, lo + C) and not hr.holds(300, 300 + C)
    assert bytes(hr[0:300].numpy()) == bytes(stream[0:300].numpy())
    with pytest.raises(KeyError):
        hr[lo:lo + C]
    com._gpu_blocks[0] = (hr, used)
    rd = OpenedReader(com, run)
    assert rd(0, "model.layers.0.self_attn.qkv_proj/0", 0, 300) == bytes(stream[0:300].numpy())
    assert rd(0, "model.layers.0.self_attn.o_proj/0", 0, 200) == bytes(stream[metas[2].dev_off:metas[2].dev_off + 200].numpy())
    assert rd(0, "fa2.m1/L0/s0/g32x8x1/S", 0, lo - 300) == bytes(stream[300:lo].numpy())   # the excluded tensor's kept boundary chunk
    assert rd(0, "fa2.m1/L0/s0/g32x8x1/S", C, 2 * C) is None                   # the excluded interior: 'not committed on host', opens by replay
    assert np.frombuffer(rd(0, "model.layers.0.self_attn.qkv_proj/0", 0, 300), dtype=np.uint8).size == 300
    assert rd.stats["reads"] == 4 and rd.stats["not_retained"] == 1 and rd.stats["failed"] == 0
