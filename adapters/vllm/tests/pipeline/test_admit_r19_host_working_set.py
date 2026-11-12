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

from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.check.replay import driver as SR
from verity_vllm.check.replay import fork as FK

# the sampled-replay world / helpers of test_sampled_replay (one fixture set, not a copy)
from tests.check import test_sampled_replay as TSR

HIDDEN_PREFIXES = ("fa2.m1/", "fa3.m1/")


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    inst, woff = TSR._program()
    n_w = max(hi for lo, hi in woff.values())
    weights = TSR._bf16(rng, n_w, 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80
    return TSR.Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


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


def test_forked_children_inherit_a_frozen_heap_and_the_parent_unfreezes_after_the_pool_joins(monkeypatch):
    monkeypatch.setattr(SR, "FORK_GC_FREEZE", True)
    picked = [{"i": i, "family": "x", "stratum": "s", "spec": "Gemm_v1", "request_id": "r0", "phase": "mid"} for i in range(6)]
    assert gc.get_freeze_count() == 0
    out = SR._parallel_replay(_child_probe, picked, workers=2)
    assert gc.get_freeze_count() == 0                                        # unfrozen in the parent (finally)
    assert [vu["i"] for vu, _r, _t in out] == list(range(6))                  # picked order, whatever the schedule
    assert all(r["pid"] != os.getpid() for _v, r, _t in out)                  # evaluated in children
    assert all(r["frozen"] > 0 for _v, r, _t in out), out                     # every child saw the permanent generation
    st = SR._FORK_STATS
    assert st["gc_freeze"] is True and st["workers"] == 2 and st["picked"] == 6 and st["gc_frozen_objects"] > 0 and st["gc_collected"] >= 0
    # the children's frozen set is the parent's at fork time (a few permanent-generation objects die in the child's fork bookkeeping)
    assert min(r["frozen"] for _v, r, _t in out) >= 0.99 * st["gc_frozen_objects"]


def test_fork_gc_freeze_opt_out_is_named_on_the_record(monkeypatch):
    monkeypatch.setattr(SR, "FORK_GC_FREEZE", False)
    picked = [{"i": i, "family": "x", "stratum": "s", "spec": "Gemm_v1", "request_id": "r0", "phase": "mid"} for i in range(3)]
    out = SR._parallel_replay(_child_probe, picked, workers=2)
    assert all(r["frozen"] == 0 for _v, r, _t in out)
    assert SR._FORK_STATS == {"gc_freeze": False, "workers": 2, "picked": 3}
    assert gc.get_freeze_count() == 0


def test_parent_unfreezes_even_when_the_pool_fails(monkeypatch):
    monkeypatch.setattr(SR, "FORK_GC_FREEZE", True)

    def boom(vu):
        raise RuntimeError("child fault")

    with pytest.raises(RuntimeError):
        SR._parallel_replay(boom, [{"i": 0}], workers=2)
    assert gc.get_freeze_count() == 0


def _killed_child(vu: dict) -> tuple[dict, dict, float]:
    if vu["i"] == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    return vu, {"result": True}, 0.0


@pytest.mark.parametrize("ooms, why", [((0, 0), "exited mid-task"), ((3, 4), "OOM-KILLED (cgroup memory.events oom_kill 3 -> 4; 2 workers")])
def test_a_worker_killed_mid_task_is_reported_by_name_not_waited_out(monkeypatch, capsys, ooms, why):
    """A SIGKILLed worker (the cgroup's OOM killer) loses its task: the pool says so, OOM-killed when memory.events counted it, at once
    rather than after its timeout."""
    counts = iter([ooms[0], *[ooms[1]] * 10**6])
    monkeypatch.setattr(FK, "oom_kills", lambda: next(counts))
    monkeypatch.setattr(FK, "POOL_POLL_S", 0.01)
    picked = [{"i": i} for i in range(4)]
    with pytest.raises(SR.WorkerDied, match=why.replace("(", r"\(")):
        SR._parallel_replay(_killed_child, picked, workers=2)
    assert why in capsys.readouterr().err
    assert gc.get_freeze_count() == 0


@pytest.mark.slow
def test_replay_record_is_identical_serial_vs_forked_and_names_the_frozen_fork(world):
    bmap, committed = world.bmap_and_reader()
    kw = dict(per_stratum=2, seed=11, retention="host")
    ser = SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), workers=1, **kw)
    par = SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), workers=3, **kw)
    assert ser["result"] is True and par["result"] is True
    strip = lambda r: {k: v for k, v in r.items() if k not in ("timing", "evaluator", "wall_s")}  # noqa: E731
    assert strip(ser) == strip(par)
    assert ser["evaluator"]["fork"] == {} and par["evaluator"]["mode"] == "forked pool"
    fk = par["evaluator"]["fork"]
    assert fk["gc_freeze"] is True and fk["workers"] == 3 and fk["gc_frozen_objects"] > 0 and fk["picked"] == par["sample"]["evaluated"]


# ------------------------------------------------------------------------------------------ the pool fitted after the prewarm
GiB = 1 << 30


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    mx, cur = tmp_path / "memory.max", tmp_path / "memory.current"
    monkeypatch.setattr(SR, "CGROUP_MEMORY", (str(mx), str(cur)))
    return mx, cur


def _prewarm_leaves(monkeypatch, cur, in_use: int) -> None:
    """The prewarm (in the Boolean replay, every picked Boolean version compiled in the parent) leaves ``in_use`` bytes in the cgroup."""
    real = SR._prewarm_evaluators

    def prewarm(picked):
        out = real(picked)
        cur.write_text(f"{in_use}\n")
        return out

    monkeypatch.setattr(SR, "_prewarm_evaluators", prewarm)


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


def test_the_pool_is_fitted_after_the_prewarm_to_what_the_parent_then_holds(world, cgroup, monkeypatch):
    """30 workers asked for under a 400 GiB memory.max, 10 GiB in use when the replay begins and 301.4 GiB once the prewarm has
    compiled the Boolean versions: workers sized before the prewarm (all 30 at 9.6 GiB) would be OOM-killed."""
    mx, cur = cgroup
    mx.write_text(f"{400 * GiB}\n")
    cur.write_text(f"{10 * GiB}\n")
    _prewarm_leaves(monkeypatch, cur, int(301.4 * GiB))
    forked = _pool_in_process(monkeypatch)
    bmap, committed = world.bmap_and_reader()
    sr = SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), workers=30, worker_peak=int(9.6 * GiB),
                           per_stratum=2, seed=11, retention="host")
    assert forked == [10] and sr["result"] is True and sr["evaluator"]["workers"] == 10
    m = sr["evaluator"]["memory"]
    assert (m["in_use"], m["room"], m["workers_requested"], m["workers_fitted"]) == (int(301.4 * GiB), 400 * GiB - int(301.4 * GiB), 30, 10)
    assert sr["evaluator"]["note"].startswith("30 requested, fitted to 10 after the prewarm: 10 workers at the largest picks' peaks (96.0 GiB")
    assert SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), workers=30, per_stratum=2, seed=11,
                             retention="host")["evaluator"]["workers"] == 30 and forked[-1] == 30   # no worker_peak: not fitted


def test_a_pick_that_cannot_fit_is_refused_before_the_fork_never_replayed_serially(world, cgroup, monkeypatch):
    """An LM-head worker peaks at 182.1 GiB; after the prewarm 98.6 GiB are left, for any worker or the parent."""
    mx, cur = cgroup
    mx.write_text(f"{400 * GiB}\n")
    cur.write_text(f"{10 * GiB}\n")
    _prewarm_leaves(monkeypatch, cur, int(301.4 * GiB))
    forked = _pool_in_process(monkeypatch)
    monkeypatch.setattr(SR, "ROW_EVALUATORS", [_Stating(lambda vu: int(182.1 * GiB))])
    bmap, committed = world.bmap_and_reader()
    with pytest.raises(SR.PoolDoesNotFit, match=r"needs 182\.3 GiB, more than the cgroup's memory\.max 400\.0 GiB less the 301\.4 GiB in "
                                                 r"use after the prewarm \(98\.6 GiB\): the job needs at least 483\.7 GiB"):
        SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), workers=30, worker_peak=250 << 20, per_stratum=2,
                          seed=11, retention="host")
    assert forked == []


def test_the_workers_hold_the_largest_picks_side_by_side(cgroup, monkeypatch):
    mx, cur = cgroup
    stated = {0: 182, 1: 17, 2: 12, 3: 12, 4: 12}
    picked = [{"i": i, "spec": f"Gemm_v3{{N={i}}}"} for i in (3, 1, 4, 0, 2)]
    monkeypatch.setattr(SR, "ROW_EVALUATORS", [_Stating(lambda vu: stated[vu["i"]] * GiB)])
    assert SR.fit_pool(8, picked, GiB) == (8, None)                     # no cgroup
    mx.write_text("max\n")
    cur.write_text(f"{6 * GiB}\n")
    assert SR.fit_pool(8, picked, GiB) == (8, None)                     # no limit
    mx.write_text(f"{400 * GiB}\n")
    cur.write_text(f"{166 * GiB}\n")
    n, m = SR.fit_pool(8, picked, GiB)                                  # 183 + 18 + 13 + 13 of 234 GiB; a fifth's 13 does not fit
    assert (n, m["fitted_bytes"], m["largest"], m["peaks_stated_by"]) == (4, 227 * GiB, {"spec": "Gemm_v3{N=0}", "bytes": 183 * GiB}, "_Stating")
    assert SR.fit_pool(3, picked, GiB)[0] == 3
    monkeypatch.setattr(SR, "ROW_EVALUATORS", [])
    assert SR.fit_pool(300, picked, GiB)[0] == 234                      # no stated peaks: the baseline alone, idle workers included


def test_a_tp_rank_forks_its_share_of_the_default_pool(monkeypatch):
    """The world's ranks replay at once on one node (`rank_replay.in_process`), each forking the default pool over the world."""
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert (SR.default_workers(), SR.default_workers(8), SR.default_workers(128)) == (32, 8, 1)


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


def test_sampled_replay_never_reads_the_hidden_m1_family(world):
    """The population has no VU row for the M1 export stream (`fa2_hidden_m1_stream: attention interior, no VU row of its own`): with the
    family absent from host the replay record is byte-identical.  Attention rows are in this world (their q/k/v/out are module outputs)."""
    bmap, committed = world.bmap_and_reader()
    # a hidden-M1 entry bound beside the module outputs at every step (the collector's tap plane), never a Program member
    for step in sorted({int(e["identity"]["step"]) for e in bmap["entries"] if "step" in e.get("identity", {})}):
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": "*", "step": step, "invocation": 0, "op_path": "fa2.m1/L0/g32x8x1",
                                             "output_member": "S", "element_range": [0, 8], "rank": 0}, "kind": "leaf", "name": "fa2.m1/L0/s0/g32x8x1/S",
                                "element_offset": 0, "encoding": {"dtype": "int32", "shape": [8]}})
    seen: list = []
    full = SR.sampled_replay(TSR._programs(world), bmap, committed, TSR._weights(world), per_stratum=2, seed=7, workers=1, retention="host")
    excl = SR.sampled_replay(TSR._programs(world), bmap, _refusing(committed, seen), TSR._weights(world), per_stratum=2, seed=7, workers=1, retention="host")
    assert full["result"] is True and full["sample"]["evaluated"] > 0
    strip = lambda r: {k: v for k, v in r.items() if k not in ("timing", "evaluator", "wall_s")}  # noqa: E731
    assert strip(full) == strip(excl)
    assert seen == []                                                          # not one read of the family


def test_oracle_compare_classifies_the_hidden_family_out_before_any_read(tmp_path):
    from verity_vllm.check import oracle_compare as OC
    from tests.check import test_oracle_compare as TOC
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
