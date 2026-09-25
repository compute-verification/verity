"""[R19 admit] F-r18-admit-1 -- the Commit host working set on the 188 GB L40S class: the two mechanisms, each pinned by a CPU test.

(c) fork-COW children: `sampled_replay._parallel_replay` freezes the parent's heap (gc.freeze) before forking the evaluator pool, so a
    child's cyclic collection never traverses (and copy-on-write duplicates) the inherited C2 heap -- the "two ForkPoolWorkers at ~72 GB"
    of R17 #23 (= the parent's programs + oracle heap, 74.8 GB modelled), the >= 17 GiB post-fork growth that killed #60 twice.  Tests:
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

import numpy as np
import pytest

from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.check.replay import sampled_replay as SR

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
    C = NC.GPU_CHUNK_BYTES
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
    hr = NC._HostRanges(ranges, used, excluded={"fa2_hidden_m1": hi - lo})
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
