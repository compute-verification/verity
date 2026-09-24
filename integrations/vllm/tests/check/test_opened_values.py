"""D1: every value a verdict-bearing check compares is opened from the committer and verified against the run root (`oracle_compare.
OpenedReader`), never read from the committer's retained memory.

The attack: after the commit (before the openings and the compare) the committer's retained copy of a member is changed so that it agrees
with a wrong oracle.  A compare of the retained memory -- what `committed_reader` did -- PASSES; the opened compare FAILS, naming the member.
Committed on both layouts the pod uses: the GPU-tree step block (`commit_block_offline`: 256 B leaves spanning tensors; needs torch) and the
per-tensor host path (`commit_offline`, whose retained copy is the caller's buffer)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from tests.check import test_oracle_compare as TOC
from tests.check.opened import commit_host_steps, commit_steps, meta, retained_block
from verity_vllm.check import oracle_compare as OC
from verity_vllm.check import sampled_replay as SR

QKV = (0, "model.layers.0.self_attn.qkv_proj/0")
IDS = (0, "model.embed_tokens/in")


def _world(seed: int = 5) -> dict[int, list[tuple[str, bytes]]]:
    rng = np.random.default_rng(seed)
    sizes = {0: (1, 255, 256, 700, 3000), 1: (9, 513, 64)}
    return {s: [(f"t{s}.{i}", rng.integers(0, 256, n, dtype=np.uint8).tobytes()) for i, n in enumerate(ns)] for s, ns in sizes.items()}


def _commit(layout: str, steps: dict):
    if layout == "block":
        pytest.importorskip("torch")
        return commit_steps(steps)
    return commit_host_steps(steps)


def _steps_of(members: dict) -> dict[int, list[tuple]]:
    steps: dict[int, list[tuple]] = {}
    for (s, name), raw in members.items():
        steps.setdefault(s, []).append((name, raw))
    return steps


def _memory_reader(com):
    """The pre-fix read: the committer's retained copy at the member's offset -- no opening, no verification."""
    def read(step, name, lo, hi):
        m = OC.meta_by_name(com._layouts.get(step) or [], name)
        if m is None:
            return None
        hi = min(hi, m.nbytes)
        if step in com._gpu_blocks:
            return bytes(com._gpu_blocks[step][0][m.dev_off + lo:m.dev_off + hi].numpy())
        return bytes(m.host[lo:hi])
    return read


@pytest.mark.parametrize("layout", ["block", "host"])
def test_opened_reads_are_the_committed_bytes_and_each_step_root_is_authenticated_once(layout):
    steps = _world()
    rd = OC.OpenedReader(*_commit(layout, steps))
    reads = nbytes = 0
    for s, ts in steps.items():
        for name, raw in ts:
            n = len(raw)
            for lo, hi in ((0, n), (0, 1), (n - 1, n), (n // 3, 2 * n // 3 + 1)):
                assert rd(s, name, lo, hi) == raw[lo:hi], (layout, s, name, lo, hi)
                reads, nbytes = reads + 1, nbytes + hi - lo
    assert rd(0, "t0.9", 0, 4) is None and rd(7, "t0.0", 0, 1) is None           # not committed at that step: None, nothing opened
    rec = rd.record()
    assert rec["source"] == "opened" and rec["reads"] == reads and rec["bytes"] == nbytes and rec["leaves"] >= reads
    assert rec["failed"] == 0 and rec["not_retained"] == 0 and rec["steps_bound"] == 2


def test_a_meta_that_is_not_an_object_of_the_committers_layout_is_refused():
    com, run = commit_host_steps(_world())
    with pytest.raises(ValueError, match="not a meta of this committer's layout"):
        OC.OpenedReader(com, run).read_meta(0, copy.copy(meta(com, 0, "t0.3")))


def test_block_layout_a_byte_changed_after_the_commit_fails_every_read_of_its_leaf_by_name():
    pytest.importorskip("torch")
    steps = _world()
    com, run = commit_steps(steps)
    rd = OC.OpenedReader(com, run)
    m = meta(com, 0, "t0.4")                                  # 3000 B at block offset 1216: leaves 4 .. 16
    retained_block(com, 0)[m.dev_off + 1500] ^= 1              # stream byte 2716: leaf 10
    with pytest.raises(OC.OpeningNotVerified, match=r"step 0 t0\.4 \(ordinal 4\) bytes \[1400, 1600\) = leaves \[10, 11\)"):
        rd(0, "t0.4", 1400, 1600)
    assert rd(0, "t0.4", 0, 1000) == steps[0][4][1][:1000]     # the member's other leaves still open and verify ...
    assert rd(0, "t0.3", 0, 700) == steps[0][3][1]             # ... and so does its neighbour (leaf 4 is shared, unchanged)
    assert rd.or_none(0, "t0.4", 0, 3000) is None              # a whole-member read includes leaf 10
    rec = rd.record()
    assert rec["failed"] == 2 and rec["reads"] == 2
    assert rec["failures"][0] == {"step": 0, "name": "t0.4", "ordinal": 4, "byte_lo": 1400, "byte_hi": 1600, "leaves": [10, 11]}


def test_host_layout_a_retained_buffer_changed_after_the_commit_fails_its_reads_by_name():
    a = bytearray(b"\x11" * 200)                               # 64 B chunks: leaves 0 .. 3
    com, run = commit_host_steps({0: [("a/0", a), ("b/0", b"\x22" * 100)]})
    rd = OC.OpenedReader(com, run)
    a[150] ^= 0xFF                                             # the per-tensor path retains `a` itself: leaf 2
    with pytest.raises(OC.OpeningNotVerified, match=r"step 0 a/0 \(ordinal 0\) bytes \[128, 192\) = leaves \[2, 3\)"):
        rd(0, "a/0", 128, 192)
    assert rd(0, "a/0", 0, 128) == b"\x11" * 128 and rd(0, "b/0", 0, 100) == b"\x22" * 100


@pytest.mark.parametrize("layout", ["block", "host"])
def test_the_pod_fault_retained_flip_fails_every_opened_read_of_the_flipped_leaf_by_name(layout):
    """`VERITY_FAULT=retained_flip:<name>` (the real-row negative; `commit_delta` calls `com.fault_retained_flip` after the commit): one bit
    of the member's retained copy flipped.  Reads of the flipped leaf fail by name; t0.3's first leaves (not shared with t0.4) still verify."""
    steps = {s: [(n, bytearray(b)) for n, b in ts] for s, ts in _world().items()}      # writable: the per-tensor path retains these buffers
    com, run = _commit(layout, steps)
    f = com.fault_retained_flip("retained_flip:t0.4")
    assert (f["fault"], f["step"], f["name"], f["byte"], f["new"]) == ("retained_flip:t0.4", 0, "t0.4", 0, f["old"] ^ 1)
    rd = OC.OpenedReader(com, run)
    with pytest.raises(OC.OpeningNotVerified, match=rf"step 0 t0\.4 \(ordinal 4\) bytes \[0, 1\) = leaves \[{f['leaf']}, {f['leaf'] + 1}\)"):
        rd(0, "t0.4", 0, 1)
    assert rd(0, "t0.3", 0, 256) == bytes(steps[0][3][1][:256]) and rd(0, "t1.1", 0, 513) == bytes(steps[1][1][1])
    with pytest.raises(RuntimeError, match="no committed member name contains 'nope'"):
        com.fault_retained_flip("retained_flip:nope")


def test_the_oracle_compare_passes_on_opened_values_of_an_honest_commit(tmp_path):
    pytest.importorskip("torch")
    d, vals = TOC.make_match(str(tmp_path))
    bmap, committed = TOC.make_bmap(vals)
    rd = OC.OpenedReader(*commit_steps(_steps_of(committed)))
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, rd)
    assert res["result"] is True and res["mismatch_n"] == 0 and res["compared"] == res["equal"] == 2 * len(vals) + 2, res.get("why")
    rec = rd.record()
    assert rec["failed"] == 0 and rec["reads"] >= res["compared"] and rec["steps_bound"] == 1


@pytest.mark.parametrize("layout", ["block", "host"])
def test_d1_a_retained_copy_changed_after_the_commit_to_agree_with_the_oracle_fails_the_opened_compare_by_name(tmp_path, layout):
    """The committer committed qkv_proj bytes the Match oracle does not hold; after the commit its retained copy is overwritten with the
    oracle's bytes.  The memory compare says PASS; the opened compare says FAIL, every entry of the changed leaves 'opened value not
    verified' -- qkv_proj's two entries among them (on the per-tensor path, exactly those two)."""
    d, vals = TOC.make_match(str(tmp_path))
    bmap, committed = TOC.make_bmap(vals)
    honest = committed[QKV]
    members = dict(committed)
    members[QKV] = bytearray(b ^ 0x5A for b in honest)                         # what the committer committed
    com, run = _commit(layout, _steps_of(members))
    if layout == "block":
        import torch

        m = meta(com, 0, QKV[1])
        retained_block(com, 0)[m.dev_off:m.dev_off + m.nbytes] = torch.frombuffer(bytearray(honest), dtype=torch.uint8)
    else:
        members[QKV][:] = honest
    oracle = OC.MatchOracle(d)
    before = OC.oracle_compare(oracle, bmap, _memory_reader(com))
    assert before["result"] is True and before["mismatch_n"] == 0            # the attack passes a compare of the committer's memory
    rd = OC.OpenedReader(com, run)
    res = OC.oracle_compare(oracle, bmap, rd)
    assert res["result"] is False and res["mismatch_n"] >= 2
    assert all(x["why"].startswith("opened value not verified") for x in res["mismatch"])
    assert sum(x["name"] == QKV[1] for x in res["mismatch"]) == 2
    if layout == "host":
        assert res["mismatch_n"] == 2 and res["equal"] == res["compared"] - 2
    assert rd.record()["failed"] == res["mismatch_n"]


def test_boundary_linkage_names_an_input_id_changed_after_the_commit_unlinked(tmp_path):
    _d, vals = TOC.make_match(str(tmp_path))
    bmap, committed = TOC.make_bmap(vals)
    members = dict(committed)
    members[IDS] = bytearray(committed[IDS])
    com, run = commit_host_steps(_steps_of(members))
    workload = [{"request_id": "r1", "arrive_step": 0, "prompt_token_ids": [100, 101, 102], "steps": [0]},
                {"request_id": "r0", "arrive_step": 0, "prompt_token_ids": [103, 104, 105, 106, 107], "steps": [0]}]
    ok = SR.boundary_linkage(bmap, OC.OpenedReader(com, run), workload)
    assert ok["pairs"] == ok["linked"] == 2 and ok["unlinked"] == [], ok
    members[IDS][0:4] = np.int32(999).tobytes()                                  # r1's first prompt id changed in the retained copy
    res = SR.boundary_linkage(bmap, OC.OpenedReader(com, run), workload)
    assert res["result"] is False and res["linked"] == 0 and len(res["unlinked"]) == 2
    assert all(u["why"].startswith("opened value not verified") for u in res["unlinked"])


def test_a_replayed_vu_whose_committed_read_does_not_verify_is_a_mismatch_by_name(monkeypatch):
    def unverified(pi, store, weights, vu, arrive, rec):
        rec.update({"i": vu["i"], "request_id": vu["request_id"]})
        raise OC.OpeningNotVerified("step 3 x/0 (ordinal 7) bytes [0, 8) = leaves [0, 1): the opening does not verify against the run root")

    monkeypatch.setattr(SR, "_replay_vu", unverified)
    r = SR.replay_vu(None, None, lambda *a: None, {"i": 5, "request_id": "r0"}, arrive=0)
    assert r["result"] is False and r["i"] == 5 and r["why"].startswith("opened value not verified: step 3 x/0 (ordinal 7)")
