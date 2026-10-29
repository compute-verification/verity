"""A w13 coordinate cut along K (`pipeline.serving_rows`' ``kchunk`` layout, `kchunk_rows`): a ``MoeExpertFp8Gemm_v1{K}`` row holds
the first, middle and last ``MoeExpertFp8CoordinateChunk`` templates at every chunk width, and a window of the three:

* reads each unit's operands off the row and its ``acc`` off the twin's chain, so the chunk Definition on each chunk's rows, parsed
  as C-Flock's stager parses them, is its committed output, and the last chunk's is the row's served word;
* wires every ``acc`` row to the output row of the chunk before it (chunk c - 1 of the same row and coordinate), commits it under
  that row's salt, so the verifier's check on the public bodies holds, and refuses a wire to another unit, an ``acc`` row that is
  not its writer's output, and a partition that leaves out a writer's template.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.ml.windows import rows as CS
from verity.ml.windows import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import kchunk_rows as KC
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.numerics import fp8_moe_rows as FR
from verity.ml.definitions.numerics.derived_rows import f2fp_bf16
from verity.ml.definitions.registry import boolean_fp8_moe as BFM
from verity.ml.definitions.registry import fp8_moe_experts as X

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32

E, K, N, G, KB, KC_ = 3, 128, 5, 32, 32, 32
NB, KG, C = -(-N // G), K // G, K // KC_
GEMM = bind(X.MoeExpertFp8Gemm, E=E, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=HopperE4m3MmaV2Dot32)


def chunk(first: bool, last: bool, k: int = KC_, boolean: bool = True) -> str:
    st = {"E": E, "K": k, "G": G, "KB": KB, "SWAP": False, "FIRST": first, "LAST": last}
    if boolean:
        return descriptor_id(bind(BFM.MoeExpertFp8CoordinateChunkV2, **st, DOT=BG.HopperE4m3MmaV2Dot32))
    return descriptor_id(bind(X.MoeExpertFp8CoordinateChunk, **st, DOT=HopperE4m3MmaV2Dot32))


FIRST, MID, LAST = chunk(True, False), chunk(False, False), chunk(False, True)
TMPLS = [FIRST, MID, LAST]
CAPS = {FIRST: 4, MID: 7, LAST: 3}


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "kchunk-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(f"<u{a.dtype.itemsize}") if a.dtype.itemsize > 1 else a


@pytest.fixture
def pairs(monkeypatch):
    """Two pairs of one layer (experts 2 and 0, one weight stack), each a ``MoeExpertFp8Gemm_v1{K=128}`` row, and the store of their
    served outputs."""
    rng = np.random.default_rng(11)

    def codes(n):
        c = rng.integers(0, 256, n, dtype=np.uint8)
        return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)

    def scales(n):
        return ((2.0 ** rng.integers(-8, 2, n)) * rng.uniform(1, 2, n)).astype(np.float32)

    w, ws = codes(E * N * K), scales(E * NB * KG)
    ops = {}
    for p, e in enumerate((2, 0)):
        ops.update({(p, 0): codes(K), (p, 1): scales(KG), (p, 2): np.array([e], np.int32)})
    ops[(9, 3)], ops[(9, 4)] = w, ws
    ys = [np.asarray(evaluate_batch(GEMM, *[_bits(ops[(p, g)]).astype(np.uint64).reshape(1, -1) for g in range(3)],
                                    _bits(w).astype(np.uint64).reshape(1, -1), _bits(ws).astype(np.uint64).reshape(1, -1))
                     ).reshape(-1).astype(np.uint16) for p in range(2)]
    rows = []
    for p in range(2):
        r = {"i": 7 + p, "spec": descriptor_id(GEMM),
             "args": [[["x", 10 * p + g, 0, ops[(p, g)].size]] for g in range(3)] + [[["x", 93, 0, w.size]], [["x", 94, 0, ws.size]]]}
        d = {"statics": {"E": E, "K": K, "N": N, "G": G, "KB": KB}, "op_path": "moe/L3", "members": [("gate_up", N, "bf16")],
             "row": p, "step": 16}
        rows.append(("r0", None, r, d))

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return np.concatenate(ys)[lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [ops[divmod(g[0][1], 10)]])
    return rows, ops, np.concatenate(ys), Store()


def window(rows, tmpls=TMPLS):
    ms = KC.wire(SRW.members(tmpls, [rows] * len(tmpls), None, CAPS), tmpls, [rows] * len(tmpls))
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*CS.population_program(defs, several=True), {"law": "subset", "k": 2}, "ab" * 32, tuple(ms))


def test_a_w13_row_holds_its_first_middle_and_last_chunks_at_each_width():
    spec = descriptor_id(GEMM)
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(bind(X.MoeExpertFp8Gemm, E=E, K=K, N=1, G=G, KB=KB, SWAP=False,
                                                                       DOT=HopperE4m3MmaV2Dot32)), N),
                                        (chunk(True, False, 64, False), N), (chunk(False, True, 64, False), N),
                                        (chunk(True, False, 32, False), N), (chunk(False, False, 32, False), N * (C - 2)),
                                        (chunk(False, True, 32, False), N))
    assert [(SRW.template(t).layout, SRW.template(t).word) for t in TMPLS] == [
        ("kchunk", chunk(f, la, boolean=False)) for f, la in ((True, False), (False, False), (False, True))]
    with pytest.raises(ValueError, match="no chunk cap"):
        SRW.members(TMPLS, [[("r0", None, {"i": 0}, {"statics": {"K": K, "N": N}})]] * 3, None, {})


def test_the_chunks_compute_their_committed_outputs_and_the_last_the_served_words(pairs):
    rows, ops, y, store = pairs
    w = window(rows)
    assert [sum(m.n for m in w.members if m.template == t) for t in TMPLS] == [2 * N, 2 * N * (C - 2), 2 * N]
    got = SRW.read_all(w.members, [rows] * 3, store, None)
    for m, (g, _entries) in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    last = np.concatenate([g["outs"]["o0"] for m, (g, _e) in zip(w.members, got) if m.template == LAST]).reshape(-1)
    assert last.tolist() == y.tolist()
    k = len(w.members) // 2
    assert w.members[k].template == MID
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        stager(MID).reference_check(got[k][0]["rows"], {"o0": got[k][0]["outs"]["o0"] ^ np.uint16(1)})
    W, WS = ops[(9, 3)].reshape(E, N, K), ops[(9, 4)].view("<u4").reshape(E, NB, KG)
    for p in range(2):
        e = int(ops[(p, 2)][0])
        accs = FR.moe_expert_fp8_block_accs(ops[(p, 0)], ops[(p, 1)].view("<u4"), W[e], WS[e], every=KC_, G=G, KB=KB, dot="mmav2")
        twin, nonfinite = FR.moe_expert_fp8_block_row(ops[(p, 0)], ops[(p, 1)].view("<u4"), W[e], WS[e], G=G, KB=KB, dot="mmav2")
        assert not nonfinite.any() and f2fp_bf16(accs[-1]).tolist() == twin.tolist() == y[p * N:(p + 1) * N].tolist()


def test_each_acc_is_wired_to_the_chunk_before_it_and_the_wires_hold_on_the_public_bodies(pairs):
    rows, _ops, _y, store = pairs
    w = window(rows)
    unit = {}
    for t, cs in ((FIRST, [0]), (MID, range(1, C - 1)), (LAST, [C - 1])):
        b = min(m.base for m in w.members if m.template == t)
        for j, (p, n, c) in enumerate((p, n, c) for p in range(2) for n in range(N) for c in cs):
            unit[(p, n, c)] = b + j
    at = {u: key for key, u in unit.items()}
    for m in w.members:
        if m.template == FIRST:
            assert m.wired == ()
            continue
        (port, out, writers), = m.wired
        assert (port, out) == ("in3", "o0") and len(writers) == m.n
        for i, wr in enumerate(writers):
            p, n, c = at[m.base + i]
            assert wr == unit[(p, n, c - 1)]
    got = SRW.read_all(w.members, [rows] * 3, store, None)
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    k = next(k for k, m in enumerate(w.members) if m.wired)
    (p, o, ws), = w.members[k].wired
    other = replace(w, members=tuple(replace(m, wired=((p, o, ws[1:] + ws[:1]),)) if j == k else m for j, m in enumerate(w.members)))
    assert list(WI.check(other.members, publics)) == [w.members[k].key]
    bad = [dict(g, rows={**g["rows"], p: g["rows"][p] ^ np.uint16(1)}) if j == k else g for j, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=f"{w.members[k].key}: a in3 row is not the o0 row"):
        CS.commit(w, bad, salt_key=bytes(32))


def test_a_partition_without_a_writer_s_template_is_refused(pairs):
    rows, *_ = pairs
    for tmpls, missing in (([MID, LAST], "first"), ([FIRST, LAST], "mid")):
        with pytest.raises(ValueError, match=rf"\['{missing}'\] ones, which the partition does not name"):
            window(rows, tmpls)
    assert all(m.wired == () for m in window(rows, [FIRST]).members)
