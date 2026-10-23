"""The fused all-reduce + residual add + RMSNorm rows' parts in chunks (`pipeline.serving_rows`' ``part`` layout, `part_rows`): a
window of the residual part at the row's width and the norm part puts each in chunks of the stager's bit rows, so each part's
Definition on a chunk's rows is its committed output, and wires the norm's ``in0`` (the row's r) to the ``o0`` row of the
residual unit on the same row, committed under that row's salt, so the verifier's check on the public bodies holds; it refuses
a wire to another unit, an r row that is not its writer's output, a norm without the residual part, and a residual part in
column blocks."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity_vllm.commit import serving_rows as CS
from verity_vllm.commit import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import part_rows as PR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.program.registry import boolean_sigma as SG
from verity_vllm.program.registry import tp_allreduce as TA

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

WORLD, N, ROWS = 2, 256, 3
STATICS = {"WORLD": WORLD, "N": N, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}
FUSED = bind(TA.AllReduceRmsNormMnnvl, **STATICS)
RESID = descriptor_id(bind(SG.AllReduceResidualMnnvl, WORLD=WORLD, N=N, SHOT=1))
NORM = descriptor_id(bind(SG.RmsNormMnnvl, N=N, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False))
CAPS = {RESID: 2, NORM: 2, descriptor_id(bind(SG.AllReduceResidualMnnvl, WORLD=WORLD, N=N // 2, SHOT=1)): 2}


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "part-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


@pytest.fixture
def rows(monkeypatch):
    """Three fused rows (layers 0, 1, 2), their partials, residual and weight read off ``_slices``, and the store of their two
    committed outputs."""
    rng = np.random.default_rng(5)

    def bf16s(n, s):
        return ((rng.standard_normal(n) * s).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    ops, vals, out = {}, {"normed": [], "residual": []}, []
    for p in range(ROWS):
        ops.update({(p, 0): bf16s(WORLD * N, 0.5), (p, 1): bf16s(N, 2.0), (p, 2): bf16s(N, 0.1)})
        both = np.asarray(evaluate_batch(FUSED, *(ops[(p, g)].astype(np.uint64).reshape(1, -1) for g in range(3)))).reshape(-1)
        vals["normed"].append(both[:N].astype(np.uint16))
        vals["residual"].append(both[N:].astype(np.uint16))
        r = {"i": 4 + p, "spec": descriptor_id(FUSED), "args": [[["x", 10 * p + g, 0, ops[(p, g)].size]] for g in range(3)]}
        d = {"statics": dict(STATICS), "op_path": f"model.layers.{p}.post_attention_layernorm",
             "members": [("normed", N, "bf16"), ("residual", N, "bf16")], "row": p, "step": 16}
        out.append(("r0", None, r, d))
    vals = {k: np.concatenate(v) for k, v in vals.items()}

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return vals[member][lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [ops[divmod(g[0][1], 10)]])
    return out, vals, Store()


def window(rows, tmpls=(RESID, NORM)):
    tmpls = list(tmpls)
    ms = PR.wire(SRW.members(tmpls, [rows] * len(tmpls), None, CAPS), tmpls, [rows] * len(tmpls))
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Window(*CS.population_program(defs, several=True), {"law": "subset", "k": 2}, "ab" * 32, tuple(ms))


def test_the_parts_are_chunks_that_compute_the_row_s_outputs(rows):
    rs, vals, store = rows
    w = window(rs)
    assert [(SRW.template(m.template).layout, m.base, m.n, m.outs) for m in w.members] == [
        ("cols", 0, 2, (("o0", N),)), ("cols", 2, 1, (("o0", N),)), ("part", 3, 2, (("o0", N),)), ("part", 5, 1, (("o0", N),))]
    got = SRW.read_all(w.members, [rs] * 2, store, None)
    for m, (g, _entries) in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    for t, member in ((RESID, "residual"), (NORM, "normed")):
        o = np.concatenate([g["outs"]["o0"] for m, (g, _e) in zip(w.members, got) if m.template == t]).reshape(-1)
        assert o.tolist() == vals[member].tolist()
    assert [e[6] for _g, es in got for e in es] == ["residual"] * 3 + ["normed"] * 3
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        stager(NORM).reference_check(got[2][0]["rows"], {"o0": got[2][0]["outs"]["o0"] ^ np.uint16(1)})


def test_the_norm_s_r_is_wired_to_its_row_s_residual_unit_and_the_wires_hold_on_the_public_bodies(rows):
    rs, _vals, store = rows
    w = window(rs)
    assert [m.wired for m in w.members] == [None, None, ("in0", "o0", (0, 1)), ("in0", "o0", (2,))]
    got = SRW.read_all(w.members, [rs] * 2, store, None)
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    p, o, ws = w.members[2].wired
    other = replace(w, members=tuple(replace(m, wired=(p, o, ws[::-1])) if j == 2 else m for j, m in enumerate(w.members)))
    assert list(WI.check(other.members, publics)) == [w.members[2].key]
    bad = [dict(g, rows={**g["rows"], "in0": g["rows"]["in0"] ^ np.uint16(1)}) if j == 2 else g for j, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=f"{w.members[2].key}: a in0 row is not the o0 row"):
        CS.commit(w, bad, salt_key=bytes(32))


def test_a_norm_without_its_residual_part_at_the_row_s_width_is_refused(rows):
    rs, *_ = rows
    with pytest.raises(ValueError, match="reads its AllReduceRmsNormMnnvl_v1 rows' committed output 1, which the partition's parts"):
        window(rs, [NORM])
    half = descriptor_id(bind(SG.AllReduceResidualMnnvl, WORLD=WORLD, N=N // 2, SHOT=1))
    with pytest.raises(ValueError, match=r"returns in column blocks: wire it to AllReduceResidualMnnvl_v1 at the rows' N"):
        window(rs, [half, NORM])
    assert all(m.wired is None for m in window(rs, [RESID]).members)
