"""A ``SLICES`` row whose slice of one element is the Definition its body calls once an element (`serving_templates.ELEMENTS`):
DSA's indexer query ``DsaIndexerQ_v1{H,D,ROPE}`` is H units of its head ``DsaIndexerQHead_v1{D,ROPE,NH=H}``, head h reading
``q[h]``, the RoPE row whole and ``w[h]`` and returning element h of each of the row's two fields (the head's e4m3 query and its
f32 weight).  The heads cover both fields with nothing left over, read their own elements and compute them; the row's own
Definition at H = 1 is no unit of it (a head's weight is scaled by its row's head count); a head's own row is its one unit; and in
a window the heads are chunks of the stager's bit rows, read once."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.definitions.registry import dsa as DSA

NH, D, ROPE, CAP = 4, 32, 16, 3
OP = "model.layers.3.self_attn.indexer"
FAMILY = "DsaIndexerQ_v1"


def indexer(h: int = NH):
    return bind(DSA.DsaIndexerQ, H=h, D=D, ROPE=ROPE)


def head():
    return bind(DSA.DsaIndexerQHead, D=D, ROPE=ROPE, NH=NH)


def leaves(a: np.ndarray, bits: int) -> np.ndarray:
    """A port's u16 words, (units, words), as its elements, (units, elements)."""
    return np.ascontiguousarray(a, dtype="<u2").view(f"<u{bits // 8}").astype(np.uint64)


def run(fn, *ops: np.ndarray) -> np.ndarray:
    """``fn`` on a batch, one array of rows per parameter: per row, its fields' elements in order."""
    return np.asarray(evaluate_batch(fn, *(np.asarray(o).astype(np.uint64) for o in ops))).reshape(len(ops[0]), -1)


@pytest.fixture
def row(monkeypatch):
    """The indexer's row of NH heads, or (``own``) one head's own row: its operands, its two committed fields and their store."""
    def make(own: bool = False, seed: int = 1):
        rng = np.random.default_rng(seed)
        q, cs, w = ((rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16) for n in (NH * D, ROPE, NH))
        ops, fn = ([q[:D], cs, w[:1]], head()) if own else ([q, cs, w], indexer())
        y = run(fn, *(o.reshape(1, -1) for o in ops)).reshape(-1)
        heads = 1 if own else NH
        out = {"index_q": y[:heads * D].astype(np.uint8), "index_w": y[heads * D:].astype(np.uint32)}
        r = {"i": 9, "spec": descriptor_id(fn), "args": [[["x", k, 0, o.size]] for k, o in enumerate(ops)]}
        statics = {"D": D, "ROPE": ROPE, "NH": NH} if own else {"H": NH, "D": D, "ROPE": ROPE}
        d = {"statics": statics, "op_path": OP, "members": [("index_q", heads * D, "u8"), ("index_w", heads, "f32")], "row": 0,
             "step": 16}

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return out[member][lo:hi], None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g, **_k: [ops[g[0][1]]])
        return ("r0", None, r, d), ops, out, Store()
    return make


def test_the_row_holds_its_heads_and_not_its_own_definition_at_one_head():
    spec, unit = descriptor_id(indexer()), descriptor_id(head())
    assert SRW.ELEMENTS[FAMILY] == ("DsaIndexerQHead_v1", {"NH": "H"})
    assert SRW.unit_templates(spec) == ((spec, 1), (unit, NH))
    assert descriptor_id(indexer(1)) not in dict(SRW.unit_templates(spec))
    t = SRW.template(unit)
    assert (t.layout, t.word, t.packed) == ("slice", unit, (16, 16, 16, 8, 32))
    assert SRW.template(spec).layout == "row" and SRW.unit_templates(unit) == ((unit, 1),)


def test_the_heads_cover_both_fields_read_their_own_elements_and_compute_them(row):
    x, (q, cs, w), out, store = row()
    T = descriptor_id(head())
    (m,) = SRW.members([T], [[x]], [NH])
    assert (m.n, m.rows, m.outs) == (NH, (("in0", D), ("in1", ROPE), ("in2", 1)), (("out.index_q", D // 2), ("out.index_w", 2)))
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, NH, "r0", 16, 3, OP, "index_q,index_w", 0, 9]]
    assert data["rows"]["in0"].tolist() == q.reshape(NH, D).tolist()
    assert data["rows"]["in1"].tolist() == [cs.tolist()] * NH
    assert data["rows"]["in2"].tolist() == w.reshape(NH, 1).tolist()
    got_q, got_w = leaves(data["outs"]["out.index_q"], 8), leaves(data["outs"]["out.index_w"], 32)
    assert got_q.reshape(-1).tolist() == out["index_q"].tolist() and got_w.reshape(-1).tolist() == out["index_w"].tolist()
    units = [leaves(data["rows"][f"in{g}"], 16) for g in range(3)]
    ref = run(head(), *units)
    assert ref[:, :D].tolist() == got_q.tolist() and ref[:, D:].tolist() == got_w.tolist()
    alone = run(indexer(1), *units)
    assert alone[:, :D].tolist() == got_q.tolist() and not (alone[:, D:] == got_w).any()


def test_a_head_s_own_row_is_its_one_unit(row):
    x, ops, out, store = row(own=True)
    T = descriptor_id(head())
    assert x[2]["spec"] == T
    (m,) = SRW.members([T], [[x]], [1])
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, 1, "r0", 16, 3, OP, "index_q,index_w", 0, 9]]
    assert [data["rows"][f"in{g}"].tolist() for g in range(3)] == [[o.tolist()] for o in ops]
    assert leaves(data["outs"]["out.index_q"], 8).reshape(-1).tolist() == out["index_q"].tolist()
    assert leaves(data["outs"]["out.index_w"], 32).reshape(-1).tolist() == out["index_w"].tolist()


def test_a_row_and_a_head_s_own_row_are_one_member_s_units(row, monkeypatch):
    x, ops, out, store = row()
    (_rid, pi, r1, d1), ops1, out1, store1 = row(own=True, seed=2)
    y = ("r1", pi, r1, d1)
    by = {"r0": (ops, store), "r1": (ops1, store1)}

    class Both:
        def words(self, rid, step, op, member, lo, hi):
            return by[rid][1].words(rid, step, op, member, lo, hi)

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, rid, _a, g, **_k: [by[rid][0][g[0][1]]])
    (m,) = SRW.members([descriptor_id(head())], [[x, y]], [NH + 1])
    data, entries = SRW.read(m, [x, y], Both(), None)
    assert [e[:3] for e in entries] == [[0, NH, "r0"], [NH, NH + 1, "r1"]]
    assert data["rows"]["in0"].tolist() == [*ops[0].reshape(NH, D).tolist(), ops1[0].tolist()]
    assert leaves(data["outs"]["out.index_w"], 32).reshape(-1).tolist() == [*out["index_w"].tolist(), *out1["index_w"].tolist()]


def test_a_row_of_another_count_of_outputs_is_refused(row):
    x, _ops, _out, store = row()
    (m,) = SRW.members([descriptor_id(head())], [[x]], [NH])
    x[3]["members"] = x[3]["members"][:1]
    with pytest.raises(ValueError, match=r"row 9 \(model.layers.3.self_attn.indexer\): 3 argument groups of 3 parameters and 1 outputs"):
        SRW.read(m, [x], store, None)


def test_a_window_s_heads_are_chunks_of_the_stager_s_bit_rows_read_once(row):
    x, _ops, _out, store = row()
    T = descriptor_id(head())
    tp = SRW.template(T)
    in_rows, out_rows = CR.unit_rows(tp)
    assert [(p, s) for p, s, _b, _ps in in_rows] == [(f"in{g}", V2) for g in range(3)] and out_rows == [(V2, (0,)), (V2, (1,))]
    ms = SRW.members([T], [[x]], [NH], {T: CAP})
    assert [(m.base, m.n) for m in ms] == [(0, CAP), (CAP, NH - CAP)]
    (whole,) = SRW.members([T], [[x]], [NH])
    data, _entries = SRW.read(whole, [x], store, None)
    got = SRW.read_all(ms, [[x]], store, None)
    for g in range(3):
        assert np.array_equal(np.concatenate([c[0]["rows"][f"in{g}"] for c in got]), data["rows"][f"in{g}"])
    for port, member in (("o0", "index_q"), ("o1", "index_w")):
        assert np.array_equal(np.concatenate([c[0]["outs"][port] for c in got]), data["outs"][f"out.{member}"])
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = tp.fn, "element-test", {}, in_rows
    for c, _e in got:
        st.reference_check(c["rows"], c["outs"])
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(got[0][0]["rows"], {**got[0][0]["outs"], "o1": got[0][0]["outs"]["o1"] ^ np.uint16(1)})
