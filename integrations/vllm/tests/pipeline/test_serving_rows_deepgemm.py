"""DeepGEMM's FP8 linear split into its parts (`pipeline.serving_rows`' ``PARTS``, one part in ``COORDS``' coordinates): a
``DeepGemmFp8Linear_v1{K,N}`` row holds one ``Fp8GroupQuantUe8m0_v1{K,G=128}``, its quantisation of x read whole (x -> q, s; the
``part`` layout), and N coordinates ``ScaledMmFp8BlockSharedScale_v1{K,N=1,G=128}`` of its GEMM, unit n reading the row's
committed q and s whole, ``w[n]`` and ``ws[n // 128]`` (the ``coord`` layout; N = 200 spans a whole weight block and a partial
one).  The parts cover the row's three outputs once (the quantisation its q and s under one index entry, the coordinates its y,
N units under one entry) and compute them: each part's Definition on its units' committed rows is their outputs.  The GEMM's own
rows split into the same coordinates.  A row is read by one member per part, never by two members of one part, nor by the row
and a part, and a committed q the GEMM's port does not take is refused."""
from __future__ import annotations

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch

from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.numerics import deepgemm_twins as DT
from verity.ml.definitions.registry import boolean_deepgemm as BDG
from verity.ml.definitions.registry import boolean_fp8 as BF8
from verity.ml.definitions.registry import deepgemm as DG
from verity.ml.definitions.registry import fp8 as F8
from verity.ml.definitions.registry.fp8 import FP8_BLOCK

K, N = 256, 200
NB, KG = -(-N // FP8_BLOCK), K // FP8_BLOCK
OP = "model.layers.5.mlp.shared_experts.gate_up_proj"


def linear():
    return bind(DG.DeepGemmFp8Linear, K=K, N=N)


def quant(boolean: bool = False):
    return bind(BDG.Fp8GroupQuantUe8m0V2 if boolean else DG.Fp8GroupQuantUe8m0, K=K, G=FP8_BLOCK)


def gemm(n: int, boolean: bool = False):
    return bind(BF8.ScaledMmFp8BlockSharedScaleV2 if boolean else F8.ScaledMmFp8BlockSharedScale, K=K, N=n, G=FP8_BLOCK)


def leaves(data: dict[str, np.ndarray], ports, packed) -> list[np.ndarray]:
    """A packed member's u16 rows as its ports' elements, one row per unit."""
    return [np.ascontiguousarray(data[p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}"))
            .reshape(len(data[p]), -1).astype(np.uint64) for (p, _w), b in zip(ports, packed)]


def ev(fn, args: list[np.ndarray]) -> list[np.ndarray]:
    out = evaluate_batch(fn, *args)
    return [np.asarray(a).reshape(len(args[0]), -1) for a in (out if isinstance(out, (tuple, list)) else [out])]


@pytest.fixture
def row(monkeypatch):
    """A synthetic row (its x a committed member, its w and ws registered weights) and its outputs by the linear's Definition,
    behind a ProgramIndex stand-in that holds it as row 7; with ``own``, a row of the GEMM's own Definition on those q and s."""
    def make(own: bool = False, q_dtype=np.uint8):
        rng = np.random.default_rng(3)
        x = ((rng.standard_normal(K) * 3).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        w = rng.integers(0, 256, N * K, dtype=np.uint8)
        w[(w & 0x7F) == 0x7F] ^= 1
        ws = DT.upcast_e8m0(rng.integers(112, 124, NB * KG))
        got = evaluate(linear(), *(a.tolist() for a in (x, w, ws)))
        q, s, y = (np.asarray(got[a:b], dtype=t) for a, b, t in ((0, K, q_dtype), (K, K + KG, np.uint32), (K + KG, K + KG + N, np.uint16)))
        assert DT.deepgemm_linear_row(x, w.reshape(N, K), ws.reshape(NB, KG))[2].tolist() == y.tolist()
        vals = {"x": x, "w": w, "ws": ws, "x_q": q, "x_s": s, "out": y}
        if own:
            spec, names, members, statics = descriptor_id(gemm(N)), ["x_q", "x_s", "w", "ws"], [("out", N, "bf16")], {"G": FP8_BLOCK}
        else:
            spec, names, statics = descriptor_id(linear()), ["x", "w", "ws"], {}
            members = [("x_q", K, "e4m3"), ("x_s", KG, "f32"), ("out", N, "bf16")]
        r = {"i": 7, "spec": spec, "args": [[["x", k, 0, vals[nm].size]] for k, nm in enumerate(names)]}
        d = {"statics": {"K": K, "N": N, **statics}, "op_path": OP, "members": members, "row": 0, "step": 16}

        class Index:
            by_i = {7: r}

            def describe(self, _r):
                return d

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return vals[member][lo:hi], None

        monkeypatch.setattr(SRW, "index_of", lambda inst, _reuse, **kw: Index(), raising=False)
        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [vals[names[group[0][1]]][:group[0][3]]])
        return {"store": Store(), **vals}
    return make


def test_a_row_holds_its_quantisation_and_n_coordinates_of_its_gemm():
    spec = descriptor_id(linear())
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(quant()), 1), (descriptor_id(gemm(N)), 1), (descriptor_id(gemm(1)), N))
    assert SRW.unit_templates(descriptor_id(gemm(N))) == ((descriptor_id(gemm(N)), 1), (descriptor_id(gemm(1)), N))
    t = SRW.template(descriptor_id(quant(True)))
    assert (t.layout, t.word, t.packed) == ("part", descriptor_id(quant()), (16, 8, 32))
    t = SRW.template(descriptor_id(gemm(1, True)))
    assert (t.layout, t.word, t.packed) == ("coord", descriptor_id(gemm(1)), (8, 32, 8, 32, 16))
    assert SRW.part(descriptor_id(quant()), spec) == SRW.PARTS["DeepGemmFp8Linear_v1"][0]
    assert SRW.part(descriptor_id(gemm(1)), spec) == SRW.PARTS["DeepGemmFp8Linear_v1"][1]
    assert SRW.part(descriptor_id(gemm(1)), descriptor_id(gemm(N))) is None


def test_the_parts_cover_the_row_s_outputs_and_compute_them(row):
    x = row()
    tq, tg = SRW.template(descriptor_id(quant(True))), SRW.template(descriptor_id(gemm(1, True)))
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [tq.id, tg.id])
    assert counts == [1, N] and [[(rid, r["i"]) for rid, _pi, r, _d in rs] for rs in rows] == [[("r0", 7)]] * 2
    mq, mg = SRW.members([tq.id, tg.id], rows, counts)
    assert (mq.n, mq.base, mq.rows, mq.outs) == (1, 0, (("in0", K),), (("out.x_q", K // 2), ("out.x_s", 2 * KG)))
    assert (mg.n, mg.base, mg.rows, mg.outs) == (N, 1, (("in0", K // 2), ("in1", 2 * KG), ("in2", K // 2), ("in3", 2 * KG)),
                                                 (("out.out", 1),))
    dq, eq = SRW.read(mq, rows[0], x["store"], None)
    dg, eg = SRW.read(mg, rows[1], x["store"], None)
    assert eq == [[0, 1, "r0", 16, 5, OP, "x_q,x_s", 0, 7]] and eg == [[1, 1 + N, "r0", 16, 5, OP, "out", 0, 7]]
    assert dq["rows"]["in0"].tobytes() == x["x"].tobytes()
    assert dq["outs"]["out.x_q"].tobytes() == x["x_q"].tobytes() and dq["outs"]["out.x_s"].tobytes() == x["x_s"].astype("<u4").tobytes()
    W, WS = x["w"].reshape(N, K), x["ws"].reshape(NB, KG).astype("<u4")
    for n in range(N):
        assert dg["rows"]["in0"][n].tobytes() == x["x_q"].tobytes()
        assert dg["rows"]["in1"][n].tobytes() == x["x_s"].astype("<u4").tobytes()
        assert dg["rows"]["in2"][n].tobytes() == W[n].tobytes()
        assert dg["rows"]["in3"][n].tobytes() == WS[n // FP8_BLOCK].tobytes()
    assert dg["outs"]["out.out"].reshape(-1).tolist() == x["out"].tolist()
    (qs,) = ev(quant(), leaves(dq["rows"], mq.rows, tq.packed))
    want = leaves(dq["outs"], mq.outs, tq.packed[1:])
    assert qs[:, :K].tolist() == want[0].tolist() and qs[:, K:].tolist() == want[1].tolist()
    (y,) = ev(gemm(1), leaves(dg["rows"], mg.rows, tg.packed))
    assert y.reshape(-1).tolist() == x["out"].tolist()


def test_a_window_s_caps_make_both_parts_chunks_of_the_stager_s_bit_rows(row):
    """With the partition file's ``chunk_cap`` naming them, the quantisation (``part``) and the GEMM's coordinates (``coord``) are
    each chunks on `UnitStager`'s bit rows, which a5 composes from the template alone; a part that is no chunk needs an input set
    there.  The chunks read the word-row members' words, and each chunk's outputs are its Definition's."""
    x = row()
    tq, tg = SRW.template(descriptor_id(quant(True))), SRW.template(descriptor_id(gemm(1, True)))
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [tq.id, tg.id])
    whole = SRW.members([tq.id, tg.id], rows, counts)
    caps = {tq.id: 1, tg.id: 64}
    assert SRW.chunked(tq.id, caps) and SRW.chunked(tg.id, caps) and not SRW.chunked(tq.id, {tg.id: 64})
    ms = SRW.members([tq.id, tg.id], rows, counts, caps)
    assert [(m.template, m.base, m.n) for m in ms] == [(tq.id, 0, 1)] + [(tg.id, 1 + lo, min(64, N - lo)) for lo in range(0, N, 64)]
    assert all(m.name and m.schemas for m in ms)
    got = SRW.read_all(ms, rows, x["store"], None)
    for w, tp, rs, chunks in ((whole[0], tq, rows[0], got[:1]), (whole[1], tg, rows[1], got[1:])):
        data, _entries = SRW.read(w, rs, x["store"], None)
        for g in range(len(w.rows)):
            assert np.array_equal(np.concatenate([c["rows"][f"in{g}"] for c, _e in chunks]), data["rows"][f"in{g}"])
        for j, (p, _w) in enumerate(w.outs):
            assert np.array_equal(np.concatenate([c["outs"][f"o{j}"] for c, _e in chunks]), data["outs"][p])
        st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
        st.definition, st.name, st.public_inputs, st.in_rows = tp.fn, "deepgemm-part-chunk-test", {}, CR.unit_rows(tp)[0]
        for c, _e in chunks:
            st.reference_check(c["rows"], c["outs"])


def test_the_gemm_s_own_rows_are_its_coordinates(row):
    x = row(own=True)
    tg = SRW.template(descriptor_id(gemm(1, True)))
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [tg.id])
    (m,) = SRW.members([tg.id], rows, counts)
    assert counts == [N] and m.outs == (("out.out", 1),)
    data, entries = SRW.read(m, rows[0], x["store"], None)
    assert entries == [[0, N, "r0", 16, 5, OP, "out", 0, 7]]
    (y,) = ev(gemm(1), leaves(data["rows"], m.rows, tg.packed))
    assert y.reshape(-1).tolist() == data["outs"]["out.out"].reshape(-1).tolist() == x["out"].tolist()


@pytest.mark.parametrize("tmpls", [lambda: [gemm(N), gemm(1)], lambda: [linear(), quant()]], ids=["two-of-one-part", "the-row-and-a-part"])
def test_a_row_read_by_two_members_that_are_not_its_parts_is_refused(row, tmpls):
    row()
    with pytest.raises(ValueError, match=r"row 7 .* one member reads a row, or one member per part of it \(PARTS\)"):
        SRW.member_rows({"r0": {}}, [descriptor_id(t) for t in tmpls()])


def test_a_committed_q_the_gemm_s_port_does_not_take_is_refused(row):
    x = row(q_dtype=np.uint16)
    T = descriptor_id(gemm(1, True))
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [T])
    (m,) = SRW.members([T], rows, counts)
    with pytest.raises(ValueError, match="row 7 output x_q: uint16 elements, the template's port is 8-bit elements"):
        SRW.read(m, rows[0], x["store"], None)
