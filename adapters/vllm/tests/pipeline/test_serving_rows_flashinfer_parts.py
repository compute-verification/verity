"""FlashInfer's FP8 linear (decode's M < 32 branch) split into its parts (`pipeline.serving_templates`' ``PARTS``), as DeepGEMM's
linear is (`test_serving_rows_deepgemm`): a ``FlashInferFp8Linear_v1{K,N}`` row holds one ``FlashInferFp8Quant1x128_v1{K}``, its
quantisation of x once (x -> q, s; the ``part`` layout), and N coordinates ``ScaledMmFp8BlockSharedScale_v1{K,N=1,G=128}`` of its
GEMM, unit n reading those q and s whole, ``w[n]`` and ``ws[n // 128]`` (the ``coord`` layout; N = 200 spans a whole weight block
and a partial one).  Serving stores only x and y, so q and s are intermediates: the readers compute them by their twin
(`part_rows.MIDS`), the coordinates' reader holds the twin's y on them to the served y, and the window wires each coordinate's q
and s rows to its row's quantisation's output rows (`part_rows.wire`), committed under that unit's salts, so the verifier's
check on the public bodies holds.  The quantisation then the coordinates compute the linear's output, word for word, and a
DeepGEMM row's coordinates in the same template read its committed q and s, unwired, in members of their own."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate, evaluate_batch
from verity.definitions.numerics import deepgemm_twins as DT
from verity.definitions.registry import boolean_deepgemm as BDG
from verity.definitions.registry import boolean_fp8 as BF8
from verity.definitions.registry import deepgemm as DG
from verity.definitions.registry import fp8 as F8
from verity.definitions.registry.fp8 import FP8_BLOCK
from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity.commitments.windows import wiring as WI
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import part_rows as PR
from verity_vllm.pipeline import serving_rows as SRW

K, N = 256, 200
NB, KG = -(-N // FP8_BLOCK), K // FP8_BLOCK
#: a window's rows' N: its population program grows with its units
WN = 40


def linear(n: int = N):
    return bind(DG.FlashInferFp8Linear, K=K, N=n)


def quant(boolean: bool = False):
    return bind(BDG.FlashInferFp8Quant1x128V2 if boolean else DG.FlashInferFp8Quant1x128, K=K)


def dg_quant(boolean: bool = False):
    return bind(BDG.Fp8GroupQuantUe8m0V2 if boolean else DG.Fp8GroupQuantUe8m0, K=K, G=FP8_BLOCK)


def gemm(n: int, boolean: bool = False):
    return bind(BF8.ScaledMmFp8BlockSharedScaleV2 if boolean else F8.ScaledMmFp8BlockSharedScale, K=K, N=n, G=FP8_BLOCK)


TQ, TDQ, TG = descriptor_id(quant(True)), descriptor_id(dg_quant(True)), descriptor_id(gemm(1, True))


def op(i: int) -> str:
    return f"model.layers.{i}.self_attn.o_proj"


def leaves(data: dict[str, np.ndarray], ports, packed) -> list[np.ndarray]:
    """A packed member's u16 rows as its ports' elements, one row per unit."""
    return [np.ascontiguousarray(data[p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}"))
            .reshape(len(data[p]), -1).astype(np.uint64) for (p, _w), b in zip(ports, packed)]


def ev(fn, args: list[np.ndarray]) -> list[np.ndarray]:
    out = evaluate_batch(fn, *args)
    return [np.asarray(a).reshape(len(args[0]), -1) for a in (out if isinstance(out, (tuple, list)) else [out])]


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "flashinfer-part-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


@pytest.fixture
def served(monkeypatch):
    """Rows 7, 8, .. of one request, each a ``FlashInferFp8Linear_v1{K,N=n}`` row (``"fi"``: x, w and ws its argument groups, y its
    one committed member) or a ``DeepGemmFp8Linear_v1{K,N=n}`` row (``"dg"``: its committed q and s too), its outputs by its
    Definition, behind a ProgramIndex stand-in; with ``bad``, row 7's served y has a word flipped."""
    def make(kinds=("fi",), bad: bool = False, n: int = N):
        args, words, by_i, ds = {}, {}, {}, {}
        nb = -(-n // FP8_BLOCK)
        for i, kind in enumerate(kinds, start=7):
            rng = np.random.default_rng(i)
            x = ((rng.standard_normal(K) * 3).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            w = rng.integers(0, 256, n * K, dtype=np.uint8)
            w[(w & 0x7F) == 0x7F] ^= 1
            ws = DT.upcast_e8m0(rng.integers(112, 124, nb * KG))
            fn = linear(n) if kind == "fi" else bind(DG.DeepGemmFp8Linear, K=K, N=n)
            got = evaluate(fn, *(a.tolist() for a in (x, w, ws)))
            if kind == "fi":
                y = np.asarray(got, dtype=np.uint16)
                twin, nonfinite = DT.flashinfer_linear_row(x, w.reshape(n, K), ws.reshape(nb, KG))
                assert not nonfinite.any() and y.tolist() == np.asarray(twin).reshape(-1).tolist()
                members, words[op(i)] = [("out", n, "bf16")], {"out": y}
            else:
                q, s, y = (np.asarray(got[a:b], dtype=t) for a, b, t in
                           ((0, K, np.uint8), (K, K + KG, np.uint32), (K + KG, K + KG + n, np.uint16)))
                members = [("x_q", K, "e4m3"), ("x_s", KG, "f32"), ("out", n, "bf16")]
                words[op(i)] = {"x_q": q, "x_s": s, "out": y}
            if bad and i == 7:
                words[op(i)]["out"] = y ^ np.uint16(np.arange(n) == 3)
            for k, a in enumerate((x, w, ws)):
                args[10 * i + k] = a
            by_i[i] = {"i": i, "spec": descriptor_id(fn), "args": [[["x", 10 * i + k, 0, a.size]] for k, a in enumerate((x, w, ws))]}
            ds[i] = {"statics": {"K": K, "N": n}, "op_path": op(i), "members": members, "row": 0, "step": 16}

        class Index:
            pnames = ["weights"]

            def __init__(self):
                self.by_i = by_i

            def describe(self, r):
                return ds[r["i"]]

        class Store:
            def words(self, rid, step, op_path, member, lo, hi):
                return words[op_path][member][lo:hi], None

        monkeypatch.setattr(SRW, "index_of", lambda inst, _pool, **kw: Index(), raising=False)
        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [args[group[0][1]][:group[0][3]]])
        return {"store": Store(), "args": args, "words": words}
    return make


def wired(tmpls: list[str], caps: dict[str, int]):
    """The window's members as `serving_rows.population` cuts and wires them, without its population program."""
    rows, _defs, counts = SRW.member_rows({"r0": {}}, tmpls)
    return PR.wire(SRW.members(tmpls, rows, counts, caps), tmpls, rows, SRW.row_units), rows


def test_a_row_holds_its_quantisation_once_and_n_coordinates_of_its_gemm():
    spec = descriptor_id(linear())
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(linear(1)), N), (descriptor_id(quant()), 1),
                                        (descriptor_id(gemm(N)), 1), *((descriptor_id(gemm(c)), N // c) for c in (8, 4, 2)),
                                        (descriptor_id(gemm(1)), N))
    t = SRW.template(TQ)
    assert (t.layout, t.word, t.packed) == ("part", descriptor_id(quant()), (16, 8, 32))
    assert SRW.part(descriptor_id(quant()), spec) == SRW.PARTS["FlashInferFp8Linear_v1"][0]
    assert SRW.part(descriptor_id(gemm(1)), spec) == SRW.PARTS["FlashInferFp8Linear_v1"][1]
    assert SRW.part(descriptor_id(gemm(1)), descriptor_id(bind(DG.DeepGemmFp8Linear, K=K, N=N))) == SRW.PARTS["DeepGemmFp8Linear_v1"][1]
    assert SRW.part(descriptor_id(linear(1)), spec) is None and SRW.part(descriptor_id(quant()), descriptor_id(quant())) is None


def test_the_quantisation_then_the_coordinates_compute_the_linear_s_output(served):
    """The composed units equal ``FlashInferFp8Linear_v1``'s output: the quantisation's Definition on its unit's x is the q and
    s it holds, every coordinate reads those q and s, and the GEMM coordinate's Definition on unit n's rows is the linear's word n."""
    x = served()
    tq, tg = SRW.template(TQ), SRW.template(TG)
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [TQ, TG])
    assert counts == [1, N] and [[(rid, r["i"]) for rid, _pi, r, _d in rs] for rs in rows] == [[("r0", 7)]] * 2
    mq, mg = SRW.members([TQ, TG], rows, counts)
    assert (mq.n, mq.base, mq.rows, mq.outs) == (1, 0, (("in0", K),), (("out.q", K // 2), ("out.s", 2 * KG)))
    assert (mg.n, mg.base, mg.rows, mg.outs) == (N, 1, (("in0", K // 2), ("in1", 2 * KG), ("in2", K // 2), ("in3", 2 * KG)),
                                                 (("out.out", 1),))
    dq, eq = SRW.read(mq, rows[0], x["store"], None)
    dg, eg = SRW.read(mg, rows[1], x["store"], None)
    assert eq == [[0, 1, "r0", 16, 7, op(7), "q,s", 0, 7]] and eg == [[1, 1 + N, "r0", 16, 7, op(7), "out", 0, 7]]
    assert dq["rows"]["in0"].tobytes() == x["args"][70].tobytes()
    (qs,) = ev(quant(), leaves(dq["rows"], mq.rows, tq.packed))
    q, s = leaves(dq["outs"], mq.outs, tq.packed[1:])
    assert qs[:, :K].tolist() == q.tolist() and qs[:, K:].tolist() == s.tolist()
    W, WS = x["args"][71].reshape(N, K), x["args"][72].reshape(NB, KG).astype("<u4")
    for n in range(N):
        assert dg["rows"]["in0"][n].tobytes() == dq["outs"]["out.q"].tobytes()
        assert dg["rows"]["in1"][n].tobytes() == dq["outs"]["out.s"].tobytes()
        assert dg["rows"]["in2"][n].tobytes() == W[n].tobytes()
        assert dg["rows"]["in3"][n].tobytes() == WS[n // FP8_BLOCK].tobytes()
    (y,) = ev(gemm(1), leaves(dg["rows"], mg.rows, tg.packed))
    assert y.reshape(-1).tolist() == dg["outs"]["out.out"].reshape(-1).tolist() == x["words"][op(7)]["out"].tolist()


def test_each_coordinate_s_q_and_s_are_wired_to_its_row_s_quantisation(served):
    """Two rows of ``WN`` coordinates, so two quantisation units (0 and 1) and 2 WN coordinates in chunks of 16: every
    coordinate's ``in0`` and ``in1`` are wired to ``o0`` and ``o1`` of its row's quantisation, and are those rows; each coordinate
    chunk's Definition on its rows is its committed output, the served words (the quantisation's chunks: the test below)."""
    x = served(("fi", "fi"), n=WN)
    ms, rows = wired([TQ, TG], {TQ: 1, TG: 16})
    quants, coords = [m for m in ms if m.template == TQ], [m for m in ms if m.template == TG]
    assert [(m.base, m.n, m.wired) for m in quants] == [(0, 1, ()), (1, 1, ())]
    assert [(m.base, m.n) for m in coords] == [(2 + lo, min(16, 2 * WN - lo)) for lo in range(0, 2 * WN, 16)]
    for m in coords:
        want = tuple((m.base + i - 2) // WN for i in range(m.n))
        assert m.wired == (("in0", "o0", want), ("in1", "o1", want))
    got = SRW.read_all(ms, rows, x["store"], None)
    for m, (g, _e) in zip(ms, got):
        if m.template == TG:
            stager(TG).reference_check(g["rows"], g["outs"])
    q = [g for m, (g, _e) in zip(ms, got) if m.template == TQ]
    for m, (g, _e) in zip(ms, got):
        for port, out, writers in m.wired:
            assert all(g["rows"][port][i].tobytes() == q[u]["outs"][out][0].tobytes() for i, u in enumerate(writers))
    y = np.concatenate([g["outs"]["o0"] for m, (g, _e) in zip(ms, got) if m.template == TG]).reshape(-1)
    assert y.tolist() == x["words"][op(7)]["out"].tolist() + x["words"][op(8)]["out"].tolist()


def test_a_deepgemm_row_s_coordinates_beside_them_read_its_committed_q_and_s_unwired(served):
    """A FlashInfer row (7) and a DeepGEMM row (8) share the coordinate template: its chunks are cut where the wires stop, row 7's
    coordinates wired to its quantisation, row 8's reading its committed q and s with no wire."""
    x = served(("fi", "dg"), n=WN)
    ms, rows = wired([TQ, TDQ, TG], {TQ: 1, TDQ: 1, TG: 16})
    coords = [m for m in ms if m.template == TG]
    ends = [2, 18, 34, 2 + WN, 50, 66, 2 + 2 * WN]
    assert [(m.name, m.base, m.n, bool(m.wired)) for m in coords] == [
        (f"{TG}/c{j}", a, b - a, a < 2 + WN) for j, (a, b) in enumerate(zip(ends, ends[1:]))]
    assert all(m.wired == (("in0", "o0", (0,) * m.n), ("in1", "o1", (0,) * m.n)) for m in coords if m.wired)
    got = SRW.read_all(ms, rows, x["store"], None)
    for m, (g, _e) in zip(ms, got):
        if m.template == TG:
            stager(TG).reference_check(g["rows"], g["outs"])
    dg_in0 = np.concatenate([g["rows"]["in0"] for m, (g, _e) in zip(ms, got) if m.template == TG and not m.wired])
    assert len(dg_in0) == WN and all(r.tobytes() == x["words"][op(8)]["x_q"].tobytes() for r in dg_in0)
    y = np.concatenate([g["outs"]["o0"] for m, (g, _e) in zip(ms, got) if m.template == TG]).reshape(-1)
    assert y.tolist() == x["words"][op(7)]["out"].tolist() + x["words"][op(8)]["out"].tolist()


@pytest.mark.slow
@pytest.mark.parametrize("kinds, tmpls", [(("fi", "fi"), [TQ, TG]), (("fi", "dg"), [TQ, TDQ, TG])], ids=["two-flashinfer", "mixed"])
def test_the_wires_hold_on_the_window_s_commitment(served, kinds, tmpls):
    """The window `serving_rows.population` derives has the members above; its circuit record counts the coordinates' q and s
    rows as wired; each quantisation chunk's Boolean Definition on its rows is its committed q and s; the window commits each
    wired row under its writer's salt, so the verifier's check on the public bodies holds, and refuses a wire to another
    quantisation and a wired row that is not its writer's output.  Slow (about 10 s a quantisation family): the population
    program and the stager's check trace each quantisation's Boolean Definition, which the tests above never build."""
    x = served(kinds, n=WN)
    caps = {**{t: 1 for t in tmpls[:-1]}, TG: 16}
    w, rows = SRW.population({"r0": {}}, {"templates": tmpls, "law": "subset:2", "chunk_cap": caps}, "ab" * 64)
    assert w.members == tuple(wired(tmpls, caps)[0])
    doc = json.loads(IF.circuit_bytes(SRW.circuit_record(w, rows)))
    assert IF.input_count(doc)["wired"] == kinds.count("fi") * WN * (8 * K + 32 * KG)
    got = SRW.read_all(w.members, rows, x["store"], None)
    for m, (g, _e) in zip(w.members, got):
        stager(m.template).reference_check(g["rows"], g["outs"])
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    assert WI.check(w.members, publics) == {}
    k = next(k for k, m in enumerate(w.members) if m.wired)
    other = replace(w, members=tuple(replace(m, wired=tuple((p, o, tuple(1 - u for u in ws)) for p, o, ws in m.wired)) if j == k
                                     else m for j, m in enumerate(w.members)))
    assert list(WI.check(other.members, publics)) == [w.members[k].key]
    bad = [dict(g, rows={**g["rows"], "in1": g["rows"]["in1"] ^ np.uint16(1)}) if j == k else g for j, (g, _e) in enumerate(got)]
    with pytest.raises(ValueError, match=f"{w.members[k].key}: a in1 row is not the o1 row"):
        CS.commit(w, bad, salt_key=bytes(32))


def test_a_partition_without_the_quantisation_s_chunks_is_refused(served):
    served()
    with pytest.raises(ValueError, match=r"row 7 \(.*\) reads its q, which its part FlashInferFp8Quant1x128_v1 returns: name that "
                                         r"part's template in the partition"):
        wired([TG], {TG: 64})
    with pytest.raises(ValueError, match=rf"{TQ}: its units are joined by a wire of row 7's q, and are no chunks"):
        wired([TQ, TG], {TG: 64})


def test_a_served_output_that_is_not_the_twin_s_on_its_intermediates_is_refused(served):
    x = served(bad=True)
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [TG])
    (m,) = SRW.members([TG], rows, counts)
    with pytest.raises(ValueError, match=r"row 7 \(.*\): the twin's output on its intermediates \['q', 's'\] is not the row's "
                                         r"served out words"):
        SRW.read(m, rows[0], x["store"], None)
