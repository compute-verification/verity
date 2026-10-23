"""The serving rows of a MoE pair row (`pipeline.serving_rows`): a ``MoeExpertFp8GemmW_v1`` Program row (one routed pair's w2
launch) is cut into its N ``MoeExpertFp8CoordinateW_v1`` instances, its weight blocks' batches counted as one template; each
coordinate's bit rows (`unit_rows`) are read off the row's operands, so that the coordinate on them, parsed as C-Flock's
`UnitStager` parses its rows, is the row's committed output word."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager
from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.program.registry import fp8_moe_experts as FE

E, K, N, G = 4, 64, 48, 32
KG, NB = K // G, -(-N // G)
STATICS = {"E": E, "K": K, "G": G, "KB": 32, "SWAP": False, "DOT": BlackwellE4m3QmmaDot32}
GEMM = bind(FE.MoeExpertFp8GemmW, N=N, **STATICS)
COORD = bind(FE.MoeExpertFp8CoordinateW, **STATICS)


def row(i: int, pair: int, op: str = "moe/L3", step: int = 0) -> SRW.Row:
    at = lambda lo, hi: [["n", pair, lo, hi]]  # noqa: E731
    r = {"i": i, "spec": descriptor_id(GEMM), "args": [at(0, K), at(K, K + KG), at(K + KG, K + KG + 1), at(K + KG + 1, K + KG + 2),
                                                        [["p", 0, 0, E * N * K]], [["p", 0, E * N * K, E * N * K + E * NB * KG]]]}
    d = {"statics": {"E": E, "K": K, "N": N, "G": G}, "op_path": op, "members": [("down", N, "bf16")], "row": pair, "step": step}
    return ("r0", None, r, d)


def test_a_pair_row_is_cut_into_its_coordinate_template_one_batch_per_weight_block():
    hits = SRW.unit_templates(descriptor_id(GEMM))
    assert hits[0] == (descriptor_id(GEMM), 1)
    assert [n for t, n in hits[1:] if t == descriptor_id(COORD)] == [G, N - G]
    tp = SRW.template(descriptor_id(COORD))
    assert (tp.word, tp.layout) == (descriptor_id(COORD), "moe")


def test_its_rows_are_the_pair_then_the_slab_and_the_scale_row():
    xbits, wbits = 8 * K + 32 * KG + 32 + 32, 8 * E * K + 32 * KG
    assert SRW.unit_rows(descriptor_id(COORD)) == ([("x", V2, xbits, ["xq", "sx", "e", "g"]), ("w", V2, wbits, ["slab", "sw"])],
                                                   [(V2, (0,))])


def test_member_rows_counts_every_weight_block_of_a_row_in_scope(monkeypatch):
    rows = {0: row(0, 0)[2], 1: row(1, 1)[2], 2: row(2, 0, op="moe/L4")[2]}
    desc = {0: row(0, 0)[3], 1: row(1, 1)[3], 2: row(2, 0, op="moe/L4")[3]}

    class Index:
        def __init__(self, _inst, **_kw):
            self.by_i = rows

        def describe(self, r):
            return desc[r["i"]]

    monkeypatch.setattr(SRW, "ProgramIndex", Index)
    got, _defs, counts = SRW.member_rows({"r0": {}}, [descriptor_id(COORD)], scope="moe/L3")
    assert [r[2]["i"] for r in got[0]] == [0, 1] and counts == [2 * N]


def test_each_chunk_is_a_member_of_bit_rows():
    ms = SRW.members([descriptor_id(COORD)], [[row(0, 0), row(1, 1)]], [2 * N], {descriptor_id(COORD): 40})
    assert [(m.key, m.base, m.n) for m in ms] == [(f"{descriptor_id(COORD)}/c{j}", 40 * j, n) for j, n in enumerate((40, 40, 16))]
    (_x, _s, xbits, _px), (_w, _s2, wbits, _pw) = SRW.unit_rows(descriptor_id(COORD))[0]
    assert all(m.rows == (("x", xbits // 16), ("w", wbits // 16)) and m.outs == (("o0", 1),)
               and dict(m.schemas) == {"x": V2, "w": V2, "o0": V2} for m in ms)


def e4m3(rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.integers(0, 0x7F, n) | (rng.integers(0, 2, n) << 7)).astype(np.uint8)


def test_a_coordinates_rows_compute_the_rows_committed_word(monkeypatch):
    rng = np.random.default_rng(3)
    W, ws = e4m3(rng, E * N * K), np.array(rng.uniform(2 ** -10, 2 ** -6, E * NB * KG), np.float32).view(np.uint32)
    pairs = [(e4m3(rng, K), np.array([0.5, 0.25], np.float32).view(np.uint32), e, np.array([g], np.float32).view(np.uint32))
             for e, g in ((2, 0.75), (7, 0.5))]
    ys = [evaluate_batch(GEMM, xq[None], sx[None], np.array([[e]], np.uint32), g[None], W[None], ws[None]).astype(np.uint16)
          for xq, sx, e, g in pairs]
    rows = [row(7, 0), row(8, 1)]

    def operand(_pi, _store, _w, _rid, _arrive, group):
        r = next(x for x in rows if group in x[2]["args"])
        xq, sx, e, g = pairs[rows.index(r)]
        return [xq, sx, np.array([e], np.uint32), g, W, ws][r[2]["args"].index(group)]

    class Store:
        def words(self, _rid, _step, _op, _member, lo, hi):
            return np.concatenate(ys, axis=None)[lo:hi], None

    monkeypatch.setattr(SRW, "_operand", operand)
    ms = SRW.members([descriptor_id(COORD)], [rows], [2 * N], {descriptor_id(COORD): 40})
    got = SRW._read_moe(ms, rows, Store(), None, None)
    X = np.concatenate([g[0]["rows"]["x"] for g in got])
    Wr = np.concatenate([g[0]["rows"]["w"] for g in got])
    Y = np.concatenate([g[0]["outs"]["o0"] for g in got])
    nan_row = np.full(KG, 0x7FC00000, "<u4")
    for i in range(2 * N):
        p, n = divmod(i, N)
        xq, sx, e, g = pairs[p]
        sw = ws.reshape(E, NB, KG)[e, n // G] if e < E else nan_row
        assert X[i].tobytes() == xq.tobytes() + sx.astype("<u4").tobytes() + np.array([e], "<u4").tobytes() + g.astype("<u4").tobytes()
        assert Wr[i].tobytes() == W.reshape(E, N, K)[:, n, :].tobytes() + sw.astype("<u4").tobytes()
    assert Y.reshape(-1).tolist() == np.concatenate(ys, axis=None).tolist()
    assert [[e[:2] for e in g[1]] for g in got] == [[[0, N]], [[0, N], [N, 2 * N]], [[N, 2 * N]]]

    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = COORD, "moe-test", {}, SRW.unit_rows(descriptor_id(COORD))[0]
    st.reference_check({"x": X, "w": Wr}, {"o0": Y})
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check({"x": X, "w": Wr}, {"o0": Y ^ np.uint16(1)})

    w = CS.Window(*CS.population_program([(COORD, 2 * N)], several=True), {"law": "subset", "k": 2}, "ab" * 32, tuple(ms))
    assert CS.commit(w, [g[0] for g in got], salt_key=bytes(32)).window.population == 2 * N
