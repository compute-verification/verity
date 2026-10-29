"""The serving rows of an int8 row (`pipeline.serving_rows`): a ``ScaledMmS8_v1`` Program row is cut into its N
``ScaledMmS8Coordinate_v1`` instances (A6's template id, the word of the Boolean ``ScaledMmS8Coordinate_v2``), each template's units are consecutive chunks of at most its chunk cap,
each coordinate's rows are A6's bit rows read off the row's operands, and a word partition is never the window's."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as OG
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import serving_rows as SRW

DOT = '{"fn":"BlackwellS8ImmaDot32_v1"}'
K = 64


def coord(k: int) -> str:
    return f"ScaledMmS8Coordinate_v1{{K={k},DOT={DOT}}}"


def bcoord(k: int) -> str:
    return f'ScaledMmS8Coordinate_v2{{K={k},DOT={{"fn":"BlackwellS8ImmaDot32_v2"}}}}'


def row(i: int, n: int, op: str = "model.model.layers.0.self_attn.qkv_proj", token: int = 0, step: int = 0) -> SRW.Row:
    spec = f"ScaledMmS8_v1{{K={K},N={n},DOT={DOT}}}"
    r = {"i": i, "spec": spec, "args": [[["n", token, 0, K]], [["n", token, K, K + 1]], [["p", 0, 0, n * K]], [["p", 0, n * K, n * K + n]]]}
    d = {"statics": {"K": K, "N": n}, "op_path": op, "members": [("y", n, "bf16")], "row": token, "step": step}
    return ("r0", None, r, d)


def test_a_scaled_mm_row_is_cut_into_a6s_coordinate_template():
    hits = SRW.unit_templates(f"ScaledMmS8_v1{{K=4096,N=6144,DOT={DOT}}}")
    assert hits[1:] == ((coord(4096), 6144),)
    assert coord(4096) == P.template_id(bind(OG.ScaledMmS8Coordinate, K=4096, **OG.GEMM_SM120_S8))
    assert [(SRW.template(t).word, SRW.template(t).layout) for t in (coord(4096), bcoord(4096))] == [(coord(4096), "s8")] * 2


def test_each_template_is_consecutive_chunks_of_at_most_its_cap():
    rows = [[row(0, 6, token=0), row(1, 6, token=1)], [row(2, 3, op="model.model.layers.0.mlp.down_proj")]]
    ms = SRW.members([coord(K), bcoord(K)], rows, [12, 3], {coord(K): 5, bcoord(K): 8})
    assert [(m.key, m.base, m.n) for m in ms] == [(f"{coord(K)}/c0", 0, 5), (f"{coord(K)}/c1", 5, 5), (f"{coord(K)}/c2", 10, 2),
                                                  (f"{bcoord(K)}/c0", 12, 3)]
    assert all(m.rows == (("x", 34), ("w", 34)) and m.outs == (("o0", 1),) and dict(m.schemas)["x"] == CS.SCHEMA_HM96_SHA512_ROW_SEG
               for m in ms)
    with pytest.raises(ValueError, match="chunk_cap"):
        SRW.members([coord(K)], rows[:1], [12], {})
    with pytest.raises(ValueError, match="12 units read off the rows, 13"):
        SRW.members([coord(K)], rows[:1], [13], {coord(K): 5})


@pytest.mark.parametrize("query,law,match", [("Q_word", "stratified:32", r"word partition \(Q_word\) with the stratified law"),
                                             ("Q_word", "subset:32", "never a word partition"),
                                             ("Q_call", "stratified:32", "never a word partition")])
def test_the_window_is_never_a_word_partition(query, law, match):
    with pytest.raises(ValueError, match=match):
        SRW.population({}, {"templates": [coord(K)], "law": law, "query": query}, "ab" * 32)


def test_a_coordinate_reads_a6s_rows_off_the_rows_operands(monkeypatch):
    rng = np.random.default_rng(1)
    toks = [rng.integers(-127, 128, K, dtype=np.int8) for _ in range(2)]
    sx = [np.array([s], np.float32).view(np.uint32) for s in (0.5, 0.25)]
    W, sw = rng.integers(-127, 128, (3, K), dtype=np.int8), np.array([1.0, 2.0, 3.0], np.float32).view(np.uint32)
    y = [rng.integers(0, 1 << 16, 3, dtype=np.uint16) for _ in range(2)]
    rows = [row(7, 3, token=0), row(8, 3, token=1)]

    def operand(_pi, _store, _w, _rid, _arrive, group):
        r = next(x for x in rows if group in x[2]["args"])
        t = rows.index(r)
        return [toks[t].view(np.uint8), sx[t], W.reshape(-1).view(np.uint8), sw][r[2]["args"].index(group)]

    class Store:
        def words(self, _rid, _step, _op, _member, lo, hi):
            return np.concatenate(y)[lo:hi], None

    monkeypatch.setattr(SRW, "_operand", operand)
    ms = SRW.members([coord(K)], [rows], [6], {coord(K): 4})
    got = SRW._read_s8(ms, rows, Store(), None, None)
    X = np.concatenate([g[0]["rows"]["x"] for g in got])
    Wr = np.concatenate([g[0]["rows"]["w"] for g in got])
    for i in range(6):
        t, c = divmod(i, 3)
        assert X[i].astype("<u2").tobytes() == toks[t].tobytes() + sx[t].astype("<u4").tobytes()
        assert Wr[i].astype("<u2").tobytes() == W[c].tobytes() + sw[c:c + 1].astype("<u4").tobytes()
    assert np.concatenate([g[0]["outs"]["o0"] for g in got]).reshape(-1).tolist() == np.concatenate(y).tolist()
    assert [len(g[0]["rows"]["x"]) for g in got] == [4, 2]
    assert [[e[:2] for e in g[1]] for g in got] == [[[0, 3], [3, 6]], [[3, 6]]]
    w = CS.Binding(*CS.population_program([(bind(OG.ScaledMmS8Coordinate, K=K, **OG.GEMM_SM120_S8), 6)], several=True),
                  {"law": "stratified", "k": 2}, "ab" * 32, tuple(ms))
    assert CS.commit(w, [g[0] for g in got], salt_key=bytes(32)).window.population == 6
