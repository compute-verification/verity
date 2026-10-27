"""A window's ``coord`` units in chunks (`pipeline.serving_rows`, ``CHUNKED``): the expert-indexed pair GEMMs' coordinates
(``COORDS``) are consecutive chunks of at most the partition's ``chunk_cap``, each a member on the bit rows C-Flock's `UnitStager`
stages (`chunk_rows.unit_rows`: ``in<g>`` per parameter, ``o0``), and the chunks are the template's units read once, so that
the coordinate on each chunk's rows, parsed as the stager parses them, is the rows' committed words, and a flipped word is not."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.registry import boolean_fp8_moe as BFM
from verity.ml.definitions.registry import fp8_moe_experts as X

E, K, N, G, KB = 3, 64, 40, 32, 32
NB, KG = -(-N // G), K // G
GEMM = bind(X.MoeExpertFp8GemmW, E=E, K=K, N=N, G=G, KB=KB, SWAP=False, DOT=HopperE4m3MmaV2Dot32)
T = descriptor_id(bind(BFM.MoeExpertFp8GemmWV2, E=E, K=K, N=1, G=G, KB=KB, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32))
CAP = 24


def _bits(a: np.ndarray) -> np.ndarray:
    return a.view(f"<u{a.dtype.itemsize}") if a.dtype.itemsize > 1 else a


@pytest.fixture
def pairs(monkeypatch):
    """Two routed pairs of one layer (two experts, one weight stack), each a ``MoeExpertFp8GemmW_v1{N}`` row, and the store of
    their committed outputs."""
    rng = np.random.default_rng(5)

    def codes(n):
        c = rng.integers(0, 256, n, dtype=np.uint8)
        return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)

    def scales(n):
        return ((2.0 ** rng.integers(-8, 2, n)) * rng.uniform(1, 2, n)).astype(np.float32)

    w, ws = codes(E * N * K), scales(E * NB * KG)
    ops = {}
    for p, e in enumerate((2, 0)):
        ops.update({(p, 0): codes(K), (p, 1): scales(KG), (p, 2): np.array([e], np.int32), (p, 3): rng.uniform(0, 1, 1).astype(np.float32)})
    ops[(9, 4)], ops[(9, 5)] = w, ws
    ys = [np.asarray(evaluate_batch(GEMM, *[_bits(ops[(p, g)]).astype(np.uint64).reshape(1, -1) for g in range(4)],
                                    _bits(w).astype(np.uint64).reshape(1, -1), _bits(ws).astype(np.uint64).reshape(1, -1))
                     ).reshape(-1).astype(np.uint16) for p in range(2)]
    rows = []
    for p in range(2):
        r = {"i": 7 + p, "spec": descriptor_id(GEMM),
             "args": [[["x", 10 * p + g, 0, ops[(p, g)].size]] for g in range(4)] + [[["x", 94, 0, w.size]], [["x", 95, 0, ws.size]]]}
        d = {"statics": {"E": E, "K": K, "N": N, "G": G, "KB": KB}, "op_path": "moe/L3", "members": [("down", N, "bf16")], "row": p,
             "step": 16}
        rows.append(("r0", None, r, d))

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return np.concatenate(ys)[lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [ops[divmod(g[0][1], 10)]])
    return rows, np.concatenate(ys), Store()


def test_a_window_s_coordinates_are_chunks_of_the_stager_s_bit_rows(pairs):
    rows, _y, _store = pairs
    in_rows, out_rows = CR.unit_rows(SRW.template(T))
    params, _ret = SRW.template(T).fn.signature()
    assert [(p, s, ps) for p, s, _b, ps in in_rows] == [(f"in{g}", V2, [nm]) for g, (nm, _t) in enumerate(params)] and out_rows == [(V2, (0,))]
    ms = SRW.members([T], [rows], [2 * N], {T: CAP})
    assert [(m.key, m.base, m.n) for m in ms] == [(f"{T}/c{j}", CAP * j, n) for j, n in enumerate((CAP, CAP, CAP, 2 * N - 3 * CAP))]
    assert all(m.rows == tuple((p, b // 16) for p, _s, b, _ps in in_rows) and m.outs == (("o0", 1),)
               and dict(m.schemas) == {**{p: V2 for p, *_ in in_rows}, "o0": V2} for m in ms)
    with pytest.raises(ValueError, match="no chunk cap; a partition file names each chunked template's flock-circuit chunk cap"):
        SRW.members([T], [rows], [2 * N], {})


def test_the_chunks_are_the_template_s_units_read_once(pairs):
    rows, y, store = pairs
    (whole,) = SRW.members([T], [rows], [2 * N])
    data, entries = SRW.read(whole, rows, store, None)
    ms = SRW.members([T], [rows], [2 * N], {T: CAP})
    got = SRW.read_all(ms, [rows], store, None)
    for p, _w in whole.rows:
        assert np.array_equal(np.concatenate([g[0]["rows"][p] for g in got]), data["rows"][p])
    assert np.concatenate([g[0]["outs"]["o0"] for g in got]).reshape(-1).tolist() == y.tolist()
    assert [[e[:2] for e in g[1]] for g in got] == [[[0, N]], [[0, N], [N, 2 * N]], [[N, 2 * N]], [[N, 2 * N]]]
    assert [e[2:] for e in entries] == [["r0", 16, None, "moe/L3", "down", p, 7 + p] for p in range(2)]
    with pytest.raises(ValueError, match=rf"/c1: its rows hold {2 * N} units, not its {CAP}"):
        SRW.read(ms[1], rows, store, None)


def test_a_chunk_on_its_rows_computes_the_committed_words_and_a_flipped_word_is_refused(pairs):
    rows, _y, store = pairs
    ms = SRW.members([T], [rows], [2 * N], {T: CAP})
    got = SRW.read_all(ms, [rows], store, None)
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(T).fn, "coord-test", {}, CR.unit_rows(SRW.template(T))[0]
    for g, _entries in got:
        st.reference_check(g["rows"], g["outs"])
    g, _entries = got[1]
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(g["rows"], {"o0": g["outs"]["o0"] ^ np.uint16(1)})
    w = CS.Binding(*CS.population_program([(SRW.template(T).fn, 2 * N)], several=True), {"law": "subset", "k": 2}, "ab" * 32, tuple(ms))
    assert CS.commit(w, [g for g, _e in got], salt_key=bytes(32)).window.population == 2 * N
