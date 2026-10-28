"""A window's ``cols`` units in chunks (`pipeline.serving_rows`, ``CHUNKED``): the column blocks of MoeSum, the MNNVL
all-reduce and the fused norm row's residual part (``COLS``) are consecutive chunks of at most the partition's ``chunk_cap``,
each a member on the bit rows Flock's `UnitStager` stages (`chunk_rows.unit_rows`: ``in<g>`` per parameter, ``o0``), and the
chunks are the template's blocks read once, so that the Definition on each chunk's rows, parsed as the stager parses them, is
the rows' committed columns, and a flipped word is not."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock.class_statement import UnitStager
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.registry import boolean_moe as BMOE
from verity.ml.definitions.registry import boolean_sigma as SG
from verity.ml.definitions.registry import boolean_tp_allreduce as BTA
from verity.ml.definitions.registry import moe as M
from verity.ml.definitions.registry import tp_allreduce as TA

TOPK, WORLD, W, w = 3, 4, 8, 2
NH, CAP = W // w, 3
OPS = {"MoeSum_v1": "model.layers.{}.mlp.experts", "AllReduceMnnvl_v1": "model.layers.{}.mlp"}
FAMILIES = list(OPS)


def defn(family: str, width: int, boolean: bool = False):
    if family == "MoeSum_v1":
        return bind(BMOE.MoeSumV3 if boolean else M.MoeSum, TOPK=TOPK, H=width)
    return bind(BTA.AllReduceMnnvl if boolean else TA.AllReduceMnnvl, WORLD=WORLD, N=width, SHOT=1)


def tmpl(family: str) -> str:
    return descriptor_id(defn(family, w, boolean=True))


@pytest.fixture
def rows(monkeypatch):
    """Two rows of one family W wide, in layers 3 and 5, and the store of their committed outputs."""
    def make(family: str):
        rng = np.random.default_rng(3)
        lead = TOPK if family == "MoeSum_v1" else WORLD
        xs = [(rng.standard_normal(lead * W).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16) for _ in range(2)]
        ys = [np.asarray(evaluate_batch(defn(family, W), x.astype(np.uint64).reshape(1, -1))).reshape(-1).astype(np.uint16) for x in xs]
        statics = {"TOPK": TOPK, "H": W} if family == "MoeSum_v1" else {"WORLD": WORLD, "N": W, "SHOT": 1}
        out = []
        for p, layer in enumerate((3, 5)):
            r = {"i": 7 + p, "spec": descriptor_id(defn(family, W)), "args": [[["x", p, 0, xs[p].size]]]}
            d = {"statics": statics, "op_path": OPS[family].format(layer), "members": [("out", W, "bf16")], "row": p, "step": 16}
            out.append(("r0", None, r, d))

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return np.concatenate(ys)[lo:hi], None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [xs[g[0][1]]])
        return out, np.concatenate(ys), Store()
    return make


@pytest.mark.parametrize("family", FAMILIES)
def test_a_window_s_column_blocks_are_chunks_of_the_stager_s_bit_rows(rows, family):
    rs, _y, _store = rows(family)
    T = tmpl(family)
    in_rows, out_rows = CR.unit_rows(SRW.template(T))
    params, _ret = SRW.template(T).fn.signature()
    assert [(p, s, ps) for p, s, _b, ps in in_rows] == [(f"in{g}", V2, [nm]) for g, (nm, _t) in enumerate(params)] and out_rows == [(V2, (0,))]
    ms = SRW.members([T], [rs], [2 * NH], {T: CAP})
    assert [(m.key, m.base, m.n) for m in ms] == [(f"{T}/c{j}", CAP * j, n) for j, n in enumerate((CAP, CAP, 2 * NH - 2 * CAP))]
    assert all(m.rows == tuple((p, b // 16) for p, _s, b, _ps in in_rows) and m.outs == (("o0", w),)
               and dict(m.schemas) == {**{p: V2 for p, *_ in in_rows}, "o0": V2} for m in ms)
    (whole,) = SRW.members([T], [rs], [2 * NH])
    assert (whole.n, whole.outs, whole.name) == (2 * NH, (("out.out", w),), "")
    with pytest.raises(ValueError, match="no chunk cap; a partition file names each chunked template's flock-circuit chunk cap"):
        SRW.members([T], [rs], [2 * NH], {})


@pytest.mark.parametrize("family", FAMILIES)
def test_the_chunks_are_the_template_s_blocks_read_once(rows, family):
    rs, y, store = rows(family)
    T = tmpl(family)
    (whole,) = SRW.members([T], [rs], [2 * NH])
    data, entries = SRW.read(whole, rs, store, None)
    ms = SRW.members([T], [rs], [2 * NH], {T: CAP})
    got = SRW.read_all(ms, [rs], store, None)
    for p, _w in whole.rows:
        assert np.array_equal(np.concatenate([g[0]["rows"][p] for g in got]), data["rows"][p])
    assert np.concatenate([g[0]["outs"]["o0"] for g in got]).reshape(-1).tolist() == y.tolist()
    assert [[e[:2] for e in g[1]] for g in got] == [[[0, NH]], [[0, NH], [NH, 2 * NH]], [[NH, 2 * NH]]]
    assert [g[1] for g in got][1] == entries
    with pytest.raises(ValueError, match=rf"/c1: its rows hold {2 * NH} units, not its {CAP}"):
        SRW.read(ms[1], rs, store, None)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_chunk_on_its_rows_computes_the_row_s_columns_and_a_flipped_word_is_refused(rows, family):
    rs, _y, store = rows(family)
    T = tmpl(family)
    ms = SRW.members([T], [rs], [2 * NH], {T: CAP})
    got = SRW.read_all(ms, [rs], store, None)
    st = stager(T)
    for g, _entries in got:
        st.reference_check(g["rows"], g["outs"])
    g, _entries = got[1]
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(g["rows"], {"o0": g["outs"]["o0"] ^ np.uint16(1)})
    win = CS.Window(*CS.population_program([(SRW.template(T).fn, 2 * NH)], several=True), {"law": "subset", "k": 2}, "ab" * 32,
                    tuple(ms))
    assert CS.commit(win, [g for g, _e in got], salt_key=bytes(32)).window.population == 2 * NH


def test_the_fused_norm_row_s_residual_part_is_chunks_of_its_blocks(monkeypatch):
    RN, rw = 256, 64
    statics = {"WORLD": WORLD, "N": RN, "SHOT": 1, "CLUSTER": 1, "EPS": 1e-5, "FMA": False}
    fused = bind(TA.AllReduceRmsNormMnnvl, **statics)
    rng = np.random.default_rng(1)
    xs = [((rng.standard_normal(n) * s).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
          for n, s in ((WORLD * RN, 0.5), (RN, 2.0), (RN, 0.1))]
    both = np.asarray(evaluate_batch(fused, *(x.astype(np.uint64).reshape(1, -1) for x in xs))).reshape(-1).astype(np.uint16)
    vals = {"normed": both[:RN], "residual": both[RN:]}
    r = {"i": 9, "spec": descriptor_id(fused), "args": [[["x", g, 0, x.size]] for g, x in enumerate(xs)]}
    d = {"statics": statics, "op_path": "model.layers.6.post_attention_layernorm", "members": [("normed", RN, "bf16"),
         ("residual", RN, "bf16")], "row": 0, "step": 16}

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return vals[member][lo:hi], None

    monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, g: [xs[g[0][1]]])
    T, row, nh = descriptor_id(bind(SG.AllReduceResidualMnnvl, WORLD=WORLD, N=rw, SHOT=1)), ("r0", None, r, d), RN // rw
    ms = SRW.members([T], [[row]], [nh], {T: CAP})
    assert [(m.n, m.rows, m.outs) for m in ms] == [(n, (("in0", WORLD * rw), ("in1", rw)), (("o0", rw),)) for n in (CAP, nh - CAP)]
    (whole,) = SRW.members([T], [[row]], [nh])
    data, entries = SRW.read(whole, [row], Store(), None)
    got = SRW.read_all(ms, [[row]], Store(), None)
    for p, _w in whole.rows:
        assert np.array_equal(np.concatenate([g[0]["rows"][p] for g in got]), data["rows"][p])
    assert np.concatenate([g[0]["outs"]["o0"] for g in got]).reshape(-1).tolist() == vals["residual"].tolist()
    assert [g[1] for g in got] == [entries, entries] and entries[0][6] == "residual"
    st = stager(T)
    for g, _entries in got:
        st.reference_check(g["rows"], g["outs"])
    with pytest.raises(AssertionError, match="outputs are not the IR reference's"):
        st.reference_check(got[0][0]["rows"], {"o0": got[0][0]["outs"]["o0"] ^ np.uint16(1)})


def stager(T: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(T).fn, "cols-test", {}, CR.unit_rows(SRW.template(T))[0]
    return st
