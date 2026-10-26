"""The elementwise rows split into column blocks (`pipeline.serving_rows`' ``cols`` layout, ``COLS``): a ``MoeSum_v1{TOPK,H}``
or ``AllReduceMnnvl_v1{WORLD,N,SHOT}`` row W wide holds W / w units of its Definition at width w, for every halving w of W, unit
j reading columns [jw, jw + w) of each of its rows and returning the row's output columns [jw, jw + w).  The units cover the
row's outputs with nothing left over (W / w units under one index entry, their outputs the row's output in order) and compute
them: the Definition at w on each unit's committed rows is the row's Definition's columns.  At w = W the member is the row as
the plain row layout reads it."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST
from verity.ml.definitions.registry import boolean_moe as BMOE
from verity.ml.definitions.registry import boolean_tp_allreduce as BTA
from verity.ml.definitions.registry import moe as M
from verity.ml.definitions.registry import tp_allreduce as TA

TOPK, WORLD, W = 3, 4, 8
OPS = {"MoeSum_v1": "model.layers.3.mlp.experts", "AllReduceMnnvl_v1": "model.layers.6.mlp"}
#: the families that are rows of their own and elementwise at every statics (a part's blocks: test_serving_rows_parts.py; torch's
#: multimem all-reduce, whose blocks `cols_refused` restricts: test_serving_rows_multimem.py)
FAMILIES = [f for f in SRW.COLS if f not in SRW.PART_OF and f != "AllReduceMultimem_v1"]


def defn(family: str, w: int, boolean: bool = False):
    if family == "MoeSum_v1":
        return bind(BMOE.MoeSumV3 if boolean else M.MoeSum, TOPK=TOPK, H=w)
    return bind(BTA.AllReduceMnnvl if boolean else TA.AllReduceMnnvl, WORLD=WORLD, N=w, SHOT=1)


def evaluate(fn, leaf: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, leaf.astype(np.uint64))).reshape(len(leaf), -1)


def bf16s(rng, n: int) -> np.ndarray:
    v = (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    v[:4] = [0x8000, 0x7F80, 0xFF80, 0x7FC1]
    return v


@pytest.fixture
def row(monkeypatch):
    def make(family: str, seed: int = 1, short: int = 0):
        rng = np.random.default_rng(seed)
        rows = TOPK if family == "MoeSum_v1" else WORLD
        x = bf16s(rng, rows * W)
        y = evaluate(defn(family, W), x.reshape(1, -1)).reshape(-1)
        r = {"i": 9, "spec": descriptor_id(defn(family, W)), "args": [[["x", 0, 0, x.size - short]]]}
        statics = {"TOPK": TOPK, "H": W} if family == "MoeSum_v1" else {"WORLD": WORLD, "N": W, "SHOT": 1}
        d = {"statics": statics, "op_path": OPS[family], "members": [("out", W, "bf16")], "row": 0, "step": 16}

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return y[lo:hi].astype(np.uint16), None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [x[:group[0][3]]])
        return ("r0", None, r, d), x.reshape(rows, W), y, Store()
    return make


@pytest.mark.parametrize("family", FAMILIES)
def test_a_row_holds_its_definition_at_every_halving_of_its_width(family):
    spec = descriptor_id(defn(family, W))
    assert SRW.unit_templates(spec) == ((spec, 1), *((descriptor_id(defn(family, W // k)), k) for k in (2, 4, 8)))
    for w in (W, W // 2, 1):
        assert SRW.template(descriptor_id(defn(family, w))).layout == "cols"
    t = SRW.template(descriptor_id(defn(family, W // 2, boolean=True)))
    assert (t.layout, t.word, t.packed) == ("cols", descriptor_id(defn(family, W // 2)), None)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("w", [W // 2, W // 4, 1])
def test_the_column_blocks_cover_the_row_s_outputs_and_compute_them(row, family, w):
    x, X, y, store = row(family)
    nh, rows = W // w, len(X)
    T = descriptor_id(defn(family, w, boolean=True))
    (m,) = SRW.members([T], [[x]], [nh])
    assert (m.n, m.rows, m.outs) == (nh, (("in0", rows * w),), (("out.out", w),))
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, nh, "r0", 16, int(OPS[family].split(".")[2]), OPS[family], "out", 0, 9]]
    assert data["outs"]["out.out"].reshape(-1).tolist() == y.tolist()
    for j in range(nh):
        assert data["rows"]["in0"][j].tolist() == X[:, j * w:(j + 1) * w].reshape(-1).tolist()
    assert evaluate(defn(family, w), data["rows"]["in0"]).tolist() == data["outs"]["out.out"].tolist()


@pytest.mark.parametrize("family", FAMILIES)
def test_at_its_own_width_the_member_is_the_row(row, family, monkeypatch):
    x, X, y, store = row(family)
    T = descriptor_id(defn(family, W))
    (m,) = SRW.members([T], [[x]], [1])
    data, entries = SRW.read(m, [x], store, None)
    monkeypatch.setitem(ST._TEMPLATES, T, SRW.Template(T, defn(family, W), T, "row"))
    (m0,) = SRW.members([T], [[x]], [1])
    plain, plain_entries = SRW.read(m0, [x], store, None)
    assert (m.n, m.rows, m.outs) == (m0.n, m0.rows, m0.outs) == (1, (("in0", X.size),), (("out.out", W),))
    assert entries == plain_entries
    for k in ("rows", "outs"):
        assert {p: a.tolist() for p, a in data[k].items()} == {p: a.tolist() for p, a in plain[k].items()}


def test_a_row_whose_operand_is_not_its_statics_is_refused(row):
    x, _X, _y, store = row("MoeSum_v1", short=2)
    (m,) = SRW.members([descriptor_id(defn("MoeSum_v1", W // 2))], [[x]], [2])
    with pytest.raises(ValueError, match=r"row 9 \(model.layers.3.mlp.experts\): inputs and outputs of \[22, 8\] words are not 2"):
        SRW.read(m, [x], store, None)
