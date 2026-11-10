"""The elementwise rows split into column blocks (`pipeline.serving_rows`' ``cols`` layout, ``COLS``): a ``MoeSum_v1{TOPK,H}``,
``AllReduceMnnvl_v1{WORLD,N,SHOT}`` or ``ResidualAdd_v1{N}`` row W wide holds W / w units of its Definition at width w, for every
halving w of W, unit j reading columns [jw, jw + w) of each row of each of its argument groups and returning the row's output
columns [jw, jw + w).  The units cover the row's outputs with nothing left over (W / w units under one index entry, their outputs
the row's output in order) and compute them: the Definition at w on each unit's committed rows is the row's Definition's
columns.  At w = W the member is the row as the plain row layout reads it."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch

from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST
from verity.definitions.registry import b1 as B1
from verity.definitions.registry import boolean_moe as BMOE
from verity.definitions.registry import boolean_sigma as SG
from verity.definitions.registry import boolean_tp_allreduce as BTA
from verity.definitions.registry import moe as M
from verity.definitions.registry import tp_allreduce as TA

TOPK, WORLD, W = 3, 4, 8
OPS = {"MoeSum_v1": "model.layers.3.mlp.experts", "AllReduceMnnvl_v1": "model.layers.6.mlp", "ResidualAdd_v1": "model.layers.5.self_attn"}
#: each family's argument groups, as the rows W wide each holds (MoeSum's top-k outputs, the all-reduce's ranks' partials, the
#: residual add's x and r)
GROUPS = {"MoeSum_v1": (TOPK,), "AllReduceMnnvl_v1": (WORLD,), "ResidualAdd_v1": (1, 1)}
STATICS = {"MoeSum_v1": {"TOPK": TOPK, "H": W}, "AllReduceMnnvl_v1": {"WORLD": WORLD, "N": W, "SHOT": 1},
           "ResidualAdd_v1": {"N": W}}
#: the families that are rows of their own and elementwise at every statics (a part's blocks: test_serving_rows_parts.py; torch's
#: multimem all-reduce, whose blocks `cols_refused` restricts: test_serving_rows_multimem.py)
FAMILIES = [f for f in SRW.COLS if f not in SRW.PART_OF and f != "AllReduceMultimem_v1"]


def defn(family: str, w: int, boolean: bool = False):
    if family == "MoeSum_v1":
        return bind(BMOE.MoeSumV3 if boolean else M.MoeSum, TOPK=TOPK, H=w)
    if family == "ResidualAdd_v1":
        return bind(SG.ResidualAdd if boolean else B1.ResidualAdd, N=w)
    return bind(BTA.AllReduceMnnvl if boolean else TA.AllReduceMnnvl, WORLD=WORLD, N=w, SHOT=1)


def evaluate(fn, *leaves: np.ndarray) -> np.ndarray:
    """``fn`` on a batch, one array of rows per parameter."""
    return np.asarray(evaluate_batch(fn, *(a.astype(np.uint64) for a in leaves))).reshape(len(leaves[0]), -1)


def bf16s(rng, n: int) -> np.ndarray:
    v = (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    v[:4] = [0x8000, 0x7F80, 0xFF80, 0x7FC1]
    return v


@pytest.fixture
def row(monkeypatch):
    """A row of ``family`` W wide (its last argument group ``short`` words short), each argument group's rows, its output and
    the store of its committed output."""
    def make(family: str, seed: int = 1, short: int = 0):
        rng = np.random.default_rng(seed)
        xs = [bf16s(rng, n * W) for n in GROUPS[family]]
        y = evaluate(defn(family, W), *(a.reshape(1, -1) for a in xs)).reshape(-1)
        last = len(xs) - 1
        args = [[[f"x{g}", 0, 0, a.size - (short if g == last else 0)]] for g, a in enumerate(xs)]
        r = {"i": 9, "spec": descriptor_id(defn(family, W)), "args": args}
        d = {"statics": STATICS[family], "op_path": OPS[family], "members": [("out", W, "bf16")], "row": 0, "step": 16}

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return y[lo:hi].astype(np.uint16), None

        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [xs[int(group[0][0][1:])][:group[0][3]]])
        return ("r0", None, r, d), [a.reshape(-1, W) for a in xs], y, Store()
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
    x, Xs, y, store = row(family)
    nh = W // w
    T = descriptor_id(defn(family, w, boolean=True))
    (m,) = SRW.members([T], [[x]], [nh])
    assert (m.n, m.rows, m.outs) == (nh, tuple((f"in{g}", len(X) * w) for g, X in enumerate(Xs)), (("out.out", w),))
    data, entries = SRW.read(m, [x], store, None)
    assert entries == [[0, nh, "r0", 16, int(OPS[family].split(".")[2]), OPS[family], "out", 0, 9]]
    assert data["outs"]["out.out"].reshape(-1).tolist() == y.tolist()
    for g, X in enumerate(Xs):
        for j in range(nh):
            assert data["rows"][f"in{g}"][j].tolist() == X[:, j * w:(j + 1) * w].reshape(-1).tolist()
    leaves = [data["rows"][f"in{g}"] for g in range(len(Xs))]
    assert evaluate(defn(family, w), *leaves).tolist() == data["outs"]["out.out"].tolist()


@pytest.mark.parametrize("family", FAMILIES)
def test_at_its_own_width_the_member_is_the_row(row, family, monkeypatch):
    x, Xs, y, store = row(family)
    T = descriptor_id(defn(family, W))
    (m,) = SRW.members([T], [[x]], [1])
    data, entries = SRW.read(m, [x], store, None)
    monkeypatch.setitem(ST._TEMPLATES, T, SRW.Template(T, defn(family, W), T, "row"))
    (m0,) = SRW.members([T], [[x]], [1])
    plain, plain_entries = SRW.read(m0, [x], store, None)
    rows = tuple((f"in{g}", X.size) for g, X in enumerate(Xs))
    assert (m.n, m.rows, m.outs) == (m0.n, m0.rows, m0.outs) == (1, rows, (("out.out", W),))
    assert entries == plain_entries
    for k in ("rows", "outs"):
        assert {p: a.tolist() for p, a in data[k].items()} == {p: a.tolist() for p, a in plain[k].items()}


def test_a_row_whose_operand_is_not_its_statics_is_refused(row):
    x, _Xs, _y, store = row("MoeSum_v1", short=2)
    (m,) = SRW.members([descriptor_id(defn("MoeSum_v1", W // 2))], [[x]], [2])
    with pytest.raises(ValueError, match=r"row 9 \(model.layers.3.mlp.experts\): inputs and outputs of \[22, 8\] words are not 2"):
        SRW.read(m, [x], store, None)
