"""torch's multimem all-reduce split into column blocks (`pipeline.serving_rows`' ``cols`` layout, ``COLS`` on SIZE): an
``AllReduceMultimem_v1{WORLD,SIZE,TREES,FTZ}`` row whose owners' TREES are one tree holds SIZE / w units of its Definition at
width w, for every halving w of SIZE that is whole 8-element vectors, unit j reading columns [jw, jw + w) of every rank's partials
and returning the row's output columns [jw, jw + w); the units cover the row's output with nothing left over and compute it.  A
row whose owners' trees differ holds no blocks: a block's elements are reduced through other owners' trees than the row's (its
owners are the block's, `tp_allreduce.multimem_owners`), so its Definition is not the row's columns, and a member of blocks
handed such a row is refused (`serving_templates.cols_refused`)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST
from verity.ml.definitions.registry import boolean_tp_allreduce as BTA
from verity.ml.definitions.registry import tp_allreduce as TA

WORLD, SIZE = 4, 32
OP = "model.layers.6.mlp"
ONE = (TA.chain(WORLD),) * WORLD
#: owner 1's tree adds rank 2 before rank 1: at SIZE = 32 each rank owns 16 elements, so ranks 0 and 1 own the row's
TWO = (TA.chain(WORLD), TA.chain(WORLD, (0, 2, 1, 3)), TA.chain(WORLD), TA.chain(WORLD))


def defn(w: int, trees=ONE, boolean: bool = False):
    return bind(BTA.AllReduceMultimem if boolean else TA.AllReduceMultimem, WORLD=WORLD, SIZE=w, TREES=trees, FTZ=False)


def evaluate(fn, leaf: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, leaf.astype(np.uint64))).reshape(len(leaf), -1)


@pytest.fixture
def row(monkeypatch):
    """A synthetic row of every rank's partials (a committed member) and its output by its Definition; under ``TWO`` every
    column is 1, -1, 2^-30, 2^-30 over the ranks, which its two trees sum to 2^-29 and 2^-30 (1 + 2^-30 rounds to 1 in f32)."""
    def make(trees=ONE):
        rng = np.random.default_rng(1)
        X = (rng.standard_normal((WORLD, SIZE)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        if trees is TWO:
            X[:] = np.array([0x3F80, 0xBF80, 0x3080, 0x3080], np.uint16)[:, None]
        x = X.reshape(-1)
        y = evaluate(defn(SIZE, trees), x.reshape(1, -1)).reshape(-1)
        r = {"i": 9, "spec": descriptor_id(defn(SIZE, trees)), "args": [[["x", 0, 0, x.size]]]}
        d = {"statics": {"WORLD": WORLD, "SIZE": SIZE, "TREES": trees, "FTZ": False}, "op_path": OP,
             "members": [("out", SIZE, "bf16")], "row": 0, "step": 16}

        class Index:
            by_i = {9: r}

            def describe(self, _r):
                return d

        class Store:
            def words(self, rid, step, op, member, lo, hi):
                return y[lo:hi].astype(np.uint16), None

        monkeypatch.setattr(SRW, "index_of", lambda inst, _reuse, **kw: Index(), raising=False)
        monkeypatch.setattr(SRW, "_slices", lambda _pi, _s, _w, _rid, _a, group: [x[:group[0][3]]])
        return ("r0", None, r, d), X, y, Store()
    return make


def test_a_row_of_one_tree_holds_its_definition_at_every_halving_of_whole_vectors():
    spec = descriptor_id(defn(SIZE))
    assert SRW.unit_templates(spec) == ((spec, 1), (descriptor_id(defn(SIZE // 2)), 2), (descriptor_id(defn(SIZE // 4)), 4))
    assert ST.cols_refused(descriptor_id(defn(SIZE // 4))) is None
    assert "8-element vectors" in ST.cols_refused(descriptor_id(defn(SIZE // 8)))
    t = SRW.template(descriptor_id(defn(SIZE // 2, boolean=True)))
    assert (t.layout, t.word, t.packed) == ("cols", descriptor_id(defn(SIZE // 2)), None)


@pytest.mark.parametrize("w", [SIZE // 2, SIZE // 4])
def test_the_column_blocks_cover_the_row_s_output_and_compute_it(row, w):
    x, X, y, store = row()
    nh = SIZE // w
    T = descriptor_id(defn(w, boolean=True))
    rows, _defs, counts = SRW.member_rows({"r0": {}}, [T])
    assert counts == [nh]
    (m,) = SRW.members([T], rows, counts)
    assert (m.n, m.rows, m.outs) == (nh, (("in0", WORLD * w),), (("out.out", w),))
    data, entries = SRW.read(m, rows[0], store, None)
    assert entries == [[0, nh, "r0", 16, 6, OP, "out", 0, 9]]
    assert data["outs"]["out.out"].reshape(-1).tolist() == y.tolist()
    for j in range(nh):
        assert data["rows"]["in0"][j].tolist() == X[:, j * w:(j + 1) * w].reshape(-1).tolist()
    assert evaluate(defn(w), data["rows"]["in0"]).tolist() == data["outs"]["out.out"].tolist()


def test_a_row_of_two_trees_holds_no_blocks_and_a_member_of_blocks_is_refused_it(row):
    x, X, y, store = row(TWO)
    spec, half = descriptor_id(defn(SIZE, TWO)), descriptor_id(defn(SIZE // 2, TWO))
    assert y[:SIZE // 2].tolist() == [0x3100] * (SIZE // 2) and y[SIZE // 2:].tolist() == [0x3080] * (SIZE // 2)
    assert evaluate(defn(SIZE // 2, TWO), X[:, SIZE // 2:].reshape(1, -1)).reshape(-1).tolist() != y[SIZE // 2:].tolist()
    assert SRW.unit_templates(spec) == ((spec, 1),)
    assert "2 trees" in ST.cols_refused(half)
    with pytest.raises(LookupError, match="no rows of"):
        SRW.member_rows({"r0": {}}, [half])
    (m,) = SRW.members([half], [[x]], [2])
    with pytest.raises(ValueError, match=rf"row 9 \({OP}\): 2 column blocks of AllReduceMultimem_v1.* its owners' TREES are 2 trees"):
        SRW.read(m, [x], store, None)
    (m,) = SRW.members([descriptor_id(defn(SIZE // 2))], [[x]], [2])
    with pytest.raises(ValueError, match="its owners' TREES are 2 trees"):
        SRW.read(m, [x], store, None)
