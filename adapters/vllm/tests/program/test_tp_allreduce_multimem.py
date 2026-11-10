"""`AllReduceMultimem` (torch's symmetric-memory all-reduce past FlashInfer's 0.5 MiB at TP 8, the NVSwitch's f32 adds in a tree
the 8-GPU capture reads off the switch): the word Definitions agree with `tp_allreduce_check.multimem_rows`, written from torch's
kernel, at chains, balanced and random trees, IEEE and FTZ; the owners are torch's 16-byte-aligned slices; a tree is recovered
from the clusters its adds absorb; the Call is a batch of 16-byte vectors per run of one tree, which the cut separates; the
Boolean versions agree with their word Definitions, are pinned by gate count and program digest, and pass
`Q_word_v1{X=16, W=32}`'s cut check."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import cut as CUT
from verity.ml.circuits import evaluation as E
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_tp_allreduce as BA
from verity.ml.definitions.registry import tp_allreduce as TA
from verity.ml.definitions.registry import tp_allreduce_check as K

SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                     0xFF80, 0x7FC0, 0x7F81, 0xFFFF], np.uint16)
CHAIN8 = TA.chain(8)
BALANCED8 = (((0, 1), (2, 3)), ((4, 5), (6, 7)))
SKEWED8 = ((((0, 4), 6), (1, 5)), ((2, 7), 3))


def _rows(rng, n: int, world: int) -> np.ndarray:
    """(n, world) partials: random words, special words, near-cancelling values of alternating sign, and subnormals."""
    base = rng.integers(0, 0x7F00, (n, 1), dtype=np.int64)
    canc = (base + rng.integers(-2, 3, (n, world))).clip(0, 0x7F7F) | (np.arange(world) % 2) << 15
    sub = rng.integers(0, 0x100, (n, world), dtype=np.int64) | rng.integers(0, 2, (n, world)) << 15
    return np.concatenate([rng.integers(0, 1 << 16, (n, world), dtype=np.int64), SPECIALS[rng.integers(0, len(SPECIALS), (n, world))],
                           canc, sub]).astype(np.uint16)


def _vectors(rows: np.ndarray) -> np.ndarray:
    """(n, world) partials as (n / 8, world, 8) vectors: each vector's 8 elements are 8 consecutive rows."""
    return np.ascontiguousarray(rows.reshape(-1, 8, rows.shape[1]).transpose(0, 2, 1))


def _out(out, n: int) -> np.ndarray:
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in parts], axis=1)


def _all_trees(ranks: tuple[int, ...]):
    if len(ranks) == 1:
        yield ranks[0]
        return
    first, rest = ranks[0], ranks[1:]
    for k in range(len(rest)):
        for other in itertools.combinations(rest, k):
            left = (first, *other)
            right = tuple(r for r in rest if r not in other)
            for a in _all_trees(left):
                for b in _all_trees(right):
                    yield (a, b)


def _random_tree(rng, world: int):
    nodes = list(range(world))
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append((a, b))
    return TA.multimem_tree(nodes[0], world)


# --- trees and owners ------------------------------------------------------------------------------------------------------------

def test_trees_are_canonical_and_checked() -> None:
    assert TA.multimem_tree((((5, 4), (7, 6)), ((1, 0), (3, 2))), 8) == BALANCED8
    assert CHAIN8 == (((((((0, 1), 2), 3), 4), 5), 6), 7)
    assert TA.chain(4, (2, 0, 3, 1)) == (((0, 2), 3), 1)
    assert TA.tree_leaves(SKEWED8) == (0, 4, 6, 1, 5, 2, 7, 3)
    for bad, match in ((((0, 1), (2, 2)), "each once"), (((0, 1), 2), "each once"), (((0, 1, 2), 3), "two"),
                       (((0, 1), (2, 3.0)), "rank or a pair"), (((0, True), (2, 3)), "rank or a pair")):
        with pytest.raises(ValueError, match=match):
            TA.multimem_tree(bad, 4)
    with pytest.raises(ValueError, match="one tree per owner"):
        TA.multimem_trees((BALANCED8,) * 7, 8)


def test_every_tree_is_its_clusters() -> None:
    """All 15 trees over 4 ranks, all 945 over 6, and 200 random ones over 8 come back from their adds' rank sets; sets that
    don't make a binary tree come back None."""
    for world, n in ((4, 15), (6, 945)):
        trees = {TA.multimem_tree(t, world) for t in _all_trees(tuple(range(world)))}
        assert len(trees) == n
        assert all(TA.tree_from_clusters(TA.tree_clusters(t), world) == t for t in trees)
    rng = np.random.default_rng(1)
    for _ in range(200):
        t = _random_tree(rng, 8)
        assert TA.tree_from_clusters(TA.tree_clusters(t), 8) == t
    full = frozenset(range(4))
    for bad in ([{0, 1}, {1, 2}, full], [{0, 1}, full], [{0, 1}, {2, 3}, {0, 1, 2}, full], [{0, 1}, {2, 3}, {0, 1, 2, 3, 4}],
                [{0, 1, 2}, full, {0}], [{0, 1, 2, 3}, {0, 1, 2}, {1, 2, 3}]):
        assert TA.tree_from_clusters([frozenset(c) for c in bad], 4) is None


@pytest.mark.parametrize("world,size", [(8, 8), (8, 120), (8, 128), (8, 136), (8, 6144), (8, 256 * 6144), (4, 40), (2, 8)])
def test_owners_are_torchs_16_byte_slices(world: int, size: int) -> None:
    """Rank r reduces elements r * P .. (r + 1) * P, P = round_up(SIZE, 16 * WORLD) / WORLD (torch's alignment, taken as elements
    of bf16), the last nonempty slice ending at SIZE."""
    runs = TA.multimem_owners(world, size)
    per = -(-size // (16 * world)) * 16
    assert runs[0][0] == 0 and runs[-1][1] == size and all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    assert all(lo == o * per and hi == min(lo + per, size) for lo, hi, o in runs)
    assert len(runs) == min(world, -(-size // per))
    with pytest.raises(ValueError, match="8-element"):
        TA.multimem_owners(world, size + 4)


def test_runs_merge_adjacent_owners_of_one_tree() -> None:
    trees = (BALANCED8,) * 3 + (CHAIN8,) * 2 + (BALANCED8,) * 3
    assert TA.multimem_runs(8, 128, trees) == [(0, 48, BALANCED8), (48, 80, CHAIN8), (80, 128, BALANCED8)]
    assert TA.multimem_runs(8, 64, (CHAIN8,) * 8) == [(0, 64, CHAIN8)]
    assert TA.SYMM_MEM_MAX_BYTES // (6144 * 2) == 5461


# --- the word Definitions against the numpy model ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,tree,ftz", [(0, CHAIN8, False), (1, BALANCED8, False), (2, SKEWED8, False), (3, CHAIN8, True),
                                            (4, BALANCED8, True)], ids=str)
def test_multimem_coordinate_is_the_kernels_sum(seed: int, tree, ftz: bool) -> None:
    x = _rows(np.random.default_rng(seed), 1500, 8)
    d = bind(TA.AllReduceMultimemCoordinate, WORLD=8, TREE=tree, FTZ=ftz)
    got = _out(E.evaluate_batch(d, x), len(x)).reshape(-1)
    want = np.concatenate([K.multimem_rows(np.ascontiguousarray(x[i:i + 8].T), (tree,) * 8, ftz) for i in range(0, len(x), 8)])
    assert (got == want).all()


@pytest.mark.slow
def test_multimem_rows_are_the_kernels_sums() -> None:
    """SIZE = 136 at WORLD 8 is 32-element slices, the last owner's empty: each slice at its owner's tree."""
    rng = np.random.default_rng(3)
    trees = tuple(_random_tree(rng, 8) for _ in range(8))
    for size, ftz in ((128, False), (136, True), (8, False)):
        parts = np.concatenate([rng.integers(0, 1 << 16, (30, 8, size)), SPECIALS[rng.integers(0, len(SPECIALS), (10, 8, size))]])
        parts = parts.astype(np.uint16)
        got = _out(E.evaluate_batch(bind(TA.AllReduceMultimem, WORLD=8, SIZE=size, TREES=trees, FTZ=ftz), parts), len(parts))
        assert (got == np.stack([K.multimem_rows(p, trees, ftz) for p in parts])).all(), size


def test_the_call_is_a_batch_of_vectors_per_tree() -> None:
    """Each run of one tree is a batch of `AllReduceMultimemVector_v1`, a member per 16-byte vector reading only the ranks'
    partials, so the cut gives every element its own unit and the vector is the Call's one callee at GLM-5.3's trees."""
    vec = lambda tree, ftz: bind(TA.AllReduceMultimemVector, WORLD=8, TREE=tree, FTZ=ftz)  # noqa: E731
    d = bind(TA.AllReduceMultimem, WORLD=8, SIZE=6144, TREES=(TA.GLM53_H200_TREE,) * 8, FTZ=False)
    assert [(n.form, n.fn.id, n.n) for n in d.body.nodes] == [("batch", vec(TA.GLM53_H200_TREE, False).id, 768)]
    assert CUT.separable(d) is not None and CUT.evaluate_definition(d, 16).units == 6144
    trees = (BALANCED8,) * 3 + (CHAIN8,) * 2 + (BALANCED8,) * 3
    d = bind(TA.AllReduceMultimem, WORLD=8, SIZE=128, TREES=trees, FTZ=True)
    assert [(n.fn.id, n.n) for n in d.body.nodes] == [(vec(t, True).id, k) for t, k in ((BALANCED8, 6), (CHAIN8, 4), (BALANCED8, 6))]
    assert CUT.evaluate_definition(d, 16).units == 128


@pytest.mark.parametrize("tree,ftz", [(TA.GLM53_H200_TREE, False), (SKEWED8, True)], ids=str)
def test_a_vector_is_its_elements_coordinates(tree, ftz: bool) -> None:
    x = _vectors(_rows(np.random.default_rng(11), 1600, 8))
    got = _out(E.evaluate_batch(bind(TA.AllReduceMultimemVector, WORLD=8, TREE=tree, FTZ=ftz), x), len(x))
    assert (got == np.stack([K.multimem_rows(v, (tree,) * 8, ftz) for v in x])).all()


def test_tree_and_ftz_are_not_vacuous() -> None:
    """2^30 - 2^30 cancels before or after a 1 is added, as the tree has it; a subnormal survives IEEE adds and not FTZ ones."""
    big, neg, one, tiny = 0x4E80, 0xCE80, 0x3F80, 0x0001
    col = [big, one, neg, 0, 0, 0, 0, 0]
    ev = lambda tree, ftz, c: E.evaluate(bind(TA.AllReduceMultimemCoordinate, WORLD=8, TREE=tree, FTZ=ftz), c)[0]  # noqa: E731
    assert ev(CHAIN8, False, col) == 0x0000 and ev(TA.chain(8, (0, 2, 1, 3, 4, 5, 6, 7)), False, col) == one
    assert ev(CHAIN8, False, [tiny] + [0] * 7) == tiny and ev(CHAIN8, True, [tiny] + [0] * 7) == 0x0000


# --- on bits ---------------------------------------------------------------------------------------------------------------

def _agree(target, x) -> None:
    word = TR.word_of(target)
    got, want = _out(BS.evaluate_words(target, x), len(x)), _out(E.evaluate_batch(word, x), len(x))
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(x)} rows differ from {word.id}, first {bad[:5]}"


def test_the_word_views() -> None:
    c = bind(BA.AllReduceMultimemCoordinate, WORLD=8, TREE=BALANCED8, FTZ=True)
    assert TR.word_of(c).id == "AllReduceMultimemCoordinate_v1{WORLD=8,TREE=(((0,1),(2,3)),((4,5),(6,7))),FTZ=True}"
    w = bind(TA.AllReduceMultimem, WORLD=4, SIZE=64, TREES=(TA.chain(4),) * 4, FTZ=False)
    assert SW.boolean_version(w).id == w.id.replace("_v1", "_v2")
    assert TR.word_of(SW.boolean_version(w)).id == w.id


@pytest.mark.parametrize("ftz", [False, True])
def test_coordinate_on_every_special_pair(ftz: bool) -> None:
    grid = np.array(list(itertools.product(SPECIALS, repeat=2)), np.uint16)
    _agree(bind(BA.AllReduceMultimemCoordinate, WORLD=2, TREE=(0, 1), FTZ=ftz), grid)


@pytest.mark.parametrize("tree,ftz", [(CHAIN8, False), (SKEWED8, True)], ids=str)
def test_coordinate_on_rows(tree, ftz: bool) -> None:
    _agree(bind(BA.AllReduceMultimemCoordinate, WORLD=8, TREE=tree, FTZ=ftz), _rows(np.random.default_rng(9), 3000, 8))


def test_rows_on_bits() -> None:
    rng = np.random.default_rng(5)
    trees = (TA.chain(4), ((0, 1), (2, 3)), ((0, 3), (1, 2)), TA.chain(4))
    _agree(bind(BA.AllReduceMultimem, WORLD=4, SIZE=64, TREES=trees, FTZ=True), rng.integers(0, 1 << 16, (40, 4, 64)).astype(np.uint16))


@pytest.mark.parametrize("tree,ftz", [(TA.GLM53_H200_TREE, False), (BALANCED8, True)], ids=str)
def test_vector_on_bits(tree, ftz: bool) -> None:
    _agree(bind(BA.AllReduceMultimemVector, WORLD=8, TREE=tree, FTZ=ftz), _vectors(_rows(np.random.default_rng(13), 256, 8)))


#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AllReduceMultimemCoordinate_v2{WORLD=8,TREE=(((((((0,1),2),3),4),5),6),7),FTZ=False}": (5211, 8375, 918, "54c7f060720f9feec924eccbc9e485f6e03cf6e75b8fdfa7c137b6b6509f133840c50ba5c76f69e8bc1671058824e9ec891a257a800b5b56327fff4ee595a045"),
    "AllReduceMultimemCoordinate_v2{WORLD=8,TREE=(((0,1),(2,3)),((4,5),(6,7))),FTZ=False}": (5013, 7943, 966, "285abf9afb624373a4f2765f7105261a9c66fdbeabc2aa2040d912a99cdbd100324580ace6af3e273ade36042b27b197ab2afc0aece0c7d9668430916fa47df0"),
    "AllReduceMultimemCoordinate_v2{WORLD=8,TREE=(((0,1),(2,3)),((4,5),(6,7))),FTZ=True}": (5233, 7957, 966, "55430db2b0ccb84ab69e7ba4118127c02dcd59703a709407f1c01fa1ad0c7cc96ba4973936db742ee9bf37cccf03424f3433d63a7faa6e086f8e9777a56d5aa2"),
    "AllReduceMultimem_v2{WORLD=4,SIZE=64,TREES=((((0,1),2),3),((0,1),(2,3)),((0,3),(1,2)),(((0,1),2),3)),FTZ=False}": (139648, 219328, 26240, "0933468cc49b8d33246ac16540ff62d89050909ed7c781d1766e7ccb47637bbb5024fea90804788aa33e51d90f0102a8297511c7f4758b9fa76715b339583a6a"),
    "AllReduceMultimemVector_v2{WORLD=8,TREE=((((((0,(1,2)),3),5),6),4),7),FTZ=False}": (41808, 67256, 7456, "1c742d1e4011baa6b030b89c7474d01d5de428900da3af7c1395ff9ac464eb732f854fcc39c87a0ab99748daa6df72e51dd8b064ad4ff6846be0084b5a36fbea"),
    "AllReduceMultimemVector_v2{WORLD=8,TREE=(((0,1),(2,3)),((4,5),(6,7))),FTZ=True}": (41864, 63656, 7728, "1caeeeb4b5b9e1d7db2286df850ce5da9ad2f25093612c6cce14a4e36cbc34381ce57b3c4a4d4886d7e8091a7e0319e75049c41b98a3b95ef2d6d094735f5466"),
}
TARGETS = [bind(BA.AllReduceMultimemCoordinate, WORLD=8, TREE=CHAIN8, FTZ=False),
           bind(BA.AllReduceMultimemCoordinate, WORLD=8, TREE=BALANCED8, FTZ=False),
           bind(BA.AllReduceMultimemCoordinate, WORLD=8, TREE=BALANCED8, FTZ=True),
           bind(BA.AllReduceMultimem, WORLD=4, SIZE=64, TREES=(TA.chain(4), ((0, 1), (2, 3)), ((0, 3), (1, 2)), TA.chain(4)),
                FTZ=False),
           bind(BA.AllReduceMultimemVector, WORLD=8, TREE=TA.GLM53_H200_TREE, FTZ=False),
           bind(BA.AllReduceMultimemVector, WORLD=8, TREE=BALANCED8, FTZ=True)]


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


@pytest.mark.slow
@pytest.mark.parametrize("target", [
    bind(BA.AllReduceMultimem, WORLD=4, SIZE=32, TREES=(((0, 1), (2, 3)),) * 2 + (TA.chain(4),) * 2, FTZ=True),
    bind(BA.AllReduceMultimemVector, WORLD=8, TREE=TA.GLM53_H200_TREE, FTZ=False)], ids=lambda t: t.id)
def test_rows_pass_the_word_querys_cut_check(target) -> None:
    prog, _ = standalone_call(target)
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
