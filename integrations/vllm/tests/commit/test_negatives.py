"""Negatives: every forgery below is rejected by the verifier under the root fixed before the challenge."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from verity_vllm.commit import hidden_engine as E
from tests.commit.fa2_prototype.derived import DerivedOpening, open_derived, verify_derived
from tests.commit.fa2_prototype.encoding import Geometry, ValueId
from tests.commit.fa2_prototype.fixture import AttentionFixture, Context, synthetic_fixture
from tests.commit.fa2_prototype.layouts import (
    HierCommitment,
    HierLayout,
    PackedCommitment,
    PackedLayout,
    Stage,
    verify_value_opening,
)
from tests.commit.fa2_prototype.reference import replay_row

GEO = Geometry()


@pytest.fixture(scope="module")
def committed():  # noqa: ANN201
    fx = synthetic_fixture(seed=21, geo=GEO, layer=5, step=0, prompt_len=140, request_id="req-A")
    HL = HierLayout(fx.ctx, GEO, 4)
    hc = HierCommitment(HL)
    RH = hc.commit(fx)
    PL = PackedLayout(fx.ctx, GEO, 64)
    pc = PackedCommitment(PL)
    RP = pc.commit(fx)
    return fx, (HL, hc, RH), (PL, pc, RP)


VID = ValueId(5, 4, 133, "P32", 1, 130, 0)     # row 133 spans 2 k-blocks; block 1 is visited first (version 0)
VID_O = ValueId(5, 4, 133, "O", 0, 17, 1)      # accumulator after the second visited block


def _flip(word: bytes, bit: int = 0) -> bytes:
    b = bytearray(word)
    b[0] ^= 1 << bit
    return bytes(b)


@pytest.mark.parametrize("which", ["hier", "packed"])
def test_changed_value_rejected(committed, which):  # noqa: ANN001, ANN201
    fx, H, P = committed
    L, c, R = H if which == "hier" else P
    op = c.open(fx, VID)
    assert verify_value_opening(R, L, op)
    # flip the claimed word only
    assert not verify_value_opening(R, L, op.replace(word=_flip(op.word)))
    # flip the word inside the leaf bytes too (consistent forgery of leaf + word): the leaf hash changes
    leaf = bytearray(op.leaf)
    leaf[op.word_offset * 4] ^= 1
    assert not verify_value_opening(R, L, op.replace(word=bytes(leaf[op.word_offset * 4:op.word_offset * 4 + 4]), leaf=bytes(leaf)))
    # flip a NEIGHBOUR word of the leaf/chunk (the claimed word untouched): still rejected -- the leaf is the unit hashed
    if len(op.leaf) > 4:
        other = (op.word_offset + 1) % (len(op.leaf) // 4)
        leaf = bytearray(op.leaf)
        leaf[other * 4] ^= 1
        assert not verify_value_opening(R, L, op.replace(leaf=bytes(leaf)))


@pytest.mark.parametrize("which", ["hier", "packed"])
def test_wrong_position_rejected(committed, which):  # noqa: ANN001, ANN201
    """A genuine opening of position i presented as identity j (same row, other column / stage / block / version)."""
    fx, H, P = committed
    L, c, R = H if which == "hier" else P
    op = c.open(fx, VID)
    for other in (
        dataclasses.replace(VID, index=131),                 # neighbouring key column
        dataclasses.replace(VID, stage="S"),                 # the score at the same column
        ValueId(5, 4, 133, "P32", 0, 2, 1),                  # other k-block
        ValueId(5, 4, 132, "P32", 1, 130, 0),                # other query row
        ValueId(5, 3, 133, "P32", 1, 130, 0),                # other head
        dataclasses.replace(VID, version=1),                 # inconsistent online-state version
        dataclasses.replace(VID, layer=6),                   # other layer
    ):
        assert not verify_value_opening(R, L, op.replace(vid=other)), other
    # a word from the same leaf presented at another offset
    if len(op.leaf) > 4:
        assert not verify_value_opening(R, L, op.replace(word_offset=(op.word_offset + 1) % (len(op.leaf) // 4)))
    # a path of the right length taken from another leaf / chunk (open another value, keep our leaf)
    op2 = c.open(fx, ValueId(5, 1, 60, "S", 0, 7, 0) if which == "packed" else VID_O)
    assert op2.stages[0].index != op.stages[0].index
    assert not verify_value_opening(R, L, op.replace(stages=op2.stages))
    # index claims outside the tree
    st0 = op.stages[0]
    assert not verify_value_opening(R, L, op.replace(stages=(Stage(st0.count, st0.count, st0.path),) + op.stages[1:]))


def test_reordered_or_substituted_chunk_rejected(committed):  # noqa: ANN001, ANN201
    fx, _H, (PL, pc, RP) = committed
    a = pc.open(fx, VID)
    b = pc.open(fx, ValueId(5, 1, 60, "S", 0, 7, 0))     # a chunk elsewhere in the stream
    ja, jb = a.stages[0].index, b.stages[0].index
    assert ja != jb
    # chunk b's bytes under chunk a's position/path
    assert not verify_value_opening(RP, PL, a.replace(leaf=b.leaf, word=b.leaf[a.word_offset * 4:a.word_offset * 4 + 4]))
    # chunk a's bytes with chunk b's path (substituted authentication path)
    assert not verify_value_opening(RP, PL, a.replace(stages=b.stages))
    # words reordered inside the chunk (the claimed word kept in place)
    w = np.frombuffer(a.leaf, dtype=np.uint32).copy()
    i, j = (a.word_offset + 1) % w.size, (a.word_offset + 2) % w.size
    if i != a.word_offset and j != a.word_offset and w[i] != w[j]:
        w[i], w[j] = w[j], w[i]
        assert not verify_value_opening(RP, PL, a.replace(leaf=w.tobytes()))
    # a shorter / longer chunk
    assert not verify_value_opening(RP, PL, a.replace(leaf=a.leaf[:-4]))
    assert not verify_value_opening(RP, PL, a.replace(leaf=a.leaf + bytes(4)))


def test_hier_substituted_subtree_rejected(committed):  # noqa: ANN001, ANN201
    """A leaf+path of the same shape from ANOTHER k-block subtree / tile / head under our identity."""
    fx, (HL, hc, RH), _P = committed
    a = hc.open(fx, VID)
    for other in (ValueId(5, 4, 133, "P32", 0, 2, 1),      # other k-block (subtree) of the same tile
                  ValueId(5, 4, 69, "P32", 0, 2, 0),        # other q-block (tile)
                  ValueId(5, 8, 133, "P32", 1, 130, 0)):    # other head
        b = hc.open(fx, other)
        forged = a.replace(leaf=b.leaf, word=b.leaf[a.word_offset * 4:a.word_offset * 4 + 4], stages=b.stages)
        assert not verify_value_opening(RH, HL, forged), other
        # mixing: our leaf-level stage with their upper stages
        forged2 = a.replace(stages=(a.stages[0],) + b.stages[1:])
        assert not verify_value_opening(RH, HL, forged2), other


@pytest.mark.parametrize("which", ["hier", "packed"])
def test_wrong_context_rejected(committed, which):  # noqa: ANN001, ANN201
    """The same values, committed for another request / step / Program: openings do not transfer across contexts."""
    fx, H, P = committed
    L, c, R = H if which == "hier" else P
    op = c.open(fx, VID)
    ctx = fx.ctx
    for other_ctx in (
        dataclasses.replace(ctx, request_id="req-B"),
        dataclasses.replace(ctx, step=1, prompt_len=139),          # decode step 1 of a 139-token prompt has the same row 139? no: row 139 only
        dataclasses.replace(ctx, program_digest=hashlib.sha256(b"other program").digest()),
        dataclasses.replace(ctx, layer=6),
    ):
        other_fx = AttentionFixture(other_ctx, fx.geo, fx.q16, fx.k16, fx.v16)
        try:
            L2 = (HierLayout(other_ctx, GEO, 4) if which == "hier" else PackedLayout(other_ctx, GEO, 64))
            c2 = (HierCommitment if which == "hier" else PackedCommitment)(L2)
            R2 = c2.commit(other_fx)
        except (IndexError, ValueError):
            continue
        assert R2 != R
        # our opening against the other context's root / layout
        assert not verify_value_opening(R2, L2, op)
        assert not verify_value_opening(R2, L, op)
        assert not verify_value_opening(R, L2, op)
        # the other context's genuine opening of the same identity (if it has one) against OUR root
        try:
            op2 = c2.open(other_fx, VID)
        except (KeyError, IndexError):
            continue
        assert verify_value_opening(R2, L2, op2)
        assert not verify_value_opening(R, L, op2)
    # a root wrapper over the right Merkle root but another geometry declaration
    other_geo = Geometry(D=64, BN=128, QB=32, NH=9, KVH=3)
    L3 = HierLayout(ctx, other_geo, 4) if which == "hier" else PackedLayout(ctx, other_geo, 64)
    assert not verify_value_opening(R, L3, op)


def test_inconsistent_openings_of_a_shared_value(committed):  # noqa: ANN001, ANN201
    """One identity, one position, one value: two openings claiming different words cannot both verify, whether the
    reader is the row-sum consumer, the bf16 consumer, or a recomputation elsewhere (split partial / another CTA)."""
    fx, (HL, hc, RH), _P = committed
    op = hc.open(fx, VID)
    forged = op.replace(word=_flip(op.word, 3))
    assert verify_value_opening(RH, HL, op) and not verify_value_opening(RH, HL, forged)
    # the derived bf16 consumer is forced to F2fpBf16 of the same committed P32
    d = open_derived(hc, fx, dataclasses.replace(VID, stage="P16"))
    assert verify_derived(RH, HL, d)
    assert not verify_derived(RH, HL, d.replace(word=_flip(d.word)))                      # other bf16 word
    assert not verify_derived(RH, HL, d.replace(source=forged))                            # inconsistent source
    assert not verify_derived(RH, HL, d.replace(conversion="Bf16ToF32"))                  # undeclared conversion
    assert not verify_derived(RH, HL, d.replace(vid=dataclasses.replace(d.vid, index=131)))  # derived id / source mismatch
    # a "second physical representation" cannot be committed: P16 has no position in any layout
    with pytest.raises(KeyError):
        HL.locate(dataclasses.replace(VID, stage="P16"))
    with pytest.raises(KeyError):
        HierLayout(fx.ctx, GEO, 4).row_layout(4, 133).rank(dataclasses.replace(VID, stage="P16"))
    # a DerivedOpening whose source is a genuine opening of ANOTHER identity
    other = hc.open(fx, dataclasses.replace(VID, index=131))
    assert not verify_derived(RH, HL, DerivedOpening(d.vid, d.word, d.conversion, other))


def test_flat_hash_is_not_a_tree_root(committed):  # noqa: ANN001, ANN201
    """A flat hash of a tile binds the tile but cannot later serve as the root of a fine Merkle tree built at challenge
    time: the fine root differs, and the only opening the flat hash admits is the whole tile (the packed layout's cost)."""
    fx, (HL, hc, RH), _P = committed
    rl = HL.row_layout(4, 133)
    words, _ = replay_row(rl, *fx.row_inputs(4, 133))
    flat = E.pos_leaf(words.tobytes())                           # what a flat "tile hash" commitment would publish
    # a fine tree over the same words built after the fact
    leaves = [E.pos_leaf(words[i:i + 1].tobytes()) for i in range(words.size)]
    fine_root = E.ft.fold(list(leaves))
    assert fine_root != flat
    # a word-level path against the flat hash fails; the same path against the fine root verifies -- but that root was
    # not fixed before the challenge, so it authenticates nothing under the commitment
    i = rl.rank(VID)
    levels = E.ft.fold_levels(list(leaves))
    path = E.path_in_levels(levels, i)
    assert E.fold_path(leaves[i], i, path) == fine_root
    assert E.fold_path(leaves[i], i, path) != flat
    # the hierarchical layout's fine subtrees DID exist before the challenge: the replayed subtree root equals the
    # retained one, and the opening verifies under RH (the original root)
    op = hc.open(fx, VID)
    assert verify_value_opening(RH, HL, op)


def test_context_row_set_mismatch_rejected():  # noqa: ANN201
    """An opening for a row the context does not contain (prompt length lies) is rejected at location time."""
    fx = synthetic_fixture(seed=2, geo=GEO, layer=0, step=0, prompt_len=20)
    L = HierLayout(fx.ctx, GEO, 4)
    c = HierCommitment(L)
    R = c.commit(fx)
    op = c.open(fx, ValueId(0, 0, 19, "S", 0, 3, 0))
    assert verify_value_opening(R, L, op)
    shorter = HierLayout(Context(fx.ctx.program_digest, fx.ctx.request_id, 0, 19, 0), GEO, 4)
    assert not verify_value_opening(R, shorter, op)
