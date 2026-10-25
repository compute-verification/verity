"""Positive round trips: commit -> discard -> replay -> open -> verify, both layouts, synthetic and archived-geometry fixtures.

Synthetic values establish the MECHANICS (positions, retention, replay-to-open, verification); they say nothing about
numerical correspondence with the FA2 kernel (that is lane B-sem's / `fa2_model.cpp`'s question)."""

from __future__ import annotations

import gc
import sys
import tracemalloc
import weakref
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.commit import scheme
from tests.commit.fa2_prototype.derived import open_derived, verify_derived
from tests.commit.fa2_prototype.encoding import Geometry, ValueId
from tests.commit.fa2_prototype.fixture import GEOMETRY_JSON, archived_geometry_fixture, synthetic_fixture
from tests.commit.fa2_prototype.layouts import (
    HierCommitment,
    HierLayout,
    PackedCommitment,
    PackedLayout,
    verify_value_opening,
)

GEO = Geometry()
INSTANCES = Path("/private/tmp/census/B0-cov_b0_b1_c256/instances.jsonl")


def _sample_ids(layout, fx, rng: np.random.Generator, n: int) -> list[ValueId]:  # noqa: ANN001
    """A spread of identities over heads, rows, stages, blocks (incl. the row that crosses a k-block boundary)."""
    rows = list(fx.ctx.qrows())
    out: list[ValueId] = []
    for _ in range(n):
        h = int(rng.integers(GEO.NH))
        r = int(rng.choice(rows))
        rl = layout.row_layout(h, r)
        out.append(rl.unrank(int(rng.integers(rl.count))))
    # deterministic extremes
    r_last = rows[-1]
    rl = layout.row_layout(0, r_last)
    out += [rl.unrank(0), rl.unrank(rl.count - 1), rl.unrank(rl.count - 2)]
    if rl.nb > 1:
        out.append(ValueId(fx.ctx.layer, 0, r_last, "ALPHA", 0, 0, rl.nb - 1))
        out.append(ValueId(fx.ctx.layer, 0, r_last, "S", 0, 0, rl.nb - 1))
    return out


@pytest.mark.parametrize("leaf_words", [1, 4, 16, 64])
def test_hier_roundtrip_synthetic_prefill(leaf_words: int) -> None:
    fx = synthetic_fixture(seed=7, geo=GEO, layer=2, step=0, prompt_len=131)  # 3 q-blocks; rows 128..130 span 2 k-blocks
    L = HierLayout(fx.ctx, GEO, leaf_words)
    c = HierCommitment(L)
    R = c.commit(fx)
    assert c.retained_bytes() == 32 * sum(len(v) for v in c.subtree_roots.values()) + 40
    rng = np.random.default_rng(leaf_words)
    for vid in _sample_ids(L, fx, rng, 12):
        op = c.open(fx, vid)
        assert verify_value_opening(R, L, op), vid
        assert len(op.leaf) <= 4 * leaf_words
    # commit is deterministic
    assert HierCommitment(HierLayout(fx.ctx, GEO, leaf_words)).commit(fx) == R


@pytest.mark.parametrize("chunk_words", [16, 64, 256, 1024])
def test_packed_roundtrip_synthetic_prefill(chunk_words: int) -> None:
    fx = synthetic_fixture(seed=8, geo=GEO, layer=2, step=0, prompt_len=131)
    L = PackedLayout(fx.ctx, GEO, chunk_words)
    c = PackedCommitment(L)
    R = c.commit(fx)
    assert c.retained_bytes() == 32 * L.n_chunks + 40
    rng = np.random.default_rng(chunk_words)
    for vid in _sample_ids(L, fx, rng, 12):
        op = c.open(fx, vid)
        assert verify_value_opening(R, L, op), vid
        assert len(op.leaf) == 4 * L.chunk_len(L.offset(vid) // chunk_words)
    # roots-only variant: same root, whole-stream replay, same verifier
    c2 = PackedCommitment(PackedLayout(fx.ctx, GEO, chunk_words), retain_chunk_digests=False)
    assert c2.commit(fx) == R
    vid = _sample_ids(L, fx, np.random.default_rng(1), 1)[0]
    op2 = c2.open(fx, vid)
    assert op2.replayed_words == L.n_words and verify_value_opening(R, L, op2)


def test_roundtrip_decode_step_both_layouts() -> None:
    fx = synthetic_fixture(seed=9, geo=GEO, layer=17, step=5, prompt_len=155)  # one row per head, T = 160 -> 2 k-blocks
    assert list(fx.ctx.qrows()) == [159]
    for layout, C in ((HierLayout(fx.ctx, GEO, 4), HierCommitment), (PackedLayout(fx.ctx, GEO, 128), PackedCommitment)):
        c = C(layout)
        R = c.commit(fx)
        rl = layout.row_layout(4, 159)
        for i in (0, 1, rl.count // 2, rl.count - 2, rl.count - 1):
            vid = rl.unrank(i)
            assert verify_value_opening(R, layout, c.open(fx, vid)), (layout.KIND, vid)


def test_derived_p16_opens_through_p32() -> None:
    fx = synthetic_fixture(seed=3, geo=GEO, layer=0, step=0, prompt_len=40)
    L = HierLayout(fx.ctx, GEO, 4)
    c = HierCommitment(L)
    R = c.commit(fx)
    d = open_derived(c, fx, ValueId(0, 2, 33, "P16", 0, 21, 0))
    assert verify_derived(R, L, d)
    assert len(d.word) == 2 and d.source.vid.stage == "P32"


def test_archived_geometry_fixture_roundtrip() -> None:
    """The REAL geometry of layer 0 of the accepted record (155 prefill rows T = 1..155 at step 0; one row T = 160 at
    step 5), values synthetic: the compact record references runtime tensors, it does not store them."""
    src = INSTANCES if INSTANCES.exists() else None
    fx0 = archived_geometry_fixture(src, layer=0, step=0)
    assert fx0.values == "synthetic" and fx0.source["kind"] == "archived-geometry"
    assert fx0.ctx.prompt_len == 155 and fx0.geo == Geometry(D=64, BN=128, QB=64, NH=9, KVH=3)
    assert fx0.ctx.program_digest.hex().startswith("5bb5ac55c1ee1101")
    L = HierLayout(fx0.ctx, GEO, 16)
    c = HierCommitment(L)
    R = c.commit(fx0)
    assert L.n_words == 9 * sum(2 * T + ((T + 127) // 128) * 70 for T in range(1, 156))
    for vid in _sample_ids(L, fx0, np.random.default_rng(0), 8):
        assert verify_value_opening(R, L, c.open(fx0, vid)), vid
    fx5 = archived_geometry_fixture(src, layer=0, step=5)
    assert list(fx5.ctx.qrows()) == [159]
    P = PackedLayout(fx5.ctx, GEO, 256)
    pc = PackedCommitment(P)
    R5 = pc.commit(fx5)
    for vid in _sample_ids(P, fx5, np.random.default_rng(0), 6):
        assert verify_value_opening(R5, P, pc.open(fx5, vid)), vid
    assert GEOMETRY_JSON.exists()


#: The code a commit runs.  A block counts only when one of these files is on the stack that allocated it, so nothing another
#: thread or an earlier test allocates is counted.
COMMIT_CODE = (str(Path(__file__).parent / "fa2_prototype" / "*"), str(Path(__file__).parent / "stream_merkle.py"), scheme.__file__)
#: Frames tracemalloc keeps per block: enough for every block the commit makes to reach a frame of `COMMIT_CODE`.
COMMIT_FRAMES = 16


def _commit_resident(snapshot: tracemalloc.Snapshot) -> int:
    """Bytes of `snapshot`'s blocks that commit code allocated.  numpy's own code keeps a varying number of small blocks after
    `np.cumsum` (raw memory, not objects anything references; the count differs run to run), so a Python-heap block counts
    only when numpy's code did not make it.  Array buffers (numpy's tracemalloc domain) always count."""
    np_dir = str(Path(np.__file__).parent)
    traces = snapshot.filter_traces([tracemalloc.Filter(True, p, all_frames=True) for p in COMMIT_CODE]).traces
    return sum(t.size for t in traces if t.domain != 0 or not t.traceback[-1].filename.startswith(np_dir))


@pytest.mark.slow
def test_transient_storage_is_released() -> None:
    """After commit only roots + declared per-tile state remain: the fixture's arrays and every leaf/level are gone."""
    # A first commit pays every first use (numpy's dispatch caches, lazy imports), so the measured one counts the same bytes
    # whatever ran before it in the process.  The layout is static context, built before the measurement.
    warm = synthetic_fixture(seed=12, geo=GEO, layer=1, step=0, prompt_len=96)
    HierCommitment(HierLayout(warm.ctx, GEO, 4)).commit(warm)
    fx = synthetic_fixture(seed=11, geo=GEO, layer=1, step=0, prompt_len=96)
    L = HierLayout(fx.ctx, GEO, 4)
    c = HierCommitment(L)
    gc.collect()
    tracemalloc.start(COMMIT_FRAMES)
    base = tracemalloc.take_snapshot()
    R = c.commit(fx)
    ref_q = weakref.ref(fx.q16)
    fx_words = int(fx.q16.nbytes + fx.k16.nbytes + fx.v16.nbytes)
    del fx
    gc.collect()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    assert ref_q() is None, "the fixture arrays are still referenced"
    grown = _commit_resident(after) - _commit_resident(base)
    retained = c.retained_bytes()
    n_sub = sum(len(v) for v in c.subtree_roots.values())
    assert retained == 32 * n_sub + 40
    # What may remain is what the commitment declares: R and the subtree roots, one list per tile keyed by (head, q-block),
    # plus numpy's shape cache (at most 7 blocks of each size: 16, 32 and 48 B for the commit's 1-, 2- and 3-D arrays, so under
    # 1 KiB).  The committed stream (4 * N bytes), the leaves, the tree levels and the fixture must not be resident.
    declared = sys.getsizeof(R) + sys.getsizeof(c.subtree_roots) + sum(
        sys.getsizeof(k) + sys.getsizeof(v) + sum(map(sys.getsizeof, v)) for k, v in c.subtree_roots.items())
    assert grown <= declared + 1024, (grown, declared, n_sub, L.n_words * 4, fx_words)
    assert grown < L.n_words * 4 // 8
    assert len(R) == 32


def test_production_verifier_authenticates_the_same_bytes() -> None:
    """C0 §C0.2-style cross-check: the words a position opening authenticates are exactly the bytes the production
    id-carrying leaf (`hashing.leaf_hash`, `MerkleTree`, `verify_opening`) authenticates for the same identity."""
    from verity_vllm.observe.capture_v1 import Value
    from verity_vllm.commit.merkle import MerkleTree, verify_opening

    fx = synthetic_fixture(seed=5, geo=GEO, layer=4, step=3, prompt_len=30)   # decode: one row per head, T = 32
    L = HierLayout(fx.ctx, GEO, 1)
    c = HierCommitment(L)
    R = c.commit(fx)
    rl = L.row_layout(0, 32)
    from tests.commit.fa2_prototype.reference import replay_row

    words, _ = replay_row(rl, *fx.row_inputs(0, 32))
    values = [Value(v.canonical_id("a594a58ec69b", "req", "decode"), "float32", (), words[i:i + 1].tobytes()) for i, v in enumerate(rl.ids())]
    tree = MerkleTree.from_values(values, "ru/req/000003")
    for i in (0, 5, rl.count // 2, rl.count - 1):
        vid = rl.unrank(i)
        mine = c.open(fx, vid)
        assert verify_value_opening(R, L, mine)
        prod = tree.open(values[i])
        assert verify_opening(tree.root, tree.leaf_count, prod)
        assert prod.value == mine.word and prod.canonical_id == vid.canonical_id("a594a58ec69b", "req", "decode")
