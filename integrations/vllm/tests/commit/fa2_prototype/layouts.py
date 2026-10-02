"""The two commitment layouts over the attention value stream, both built on the imported C0/C1 engine (:mod:`engine`).

Hierarchical (`HierCommitment`)
    head root -> (head, q-block) tile root -> (head, q-block, k-block) SUBTREE root -> leaves of ``leaf_words`` words.
    Every subtree root is built during commit (streaming, O(log n) frontier) and RETAINED (32 B each: the "declared
    per-tile state"); leaves, values and tree levels are discarded.  Opening a value: replay regenerates ONLY the rows of
    its tile up to the challenged k-block, rebuilds that subtree with a tracked path (`StreamingMerkle(track=i)`), and
    asserts the regenerated subtree root equals the retained one; the upper path is read from the retained roots.

Packed / chunk (`PackedCommitment`)
    one stream (head-major, then row, then position) cut into chunks of ``chunk_words`` words; leaf = pos_leaf(chunk bytes).
    Opening a value reveals the WHOLE chunk (its bytes are the authentication payload) plus one path.  Retained: root +
    optionally the chunk digests (32 B per chunk) so that a challenge replays only the rows overlapping the chunk.

What a flat tile hash cannot do
    ``H(tile bytes)`` binds the tile but has no positions: a fine Merkle tree constructed AFTER the challenge has a
    different root, and an opening against it authenticates nothing under the commitment fixed before the challenge
    (`test_flat_hash_is_not_a_tree_root`).  Replay must reproduce authentication under the ORIGINAL root: the hierarchical
    layout does that because its subtree roots existed before the challenge; the packed layout because its chunk leaf is
    the unit that was hashed -- its opening cost is the chunk.

One logical value, many physical computations
    A value that the kernel computes more than once (a split-KV partial recomputed by another CTA, a rescaled accumulator
    re-derived in the combine step, the bf16 word derived from a committed fp32 P) has ONE identity here and ONE position.
    Readers rank the identity; a second physical copy is never a separate leaf, so it cannot carry an independently chosen
    value (every shared gate has one value).  Derived words open through :mod:`derived`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity_vllm.commit import scheme as E
from tests.commit.stream_merkle import StreamingMerkle
from .encoding import STAGE_WIDTH, Geometry, RowLayout, ValueId
from .fixture import AttentionFixture, Context

WORD = 4  # every committed stage is fp32


def layout_digest(kind: str, params: dict[str, Any]) -> bytes:
    return hashlib.sha256(json.dumps({"kind": kind, **params}, sort_keys=True, separators=(",", ":")).encode()).digest()


@dataclass(frozen=True, slots=True)
class Stage:
    """One fold stage of an opening: a tree of ``count`` children in which our node has ``index`` and ``path``."""

    count: int
    index: int
    path: tuple[bytes | None, ...]


@dataclass(frozen=True, slots=True)
class ValueOpening:
    layout: str                      # "hier" | "packed"
    root: bytes                      # the published R the opening claims to authenticate under
    vid: ValueId
    word: bytes                      # the claimed value word (little-endian bit pattern)
    leaf: bytes                      # the whole leaf / chunk bytes that were hashed
    word_offset: int                 # word index inside ``leaf``
    stages: tuple[Stage, ...]        # bottom-up: leaf tree, then the upper trees
    replayed_words: int = 0          # replay work spent to regenerate this opening (prover-side accounting)

    def auth_bytes(self) -> int:
        """Bytes a verifier needs beyond the identity: the leaf/chunk payload + 32 B per sibling."""
        return len(self.leaf) + sum(32 for st in self.stages for s in st.path if s is not None)

    def hash_evals(self) -> int:
        """Hashes the verifier evaluates: the leaf hash, one per path level (lifts included), the root wrapper."""
        return 1 + sum(len(st.path) for st in self.stages) + 1

    def replace(self, **changes: Any) -> ValueOpening:
        import dataclasses

        return dataclasses.replace(self, **changes)


class _LeafPacker:
    """Accumulates words and feeds leaves of ``leaf_words`` words to a `StreamingMerkle`; holds < 1 leaf of pending data."""

    __slots__ = ("leaf_words", "sm", "pending", "n_words", "new", "leaves")

    def __init__(self, leaf_words: int, new: Callable[..., Any], track: int | None = None) -> None:
        self.leaf_words = leaf_words
        self.sm = StreamingMerkle(new, track=track)
        self.pending: list[np.ndarray] = []
        self.n_words = 0
        self.new = new
        self.leaves = 0

    def feed(self, words: np.ndarray) -> None:
        self.n_words += int(words.size)
        self.pending.append(words)
        have = sum(int(w.size) for w in self.pending)
        if have >= self.leaf_words:
            buf = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
            full = (buf.size // self.leaf_words) * self.leaf_words
            for lo in range(0, full, self.leaf_words):
                self.sm.add_digest(E.pos_leaf(buf[lo:lo + self.leaf_words].tobytes(), self.new))
                self.leaves += 1
            self.pending = [buf[full:]] if full < buf.size else []

    def finish(self) -> bytes:
        rest = np.concatenate(self.pending) if len(self.pending) > 1 else (self.pending[0] if self.pending else None)
        if rest is not None and rest.size:
            self.sm.add_digest(E.pos_leaf(rest.tobytes(), self.new))
            self.leaves += 1
        self.pending = []
        return self.sm.finish()


def _leaf_bytes_at(words: np.ndarray, leaf_words: int, leaf_index: int) -> bytes:
    return words[leaf_index * leaf_words:(leaf_index + 1) * leaf_words].tobytes()


# ==================================================================================================================
# hierarchical
# ==================================================================================================================


@dataclass
class _TileGeom:
    head: int
    qb: int
    rows: list[int]                # ascending qrows in this q-block
    nb_max: int                    # k-blocks of the longest row -> subtrees 0..nb_max-1, epilogue = nb_max

    @property
    def n_subtrees(self) -> int:
        return self.nb_max + 1


class HierLayout:
    """Static geometry of the hierarchical layout for a (layer, step): tiles, subtrees, and value -> (leaf, offset)."""

    KIND = "hier"

    def __init__(self, ctx: Context, geo: Geometry, leaf_words: int) -> None:
        if leaf_words < 1:
            raise ValueError("leaf_words >= 1")
        self.ctx, self.geo, self.leaf_words = ctx, geo, leaf_words
        rows = list(ctx.qrows())
        self.qblocks = sorted({r // geo.QB for r in rows})
        self.tiles: dict[tuple[int, int], _TileGeom] = {}
        for h in range(geo.NH):
            for qb in self.qblocks:
                trows = [r for r in rows if r // geo.QB == qb]
                nb_max = max(self.row_layout(h, r).nb for r in trows)
                self.tiles[(h, qb)] = _TileGeom(h, qb, trows, nb_max)
        self.n_words = sum(self.row_layout(h, r).count for h in range(geo.NH) for r in rows)
        self.digest = layout_digest(self.KIND, {"leaf_words": leaf_words, "QB": geo.QB})

    def row_layout(self, head: int, qrow: int) -> RowLayout:
        return RowLayout(self.geo, self.ctx.layer, head, qrow)   # static context: recomputed, never cached (O(nb))

    # -- subtree arithmetic ----------------------------------------------------------------------------------------
    def segment(self, rl: RowLayout, kb: int) -> tuple[int, int] | None:
        """Positions of block kb of a row (None when the row has no block kb); the epilogue is kb == tile.nb_max."""
        if kb < rl.nb:
            return rl.block_span(rl.version_of_block(kb))
        return None

    def subtree_words(self, tile: _TileGeom, kb: int) -> int:
        if kb == tile.nb_max:
            return len(tile.rows)
        return sum((lambda sp: sp[1] - sp[0])(self.segment(self.row_layout(tile.head, r), kb)) for r in tile.rows
                   if self.segment(self.row_layout(tile.head, r), kb) is not None)

    def n_leaves(self, tile: _TileGeom, kb: int) -> int:
        return -(-self.subtree_words(tile, kb) // self.leaf_words)

    def locate(self, vid: ValueId) -> tuple[_TileGeom, int, int, int, int]:
        """-> (tile, subtree index kb, offset of the word in the subtree stream, leaf index, word offset in the leaf)."""
        if vid.layer != self.ctx.layer or vid.qrow not in self.ctx.qrows() or not 0 <= vid.head < self.geo.NH:
            raise KeyError(vid)
        tile = self.tiles[(vid.head, vid.qrow // self.geo.QB)]
        rl = self.row_layout(vid.head, vid.qrow)
        rank = rl.rank(vid)
        if vid.stage == "INV":
            kb = tile.nb_max
            offset = tile.rows.index(vid.qrow)
        else:
            kb = vid.block
            offset = 0
            for r in tile.rows:
                if r == vid.qrow:
                    break
                sp = self.segment(self.row_layout(vid.head, r), kb)
                if sp is not None:
                    offset += sp[1] - sp[0]
            lo, _hi = self.segment(rl, kb)
            offset += rank - lo
        return tile, kb, offset, offset // self.leaf_words, offset % self.leaf_words

    def expected_stages(self, vid: ValueId) -> tuple[list[tuple[int, int]], int]:
        """Verifier-side: (count, index) per stage bottom-up, and the expected leaf length in words."""
        tile, kb, _off, leaf_index, _wo = self.locate(vid)
        n_leaves = self.n_leaves(tile, kb)
        n_words = self.subtree_words(tile, kb)
        leaf_len = min(self.leaf_words, n_words - leaf_index * self.leaf_words)
        tile_index = self.qblocks.index(tile.qb)
        return ([(n_leaves, leaf_index), (tile.n_subtrees, kb), (len(self.qblocks), tile_index), (self.geo.NH, tile.head)], leaf_len)


class HierCommitment:
    """Commit phase + replay-to-open for :class:`HierLayout`.  Retains only R, N and the subtree roots."""

    def __init__(self, layout: HierLayout, new: Callable[..., Any] = hashlib.sha256) -> None:
        self.layout = layout
        self.new = new
        self.subtree_roots: dict[tuple[int, int], list[bytes]] = {}   # (head, qb) -> roots per subtree (kb order, epilogue last)
        self.root: bytes | None = None
        self.peak_pending_words = 0

    # -- commit ------------------------------------------------------------------------------------------------------
    def commit(self, fx: AttentionFixture) -> bytes:
        L = self.layout
        if fx.ctx != L.ctx or fx.geo != L.geo:
            raise ValueError("fixture context/geometry differ from the layout's")
        for (h, qb), tile in L.tiles.items():
            packers = [_LeafPacker(L.leaf_words, self.new) for _ in range(tile.n_subtrees)]
            for r in tile.rows:
                rl = L.row_layout(h, r)
                words, _out = fx.replay(rl, *fx.row_inputs(h, r))
                for kb in range(rl.nb):
                    lo, hi = rl.block_span(rl.version_of_block(kb))
                    packers[kb].feed(words[lo:hi])
                packers[tile.nb_max].feed(words[rl.count - 1:rl.count])
                self.peak_pending_words = max(self.peak_pending_words, sum(sum(int(w.size) for w in p.pending) for p in packers))
                del words  # the row's values are not retained
            self.subtree_roots[(h, qb)] = [p.finish() for p in packers]
            del packers
        self.root = E.step_root(L.ctx.program_digest, L.ctx.digest(), L.geo.digest(), L.digest, L.n_words, self.merkle_root())
        return self.root

    def _upper_levels(self) -> tuple[dict[tuple[int, int], list[list[bytes]]], dict[int, list[list[bytes]]], list[list[bytes]]]:
        L = self.layout
        tile_levels = {k: E.fold_levels(list(v), self.new) for k, v in self.subtree_roots.items()}
        head_levels = {h: E.fold_levels([tile_levels[(h, qb)][-1][0] for qb in L.qblocks], self.new) for h in range(L.geo.NH)}
        top = E.fold_levels([head_levels[h][-1][0] for h in range(L.geo.NH)], self.new)
        return tile_levels, head_levels, top

    def merkle_root(self) -> bytes:
        return self._upper_levels()[2][-1][0]

    def retained_bytes(self) -> int:
        return 32 * sum(len(v) for v in self.subtree_roots.values()) + 32 + 8

    # -- replay-to-open ------------------------------------------------------------------------------------------------
    def open(self, fx: AttentionFixture, vid: ValueId) -> ValueOpening:
        if self.root is None:
            raise RuntimeError("commit first")
        L = self.layout
        tile, kb, _offset, leaf_index, word_off = L.locate(vid)
        packer = _LeafPacker(L.leaf_words, self.new, track=leaf_index)
        chunks: list[np.ndarray] = []
        replayed = 0
        for r in tile.rows:
            rl = L.row_layout(tile.head, r)
            if kb == tile.nb_max:
                words, _ = fx.replay(rl, *fx.row_inputs(tile.head, r))          # INV needs the whole row
                seg = words[rl.count - 1:rl.count]
                replayed += rl.count
            else:
                if kb >= rl.nb:
                    continue
                v = rl.version_of_block(kb)
                words, _ = fx.replay(rl, *fx.row_inputs(tile.head, r), upto_version=v)  # only the blocks visited so far
                lo, hi = rl.block_span(v)
                seg = words[lo:hi]
                replayed += hi
            packer.feed(seg)
            chunks.append(seg)
        sub_root = packer.finish()
        retained = self.subtree_roots[(tile.head, tile.qb)][kb]
        if sub_root != retained:
            raise RuntimeError("replay did not reproduce the retained subtree root (non-deterministic replay?)")
        stream = np.concatenate(chunks)
        leaf = _leaf_bytes_at(stream, L.leaf_words, leaf_index)
        word = leaf[word_off * WORD:(word_off + 1) * WORD]
        tile_levels, head_levels, top = self._upper_levels()
        tile_index = L.qblocks.index(tile.qb)
        stages = (
            Stage(packer.leaves, leaf_index, tuple(packer.sm.path)),
            Stage(tile.n_subtrees, kb, E.path_in_levels(tile_levels[(tile.head, tile.qb)], kb)),
            Stage(len(L.qblocks), tile_index, E.path_in_levels(head_levels[tile.head], tile_index)),
            Stage(L.geo.NH, tile.head, E.path_in_levels(top, tile.head)),
        )
        return ValueOpening("hier", self.root, vid, word, leaf, word_off, stages, replayed)


# ==================================================================================================================
# packed / chunk
# ==================================================================================================================


class PackedLayout:
    KIND = "packed"

    def __init__(self, ctx: Context, geo: Geometry, chunk_words: int) -> None:
        if chunk_words < 1:
            raise ValueError("chunk_words >= 1")
        self.ctx, self.geo, self.chunk_words = ctx, geo, chunk_words
        self.rows = list(ctx.qrows())
        self.row_base: dict[tuple[int, int], int] = {}
        off = 0
        for h in range(geo.NH):
            for r in self.rows:
                self.row_base[(h, r)] = off
                off += self.row_layout(h, r).count
        self.n_words = off
        self.n_chunks = -(-off // chunk_words)
        self.digest = layout_digest(self.KIND, {"chunk_words": chunk_words})

    def row_layout(self, head: int, qrow: int) -> RowLayout:
        return RowLayout(self.geo, self.ctx.layer, head, qrow)

    def offset(self, vid: ValueId) -> int:
        if vid.layer != self.ctx.layer or (vid.head, vid.qrow) not in self.row_base:
            raise KeyError(vid)
        return self.row_base[(vid.head, vid.qrow)] + self.row_layout(vid.head, vid.qrow).rank(vid)

    def chunk_len(self, j: int) -> int:
        return min(self.chunk_words, self.n_words - j * self.chunk_words)

    def expected_stages(self, vid: ValueId) -> tuple[list[tuple[int, int]], int]:
        off = self.offset(vid)
        j = off // self.chunk_words
        return [(self.n_chunks, j)], self.chunk_len(j)

    def rows_overlapping(self, j: int) -> list[tuple[int, int]]:
        lo, hi = j * self.chunk_words, j * self.chunk_words + self.chunk_len(j)
        out = []
        for key, base in self.row_base.items():
            end = base + self.row_layout(*key).count
            if base < hi and end > lo:
                out.append(key)
        return out

    def stream(self, fx: AttentionFixture) -> Iterator[np.ndarray]:
        for h in range(self.geo.NH):
            for r in self.rows:
                words, _ = fx.replay(self.row_layout(h, r), *fx.row_inputs(h, r))
                yield words


class PackedCommitment:
    """Commit + open for :class:`PackedLayout`.  Retains R, N and (optionally) the chunk digests."""

    def __init__(self, layout: PackedLayout, new: Callable[..., Any] = hashlib.sha256, retain_chunk_digests: bool = True) -> None:
        self.layout, self.new, self.retain = layout, new, retain_chunk_digests
        self.root: bytes | None = None
        self.chunk_digests: list[bytes] = []
        self.merkle: bytes | None = None

    def commit(self, fx: AttentionFixture) -> bytes:
        L = self.layout
        if fx.ctx != L.ctx or fx.geo != L.geo:
            raise ValueError("fixture context/geometry differ from the layout's")
        sm = StreamingMerkle(self.new)
        pending: list[np.ndarray] = []
        have = 0
        digests: list[bytes] = []

        def flush(buf: np.ndarray) -> None:
            full = (buf.size // L.chunk_words) * L.chunk_words
            for lo in range(0, full, L.chunk_words):
                d = E.pos_leaf(buf[lo:lo + L.chunk_words].tobytes(), self.new)
                sm.add_digest(d)
                if self.retain:
                    digests.append(d)
            return buf[full:]

        rest = np.zeros(0, dtype=np.uint32)
        for words in L.stream(fx):
            pending.append(words)
            have += words.size
            if have + rest.size >= L.chunk_words:
                rest = flush(np.concatenate([rest, *pending]))
                pending, have = [], 0
        if pending:
            rest = flush(np.concatenate([rest, *pending]))
        if rest.size:
            d = E.pos_leaf(rest.tobytes(), self.new)
            sm.add_digest(d)
            if self.retain:
                digests.append(d)
        self.merkle = sm.finish()
        self.chunk_digests = digests
        self.root = E.step_root(L.ctx.program_digest, L.ctx.digest(), L.geo.digest(), L.digest, L.n_words, self.merkle)
        return self.root

    def retained_bytes(self) -> int:
        return 32 * len(self.chunk_digests) + 32 + 8

    def _chunk_words(self, fx: AttentionFixture, j: int) -> tuple[np.ndarray, int]:
        """Regenerate chunk j: replay only the rows that overlap it (needs the retained chunk digests for the path)."""
        L = self.layout
        lo, hi = j * L.chunk_words, j * L.chunk_words + L.chunk_len(j)
        buf = np.zeros(hi - lo, dtype=np.uint32)
        replayed = 0
        for key in L.rows_overlapping(j):
            base = L.row_base[key]
            words, _ = fx.replay(L.row_layout(*key), *fx.row_inputs(*key))
            replayed += words.size
            a, b = max(lo, base), min(hi, base + words.size)
            buf[a - lo:b - lo] = words[a - base:b - base]
        return buf, replayed

    def open(self, fx: AttentionFixture, vid: ValueId) -> ValueOpening:
        if self.root is None:
            raise RuntimeError("commit first")
        L = self.layout
        off = L.offset(vid)
        j, wo = divmod(off, L.chunk_words)
        if self.retain:
            buf, replayed = self._chunk_words(fx, j)
            if E.pos_leaf(buf.tobytes(), self.new) != self.chunk_digests[j]:
                raise RuntimeError("replay did not reproduce the retained chunk digest")
            levels = E.fold_levels(list(self.chunk_digests), self.new)
            path = E.path_in_levels(levels, j)
            n_leaves = len(self.chunk_digests)
        else:  # roots-only: replay the whole stream with a tracked path
            sm = StreamingMerkle(self.new, track=j)
            replayed = 0
            buf = None
            rest = np.zeros(0, dtype=np.uint32)
            k = 0
            for words in L.stream(fx):
                replayed += words.size
                cur = np.concatenate([rest, words])
                full = (cur.size // L.chunk_words) * L.chunk_words
                for lo in range(0, full, L.chunk_words):
                    if k == j:
                        buf = cur[lo:lo + L.chunk_words].copy()
                    sm.add_digest(E.pos_leaf(cur[lo:lo + L.chunk_words].tobytes(), self.new))
                    k += 1
                rest = cur[full:]
            if rest.size:
                if k == j:
                    buf = rest.copy()
                sm.add_digest(E.pos_leaf(rest.tobytes(), self.new))
                k += 1
            if sm.finish() != self.merkle:
                raise RuntimeError("replay did not reproduce the retained root")
            path, n_leaves = tuple(sm.path), k
        leaf = buf.tobytes()
        word = leaf[wo * WORD:(wo + 1) * WORD]
        return ValueOpening("packed", self.root, vid, word, leaf, wo, (Stage(n_leaves, j, path),), replayed)


# ==================================================================================================================
# verification (the C0 verifier core: pos_leaf + path_shape + fold_path + root wrapper), for both layouts
# ==================================================================================================================


def verify_value_opening(R: bytes, layout: HierLayout | PackedLayout, opening: ValueOpening) -> bool:
    """Authenticate ``opening`` against the published ``R`` using only static context (ctx, geo, layout parameters).

    Checks, all fail-closed: the claimed root and layout kind; the identity is a committed stage located by arithmetic at
    the claimed (leaf, word offset) -- a value presented at another position is rejected; the leaf length is what the
    layout says; the word is the slice of the leaf at that offset; every path has the production `path_shape`; the folds
    reach a Merkle root whose context-bound wrapper equals R."""
    try:
        if opening.root != R or opening.layout != layout.KIND:
            return False
        if opening.vid.stage not in STAGE_WIDTH or STAGE_WIDTH[opening.vid.stage] != WORD:
            return False
        expected, leaf_len = layout.expected_stages(opening.vid)
        if len(opening.stages) != len(expected):
            return False
        for st, (count, index) in zip(opening.stages, expected, strict=True):
            if st.count != count or st.index != index:
                return False
            shape = E.path_shape(count, index)
            if len(st.path) != len(shape):
                return False
            for sib, has in zip(st.path, shape, strict=True):
                if has != (sib is not None) or (sib is not None and len(sib) != 32):
                    return False
        if len(opening.leaf) != leaf_len * WORD:
            return False
        if isinstance(layout, HierLayout):
            _tile, _kb, _off, _li, wo = layout.locate(opening.vid)
        else:
            wo = layout.offset(opening.vid) % layout.chunk_words
        if opening.word_offset != wo or opening.leaf[wo * WORD:(wo + 1) * WORD] != opening.word or len(opening.word) != WORD:
            return False
        digest = E.pos_leaf(opening.leaf)
        for st in opening.stages:
            digest = E.fold_path(digest, st.index, st.path)
        return E.step_root(layout.ctx.program_digest, layout.ctx.digest(), layout.geo.digest(), layout.digest, layout.n_words, digest) == R
    except (KeyError, IndexError, ValueError, TypeError):
        return False
