"""One-stage commitment of the FA2 *hidden* value stream (scores, P, running max, rescale, lane
sums, PV accumulators, epilogue) as acquired from the pinned kernel by the v7 taps:

* **M1 stream** (`VERITY_TAP=3`): the kernel materialises the selected classes into one flat u32 buffer whose layout is
  arithmetic in (HB = b*h slabs, M = seqlen_q rows, NB = key-block cap) -- `StreamLayout` below mirrors
  `Verity_mat_layout` in `verity_tap.h`; every canonical (class, slab, row, col | block, lane | d) position is one word.
  The commitment is a Merkle tree (production node/lift rule, `verity_vllm.commit.scheme`) over
  fixed-size chunks of that buffer; leaf = SHA-256(16-word header || chunk words) where the header carries the chunk
  index, the launch tag and the geometry, so a leaf is bound to its position, not to a multiset.
* **H1 digests** (`VERITY_TAP=4`): every kernel thread hashes, per visited 128-key block, its own register words with a
  real SHA-256 behind the same 16-word header (slab, m_block, n_block, tidx, seqlens, launch tag, n_block_max, row
  validity); only the 32-byte digests leave the SM.  `ThreadLeafCommitment` builds the tree over those digests; an
  opening reveals the thread leaf's message (regenerated on replay from the M1 stream / the Definition) plus a path.

Validity conditions: (1) binding under SHA-256 (leaf) + the production Merkle rule (nodes); (2) the leaf
preimage contains the position header the verifier derives itself from the claimed identity, and a word's offset inside
the leaf is fixed by the layout -- identity/position, not a multiset; (3) openings fold to the *prechallenge* root;
(4) `test_hidden_stream.py` rejects altered / reordered / substituted words, substituted leaves, wrong launch tag.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity_vllm.commit import scheme

TAG = scheme.FA2H_TAG
VERSION = scheme.FA2H_VERSION
KIND_CHUNK = scheme.FA2H_KIND_CHUNK
KIND_THREAD = 2             # per-thread per-block leaf computed inside the kernel (H1)
KBLOCK_N = 128
KBLOCK_M = 64
CLASSES = ("S", "P", "RP", "ROW", "O", "FIN")
#: MS (64): the seventh class, appended after FIN only by the guarded-max tap builds (`VERITY_MAT_SRC & 64`, `hidden_source.TAP_ROW3`): one f32
#: per (slab, row, key block), the `max * scale` that block's exp2 FMAs read.  Without it the layout, digest and bytes are the six classes'.
SRC_BITS = {"S": 1, "P": 2, "RP": 4, "ROW": 8, "O": 16, "FIN": 32, "MS": 64}
#: ROW word 3: never written, except by the guarded-max tap builds (`verity_tap_row3() == 1`, `hidden_source.TAP_ROW3`), which store
#: GuardNegInfZero(row_max) there at every key block after a row's first and 0 at the first
ROW_GUARD_FIELD = 3


@dataclass(frozen=True)
class StreamLayout:
    """Word offsets of the M1 stream (mirror of `Verity_mat_layout(HB, M, NB, BN, D)`).

    Tap layout v8: the layout is arithmetic in the kernel tile -- `BN` = kBlockN of the launched kernel (128 for head
    dim 64, 64 for 96/128/256 on the num_splits == 1 kernel; `hidden_source.fa2_kblock_n`) and `D` = head_dim (the O plane's row
    width).  The defaults reproduce the legacy hd-64 layout, digest and chunk header word for word; other tiles enter the digest and the
    spare header words 8/9, so a chunk of a (64, 128) stream can never verify against a (128, 64) declaration.  `ms` appends the MS
    class after FIN (and enters the digest); without it every offset, the total and the digest are the six classes'."""

    HB: int
    M: int
    NB: int
    BN: int = KBLOCK_N
    D: int = 64
    ms: bool = False

    @property
    def N(self) -> int:
        return self.NB * self.BN

    @property
    def offsets(self) -> dict[str, int]:
        HB, M, N, NB, D = self.HB, self.M, self.N, self.NB, self.D
        off_s = 0
        off_p = off_s + HB * M * N
        off_rp = off_p + HB * M * N
        off_row = off_rp + HB * M * (N // 2)
        off_o = off_row + HB * M * NB * 8
        off_fin = off_o + HB * M * NB * D
        o = {"S": off_s, "P": off_p, "RP": off_rp, "ROW": off_row, "O": off_o, "FIN": off_fin, "total": off_fin + HB * M * 4}
        if self.ms:
            o["MS"] = o["total"]
            o["total"] = o["MS"] + HB * M * NB
        return o

    @property
    def legacy_tile(self) -> bool:
        return (self.BN, self.D) == (KBLOCK_N, 64)

    @property
    def total_words(self) -> int:
        return self.offsets["total"]

    def padded_words(self, chunk_words: int = 64) -> int:
        """Stream length rounded up to whole chunks (the tail chunk is zero-padded; padding words are outside the layout)."""
        return -(-self.total_words // chunk_words) * chunk_words

    def word_index(self, cls: str, slab: int, row: int, *, col: int | None = None, block: int | None = None,
                   lane: int | None = None, d: int | None = None, field: int | None = None) -> int:
        """Canonical position -> word index.  ROW fields: 0 row_max, 1 rescale, 2 step, 3 the guarded max (`ROW_GUARD_FIELD`),
        4+lane l_lane.  FIN: 0 sum, 1 inv_sum, 2 final row_max, 3 lse.  MS (with `ms`): one word per block."""
        o = self.offsets
        if not (0 <= slab < self.HB and 0 <= row < self.M):
            raise ValueError("position outside the layout")
        if cls in ("S", "P"):
            assert col is not None and 0 <= col < self.N
            return o[cls] + (slab * self.M + row) * self.N + col
        if cls == "RP":
            assert col is not None and 0 <= col < self.N
            return o["RP"] + (slab * self.M + row) * (self.N // 2) + col // 2   # word holds cols (c, c+1); bf16 halves
        if cls == "ROW":
            assert block is not None and 0 <= block < self.NB
            f = field if field is not None else 4 + lane
            return o["ROW"] + ((slab * self.M + row) * self.NB + block) * 8 + f
        if cls == "O":
            assert block is not None and d is not None and 0 <= d < self.D
            return o["O"] + ((slab * self.M + row) * self.NB + block) * self.D + d
        if cls == "FIN":
            return o["FIN"] + (slab * self.M + row) * 4 + field
        if cls == "MS" and self.ms:
            assert block is not None and 0 <= block < self.NB
            return o["MS"] + (slab * self.M + row) * self.NB + block
        raise ValueError(cls)

    def unrank(self, word: int) -> dict[str, Any]:
        o = self.offsets
        if word >= o["total"] or word < 0:
            raise ValueError("word outside the layout")
        if word < o["P"]:
            i = word - o["S"]; sr, col = divmod(i, self.N); slab, row = divmod(sr, self.M)
            return {"cls": "S", "slab": slab, "row": row, "col": col}
        if word < o["RP"]:
            i = word - o["P"]; sr, col = divmod(i, self.N); slab, row = divmod(sr, self.M)
            return {"cls": "P", "slab": slab, "row": row, "col": col}
        if word < o["ROW"]:
            i = word - o["RP"]; sr, cw = divmod(i, self.N // 2); slab, row = divmod(sr, self.M)
            return {"cls": "RP", "slab": slab, "row": row, "col": 2 * cw}
        if word < o["O"]:
            i = word - o["ROW"]; srb, f = divmod(i, 8); sr, block = divmod(srb, self.NB); slab, row = divmod(sr, self.M)
            return {"cls": "ROW", "slab": slab, "row": row, "block": block, "field": f}
        if word < o["FIN"]:
            i = word - o["O"]; srb, d = divmod(i, self.D); sr, block = divmod(srb, self.NB); slab, row = divmod(sr, self.M)
            return {"cls": "O", "slab": slab, "row": row, "block": block, "d": d}
        if self.ms and word >= o["MS"]:
            sr, block = divmod(word - o["MS"], self.NB); slab, row = divmod(sr, self.M)
            return {"cls": "MS", "slab": slab, "row": row, "block": block}
        i = word - o["FIN"]; sr, f = divmod(i, 4); slab, row = divmod(sr, self.M)
        return {"cls": "FIN", "slab": slab, "row": row, "field": f}

    def digest(self) -> bytes:
        d = {"kind": "fa2-hidden-m1", "v": VERSION, "HB": self.HB, "M": self.M, "NB": self.NB}
        if not self.legacy_tile:
            d.update({"BN": self.BN, "D": self.D})
        if self.ms:
            d["ms"] = True
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).digest()


# ------------------------------------------------------------------ leaf framing (shared with the kernel's H1 leaf)
def pack_words(words: list[int]) -> bytes:
    return struct.pack("<%dI" % len(words), *[w & 0xFFFFFFFF for w in words])


def thread_of(row: int, col_or_d: int) -> tuple[int, int, int]:
    """(tidx, row_slot mi, quad lane t) owning value (row, col) of a tile: rows 16w+g (+8), cols 8n+2t+{0,1}."""
    r = row % KBLOCK_M
    w, g, mi = r // 16, (r % 16) % 8, (r % 16) // 8
    t = (col_or_d % 8) // 2
    return 32 * w + 4 * g + t, mi, t


# ------------------------------------------------------------------ M1 chunk commitment
@dataclass(frozen=True)
class ChunkOpening:
    index: int                 # leaf (chunk) index
    offset: int                # word offset inside the chunk
    chunk: bytes               # the chunk's words (little-endian u32)
    path: tuple[bytes | None, ...]

    @property
    def word(self) -> int:
        return struct.unpack_from("<I", self.chunk, 4 * self.offset)[0]


@dataclass
class StreamCommitment:
    layout: StreamLayout
    launch_tag: int
    src_mask: int
    chunk_words: int
    leaf_count: int
    merkle_root: bytes
    root: bytes                       # context-bound root
    levels: list[list[bytes]] | None  # retained for openings (prover side); the verifier needs only `root`
    timing: dict[str, float]

    @property
    def retained_bytes(self) -> int:
        return 32 * sum(len(l) for l in self.levels) if self.levels else 32

    def open(self, stream: np.ndarray, word_index: int) -> ChunkOpening:
        ci, off = divmod(word_index, self.chunk_words)
        chunk = stream[ci * self.chunk_words:(ci + 1) * self.chunk_words].astype(np.uint32).tobytes()
        assert self.levels is not None
        return ChunkOpening(ci, off, chunk, scheme.path_in_levels(self.levels, ci))


def commit_stream(stream: np.ndarray, layout: StreamLayout, launch_tag: int, src_mask: int, chunk_words: int = 64,
                  retain_levels: bool = True) -> StreamCommitment:
    """Host reference commitment (hashlib SHA-256) over the whole M1 stream, chunk leaves in layout order."""
    stream = np.ascontiguousarray(stream, dtype=np.uint32)
    padded = layout.padded_words(chunk_words)
    if stream.size == layout.total_words and padded != stream.size:
        stream = np.concatenate([stream, np.zeros(padded - stream.size, dtype=np.uint32)])
    if stream.size != padded:
        raise ValueError(f"stream has {stream.size} words, layout says {layout.total_words} (padded {padded})")
    n = stream.size // chunk_words
    mv = memoryview(stream.tobytes())
    t0 = time.perf_counter()
    cb = 4 * chunk_words
    leaves = [scheme.chunk_leaf(scheme.chunk_header(layout, launch_tag, i, chunk_words, src_mask), mv[i * cb:(i + 1) * cb]) for i in range(n)]
    t1 = time.perf_counter()
    levels = scheme.fold_levels(list(leaves))
    t2 = time.perf_counter()
    merkle_root = levels[-1][0]
    root = scheme.stream_root(layout.digest(), launch_tag, src_mask, chunk_words, n, merkle_root)
    return StreamCommitment(layout, launch_tag, src_mask, chunk_words, n, merkle_root, root, levels if retain_levels else None,
                            {"leaf_hash_s": t1 - t0, "tree_fold_s": t2 - t1, "leaf_count": n, "bytes": stream.size * 4})


def verify_chunk_opening(root: bytes, layout: StreamLayout, launch_tag: int, src_mask: int, chunk_words: int, leaf_count: int,
                         word_index: int, opening: ChunkOpening) -> bool:
    """Verifier: derive index/offset/header from the claimed position, recompute the leaf, fold to the root."""
    ci, off = divmod(word_index, chunk_words)
    if opening.index != ci or opening.offset != off or len(opening.chunk) != 4 * chunk_words:
        return False
    if len(opening.path) != len(scheme.path_shape(leaf_count, ci)):
        return False
    leaf = scheme.chunk_leaf(scheme.chunk_header(layout, launch_tag, ci, chunk_words, src_mask), opening.chunk)
    mr = scheme.fold_path(leaf, ci, opening.path)
    return scheme.stream_root(layout.digest(), launch_tag, src_mask, chunk_words, leaf_count, mr) == root


# ------------------------------------------------------------------ H1 thread-leaf commitment
@dataclass(frozen=True)
class ThreadGeometry:
    HB: int
    b: int
    h: int
    M: int
    NB: int
    num_m_block: int
    seqlens_q: tuple[int, ...]
    seqlens_k: tuple[int, ...]
    causal: bool

    def n_block_max(self, seq_idx: int, m_block: int) -> int:
        sk, sq = self.seqlens_k[seq_idx], self.seqlens_q[seq_idx]
        nb = (sk + KBLOCK_N - 1) // KBLOCK_N
        if self.causal:
            nb = min(nb, ((m_block + 1) * KBLOCK_M + sk - sq + KBLOCK_N - 1) // KBLOCK_N)
        return max(nb, 0)

    def leaf_index(self, slab: int, m_block: int, n_block: int, tidx: int) -> int:
        return ((slab * self.num_m_block + m_block) * self.NB + n_block) * 128 + tidx

    @property
    def leaf_count(self) -> int:
        return self.HB * self.num_m_block * self.NB * 128

    def digest(self) -> bytes:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, default=list).encode()).digest()


@dataclass(frozen=True)
class ThreadOpening:
    index: int
    message: bytes             # header (64 B) + the thread's words, exactly what the kernel hashed
    path: tuple[bytes | None, ...]


def thread_message(stream: np.ndarray, layout: StreamLayout, geo: ThreadGeometry, src_mask: int, launch_tag: int,
                   seq_idx: int, slab: int, m_block: int, n_block: int, tidx: int) -> bytes:
    """Regenerate the exact H1 leaf message for one thread from the M1 stream (replay side)."""
    o = layout.offsets
    N = layout.N
    w, lane = tidx // 32, tidx % 32
    r0 = m_block * KBLOCK_M + 16 * w + lane // 4
    r1 = r0 + 8
    t = lane % 4
    sq, sk = geo.seqlens_q[seq_idx], geo.seqlens_k[seq_idx]
    ok0, ok1 = r0 < sq, r1 < sq
    words: list[int] = []
    cols = np.array([n_block * KBLOCK_N + 8 * (ni >> 1) + 2 * t + (ni & 1) for ni in range(32)])

    def f32plane(base: int, row: int, ok: bool) -> list[int]:
        return stream[base + (slab * layout.M + row) * N + cols].tolist() if ok else [0] * 32

    if src_mask & 1:
        words += f32plane(o["S"], r0, ok0) + f32plane(o["S"], r1, ok1)
    if src_mask & 8:
        def rowf(row: int, f: int, ok: bool) -> int:
            return int(stream[o["ROW"] + ((slab * layout.M + row) * layout.NB + n_block) * 8 + f]) if ok else 0
        step = rowf(r0 if ok0 else r1, 2, ok0 or ok1)
        words += [rowf(r0, 0, ok0), rowf(r1, 0, ok1), rowf(r0, 1, ok0), rowf(r1, 1, ok1), rowf(r0, 4 + t, ok0), rowf(r1, 4 + t, ok1),
                  n_block, step, 0, 0, 0, 0, 0, 0, 0, 0]
    if src_mask & 2:
        words += f32plane(o["P"], r0, ok0) + f32plane(o["P"], r1, ok1)
    if src_mask & 4:
        wc = (n_block * KBLOCK_N) // 2 + t + 4 * np.arange(16)
        for row, ok in ((r0, ok0), (r1, ok1)):
            words += stream[o["RP"] + (slab * layout.M + row) * (N // 2) + wc].tolist() if ok else [0] * 16
    if src_mask & 16:
        dd = np.array([8 * (i >> 1) + 2 * t + (i & 1) for i in range(16)])
        for row, ok in ((r0, ok0), (r1, ok1)):
            words += stream[o["O"] + ((slab * layout.M + row) * layout.NB + n_block) * 64 + dd].tolist() if ok else [0] * 16
    hdr = scheme.thread_header(src_mask, slab, m_block, n_block, tidx, sq, sk, launch_tag, geo.n_block_max(seq_idx, m_block), ok0, ok1)
    return hdr + pack_words([int(x) for x in words])


@dataclass
class ThreadLeafCommitment:
    geo: ThreadGeometry
    launch_tag: int
    src_mask: int
    merkle_root: bytes
    root: bytes
    levels: list[list[bytes]] | None
    timing: dict[str, float]

    @classmethod
    def from_digests(cls, digests_u32: np.ndarray, geo: ThreadGeometry, launch_tag: int, src_mask: int, retain_levels: bool = True) -> ThreadLeafCommitment:
        """Tree over the kernel-produced 32-byte leaf digests (index order = layout order; unvisited slots are zero leaves)."""
        d = np.ascontiguousarray(digests_u32, dtype=np.uint32).reshape(-1, 8)
        if d.shape[0] != geo.leaf_count:
            raise ValueError(f"{d.shape[0]} digests, geometry says {geo.leaf_count}")
        t0 = time.perf_counter()
        raw = d.tobytes()
        leaves = [raw[32 * i:32 * (i + 1)] for i in range(d.shape[0])]
        levels = scheme.fold_levels(leaves)
        t1 = time.perf_counter()
        mr = levels[-1][0]
        return cls(geo, launch_tag, src_mask, mr, scheme.thread_root(geo.digest(), launch_tag, src_mask, mr), levels if retain_levels else None,
                   {"tree_fold_s": t1 - t0, "leaf_count": d.shape[0], "digest_bytes": raw.__len__()})

    def open(self, message: bytes, index: int) -> ThreadOpening:
        assert self.levels is not None
        return ThreadOpening(index, message, scheme.path_in_levels(self.levels, index))


def verify_thread_opening(root: bytes, geo: ThreadGeometry, launch_tag: int, src_mask: int, seq_idx: int, slab: int, m_block: int,
                          n_block: int, tidx: int, opening: ThreadOpening) -> bool:
    """Verifier: derive the header from the claimed position, require it as the message prefix, hash, fold to the root."""
    idx = geo.leaf_index(slab, m_block, n_block, tidx)
    if opening.index != idx:
        return False
    w, lane = tidx // 32, tidx % 32
    r0 = m_block * KBLOCK_M + 16 * w + lane // 4
    sq, sk = geo.seqlens_q[seq_idx], geo.seqlens_k[seq_idx]
    hdr = scheme.thread_header(src_mask, slab, m_block, n_block, tidx, sq, sk, launch_tag, geo.n_block_max(seq_idx, m_block), r0 < sq, r0 + 8 < sq)
    if opening.message[:64] != hdr:
        return False
    if len(opening.path) != len(scheme.path_shape(geo.leaf_count, idx)):
        return False
    return scheme.thread_root(geo.digest(), launch_tag, src_mask, scheme.fold_path(scheme.thread_leaf(opening.message), idx, opening.path)) == root


def thread_word_offset(src_mask: int, cls: str, mi: int, ni_or_i: int) -> int:
    """Where value (class, row slot mi, fragment index) sits in the thread message (word offset incl. the 16 header words).
    S/P: ni in 0..31 (ascending column within the thread's 32), RP: i in 0..15 (u32 word), O: i in 0..15 (fragment order)."""
    off = 16
    if cls == "S":
        return off + mi * 32 + ni_or_i
    off += 64 if src_mask & 1 else 0
    if cls == "ROW":
        return off + ni_or_i
    off += 16 if src_mask & 8 else 0
    if cls == "P":
        return off + mi * 32 + ni_or_i
    off += 64 if src_mask & 2 else 0
    if cls == "RP":
        return off + mi * 16 + ni_or_i
    off += 32 if src_mask & 4 else 0
    if cls == "O":
        return off + mi * 16 + ni_or_i
    raise ValueError(cls)
