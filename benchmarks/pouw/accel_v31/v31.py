"""Dither's no-fold call (verity/core/protocols/pouw/circuit/dither.py) on the CPU: the reference prover's committed
values and tree, and the verifier's checks, so the GPU prover's output (v31.cu's Fill, bound per call by v31lib.py) is
judged bit for bit and a call is accepted or refused by the checks its proof blocks' Programs make.

A call is O = (A + d_A)(B + d_B)^T over the integers, A m x K and B N x K int8. A served call's A is the caller's
activations and its seeds come from the challenger's raw coin after the operand roots (dither.seeds); a filler call's A
is its pad and its seeds come from its live coin after its monitoring period's salt (dither.fill_seeds). Each seed is a
stream of SHA-512 counter-mode chunks ``prg(seed, tag, idx) = SHA-512(seed || tag || be64(idx))``, the tag zero-padded
to 8 bytes, one compression a 512-bit chunk:
    d_A[i, k]   bit k % 512 (LSB first) of prg(s_dA, "dA", i ceil(K / 512) + k // 512); d_B likewise under "dB"
    V[i, c]     ((r 161) >> 32) - 80, r the little-endian u32 word (i L' + c) of the "V" stream (16 words a chunk):
                uniform on -80..80 within 161 / 2^32; U[j, c] likewise under "U"
    pi_tau      Fisher-Yates over L' with word tau L' + c of the "P" stream: c from L' - 1 down to 1, x = (w_c (c + 1)) >> 32,
                swap perm[c] and perm[x]; pi_tau(c) = perm[c]
    pad[i]      a filler call's A row i: the first K bytes of chunks i ceil(K / 64) .. of the "pad" stream, as int8
The operands are X = wrap8(A + d_A + V[:, k % L']) and W' = wrap8(B + d_B + U[:, pi_tau(k % L')]), tau = k // L', L' a
noise tile's length. A segment is L inner indices (S = ceil(K / L) of them, t = min(L, K) / L' at most 8); per output cell the
prefix sums are T_s = X W'^T over the first min(s L, K) indices mod 2^16 (s = 1..S). A served call commits T_1 .. T_{S-1}
and O mod 2^32 (T_S is implied by O); a filler call commits T_1 .. T_S.

Layout (public structure). Output tiles are 128 x 128 (tile id tm (N / 128) + tn), a tile's cells in row-major order.
Per tile, its committed prefix sums (u16, 32 KiB a segment), then a served call's O (u32, 64 KiB), little-endian
(dither.values_buffer), cut into leaf rows of ROW bytes. Each row is an hm96-sha512/row/v2 row of 8 ROW bits
(``verity.ml.commitments``) in one frame-v3-sha512 tree a call, under dither.Shape.domain (SERVED_TAG or FILL_TAG); the
leaf at position pos of a call at salt offset off is salted ``prf_salt(Ks, off + pos)`` (commit_gpu's PRF). A proof block
(i, j, s) opens the rows that hold cell (i, j)'s T_s and T_{s-1}, and at a served call's s = S its O (`Fill.opened`).

The verifier opens q proof blocks drawn by the check coin (a cell uniformly, then s with probability its segment's share
of K) and runs the proof block's checks in the clear on the opened rows, a stand-in for its C-Flock --zk proof: the rows
against the root, T_s - T_{s-1} against the segment's X W'^T mod 2^16, and at a served call's s = S, O against
(A + d_A)(B + d_B)^T mod 2^32 and T_{S-1} against X W'^T over k < (S - 1) L. The operand rows A[i] and B[j] are taken in
the clear.

    python3 v31.py fill-params DIR NAME:M:N:K ... [--L L] [--served]   CGP1 params for `v31 fill-bench` and
                                                `v31 fill-once` (served: O committed; else the filler's), a fresh Ks each
    python3 v31.py fill-sample DIR              a fill-bench shape's two sampled tiles against the reference
    python3 v31.py fill-table BENCH.jsonl ... [--bar B]   gamma over cuBLASLt and CUTLASS, where the call's time goes,
                                                and the commit's compressions a SM-cycle over the ruler (and B)
    python3 v31.py timeline TRACE.csv           a kernel timeline of `v31 fill-once`'s serial and overlapped calls
                                                (run.sh's FILL_NSYS)

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "commit_gpu"))
import reference as cref

from verity.core.protocols.pouw.circuit import dither
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import (
    Commitment,
    CommitmentDomain,
    Opening,
    tree_levels,
    verify_opening,
)
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest

INST, KEY = cref.INST, cref.KEY
TILE = 128
NOISE = 80
STREAMS = ("dA", "dB", "U", "V", "P")


# ---------------------------------------------------------------------------------------------------------- the coins
prg = dither.prg


def stream(seed: bytes, tag: str, first: int, blocks: int) -> bytes:
    return b"".join(prg(seed, tag, first + b) for b in range(blocks))


def bits(seed: bytes, tag: str, rows: int, k: int) -> np.ndarray:
    kb = -(-k // 512)
    raw = np.frombuffer(stream(seed, tag, 0, rows * kb), dtype=np.uint8).reshape(rows, kb * 64)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :k].astype(np.int64)


def words32(seed: bytes, tag: str, count: int) -> np.ndarray:
    return np.frombuffer(stream(seed, tag, 0, -(-count // 16)), dtype="<u4")[:count].astype(np.uint64)


def noise(seed: bytes, tag: str, rows: int, lp: int) -> np.ndarray:
    r = words32(seed, tag, rows * lp)
    return (((r * np.uint64(2 * NOISE + 1)) >> np.uint64(32)).astype(np.int64) - NOISE).reshape(rows, lp)


def perms(seed: bytes, tiles: int, lp: int) -> np.ndarray:
    w = words32(seed, "P", tiles * lp).reshape(tiles, lp)
    out = np.empty((tiles, lp), dtype=np.int64)
    for tau in range(tiles):
        p = list(range(lp))
        for c in range(lp - 1, 0, -1):
            x = (int(w[tau, c]) * (c + 1)) >> 32
            p[c], p[x] = p[x], p[c]
        out[tau] = p
    return out


def bits_rows(seed: bytes, tag: str, rows: range, k: int) -> np.ndarray:
    """``bits`` for some rows only (a sampled tile's)."""
    kb = -(-k // 512)
    raw = np.frombuffer(b"".join(prg(seed, tag, i * kb + b) for i in rows for b in range(kb)), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(rows), kb * 64), axis=1, bitorder="little")[:, :k].astype(np.int64)


def noise_rows(seed: bytes, tag: str, rows: range, lp: int) -> np.ndarray:
    """``noise`` for some rows only: row r's words are chunks r L' / 16 .. (L' a multiple of 16)."""
    per = lp // 16
    w = np.frombuffer(b"".join(prg(seed, tag, r * per + b) for r in rows for b in range(per)), dtype="<u4").astype(np.uint64)
    return (((w * np.uint64(2 * NOISE + 1)) >> np.uint64(32)).astype(np.int64) - NOISE).reshape(len(rows), lp)


@dataclass
class CallNoise:
    """A call's noise, the public inputs its checks read: the dithers d_A, d_B, the noise V, U and the permutations,
    expanded from the call's seeds as dither.py expands its coins."""
    dA: np.ndarray
    dB: np.ndarray
    U: np.ndarray
    V: np.ndarray
    perm: np.ndarray

    @staticmethod
    def expand(sh: Fill, seeds: dict[str, bytes]) -> CallNoise:
        return CallNoise(bits(seeds["dA"], "dA", sh.m, sh.k), bits(seeds["dB"], "dB", sh.n, sh.k),
                         noise(seeds["U"], "U", sh.n, sh.lp), noise(seeds["V"], "V", sh.m, sh.lp),
                         perms(seeds["P"], sh.k // sh.lp, sh.lp))

    @staticmethod
    def rows(sh: Fill, seeds: dict[str, bytes], ia: range, ja: range) -> CallNoise:
        """The noise of A's rows ia and B's rows ja (and every permutation)."""
        return CallNoise(bits_rows(seeds["dA"], "dA", ia, sh.k), bits_rows(seeds["dB"], "dB", ja, sh.k),
                         noise_rows(seeds["U"], "U", ja, sh.lp), noise_rows(seeds["V"], "V", ia, sh.lp),
                         perms(seeds["P"], sh.k // sh.lp, sh.lp))


# ------------------------------------------------------------------------------------------------------------ forming
def wrap8(x: np.ndarray) -> np.ndarray:
    return ((x + 128) & 255) - 128


def v_full(sh: Fill, c: CallNoise, rows: slice | None = None) -> np.ndarray:
    V = c.V if rows is None else c.V[rows]
    return np.tile(V, (1, sh.k // sh.lp))


def u_full(sh: Fill, c: CallNoise, rows: slice | None = None) -> np.ndarray:
    U = c.U if rows is None else c.U[rows]
    return np.concatenate([U[:, c.perm[tau]] for tau in range(sh.k // sh.lp)], axis=1)


@dataclass
class Formed:
    X: np.ndarray
    Wp: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray


def form(sh: Fill, A: np.ndarray, B: np.ndarray, c: CallNoise) -> Formed:
    Ad, Bd = A.astype(np.int64) + c.dA, B.astype(np.int64) + c.dB
    return Formed(wrap8(Ad + v_full(sh, c)), wrap8(Bd + u_full(sh, c)), Ad, Bd)


def imatmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a b^T exactly (float64 BLAS: every partial sum here is below 2^53)."""
    assert a.shape[1] * 256 * 256 < 2 ** 53
    return np.rint(a.astype(np.float64) @ b.astype(np.float64).T).astype(np.int64)


# ------------------------------------------------------------------------------------------------------------- leaves
def leaf_h(row: bytes, salt: bytes) -> bytes:
    x = sha512_row_v2_digest(row, 8 * len(row))
    return INST.tree_leaf(KEY, INST.commit_string(KEY, x, salt))


def levels_bytes(levels: list[list[bytes]]) -> bytes:
    return b"".join(b"".join(lv) for lv in levels)


# ----------------------------------------------------------------------------------------------------------- the call
SERVED_TAG = "verity/pouw/dither/call/v1"
FILL_TAG = "verity/pouw/dither/fill/v1"
FILL_STREAMS = ("pad", "dA", "dB", "U", "V", "P")
K_LIMIT = 1 << 17


@dataclass(frozen=True)
class Fill:
    """A no-fold call: m x n output cells at contraction k, noise tiles of lp (L'), segments of L, leaf rows of `row`
    bytes, O committed (`o`: a served call; a filler's commits none) (dither.Shape)."""

    m: int
    n: int
    k: int
    lp: int = 256
    L: int = 2048
    row: int = 1024
    o: bool = False

    def __post_init__(self) -> None:
        if self.m % TILE or self.n % TILE or self.m <= 0 or self.n <= 0:
            raise ValueError(f"m and N multiples of {TILE}, not {self.m} x {self.n}")
        if self.lp % 32 or self.lp <= 0 or self.k % self.lp:
            raise ValueError(f"L' a multiple of 32 dividing K, not L' {self.lp} at K {self.k}")
        if self.k >= K_LIMIT:
            raise ValueError(f"K below {K_LIMIT}, not {self.k}")
        if self.L <= 0 or self.L % self.lp:
            raise ValueError(f"L a multiple of L' {self.lp}, not {self.L}")
        if min(self.L, self.k) > 8 * self.lp:
            raise ValueError(f"t = min(L, K) / L' is at most 8 (verity/core/protocols/pouw/dither/PROTOCOL.md, The segment "
                             f"and the noise tile), not {min(self.L, self.k)} / {self.lp} at L {self.L}, K {self.k}")
        if self.row % 16 or 32768 % self.row:
            raise ValueError(f"rows of a multiple of 16 bytes dividing 32 KiB, not {self.row}")

    @property
    def S(self) -> int:
        return -(-self.k // self.L)

    @property
    def W(self) -> int:
        """Segments whose prefix sums are committed: S - 1 when O is (T_S is implied by it), else S."""
        return self.S - 1 if self.o else self.S

    @property
    def tiles(self) -> int:
        return (self.m // TILE) * (self.n // TILE)

    @property
    def row_w(self) -> int:
        return 32768 // self.row

    @property
    def row_o(self) -> int:
        return 65536 // self.row if self.o else 0

    @property
    def lpt(self) -> int:
        return self.W * self.row_w + self.row_o

    @property
    def leaves(self) -> int:
        return self.tiles * self.lpt

    def seg(self, s: int) -> tuple[int, int]:
        return (s - 1) * self.L, min(s * self.L, self.k)

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "lp": self.lp, "t": self.L // self.lp, "row": self.row,
                "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(SERVED_TAG if self.o else FILL_TAG, self.label()), -1,
                                RangeIndexedDomain(0, self.leaves), hash="sha512")

    def tile_of(self, i: int, j: int) -> tuple[int, int]:
        return (i // TILE) * (self.n // TILE) + j // TILE, (i % TILE) * TILE + j % TILE

    def word_leaf(self, i: int, j: int, s: int) -> tuple[int, int]:
        """(leaf position, byte offset) of T_s at cell (i, j), s <= W."""
        if not 1 <= s <= self.W:
            raise ValueError(f"T_{s} isn't committed (segments 1 .. {self.W})")
        t, c = self.tile_of(i, j)
        per = self.row // 2
        return t * self.lpt + (s - 1) * self.row_w + c // per, 2 * (c % per)

    def o_leaf(self, i: int, j: int) -> tuple[int, int]:
        """(leaf position, byte offset) of O at cell (i, j)."""
        if not self.o:
            raise ValueError("a filler call commits no O")
        t, c = self.tile_of(i, j)
        per = self.row // 4
        return t * self.lpt + self.W * self.row_w + c // per, 4 * (c % per)

    def opened(self, i: int, j: int, s: int) -> list[int]:
        """The leaves proof block (i, j, s) opens: T_s's (s <= W), T_{s-1}'s (s > 1), and at s = S O's."""
        pos = [self.word_leaf(i, j, s)[0]] if s <= self.W else []
        pos += [self.word_leaf(i, j, s - 1)[0]] if s > 1 else []
        return sorted(set(pos + ([self.o_leaf(i, j)[0]] if s == self.S and self.o else [])))


served_seeds = dither.seeds
fill_seeds = call_seeds = dither.fill_seeds


def fill_seeds_bytes(seeds: dict[str, bytes]) -> bytes:
    """The six seeds in FILL_STREAMS' order, as v31.cu takes them (a served call's pad seed zero: it has no pad)."""
    return b"".join(seeds.get(s, bytes(64)) for s in FILL_STREAMS)


def fill_seeds_from(b: bytes, served: bool = False) -> dict[str, bytes]:
    return {s: b[64 * i:64 * i + 64] for i, s in enumerate(FILL_STREAMS) if not (served and s == "pad")}


def pad_rows(seeds: dict[str, bytes], rows: range, k: int) -> np.ndarray:
    """dither.pad_rows: row i is the first k bytes of chunks i ceil(k / 64) .. of the pad stream, as int8."""
    kb = -(-k // 64)
    raw = np.frombuffer(stream(seeds["pad"], "pad", rows.start * kb, len(rows) * kb), dtype=np.int8)
    return raw.reshape(len(rows), kb * 64)[:, :k].copy()


def fill_words(f: Fill, A: np.ndarray, B: np.ndarray, c: CallNoise, skip: int | None = None, upto: int | None = None) -> np.ndarray:
    """T_1 .. T_upto (default S; upto x rows(A) x rows(B)) mod 2^16 (skip: a prover that leaves noise tile `skip` out of
    X W'^T)."""
    fm = form(f, A, B, c)
    X = fm.X.copy()
    if skip is not None:
        X[:, skip * f.lp:(skip + 1) * f.lp] = 0
    upto = f.S if upto is None else upto
    acc = np.zeros((A.shape[0], B.shape[0]), dtype=np.int64)
    T = np.zeros((upto, A.shape[0], B.shape[0]), dtype=np.int64)
    for s in range(1, upto + 1):
        a, b = f.seg(s)
        acc += imatmul(X[:, a:b], fm.Wp[:, a:b])
        T[s - 1] = acc & 0xFFFF
    return T


def fill_values(f: Fill, A: np.ndarray, B: np.ndarray, c: CallNoise, skip: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The call's committed values: T_1 .. T_W (W x rows(A) x rows(B), mod 2^16) and a served call's O (mod 2^32; None for
    a filler's)."""
    T = fill_words(f, A, B, c, skip, f.W)
    if not f.o:
        return T, None
    Ad, Bd = A.astype(np.int64) + c.dA, B.astype(np.int64) + c.dB
    return T, imatmul(Ad, Bd) & 0xFFFFFFFF


def fill_buffer(f: Fill, T: np.ndarray, O: np.ndarray | None = None) -> bytes:
    """dither.values_buffer: per tile, its T_1 .. T_W (u16) and then its O (u32), cells row-major, little-endian."""
    tm, tn = f.m // TILE, f.n // TILE
    w = T[:f.W].astype("<u2").reshape(f.W, tm, TILE, tn, TILE).transpose(1, 3, 0, 2, 4).reshape(tm * tn, -1)
    parts = [w.view(np.uint8)]
    if f.o:
        if O is None:
            raise ValueError("a served call commits O: pass it")
        parts.append(O.astype("<u4").reshape(tm, TILE, tn, TILE).transpose(0, 2, 1, 3).reshape(tm * tn, -1).view(np.uint8))
    return np.ascontiguousarray(np.concatenate(parts, axis=1)).tobytes()


@dataclass
class FillCommitted:
    f: Fill
    ks: bytes
    values: bytes
    off: int = 0
    levels: list[list[bytes]] = field(default_factory=list)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def row(self, pos: int) -> bytes:
        return self.values[pos * self.f.row:(pos + 1) * self.f.row]

    def salt(self, pos: int) -> bytes:
        return cref.prf_salt(self.ks, self.off + pos)

    def path(self, pos: int) -> tuple[bytes, ...]:
        return tuple(self.levels[d][(pos >> d) ^ 1] for d in range(len(self.levels) - 1))


def fill_commit(f: Fill, values: bytes, ks: bytes, off: int = 0) -> FillCommitted:
    """The call's values (`fill_buffer`) under its root, leaf pos salted prf_salt(ks, off + pos) (dither.commit under
    dither.salts(ks, off))."""
    if len(values) != f.leaves * f.row:
        raise ValueError(f"the call's values are {f.leaves * f.row} bytes, not {len(values)}")
    cm = FillCommitted(f, ks, values, off)
    dom = f.domain()
    cm.levels = [list(lv) for lv in tree_levels(dom, [row_leaf(dom, p, leaf_h(cm.row(p), cm.salt(p)), V2)
                                                      for p in range(f.leaves)])]
    return cm


def fill_params(f: Fill, ks: bytes) -> bytes:
    """The CGP1 params of a no-fold call's tree: its leaves, row bits, the prover's private Ks and the domain."""
    return cref.params(f.leaves, 8 * f.row, ks, KEY, False, f.domain().domain_id)


def fill_picks(f: Fill, coin: bytes, q: int) -> list[tuple[int, int, int]]:
    """q proof blocks (i, j, s) from a check coin: a cell uniformly, then s with probability its segment's share of K."""
    picks = []
    for b in range(q):
        r = int.from_bytes(hashlib.sha512(coin + b.to_bytes(8, "big")).digest(), "big")
        i, r = r % f.m, r // f.m
        j, r = r % f.n, r // f.n
        picks.append((i, j, (r % f.k) // f.L + 1))
    return picks


def fill_check(f: Fill, seeds: dict[str, bytes], root: bytes, i: int, j: int, s: int, ans: dict) -> str | None:
    """A no-fold proof block's checks in the clear (the stand-in for its C-Flock --zk proof; None: passed): the opened rows
    (`Fill.opened`) against the root; for s < S (or a filler's s = S), T_s - T_{s-1} against the segment's X W'^T mod 2^16
    (PouwDitherBlock); for a served call's s = S (PouwDitherLast), O against (A + d_A)(B + d_B)^T mod 2^32 and T_{S-1}
    against X W'^T over k < (S - 1) L mod 2^16. ans: {"rows": {pos: (row, salt, path)}, "A": A's row i (the pad's when
    absent), "B": B's row j}."""
    dom, cmt, lim = f.domain(), Commitment(root, f.leaves), VerificationLimits()
    rows = {}
    for pos, (row, salt, path) in ans["rows"].items():
        if not verify_opening(dom, cmt, Opening(pos, leaf_h(row, salt), tuple(path)), V2, lim):
            return f"leaf {pos}: its row and salt don't open the root"
        rows[pos] = row

    def word(s_: int) -> int:
        if s_ == 0:
            return 0
        p, o = f.word_leaf(i, j, s_)
        if p not in rows:
            raise KeyError(f"leaf {p} of T_{s_} wasn't opened")
        return int.from_bytes(rows[p][o:o + 2], "little")

    if ans.get("A") is None and f.o:
        return "a served call's proof block reads A's row"
    Ai = pad_rows(seeds, range(i, i + 1), f.k) if ans.get("A") is None else np.asarray(ans["A"], dtype=np.int8)[None, :]
    Bj = np.asarray(ans["B"], dtype=np.int8)[None, :]
    c = CallNoise.rows(f, seeds, range(i, i + 1), range(j, j + 1))
    fm = form(f, Ai, Bj, c)
    try:
        if f.o and s == f.S:
            p, o = f.o_leaf(i, j)
            if p not in rows:
                return f"leaf {p} of O wasn't opened"
            if (int.from_bytes(rows[p][o:o + 4], "little") - int((fm.Ad[0] * fm.Bd[0]).sum())) % (1 << 32):
                return "O isn't (A + d_A)(B + d_B)^T mod 2^32"
            lo = f.seg(s)[0]
            if lo and (word(s - 1) - int((fm.X[0, :lo] * fm.Wp[0, :lo]).sum())) % (1 << 16):
                return f"T_{s - 1} isn't X W'^T over k < (S - 1) L mod 2^16"
            return None
        a, b = f.seg(s)
        if (word(s) - word(s - 1) - int((fm.X[0, a:b] * fm.Wp[0, a:b]).sum())) % (1 << 16):
            return f"T_{s} - T_{s - 1} isn't the segment's X W'^T mod 2^16"
    except KeyError as e:
        return str(e)
    return None


def fill_cells_positions(f: Fill, i: int, j: int, s: int) -> list[int]:
    """The leaves a proof block (i, j, s) opens (dither.Shape.opened)."""
    return f.opened(i, j, s)


def fill_verdict(f: Fill, seeds: dict[str, bytes], root: bytes, picks: list[tuple[int, int, int]], answer, label: str) -> tuple[bool, list[str], str]:
    """The verifier's verdict on q opened proof blocks: answer(i, j, s) gives each proof block's opened rows and operand rows."""
    fails = []
    for (i, j, s) in picks:
        why = fill_check(f, seeds, root, i, j, s, answer(i, j, s))
        if why:
            fails.append(f"({i}, {j}, s {s}): {why}")
    ok = not fails
    line = (f"{'ACCEPT' if ok else 'REFUSE'} {label} {'served' if f.o else 'filler'} {f.m}x{f.n}x{f.k} L' {f.lp} L {f.L} "
            f"S {f.S}: {len(picks)} proof blocks "
            f"opened, {len(picks) - len(fails)} passed, {len(fails)} failed" + (f"; first: {fails[0]}" if fails else "")
            + f"; root {root.hex()[:16]}")
    return ok, fails, line


# ----------------------------------------------------------------------------------------------------------- operands
def operands(sh: Fill, rng: np.random.Generator, kind: str = "llm") -> tuple[np.ndarray, np.ndarray]:
    """int8 operands. ``llm``: rows quantized to +-127 by their absolute maximum, activations from a Student t (heavy tails,
    so few wrap at V +-80) and weights from a Gaussian (more wrap at U +-80);
    ``uniform``: every value on -128..127 (a third of the entries wrap); ``zero``: A = B = 0; ``max``: A = B = 127 (half
    the entries wrap)."""
    if kind == "zero":
        return np.zeros((sh.m, sh.k), dtype=np.int8), np.zeros((sh.n, sh.k), dtype=np.int8)
    if kind == "max":
        return np.full((sh.m, sh.k), 127, dtype=np.int8), np.full((sh.n, sh.k), 127, dtype=np.int8)
    if kind == "uniform":
        return (rng.integers(-128, 128, size=(sh.m, sh.k)).astype(np.int8), rng.integers(-128, 128, size=(sh.n, sh.k)).astype(np.int8))
    a = rng.standard_t(3, size=(sh.m, sh.k))
    b = rng.standard_normal(size=(sh.n, sh.k))
    def q(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(127 * x / np.abs(x).max(axis=1, keepdims=True)), -127, 127).astype(np.int8)

    return q(a), q(b)


# ---------------------------------------------------------------------------------------------------------------- CLI
def fill_params_name(m: int, n: int, k: int, lp: int, L: int, row: int, o: bool = False) -> str:
    return f"{'served' if o else 'fill'}-{m}x{n}x{k}-lp{lp}-L{L}-r{row}.bin"


def params_ks(p: bytes) -> bytes:
    """The private Ks a CGP1 params file carries."""
    return p[4 + 3 * 128 + 64:4 + 3 * 128 + 128]


def cmd_fill_params(a: argparse.Namespace) -> int:
    """CGP1 params (a fresh Ks each) for `v31 fill-bench`'s shapes at L (0: L = K), named as it looks them up."""
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    for s in a.shapes:
        m, n, k = (int(x) for x in s.split(":")[-3:])
        L = a.L or k
        f = Fill(m, n, k, a.lp, L, a.row, a.served)
        (d / fill_params_name(m, n, k, a.lp, L, a.row, a.served)).write_bytes(fill_params(f, os.urandom(64)))
    return 0


def cmd_fill_sample(a: argparse.Namespace) -> int:
    """A fill-bench shape's two sampled tiles (DIR from `v31 fill-bench`): each tile's values and leaves against the
    reference, from the tile's A and B rows and those rows' coins alone."""
    d = Path(a.dir)
    meta = json.loads((d / "sample.json").read_text())
    o = bool(meta.get("o", False))
    f = Fill(meta["m"], meta["n"], meta["k"], meta["lp"], meta["L"], meta["row"], o)
    tl = Fill(TILE, TILE, f.k, f.lp, f.L, f.row, o)
    seeds = fill_seeds_from((d / "seeds.bin").read_bytes(), o)
    p = (d / "params.bin").read_bytes()
    ks, off = params_ks(p), int(meta.get("off", 0))
    rep: dict = {"shape": f.label(), "params_match": p == fill_params(f, ks), "tiles": []}
    dom = f.domain()
    for idx, tile in enumerate(meta["tiles"]):
        tm, tn = divmod(tile, f.n // TILE)
        A = np.frombuffer((d / f"t{idx}_A.bin").read_bytes(), dtype=np.int8).reshape(TILE, f.k)
        B = np.frombuffer((d / f"t{idx}_B.bin").read_bytes(), dtype=np.int8).reshape(TILE, f.k)
        c = CallNoise.rows(f, seeds, range(tm * TILE, (tm + 1) * TILE), range(tn * TILE, (tn + 1) * TILE))
        words = fill_buffer(tl, *fill_values(tl, A, B, c))
        g_leaves = (d / f"t{idx}_leaves.bin").read_bytes()
        bad = 0
        for j in range(f.lpt):
            pos = tile * f.lpt + j
            leaf = row_leaf(dom, pos, leaf_h(words[j * f.row:(j + 1) * f.row], cref.prf_salt(ks, off + pos)), V2)
            bad += leaf != g_leaves[64 * j:64 * j + 64]
        rep["tiles"].append({"tile": tile, "words_equal": (d / f"t{idx}_words.bin").read_bytes() == words, "leaves": f.lpt,
                             "leaves_bad": bad})
    rep["exact"] = rep["params_match"] and all(t["words_equal"] and not t["leaves_bad"] for t in rep["tiles"])
    print(json.dumps(rep))
    (d / "sample-check.json").write_text(json.dumps(rep, indent=1))
    return 0 if rep["exact"] else 1


def commit_labels(r: dict) -> str:
    """A fill-bench line's commit knobs off their defaults, as fill-table labels its rows after the call's knobs."""
    return ((f" m{r['lminb']}" if r.get("lminb", 5) not in (4, 5) else "")
            + (f" sm{r['sminb']}" if r.get("sminb", 4) != 4 else "") + (" cf" if r.get("lfold") == 1 else "")
            + (" nofull" if r.get("full") == 0 else "") + (f" mk{r['mask']}" if r.get("mask") else ""))


def cmd_fill_table(a: argparse.Namespace) -> int:
    """`v31 fill-bench`'s JSON lines: gamma of Dither's no-fold call overlapped (a served call's, or a filler's), over
    cuBLASLt and CUTLASS, with the weight side a monitoring period's (ovl), a call's (weight coins drawn every call: ovl_w)
    and the filler's pad (ovl_pad), ovl with the commit's loads left out (ovl_nomem: its compressions on bytes made in
    registers; wrong roots, timing only), the GEMM's tensor duty (cuBLASLt's time over the call's period, per-call weight
    coins) and where the call's time goes. An arm the call doesn't run (a served call's pad) is "-". Then wgmma.linear's
    time (a -DV31_WG build's served bench), gamma ovl_w over it, and the share of O's rows the GEMM's hash warps hashed
    (V31_RP 4, the last overlapped call).

    Then the commit's parts alone (lines that carry them): each part's SHA-512 compressions a SM-cycle at its median time
    and clock, over the build's own compression rate (the sha arm, commit_gpu's ruler), and with ``--bar`` (the card's
    fastest full-output compression a SM-cycle) over that bar."""
    rows, named = [], []
    for path in a.files:
        for line in Path(path).read_text().splitlines():
            if line.startswith("{") and '"fill": true' in line:
                rows.append(json.loads(line))

    def g(x) -> str:
        return "-" if x is None else f"{x:.3f}"

    print("| shape | m x N x K | L | S | variant | lt ms | CUTLASS ms | ovl ms | gamma ovl | / CUTLASS | gamma ovl_w | / CUTLASS "
          "| gamma ovl_pad | gamma nomem | gamma call | gamma gemm | tensor duty | coins, coins_w, pad, form_a, form_b, gemm, "
          "commit ms | wgl ms | gamma ovl_w / wgl | rows hashed in the GEMMs: rows / chunks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ms = r["arms"]
        variant = (("served " if r.get("o") else "") + f"r{r['row']} {r['cprio']} k{r['ccta']} s{r['sched']}"
                   + (f" p{r['pf']}" if r.get("pf") else "") + (" ltw" if r.get("ltw") else "")
                   + (f" wg{r.get('wbn', '')}" if r.get("wg") else "") + (f" rp{r['rp']}" if r.get("wg") and r.get("rp") else "")
                   + (" hall" if r.get("hall") else "") + (f" c{r['hchunk']}" if r.get("hchunk") else "")
                   + (" w0" if r.get("wg") and r.get("W") and not r.get("wgw", 1) else "")
                   + (" r5w0" if r.get("rp") == 5 and r.get("W") and r.get("wgw", 1) and not r.get("r5w", 1) else "")
                   + (" tst" if r.get("rp") == 5 and r.get("tst") else "")
                   + (" m4" if r.get("lminb") == 4 else "") + (" sa" if r.get("sahead") else ""))
        variant += commit_labels(r)
        named.append((r, variant))
        fused = "-"
        rows_all = r.get("leaves") or r.get("o_leaves")
        if r.get("fused_leaves", -1) >= 0 and rows_all:
            fused = f"{100 * r['fused_leaves'] / rows_all:.1f}%"
            if r.get("fused_blocks", -1) >= 0 and r.get("row_blocks"):
                fused += f" / {100 * r['fused_blocks'] / (rows_all * r['row_blocks']):.1f}%"
        cut = g(ms["cutlass"]["ms"]) if "cutlass" in ms else "-"
        gcut, gcut_w = (r.get("gamma_ovl_cutlass"), r.get("gamma_ovl_w_cutlass")) if "cutlass" in ms else (None, None)
        parts = ", ".join(g(ms[x]["ms"]) if x in ms else "-" for x in ("coins", "coins_w", "pad", "form_a", "form_b", "gemm", "commit"))
        duty = r.get("gamma_ovl_w") if r.get("o") else r["gamma_ovl"]
        lt, ovl = (g(ms[x]["ms"]) if x in ms else "-" for x in ("lt", "ovl"))
        print(f"| {r['shape']} | {r['m']} x {r['n']} x {r['k']} | {r['L']} | {r['S']} | {variant} | {lt} | {cut} | "
              f"{ovl} | {g(r['gamma_ovl'])} | {g(gcut)} | {g(r.get('gamma_ovl_w'))} | {g(gcut_w)} | "
              f"{g(r.get('gamma_ovl_pad'))} | {g(r.get('gamma_ovl_nomem'))} | {g(r['gamma_call'])} | {g(r['gamma_gemm'])} | "
              f"{f'{100 / duty:.1f}%' if duty else '-'} | {parts} | {g(ms['wgl']['ms']) if 'wgl' in ms else '-'} | {g(r.get('gamma_ovl_w_wgl'))} | "
              f"{fused} |")
    parts = [(r, variant) for r, variant in named if "per_sm_cycle" in r]
    if not parts:
        return 0

    def c(x) -> str:
        return "-" if x is None else f"{x:.5f}"

    def of(x, bar) -> str:
        return "-" if x is None or not bar else f"{x / bar:.3f}"

    print()
    print("| shape | variant | sha / SM-cycle | part | ms | MHz | compressions | / SM-cycle | of sha | of bar |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r, variant in parts:
        pc, cm = r["per_sm_cycle"], r["comp"]
        for part, comp in (("leaf", cm["leaf"]), ("leaf_nomem", cm["leaf"]), ("salt", cm["salt"]),
                           ("commit", cm["leaf"] + cm["salt"])):
            arm = r["arms"].get(part, {})
            print(f"| {r['shape']} | {variant} | {c(pc['sha'])} | {part} | {g(arm.get('ms'))} | {arm.get('mhz', '-')} | "
                  f"{comp:.0f} | {c(pc[part])} | {g(r['of_sha'][part])} | {of(pc[part], a.bar)} |")
    return 0


# ----------------------------------------------------------------------------------------------------------- timeline
COMPUTE_KERNELS = ("bits_kernel", "noise_kernel", "perm_kernel", "fill_a_kernel", "fill_b_kernel", "serve_rows_kernel",
                   "gemm_kernel")


def kernel_name(full: str) -> str:
    """A demangled kernel name without `void`, its parameters and its argument types' casts: `gemm_kernel<true>`."""
    s, depth, out = full.strip().removeprefix("void "), 0, []
    for ch in s:
        if ch == "(" and depth == 0:
            break
        depth += (ch == "<") - (ch == ">")
        out.append(ch)
    return "".join(out).replace("(bool)1", "true").replace("(bool)0", "false").replace("(int)", "")


def spans_union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def covered(spans: list[tuple[int, int]], s: int, e: int) -> int:
    """The time in [s, e) under the (disjoint) spans."""
    return sum(max(0, min(e, b) - max(s, a)) for a, b in spans)


def cmd_timeline(a: argparse.Namespace) -> int:
    """`v31 fill-once` in Nsight Systems (its `nsys stats -r cuda_gpu_trace` CSV): each kernel's time in the
    second serial call and its median over the steady overlapped calls (each one's compute beside the previous call's
    commit, the first overlapped call's left out), with the share of it that ran beside the other stream; the steady
    period (GEMM end to GEMM end); and the steady calls' time with compute alone, the commit alone, both, and neither.
    The GEMM is the kernel whose name --gemm matches (cuBLASLt's under V31_LTW), and it counts the calls. A kernel's k-th
    launch belongs to call k // (its launches / the calls); a compute kernel launched in some calls only (`v31 fill-once`'s
    weight side, formed once a monitoring period) to the call whose GEMM comes next. --from NAME drops what ran before
    NAME's first launch (the operands' generation and cuBLASLt's autotuning). Each kernel's grid, CTA, registers a thread and shared memory are its first launch's."""
    ks, res = [], {}
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Name", "").startswith("[CUDA") or not r.get("Start (ns)"):
                continue
            n = kernel_name(r["Name"])
            ks.append((int(r["Start (ns)"]), int(r["Start (ns)"]) + int(r["Duration (ns)"]), n))
            if n not in res:
                smem = sum(float(r.get(c) or 0) for c in ("StcSMem (MB)", "DymSMem (MB)"))
                res[n] = f"{r.get('GrdX', '?')} x {r.get('BlkX', '?')}, {r.get('Reg/Trd', '?')} regs, {smem * 1024:.0f} KB"
    ks.sort()
    if a.start:
        first = min((s for s, _, n in ks if n.startswith(a.start)), default=None)
        if first is None:
            print(f"no {a.start} launch", file=sys.stderr)
            return 2
        ks = [k for k in ks if k[0] >= first]
    gemm = re.compile(a.gemm)
    gstart = [s for s, _, n in ks if gemm.search(n)]
    ncalls = len(gstart)
    if ncalls < a.serial + 3:
        print(f"{ncalls} calls: want {a.serial} serial and at least 3 overlapped", file=sys.stderr)
        return 2
    by_name: dict[str, list[tuple[int, int]]] = {}
    for s, e, n in ks:
        by_name.setdefault(n, []).append((s, e))
    is_compute = {n: n.startswith(COMPUTE_KERNELS) or bool(gemm.search(n)) for n in by_name}
    calls: list[list[tuple[int, int, str]]] = [[] for _ in range(ncalls)]
    for n, v in by_name.items():
        if len(v) % ncalls == 0:
            per = len(v) // ncalls
            for j, (s, e) in enumerate(v):
                calls[j // per].append((s, e, n))
        elif is_compute[n]:
            for s, e in v:
                calls[min(bisect.bisect_left(gstart, s), ncalls - 1)].append((s, e, n))
    first_ovl = a.serial
    steady = range(first_ovl + 1, ncalls)
    comp_spans = spans_union([(s, e) for c in calls[first_ovl:] for s, e, n in c if is_compute[n]])
    comm_spans = spans_union([(s, e) for c in calls[first_ovl:] for s, e, n in c if not is_compute[n]])
    print(f"{a.csv}: {ncalls} calls ({a.serial} serial), steady overlapped calls {steady.start}..{steady.stop - 1}\n")
    print("| kernel | grid x CTA, regs, smem | launches a call | serial ms | overlapped ms (median) | overlapped / serial "
          "| beside the other stream |")
    print("|---|---|---|---|---|---|---|")
    names = list(dict.fromkeys(n for _, _, n in calls[a.serial - 1]))
    for n in names:
        ser = sum(e - s for s, e, m in calls[a.serial - 1] if m == n)
        per_call, beside = [], []
        for i in steady:
            mine = [(s, e) for s, e, m in calls[i] if m == n]
            per_call.append(sum(e - s for s, e in mine))
            other = comm_spans if is_compute[n] else comp_spans
            beside.append(sum(covered(other, s, e) for s, e in mine))
        med = float(np.median(per_call))
        side = sum(beside) / max(1, sum(per_call))
        print(f"| {n} | {res[n]} | {len([m for _, _, m in calls[a.serial - 1] if m == n])} | {ser / 1e6:.3f} | "
              f"{med / 1e6:.3f} | {med / max(ser, 1):.2f} | {100 * side:.0f}% |")
    gend = [max(e for s, e, n in c if gemm.search(n)) for c in calls]
    periods = [gend[i] - gend[i - 1] for i in steady]
    ser_call = calls[a.serial - 1]
    ser_span = max(e for s, e, _ in ser_call) - min(s for s, _, _ in ser_call)
    print(f"\nserial call: {ser_span / 1e6:.3f} ms; steady period (GEMM end to GEMM end): median "
          f"{np.median(periods) / 1e6:.3f} ms, {', '.join(f'{p / 1e6:.3f}' for p in periods)}")
    w0, w1 = gend[steady.start - 1], gend[steady.stop - 1]
    cs, ms = ([(max(s, w0), min(e, w1)) for s, e in sp if e > w0 and s < w1] for sp in (comp_spans, comm_spans))
    comp, comm = sum(e - s for s, e in cs), sum(e - s for s, e in ms)
    both = sum(covered(ms, s, e) for s, e in cs)
    win = w1 - w0
    print(f"steady calls {win / 1e6:.3f} ms: compute alone {(comp - both) / win:.1%}, commit alone {(comm - both) / win:.1%}, "
          f"both {both / win:.1%}, neither {(win - comp - comm + both) / win:.1%}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("timeline")
    p.add_argument("csv")
    p.add_argument("--serial", type=int, default=2, help="the serial calls before the overlapped ones (`v31 fill-once`'s 2)")
    p.add_argument("--gemm", default="gemm", help="a regex for the GEMM's kernel name")
    p.add_argument("--from", dest="start", default="bits_kernel", help="drop kernels before this one's first launch")
    p = sub.add_parser("fill-params")
    p.add_argument("dir")
    p.add_argument("shapes", nargs="+", help="[name:]M:N:K")
    p.add_argument("--lp", type=int, default=256)
    p.add_argument("--L", type=int, default=2048, help="the segment length (0: K), at most 8 L' below K")
    p.add_argument("--row", type=int, default=1024)
    p.add_argument("--served", action="store_true", help="served calls (O committed, dither.TAG), not the filler's")
    p = sub.add_parser("fill-sample")
    p.add_argument("dir")
    p = sub.add_parser("fill-table")
    p.add_argument("files", nargs="+")
    p.add_argument("--bar", type=float, default=None,
                   help="the card's fastest full-output SHA-512 compressions a SM-cycle, for the commit's parts")
    a = ap.parse_args(argv[1:])
    return {"timeline": cmd_timeline, "fill-params": cmd_fill_params, "fill-sample": cmd_fill_sample,
            "fill-table": cmd_fill_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
