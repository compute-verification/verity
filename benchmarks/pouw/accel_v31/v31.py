"""ACCEL v3.1's credited static matmul (verity/core/protocols/pouw/accel/PROTOCOL.md on #4746): the prover's values and
commitment on the CPU, a stub challenger, and the reference verifier, so v31.cu's output is judged bit for bit and a run is
accepted or refused by the checks the proof block's Program makes.

A call is O = (A + d_A)(B + d_B)^T over the integers, A m x K and B N x K int8. The challenger's seeds (after the operand
roots) give the coins, each a stream of SHA-512 counter-mode blocks ``prg(seed, tag, idx) = SHA-512(seed || tag || be64(idx))``
(one compression a 512-bit block; a stub for the live seed stream, which would give fresh 512-bit seeds at the coin rate
the spec names, at the same cost a bit):
    d_A[i, k]   bit k % 512 (LSB first) of prg(s_dA, "dA", i ceil(K / 512) + k // 512); d_B likewise under "dB"
    V[i, c]     ((r 161) >> 32) - 80, r the little-endian u32 word (i L' + c) of the "V" stream (16 words a block):
                uniform on -80..80 within 161 / 2^32; U[j, c] likewise under "U"
    pi_tau      Fisher-Yates over L' with word tau L' + c of the "P" stream: c from L' - 1 down to 1, x = (w_c (c + 1)) >> 32,
                swap perm[c] and perm[x]; pi_tau(c) = perm[c]
The operands are X = wrap8(A + d_A + V[:, k % L']) and W' = wrap8(B + d_B + U[:, pi_tau(k % L')]), tau = k // L', with
wrap indicators w_X, w_W in {-1, 0, 1}. A segment is L = t L' inner indices (S = ceil(K / L) of them); per output cell the
committed values are T_s = sum over the first s L indices of X W'^T mod 2^16 (s = 1..S) and O mod 2^32, bound by
O = T_S + 256 (w_X W'^T + X w_W^T + 256 w_X w_W^T) - Z_X U^T - V Y_W^T, Z_X = sum_tau (A + d_A + V)_tau pi_tau^T,
Y_W = sum_tau (B + d_B)_tau (T_S at full width).

Layout (public structure). Output tiles are 128 x 128 (tile id tm (N / 128) + tn), a tile's cells in row-major order
(c = 128 r + col). The words buffer holds, per tile and segment, the tile's 16,384 T_s as little-endian u16 (32 KiB); O is
the row-major m x N output as little-endian u32. Leaves are rows of ROW bytes (default 1 KiB), LPT = S ROW_W + ROW_O a
tile, ROW_W = 32768 / ROW and ROW_O = 65536 / ROW: leaf tile LPT + (s - 1) ROW_W + r is bytes [r ROW, (r + 1) ROW) of the
tile's segment-s words, and leaf tile LPT + S ROW_W + h is the O values of the tile's cells [h ROW / 4, (h + 1) ROW / 4).
Each row is an hm96-sha512/row/v2 row of 8 ROW bits (``verity.ml.commitments``: rowleaf, hm96.SHA512, merkle) in one
frame-v3-sha512 tree a call, domain ``CommitmentDomain(identity_digest_sha512(TAG, {shape}), -1, RangeIndexedDomain(0, n))``,
salts ``prf_salt(Ks, off + pos)`` (commit_gpu's PRF). A proof block (i, j, s) opens the words rows of T_s and T_{s-1}
(s > 1), and at s = S O's row too.

The verifier opens q proof blocks drawn by the check coin (a cell uniformly, then s with probability its segment's share
of K: each block's share of the call's MACs), and runs the Program's checks in the clear on the opened rows, a stand-in
for the C-Flock --zk proof (the Definition isn't written yet): the rows' leaves and paths against the root, T_s - T_{s-1}
against the segment's X W'^T mod 2^16, and at s = S O against (A + d_A)(B + d_B)^T mod 2^32 and the binding identity
mod 2^16. The operand rows A[i], B[j] are taken in the clear (their commitments are the previous call's and the period's).

    python3 v31.py cpu-run OUT M N K [--lp 256] [--t 3] [--row 1024] [--q 64] [--seed S] [--cheat KIND]
        one call on the CPU, honest or cheating (skip:TAU, forge:S0[:FRAC], forgeO[:FRAC]), and the verdict line
    python3 v31.py gpu-prep DIR M N K [...]     the challenger's side before a GPU run: A, B, seeds, params (CGP1), meta
    python3 v31.py gpu-check DIR [--q 64]       a GPU run's dump against the reference (every word, O, tree level), then
                                                the protocol's verdict on the GPU's own commitment and values
    python3 v31.py gpu-sample DIR               a bench shape's two sampled tiles (from `v31 bench`) against the reference,
                                                built from the tiles' A and B rows and those rows' coins alone
    python3 v31.py params DIR NAME:M:N:K ...    CGP1 params for `v31 bench` (a fresh Ks each)
    python3 v31.py table BENCH.jsonl ...        gamma beside D6 v1.1's prediction, shape by shape

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "commit_gpu"))
import reference as cref

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

TAG = "verity/accel-v31/call/v1"
INST, KEY = cref.INST, cref.KEY
TILE = 128
NOISE = 80
STREAMS = ("dA", "dB", "U", "V", "P")


# ---------------------------------------------------------------------------------------------------------- the shape
@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    lp: int = 256
    t: int = 3
    row: int = 1024

    def __post_init__(self) -> None:
        if self.m % TILE or self.n % TILE or self.m <= 0 or self.n <= 0:
            raise ValueError(f"m and N multiples of {TILE}, not {self.m} x {self.n}")
        if self.lp % 32 or self.lp <= 0 or self.k % self.lp:
            raise ValueError(f"L' a multiple of 32 dividing K, not L' {self.lp} at K {self.k}")
        if self.row % 16 or 32768 % self.row:
            raise ValueError(f"rows of a multiple of 16 bytes dividing 32 KiB, not {self.row}")

    @property
    def L(self) -> int:
        return self.t * self.lp

    @property
    def S(self) -> int:
        return -(-self.k // self.L)

    @property
    def tiles(self) -> int:
        return (self.m // TILE) * (self.n // TILE)

    @property
    def row_w(self) -> int:
        return 32768 // self.row

    @property
    def row_o(self) -> int:
        return 65536 // self.row

    @property
    def lpt(self) -> int:
        return self.S * self.row_w + self.row_o

    @property
    def leaves(self) -> int:
        return self.tiles * self.lpt

    def seg(self, s: int) -> tuple[int, int]:
        """Segment s's inner range (s = 1..S)."""
        return (s - 1) * self.L, min(s * self.L, self.k)

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "lp": self.lp, "t": self.t, "row": self.row, "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(TAG, self.label()), -1, RangeIndexedDomain(0, self.leaves), hash="sha512")

    def tile_of(self, i: int, j: int) -> tuple[int, int]:
        """(tile id, local cell index) of cell (i, j)."""
        return (i // TILE) * (self.n // TILE) + j // TILE, (i % TILE) * TILE + j % TILE

    def word_leaf(self, i: int, j: int, s: int) -> tuple[int, int]:
        """(leaf position, byte offset) of T_s at cell (i, j)."""
        t, c = self.tile_of(i, j)
        per = self.row // 2
        return t * self.lpt + (s - 1) * self.row_w + c // per, 2 * (c % per)

    def o_leaf(self, i: int, j: int) -> tuple[int, int]:
        t, c = self.tile_of(i, j)
        per = self.row // 4
        return t * self.lpt + self.S * self.row_w + c // per, 4 * (c % per)


# ---------------------------------------------------------------------------------------------------------- the coins
def prg(seed: bytes, tag: str, idx: int) -> bytes:
    assert len(seed) == 64
    return hashlib.sha512(seed + tag.encode().ljust(8, b"\0") + idx.to_bytes(8, "big")).digest()


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
    """``noise`` for some rows only: row r's words are blocks r L' / 16 .. (L' a multiple of 16)."""
    per = lp // 16
    w = np.frombuffer(b"".join(prg(seed, tag, r * per + b) for r in rows for b in range(per)), dtype="<u4").astype(np.uint64)
    return (((w * np.uint64(2 * NOISE + 1)) >> np.uint64(32)).astype(np.int64) - NOISE).reshape(len(rows), lp)


@dataclass
class Coins:
    dA: np.ndarray
    dB: np.ndarray
    U: np.ndarray
    V: np.ndarray
    perm: np.ndarray

    @staticmethod
    def expand(sh: Shape, seeds: dict[str, bytes]) -> Coins:
        return Coins(bits(seeds["dA"], "dA", sh.m, sh.k), bits(seeds["dB"], "dB", sh.n, sh.k), noise(seeds["U"], "U", sh.n, sh.lp),
                     noise(seeds["V"], "V", sh.m, sh.lp), perms(seeds["P"], sh.k // sh.lp, sh.lp))

    @staticmethod
    def rows(sh: Shape, seeds: dict[str, bytes], ia: range, ja: range) -> Coins:
        """The coins of A's rows ia and B's rows ja (and every permutation)."""
        return Coins(bits_rows(seeds["dA"], "dA", ia, sh.k), bits_rows(seeds["dB"], "dB", ja, sh.k),
                     noise_rows(seeds["U"], "U", ja, sh.lp), noise_rows(seeds["V"], "V", ia, sh.lp),
                     perms(seeds["P"], sh.k // sh.lp, sh.lp))


def seeds_bytes(seeds: dict[str, bytes]) -> bytes:
    return b"".join(seeds[s] for s in STREAMS)


def seeds_from(b: bytes) -> dict[str, bytes]:
    return {s: b[64 * i:64 * i + 64] for i, s in enumerate(STREAMS)}


# ------------------------------------------------------------------------------------------------ forming and values
def wrap8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = ((x + 128) & 255) - 128
    return w, (x - w) // 256


def v_full(sh: Shape, c: Coins, rows: slice | None = None) -> np.ndarray:
    V = c.V if rows is None else c.V[rows]
    return np.tile(V, (1, sh.k // sh.lp))


def u_full(sh: Shape, c: Coins, rows: slice | None = None) -> np.ndarray:
    U = c.U if rows is None else c.U[rows]
    return np.concatenate([U[:, c.perm[tau]] for tau in range(sh.k // sh.lp)], axis=1)


@dataclass
class Formed:
    X: np.ndarray
    wX: np.ndarray
    Wp: np.ndarray
    wW: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray


def form(sh: Shape, A: np.ndarray, B: np.ndarray, c: Coins) -> Formed:
    Ad, Bd = A.astype(np.int64) + c.dA, B.astype(np.int64) + c.dB
    X, wX = wrap8(Ad + v_full(sh, c))
    Wp, wW = wrap8(Bd + u_full(sh, c))
    return Formed(X, wX, Wp, wW, Ad, Bd)


def imatmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a b^T exactly (float64 BLAS: every partial sum here is below 2^53)."""
    assert a.shape[1] * 256 * 256 < 2 ** 53
    return np.rint(a.astype(np.float64) @ b.astype(np.float64).T).astype(np.int64)


def peels(sh: Shape, f: Formed, c: Coins) -> tuple[np.ndarray, np.ndarray]:
    """Z_X (m x L') and Y_W (N x L')."""
    T = sh.k // sh.lp
    At = f.Ad + v_full(sh, c)
    Z = np.zeros((At.shape[0], sh.lp), dtype=np.int64)
    Y = np.zeros((f.Bd.shape[0], sh.lp), dtype=np.int64)
    for tau in range(T):
        blk = slice(tau * sh.lp, (tau + 1) * sh.lp)
        Z[:, c.perm[tau]] += At[:, blk]
        Y += f.Bd[:, blk]
    return Z, Y


@dataclass
class Values:
    T: np.ndarray        # S x m x N, mod 2^16
    O: np.ndarray        # m x N, mod 2^32
    T_full: np.ndarray   # m x N, T_S at full width


def values(sh: Shape, f: Formed, skip: int | None = None) -> Values:
    acc = np.zeros((sh.m, sh.n), dtype=np.int64)
    T = np.zeros((sh.S, sh.m, sh.n), dtype=np.int64)
    for s in range(1, sh.S + 1):
        a, b = sh.seg(s)
        if skip is not None:
            ka, kb = skip * sh.lp, (skip + 1) * sh.lp
            keep = [x for x in range(a, b) if not ka <= x < kb]
            if keep:
                acc += imatmul(f.X[:, keep], f.Wp[:, keep])
        else:
            acc += imatmul(f.X[:, a:b], f.Wp[:, a:b])
        T[s - 1] = acc & 0xFFFF
    O = imatmul(f.Ad, f.Bd) & 0xFFFFFFFF
    return Values(T, O, acc)


def binding(sh: Shape, f: Formed, c: Coins, T_full: np.ndarray) -> np.ndarray:
    """O from the tensor product's full-width T_S, the wraps and the peels (the honest prover's route), mod 2^32."""
    Z, Y = peels(sh, f, c)
    wrap = imatmul(f.wX, f.Wp) + imatmul(f.X, f.wW) + 256 * imatmul(f.wX, f.wW)
    return (T_full + 256 * wrap - imatmul(Z, c.U) - imatmul(c.V, Y)) & 0xFFFFFFFF


# --------------------------------------------------------------------------------------------------- rows and leaves
def words_buffer(sh: Shape, T: np.ndarray) -> bytes:
    """Per tile, per segment, the tile's T_s in row-major cell order, little-endian u16."""
    tm, tn = sh.m // TILE, sh.n // TILE
    w = T.astype("<u2").reshape(sh.S, tm, TILE, tn, TILE).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(w).tobytes()


def o_buffer(O: np.ndarray) -> bytes:
    return np.ascontiguousarray(O.astype("<u4")).tobytes()


def row_bytes(sh: Shape, words: bytes, obuf: bytes, pos: int) -> bytes:
    t, j = divmod(pos, sh.lpt)
    if j < sh.S * sh.row_w:
        off = t * sh.S * 32768 + j * sh.row
        return words[off:off + sh.row]
    h = j - sh.S * sh.row_w
    tm, tn = divmod(t, sh.n // TILE)
    cells = sh.row // 4
    out = []
    for c0 in range(h * cells, (h + 1) * cells, TILE):
        r = c0 // TILE
        off = 4 * ((tm * TILE + r) * sh.n + tn * TILE)
        out.append(obuf[off:off + 4 * min(TILE, (h + 1) * cells - c0)])
    return b"".join(out)


def leaf_h(row: bytes, salt: bytes) -> bytes:
    x = sha512_row_v2_digest(row, 8 * len(row))
    return INST.tree_leaf(KEY, INST.commit_string(KEY, x, salt))


@dataclass
class Committed:
    sh: Shape
    ks: bytes
    off: int
    words: bytes
    obuf: bytes
    hs: list[bytes] = field(default_factory=list)
    levels: list[list[bytes]] = field(default_factory=list)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def salt(self, pos: int) -> bytes:
        return cref.prf_salt(self.ks, self.off + pos)

    def row(self, pos: int) -> bytes:
        return row_bytes(self.sh, self.words, self.obuf, pos)

    def path(self, pos: int) -> tuple[bytes, ...]:
        return tuple(self.levels[d][(pos >> d) ^ 1] for d in range(len(self.levels) - 1))


def commit(sh: Shape, words: bytes, obuf: bytes, ks: bytes, off: int) -> Committed:
    cm = Committed(sh, ks, off, words, obuf)
    dom = sh.domain()
    cm.hs = [leaf_h(cm.row(p), cm.salt(p)) for p in range(sh.leaves)]
    cm.levels = [list(lv) for lv in tree_levels(dom, [row_leaf(dom, p, h, V2) for p, h in enumerate(cm.hs)])]
    return cm


def levels_bytes(levels: list[list[bytes]]) -> bytes:
    return b"".join(b"".join(lv) for lv in levels)


def levels_from(sh: Shape, b: bytes) -> list[list[bytes]]:
    n = sh.leaves
    depth = (n - 1).bit_length() if n > 1 else 0
    out, off = [], 0
    for d in range(depth + 1):
        w = (1 << depth) >> d
        out.append([b[off + 64 * i:off + 64 * i + 64] for i in range(w)])
        off += 64 * w
    assert off == len(b), f"levels: {len(b)} bytes, want {off}"
    return out


def params_file(sh: Shape, ks: bytes) -> bytes:
    return cref.params(sh.leaves, 8 * sh.row, ks, KEY, False, sh.domain().domain_id)


# ------------------------------------------------------------------------------------------------- the stub challenger
class StubChallenger:
    """Issues a call's coins in the protocol's order from a secret the prover never sees: the seeds only after the operand
    roots, the selection and check coins only after the call's root. A live challenger draws them fresh; this one derives
    them from its secret and the transcript so far, so a run is reproducible from the secret."""

    def __init__(self, secret: bytes | None = None):
        self.secret = os.urandom(64) if secret is None else secret
        self.calls: dict[int, dict] = {}

    def _coin(self, label: str, *parts: bytes) -> bytes:
        h = hashlib.sha512(b"accel-v31/stub-challenger/" + label.encode() + b"\0" + self.secret)
        for p in parts:
            h.update(len(p).to_bytes(8, "big") + p)
        return h.digest()

    def seeds(self, call: int, operand_roots: bytes) -> dict[str, bytes]:
        if call in self.calls:
            raise RuntimeError(f"call {call}'s seeds were already issued: a retry never redraws")
        self.calls[call] = {"operands": operand_roots}
        return {s: self._coin("seed/" + s, call.to_bytes(8, "big"), operand_roots) for s in STREAMS}

    def check(self, call: int, root: bytes, sh: Shape, q: int) -> list[tuple[int, int, int]]:
        st = self.calls.get(call)
        if st is None or "root" in st:
            raise RuntimeError(f"call {call}: check coins only once, after its seeds and its root")
        st["root"] = root
        coin = self._coin("check", call.to_bytes(8, "big"), st["operands"], root)
        picks = []
        for b in range(q):
            r = int.from_bytes(hashlib.sha512(coin + b.to_bytes(8, "big")).digest(), "big")
            i, r = r % sh.m, r // sh.m
            j, r = r % sh.n, r // sh.n
            x = r % sh.k
            picks.append((i, j, x // sh.L + 1))
        return picks


def operand_roots(A: np.ndarray, B: np.ndarray) -> bytes:
    """Stand-in for the operands' roots: A is the previous call's committed O under the model's map, B the period's."""
    return hashlib.sha512(b"A" + A.astype(np.int8).tobytes()).digest() + hashlib.sha512(b"B" + B.astype(np.int8).tobytes()).digest()


# --------------------------------------------------------------------------------------------------------- the prover
@dataclass
class Prover:
    sh: Shape
    A: np.ndarray
    B: np.ndarray
    seeds: dict[str, bytes]
    cm: Committed

    def answer(self, i: int, j: int, s: int) -> dict:
        """The proof block's opened rows (each with its salt and path) and the operand rows."""
        sh, cm = self.sh, self.cm
        pos = [sh.word_leaf(i, j, s)[0]] + ([sh.word_leaf(i, j, s - 1)[0]] if s > 1 else []) + ([sh.o_leaf(i, j)[0]] if s == sh.S else [])
        rows = {}
        for p in pos:
            row, salt = cm.row(p), cm.salt(p)
            rows[p] = (row, salt, cm.hs[p] if cm.hs else leaf_h(row, salt), cm.path(p))
        return {"rows": rows, "A": self.A[i], "B": self.B[j]}


def cheat_values(sh: Shape, v: Values, kind: str, rng: np.random.Generator) -> Values:
    if kind.startswith("forge:"):
        parts = kind.split(":")
        s0, frac = int(parts[1]), float(parts[2]) if len(parts) > 2 else 1.0
        T = v.T.copy()
        mask = tile_mask(sh, frac, rng)
        T[s0 - 1][mask] = rng.integers(0, 1 << 16, size=int(mask.sum()))
        return Values(T, v.O, v.T_full)
    if kind.startswith("forgeO"):
        parts = kind.split(":")
        frac = float(parts[1]) if len(parts) > 1 else 1.0
        O = v.O.copy()
        mask = tile_mask(sh, frac, rng)
        O[mask] = rng.integers(0, 1 << 32, size=int(mask.sum()), dtype=np.int64)
        return Values(v.T, O, v.T_full)
    raise ValueError(f"cheat {kind}")


def tile_mask(sh: Shape, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Cells of a fraction of the output tiles (at least one)."""
    tn = sh.n // TILE
    pick = rng.permutation(sh.tiles)[:max(1, round(frac * sh.tiles))]
    mask = np.zeros((sh.m, sh.n), dtype=bool)
    for t in pick:
        a, b = divmod(int(t), tn)
        mask[a * TILE:(a + 1) * TILE, b * TILE:(b + 1) * TILE] = True
    return mask


def prove(sh: Shape, A: np.ndarray, B: np.ndarray, seeds: dict[str, bytes], ks: bytes, off: int = 0,
          cheat: str | None = None) -> tuple[Prover, Values]:
    c = Coins.expand(sh, seeds)
    f = form(sh, A, B, c)
    skip = int(cheat.split(":")[1]) if cheat and cheat.startswith("skip:") else None
    v = values(sh, f, skip)
    if skip is not None:
        # the skipping prover's O is its own T_S through the binding route, which it can afford
        v = Values(v.T, binding(sh, f, c, v.T_full), v.T_full)
    elif cheat:
        v = cheat_values(sh, v, cheat, np.random.default_rng(int.from_bytes(seeds["dA"][:8], "big")))
    cm = commit(sh, words_buffer(sh, v.T), o_buffer(v.O), ks, off)
    return Prover(sh, A, B, seeds, cm), v


# ------------------------------------------------------------------------------------------------------- the verifier
def check_block(sh: Shape, seeds: dict[str, bytes], root: bytes, i: int, j: int, s: int, ans: dict,
                coins: Coins | None = None) -> str | None:
    """The proof block's checks (None: passed; else the first failure)."""
    dom, cmt, lim = sh.domain(), Commitment(root, sh.leaves), VerificationLimits()
    rows = {}
    for pos, (row, salt, h, path) in ans["rows"].items():
        if leaf_h(row, salt) != h:
            return f"leaf {pos}: row and salt don't give its digest"
        if not verify_opening(dom, cmt, Opening(pos, h, path), V2, lim):
            return f"leaf {pos}: path doesn't reach the root"
        rows[pos] = row

    def word(s_: int) -> int:
        if s_ == 0:
            return 0
        p, o = sh.word_leaf(i, j, s_)
        return int.from_bytes(rows[p][o:o + 2], "little")

    c = coins or Coins.expand(sh, seeds)
    Ai, Bj = ans["A"].astype(np.int64)[None, :], ans["B"].astype(np.int64)[None, :]
    one = Coins(c.dA[i:i + 1], c.dB[j:j + 1], c.U[j:j + 1], c.V[i:i + 1], c.perm)
    f = form(sh, Ai, Bj, one)
    a, b = sh.seg(s)
    seg = int((f.X[0, a:b] * f.Wp[0, a:b]).sum())
    if (word(s) - word(s - 1) - seg) % (1 << 16):
        return f"T_{s} - T_{s - 1} isn't the segment's X W'^T mod 2^16"
    if s == sh.S:
        p, o = sh.o_leaf(i, j)
        Ov = int.from_bytes(rows[p][o:o + 4], "little")
        if (Ov - int((f.Ad * f.Bd).sum())) % (1 << 32):
            return "O isn't (A + d_A)(B + d_B)^T mod 2^32"
        Z, Y = peels(sh, f, one)
        wrap = int((f.wX * f.Wp).sum() + (f.X * f.wW).sum() + 256 * (f.wX * f.wW).sum())
        rhs = word(sh.S) + 256 * wrap - int((Z * one.U).sum()) - int((one.V * Y).sum())
        if (Ov - rhs) % (1 << 16):
            return "the binding identity fails mod 2^16"
    return None


def verdict(sh: Shape, ch: StubChallenger, call: int, prover: Prover, q: int) -> tuple[bool, list[str], str]:
    picks = ch.check(call, prover.cm.root, sh, q)
    coins = Coins.expand(sh, prover.seeds)
    fails = []
    for (i, j, s) in picks:
        why = check_block(sh, prover.seeds, prover.cm.root, i, j, s, prover.answer(i, j, s), coins)
        if why:
            fails.append(f"({i}, {j}, s {s}): {why}")
    ok = not fails
    line = (f"{'ACCEPT' if ok else 'REFUSE'} call {call} {sh.m}x{sh.n}x{sh.k} L' {sh.lp} t {sh.t} S {sh.S}: {len(picks)} proof "
            f"blocks opened, {len(picks) - len(fails)} passed, {len(fails)} failed"
            + (f"; first: {fails[0]}" if fails else "") + f"; root {prover.cm.root.hex()[:16]}")
    return ok, fails, line


# ---------------------------------------------------------------------------------------------------------- operands
def operands(sh: Shape, rng: np.random.Generator, kind: str = "llm") -> tuple[np.ndarray, np.ndarray]:
    """int8 operands. ``llm``: rows quantized to +-127 by their absolute maximum, activations from a Student t (heavy tails,
    so few wrap at V +-80) and weights from a Gaussian (more wrap at U +-80), the shape of BET-HYBRID-2's densities;
    ``uniform``: every value on -128..127 (a third of the entries wrap); ``zero``: A = B = 0."""
    if kind == "zero":
        return np.zeros((sh.m, sh.k), dtype=np.int8), np.zeros((sh.n, sh.k), dtype=np.int8)
    if kind == "uniform":
        return (rng.integers(-128, 128, size=(sh.m, sh.k)).astype(np.int8), rng.integers(-128, 128, size=(sh.n, sh.k)).astype(np.int8))
    a = rng.standard_t(3, size=(sh.m, sh.k))
    b = rng.standard_normal(size=(sh.n, sh.k))
    q = lambda x: np.clip(np.rint(127 * x / np.abs(x).max(axis=1, keepdims=True)), -127, 127).astype(np.int8)
    return q(a), q(b)


# ---------------------------------------------------------------------------------------------------------------- CLI
def shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("m", type=int)
    p.add_argument("n", type=int)
    p.add_argument("k", type=int)
    p.add_argument("--lp", type=int, default=256)
    p.add_argument("--t", type=int, default=3)
    p.add_argument("--row", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--operands", default="llm")


def cmd_cpu_run(a: argparse.Namespace) -> int:
    sh = Shape(a.m, a.n, a.k, a.lp, a.t, a.row)
    rng = np.random.default_rng(a.seed)
    A, B = operands(sh, rng, a.operands)
    ch = StubChallenger(None if a.secret is None else bytes.fromhex(a.secret))
    seeds = ch.seeds(0, operand_roots(A, B))
    prover, _ = prove(sh, A, B, seeds, os.urandom(64), 0, a.cheat)
    ok, fails, line = verdict(sh, ch, 0, prover, a.q)
    print(line)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps({"shape": sh.label(), "cheat": a.cheat, "q": a.q, "accepted": ok, "line": line,
                                                   "failures": fails[:16]}, indent=1))
    return 0 if ok == (a.cheat is None) else 1


def cmd_gpu_prep(a: argparse.Namespace) -> int:
    sh = Shape(a.m, a.n, a.k, a.lp, a.t, a.row)
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    A, B = operands(sh, rng, a.operands)
    secret = os.urandom(64)
    ch = StubChallenger(secret)
    seeds = ch.seeds(0, operand_roots(A, B))
    ks = os.urandom(64)
    (d / "A.bin").write_bytes(A.tobytes())
    (d / "B.bin").write_bytes(B.tobytes())
    (d / "seeds.bin").write_bytes(seeds_bytes(seeds))
    (d / "params.bin").write_bytes(params_file(sh, ks))
    (d / "challenger.bin").write_bytes(secret)
    (d / "meta.json").write_text(json.dumps({**sh.label(), "S": sh.S, "L": sh.L, "operands": a.operands, "seed": a.seed,
                                             "salt_off": 0, "cheat": a.cheat}))
    print(json.dumps({"dir": str(d), **sh.label()}))
    return 0


def cmd_gpu_check(a: argparse.Namespace) -> int:
    """The GPU run in DIR (words.bin, O.bin, levels.bin) against the reference, then the verdict on the GPU's own output."""
    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    sh = Shape(meta["m"], meta["n"], meta["k"], meta["lp"], meta["t"], meta["row"])
    A = np.frombuffer((d / "A.bin").read_bytes(), dtype=np.int8).reshape(sh.m, sh.k)
    B = np.frombuffer((d / "B.bin").read_bytes(), dtype=np.int8).reshape(sh.n, sh.k)
    seeds = seeds_from((d / "seeds.bin").read_bytes())
    p = (d / "params.bin").read_bytes()
    ks = p[4 + 3 * 128 + 64:4 + 3 * 128 + 128]
    if p != params_file(sh, ks):
        print(json.dumps({"ok": False, "error": "params.bin isn't this shape's CGP1 file"}))
        return 1
    g_words, g_o, g_lv = (d / "words.bin").read_bytes(), (d / "O.bin").read_bytes(), (d / "levels.bin").read_bytes()
    c = Coins.expand(sh, seeds)
    f = form(sh, A, B, c)
    v = values(sh, f)
    r_words, r_o = words_buffer(sh, v.T), o_buffer(v.O)
    cm = commit(sh, r_words, r_o, ks, meta.get("salt_off", 0))
    r_lv = levels_bytes(cm.levels)
    rep = {"shape": sh.label(), "cheat": meta.get("cheat"), "words_equal": g_words == r_words, "O_equal": g_o == r_o,
           "levels_equal": g_lv == r_lv, "root_gpu": levels_from(sh, g_lv)[-1][0].hex()[:32], "root_ref": cm.root.hex()[:32],
           "binding_identity": bool((binding(sh, f, c, v.T_full) == v.O).all())}
    if not rep["words_equal"]:
        gw, rw = np.frombuffer(g_words, "<u2"), np.frombuffer(r_words, "<u2")
        rep["words_bad"] = int((gw != rw).sum()) if gw.size == rw.size else f"{gw.size} vs {rw.size} words"
    if not rep["O_equal"]:
        go, ro = np.frombuffer(g_o, "<u4"), np.frombuffer(r_o, "<u4")
        rep["O_bad"] = int((go != ro).sum()) if go.size == ro.size else f"{go.size} vs {ro.size}"
    for name, ref in (("X", f.X), ("Wp", f.Wp)):
        fp = d / f"{name}.bin"
        if fp.exists():
            rep[f"{name}_equal"] = fp.read_bytes() == ref.astype(np.int8).tobytes()
    ctl = d / "control"
    if ctl.exists():
        # the no-write control (the same buffers poisoned, nothing run) must fail every comparison
        rep["control_rejected"] = ((ctl / "words.bin").read_bytes() != r_words and (ctl / "O.bin").read_bytes() != r_o
                                   and (ctl / "levels.bin").read_bytes() != r_lv)
    # the protocol's verdict on the GPU's own commitment: the prover's answers come from the GPU's rows and levels
    gcm = Committed(sh, ks, meta.get("salt_off", 0), g_words, g_o)
    gcm.levels = levels_from(sh, g_lv)
    prover = Prover(sh, A, B, seeds, gcm)
    ch = StubChallenger((d / "challenger.bin").read_bytes())
    ch.seeds(0, operand_roots(A, B))
    ok, fails, line = verdict(sh, ch, 0, prover, a.q)
    rep.update(accepted=ok, verdict=line, failures=fails[:8])
    rep["exact"] = rep["words_equal"] and rep["O_equal"] and rep["levels_equal"]
    print(json.dumps(rep))
    print(line)
    (d / "check.json").write_text(json.dumps(rep, indent=1))
    good = (rep["exact"] and ok and rep.get("control_rejected", True)) if not meta.get("cheat") else not ok
    return 0 if good else 1


def cmd_gpu_sample(a: argparse.Namespace) -> int:
    """A bench shape's sampled tiles (DIR from `v31 bench`): each tile's words, O and leaves against the reference, from
    the tile's A and B rows and the call's seeds (the coins of those rows only)."""
    d = Path(a.dir)
    meta = json.loads((d / "sample.json").read_text())
    sh = Shape(meta["m"], meta["n"], meta["k"], meta["lp"], meta["t"], meta["row"])
    tl = Shape(TILE, TILE, sh.k, sh.lp, sh.t, sh.row)
    seeds = seeds_from((d / "seeds.bin").read_bytes())
    p = (d / "params.bin").read_bytes()
    ks = p[4 + 3 * 128 + 64:4 + 3 * 128 + 128]
    rep: dict = {"shape": sh.label(), "params_match": p == params_file(sh, ks), "tiles": []}
    dom = sh.domain()
    for idx, tile in enumerate(meta["tiles"]):
        tm, tn = divmod(tile, sh.n // TILE)
        A = np.frombuffer((d / f"t{idx}_A.bin").read_bytes(), dtype=np.int8).reshape(TILE, sh.k)
        B = np.frombuffer((d / f"t{idx}_B.bin").read_bytes(), dtype=np.int8).reshape(TILE, sh.k)
        c = Coins.rows(sh, seeds, range(tm * TILE, (tm + 1) * TILE), range(tn * TILE, (tn + 1) * TILE))
        f = form(tl, A, B, c)
        v = values(tl, f)
        words, obuf = words_buffer(tl, v.T), o_buffer(v.O)
        g_leaves = (d / f"t{idx}_leaves.bin").read_bytes()
        bad = 0
        for j in range(sh.lpt):
            pos = tile * sh.lpt + j
            leaf = row_leaf(dom, pos, leaf_h(row_bytes(tl, words, obuf, j), cref.prf_salt(ks, pos)), V2)
            bad += leaf != g_leaves[64 * j:64 * j + 64]
        rep["tiles"].append({"tile": tile, "words_equal": (d / f"t{idx}_words.bin").read_bytes() == words,
                             "O_equal": (d / f"t{idx}_O.bin").read_bytes() == obuf, "leaves": sh.lpt, "leaves_bad": bad,
                             "binding_identity": bool((binding(tl, f, c, v.T_full) == v.O).all())})
    rep["exact"] = rep["params_match"] and all(t["words_equal"] and t["O_equal"] and not t["leaves_bad"] and t["binding_identity"]
                                               for t in rep["tiles"])
    print(json.dumps(rep))
    (d / "sample-check.json").write_text(json.dumps(rep, indent=1))
    return 0 if rep["exact"] else 1


def params_name(m: int, n: int, k: int, lp: int, t: int, row: int) -> str:
    return f"{m}x{n}x{k}-lp{lp}-t{t}-r{row}.bin"


def cmd_params(a: argparse.Namespace) -> int:
    """CGP1 params (a fresh Ks each) for `v31 bench`'s shapes, named as it looks them up."""
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    for s in a.shapes:
        m, n, k = (int(x) for x in s.split(":")[-3:])
        sh = Shape(m, n, k, a.lp, a.t, a.row)
        (d / params_name(m, n, k, a.lp, a.t, a.row)).write_bytes(params_file(sh, os.urandom(64)))
    return 0


# D6 v1.1's predictions (art:6926be1c98b7855aca699278551040538e83843d28b72fdf85044cfd291bf3ef), RTX PRO 6000, v3.1 at
# L' 256, t 3, m 8192, overlapped as served: (K, N) -> (model, gamma). V4's 7168 x 64 and the F32 mHC 28672 x 24 aren't
# 128-column tiles and are left out here.
D6 = {
    (4096, 6144): ("8B", 1.70), (4096, 4096): ("8B", 1.71), (4096, 28672): ("8B", 1.70), (14336, 4096): ("8B", 1.29),
    (4096, 128256): ("8B", 1.70),
    (8192, 10240): ("70B", 1.39), (8192, 8192): ("70B", 1.39), (8192, 57344): ("70B", 1.39), (28672, 8192): ("70B", 1.23),
    (8192, 128256): ("70B", 1.39),
    (7168, 43008): ("V4", 1.46), (3072, 50176): ("V4", 1.70), (7168, 1536): ("V4", 1.48), (1536, 65536): ("V4", 2.25),
    (7168, 512): ("V4", 1.52), (4096, 16384): ("V4", 1.70), (16384, 7168): ("V4", 1.29), (7168, 129280): ("V4", 1.46),
    (7168, 2048): ("V4", 1.48), (7168, 1024): ("V4", 1.49), (1536, 8192): ("V4", 2.26), (7168, 384): ("V4", 1.53),
}


def cmd_table(a: argparse.Namespace) -> int:
    """`v31 bench`'s JSON lines beside D6's predictions: gamma overlapped (and serial, and the GEMM alone) and the ratio,
    with the variant (ROW, the commit stream's priority, the carveout, the GEMM's tile counter) and the arms' ms."""
    rows = []
    for path in a.files:
        for line in Path(path).read_text().splitlines():
            if line.startswith("{") and '"gamma_ovl"' in line:
                rows.append(json.loads(line))
    print("| shape | model | m x N x K | variant | lt ms | ovl ms | gamma ovl | gamma call | gamma gemm | D6 | ovl / D6 "
          "| form, wrap, gemm, commit ms | wraps A, B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        model, pred = D6.get((r["k"], r["n"]), ("", float("nan")))
        ms = r["arms"]
        variant = f"r{r.get('row', '?')} {r.get('cprio', 'hi')} c{r.get('carve', -1)} d{r.get('dyn', 0)}"
        parts = ", ".join(f"{ms[x]['ms']:.3f}" if x in ms else "-" for x in ("form", "wrap", "gemm", "commit"))
        print(f"| {r['shape']} | {model} | {r['m']} x {r['n']} x {r['k']} | {variant} | {ms['lt']['ms']:.3f} | {ms['ovl']['ms']:.3f} | "
              f"{r['gamma_ovl']:.2f} | {r['gamma_call']:.2f} | {r['gamma_gemm']:.2f} | {pred:.2f} | {r['gamma_ovl'] / pred:.2f} | "
              f"{parts} | {100 * r['wrap_rate_A']:.2f}%, {100 * r['wrap_rate_B']:.2f}% |")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cpu-run")
    p.add_argument("out")
    shape_args(p)
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--cheat", default=None)
    p.add_argument("--secret", default=None)
    p = sub.add_parser("gpu-prep")
    p.add_argument("dir")
    shape_args(p)
    p.add_argument("--cheat", default=None)
    p = sub.add_parser("gpu-check")
    p.add_argument("dir")
    p.add_argument("--q", type=int, default=64)
    p = sub.add_parser("gpu-sample")
    p.add_argument("dir")
    p = sub.add_parser("params")
    p.add_argument("dir")
    p.add_argument("shapes", nargs="+", help="[name:]M:N:K")
    p.add_argument("--lp", type=int, default=256)
    p.add_argument("--t", type=int, default=3)
    p.add_argument("--row", type=int, default=1024)
    p = sub.add_parser("table")
    p.add_argument("files", nargs="+")
    a = ap.parse_args(argv[1:])
    return {"cpu-run": cmd_cpu_run, "gpu-prep": cmd_gpu_prep, "gpu-check": cmd_gpu_check, "gpu-sample": cmd_gpu_sample,
            "params": cmd_params, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
