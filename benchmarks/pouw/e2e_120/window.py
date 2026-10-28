"""One served int8 window under today's spot check: the protocol's round, its commitments and its honest prover, so an attack
is a prover that departs from `Honest` and the round's verdict says whether the check caught it.

Today's check is `PouwSegCheckDither_v2{S, FOLD}` (benchmarks/pouw/hashed_gemm/segment_e2e.py, pinned by its sha256 as
hgref.py pins it) behind the e2e slice's round (cursor/e2e-slice-ea24's challenger.py and statement.py):

    (a) the weights W (m x k int8, out features x in) are registered: their tile tree's root is known before any round
    (b) the prover commits the window's activations X (n x k int8, tokens x in; X = floor(x / s), committed before the coin)
    (c) the coin sigma, 64 bytes, only after (b)
    (d) the prover commits the call's receipts: a word a thread every S 32-steps (`segment_e2e`'s windowed layout,
        (tile, window, thread), RW words a leaf), the products W' X'^T with W' = W + U (U in ±1..±16, period 32 along k, H1's
        noise on the weights) and X' = X + d (decision 29's dither, d in {0, 1}), U, d and the fold's keys from sigma
    (e) after (d), the draw with fresh coins: a 128 x 128 tile, one of its 512 threads and a 32-step j
    (f) the statement: the thread's W and X tile leaves at j and the receipt leaf holding its window's word, each a
        committed hm96-sha512/row/v2 row whose commit string b || c the public file carries; the public rows (asel, xu,
        wsel, wf, st, rsel) recomputed from sigma and the draw under public salts; the output row o0 held to 1; the proof
        verified (C-Flock --zk, the Lean verifier of record on the pod; here `ideal_proof`, the statement's public file
        computed from its rows)

Nothing is retried or redrawn: a round refused at any step is rejected (`Refused`). Real values plug in at `ServedValues` (W
and X: vLLM's int8 window, once it exists), `RoundCoins` (H1's derivations of U, d and the receipt key from sigma: cgemm.cu's
tags) and `Honest.receipts` (the H1 kernel's receipts, once it writes them in this layout).
"""
from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest

HERE = Path(__file__).resolve().parent
SEG_PATH = HERE.parent / "hashed_gemm" / "segment_e2e.py"
SEG_SHA256 = "4aff8b61dcacf1cf32f11c8ec4e4bd19573fc84e68267a34ac69e8f6f10f28f7"


def _segment():
    name = "verity.core.protocols.pouw.circuit.segment_e2e"
    if name not in sys.modules:
        if hashlib.sha256(SEG_PATH.read_bytes()).hexdigest() != SEG_SHA256:
            raise RuntimeError(f"{SEG_PATH} is not segment.py of cursor/e2e-slice-ea24@3341b0f46: restore it, don't edit it")
        spec = importlib.util.spec_from_file_location(name, SEG_PATH)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


SG = _segment()
INST = hm96.SHA512
KEY = INST.default_key
SALT = INST.salt_bytes
M32 = SG.M32
NCT, RL, RW, NCH, TM = SG.NCT, SG.RL, SG.RW, SG.NCH, 128
ROW_BITS = RL * 32 * 8
TAG = b"verity/e2e-120/v1\0"
TREE_TAG = "verity/e2e-120/tree/v1"
#: fx1's multiplier of entry q = 4 t + e (decision 83; cgemm_ref.fx1_keys: key_mul(1, 32 q) of floored_gemm's ref.py)
FX1 = tuple(SG.mix32(0xA0000000 + 2 * (16 + 32 * q)) | 1 for q in range(32))
#: receipt folds: the check's (`segment_e2e.FOLDS`) and the served design's fx1 (one keyed IMAD a word, then duo)
FOLDS = ("duo", "xin", "fx1")


class Refused(Exception):
    """The round refuses the prover's message: out of order, or a second commitment."""


@dataclass(frozen=True)
class Shape:
    """The window's GEMM: W m x k, X n x k, S 32-steps a receipt window. The default is the tests' stand-in (two tiles, two
    windows a thread at S = 16)."""

    m: int = 256
    n: int = 128
    k: int = 1024
    S: int = 16

    def __post_init__(self):
        if self.m % TM or self.n % TM or self.k % (32 * self.S):
            raise ValueError(f"{self}: m and n multiples of {TM}, k of 32 S")

    @property
    def ks(self) -> int:
        return self.k // 32

    @property
    def nw(self) -> int:
        return self.ks // self.S

    @property
    def tiles_n(self) -> int:
        return self.n // TM

    @property
    def tiles(self) -> int:
        return self.m // TM * self.tiles_n


@dataclass
class ServedValues:
    """A served window's values: the registered weights W and the activations X it serves (int8). Stand-in values until
    vLLM's int8 window exists (`stand_in`)."""

    W: np.ndarray
    X: np.ndarray

    @staticmethod
    def stand_in(shape: Shape, seed: int = 0) -> "ServedValues":
        rng = np.random.default_rng(seed)
        return ServedValues(rng.integers(-100, 101, (shape.m, shape.k), dtype=np.int8),
                      rng.integers(-127, 127, (shape.n, shape.k), dtype=np.int8))


def _h(sigma: bytes, tag: bytes, *ints: int) -> bytes:
    return hashlib.sha512(TAG + tag + b"\0" + sigma + b"".join(i.to_bytes(8, "big") for i in ints)).digest()


@dataclass(frozen=True)
class RoundCoins:
    """What the coin gives a round (stand-ins for H1's derivations): U's 32 values a weight row in ±1..±16, d's bit an
    activation entry, and the receipts' fold key."""

    sigma: bytes

    def u_rows(self, rows) -> np.ndarray:
        out = np.empty((len(rows), 32), np.int8)
        for i, r in enumerate(rows):
            b = np.frombuffer(_h(self.sigma, b"u", int(r)), np.uint8)[:32].astype(np.int16)
            out[i] = np.where(b & 16, -((b & 15) + 1), (b & 15) + 1)
        return out

    def d_bits(self, rows, j: int) -> np.ndarray:
        """d at activation rows ``rows`` and 32-step j, (len(rows), 32) in {0, 1}."""
        return np.stack([np.unpackbits(np.frombuffer(_h(self.sigma, b"d", int(r), j)[:4], np.uint8), bitorder="little")
                         for r in rows])

    def d_all(self, n: int, ks: int) -> np.ndarray:
        return np.concatenate([self.d_bits(range(n), j) for j in range(ks)], axis=1)

    @property
    def seed(self) -> int:
        return int.from_bytes(_h(self.sigma, b"receipt-key")[:4], "big")


def public_salt(sigma: bytes, port: str) -> bytes:
    return hashlib.shake_256(TAG + b"public-salt\0" + sigma + port.encode()).digest(SALT)


# --- commitments -------------------------------------------------------------------------------------------------------

def domain(label: dict, count: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(TREE_TAG, label), -1, RangeIndexedDomain(0, count), hash="sha512")


def commit_string(row: np.ndarray, salt: bytes, bits: int) -> bytes:
    """A u16 row's hm96-sha512/row/v2 commit string b || c (the statement's public file carries one a row)."""
    raw = np.ascontiguousarray(row, "<u2").tobytes()[:(bits + 7) // 8]
    return INST.commit_string(KEY, sha512_row_v2_digest(raw, bits), salt)


class Tree:
    """A committed tree of hm96-sha512/row/v2 rows of ROW_BITS, its salts the committer's (from ``salt_key``)."""

    def __init__(self, label: dict, rows: np.ndarray, salt_key: bytes):
        self.label, self.rows, self.count = label, rows, len(rows)
        ks = hashlib.sha512(TAG + b"salt\0" + salt_key + identity_digest_sha512(TREE_TAG, label)).digest()
        self.salts = [b"".join(hashlib.sha512(ks + p.to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))
                      for p in range(self.count)]
        self.bc = [commit_string(rows[p], self.salts[p], ROW_BITS) for p in range(self.count)]
        dom = domain(label, self.count)
        self.levels = tree_levels(dom, [row_leaf(dom, p, INST.tree_leaf(KEY, bc), V2) for p, bc in enumerate(self.bc)])
        self.root = self.levels[-1][0]

    def path(self, pos: int) -> list[bytes]:
        return [self.levels[d][(pos >> d) ^ 1] for d in range(len(self.levels) - 1)]


def verify_leaf(label: dict, count: int, root: bytes, pos: int, bc: bytes, path: list[bytes]) -> bool:
    if not 0 <= pos < count or len(bc) != 2 * 64:
        return False
    return verify_opening(domain(label, count), Commitment(root, count), Opening(pos, INST.tree_leaf(KEY, bc), tuple(path)), V2,
                          VerificationLimits())


def tile_rows(A: np.ndarray) -> np.ndarray:
    """Decision 46's tile leaves of an operand (rows x k int8): leaf j (rows / RL) + i' is rows RL i' .. RL i' + RL - 1 at
    values 32 j .. 32 j + 31, row-major, as (leaves, RL 16) u16."""
    rows, k = A.shape
    blocks = np.ascontiguousarray(A).reshape(rows // RL, RL, k // 32, 32).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(blocks).reshape(-1, RL * 32).view("<u2")


def receipt_rows(R: np.ndarray) -> np.ndarray:
    """A call's receipts (tiles, NW, NCT) u32 as leaves of RW words: leaf tile NW + w is one window of one tile."""
    return np.ascontiguousarray(R.astype("<u4")).reshape(-1, RW).view("<u2")


def label(session: str, rnd: int, what: str) -> dict:
    return {"session": session, "round": rnd, "tree": what} if what != "W" else {"session": session, "tree": "W"}


# --- the products and their receipts -----------------------------------------------------------------------------------

def thread_slots() -> tuple[np.ndarray, np.ndarray]:
    """Every thread's 32 entries' (row, column) in its tile, (NCT, 32) each (entry q at row_slot(q), col_slot(q))."""
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, q = warp // 4, warp % 4, lane >> 2, lane & 3
    rs = np.array([SG.row_slot(e) for e in range(32)])
    cs = np.array([SG.col_slot(e) for e in range(32)])
    return wm[:, None] * 32 + g[:, None] + 8 * rs, wn[:, None] * 32 + 2 * q[:, None] + 8 * (cs >> 1) + (cs & 1)


def _sx8(v: np.ndarray) -> np.ndarray:
    return ((v.astype(np.int64) + 128) % 256 - 128).astype(np.int64)


def noised(window: ServedValues, coins: RoundCoins, shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """W' = W + U (period 32 along k) and X' = X + d, mod 2^8 as int8, as the check computes them."""
    U = coins.u_rows(range(shape.m))
    Wp = _sx8(window.W.astype(np.int64) + np.tile(U, (1, shape.ks)))
    Xp = _sx8(window.X.astype(np.int64) + coins.d_all(shape.n, shape.ks))
    return Wp, Xp


def prefix_words(Wp: np.ndarray, Xp: np.ndarray, shape: Shape, skip: np.ndarray | None = None) -> np.ndarray:
    """Every thread's prefix words: P[tile, j + 1, thread] = T_j (32 words), P[:, 0] = 0, (tiles, KS + 1, NCT, 32) u32.
    ``skip`` (tiles, KS) True leaves a tile's step j out (its T_j = T_(j-1))."""
    rows, cols = thread_slots()
    P = np.zeros((shape.tiles, shape.ks + 1, NCT, 32), np.uint32)
    for tile in range(shape.tiles):
        tm, tn = divmod(tile, shape.tiles_n)
        a = Wp[TM * tm:TM * tm + TM].reshape(TM, shape.ks, 32)
        b = Xp[TM * tn:TM * tn + TM].reshape(TM, shape.ks, 32)
        inc = np.einsum("isk,jsk->sij", a, b)
        if skip is not None:
            inc[skip[tile]] = 0
        P[tile, 1:] = (np.cumsum(inc, axis=0) & M32).astype(np.uint32)[:, rows, cols]
    return P


def _mix32(x: np.ndarray) -> np.ndarray:
    x = x & M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return ((x << r) | (x >> (32 - r))) & M32


def receipts(P: np.ndarray, seed: int, shape: Shape, fold: str = "duo") -> np.ndarray:
    """Every (tile, window, thread)'s word, (tiles, NW, NCT) u32: the window's keys (`segment_e2e.fold_keys`), its S + 1
    absorbed sets, its final; under fx1 each absorbed word is first multiplied by FX1 (then duo)."""
    if fold not in FOLDS:
        raise ValueError(f"fold {fold!r} is none of {FOLDS}")
    S, nw = shape.S, shape.nw
    out = np.zeros((shape.tiles, nw, NCT), np.uint32)
    ctid = np.arange(NCT, dtype=np.uint64)
    mul = np.array(FX1, np.uint64)
    for tile in range(shape.tiles):
        for w in range(nw):
            idx = ((tile * NCT + ctid[:, None]) * nw + w) * NCH + np.arange(NCH, dtype=np.uint64)[None, :]
            y = _mix32(np.uint64(seed) ^ _mix32(idx))
            z = _mix32(y + np.uint64(SG.GOLDEN))
            for i in range(S + 1):
                p = P[tile, w * S + i].astype(np.uint64)
                if fold == "fx1":
                    p = (p * mul) & M32
                if fold == "xin":
                    for t in range(8):
                        d = [p[:, 4 * t + e] for e in range(4)]
                        y[:, t] = (((_rotl(y[:, t], SG.XROT[0]) ^ d[0]) * SG.XA0[t] + d[1]) & M32)
                        y[:, t] = (((_rotl(y[:, t], SG.XROT[1]) ^ d[2]) * SG.XA1[t] + d[3]) & M32)
                    continue
                for t in range(8):
                    h = 4 * (t % 2)
                    u = y[:, h:h + 4] ^ p[:, 4 * t:4 * t + 4] ^ np.uint64(SG.KK[t])
                    y[:, h:h + 4] = (u * np.uint64(SG.KA[t]) + z[:, h:h + 4]) & M32
                    z[:, h:h + 4] = (_rotl(u, SG.ROT[t]) + y[:, h:h + 4]) & M32
            fin = np.bitwise_xor.reduce(y, axis=1) if fold == "xin" else np.bitwise_xor.reduce(y ^ _rotl(z, 16), axis=1)
            out[tile, w] = fin.astype(np.uint32)
    return out


# --- the statement -----------------------------------------------------------------------------------------------------

def in_rows(S: int, fold: str = "duo") -> list[tuple[str, str, int, list[str]]]:
    """The spot check's rows (statement.py's on cursor/e2e-slice-ea24): ``(port, schema, bits, parameters)``."""
    return [("xa", V2, ROW_BITS, ["a"]), ("asel", V2, 16, ["asel"]), ("xu", V2, 1024, ["u"]), ("wb", V2, ROW_BITS, ["b"]),
            ("wsel", V2, 9 + SG.jsel_bits(S), ["bsel", "jsel", "first"]), ("wf", V2, 256, ["f"]),
            ("p", V2, 1024 * (S + 1), ["p"]), ("st", V2, SG.FOLDS[fold][2], ["st"]), ("rc", V2, 32 * RW, ["rc"]),
            ("rsel", V2, RW, ["rsel"])]


COMMITTED = ("xa", "wb", "rc")
PUBLIC = ("asel", "xu", "wsel", "wf", "st", "rsel")
OUT_BITS = 16


def ok_row() -> np.ndarray:
    return np.ones(1, np.uint16)


def _u16(b) -> np.ndarray:
    b = np.ascontiguousarray(b, np.uint8).reshape(-1)
    return np.concatenate([b, np.zeros(len(b) % 2, np.uint8)]).view("<u2")


def _bits(v) -> np.ndarray:
    return _u16(np.packbits(np.asarray(v, np.uint8).reshape(-1), bitorder="little"))


@dataclass(frozen=True)
class TilePick:
    tile: int
    thread: int
    seg: int

    def place(self, shape: Shape) -> dict:
        """Where the thread reads at segment j: its rows' offset indices in W's and X's leaves (asel, bsel), the leaves'
        positions, its window and j's place in it, and its receipt leaf."""
        warp, lane = self.thread // 32, self.thread % 32
        wm, wn, g, q = warp // 4, warp % 4, lane >> 2, lane & 3
        tm, tn = divmod(self.tile, shape.tiles_n)
        j = self.seg
        return {"ia": 8 * (wm % 2) + g, "ib": 4 * (wn % 2) + q, "w": j // shape.S, "i": j % shape.S,
                "pos_a": j * (shape.m // RL) + 2 * tm + wm // 2, "pos_b": j * (shape.n // RL) + 2 * tn + wn // 2,
                "pos_r": self.tile * shape.nw + j // shape.S, "a_rows": TM * tm + 32 * wm + g + 8 * np.arange(4),
                "b_rows": TM * tn + 32 * wn + 2 * q + 8 * (np.arange(8) >> 1) + (np.arange(8) & 1)}


def public_rows(shape: Shape, coins: RoundCoins, draw: TilePick, fold: str = "duo") -> dict[str, np.ndarray]:
    """The rows the challenger recomputes from the coin and the draw."""
    pl = draw.place(shape)
    st = np.array(SG.fold_keys(fold, coins.seed, draw.tile, draw.thread, pl["w"], shape.nw), "<u4")
    return {"asel": _bits(np.arange(16) == pl["ia"]), "xu": _u16(coins.u_rows(pl["a_rows"]).view(np.uint8)),
            "wsel": _bits(np.concatenate([np.arange(8) == pl["ib"], np.arange(SG.jsel_bits(shape.S)) == pl["i"], [pl["w"] == 0]])),
            "wf": _bits(coins.d_bits(pl["b_rows"], draw.seg)), "st": _u16(st.view(np.uint8)),
            "rsel": _bits(np.arange(RW) == draw.thread)}


def p_row(sets) -> np.ndarray:
    return _u16(np.asarray(sets, "<u4").view(np.uint8))


def word_args(rows: dict[str, np.ndarray], S: int) -> tuple:
    """The rows as `segment_e2e.win_check`'s arguments (the circuit's leaves, decoded)."""
    by = lambda r: np.ascontiguousarray(rows[r], "<u2").view(np.uint8)  # noqa: E731
    bits = lambda r, n: np.unpackbits(by(r), bitorder="little")[:n].tolist()  # noqa: E731
    w32 = lambda r: np.ascontiguousarray(rows[r], "<u2").view("<u4").astype(np.int64).tolist()  # noqa: E731
    sel = bits("wsel", 9 + SG.jsel_bits(S))
    return (by("xa").reshape(RL, 32).tolist(), bits("asel", 16), by("xu").reshape(4, 32).tolist(), by("wb").reshape(RL, 32).tolist(),
            sel[:8], np.array(bits("wf", 256)).reshape(8, 32).tolist(), np.array(w32("p")).reshape(S + 1, 32).tolist(), w32("st"),
            w32("rc"), bits("rsel", RW), sel[8:8 + S], sel[-1])


def word_check(rows: dict[str, np.ndarray], S: int, fold: str = "duo") -> int:
    """Today's check on words: `PouwSegCheckDither_v2{S, FOLD}`'s word view (`segment_e2e.win_check`, dither served)."""
    a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first = word_args(rows, S)
    return SG.win_check(a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first, fold=fold, dither=True)


def ideal_proof(rows: dict[str, np.ndarray], salts: dict[str, bytes], sigma: bytes, S: int, fold: str = "duo") -> dict[str, bytes]:
    """The public file a sound proof's statement carries: every row's commit string (its salt the committer's, or the coin's
    public salt), o0's the check's output. On the pod C-Flock's staging writes it (pub.bin) and the Lean verifier checks
    the proof of it."""
    bits = {r: b for r, _s, b, _ps in in_rows(S, fold)}
    pub = {r: commit_string(rows[r], salts.get(r) or public_salt(sigma, r), bits[r]) for r, *_ in in_rows(S, fold)}
    o0 = np.array([word_check(rows, S, fold)], np.uint16)
    pub["o0"] = commit_string(o0, public_salt(sigma, "o0"), OUT_BITS)
    return pub


# --- the round ---------------------------------------------------------------------------------------------------------

@dataclass
class Opened:
    """What the prover sends after the draw: the statement's rows and their salts (to the prover's proof, never in the clear
    to the challenger), the public file (`ideal_proof`'s, or pub.bin on the pod) and the committed rows' paths."""

    rows: dict[str, np.ndarray]
    salts: dict[str, bytes]
    paths: dict[str, list[bytes]]
    public: dict[str, bytes] | None = None


@dataclass
class RoundResult:
    accepted: bool
    why: str | None
    draw: TilePick | None = None
    sigma: bytes | None = None
    opened: Opened | None = None
    log: list[str] = field(default_factory=list)


class Round:
    """The challenger's side of one round: the order (b) .. (f), the coin and the draw from ``rng`` (the OS generator in a
    run; seeded in a test), and the judgement of the opened statement."""

    def __init__(self, shape: Shape, w_root: bytes, session: str, rnd: int, rng: random.Random, fold: str = "duo"):
        self.shape, self.w_root, self.session, self.rnd, self.rng, self.fold = shape, w_root, session, rnd, rng, fold
        self.x_root = self.r_root = self.sigma = self.draw_ = None
        self.log: list[str] = []

    def operands(self, x_root: bytes) -> bytes:
        """(b) then (c): the activations' root, answered by the coin."""
        if self.sigma is not None or self.x_root is not None:
            raise Refused("operands: once a round, before its coin")
        self.x_root = x_root
        self.sigma = self.rng.randbytes(64)
        self.log.append("operands, coin")
        return self.sigma

    def receipts(self, r_root: bytes) -> TilePick:
        """(d) then (e): the receipts' root, answered by the draw."""
        if self.sigma is None or self.r_root is not None:
            raise Refused("receipts: once a round, after its coin")
        self.r_root = r_root
        sh = self.shape
        self.draw_ = TilePick(self.rng.randrange(sh.tiles), self.rng.randrange(NCT), self.rng.randrange(sh.ks))
        self.log.append(f"receipts, draw {self.draw_}")
        return self.draw_

    def judge(self, op: Opened, proof_ok: bool = True) -> str | None:
        """(f): why the statement is rejected, or None. ``proof_ok``: the proof's verdict (the Lean verifier's on the pod;
        `ideal_proof` here, sound by construction)."""
        if self.draw_ is None:
            return "opening before its draw"
        sh, d, coins = self.shape, self.draw_, RoundCoins(self.sigma)
        pub = op.public if op.public is not None else ideal_proof(op.rows, op.salts, self.sigma, sh.S, self.fold)
        bits = {r: b for r, _s, b, _ps in in_rows(sh.S, self.fold)}
        for port, row in public_rows(sh, coins, d, self.fold).items():
            if pub.get(port) != commit_string(row, public_salt(self.sigma, port), bits[port]):
                return f"{port}: not the coin's row"
        if pub.get("o0") != commit_string(ok_row(), public_salt(self.sigma, "o0"), OUT_BITS):
            return "o0: the statement's output is not 1"
        pl = d.place(sh)
        trees = {"xa": (label(self.session, 0, "W"), sh.m // RL * sh.ks, self.w_root, pl["pos_a"]),
                 "wb": (label(self.session, self.rnd, "X"), sh.n // RL * sh.ks, self.x_root, pl["pos_b"]),
                 "rc": (label(self.session, self.rnd, "R"), sh.tiles * sh.nw, self.r_root, pl["pos_r"])}
        names = {"xa": "the registered weights' tile leaf", "wb": "the activations' committed tile leaf",
                 "rc": "the committed receipt leaf"}
        for port, (lab, count, root, pos) in trees.items():
            if not verify_leaf(lab, count, root, pos, pub.get(port, b""), op.paths.get(port, [])):
                return f"{port}: not {names[port]} {pos}"
        if not proof_ok:
            return "proof: rejected"
        return None


class Honest:
    """The honest prover of one session: the registered weights' tree once, then per round the activations' tree, the
    receipts under the coin and the statement at the draw. ``receipt_fold``: the fold its receipts use (the check's, duo,
    by default; H1 + fx1's kernel folds fx1)."""

    def __init__(self, window: ServedValues, shape: Shape, session: str = "e2e-120", salt_key: bytes = b"stand-in",
                 receipt_fold: str = "duo"):
        self.window, self.shape, self.session, self.salt_key, self.receipt_fold = window, shape, session, salt_key, receipt_fold
        self.w_tree = Tree(label(session, 0, "W"), tile_rows(window.W), salt_key)
        self.open_w_tree = self.w_tree               # the tree xa is opened from (the registered one; a swap opens another)
        self.x_trees: dict[tuple, Tree] = {}

    def x_tree(self, rnd: int, X: np.ndarray) -> Tree:
        key = (rnd, hashlib.sha256(X.tobytes()).digest())
        if key not in self.x_trees:
            self.x_trees[key] = Tree(label(self.session, rnd, "X"), tile_rows(X), self.salt_key)
        return self.x_trees[key]

    # the steps an attack overrides
    def served_weights(self) -> np.ndarray:
        return self.window.W

    def products(self, coins: RoundCoins) -> np.ndarray:
        Wp, Xp = noised(ServedValues(self.served_weights(), self.window.X), coins, self.shape)
        return prefix_words(Wp, Xp, self.shape)

    def receipts(self, P: np.ndarray, coins: RoundCoins) -> np.ndarray:
        return receipts(P, coins.seed, self.shape, self.receipt_fold)

    def witness(self, P: np.ndarray, R: np.ndarray, coins: RoundCoins, draw: TilePick, rows: dict) -> list[list[int]]:
        pl = draw.place(self.shape)
        return P[draw.tile, pl["w"] * self.shape.S:pl["w"] * self.shape.S + self.shape.S + 1, draw.thread].astype(np.int64).tolist()

    def run(self, rd: Round) -> Opened:
        sh = self.shape
        xt = self.x_tree(rd.rnd, self.window.X)
        coins = RoundCoins(rd.operands(xt.root))
        P = self.products(coins)
        R = self.receipts(P, coins)
        rt = Tree(label(self.session, rd.rnd, "R"), receipt_rows(R), self.salt_key)
        draw = rd.receipts(rt.root)
        return self.open(draw, coins, P, R, xt, rt)

    def open(self, draw: TilePick, coins: RoundCoins, P: np.ndarray, R: np.ndarray, xt: Tree, rt: Tree) -> Opened:
        pl = draw.place(self.shape)
        wt = self.open_w_tree
        rows = {"xa": wt.rows[pl["pos_a"]], "wb": xt.rows[pl["pos_b"]], "rc": rt.rows[pl["pos_r"]],
                **public_rows(self.shape, coins, draw)}
        rows["p"] = p_row(self.witness(P, R, coins, draw, rows))
        salts = {"xa": wt.salts[pl["pos_a"]], "wb": xt.salts[pl["pos_b"]], "rc": rt.salts[pl["pos_r"]],
                 "p": hashlib.shake_256(TAG + b"p-salt\0" + self.salt_key + coins.sigma).digest(SALT)}
        return Opened(rows, salts, {"xa": wt.path(pl["pos_a"]), "wb": xt.path(pl["pos_b"]), "rc": rt.path(pl["pos_r"])})


def play(prover, shape: Shape, rnd: int = 1, seed: int | None = None, fold: str = "duo") -> RoundResult:
    """One round of ``prover`` against today's check, its coins and draw from ``seed`` (None: the OS generator)."""
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    rd = Round(shape, prover.w_tree.root, prover.session, rnd, rng, fold)
    try:
        op = prover.run(rd)
    except Refused as e:
        return RoundResult(False, f"refused: {e}", rd.draw_, rd.sigma, log=rd.log)
    why = rd.judge(op)
    return RoundResult(why is None, why, rd.draw_, rd.sigma, op, rd.log)
