"""Pearl-C's fixed-shape bench, its CPU half (needs verity and numpy): the shapes, their keys and input seeds, and for a few
sampled 64 x 64 tiles per shape the bytes the H100 kernels must write there, from a numpy twin of the kernels:

* `gen_f32`'s inputs (A's real rows; filler rows are zero);
* §5's forming with its noise atoms batched, the chain, the fused A′·F_B and B̃·F_A, the peel factors and the peel;
* the leaf and Pearl's tickets, and the A row's leaf in A's tree, all in hashing format -h1 (the verifier's: BLAKE3
  messages and leaves, frame-b3 trees);
* the adopted baseline's y (e4m3(αx) through the same chain, then the glue).

bench.py compares the device's buffers with them before it times a shape.  benchmarks/pouw/tests/test_pearl_c_bench.py holds
the twin to `pearl_c`'s scalar reference and the fixture, and `gen_f32` to its CUDA source.  Every step runs on a device
record (`pearl_c_device`, the H100's by default; the sm_120 arm uses SM120 and SM120_UNPROMOTED), and `tile` takes the rows
themselves when the caller generated them.

  uv run python benchmarks/pouw/pearl_c/twin.py OUT.json [label ...]     (default: every shape in SHAPES)
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from functools import partial
import multiprocessing
from pathlib import Path

import numpy as np
from verity.ml.kernels.tc import group_sum_total_batch
from verity.core.protocols.pouw.audit import ROW_SCHEMA, Epoch, Workload, activation_binding, rows_domain, transcript_domain
from verity.core.protocols.pouw.protocol import Matmul
from verity.core.protocols.pouw.protocol import Shape as MShape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c_u as U
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity.core.protocols.pouw.schemes.pearl_c_device import H100, Device

SCHEMA = "verity-pouw/pearl-c-bench/v2"
T = C.TILE                        # the leaf's tile, 64 x 64
ALIGN = 128                       # the kernels' granularity in m and n
HASHING = "h1"
QWEN7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}   # (n, k) of each linear
PREFILL_M, DECODE_M = 2048, 32


@dataclass(frozen=True)
class Shape:
    label: str
    m_real: int                   # the matmul's rows; the kernels run m = m_real rounded up to 128, the rest zero filler
    n: int
    k: int

    @property
    def m(self) -> int:
        return -(-self.m_real // ALIGN) * ALIGN


SHAPES = ([Shape(f"w1-{s}", s, s, s) for s in (2048, 4096, 8192, 16384, 32768)]
          + [Shape(f"qwen7b-{name}-{phase}", m, n, k) for phase, m in (("prefill", PREFILL_M), ("decode", DECODE_M))
             for name, (n, k) in QWEN7B.items()])


def _h(shape: Shape, role: str) -> bytes:
    return hashlib.sha256(f"{SCHEMA}/{shape.label}/{shape.m_real}x{shape.n}x{shape.k}/{role}".encode()).digest()


def seeds(shape: Shape) -> dict:
    """seed_A, seed_B (the scheme's noise seeds, fixed per shape), the salt A's rows are committed under (unit 0), and
    gen_f32's 31-bit seeds for A and B."""
    return {"seed_a": _h(shape, "seed-A"), "seed_b": _h(shape, "seed-B"), "salt": _h(shape, "salt"), "root_b": _h(shape, "root-B"),
            "gen_a": int.from_bytes(_h(shape, "gen-A")[:4], "little") >> 1,
            "gen_b": int.from_bytes(_h(shape, "gen-B")[:4], "little") >> 1}


def samples(shape: Shape) -> list[tuple[int, int]]:
    """The first tile, the last tile with real rows, and one keyed by the shape: (first row, first column)."""
    tm, tn = -(-shape.m_real // T), shape.n // T
    key = _h(shape, "samples")
    mid = (int.from_bytes(key[:4], "little") % tm * T, int.from_bytes(key[4:8], "little") % tn * T)
    return sorted({(0, 0), mid, ((tm - 1) * T, (tn - 1) * T)})


# ---- the kernels' pieces in numpy -------------------------------------------------------------------------------------

def mix32(x: np.ndarray) -> np.ndarray:
    x = np.array(x, np.uint32)
    x ^= x >> 16
    x *= np.uint32(0x7FEB352D)
    x ^= x >> 15
    x *= np.uint32(0x846CA68B)
    x ^= x >> 16
    return x


def gen_f32(rows: np.ndarray, k: int, seed: int) -> np.ndarray:
    """pearl_c.cu's gen_f32 at the given rows of a (·, k) matrix: FP32 words."""
    w = (np.asarray(rows, np.uint64)[:, None] * np.uint64(k) + np.arange(k, dtype=np.uint64)[None, :])
    h = mix32(np.uint32(seed) ^ mix32(((w * np.uint64(2) + np.uint64(1)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)))
    b = h & np.uint32(0xFF)
    code = np.where(b == 0x80, np.uint32(0x81), b).astype(np.uint8).view(np.int8)
    scale = ((np.uint32(123) - (h >> 8) % np.uint32(5)) << 23).astype(np.uint32).view(np.float32)
    x = (code.astype(np.float32) * scale).view(np.uint32)
    return np.where(code != 0, x | (h >> 16), np.uint32(0)).astype(np.uint32)


def fp8(words: np.ndarray) -> np.ndarray:
    """Pearl's f32_to_fp8_e4m3 (`pearl_kw.f32_to_fp8`: nearest-even, saturating to 448) on FP32 words."""
    w = np.asarray(words, np.uint32)
    if ((w >> 23) & 0xFF == 0xFF).any():
        raise P.Reject("a non-finite FP32 word")
    e = ((w >> 23) & 0xFF).astype(np.int64) - 127
    small = np.rint(np.where(e < -6, np.abs(w.view(np.float32)).astype(np.float64), 0.0) * 512.0).astype(np.int64)
    fm = (w & 0x7FFFFF).astype(np.int64)
    mant, rem = fm >> 20, fm & 0xFFFFF
    mant = mant + ((rem > 0x80000) | ((rem == 0x80000) & ((mant & 1) == 1)))
    exp = e + 7 + (mant == 8)
    mant = np.where(mant == 8, 0, mant)
    big = np.where((exp > 15) | ((exp == 15) & (mant >= 7)), 0x7E, (exp << 3) | mant)
    return (((w >> 31) << 7) | np.where(e < -6, small, big)).astype(np.uint8)


def f32(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, np.uint32).view(np.float32)


def chain(a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray, block: int = 1 << 16, device: Device = H100
          ) -> np.ndarray:
    """`pearl_c.chain` on `device` for the cells (ii, jj) of code matrices a and b: +0 per window of G atoms, one FP32 add
    per window (on the H100, 128 products); unpromoted, one accumulator over all of k."""
    out = np.empty(len(ii), np.uint32)
    k = a.shape[1]
    window = device.window if device.g else k
    for lo in range(0, len(ii), block):
        ci, cj = ii[lo:lo + block], jj[lo:lo + block]
        total = np.zeros(len(ci), np.uint32)
        for ws in range(0, k, window):
            local = np.zeros(len(ci), np.uint32)
            for s in range(ws, min(ws + window, k), C.SLICE):
                local = group_sum_total_batch(device.atom, local, a[ci, s:s + C.SLICE], b[cj, s:s + C.SLICE])[0]
            total = (f32(total) + f32(local)).view(np.uint32) if device.g else local
        if device.g:
            total[total == 0x80000000] = 0
        out[lo:lo + block] = total
    return out


def cells(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return ii.ravel(), jj.ravel()


@dataclass
class Side:
    """Rows formed by v1 as the kernels form them."""

    words: np.ndarray             # rows x k FP32 input
    alpha: np.ndarray             # FP32 words
    beta: np.ndarray
    ep: np.ndarray                # rows x 32, E′ codes
    codes: np.ndarray             # rows x k, E4M3: e4m3(atom(α·x, E′, F line))
    quant: np.ndarray             # rows x k, E4M3: the adopted quantizer's e4m3(α·x)


def form(words: np.ndarray, e_rows: list[list[int]], basis: np.ndarray, block: int = 1 << 18, device: Device = H100) -> Side:
    """`pearl_c.form_v1` with the noise atoms batched: per row the pinned statistics and E′, then per element one step of
    the device's atom with α·x as its accumulator, then the cast."""
    rows, k = words.shape
    stats = [C.s5_stats([int(v) for v in row]) for row in words]
    alpha, beta = (np.array([s[i] for s in stats], np.uint32) for i in (2, 3))
    ep = np.array([C.eprime(int(b), e) for b, e in zip(beta, e_rows)], np.int64).reshape(rows, U.R)
    acc = (f32(words) * f32(alpha)[:, None]).view(np.uint32).ravel()
    ii, tt = np.divmod(np.arange(rows * k), k)
    d = np.empty(rows * k, np.uint32)
    for lo in range(0, rows * k, block):
        sl = slice(lo, lo + block)
        d[sl] = group_sum_total_batch(device.atom, acc[sl], ep[ii[sl]], basis[tt[sl]])[0]
    return Side(words, alpha, beta, ep.astype(np.uint8), fp8(d).reshape(rows, k), fp8(acc).reshape(rows, k))


def rne_bf16(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.uint64)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16).astype(np.uint16)


# ---- one shape --------------------------------------------------------------------------------------------------------

@dataclass
class Job:
    """What every sampled tile of a shape shares: the bases F_A and F_B (k x 32 codes, at v1's norm) and G."""

    shape: Shape
    seeds: dict
    fa: np.ndarray
    fb: np.ndarray
    g: np.ndarray
    device: Device = field(default=H100)


def job(shape: Shape, device: Device = H100, sd: dict | None = None) -> Job:
    sd = sd or seeds(shape)
    fa, fb = (np.array(C._basis(sd["seed_b"], side, shape.k, C.LINE_NORM_V1), np.int64) for side in (0, 1))
    gi, gj = cells(U.R, U.R)
    g = chain(fb.T.copy(), fa.T.copy(), gi, gj, device=device).reshape(U.R, U.R)
    return Job(shape, sd, fa, fb, g, device)


def a_words(shape: Shape, sd: dict, rows: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), shape.k), np.uint32)
    real = rows < shape.m_real
    out[real] = gen_f32(rows[real], shape.k, sd["gen_a"])
    return out


@dataclass
class Tile:
    r0: int
    c0: int
    a: Side
    b: Side
    c: np.ndarray                 # 64 x 64 FP32 words
    u: np.ndarray
    leaf: bytes
    tickets: list[bytes]          # the tile's 16 subtiles, row-major
    y_adopt: np.ndarray           # 64 x 64 BF16 words
    row_leaf: bytes               # row r0's leaf in A's frame-b3 tree

    def expected(self) -> dict:
        sha = lambda raw: hashlib.sha256(raw).hexdigest()  # noqa: E731
        return {"tile": [self.r0, self.c0], "leaf": self.leaf.hex(), "c": sha(self.c.astype("<u4").tobytes()),
                "u": sha(self.u.astype("<u4").tobytes()), "tickets": sha(b"".join(self.tickets)),
                "y_adopt": sha(self.y_adopt.astype("<u2").tobytes()), "alpha_a": sha(self.a.alpha.astype("<u4").tobytes()),
                "alpha_b": sha(self.b.alpha.astype("<u4").tobytes()), "row_leaf": sha(self.row_leaf)}


def tile(jb: Job, r0: int, c0: int, a_rows: np.ndarray | None = None, b_rows: np.ndarray | None = None) -> Tile:
    """The tile at (r0, c0) as the kernels compute it on the job's device, from its 64 A rows and 64 B rows alone (FP32
    words; `gen_f32`'s unless given)."""
    shape, sd, k, dev = jb.shape, jb.seeds, jb.shape.k, jb.device
    rows, cols = np.arange(r0, r0 + T), np.arange(c0, c0 + T)
    a_rows = a_words(shape, sd, rows) if a_rows is None else a_rows
    b_rows = gen_f32(cols, k, sd["gen_b"]) if b_rows is None else b_rows
    a = form(a_rows, [P.sample_line(sd["seed_a"], 0, 0, int(i)) for i in rows], jb.fa, device=dev)
    b = form(b_rows, [P.sample_line(sd["seed_b"], 1, 0, int(j)) for j in cols], jb.fb, device=dev)
    ri, rq = cells(T, U.R)
    af = chain(a.codes.astype(np.int64), jb.fb.T.copy(), ri, rq, device=dev).reshape(T, U.R)
    bf = chain(b.codes.astype(np.int64), jb.fa.T.copy(), ri, rq, device=dev).reshape(T, U.R)
    bf16 = lambda ep: [[C._bf16_of_code(int(x)) for x in row] for row in ep]  # noqa: E731
    pa = np.array([U.pa_row(e, [int(v) for v in f]) for e, f in zip(bf16(a.ep), af)], np.int64)
    pb = np.array([U.pb_row(e, [int(v) for v in f], jb.g.tolist()) for e, f in zip(bf16(b.ep), bf)], np.int64)
    ti, tj = cells(T, T)
    c = chain(a.codes.astype(np.int64), b.codes.astype(np.int64), ti, tj, device=dev)
    u = c.copy()
    for s in range(0, U.PEEL_K, 16):
        u = group_sum_total_batch(dev.peel, u, pa[ti, s:s + 16], pb[tj, s:s + 16])[0]
    c, u = c.reshape(T, T), u.reshape(T, T)
    checked = C.Checked(c.tolist(), u.tolist())
    tickets = C.PearlC(forming="v1").tickets(C.Noise(sd["seed_a"], sd["seed_b"]), checked)
    ia, ib = (np.float32(1.0) / f32(side.alpha) for side in (a, b))
    q = chain(a.quant.astype(np.int64), b.quant.astype(np.int64), ti, tj, device=dev).reshape(T, T)
    y = rne_bf16(((f32(q) * ia[:, None]).astype(np.float32) * ib[None, :]).astype(np.float32).view(np.uint32))
    row_leaf = a_domain(shape, sd).leaf(r0, r0, ROW_SCHEMA, a.words[0].astype("<u4").tobytes())
    return Tile(r0, c0, a, b, c, u, C.leaf(checked, HASHING), tickets, y, row_leaf)


def a_domain(shape: Shape, sd: dict):
    """The domain of A's rows (`verity.core.protocols.pouw.audit`, frame-b3): unit 0 under the shape's salt, all m rows filler included."""
    return rows_domain(activation_binding(sd["salt"], 0), 0, shape.m, "blake3")


def t_domain(shape: Shape, sd: dict):
    """The tile tree's domain for a one-unit epoch over the shape's tiles."""
    epoch = Epoch(Workload((Matmul(0, MShape(shape.m, shape.k, shape.n), 0),)), {0: sd["root_b"]}, sd["salt"])
    return transcript_domain(epoch, (shape.m // T) * (shape.n // T), "blake3")


def keys(shape: Shape, sd: dict) -> dict[str, str]:
    """The kernels' 32-byte keys: the noise lines' (A's, B's), the tickets', the two trees' domain ids and the -h1 labels'."""
    return {"line_a": P._line_key(sd["seed_a"]).hex(), "line_b": P._line_key(sd["seed_b"]).hex(),
            "jackpot": P.subkey("jackpot", sd["seed_a"]).hex(), "dom_a": a_domain(shape, sd).domain_id.hex(),
            "dom_t": t_domain(shape, sd).domain_id.hex(), "k_msg": C.H1_KEYS["message"].hex(), "k_leaf": C.H1_KEYS["leaf"].hex()}


def entry(shape: Shape, expected: list[dict]) -> dict:
    sd = seeds(shape)
    return {"label": shape.label, "m_real": shape.m_real, "m": shape.m, "n": shape.n, "k": shape.k,
            "seeds": {"seed_a": sd["seed_a"].hex(), "seed_b": sd["seed_b"].hex(), "salt": sd["salt"].hex(), "root_b": sd["root_b"].hex()},
            "gen": {"a": sd["gen_a"], "b": sd["gen_b"]},
            "keys": keys(shape, sd), "samples": expected}


def _tile_job(args) -> dict:
    jb, r0, c0 = args
    return tile(jb, r0, c0).expected()


def build(shapes: list[Shape], processes: int | None = None, device: Device = H100) -> dict:
    # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
    with multiprocessing.get_context("fork").Pool(processes) as pool:
        jobs = pool.map(partial(job, device=device), shapes)
        work = [(jb, r0, c0) for jb in jobs for r0, c0 in samples(jb.shape)]
        done = pool.map(_tile_job, work, chunksize=1)
    per = {s.label: [] for s in shapes}
    for (jb, _, _), exp in zip(work, done):
        per[jb.shape.label].append(exp)
    return {"schema": SCHEMA, "scheme": C.PearlC(forming="v1", device=device, hashing=HASHING).name, "tile": T, "hashing": HASHING,
            "shapes": [entry(s, per[s.label]) for s in shapes]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    want = set(sys.argv[2:])
    chosen = [s for s in SHAPES if not want or s.label in want]
    out = build(chosen)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(f"bench {len(out['shapes'])} shapes, {sum(len(s['samples']) for s in out['shapes'])} sampled tiles -> {sys.argv[1]}")
