"""The frozen shape's J10-R calls: the operands, salt, roots and positions of each (linear, M), and the proof blocks
the fixtures sample. Shared by `fixtures.py` (the expected values, CPU), `check.py` and `bench.py` (the GPU).

* **Linears** (`LINEARS`): the H200 demo's MQA decoder, width 16384, 256 query heads of 64,
  one KV head, MLP 65536: q, kv (wk's rows then wv's), o, up, down, and the head once a step. A call is A [M, K] (token
  rows) times B [N, K] (the linear's weight), its proof blocks the 32 x 32 tiles, proof block u = ri (N / 32) + cj.
* **B** is layer 0's weight of the demo's served weights: the seeded generator (`e2e_h200.served_weights`,
  shape H200, seed `WEIGHTS_SEED`), restated here for its s8 codes and f32 scales
  (`weight_rows`, `weight_scales`): row g of tensor t is the SHA-512 counter stream SHA-512(K_t || u64le(g) || u64le(i)),
  K_t = SHA-512("verity/seeded-weights/v1" 0x00 seed 0x00 t), truncated to the row; tests/ pins its first bytes.
* **A** is the same generator under `X_SEED`, tensor ``x/<linear>``, row r its token r, with `SPECIAL` code patterns in
  the first and the last rows of the call (max-magnitude codes, zero rows, sums past 2^24): row tile 0 and the last row
  tile, which every sample reads.
* **Salt, roots, positions**: one live salt for the step (`SALT`, a period's `pouw.salt` stand-in); root(A) per call and
  root(B) per linear are stand-ins too, SHA-512 of a label (`roots`): the proof block reads them only through seed_c and
  sigma_R, so any 64-byte value checks the same arithmetic. A step's proof blocks are numbered in `ORDER`, the call's first
  at `base` (J10-R's ``call``), proof block u at base + u.
* **Samples** (`picks`): the proof-block grid's four corners, its middle and `PICKS` - 5 drawn proof blocks, at most `PICKS` row
  tiles of 32 rows: 512 of M = 4,096's rows and 512 of 16,384's at the default 16.
"""
from __future__ import annotations

import hashlib
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: (n, k) of the frozen shape's linears
LINEARS = {"q": (16384, 16384), "kv": (128, 16384), "o": (16384, 16384), "up": (65536, 16384), "down": (16384, 65536),
           "head": (256, 16384)}
#: one decoder layer's linears; the head runs once a step
LAYER = ("q", "kv", "o", "up", "down")
ORDER = (*LAYER, "head")
#: decoder layers a step (the H200 shape's)
LAYERS = 52
#: token rows: the skeleton's batch and the block's
MS = (4096, 16384)
#: layer 0's tensors of each linear, rows in this order
SERVED = {"q": ("l0.wq",), "kv": ("l0.wk", "l0.wv"), "o": ("l0.wo",), "up": ("l0.wup",), "down": ("l0.wdown",),
          "head": ("head",)}
#: each served tensor's rows and the scale exponent's shift (wq takes 1/sqrt(d) as -floor(log2 64 / 2))
TENSOR_ROWS = {"l0.wq": 16384, "l0.wk": 64, "l0.wv": 64, "l0.wo": 16384, "l0.wup": 65536, "l0.wdown": 16384, "head": 256}
SCALE_SHIFT = {"l0.wq": -3}
WEIGHTS_FORMAT = "verity/seeded-weights/v1"
#: the demo's public weights seed
WEIGHTS_SEED = b"verity/cmp92/h200/weights/v1"
X_SEED = b"verity/j10r-sm90/activations/v1"
SALT = hashlib.sha512(b"verity/j10r-sm90/fixture/salt/v1").digest()
#: the first proof block's position in each M's step
POS0 = {4096: 1 << 40, 16384: 2 << 40}
T = 32
PICKS = 16
#: code patterns of the first rows of A, and of its last rows in reverse
SPECIAL = ("max", "min", "alt", "zero", "big+", "big-")


def tensor_key(seed: bytes, name: str) -> bytes:
    return hashlib.sha512(WEIGHTS_FORMAT.encode() + b"\0" + seed + b"\0" + name.encode()).digest()


def stream_row(seed: bytes, name: str, row: int, n: int) -> bytes:
    """The first n bytes of generation row `row`'s counter stream of tensor `name` (the generator's `row_bytes`)."""
    key, g, sha = tensor_key(seed, name), row.to_bytes(8, "little"), hashlib.sha512
    return b"".join(sha(key + g + i.to_bytes(8, "little")).digest() for i in range(-(-n // 64)))[:n]


def _codes(seed: bytes, name: str, rows, k: int) -> np.ndarray:
    out = np.empty((len(rows), k), np.int8)
    for j, r in enumerate(rows):
        out[j] = np.frombuffer(stream_row(seed, name, int(r), k), np.int8)
    return out


def weight_rows(linear: str, rows, k: int | None = None) -> np.ndarray:
    """Rows `rows` of the linear's B, s8 [len(rows), K]: its served tensors' generation rows, in `SERVED` order."""
    k = LINEARS[linear][1] if k is None else k
    out, first = np.empty((len(rows), k), np.int8), 0
    rows = np.asarray(rows, np.int64)
    for t in SERVED[linear]:
        sel = np.nonzero((rows >= first) & (rows < first + TENSOR_ROWS[t]))[0]
        if len(sel):
            out[sel] = _codes(WEIGHTS_SEED, t, rows[sel] - first, k)
        first += TENSOR_ROWS[t]
    return out


def weight_scales(linear: str) -> np.ndarray:
    """The linear's f32 channel scales [N]: (127 + E) << 23 | u32le >> 9, E = -(7 + floor(log2 K / 2)) + shift."""
    k = LINEARS[linear][1]
    out = []
    for t in SERVED[linear]:
        n = TENSOR_ROWS[t]
        u = np.frombuffer(stream_row(WEIGHTS_SEED, f"{t}.scale", 0, 4 * n), "<u4").astype(np.uint32)
        e = -(7 + (k.bit_length() - 1) // 2) + SCALE_SHIFT.get(t, 0)
        out.append((np.uint32((127 + e) << 23) | (u >> np.uint32(9))).view(np.float32))
    return np.concatenate(out)


def _pattern(name: str, raw: np.ndarray) -> np.ndarray:
    if name == "big+":
        return (100 + raw.view(np.uint8) % 28).astype(np.int8)
    if name == "big-":
        return (-128 + (raw.view(np.uint8) % 29).astype(np.int16)).astype(np.int8)
    if name == "alt":
        return np.resize(np.array([-128, 127], np.int8), raw.shape[0])
    return np.full(raw.shape[0], {"max": 127, "min": -128, "zero": 0}[name], np.int8)


def special_of(m: int, r: int) -> str | None:
    if r < len(SPECIAL):
        return SPECIAL[r]
    if r >= m - len(SPECIAL):
        return SPECIAL[m - 1 - r]
    return None


def activation_rows(linear: str, m: int, rows) -> np.ndarray:
    """Rows `rows` of the call's A, s8 [len(rows), K]."""
    k = LINEARS[linear][1]
    out = _codes(X_SEED, f"x/{linear}", rows, k)
    for j, r in enumerate(rows):
        if (p := special_of(m, int(r))) is not None:
            out[j] = _pattern(p, out[j])
    return out


def activation_scales(linear: str, m: int) -> np.ndarray:
    """The call's f32 token scales [M], typical (|N(0, 1)| / 127): only the plain arm's epilogue reads them."""
    g = np.random.default_rng(int.from_bytes(hashlib.sha512(f"j10r-sm90/sx/{linear}/{m}".encode()).digest()[:8], "little"))
    return (np.abs(g.standard_normal(m)) / 127).astype(np.float32)


def roots(linear: str, m: int) -> tuple[bytes, bytes]:
    """The call's stand-in root(A) and its linear's root(B)."""
    return (hashlib.sha512(f"verity/j10r-sm90/fixture/root-a/v1\0{linear}/{m}".encode()).digest(),
            hashlib.sha512(f"verity/j10r-sm90/fixture/root-b/v1\0{linear}".encode()).digest())


def units(linear: str, m: int) -> int:
    return (m // T) * (LINEARS[linear][0] // T)


def base(linear: str, m: int) -> int:
    """The call's first proof-block position: the step's proof blocks numbered in `ORDER` from POS0[m]."""
    return POS0[m] + sum(units(x, m) for x in ORDER[:ORDER.index(linear)])


def call(linear: str, m: int) -> dict:
    n, k = LINEARS[linear]
    ra, rb = roots(linear, m)
    return {"linear": linear, "m": m, "n": n, "k": k, "units": units(linear, m), "base": base(linear, m),
            "root_a": ra.hex(), "root_b": rb.hex()}


def picks(linear: str, m: int, count: int = PICKS) -> list[int]:
    nrb, ncb = m // T, LINEARS[linear][0] // T
    rng = random.Random(f"j10r-sm90/picks/{linear}/{m}")
    pts = {(0, 0), (0, ncb - 1), (nrb - 1, 0), (nrb - 1, ncb - 1), (nrb // 2, ncb // 2)}
    while len(pts) < min(count, nrb * ncb):
        pts.add((rng.randrange(nrb), rng.randrange(ncb)))
    return sorted(ri * ncb + cj for ri, cj in pts)


def unit_operands(linear: str, m: int, u: int) -> tuple[np.ndarray, np.ndarray]:
    """Proof block u's 32 rows of A and 32 rows of B, s8."""
    ncb = LINEARS[linear][0] // T
    ri, cj = divmod(u, ncb)
    return activation_rows(linear, m, range(T * ri, T * ri + T)), weight_rows(linear, range(T * cj, T * cj + T))


# -- whole operands, for the GPU ----------------------------------------------------------------------------------------

def _fill(job: tuple) -> None:
    kind, linear, m, path, lo, hi = job
    out = np.load(path, mmap_mode="r+")
    out[lo:hi] = activation_rows(linear, m, range(lo, hi)) if kind == "a" else weight_rows(linear, range(lo, hi))
    out.flush()


def whole(kind: str, linear: str, m: int, scratch: Path, jobs: int) -> np.ndarray:
    """The call's whole A ("a") or B ("b") as an int8 memmap in `scratch`, generated over `jobs` processes once."""
    n, k = LINEARS[linear]
    rows = m if kind == "a" else n
    path = scratch / (f"a-{linear}-{m}.npy" if kind == "a" else f"b-{linear}.npy")
    done = path.with_suffix(".done")
    if not done.exists():
        np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=(rows, k)).flush()
        step = max(1, (1 << 22) // k)
        work = [(kind, linear, m, str(path), lo, min(rows, lo + step)) for lo in range(0, rows, step)]
        with ProcessPoolExecutor(max(1, min(jobs, len(work)))) as ex:
            list(ex.map(_fill, work, chunksize=4))
        done.write_text("")
    return np.load(path, mmap_mode="r")


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()
