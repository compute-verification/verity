"""Option A's prover side (CA, 21 Oct 16:04Z): a drawn leaf's 32 lane partials, recomputed after the draw from what serving
committed, bit for bit the words the served receipt XORed.  Serving is unchanged: one receipt a 32 × 32 leaf (`receipts`).
The proof of a drawn leaf is 32 lane sub-statements and a combine that checks the XOR of their partials is the committed
receipt; this module computes those partials.

A window (`Window`) holds each call's B [N, K] (s8, the int8 pin's weight) and B root, the call's first position in a step
(`offs`) and the positions a step (`u_step`); a step (`Step`) its calls' committed A [t, K] (s8, every slot), their A
roots, its live rows m and its salt, `h1tile.live_salt` of its coin (`Step.from_coin`).  A leaf (`Leaf`) is unit u of
call w at step s: row block u // (N / 32), column block u % (N / 32), position s u_step + offs[w] + u, seeds (seed_j, σ_R)
from (salt, position, A root, B root) (`seeds.seed` / `receipt_seed`).

`lane_partials(window, leaf)` is (32, W) uint32, W = 2 (KS + m) (KS = K / 32, m = `segment.n_digits(KS)`: W = 260 at
K = 4096, 902 at K = 14336), the layout `kh_lanes` writes a leaf's record in:

* lane order: row l is lane l = 4 g + q (g = l >> 2, q = l & 3), the lane that holds the tile's rows g + 8 s (s = 0..3)
  and columns 2 q + 8 (s >> 1) + (s & 1) (s = 0..7) (`h1tile.Layout(32, 32, K).rows` / `cols`), whose 16 fold keys are
  block l of σ_R's stream (`h1tile.fold_keys`);
* word order: word 2 s is the XOR of the lane's 8 duo chains' y after it absorbs set s, word 2 s + 1 of their z, sets
  s = 0 .. KS - 1 the prefix sums T_s, s = KS the O' set and KS + 1 .. KS + m - 1 the digits' D_d (`h1tile.lane_sets`).

So the XOR over the 32 rows is the leaf's receipt (`combine`).  A leaf in a dead row block (32 · row block >= m) has the
zero receipt and zero partials (`PouwH1Tile_v4`'s dead tile, as serving writes it).

Two backends: "cpu", receipts.py's chain split by lane (`tile_receipts(lanes=True)` on hashlib's seeds and noise, the
reference the served units kernels are gated against), and "gpu", h1u.cu's `kh_lanes` (`Kernel`: the fused units kernel
in its own instantiation, run on just the drawn leaves' gathered operands, batched, their seeds from `kh_seeds_at`).
"""
from __future__ import annotations

import ctypes as C
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import receipts as RC  # noqa: E402

from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402

LANES = RC.LANES


@dataclass(frozen=True)
class Leaf:
    """Unit `unit` (row-block-major within the call) of call `call` at step `step`."""

    step: int
    call: int
    unit: int


@dataclass
class Step:
    """A step's committed A [t, K] s8 a call (every slot, as its A table commits them), their A roots, its live rows m
    and its salt."""

    a: list[torch.Tensor]
    a_roots: list[bytes]
    m: int
    salt: bytes

    @classmethod
    def from_coin(cls, a: list[torch.Tensor], a_roots: list[bytes], m: int, commitment: bytes, coin: bytes, index: int,
                  round_root: bytes) -> "Step":
        """The step whose units' salt is `h1tile.live_salt` of its coin (the challenger's opening at `index`, flipped after
        `round_root`)."""
        return cls(a, a_roots, m, HT.live_salt(commitment, coin, index, round_root))


@dataclass
class Window:
    """A window's calls: B [N, K] s8 and its root a call, call w's first position in a step, the positions a step, and
    its steps by number."""

    b: list[torch.Tensor]
    b_roots: list[bytes]
    offs: list[int]
    u_step: int
    steps: dict[int, Step] = field(default_factory=dict)

    def k(self, call: int) -> int:
        return self.b[call].shape[1]

    def words(self, call: int) -> int:
        return 2 * RC.layout(self.k(call)).sets


@dataclass(frozen=True)
class _At:
    leaf: Leaf
    ri: int
    cj: int
    position: int
    live: bool


def _at(win: Window, leaf: Leaf) -> _At:
    st = win.steps[leaf.step]
    a, b = st.a[leaf.call], win.b[leaf.call]
    if a.shape[1] != b.shape[1] or b.shape[0] % RC.C_:
        raise ValueError(f"call {leaf.call}: A {tuple(a.shape)} and B {tuple(b.shape)} are no call's")
    ncb = b.shape[0] // RC.C_
    if not 0 <= leaf.unit < (a.shape[0] // RC.R) * ncb:
        raise IndexError(f"{leaf}: call {leaf.call} has {(a.shape[0] // RC.R) * ncb} units")
    ri, cj = divmod(leaf.unit, ncb)
    return _At(leaf, ri, cj, leaf.step * win.u_step + win.offs[leaf.call] + leaf.unit, RC.R * ri < st.m)


def _seeds(win: Window, ats: list[_At]) -> tuple[list[bytes], list[bytes]]:
    sj, sr = [], []
    for x in ats:
        st = win.steps[x.leaf.step]
        ra, rb = st.a_roots[x.leaf.call], win.b_roots[x.leaf.call]
        sj.append(SD.seed(st.salt, x.position, ra, rb))
        sr.append(SD.receipt_seed(st.salt, x.position, ra, rb))
    return sj, sr


def _be(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


def _seed_inputs(win: Window, ats: list[_At], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`kh_seeds_at`'s inputs: a record (kh_seeds' data: SEED_TAG, RECEIPT_TAG, salt, root_a, root_b, 30 big-endian
    words) a step and call, each leaf's record index and its position."""
    recs: dict[tuple[int, int], int] = {}
    idx = np.empty(len(ats), dtype=np.int32)
    for i, x in enumerate(ats):
        idx[i] = recs.setdefault((x.leaf.step, x.leaf.call), len(recs))
    tags = np.concatenate([_be(SD.SEED_TAG), _be(SD.RECEIPT_TAG)])
    data = np.stack([np.concatenate([tags, _be(win.steps[s].salt), _be(win.steps[s].a_roots[w]), _be(win.b_roots[w])])
                     for s, w in recs])
    pos = np.array([x.position for x in ats], dtype=np.uint64)
    return (torch.from_numpy(data.view(np.int64)).to(device), torch.from_numpy(idx).to(device),
            torch.from_numpy(pos.view(np.int64)).to(device))


def _rows(t: torch.Tensor, blocks: list[int], device) -> torch.Tensor:
    """Row blocks `blocks` of t [rows, K], (len, 32, K), on `device`."""
    idx = (torch.tensor(blocks, device=t.device)[:, None] * RC.R + torch.arange(RC.R, device=t.device)).flatten()
    return t.index_select(0, idx).view(len(blocks), RC.R, t.shape[1]).to(device)


def _gather(win: Window, ats: list[_At], device) -> tuple[torch.Tensor, torch.Tensor]:
    """The leaves' A and B row blocks, (L, 32, K) s8 each on `device`, in `ats`' order (one index_select a step and call)."""
    k = win.k(ats[0].leaf.call)
    a = torch.empty(len(ats), RC.R, k, dtype=torch.int8, device=device)
    b = torch.empty_like(a)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, x in enumerate(ats):
        groups[(x.leaf.step, x.leaf.call)].append(i)
    for (s, w), ix in groups.items():
        at = torch.tensor(ix, device=device)
        a[at] = _rows(win.steps[s].a[w], [ats[i].ri for i in ix], device)
        b[at] = _rows(win.b[w], [ats[i].cj for i in ix], device)
    return a, b


class Kernel:
    """h1u.cu's `kh_lanes` and `kh_seeds_at` from any build of it (sbench.sh's h1u_kc.so, job.sh's h1u*.so): leaves'
    seeds, receipts and lane records on gathered operands, on torch's current stream."""

    def __init__(self, lib: str | Path | C.CDLL):
        h = lib if isinstance(lib, C.CDLL) else C.CDLL(str(lib))
        vp, u32 = C.c_void_p, C.c_uint
        h.kh_lanes.argtypes = [vp, vp, u32, u32, vp, vp, vp, vp]
        h.kh_lanes.restype = C.c_int
        h.kh_seeds_at.argtypes = [vp, vp, vp, u32, vp, vp]
        h.kh_seeds_at.restype = C.c_int
        h.kh_words.argtypes = [u32]
        h.kh_words.restype = u32
        self.h = h

    def seeds(self, data: torch.Tensor, idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """(L, 16) int64 on the device, leaf i's seed_j then σ_R as big-endian words (`receipts.seed_words`'), at
        position pos[i] from record idx[i] of data (G, 30) int64 (`_seed_inputs`), asynchronously."""
        n = idx.shape[0]
        if data.dim() != 2 or data.shape[1] != 30 or pos.shape != (n,) or idx.dtype != torch.int32:
            raise ValueError(f"kh_seeds_at: data {tuple(data.shape)}, idx {tuple(idx.shape)} {idx.dtype}, "
                             f"pos {tuple(pos.shape)}")
        data, idx, pos = data.contiguous(), idx.contiguous(), pos.contiguous()
        out = torch.empty(n, 16, dtype=torch.int64, device=idx.device)
        rc = self.h.kh_seeds_at(data.data_ptr(), idx.data_ptr(), pos.data_ptr(), n, out.data_ptr(),
                                torch.cuda.current_stream(idx.device).cuda_stream)
        if rc:
            raise RuntimeError(f"kh_seeds_at returned {rc} (L = {n})")
        return out

    def run(self, a: torch.Tensor, b: torch.Tensor, seeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """a, b (L, 32, K) s8 on the device (leaf i's A and B row blocks), seeds (L, 16) int64 (seed_j's 8 big-endian
        words, then σ_R's: `receipts.seed_words`) -> (receipts (L, W), records (L, 32, W)) int32, asynchronously."""
        n, r, k = a.shape
        if b.shape != a.shape or r != RC.R or seeds.shape != (n, 16) or a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError(f"kh_lanes: a {tuple(a.shape)}, b {tuple(b.shape)} s8 and seeds {tuple(seeds.shape)}")
        w = int(self.h.kh_words(k))
        a, b, seeds = a.contiguous(), b.contiguous(), seeds.contiguous()
        out = torch.empty(n, w, dtype=torch.int32, device=a.device)
        rec = torch.empty(n, LANES, w, dtype=torch.int32, device=a.device)
        rc = self.h.kh_lanes(a.data_ptr(), b.data_ptr(), n, k, seeds.data_ptr(), out.data_ptr(), rec.data_ptr(),
                             torch.cuda.current_stream(a.device).cuda_stream)
        if rc:
            raise RuntimeError(f"kh_lanes returned {rc} (L = {n}, K = {k})")
        return out, rec


def _cpu(win: Window, ats: list[_At], chunk: int) -> np.ndarray:
    k = win.k(ats[0].leaf.call)
    lay = RC.layout(k)
    out = np.empty((len(ats), LANES, 2 * lay.sets), dtype=np.uint32)
    cpu = torch.device("cpu")
    for at in range(0, len(ats), chunk):
        part = ats[at:at + chunk]
        sj, sr = _seeds(win, part)
        a, b = _gather(win, part, cpu)
        per = RC.tile_receipts(a.to(torch.int64), b.to(torch.int64), RC.noise_blocks_cpu(sj, lay.blocks),
                               RC.noise_blocks_cpu(sr, LANES), lanes=True)
        out[at:at + len(part)] = per.numpy().astype(np.uint32)
    return out


def _live_by_k(win: Window, ats: list[_At]) -> dict[int, list[int]]:
    by_k: dict[int, list[int]] = defaultdict(list)
    for i, x in enumerate(ats):
        if x.live:
            by_k[win.k(x.leaf.call)].append(i)
    return by_k


def gpu_batches(win: Window, leaves: list[Leaf], kernel: Kernel, chunk: int | None = None, device=None):
    """The live leaves on `kernel`, `chunk` of one K at a time (2^30 / (64 K) by default: a GiB of gathered operands), on
    `device` (the B tensors' by default), their seeds from `kh_seeds_at`: yields (indices into `leaves`, receipts (n, W),
    records (n, 32, W)), int32 on the device; a dead leaf is in no batch (its partials are zero)."""
    ats = [_at(win, x) for x in leaves]
    for k, ix in _live_by_k(win, ats).items():
        dev = device if device is not None else win.b[ats[ix[0]].leaf.call].device
        n = chunk or max(1, (1 << 30) // (64 * k))
        for at in range(0, len(ix), n):
            sel = ix[at:at + n]
            part = [ats[i] for i in sel]
            seeds = kernel.seeds(*_seed_inputs(win, part, dev))
            a, b = _gather(win, part, dev)
            out, rec = kernel.run(a, b, seeds)
            yield sel, out, rec


def lane_partials_many(win: Window, leaves: list[Leaf], backend: str = "cpu", kernel: Kernel | None = None,
                       chunk: int | None = None, device=None) -> list[np.ndarray]:
    """Each leaf's lane partials, (32, W) uint32 (the module docstring's lane and word order), in `leaves`' order: the
    live leaves of each K batched, `chunk` at a time (cpu: 2^20 / K; gpu: `gpu_batches`), on `backend` ("cpu", or "gpu":
    `kernel` on `device`, the B tensors' by default)."""
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend: cpu or gpu, not {backend!r}")
    if backend == "gpu" and kernel is None:
        raise ValueError("the gpu backend needs a Kernel (an h1u.cu build with kh_lanes)")
    ats = [_at(win, x) for x in leaves]
    res = [None if x.live else np.zeros((LANES, win.words(x.leaf.call)), dtype=np.uint32) for x in ats]
    if backend == "cpu":
        for k, ix in _live_by_k(win, ats).items():
            got = _cpu(win, [ats[i] for i in ix], chunk or max(1, (1 << 20) // k))
            for j, i in enumerate(ix):
                res[i] = got[j]
    else:
        for sel, _, rec in gpu_batches(win, leaves, kernel, chunk, device):
            got = rec.cpu().numpy().view(np.uint32)
            for j, i in enumerate(sel):
                res[i] = got[j]
    return res  # type: ignore[return-value]


def lane_partials(win: Window, leaf: Leaf, backend: str = "cpu", kernel: Kernel | None = None,
                  device=None) -> np.ndarray:
    """Leaf `leaf`'s 32 lane partials, (32, W) uint32: row l lane l = 4 g + q, word 2 s its chains' y after set s and
    2 s + 1 their z (the module docstring); their XOR over rows is the leaf's receipt."""
    return lane_partials_many(win, [leaf], backend, kernel, device=device)[0]


def combine(partials: np.ndarray) -> np.ndarray:
    """The combine's statement: the XOR over the lanes (the first axis of (32, W), or the second of (L, 32, W)), which
    must be the committed receipt."""
    return np.bitwise_xor.reduce(partials, axis=-2)


__all__ = ["Kernel", "LANES", "Leaf", "Step", "Window", "combine", "gpu_batches", "lane_partials", "lane_partials_many"]
