"""`eps1-sm120-h1fx1`'s unit (`pouw.circuit.h1tile`) as a side computation on a served int8 call, vectorized over the call's
r × c tiles in torch, bit for bit `h1tile.reference`'s receipt.

A call is A [M, K] (s8, the served `Int8RowQuant_v1` rows) and B [N, K] (s8, the int8 pin's weight), 32 | K, K >= 64.  Its
units are the full r × c tiles, in row-block-major order: unit u is rows ``r (u // (N / c))`` .. and columns
``c (u % (N / c))`` ..; a call's rows past the last full row block are no unit's (`units`).  Unit u's position is
``base + u``.

Per unit (`h1tile`'s module doc): seed_j and σ_R from (salt, position, root(A), root(B)) (`seeds.seed`,
`seeds.receipt_seed`, hashlib); seed_j's noise blocks and σ_R's key blocks (`noise_blocks`: the CUDA kernel `noise.cu`
through the driver API when a cubin is given, else hashlib); then on the device

* X = A + b and W' = B + c + U, mod 2^8 (U = `h1tile.u129` of the column's 32 noise bytes, repeated every 32 columns of K);
* T_s, the prefix sums over segments of X_s W'_sᵀ (each segment's 32-term sum as a float32 matmul: |sum| <= 32 · 128² < 2^24,
  exact), mod 2^32; the residue sums' digits A_0 .. A_(m-1) (`segment.digit_words`); O' = T_last - A_0 Uᵀ and
  D_d = A_d Uᵀ, mod 2^32;
* each lane's entries (`h1tile.Layout.rows` / `cols` at `segment.row_slot` / `col_slot`) times `segment.fx1_key`, folded
  into the lane's 16 keys with duo (`segment._duo_absorb`, its two chain halves at once), and after each set the XOR over
  lanes and chains of y, then of z: the receipt's 2 (KS + m) u32 words.

z, the plain A Bᵀ cell, is the served y's integer product and no part of this computation.
"""
from __future__ import annotations

import ctypes as C
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG

M32 = (1 << 32) - 1
R = C_ = 32
LANES = HT.Layout(R, C_, 64).lanes
HERE = Path(__file__).resolve().parent


def layout(k: int) -> HT.Layout:
    return HT.Layout(R, C_, k)


def units(m: int, n: int) -> list[tuple[int, int]]:
    """A call's units, (row block, column block), in position order: the full 32-row blocks of its M rows, by column block."""
    return [(i, j) for i in range(m // R) for j in range(n // C_)]


# -- seeds and noise -----------------------------------------------------------------------------------------------------

def unit_seeds(salt: bytes, positions: list[int], root_a: bytes, root_b: bytes) -> tuple[list[bytes], list[bytes]]:
    return ([SD.seed(salt, p, root_a, root_b) for p in positions],
            [SD.receipt_seed(salt, p, root_a, root_b) for p in positions])


def _halves(digests: bytes) -> np.ndarray:
    return np.frombuffer(digests, dtype=">u4").astype(np.int64)


def noise_blocks_cpu(seeds: list[bytes], n: int) -> torch.Tensor:
    """Blocks 0 .. n - 1 of each seed's stream as big-endian halves, (U, n, 16) int64 (`seeds.noise_block`)."""
    out = bytearray()
    for s in seeds:
        head = SD.NOISE_TAG + bytes(s)
        for t in range(n):
            out += hashlib.sha512(head + t.to_bytes(8, "big")).digest()
    return torch.from_numpy(_halves(bytes(out)).reshape(len(seeds), n, 16))


class Cubin:
    """`noise.cu` built for this device, loaded through the driver API into torch's current context; launches on torch's
    current stream."""

    def __init__(self, path: str | Path):
        torch.zeros(1, device="cuda")
        self.lib = C.CDLL("libcuda.so.1")
        self.image = Path(path).read_bytes()
        self.mod, self.fn = C.c_void_p(), C.c_void_p()
        self._ck(self.lib.cuModuleLoadData(C.byref(self.mod), C.c_char_p(self.image)), "cuModuleLoadData")
        self._ck(self.lib.cuModuleGetFunction(C.byref(self.fn), self.mod, b"h1_noise"), "cuModuleGetFunction")
        self.tag = [int.from_bytes(SD.NOISE_TAG[8 * i:8 * i + 8], "big") for i in range(3)]

    @staticmethod
    def _ck(rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what} returned CUresult {rc}")

    def launch(self, seeds: torch.Tensor, n: int, out: torch.Tensor, sink: bool = False, block: int = 128) -> None:
        """seeds (U, 8) int64 (the digest's big-endian 64-bit words) on the device -> out (U, n, 16) int32 halves, or with
        `sink` (U, n) int32, each block's XOR of its halves (the hashing without the stream's writes)."""
        u = seeds.shape[0]
        total = u * n
        args = [C.c_uint64(seeds.data_ptr()), C.c_uint64(u), C.c_uint32(n), *[C.c_uint64(t) for t in self.tag],
                C.c_uint64(out.data_ptr()), C.c_int32(int(sink))]
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._ck(self.lib.cuLaunchKernel(self.fn, C.c_uint32(-(-total // block)), 1, 1, C.c_uint32(block), 1, 1, 0, stream,
                                         params, None), "cuLaunchKernel")


def seed_words(seeds: list[bytes], device) -> torch.Tensor:
    w = np.frombuffer(b"".join(seeds), dtype=">u8").astype("<u8").view("<i8").reshape(len(seeds), 8)
    return torch.from_numpy(w.copy()).to(device)


def noise_blocks(seeds: list[bytes], n: int, device, cubin: Cubin | None) -> torch.Tensor:
    """(U, n, 16) int64 halves on `device`: the kernel's when `cubin` is given, else hashlib's."""
    if cubin is None:
        return noise_blocks_cpu(seeds, n).to(device)
    out = torch.empty(len(seeds), n, 16, dtype=torch.int32, device=device)
    cubin.launch(seed_words(seeds, device), n, out)
    return out.to(torch.int64) & M32


# -- the unit on the device ----------------------------------------------------------------------------------------------

def _consts(device) -> dict:
    lay = layout(64)
    ri = torch.tensor([[lay.rows(ln)[SG.row_slot(e)] for e in range(32)] for ln in range(LANES)], device=device)
    ci = torch.tensor([[lay.cols(ln)[SG.col_slot(e)] for e in range(32)] for ln in range(LANES)], device=device)
    fx1 = torch.tensor([SG.fx1_key(q) for q in range(32)], dtype=torch.int64, device=device)
    step = range(4)
    ka = torch.tensor([[SG.KA[2 * i]] * 4 + [SG.KA[2 * i + 1]] * 4 for i in step], dtype=torch.int64, device=device)
    kk = torch.tensor([[SG.KK[2 * i]] * 4 + [SG.KK[2 * i + 1]] * 4 for i in step], dtype=torch.int64, device=device)
    rot = torch.tensor([[SG.ROT[2 * i]] * 4 + [SG.ROT[2 * i + 1]] * 4 for i in step], dtype=torch.int64, device=device)
    return {"ri": ri, "ci": ci, "fx1": fx1, "ka": ka, "kk": kk, "rot": rot}


_CONSTS: dict = {}


def consts(device) -> dict:
    key = str(device)
    if key not in _CONSTS:
        _CONSTS[key] = _consts(device)
    return _CONSTS[key]


def mul32(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """u k mod 2^32 for u, k < 2^32 without overflowing int64."""
    return (u * (k & 0xFFFF) + (((u * (k >> 16)) & 0xFFFF) << 16)) & M32


def rotl(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return ((u << r) | (u >> (32 - r))) & M32


def wrap8(v: torch.Tensor) -> torch.Tensor:
    """An integer as the s8 its low byte encodes."""
    return ((v + 128) & 0xFF) - 128


def _bits(h: torch.Tensor) -> torch.Tensor:
    """Halves (..., n) -> their bits (..., 32 n), least significant first."""
    sh = torch.arange(32, device=h.device)
    return ((h[..., None] >> sh) & 1).flatten(-2)


def _xor_reduce(x: torch.Tensor) -> torch.Tensor:
    """XOR over the last dimension (a power of two)."""
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] ^ x[..., h:]
    return x[..., 0]


def tile_receipts(a: torch.Tensor, b: torch.Tensor, noise: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Receipts (U, 2 sets) int64 of U units: a (U, 32, K) and b (U, 32, K) int64 s8 values, noise (U, blocks, 16) seed_j's
    halves, keys (U, LANES, 16) σ_R's."""
    nu, r, k = a.shape
    lay, dev = layout(k), a.device
    ks, m = lay.ks, lay.m
    cs = consts(dev)
    rows = noise[:, :r * lay.rb].reshape(nu, r, lay.rb * 16)
    cols = noise[:, r * lay.rb:].reshape(nu, C_, lay.cb * 16)
    x = wrap8(a + _bits(rows[..., :ks]))
    u = cols[..., ks:ks + 8]
    u = (u[..., None] >> (8 * torch.arange(4, device=dev))) & 0xFF
    u = (u.flatten(-2) % 129) - 64
    w = wrap8(b + _bits(cols[..., :ks]) + u.repeat(1, 1, ks))
    xs = x.view(nu, r, ks, 32).transpose(1, 2).to(torch.float32)
    ws = w.view(nu, C_, ks, 32).transpose(1, 2).to(torch.float32)
    t = (torch.matmul(xs, ws.transpose(-1, -2)).round().to(torch.int64).cumsum(1)) & M32
    y = x.view(nu, r, ks, 32).sum(2)
    z = _bits(rows[..., ks:ks + m - 1]).view(nu, r, m - 1, 32)
    digits = []
    for d in range(m - 1):
        q = y >> 7
        zd = z[:, :, d]
        if d == m - 2:
            zd = torch.where(q == 127, torch.zeros_like(zd), zd)
        digits.append(wrap8((y & 127) | (zd << 7)))
        y = q + zd
    digits.append(wrap8(y & 0xFF))
    uf = u.to(torch.float32).transpose(-1, -2)
    op = (t[:, -1] - torch.matmul(digits[0].to(torch.float32), uf).round().to(torch.int64)) & M32
    ds = [torch.matmul(dg.to(torch.float32), uf).round().to(torch.int64) & M32 for dg in digits[1:]]
    sets = torch.cat([t, op[:, None], *[d_[:, None] for d_ in ds]], 1)
    p = mul32(sets[:, :, cs["ri"], cs["ci"]], cs["fx1"])
    st = keys.clone()
    out = torch.empty(nu, lay.sets, 2, dtype=torch.int64, device=dev)
    for s in range(lay.sets):
        yv, zv = st[..., :8], st[..., 8:]
        ps = p[:, s]
        for i in range(4):
            uu = yv ^ ps[..., 8 * i:8 * i + 8] ^ cs["kk"][i]
            yv = (mul32(uu, cs["ka"][i]) + zv) & M32
            zv = (rotl(uu, cs["rot"][i]) + yv) & M32
        st = torch.cat([yv, zv], -1)
        out[:, s] = _xor_reduce(st.view(nu, LANES, 2, 8).transpose(1, 2).reshape(nu, 2, LANES * 8))
    return out.reshape(nu, 2 * lay.sets)


@dataclass
class Call:
    """One served call's operands on the device: A [M, K] and B [N, K] s8."""

    a: torch.Tensor
    b: torch.Tensor

    @property
    def units(self) -> list[tuple[int, int]]:
        return units(self.a.shape[0], self.b.shape[0])


def call_receipts(call: Call, salt: bytes, base: int, root_a: bytes, root_b: bytes, cubin: Cubin | None = None,
                  chunk: int | None = None, only: list[int] | None = None) -> np.ndarray:
    """The call's receipts (units, 2 sets) uint32, unit u at position base + u (`only`: those units, in that order)."""
    k = call.a.shape[1]
    lay = layout(k)
    chunk = chunk or max(16, (1 << 20) // k)
    us = call.units
    pick = list(range(len(us))) if only is None else list(only)
    dev = call.a.device
    ar = torch.arange(R, device=dev)
    out = np.empty((len(pick), 2 * lay.sets), dtype=np.uint32)
    for at in range(0, len(pick), chunk):
        part = pick[at:at + chunk]
        seeds, sigmas = unit_seeds(salt, [base + u for u in part], root_a, root_b)
        noise = noise_blocks(seeds, lay.blocks, dev, cubin)
        keys = noise_blocks(sigmas, LANES, dev, cubin)
        ri = torch.tensor([us[u][0] for u in part], device=dev)
        cj = torch.tensor([us[u][1] for u in part], device=dev)
        a = call.a[ri[:, None] * R + ar].to(torch.int64)
        b = call.b[cj[:, None] * C_ + ar].to(torch.int64)
        out[at:at + len(part)] = tile_receipts(a, b, noise, keys).cpu().numpy().astype(np.uint32)
    return out


def reference(call: Call, salt: bytes, base: int, root_a: bytes, root_b: bytes, u: int) -> tuple[list[int], list[int]]:
    """`h1tile.reference` on unit u of the call: (z, receipt)."""
    i, j = call.units[u]
    a = (call.a[R * i:R * i + R].to(torch.int64).cpu() & 0xFF).tolist()
    b = (call.b[C_ * j:C_ * j + C_].to(torch.int64).cpu() & 0xFF).tolist()
    return HT.reference(a, b, salt, base + u, (root_a, root_b))


__all__ = ["Call", "Cubin", "LANES", "call_receipts", "layout", "noise_blocks", "noise_blocks_cpu", "reference",
           "tile_receipts", "unit_seeds", "units"]
