"""Each served call's receipts, `Served.receipts` (BRIDGE2): one a proof block of the call in the window Program's tile order,
its receipt, the row its call's ``receipt@<c>`` registers. A v1 window's is `h1tile.reference`'s receipt of the proof
block's A and B rows, which `h1zy.reference` (without J10-R) returns too: the receipt a lanes-form K's combine statement
holds its 32 lane partials to (`h1zy.combine_reference`). serve-openai's ``--h1-receipts`` (`serve_h1`) sets
`Receipts.serving` on each window's `Served`. Receipts are values: they are committed under hiding leaves in the window's
record, and nothing here keeps or prints them.

**The torch chain.** `eps1-sm120-h1fx1`'s unit (`pouw.circuit.h1tile`) as a side computation on a served int8 call,
vectorized over the call's r × c tiles in torch, bit for bit `h1tile.reference`'s receipt.

A call is A [M, K] (s8, the served `Int8RowQuant_v1` rows) and B [N, K] (s8, the int8 pin's weight), 32 | K, K >= 64.  Its
units are the full r × c tiles, in row-block-major order: unit u is rows ``r (u // (N / c))`` .. and columns
``c (u % (N / c))`` ..; a call's rows past the last full row block are no unit's (`units`).  Unit u's position is
``base + u``.

Per unit (`h1tile`'s module doc): seed_j and σ_R from (salt, position, root(A), root(B)) (`seeds.seed`,
`seeds.receipt_seed`, hashlib); seed_j's noise blocks and σ_R's key blocks (`noise_blocks`: the CUDA kernel `noise.cu`
of `pouw_cuda`, built for the device (`noise_cubin`), through the driver API when a cubin is given, else hashlib); then on
the device

* X = A + b and W' = B + c + U, mod 2^8 (U = `h1tile.u129` of the column's 32 noise bytes, repeated every 32 columns of K);
* T_s, the prefix sums over segments of X_s W'_sᵀ (each segment's 32-term sum as a float32 matmul: |sum| <= 32 · 128² < 2^24,
  exact), mod 2^32; the residue sums' digits A_0 .. A_(m-1) (`segment.digit_words`); O' = T_last - A_0 Uᵀ and
  D_d = A_d Uᵀ, mod 2^32;
* each lane's entries (`h1tile.Layout.rows` / `cols` at `segment.row_slot` / `col_slot`) times `segment.fx1_key`, folded
  into the lane's 16 keys with duo (`segment._duo_absorb`, its two chain halves at once), and after each set the XOR over
  lanes and chains of y, then of z: the receipt's 2 (KS + m) u32 words.

z, the plain A Bᵀ cell, is the served y's integer product and no part of this computation.

Under J10-R (``j10r``: `h1tile.PouwH1Tile_v5`, `h1tile.j10r_reference`) the call has one noise seed, seed_c =
`seeds.call_seed` (salt, base, root(A), root(B)), ``base`` the call's first unit position (J10-R's ``call``), and one stream,
blocks 0 .. M RB + N CB - 1 of seed_c's at call scale (`h1tile.Layout(M, N, K)`: call row i at i RB, call column j at
M RB + j CB), hashed once a call (`call_streams`: `noise.cu`'s `h1_noise_calls`, one launch for many calls, else
hashlib). Unit (I, J) gathers call rows 32 I .. and columns 32 J .. from it (`j10r_counters`, `h1tile.j10r_unit_blocks`
at `h1tile.j10r_blocks`), in seed_j's noise layout, so the unit's chain above is v1's; its fold keys stay σ_R's at its own
position. M is the call's A slot, whole row blocks.

**The producer.** `Receipts` is `Served.receipts` for one window: ``(call, codes, s_a, weight codes, s_w)`` -> the call's
receipts (units, 2 (KS + m)) uint32, the input of `pouw_h1_served.receipt_words`, which `Served` refuses at any other shape.
Call c of the window at contraction K, as the Program's proof blocks read it (`h1window.program`):

* A is the call's M slot rows (`Served.m`, the Program's slot: its served codes, then zero rows) and B its linear's N
  weight rows; proof block u of the call is row block u // (N / 32) and column block u % (N / 32), at position u0 + u, u0
  the call's first proof block of the window Program (`h1window.call_units`), the position its public row carries;
* its seeds take the window Program's salt (`H1Run.salt`) and the call's roots (`Receipts.roots`): root(A) its step's A
  table ``a/<K>@<s>``'s and root(B) ``b/<K>``'s, as `Served` registered them. `Served` asks for a step's receipts of K at
  its last call of K, once both are registered;
* the proof block is v1's (`h1tile.PouwH1Tile`): no admit gating, so a row block past the served rows has the receipt its
  zero codes and its noise give it. With ``skip_dead`` (a v4 window's) such a row block's proof blocks are skipped and
  get v4's dead-tile receipt, zero words; under J10-R (`h1window.is_j10r`) they are J10-R's receipts, from the call's one
  stream.

It takes no coin: the noise is the salt's and the proof block's position's. With ``units`` (h1u.cu's units kernel,
kernels' `Units`, whose -hbc build needs 512 | K) `prepare` computes a step's calls of one K in a few launches; else the
torch chain, any 32 | K, K >= 64.
"""
from __future__ import annotations

import ctypes as C
import functools
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.prover.circuit import h1tile as HT
from verity.protocols.pouw.prover.circuit import h1window as HW
from verity.protocols.pouw.prover.circuit import seeds as SD
from verity.protocols.pouw.prover.circuit import segment as SG

M32 = (1 << 32) - 1
R = C_ = 32
LANES = HT.Layout(R, C_, 64).lanes
#: the noise kernel, of `pouw_cuda`
NOISE_CU = "noise.cu"


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


def noise_cubin(arch: str) -> Path:
    """`NOISE_CU` built with the nvcc on the path for ``sm_<arch>`` (``"90"``: an H200, ``"120"``: an RTX PRO 6000) into a
    cubin, with `verity.definitions.kernels`' ``sha512.cuh``, in the integration's build cache (`row_tables_device.built`)."""
    from verity.definitions import kernels

    from verity_vllm.commit import row_tables_device as RTD

    tool = RTD.nvcc()
    if tool is None:
        raise RuntimeError(f"no nvcc on the path: the receipts' noise kernel ({NOISE_CU}) is built with it")
    inc = Path(kernels.__file__).resolve().parent
    return RTD.built(tool, (NOISE_CU,), ("-O3", "-std=c++17", f"-I{inc}", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                         "-cubin"), "noise.cubin", headers=(inc / "sha512.cuh",))


class Cubin:
    """`noise.cu` built for this device, loaded through the driver API into torch's current context; launches on torch's
    current stream."""

    @classmethod
    def built(cls, device: Any = None) -> "Cubin":
        """`noise_cubin` for ``device``'s compute capability (torch's current device by default), loaded."""
        major, minor = torch.cuda.get_device_capability(device)
        return cls(noise_cubin(f"{major}{minor}"))

    def __init__(self, path: str | Path) -> None:
        torch.zeros(1, device="cuda")
        self.lib = C.CDLL("libcuda.so.1")
        self.image = Path(path).read_bytes()
        self.mod, self.fn = C.c_void_p(), C.c_void_p()
        self._ck(self.lib.cuModuleLoadData(C.byref(self.mod), C.c_char_p(self.image)), "cuModuleLoadData")
        self._ck(self.lib.cuModuleGetFunction(C.byref(self.fn), self.mod, b"h1_noise"), "cuModuleGetFunction")
        self.fn_calls = C.c_void_p()
        self._ck(self.lib.cuModuleGetFunction(C.byref(self.fn_calls), self.mod, b"h1_noise_calls"), "cuModuleGetFunction")
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
        self._go(self.fn, total, block, args)

    def launch_calls(self, seeds: torch.Tensor, starts: torch.Tensor, out: torch.Tensor, sink: bool = False,
                     block: int = 128) -> None:
        """J10-R's call streams, `h1_noise_calls`: seeds (calls, 8) int64 (each call's seed_c) and starts (calls + 1,)
        int64 (call c's stream is the launch's blocks starts[c] .. starts[c + 1] - 1) on the device -> out (starts[-1], 16)
        int32 halves, or with `sink` (starts[-1],) int32, each block's XOR of its halves."""
        total = int(starts[-1])
        args = [C.c_uint64(seeds.data_ptr()), C.c_uint64(starts.data_ptr()), C.c_uint32(seeds.shape[0]),
                *[C.c_uint64(t) for t in self.tag], C.c_uint64(out.data_ptr()), C.c_int32(int(sink))]
        self._go(self.fn_calls, total, block, args)

    def _go(self, fn: C.c_void_p, total: int, block: int, args: list) -> None:
        if not total:
            return
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._ck(self.lib.cuLaunchKernel(fn, C.c_uint32(-(-total // block)), 1, 1, C.c_uint32(block), 1, 1, 0, stream,
                                         params, None), "cuLaunchKernel")


@functools.cache
def loaded(device: torch.device) -> Cubin:
    """`Cubin.built` for ``device``, once a process: every window's producer launches from the one loaded module."""
    return Cubin.built(device)


def seed_words(seeds: list[bytes], device) -> torch.Tensor:
    w = np.frombuffer(b"".join(seeds), dtype=">u8").astype("<u8").view("<i8").reshape(len(seeds), 8)
    return torch.from_numpy(w.copy()).to(device)


def noise_blocks(seeds: list[bytes], n: int, device: torch.device | str, cubin: Cubin | None) -> torch.Tensor:
    """(U, n, 16) int64 halves on `device`: the kernel's when `cubin` is given, else hashlib's."""
    if cubin is None:
        return noise_blocks_cpu(seeds, n).to(device)
    out = torch.empty(len(seeds), n, 16, dtype=torch.int32, device=device)
    cubin.launch(seed_words(seeds, device), n, out)
    return out.to(torch.int64) & M32


# -- J10-R's call streams ------------------------------------------------------------------------------------------------

def stream_blocks(m: int, n: int, k: int) -> int:
    """A call's J10-R stream length in blocks: its M rows' and N columns' at call scale, `h1tile.Layout(M, N, K).blocks`."""
    return HT.Layout(m, n, k).blocks


def call_streams(seeds: list[bytes], counts: list[int], device: torch.device | str,
                 cubin: Cubin | None) -> list[torch.Tensor]:
    """Each call's stream, blocks 0 .. counts[c] - 1 of seeds[c] (its seed_c), as (counts[c], 16) int64 halves on `device`:
    the kernel's (`h1_noise_calls`, one launch for every call) when `cubin` is given, else hashlib's."""
    if len(seeds) != len(counts):
        raise ValueError(f"{len(seeds)} seeds for {len(counts)} calls")
    if cubin is None:
        out = []
        for s, nb in zip(seeds, counts):
            head = SD.NOISE_TAG + bytes(s)
            raw = b"".join(hashlib.sha512(head + t.to_bytes(8, "big")).digest() for t in range(nb))
            out.append(torch.from_numpy(_halves(raw).reshape(nb, 16)).to(device))
        return out
    starts = [0]
    for nb in counts:
        starts.append(starts[-1] + int(nb))
    got = torch.empty(starts[-1], 16, dtype=torch.int32, device=device)
    cubin.launch_calls(seed_words(seeds, device), torch.tensor(starts, dtype=torch.int64, device=device), got)
    got = got.to(torch.int64) & M32
    return [got[lo:hi] for lo, hi in zip(starts[:-1], starts[1:])]


def j10r_counters(ri: torch.Tensor, cj: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """The counters of seed_c's blocks units (ri, cj) read, (U, blocks) int64, rows' then columns', each row's or column's
    blocks in turn: `h1tile.j10r_unit_blocks` at `h1tile.j10r_blocks`(M = m, K = k, the 32 × 32 tile, ri, cj), seed_j's
    noise layout (`tile_receipts`' ``noise``)."""
    lay = layout(k)
    rb, cb, ar = lay.rb, lay.cb, torch.arange(R, device=ri.device)
    rows = ((R * ri[:, None] + ar)[:, :, None] * rb + torch.arange(rb, device=ri.device)).flatten(1)
    cols = (m * rb + (C_ * cj[:, None] + ar)[:, :, None] * cb + torch.arange(cb, device=ri.device)).flatten(1)
    return torch.cat([rows, cols], 1)


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


def tile_receipts(a: torch.Tensor, b: torch.Tensor, noise: torch.Tensor, keys: torch.Tensor,
                  lanes: bool = False) -> torch.Tensor:
    """Receipts (U, 2 sets) int64 of U units: a (U, 32, K) and b (U, 32, K) int64 s8 values, noise (U, blocks, 16) seed_j's
    halves, keys (U, LANES, 16) σ_R's.  With `lanes`, each lane's words before the XOR across lanes, (U, LANES, 2 sets):
    word 2 s the XOR of the lane's 8 chains' y after set s, 2 s + 1 of their z."""
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
    per = torch.empty(nu, LANES, lay.sets, 2, dtype=torch.int64, device=dev) if lanes else None
    for s in range(lay.sets):
        yv, zv = st[..., :8], st[..., 8:]
        ps = p[:, s]
        for i in range(4):
            uu = yv ^ ps[..., 8 * i:8 * i + 8] ^ cs["kk"][i]
            yv = (mul32(uu, cs["ka"][i]) + zv) & M32
            zv = (rotl(uu, cs["rot"][i]) + yv) & M32
        st = torch.cat([yv, zv], -1)
        if lanes:
            per[:, :, s] = _xor_reduce(st.view(nu, LANES, 2, 8))
        else:
            out[:, s] = _xor_reduce(st.view(nu, LANES, 2, 8).transpose(1, 2).reshape(nu, 2, LANES * 8))
    return per.reshape(nu, LANES, 2 * lay.sets) if lanes else out.reshape(nu, 2 * lay.sets)


@dataclass
class Call:
    """One served call's operands on the device: A [M, K] and B [N, K] s8."""

    a: torch.Tensor
    b: torch.Tensor

    @property
    def units(self) -> list[tuple[int, int]]:
        return units(self.a.shape[0], self.b.shape[0])


def call_stream(call: Call, salt: bytes, base: int, root_a: bytes, root_b: bytes, cubin: Cubin | None = None) -> torch.Tensor:
    """The call's J10-R stream on its device, (M RB + N CB, 16) int64 halves: seed_c's (`seeds.call_seed` at the call's first
    position ``base``) at call scale. ValueError unless M is whole row blocks (the window Program's slot is)."""
    (m, k), n = call.a.shape, call.b.shape[0]
    if m % R or not m:
        raise ValueError(f"a J10-R call's A is its slot, whole blocks of {R} rows, not {m}")
    return call_streams([SD.call_seed(salt, base, root_a, root_b)], [stream_blocks(m, n, k)], call.a.device, cubin)[0]


def call_receipts(call: Call, salt: bytes, base: int, root_a: bytes, root_b: bytes, cubin: Cubin | None = None,
                  chunk: int | None = None, only: list[int] | None = None, lanes: bool = False, j10r: bool = False,
                  stream: torch.Tensor | None = None) -> np.ndarray:
    """The call's receipts (units, 2 sets) uint32, unit u at position base + u (`only`: those units, in that order); with
    `lanes` each lane's words (units, LANES, 2 sets), whose XOR over lanes is the receipt (`tile_receipts`). `j10r`: J10-R's
    unit (`h1tile.PouwH1Tile_v5`), each unit's noise gathered from the call's stream (``stream``, else `call_stream`)."""
    k = call.a.shape[1]
    lay = layout(k)
    chunk = chunk or max(16, (1 << 20) // k)
    us = call.units
    pick = list(range(len(us))) if only is None else list(only)
    dev = call.a.device
    ar = torch.arange(R, device=dev)
    if j10r and stream is None:
        stream = call_stream(call, salt, base, root_a, root_b, cubin)
    out = np.empty((len(pick),) + ((LANES,) if lanes else ()) + (2 * lay.sets,), dtype=np.uint32)
    for at in range(0, len(pick), chunk):
        part = pick[at:at + chunk]
        ri = torch.tensor([us[u][0] for u in part], device=dev)
        cj = torch.tensor([us[u][1] for u in part], device=dev)
        if j10r:
            sigmas = [SD.receipt_seed(salt, base + u, root_a, root_b) for u in part]
            noise = stream[j10r_counters(ri, cj, call.a.shape[0], k)]
        else:
            seeds, sigmas = unit_seeds(salt, [base + u for u in part], root_a, root_b)
            noise = noise_blocks(seeds, lay.blocks, dev, cubin)
        keys = noise_blocks(sigmas, LANES, dev, cubin)
        a = call.a[ri[:, None] * R + ar].to(torch.int64)
        b = call.b[cj[:, None] * C_ + ar].to(torch.int64)
        out[at:at + len(part)] = tile_receipts(a, b, noise, keys, lanes).cpu().numpy().astype(np.uint32)
    return out


def reference(call: Call, salt: bytes, base: int, root_a: bytes, root_b: bytes, u: int,
              gated: np.ndarray | None = None, j10r: bool = False) -> tuple[list[int], list[int]]:
    """`h1tile.reference` on unit u of the call: (z, receipt); `gated`, the tile's 32 admit bits, is `PouwH1Tile_v4`'s
    input (a dead tile, every bit 0, has the zero receipt), None the ungated unit (a dead tile's receipt as its noise gives
    it). `j10r`: `h1tile.j10r_reference`, ``call`` = base and ``blocks`` its tile's `h1tile.j10r_blocks` at the call's M."""
    i, j = call.units[u]
    a = (call.a[R * i:R * i + R].to(torch.int64).cpu() & 0xFF).tolist()
    b = (call.b[C_ * j:C_ * j + C_].to(torch.int64).cpu() & 0xFF).tolist()
    if j10r:
        if gated is not None:
            raise ValueError("J10-R's unit (PouwH1Tile_v5) takes no admit bits")
        blocks = HT.j10r_blocks(call.a.shape[0], call.a.shape[1], (R, C_), i, j)
        return HT.j10r_reference(a, b, salt, base + u, (root_a, root_b), base, blocks)
    bits = None if gated is None else [int(x) for x in np.asarray(gated, dtype=np.uint8)]
    return HT.reference(a, b, salt, base + u, (root_a, root_b), gated=bits)


def gated_receipts(call: Call, admit: np.ndarray, salt: bytes, base: int, root_a: bytes, root_b: bytes,
                   cubin: Cubin | None = None, chunk: int | None = None) -> tuple[np.ndarray, list[int]]:
    """The call's receipts under its admit bits, `PouwH1Tile_v4`'s (a tile's live bit is the OR of its 32 rows' admit
    bits, the bits the tally reads): a live unit's is `h1tile`'s, whatever its rows' bytes, a dead unit's the canonical
    empty receipt (zero words) with no noise derived; and the live units."""
    live = [ri * (call.b.shape[0] // C_) + cj for ri in range(call.a.shape[0] // R)
            if bool(np.asarray(admit[R * ri:R * ri + R]).any()) for cj in range(call.b.shape[0] // C_)]
    out = np.zeros((len(call.units), 2 * layout(call.a.shape[1]).sets), dtype=np.uint32)
    if live:
        out[live] = call_receipts(call, salt, base, root_a, root_b, cubin, chunk=chunk, only=live)
    return out, live


def gated_reference(call: Call, admit: np.ndarray, salt: bytes, base: int, root_a: bytes, root_b: bytes,
                    u: int) -> list[int]:
    """Unit u's receipt under the call's admit bits: `PouwH1Tile_v4`'s on its tile's 32 bits (`reference(gated=...)`)."""
    i, _ = call.units[u]
    return reference(call, salt, base, root_a, root_b, u, gated=np.asarray(admit)[R * i:R * i + R])[1]


# -- the producer ----------------------------------------------------------------------------------------------------------

def _be(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


def seed_record(salt: bytes, root_a: bytes, root_b: bytes, j10r: bool = False) -> np.ndarray:
    """h1u.cu's `kh_seeds`' data, 30 int64: SEED_TAG and RECEIPT_TAG (3 big-endian u64 words each), then the salt, root(A)
    and root(B) (8 each); with `j10r`, `kh_seeds_j10r`'s, CALL_SEED_TAG in SEED_TAG's place. ValueError unless each of the
    three is 64 bytes."""
    for name, v in (("salt", salt), ("root(A)", root_a), ("root(B)", root_b)):
        if len(v) != 64:
            raise ValueError(f"a unit's seeds take a 64-byte {name}, not {len(v)} bytes")
    tag = SD.CALL_SEED_TAG if j10r else SD.SEED_TAG
    return np.concatenate([_be(tag), _be(SD.RECEIPT_TAG), _be(salt), _be(root_a), _be(root_b)]).view(np.int64)


class Receipts:
    """`Served.receipts` for the window ``served`` (`pouw_h1_served.Served`) of the run ``run`` (`pouw_h1.H1Run`, its
    salt; `bind` when the run is made after the producer): ``(call, codes, s_a, weight codes, s_w)`` -> the call's
    receipts (units, 2 (KS + m)) uint32 (the module docstring). With ``units`` (kernels' h1u.cu `Units`) on ``device``
    (CUDA), else the torch chain on ``device`` (the CPU by default).

    ``weight(call)``: the call's B [N, K] s8 on ``device`` (the serving engine's own int8 weight, so the GPU holds no second
    copy); by default the weight codes `Served` hands over, copied to ``device`` once a linear. ``roots(call)``: the call's
    (root(A), root(B)), by default `Receipts.roots`. ``keep(call)``: keep that call's operands, roots and receipts in
    ``kept`` (by call index), for a check against `h1tile.reference`. ``skip_dead``: skip the proof blocks of row blocks
    wholly past the served rows (v4's dead tile; refused under J10-R). ``launches``: `prepare` splits a batch into that many
    launches of whole calls, about equal in proof blocks, each one's codes staged and copied to the device while the one
    before computes, and its receipts copied back while the one after does. ``cubin``: the torch chain's noise from
    `noise.cu` (`Cubin`) on ``device``, else hashlib's. ``j10r``: J10-R's receipts, by default the window Program's form
    (`h1window.is_j10r`). ``calls`` and ``seconds``: the calls it computed and the time they took."""

    def __init__(self, served: Any, run: Any = None, units: Any = None, device: Any = None,
                 weight: Callable[[Any], torch.Tensor] | None = None,
                 roots: Callable[[Any], tuple[bytes, bytes]] | None = None, keep: Callable[[Any], bool] | None = None,
                 skip_dead: bool = False, launches: int = 4, cubin: Any = None, j10r: bool | None = None) -> None:
        if launches < 1:
            raise ValueError(f"a batch takes at least one launch, not {launches}")
        self.served, self.run, self.units, self.launches, self.cubin = served, run, units, launches, cubin
        self._j10r = j10r
        self.skip_dead = bool(skip_dead)
        d = torch.device(device if device is not None else ("cuda" if units is not None or cubin is not None else "cpu"))
        self.device = torch.device("cuda", torch.cuda.current_device()) if d.type == "cuda" and d.index is None else d
        self._weight, self._roots, self._keep = weight, roots, keep
        self._b: dict[int, tuple[Any, torch.Tensor]] = {}
        self._starts: tuple[Any, list[tuple[int, int]]] | None = None
        self._records: dict[tuple[bytes, bytes, bytes, bool], torch.Tensor] = {}
        self._ready: dict[int, tuple[Any, np.ndarray, Any]] = {}
        self._copy: Any = None
        self._up: Any = None
        self.kept: dict[int, dict[str, Any]] = {}
        self.calls, self.seconds = 0, 0.0

    def bind(self, run: Any) -> "Receipts":
        self.run = run
        return self

    @classmethod
    def serving(cls, served: Any, run: Any, layers: list[Any]) -> "Receipts":
        """serve-openai's producer (``--h1-receipts``) for the window ``served`` of ``run``: on the device the served
        ``layers``' weights are on, each call's B its linear's own weight (no second copy) when every layer holds one, and
        on a CUDA device the noise from `noise.cu` (`loaded`), else hashlib's."""
        held = [m.weight for m in layers if isinstance(getattr(m, "weight", None), torch.Tensor)]
        device = held[0].device if held else torch.device("cpu")
        weight = (lambda call: layers[call.layer].weight) if len(held) == len(layers) else None
        return cls(served, run, device=device, weight=weight, cubin=loaded(device) if device.type == "cuda" else None)

    def roots(self, call: Any) -> tuple[bytes, bytes]:
        """Call ``call``'s (root(A), root(B)): its step's A table ``a/<K>@<s>``'s and the weights' ``b/<K>``'s registered
        roots (`Served.registered`), the Registered row/v2 roots its tables were committed under. ValueError while
        either isn't registered."""
        if self._roots is not None:
            return self._roots(call)
        s = self.served
        names = (s.a_table(call.index), s.program.fn.bindings["PREFIX"] + f"b/{call.k}")
        if missing := [v for v in names if v not in s.registered]:
            raise ValueError(f"call {call.index}'s seeds take the roots of {names[0]} and {names[1]}, and {missing} isn't "
                             "registered yet: a step's receipts of K are asked for at its last call of K")
        return s.registered[names[0]].root, s.registered[names[1]].root

    @property
    def j10r(self) -> bool:
        """Whether the producer computes J10-R's receipts: ``j10r`` if given, else the window Program's form."""
        return HW.is_j10r(self.served.program) if self._j10r is None else bool(self._j10r)

    def public(self, call: Any) -> list[list[int]]:
        """Call ``call``'s public rows as the verifier builds them (`credit.public_table`, the statement's ``pub`` port):
        row t proof block u0 + t's, the window Program's salt, u0 + t and the call's roots (`roots`), and under J10-R u0 and
        the proof block's blocks (`h1window.call_blocks`)."""
        s, (u0, n), (ra, rb) = self.served, self.start(call), self.roots(call)
        blocks = HW.call_blocks(s.program, call.index) if self.j10r else None
        rows, _salts, _reg = CR.public_table(s.program_digest, HW.table(s.program, HW.PUBS, call.index), self.run.salt.hex(),
                                             s.root, u0, n, (ra.hex(), rb.hex()), blocks)
        return rows

    def start(self, call: Any) -> tuple[int, int]:
        """Call ``call``'s first proof block of the window Program and its proof blocks (`h1window.call_units`): their
        positions are u0 .. u0 + n - 1."""
        prog = self.served.program
        if self._starts is None or self._starts[0] is not prog:
            self._starts = (prog, HW.call_units(prog))
        return self._starts[1][call.index]

    def weight_of(self, call: Any, w: Any) -> torch.Tensor:
        """Call ``call``'s B [N, K] s8 on the device (``weight``, else ``w`` copied there once a linear)."""
        if self._weight is not None:
            b = self._weight(call)
        else:
            held = self._b.get(call.layer)
            if held is None or held[0] is not w:
                if w is None:
                    raise ValueError(f"call {call.index}: no weight of linear {call.layer} was handed over")
                held = self._b[call.layer] = (w, torch.from_numpy(np.ascontiguousarray(w, np.int8)).to(self.device))
            b = held[1]
        if b.dtype != torch.int8 or tuple(b.shape) != (call.n, call.k) or b.device != self.device or not b.is_contiguous():
            raise ValueError(f"call {call.index}: its weight is {b.dtype} {tuple(b.shape)} on {b.device}, not its {call.n} × "
                             f"{call.k} contiguous int8 codes on {self.device}")
        return b

    def operands(self, call: Any, codes: Any, w: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Call ``call``'s A [M, K] (its served ``codes`` [m, K], then zero rows to the Program's slot M) and B [N, K], s8
        on the device."""
        m, k = int(self.served.m), int(call.k)
        x = np.asarray(codes)
        if x.dtype != np.int8 or x.ndim != 2 or x.shape[1] != k or len(x) > m:
            raise ValueError(f"call {call.index}: its codes are {x.dtype} {x.shape}, not at most {m} rows of {k} int8 codes")
        a = torch.zeros(m, k, dtype=torch.int8, device=self.device)
        if len(x):
            a[:len(x)] = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        return a, self.weight_of(call, w)

    def _record(self, salt: bytes, ra: bytes, rb: bytes) -> torch.Tensor:
        key = (salt, ra, rb, self.j10r)
        if key not in self._records:
            self._records = {key: torch.from_numpy(seed_record(salt, ra, rb, self.j10r)).to(self.device)}
        return self._records[key]

    def _live(self, m: int) -> torch.Tensor | None:
        if not self.skip_dead:
            return None
        if self.j10r:
            raise ValueError("skip_dead under J10-R: its unit (PouwH1Tile_v5) is ungated, a dead tile has its noise's "
                             "receipt")
        return (torch.arange(int(self.served.m) // R, device=self.device) * R < m).to(torch.uint8)

    def prepare(self, batch: list[tuple[Any, Any, Any, Any, Any]]) -> None:
        """`Served`'s hook before it asks a step's calls of one K, ``batch`` their ``(call, codes, s_a, weight codes, s_w)``
        in the order it asks them: with the units kernel, every proof block of them in `launches` launches
        (`Units.multi`) and their receipts copied to the host on a stream of their own, so each call's ask returns its rows
        (the same words) with no launch of its own, once its launch's copy is done. A launch's codes are staged in pinned
        memory and go to the device in one copy on a stream of their own. Nothing with the torch chain, or unless every
        call of the batch serves the same rows at one K: each is then asked on its own."""
        self._ready = {}
        if self.units is None or self.run is None or len(batch) < 2:
            return
        t0 = time.perf_counter()
        m, k, slot = len(batch[0][1]), int(batch[0][0].k), int(self.served.m)
        if any(len(codes) != m or int(call.k) != k for call, codes, *_rest in batch):
            return
        xs = [np.asarray(codes) for _call, codes, *_rest in batch]
        if any(x.dtype != np.int8 or x.shape != (m, k) for x in xs) or m > slot:
            raise ValueError(f"calls {[c.index for c, *_r in batch]}: their codes are not {m} rows of {k} int8 codes each, "
                             f"at most {slot}")
        bs = [self.weight_of(call, w) for call, _codes, _s_a, w, _s_w in batch]
        spans = [self.start(call) for call, *_rest in batch]
        for (call, *_rest), b, (_u0, n) in zip(batch, bs, spans):
            if (slot // R) * (b.shape[0] // C_) != n:
                raise ValueError(f"call {call.index}: {n} units in the window Program, not its {slot} × {b.shape[0]} tiles")
        offs = [int(x) for x in np.cumsum([0] + [n for _u0, n in spans])]
        n = min(self.launches, len(batch))
        cuts = sorted({0, len(batch)} | {int(np.searchsorted(offs, offs[-1] * j / n)) for j in range(1, n)})
        recs = [self._record(self.run.salt, *self.roots(call)) for call, *_rest in batch]
        main = torch.cuda.current_stream(self.device)
        if self._copy is None:
            self._copy, self._up = torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)
        a = (torch.empty if m == slot else torch.zeros)(len(batch), slot, k, dtype=torch.int8, device=self.device)
        self._up.wait_stream(main)
        a.record_stream(self._up)
        host = torch.empty(len(batch), m, k, dtype=torch.int8, pin_memory=True) if m else None
        seeds = torch.empty(offs[-1], 16, dtype=torch.int64, device=self.device)
        got = torch.empty(offs[-1], self.units.words(k), dtype=torch.int32, pin_memory=True)
        words, live, ready, j10r = got.numpy().view(np.uint32), self._live(m), {}, self.j10r
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if host is not None:
                hn = host.numpy()
                for i in range(lo, hi):
                    hn[i] = xs[i]
                with torch.cuda.stream(self._up):
                    a[lo:hi, :m].copy_(host[lo:hi], non_blocking=True)
                main.wait_stream(self._up)
            i = lo
            while i < hi:
                j = i + 1
                while not j10r and j < hi and recs[j] is recs[i] and spans[j][0] == spans[j - 1][0] + spans[j - 1][1]:
                    j += 1
                self.units.seeds(recs[i], spans[i][0], offs[j] - offs[i], out=seeds[offs[i]:offs[j]],
                                 call=spans[i][0] if j10r else None)
                i = j
            out = self.units.multi([(a[i], bs[i], seeds[offs[i]:offs[i + 1]]) for i in range(lo, hi)], live, j10r)
            self._copy.wait_stream(main)
            with torch.cuda.stream(self._copy):
                got[offs[lo]:offs[hi]].copy_(out, non_blocking=True)
            out.record_stream(self._copy)
            copied = torch.cuda.Event()
            copied.record(self._copy)
            for i in range(lo, hi):
                ready[batch[i][0].index] = (batch[i][1], words[offs[i]:offs[i + 1]], copied)
        self._ready = ready
        self.seconds += time.perf_counter() - t0

    def __call__(self, call: Any, codes: Any, s_a: Any, w: Any, s_w: Any) -> np.ndarray:
        if self.run is None:
            raise ValueError("the receipt producer is bound to no run: its units' seeds take the run's salt (`bind`)")
        t0 = time.perf_counter()
        salt, j10r = self.run.salt, self.j10r
        u0, n = self.start(call)
        ra, rb = self.roots(call)
        m = len(codes)
        live = self._live(m)
        ready = self._ready.pop(call.index, None)
        if ready is not None and ready[0] is codes:
            ready[2].synchronize()
            out, a, b = ready[1], None, None
        else:
            a, b = self.operands(call, codes, w)
            if (a.shape[0] // R) * (b.shape[0] // C_) != n:
                raise ValueError(f"call {call.index}: {n} units in the window Program, not its {a.shape[0]} × "
                                 f"{b.shape[0]} tiles")
            if self.units is not None:
                seeds = self.units.seeds(self._record(salt, ra, rb), u0, n, call=u0 if j10r else None)
                out = self.units.receipts(a, b, seeds, live, j10r).cpu().numpy().view(np.uint32)
            elif self.skip_dead:
                out = gated_receipts(Call(a, b), np.arange(a.shape[0]) < m, salt, u0, ra, rb, self.cubin)[0]
            else:
                out = call_receipts(Call(a, b), salt, u0, ra, rb, self.cubin, j10r=j10r)
        if self._keep is not None and self._keep(call):
            if a is None:
                a, b = self.operands(call, codes, w)
            self.kept[call.index] = {"a": a.cpu(), "b": b.cpu(), "salt": salt, "base": u0, "roots": (ra, rb),
                                     "receipts": out.copy(), "m": m, "pub": self.public(call), "skip": self.skip_dead,
                                     "j10r": j10r}
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return out


__all__ = ["Call", "Cubin", "LANES", "NOISE_CU", "Receipts", "call_receipts", "call_stream", "call_streams",
           "gated_receipts", "gated_reference", "j10r_counters", "layout", "loaded", "noise_blocks", "noise_blocks_cpu",
           "noise_cubin", "reference", "seed_record", "stream_blocks", "tile_receipts", "unit_seeds", "units"]
