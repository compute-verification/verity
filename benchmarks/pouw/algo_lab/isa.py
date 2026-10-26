"""A counted instruction set: exact lane semantics on numpy arrays, one counter per instruction class.

Every operation returns its exact result (words are uint32 and wrap mod 2^32; IMAD.WIDE's are uint64) and adds the number
of lane operations it performed (the result's size) to its class, under the current phase. An MMA counts multiply-
accumulates instead, and refuses an operand outside its format's exact domain (P1): s8 operands must lie in [-128, 127],
and a bf16 product is exact only for integer operands of magnitude at most 256 with every partial sum under 2^24 in its
fp32 accumulator. Shared-memory traffic is counted in bytes (class ``SMEM.B``) beside the LDS or STS lane operations.

Phases: ``word`` (the default) is work that scales with the number of committed words, n^3 / L; ``setup`` is work that
scales with n^2 or less (noise forming, operand combinations, tables), reported apart and divided out at the reference n.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import numpy as np

M32 = 0xFFFFFFFF
U32 = np.uint32
U64 = np.uint64

#: lanes of one wgmma m64n256k32 s8 warp instruction (a quarter of the warpgroup's 524,288 MACs)
WGMMA_S8_MACS_PER_WARP_INSTR = 64 * 256 * 32 // 4
BF16_EXACT_OPERAND = 256
FP32_EXACT = 1 << 24


class DomainError(ValueError):
    """An MMA operand or accumulator left its format's exact domain (P1)."""


def u32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).astype(U32) if not (isinstance(x, np.ndarray) and x.dtype == U32) else x


def rotl(x: np.ndarray, r) -> np.ndarray:
    r = np.asarray(r, dtype=U32)
    return ((x << r) | (x >> (U32(32) - r))) & U32(M32)


class Kernel:
    """The counter. ``counts[(phase, cls)]`` is lane operations (MACs for ``MMA.*``, bytes for ``SMEM.B``)."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.phase_name = "word"

    @contextmanager
    def phase(self, name: str):
        old, self.phase_name = self.phase_name, name
        try:
            yield
        finally:
            self.phase_name = old

    def add(self, cls: str, n: int) -> None:
        if n:
            self.counts[(self.phase_name, cls)] += int(n)

    def by_phase(self, phase: str) -> dict[str, int]:
        return {c: v for (p, c), v in self.counts.items() if p == phase}

    # ---- lane ALU and IMAD-pipe instructions (uint32 in, uint32 out) -----------------------------------------------

    def lop3(self, a, b, c, f) -> np.ndarray:
        """Any 3-input bitwise function ``f(a, b, c)`` (one LOP3 a lane)."""
        out = (f(u32(a), u32(b), u32(c)) & U32(M32)).astype(U32)
        self.add("LOP3", out.size)
        return out

    def xor3(self, a, b, c) -> np.ndarray:
        return self.lop3(a, b, c, lambda x, y, z: x ^ y ^ z)

    def xor2(self, a, b) -> np.ndarray:
        return self.lop3(a, b, 0, lambda x, y, z: x ^ y)

    def and2(self, a, b) -> np.ndarray:
        return self.lop3(a, b, 0, lambda x, y, z: x & y)

    def shf_l_w(self, u, r: int) -> np.ndarray:
        """rotl(u, r): SHF.L.W, the funnel shift of (u, u)."""
        out = rotl(u32(u), r).astype(U32)
        self.add("SHF", out.size)
        return out

    def redux_xor(self, x: np.ndarray) -> np.ndarray:
        """redux.sync.xor across each warp (axis 0 grouped by 32 lanes): one warp instruction a warp, class ``REDUX``,
        counted in lanes (32 a warp) like every other class."""
        x = u32(x)
        out = np.bitwise_xor.reduce(x.reshape(-1, 32, *x.shape[1:]), axis=1).astype(U32)
        self.add("REDUX", x.size)
        return out

    def shfl_bfly(self, x: np.ndarray, m: int) -> np.ndarray:
        """SHFL.BFLY by lane mask ``m`` within each warp (axis 0 grouped by 32 lanes)."""
        x = u32(x)
        idx = np.arange(x.shape[0])
        out = x[(idx & ~31) | ((idx & 31) ^ m)]
        self.add("SHFL", x.size)
        return out

    def stg(self, lanes: int, nbytes: int) -> None:
        """A global store of ``lanes`` lanes (class ``STG``, values unchanged)."""
        self.add("STG", lanes)
        self.add("GMEM.B", nbytes)

    def iadd3(self, a, b, c=0, sb: int = 1, sc: int = 1) -> np.ndarray:
        """a + sb b + sc c mod 2^32, sb and sc in {1, -1} (IADD3's input negation)."""
        a, b, c = u32(a), u32(b), u32(c)
        out = a + (b if sb == 1 else U32(0) - b) + (c if sc == 1 else U32(0) - c)
        self.add("IADD3", np.broadcast(a, b, c).size)
        return out.astype(U32)

    def imad(self, a, b, c) -> np.ndarray:
        out = (u32(a) * u32(b) + u32(c)).astype(U32)
        self.add("IMAD", out.size)
        return out

    def imad_hi(self, a, b, c) -> np.ndarray:
        """hi32(a b) + c mod 2^32 (IMAD.HI.U32)."""
        hi = (u32(a).astype(U64) * u32(b).astype(U64)) >> U64(32)
        out = (hi.astype(U32) + u32(c)).astype(U32)
        self.add("IMAD.HI", out.size)
        return out

    def imad_wide(self, a, b, c64) -> np.ndarray:
        """a b + c mod 2^64 for 32-bit a, b (IMAD.WIDE.U32)."""
        out = u32(a).astype(U64) * u32(b).astype(U64) + np.asarray(c64, dtype=U64)
        self.add("IMAD.WIDE", out.size)
        return out

    def lea(self, a, b, s: int) -> np.ndarray:
        """(a << s) + b mod 2^32."""
        out = ((u32(a) << U32(s)) + u32(b)).astype(U32)
        self.add("LEA", out.size)
        return out

    def lea_hi_rot(self, u, r, y) -> np.ndarray:
        """rotl(u, r) + y (LEA.HI with both halves of its funnel u)."""
        out = (rotl(u32(u), r) + u32(y)).astype(U32)
        self.add("LEA.HI", out.size)
        return out

    def shf_r(self, a, s: int) -> np.ndarray:
        out = (u32(a) >> U32(s)).astype(U32)
        self.add("SHF", out.size)
        return out

    def shf_l(self, a, s: int) -> np.ndarray:
        out = ((u32(a) << U32(s)) & U32(M32)).astype(U32)
        self.add("SHF", out.size)
        return out

    def shf_r_wide(self, v64, s: int) -> np.ndarray:
        """The low word of a 64-bit pair shifted right by s (SHF.R.U64's funnel)."""
        out = ((np.asarray(v64, dtype=U64) >> U64(s)) & U64(M32)).astype(U32)
        self.add("SHF", out.size)
        return out

    def popc(self, a) -> np.ndarray:
        x = u32(a)
        out = np.zeros(x.shape, dtype=U32)
        for b in range(32):
            out += (x >> U32(b)) & U32(1)
        self.add("POPC", x.size)
        return out

    def idp4a(self, a, b, c) -> np.ndarray:
        """c + sum of the four s8 x s8 byte products of a and b, mod 2^32 (IDP4A.S8.S8)."""
        a, b = u32(a), u32(b)
        acc = u32(c).astype(np.int64)
        for k in range(4):
            x = ((a >> U32(8 * k)) & U32(0xFF)).astype(np.int64)
            y = ((b >> U32(8 * k)) & U32(0xFF)).astype(np.int64)
            acc = acc + np.where(x > 127, x - 256, x) * np.where(y > 127, y - 256, y)
        out = (acc & M32).astype(U32)
        self.add("IDP4A", out.size)
        return out

    def ffma(self, a, b, c) -> np.ndarray:
        """fp32 a b + c, held exact: every operand and the result an integer under 2^24 in magnitude."""
        a, b, c = (np.asarray(v, dtype=np.int64) for v in (a, b, c))
        out = a * b + c
        if np.abs(out).max(initial=0) >= FP32_EXACT or np.abs(a * b).max(initial=0) >= FP32_EXACT:
            raise DomainError("FFMA: an integer at or past 2^24 is not exact in fp32")
        self.add("FFMA", out.size)
        return out

    def f2i(self, x) -> np.ndarray:
        out = (np.asarray(x, dtype=np.int64) & M32).astype(U32)
        self.add("F2I", out.size)
        return out

    def uni(self, n: int) -> None:
        """n uniform-datapath instructions (one a warp, not a lane)."""
        self.add("UNI", n)

    # ---- memory -----------------------------------------------------------------------------------------------------

    def lds(self, table: np.ndarray, idx, bytes_per_lane: int = 4) -> np.ndarray:
        """Each lane reads table[idx] from shared memory (LDS.32 at 4 bytes a lane)."""
        out = table[idx]
        self.add("LDS", out.size)
        self.add("SMEM.B", out.size * bytes_per_lane)
        return out

    def lds_bulk(self, lane_ops: int, nbytes: int) -> None:
        """Operand loads a register-tiled kernel issues (counted from its tiling, values unchanged)."""
        self.add("LDS", lane_ops)
        self.add("SMEM.B", nbytes)

    def sts_bulk(self, lane_ops: int, nbytes: int) -> None:
        self.add("STS", lane_ops)
        self.add("SMEM.B", nbytes)

    def smem_read(self, nbytes: int) -> None:
        """Bytes an MMA reads from shared memory by descriptor (wgmma's operands): no lane instruction."""
        self.add("SMEM.B", nbytes)

    # ---- tensor cores -----------------------------------------------------------------------------------------------

    def mma_s8(self, a: np.ndarray, b: np.ndarray, c=None) -> np.ndarray:
        """c + a @ b mod 2^32 on the s8 tensor path; a is (m, k), b is (k, n), both in [-128, 127]."""
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        for name, x in (("a", a), ("b", b)):
            if x.size and (x.min() < -128 or x.max() > 127):
                raise DomainError(f"MMA.S8: operand {name} spans [{x.min()}, {x.max()}], outside s8")
        prod = (a @ b) & M32
        out = prod.astype(U32) if c is None else (prod.astype(U32) + u32(c)).astype(U32)
        macs = a.shape[0] * a.shape[1] * b.shape[1]
        self.add("MMA.S8", macs)
        return out

    def mma_bf16(self, a: np.ndarray, b: np.ndarray, c=None) -> np.ndarray:
        """c + a @ b on the bf16 path with fp32 accumulation, held exact (integers, |operand| <= 256, sums < 2^24)."""
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        for name, x in (("a", a), ("b", b)):
            if x.size and np.abs(x).max() > BF16_EXACT_OPERAND:
                raise DomainError(f"MMA.BF16: operand {name} reaches {np.abs(x).max()}, past bf16's exact integers")
        out = a @ b + (0 if c is None else np.asarray(c, dtype=np.int64))
        if np.abs(out).max(initial=0) >= FP32_EXACT:
            raise DomainError("MMA.BF16: an fp32 accumulator reached 2^24")
        self.add("MMA.BF16", a.shape[0] * a.shape[1] * b.shape[1])
        return out
