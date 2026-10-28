"""The J10-R kernels `check.py` and `bench.py` drive, through ctypes on torch's current stream:

* `H1u`: a build of benchmarks/hash/dither/h1u.cu (its sm_90a arms; any build exporting
  kh_seeds_j10r and kh_units_multi_j10r): one call's seeds (kh_seeds_j10r: seed_c at the call's first position, sigma_R
  at base + u) and every unit's receipt (kh_units_multi_j10r, mode 0: the call's row tiles, then its proof blocks; with ``sm``
  kh_units_multi_j10r_sm on every SM, the served launch). Receipts only: z is the served GEMM's.
* `Ref`: j10r_ref.so (j10r_ref.cu), the fallback: seeds, receipts and z of listed proof blocks.

Both take A [M, K] and B [N, K] int8 on the GPU, and the call's (salt, root_a, root_b, base).
"""
from __future__ import annotations

import ctypes

import numpy as np
import torch

from verity.core.protocols.pouw.circuit import seeds as SD

T = 32
#: 0xA5A5A5A5 and 0x5A5A5A5A as s32: a receipt word equals either with probability 2^-32, so over a call of 10^9 words
#: one poison alone flags real words; a word is unwritten only where it holds each poison after the run under it
POISONS = (-1515870811, 1515870810)
POISON = POISONS[0]
#: h1u.cu's UCallR, field for field
UCALLR = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"), ("rws", "<u8"), ("N", "<u4"),
                   ("K", "<u4"), ("u0", "<u4"), ("md", "<u4")])


def be_words(b: bytes) -> list[int]:
    return [int.from_bytes(b[i:i + 8], "big") for i in range(0, len(b), 8)]


def seed_data(salt: bytes, root_a: bytes, root_b: bytes) -> torch.Tensor:
    """kh_seeds_j10r's words (CALL_SEED_TAG, RECEIPT_TAG, salt, root_a, root_b; big-endian u64) and NOISE_TAG's, on the GPU."""
    w = be_words(SD.CALL_SEED_TAG + SD.RECEIPT_TAG + salt + root_a + root_b + SD.NOISE_TAG)
    return torch.tensor(np.array(w, dtype=np.uint64).view(np.int64), device="cuda")


def digest(words) -> bytes:
    """A digest from its 8 big-endian u64 words (int64 on the device)."""
    return b"".join((int(v) & ((1 << 64) - 1)).to_bytes(8, "big") for v in words)


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def twice(call, run=None) -> dict:
    """``run`` (default ``call.run``) under each of `POISONS` in turn: the receipt words left unwritten (each poison where
    it was), the words that differ between the two runs otherwise, and the words equal to the first poison after its run
    (written words among them: the single-poison test's false alarms). ``call.out`` holds the second run's."""
    run = run or call.run
    call.poison(POISONS[0])
    run()
    torch.cuda.synchronize()
    first = call.out.clone()
    call.poison(POISONS[1])
    run()
    torch.cuda.synchronize()
    unwritten = (first == POISONS[0]) & (call.out == POISONS[1])
    unstable = int(((first != call.out) & ~unwritten).sum())
    left, hits = int(unwritten.sum()), int((first == POISONS[0]).sum())
    del first, unwritten
    return {"unwritten": left, "unstable": unstable, "words_equal_to_the_first_poison": hits}


class H1u:
    def __init__(self, path: str, sm: bool = False):
        h = self.h = ctypes.CDLL(path)
        vp, u32, u64 = ctypes.c_void_p, ctypes.c_uint, ctypes.c_ulonglong
        self.path, self.sm = path, sm
        h.kh_seeds_j10r.argtypes = [vp, u64, u64, u32, vp, vp]
        h.kh_units_multi_j10r.argtypes = [vp, u32, u32, u32, vp, ctypes.c_int, vp]
        h.kh_ucallr_bytes.restype = u32
        h.kh_j10r_rows_bytes.restype = u64
        h.kh_j10r_rows_bytes.argtypes = [u32, u32]
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        h.kh_digits.argtypes = [u32]
        if sm:
            h.kh_units_multi_j10r_sm.argtypes = [vp, u32, u32, u32, vp, u32, u32, vp, vp]
            h.kh_smc_words.restype = u32
        if int(h.kh_ucallr_bytes()) != UCALLR.itemsize:
            raise RuntimeError(f"{path}: UCallR is {int(h.kh_ucallr_bytes())} bytes, not {UCALLR.itemsize}")

    @staticmethod
    def has_sm(path: str) -> bool:
        h = ctypes.CDLL(path)
        return hasattr(h, "kh_units_multi_j10r_sm") and hasattr(h, "kh_smc_words")

    def words(self, k: int) -> int:
        return int(self.h.kh_words(k))

    def ws_bytes(self, m: int, n: int, k: int) -> int:
        """A call's row workspace: kh_j10r_jw_bytes (an H1U_JW build's: its column blocks and W' too), else
        kh_j10r_rows_bytes."""
        if hasattr(self.h, "kh_j10r_jw_bytes"):
            self.h.kh_j10r_jw_bytes.restype = ctypes.c_ulonglong
            self.h.kh_j10r_jw_bytes.argtypes = [ctypes.c_uint] * 3
            return int(self.h.kh_j10r_jw_bytes(m, n, k))
        return int(self.h.kh_j10r_rows_bytes(m, k))


class H1uCall:
    """One call on one H1u build: its seeds [units, 16] int64 ([seed_c, sigma_R] words), row workspace and receipts
    [units, words] int32."""

    def __init__(self, lib: H1u, a: torch.Tensor, b: torch.Tensor, salt: bytes, roots: tuple[bytes, bytes], base: int):
        self.lib, self.a, self.b, self.base = lib, a, b, base
        m, k = a.shape
        n = b.shape[0]
        self.units = (m // T) * (n // T)
        self.data = seed_data(salt, *roots)
        self.live = torch.ones(m // T, dtype=torch.uint8, device="cuda")
        self.seeds = torch.empty((self.units, 16), dtype=torch.int64, device="cuda")
        self.out = torch.empty((self.units, lib.words(k)), dtype=torch.int32, device="cuda")
        self.rws = torch.empty((lib.ws_bytes(m, n, k),), dtype=torch.uint8, device="cuda")
        self.next = torch.zeros(int(lib.h.kh_smc_words()), dtype=torch.int32, device="cuda") if lib.sm else None
        t = np.zeros(2, dtype=UCALLR)
        t[0] = (a.data_ptr(), b.data_ptr(), self.seeds.data_ptr(), self.out.data_ptr(), self.rws.data_ptr(), n, k, 0,
                int(lib.h.kh_digits(k)))
        t[1]["u0"] = self.units
        self.tab = torch.from_numpy(t.view(np.uint8).copy()).to("cuda")

    def poison(self, p: int = POISON) -> None:
        self.out.fill_(p)
        self.rws.fill_(p & 0xFF)
        self.seeds.fill_(p)

    def run(self) -> None:
        h, st, m = self.lib.h, _stream(), self.a.shape[0]
        if rc := h.kh_seeds_j10r(self.data.data_ptr(), self.base, self.base, self.units, self.seeds.data_ptr(), st):
            raise RuntimeError(f"kh_seeds_j10r rc {rc}")
        if self.lib.sm:
            rc = h.kh_units_multi_j10r_sm(self.tab.data_ptr(), 1, self.units, m, self.live.data_ptr(), 1, 1,
                                          self.next.data_ptr(), st)
        else:
            rc = h.kh_units_multi_j10r(self.tab.data_ptr(), 1, self.units, m, self.live.data_ptr(), 0, st)
        if rc:
            raise RuntimeError(f"kh_units_multi_j10r{'_sm' if self.lib.sm else ''} rc {rc}")


class Ref:
    def __init__(self, path: str):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        self.path = path
        h.jr_words.restype = u32
        h.jr_words.argtypes = [u32]
        h.jr_digits.argtypes = [u32]
        h.jr_seeds.argtypes = [vp, ctypes.c_ulonglong, vp, u32, vp, vp, vp]
        h.jr_units.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp]
        h.jr_z.argtypes = [vp, vp, u32, u32, u32, vp, u32, vp, vp]

    def words(self, k: int) -> int:
        return int(self.h.jr_words(k))


class RefCall:
    """Listed units of one call on j10r_ref.so: seed_c [8] and sigma [n, 8] int64 (big-endian digest words), receipts
    [n, words] int32 and z [n, 32, 32] int32."""

    def __init__(self, lib: Ref, a: torch.Tensor, b: torch.Tensor, salt: bytes, roots: tuple[bytes, bytes], base: int,
                 units: list[int] | None = None, z: bool = True):
        self.lib, self.a, self.b, self.base = lib, a, b, base
        m, k = a.shape
        n = b.shape[0]
        us = range((m // T) * (n // T)) if units is None else units
        self.units = torch.tensor(np.asarray(us, dtype=np.uint32).view(np.int32), device="cuda")
        self.n = len(us)
        self.data = seed_data(salt, *roots)
        self.seedc = torch.empty((8,), dtype=torch.int64, device="cuda")
        self.sigma = torch.empty((self.n, 8), dtype=torch.int64, device="cuda")
        self.out = torch.empty((self.n, lib.words(k)), dtype=torch.int32, device="cuda")
        self.z = torch.empty((self.n, T, T), dtype=torch.int32, device="cuda") if z else None

    def poison(self, p: int = POISON) -> None:
        self.out.fill_(p)
        self.seedc.fill_(p)
        self.sigma.fill_(p)
        if self.z is not None:
            self.z.fill_(p)

    def run(self, z: bool = True) -> None:
        h, st = self.lib.h, _stream()
        (m, k), n = self.a.shape, self.b.shape[0]
        if rc := h.jr_seeds(self.data.data_ptr(), self.base, self.units.data_ptr(), self.n, self.seedc.data_ptr(),
                            self.sigma.data_ptr(), st):
            raise RuntimeError(f"jr_seeds rc {rc}")
        if rc := h.jr_units(self.a.data_ptr(), self.b.data_ptr(), m, n, k, self.data.data_ptr(), self.seedc.data_ptr(),
                            self.sigma.data_ptr(), self.units.data_ptr(), self.n, self.out.data_ptr(), st):
            raise RuntimeError(f"jr_units rc {rc}")
        if z and self.z is not None and (rc := h.jr_z(self.a.data_ptr(), self.b.data_ptr(), m, n, k,
                                                       self.units.data_ptr(), self.n, self.z.data_ptr(), st)):
            raise RuntimeError(f"jr_z rc {rc}")
