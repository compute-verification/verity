"""R1, the rotated chain (the store's `docs/pouw/new-designs.md` §2.1), as an arm of the bench harness (interface v0.5): its
deciding primitive, the canonical unpromoted sm_120 chain with C's FP32 words hashed, timed against the harness's autotuned
plain FP8 GEMMs at the exact shape.  **Not a PoUW scheme yet**: no epoch, beacon or transcript, and the keyed rotation is
folded into the weights offline, so on the harness's E4M3 operands the call is the plain GEMM plus binding C.

`launch`, with `hash`: at m a multiple of 128, `gemm_r1_b_u` (y and C's message digests from registers, C never stored);
else A's rows into a zero-padded 64-row buffer, `gemm_r1_c_d_u` (y and C stored, 64 x 64 tiles, one accumulator over all of k:
no split-K) and `r1_digests` over C's real rows.  Without `hash` (`nohash`), the same call with y alone: `gemm_fast`, or
`gemm_plain_d_u` at 64 x 64.  `phases` splits the call for the dependent chain: `before` (the pad copy), `gemm`, and `after`
(`r1_digests`, unfused only), which the chain may run on a side stream.  The digest key is a stand-in, blake3("r1/message").

The gate, at the exact shape and flags: a second launch on another kernel and tile schedule, on 3 CTAs (`gemm_ct_u` or
`gemm_ct_d_u` for C, `gemm_fast` or `gemm_plain_d_u` for y, then `r1_digests` over that C), must equal the call's y, C where it
stores C, and digests, word for word; that C must equal `SM120_UNPROMOTED`'s CPU chain (the captured atom, one accumulator
from +0 over all of k) at sampled cells; with `hash`, sampled digests must equal keyed BLAKE3
(`verity.commitments.blake3.keyed`) over those cells' rows of C.  `known_bad` flips C's first word between the product and the
hash (the first digest's word, fused).

Shortcuts, flagged: decode's m = 32 is padded to 64 rows (2x its tensor work; `work` counts issued MACs); the scales are 1;
nothing commits A or the weights, which Verity's own records do.

  python bench.py --arm-path <this directory> --arm r1_arm:R1ChainSm120 ...

PEARLC_SHIP names the directory holding pearl_c_sm120.cubin (build.sh's output; default this directory).  Needs numpy, and
`verity` and `verity_pouw` importable (the atom, BLAKE3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
from arm import Buffer, Gate, Work

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = _load("pearl_c_sm120_run", HERE / "run.py")

from verity.commitments.blake3 import blake3, keyed  # noqa: E402
from verity.ml.kernels import group_sum_total_batch  # noqa: E402
from verity_pouw.schemes.pearl_c_device import SM120_UNPROMOTED  # noqa: E402

KEY = blake3(b"r1/message")
REF_GRID = 3
SAMPLE_ROWS = SAMPLE_COLS = 16
SAMPLE_DIGESTS = 64


def padded(m: int) -> int:
    return -(-m // R.BM_D) * R.BM_D


def fused(m: int) -> bool:
    return m % R.BM == 0


def chain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The unpromoted sm_120 chain's FP32 words for every (row of a, row of b): one accumulator from +0, k32 atoms ascending."""
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij"))
    acc = np.zeros(len(ii), np.uint32)
    for s in range(0, a.shape[1], 32):
        acc = group_sum_total_batch(SM120_UNPROMOTED.atom, acc, a[ii, s:s + 32], b[jj, s:s + 32])[0]
    return acc.reshape(len(a), len(b))


_DEV = None


def device(cubin: Path):
    """The kernels' module in the device's primary context, loaded once per process (the harness's context too)."""
    global _DEV
    if _DEV is None:
        _DEV = R.Dev(cubin.read_bytes())
        _DEV.open()
    return _DEV


class _Call:
    def __init__(self, arm: R1ChainSm120, hdev, shape, flags: frozenset[str], ops, corrupt: bool = False):
        rd = self.rd = arm.rd
        rd.sync()                          # the operands were filled on the harness's non-blocking stream; this runs on the null one
        self.arm, self.hdev, self.shape, self.ops, self.corrupt = arm, hdev, shape, ops, corrupt
        self.hashing = "hash" in flags
        m, n, k = self.m, self.n, self.k = padded(shape.m), shape.n, shape.k
        self.fused = fused(shape.m)
        self.stores_c = self.hashing and not self.fused
        self.bufs: list[int] = []
        new = self._new
        self.xa = ops.a.ptr if m == shape.m else new(m * k)
        self.y, self.rs, self.cs = new(2 * m * n), rd.put(np.ones(m, "<f4").tobytes()), rd.put(np.ones(n, "<f4").tobytes())
        self.bufs += [self.rs, self.cs]
        self.c = new(4 * m * n) if self.stores_c else 0
        self.digests = new(32 * shape.m * (n // 32)) if self.hashing else 0
        self.key = rd.put(KEY)
        self.bufs.append(self.key)
        self.tma = 0
        if self.fused:
            ab = rd.tmap(self.xa, k, m) + rd.tmap(ops.b.ptr, k, n)
            self.tma = rd.put(ab + ab)
            self.bufs.append(self.tma)
        copy = [lambda: rd.copy_async(self.xa, ops.a.ptr, shape.m * k)] if self.xa != ops.a.ptr else []
        flip = [lambda: hdev.xor_byte(self.c if self.stores_c else self.digests, 0, 0x01, rd.stream)] if corrupt else []
        for name in ("gemm_r1_b_u", "gemm_fast", "gemm_ct_u") if self.fused else ("gemm_r1_c_d_u", "gemm_plain_d_u", "gemm_ct_d_u"):
            arm.ensure_smem(name)
        if self.hashing and self.fused:
            gemm = lambda: self.gemm("gemm_r1_b_u", y=self.y, dg=self.digests)  # noqa: E731
            after = []
        elif self.hashing:
            gemm = lambda: self.gemm("gemm_r1_c_d_u", c=self.c, y=self.y)  # noqa: E731
            after = flip + [lambda: self.digest(self.c, self.digests)]
        else:
            gemm = lambda: self.gemm("gemm_fast" if self.fused else "gemm_plain_d_u", y=self.y)  # noqa: E731
            after = []
        if corrupt and self.fused:
            after = flip
        self.parts = {"before": copy, "gemm": [gemm], "after": after}
        self.phases = {name: (lambda stream, fs=fs: self.enqueue(fs, stream)) for name, fs in self.parts.items()}
        self.outputs = {"y": Buffer(self.y, 2 * m * n)}
        if self.stores_c:
            self.outputs["c"] = Buffer(self.c, 4 * m * n)
        if self.hashing:
            self.outputs["digests"] = Buffer(self.digests, 32 * shape.m * (n // 32))
        self.enqueue(copy, None)
        rd.sync()

    def _new(self, n: int) -> int:
        p = self.rd.alloc(n)
        self.bufs.append(p)
        return p

    def gemm(self, name: str, c: int = 0, y: int = 0, dg: int = 0, grid: int | None = None, a: int | None = None,
             tma: int | None = None) -> None:
        rd, arm = self.rd, self.arm
        tm, bn = R.gemm_tile(name)
        tiles = (self.m // tm) * (self.n // bn)
        ptrs = [a or self.xa, self.ops.b.ptr, c, 0, y, 0, 0, self.rs, self.cs, dg, 0, self.key]
        block = R.NT
        if R.is_ml(name):
            ptrs[10], block = tma or self.tma, R.ML_NT
        rd.launch(name, min(tiles, grid or rd.sms), block, arm.smem[name], p=ptrs, i=(self.m, self.n, self.k, R.RASTER))

    def digest(self, c: int, out: int) -> None:
        rows, n = self.shape.m, self.n
        self.rd.launch("r1_digests", R.cdiv(rows * (n // 32), 128), 128, p=(c, self.key, out), i=(rows, n))

    def enqueue(self, steps: list, stream) -> None:
        rd = self.rd
        rd.stream = stream
        try:
            for f in steps:
                f()
        finally:
            rd.stream = None

    def launch(self, stream) -> None:
        self.enqueue([f for part in self.parts.values() for f in part], stream)

    def weight_side(self, stream) -> None:
        pass

    def free(self) -> None:
        for p in self.bufs:
            self.rd.free(p)
        self.bufs.clear()

    # ---- the gate ----------------------------------------------------------------------------------------------------------

    def reference(self) -> dict[str, int]:
        """The second launch, unfused on 3 CTAs: C by gemm_ct(_d)_u, y by gemm_fast or gemm_plain_d_u, digests by r1_digests."""
        rd, m, n = self.rd, self.m, self.n
        ref = {"c": rd.alloc(4 * m * n), "y": rd.alloc(2 * m * n), "digests": rd.alloc(32 * self.shape.m * (n // 32))}
        self.gemm("gemm_ct_u" if self.fused else "gemm_ct_d_u", c=ref["c"], grid=REF_GRID)
        self.gemm("gemm_fast" if self.fused else "gemm_plain_d_u", y=ref["y"], grid=REF_GRID)
        self.digest(ref["c"], ref["digests"])
        rd.sync()
        return ref

    def words(self, p: int, nbytes: int) -> np.ndarray:
        return np.frombuffer(self.rd.read(p, nbytes), "<u4")


class R1ChainSm120:
    name = "r1-chain-sm120"
    stream = "fp8"
    precision = "fp8-e4m3"
    inputs = "e4m3"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset()}

    def __init__(self):
        self.ship = Path(os.environ.get("PEARLC_SHIP", HERE))
        self.cubin = self.ship / "pearl_c_sm120.cubin"
        self.rd = None
        self.smem: dict[str, int] = {}

    def ensure_smem(self, name: str) -> None:
        """A GEMM's dynamic shared memory, from its `_smem` probe, set on the kernel once (before any capture)."""
        rd = self.rd
        if name not in self.smem:
            probe = rd.alloc(16)
            rd.launch(name + "_smem", 1, 1, p=(probe,))
            rd.sync()
            self.smem[name] = int.from_bytes(rd.read(probe, 4), "little")
            rd.free(probe)
            rd.ck("cuFuncSetAttribute", rd.fn(name), R.MAX_DYN, self.smem[name])

    def build(self, ctx: dict) -> dict:
        self.rd = device(self.cubin)
        return {"kernels": "pearl_c_sm120.cu's gemm_r1_b_u, gemm_r1_c_d_u, gemm_plain_d_u, gemm_fast and r1_digests",
                "binaries": [str(self.cubin)], "cubin_sha256": hashlib.sha256(self.cubin.read_bytes()).hexdigest(),
                "not_a_pouw_scheme": "R1's deciding primitive: no epoch, beacon or transcript",
                "shortcuts": ["m padded to 64 rows at decode; work counts issued MACs", "scales 1",
                              "digest key a stand-in, blake3('r1/message')"]}

    def domain(self, shape) -> str | None:
        if shape.n % 128 or shape.k % 128:
            return "n and k must be multiples of 128"
        if not fused(shape.m) and padded(shape.m) * shape.n * 4 > 1 << 31:
            return "C over 2 GiB at an unfused shape"
        return None

    def work(self, shape) -> Work:
        m = padded(shape.m)
        return Work(tensor_macs={"fp8-e4m3": m * shape.k * shape.n}, w_ref=str(shape.mkn), credited=str(shape.mkn), unit="mac",
                    hashed_bytes={"c-words": 4 * shape.m * shape.n})

    def prepare(self, dev, shape, flags: frozenset[str], operands, corrupt: bool = False) -> _Call:
        return _Call(self, dev, shape, flags, operands, corrupt)

    def known_bad(self, dev, shape, operands) -> _Call:
        return self.prepare(dev, shape, self.flag_sets["hash"], operands, corrupt=True)

    def gate(self, dev, call: _Call, shape, flags: frozenset[str]) -> Gate:
        rd, m, n, k = self.rd, call.m, call.n, call.k
        ref = call.reference()
        try:
            differ = {name: int(np.count_nonzero(call.words(buf.ptr, buf.nbytes) != call.words(ref[name], buf.nbytes)))
                      for name, buf in call.outputs.items()}
            rng = random.Random(shape.mkn)
            rows = sorted(rng.sample(range(shape.m), min(SAMPLE_ROWS, shape.m)))
            cols = sorted(rng.sample(range(n), min(SAMPLE_COLS, n)))
            a = np.stack([np.frombuffer(rd.read(call.xa + r * k, k), np.uint8) for r in rows])
            b = np.stack([np.frombuffer(rd.read(call.ops.b.ptr + c * k, k), np.uint8) for c in cols])
            c_ref = call.words(ref["c"], 4 * m * n).reshape(m, n)
            twin = chain(a, b)
            twin_bad = int(np.count_nonzero(twin != c_ref[np.ix_(rows, cols)]))
            detail = {"words_differing_from_relaunch": differ, "twin_cells": len(rows) * len(cols), "twin_cells_differing": twin_bad}
            ok = twin_bad == 0 and not any(differ.values())
            checked = {name: "all" for name in call.outputs}
            if call.hashing:
                per = n // 32
                got = rd.read(call.digests, 32 * shape.m * per)
                picks = [(rng.randrange(shape.m), rng.randrange(per)) for _ in range(SAMPLE_DIGESTS)]
                bad = sum(got[32 * (r * per + h):32 * (r * per + h + 1)] != keyed(KEY, c_ref[r, 32 * h:32 * h + 32].astype("<u4").tobytes())
                          for r, h in picks)
                detail["blake3_digests_sampled"], detail["blake3_digests_differing"] = len(picks), int(bad)
                ok = ok and bad == 0
            return Gate(ok, checked, detail)
        finally:
            for p in ref.values():
                rd.free(p)
