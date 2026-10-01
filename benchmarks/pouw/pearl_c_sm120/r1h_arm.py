"""R1-H, the rotated chain from a keyed hot start (the store's `docs/pouw/new-designs.md` §2.1), as an arm of the bench harness
(interface v0.5): R1's arm (`r1_arm.py`) with the chain started at H = P_i Q_j instead of +0, U = fl(C - H') as the useful
output, and decode's k split into S parts that merge exactly by the per-atom lemma (§1.3).  **Not a PoUW scheme yet**, as R1's
arm: no epoch, beacon or transcript, and the keyed rotation is folded into the weights offline.

The hot start is r1_binade's centred one: P_i = s_i 2^e_i and Q_j = t_j 1.5 2^f_j, with keyed signs, e_i the rounded log2 of
A's row norm and f_j that of B's column norm times sqrt(32) 2^h / k (h = 10), so H = +-1.5 2^E exactly.  They are computed at
prepare, untimed: O(m k + n k), the weights' side once per epoch and A's norms from the quantizer that writes its codes.

`launch`, with `hash`: at m a multiple of 128, `gemm_r1h_b_u` (y and C's message digests from registers); else A's rows into a
zero-padded 64-row buffer, `gemm_r1h_d_u` over S parts of k (C and y stored; each tile's last part merges the others' C_s - H),
then `r1_digests` over C's real rows.  `nohash`: `gemm_r1h_s_u`, or `gemm_r1h_d_u` alone.  `nocheck` (decode only; elsewhere
the `hash` call): `gemm_r1h_n_d_u`, the split without the per-step check, to price the check.  S is the least power of two
with tiles x S at least twice the SMs (S | k / 64, at most 16), or R1H_SPLIT.

The gate, at the exact shape and flags: a second launch at S = 1, the serial chain and so the definition of C, on 3 CTAs
(`gemm_r1h_d_u`, then `r1_digests`), whose C, y and digests must equal the call's word for word (C and y where the call stores
them); that C must equal `SM120_UNPROMOTED`'s CPU chain from H at sampled cells, and y must equal bf16(fl(C - H')) there; with
`hash`, sampled digests must equal keyed BLAKE3 over C's rows; and the check must count no word whose split may differ from
the chain (no path recomputes one yet).  `known_bad` flips C's first word between the product and the hash, as R1's.

Shortcuts, flagged: R1's (m = 32 padded to 64 rows, scales 1, nothing commits A or the weights), and no recompute path for a
word that fails the split's check.

  python bench.py --arm-path <this directory> --arm r1h_arm:R1HChainSm120 ...
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import r1_arm as r1
from arm import Buffer, Gate, Work

R = r1.R
H_SHIFT = 10
MAX_SPLIT = 16


def _e4m3_values() -> np.ndarray:
    c = np.arange(256)
    e, m = (c >> 3) & 0xF, c & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((c & 0x7F) == 0x7F, 0.0, v)
    return np.where(c & 0x80, -v, v).astype(np.float32)


E4M3 = _e4m3_values()


def norms(codes: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Each row's Euclidean norm, from its E4M3 codes."""
    out = np.empty(len(codes))
    for s in range(0, len(codes), chunk):
        v = E4M3[codes[s:s + chunk]].astype(np.float64)
        out[s:s + chunk] = np.sqrt(np.einsum("ij,ij->i", v, v))
    return out


def hot_start(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """P (len(a)) and Q (len(b)) FP32, r1_binade.centred_start's with h = H_SHIFT: P_i = s_i 2^e_i, Q_j = t_j 1.5 2^f_j."""
    k = a.shape[1]
    g = np.random.default_rng(seed)

    def side(n: np.ndarray, shift: float, mant: float) -> np.ndarray:
        e = np.round(np.log2(np.where(n > 0, n * shift, 1.0)))
        return (np.where(g.integers(0, 2, len(n)) == 1, -1.0, 1.0) * np.exp2(e) * mant).astype(np.float32)

    return side(norms(a), 1.0, 1.0), side(norms(b), np.sqrt(32) / k * 2.0 ** H_SHIFT, 1.5)


def h_words(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """H's FP32 words, (len(p), len(q)): P_i Q_j, exact."""
    return np.outer(p, q).astype(np.float32).view(np.uint32)


def chain_from(a: np.ndarray, b: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The unpromoted sm_120 chain's FP32 words for every (row of a, row of b) from the hot start h (their H words), k32 atoms
    ascending."""
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij"))
    acc = h.ravel().astype(np.uint32)
    for s in range(0, a.shape[1], 32):
        acc = r1.group_sum_total_batch(r1.SM120_UNPROMOTED.atom, acc, a[ii, s:s + 32], b[jj, s:s + 32])[0]
    return acc.reshape(len(a), len(b))


def u_words(c: np.ndarray, h: np.ndarray, k: int) -> np.ndarray:
    """U = fl(C - H') in FP32, H' = H (1 - (k / 32) 2^-25) (exact at 256 | k)."""
    hf = np.float32(1.0) - np.float32(k // 32) * np.float32(2.0 ** -25)
    return (c.view(np.float32) - h.view(np.float32) * hf).astype(np.float32)


def bf16(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)


def split(m: int, n: int, k: int, sms: int) -> int:
    env = os.environ.get("R1H_SPLIT")
    if env:
        return int(env)
    tiles, kb, s = (m // R.BM_D) * (n // R.BM_D), k // 64, 1
    while tiles * s < 2 * sms and s < MAX_SPLIT and kb % (2 * s) == 0:
        s *= 2
    return s


class _Call:
    def __init__(self, arm: R1HChainSm120, hdev, shape, flags: frozenset[str], ops, corrupt: bool = False):
        rd = self.rd = arm.rd
        rd.sync()                          # the operands were filled on the harness's non-blocking stream; this runs on the null one
        self.arm, self.hdev, self.shape, self.ops, self.corrupt = arm, hdev, shape, ops, corrupt
        self.hashing = "hash" in flags
        m, n, k = self.m, self.n, self.k = r1.padded(shape.m), shape.n, shape.k
        self.fused = r1.fused(shape.m)
        self.stores_c = not self.fused
        if self.fused:
            self.kernel = "gemm_r1h_b_u" if self.hashing else "gemm_r1h_s_u"
        else:
            self.kernel = "gemm_r1h_n_d_u" if "nocheck" in flags else "gemm_r1h_d_u"
        self.s = 1 if self.fused else split(m, n, k, rd.sms)
        self.bufs: list[int] = []
        new = self._new
        self.xa = ops.a.ptr if m == shape.m else new(m * k)
        self.y, self.rs, self.cs = new(2 * m * n), rd.put(np.ones(m, "<f4").tobytes()), rd.put(np.ones(n, "<f4").tobytes())
        self.bufs += [self.rs, self.cs]
        self.c = new(4 * m * n) if self.stores_c else 0
        self.digests = new(32 * shape.m * (n // 32)) if self.hashing else 0
        self.key = rd.put(r1.KEY)
        self.bufs.append(self.key)
        a_codes = np.frombuffer(rd.read(ops.a.ptr, shape.m * k), np.uint8).reshape(shape.m, k)
        b_codes = np.frombuffer(rd.read(ops.b.ptr, n * k), np.uint8).reshape(n, k)
        p, self.qv = hot_start(a_codes, b_codes, shape.mkn)
        self.pv = np.concatenate([p, np.ones(m - shape.m, np.float32)])
        top = np.log2(np.abs(self.pv[:shape.m])).min() + np.log2(np.abs(self.qv)).min()
        if top < 18:
            raise ValueError(f"H's top bit {top:.2f} at or below a product's (448^2 < 2^18)")
        self.pq = rd.put(self.pv.astype("<f4").tobytes()), rd.put(self.qv.astype("<f4").tobytes())
        self.bufs += list(self.pq)
        if self.s > 1:
            self.ws = (new(4 * m * n), new(4 * m * n), new(4 * (m // R.BM_D) * (n // R.BM_D)), new(16))
        else:
            self.ws = (0, 0, 0, 0)
        self.fails = self.ws[3]
        self.tma = 0
        if self.fused:
            ab = rd.tmap(self.xa, k, m) + rd.tmap(ops.b.ptr, k, n)
            self.tma = rd.put(ab + ab)
            self.bufs.append(self.tma)
        for name in (self.kernel, "gemm_r1h_d_u"):
            arm.ensure_smem(name)
        copy = [lambda: rd.copy_async(self.xa, ops.a.ptr, shape.m * k)] if self.xa != ops.a.ptr else []
        flip = [lambda: hdev.xor_byte(self.c if self.stores_c else self.digests, 0, 0x01, rd.stream)] if corrupt else []
        gemm = lambda: self.gemm(self.kernel, c=self.c, y=self.y, dg=self.digests, s=self.s)  # noqa: E731
        after = flip + ([lambda: self.digest(self.c, self.digests)] if self.hashing and not self.fused else [])
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

    def gemm(self, name: str, c: int = 0, y: int = 0, dg: int = 0, s: int = 1, grid: int | None = None) -> None:
        rd, arm = self.rd, self.arm
        ptrs = [self.xa, self.ops.b.ptr, c, 0, y, *self.pq, self.rs, self.cs, dg, 0, self.key, 0, 0, 0, 0]
        if R.is_ml(name):
            ptrs[10], block = self.tma, R.ML_NT
            items, cap = (self.m // R.BM) * (self.n // R.BM), rd.sms
        else:
            ptrs[12:16] = self.ws if s > 1 else (0, 0, 0, 0)
            block = R.NT
            items, cap = (self.m // R.BM_D) * (self.n // R.BM_D) * s, 8 * rd.sms
        rd.launch(name, min(items, grid or cap), block, arm.smem[name], p=ptrs, i=(self.m, self.n, self.k, R.RASTER, s))

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

    def reference(self) -> dict[str, int]:
        """The serial chain: gemm_r1h_d_u at S = 1 on 3 CTAs (C and y), and r1_digests over that C."""
        rd, m, n = self.rd, self.m, self.n
        ref = {"c": rd.alloc(4 * m * n), "y": rd.alloc(2 * m * n), "digests": rd.alloc(32 * self.shape.m * (n // 32))}
        self.gemm("gemm_r1h_d_u", c=ref["c"], y=ref["y"], s=1, grid=r1.REF_GRID)
        self.digest(ref["c"], ref["digests"])
        rd.sync()
        return ref

    def words(self, p: int, nbytes: int, dtype: str = "<u4") -> np.ndarray:
        return np.frombuffer(self.rd.read(p, nbytes), dtype)


class R1HChainSm120:
    name = "r1h-chain-sm120"
    stream = "fp8"
    precision = "fp8-e4m3"
    inputs = "e4m3"
    flag_sets = {"hash": frozenset({"hash"}), "nohash": frozenset(), "nocheck": frozenset({"hash", "nocheck"})}

    def __init__(self):
        self.ship = r1.Path(os.environ.get("PEARLC_SHIP", r1.HERE))
        self.cubin = self.ship / "pearl_c_sm120.cubin"
        self.rd = None
        self.smem: dict[str, int] = {}

    ensure_smem = r1.R1ChainSm120.ensure_smem

    def build(self, ctx: dict) -> dict:
        self.rd = r1.device(self.cubin)
        return {"kernels": "pearl_c_sm120.cu's gemm_r1h_b_u, gemm_r1h_s_u, gemm_r1h_d_u, gemm_r1h_n_d_u and r1_digests",
                "binaries": [str(self.cubin)], "cubin_sha256": hashlib.sha256(self.cubin.read_bytes()).hexdigest(),
                "not_a_pouw_scheme": "R1-H's deciding primitive: no epoch, beacon or transcript",
                "conditional_on": "Daniel's approval ruling and the red team's glue-unit or residual-state condition",
                "hot_start": f"centred, h = {H_SHIFT}: P_i = s_i 2^e_i, Q_j = t_j 1.5 2^f_j from the operands' norms (prepare)",
                "split_k": "decode: S parts from H, merged exactly; S = least power of two with tiles S >= 2 SMs, or R1H_SPLIT",
                "shortcuts": ["m padded to 64 rows at decode; work counts issued MACs", "scales 1",
                              "digest key a stand-in, blake3('r1/message')", "no recompute path for a word failing the check"]}

    def domain(self, shape) -> str | None:
        if shape.n % 128 or shape.k % 256:
            return "n a multiple of 128 and k of 256 (H' exact)"
        if not r1.fused(shape.m) and r1.padded(shape.m) * shape.n * 4 > 1 << 31:
            return "C over 2 GiB at an unfused shape"
        return None

    def work(self, shape) -> Work:
        m = r1.padded(shape.m)
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
            rows = sorted(rng.sample(range(shape.m), min(r1.SAMPLE_ROWS, shape.m)))
            cols = sorted(rng.sample(range(n), min(r1.SAMPLE_COLS, n)))
            a = np.stack([np.frombuffer(rd.read(call.xa + r * k, k), np.uint8) for r in rows])
            b = np.stack([np.frombuffer(rd.read(call.ops.b.ptr + c * k, k), np.uint8) for c in cols])
            h = h_words(call.pv[rows], call.qv[cols])
            c_ref = call.words(ref["c"], 4 * m * n).reshape(m, n)
            twin = chain_from(a, b, h)
            twin_bad = int(np.count_nonzero(twin != c_ref[np.ix_(rows, cols)]))
            y_ref = call.words(ref["y"], 2 * m * n, "<u2").reshape(m, n)
            y_bad = int(np.count_nonzero(bf16(u_words(twin, h, k)) != y_ref[np.ix_(rows, cols)]))
            fails = int(call.words(call.fails, 4)[0]) if call.fails else 0
            hv = h.view(np.float32).astype(np.float64)
            excursion = np.abs(twin.view(np.float32).astype(np.float64) - hv) / np.abs(hv)
            detail = {"split": call.s, "kernel": call.kernel, "words_differing_from_serial": differ,
                      "twin_cells": len(rows) * len(cols), "twin_cells_differing": twin_bad, "twin_y_differing": y_bad,
                      "check_failures": fails, "twin_max_c_minus_h_over_h": float(excursion.max())}
            ok = twin_bad == 0 and y_bad == 0 and fails == 0 and not any(differ.values())
            checked = {name: "all" for name in call.outputs}
            if call.hashing:
                per = n // 32
                got = rd.read(call.digests, 32 * shape.m * per)
                picks = [(rng.randrange(shape.m), rng.randrange(per)) for _ in range(r1.SAMPLE_DIGESTS)]
                bad = sum(got[32 * (r * per + j):32 * (r * per + j + 1)]
                          != r1.keyed(r1.KEY, c_ref[r, 32 * j:32 * j + 32].astype("<u4").tobytes()) for r, j in picks)
                detail["blake3_digests_sampled"], detail["blake3_digests_differing"] = len(picks), int(bad)
                ok = ok and bad == 0
            return Gate(ok, checked, detail)
        finally:
            for p in ref.values():
                rd.free(p)
