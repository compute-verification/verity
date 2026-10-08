#!/usr/bin/env python3
"""Pearl-C4 (``pearl-c-nvfp4-v0``, ``pearl-c-mxfp4-v0``) on one sm_120 GPU from the prebuilt pearl_c4.cubin, through the CUDA
driver API (ctypes on libcuda.so.1; stdlib and numpy, nothing built on the machine).

  python3 dev.py check CHECK_DIR [--expect-uuid GPU-...]      every kernel on fixture.py's inputs, every buffer byte for byte
  python3 dev.py time MxNxK [...] [--fmt nvfp4|mxfp4] [--reps R] [--expect-uuid GPU-...]
                                                              self-timing for hillclimbing (the harness's numbers are the record)

The check runs the weight side and one call in each mode (hash with C~ and U stored, nohash, plain) at every GEMM geometry,
and compares every buffer the kernels write with the fixture, the rows' commitment roots and the seeds included.  One JSON
object per line on stdout.  Exit 3: identity refused; 4: a buffer differs.

The host supplies only public inputs: the operands, the epoch's salt and each commitment's domain id (`dom_a`, `dom_b`: the
32-byte ids verity.ml.commitments derives from the salt, the matmul's index and the row count).  The device commits A's and
B's rows as the verifier checks them (`commit_rows`) and derives the seeds from the roots; self-timing takes labels for the
salt and the domain ids.

The pipeline (`Pipeline`) is also the FP4 bench arm's (arm.py): `weight_side` is the per-(weight, epoch) work, `launch` the
per-call prover step, each on a given stream and capturable into a CUDA graph.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SM_COUNT, MAX_DYN_SMEM, CARVEOUT, CARVEOUT_MAX_SHARED = 16, 8, 9, 100
ROLE_EA, ROLE_FA, ROLE_EB, ROLE_FB = 0, 1, 2, 3
FMTS = {"nvfp4": ("nv", 16), "mxfp4": ("mx", 32)}
MODES = ("hash", "nohash", "plain")

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGet": [P(i32), i32], "cuDeviceGetCount": [P(i32)],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuMemcpyDtoDAsync_v2": [u64, u64, sz, vp],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuEventDestroy_v2": [vp], "cuGetErrorString": [i32, P(C.c_char_p)],
}


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Dev:
    """The device's primary context (shared with the harness), the module, allocations and launches."""

    def __init__(self, cubin: bytes, expect_uuid: str | None = None):
        self.cu = C.CDLL(os.environ.get("PEARLC4_LIBCUDA", "libcuda.so.1"))
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        self.ck("cuInit", 0)
        count = self.get("cuDeviceGetCount")
        self.dev = self.get("cuDeviceGet", 0)
        name = C.create_string_buffer(256)
        self.ck("cuDeviceGetName", name, 256, self.dev)
        self.name = name.value.decode()
        raw = C.create_string_buffer(16)
        self.ck("cuDeviceGetUuid_v2", raw, self.dev)
        h = raw.raw.hex()
        self.uuid = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
        self.cc = (self.get("cuDeviceGetAttribute", 75, self.dev), self.get("cuDeviceGetAttribute", 76, self.dev))
        self.sms = self.get("cuDeviceGetAttribute", SM_COUNT, self.dev)
        self.driver = self.get("cuDriverGetVersion")
        self.identity = {"name": self.name, "uuid": self.uuid, "cc": list(self.cc), "sms": self.sms, "driver_api": self.driver,
                         "visible": count, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if expect_uuid is not None and (count != 1 or self.uuid != expect_uuid):
            raise PermissionError(f"identity refused: {count} visible device(s), UUID {self.uuid}, expected {expect_uuid}")
        if self.cc != (12, 0):
            raise PermissionError(f"identity refused: compute capability {self.cc}, pearl_c4.cubin is sm_120a")
        ctx = vp()
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), self.dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), cubin)
        self.fns: dict[str, vp] = {}
        self.smem: dict[str, int] = {}

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def get(self, f: str, *a) -> int:
        v = i32()
        self.ck(f, C.byref(v), *a)
        return v.value

    def fn(self, name: str) -> vp:
        if name not in self.fns:
            f = vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
            self.fns[name] = f
        return self.fns[name]

    def dyn_smem(self, name: str) -> int:
        """A GEMM's dynamic shared memory, from its _smem kernel, and the attribute set to allow it."""
        if name not in self.smem:
            p = self.alloc(4)
            self.launch(name + "_smem", 1, 1, p=[p])
            self.sync()
            self.smem[name] = int.from_bytes(self.read(p, 4), "little")
            self.free(p)
            self.ck("cuFuncSetAttribute", self.fn(name), MAX_DYN_SMEM, self.smem[name])
        return self.smem[name]

    def alloc(self, n: int) -> int:
        p = u64()
        self.ck("cuMemAlloc_v2", C.byref(p), max(n, 16))
        self.ck("cuMemsetD8_v2", p, 0, max(n, 16))
        return p.value

    def free(self, p: int) -> None:
        self.ck("cuMemFree_v2", u64(p))

    def write(self, p: int, data: bytes) -> None:
        self.ck("cuMemcpyHtoD_v2", u64(p), C.c_char_p(data), len(data))

    def read(self, p: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ck("cuMemcpyDtoH_v2", buf, u64(p), n)
        return buf.raw

    def copy(self, dst: int, src: int, n: int, stream: int = 0) -> None:
        self.ck("cuMemcpyDtoDAsync_v2", u64(dst), u64(src), n, vp(stream or None))

    def launch(self, name: str, grid: int | tuple[int, int], block: int, smem: int = 0, p=(), i=(), stream: int = 0) -> None:
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        params = (vp * 1)(C.addressof(a))
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        self.ck("cuLaunchKernel", self.fn(name), gx, gy, 1, block, 1, 1, smem, vp(stream or None), params, None)

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def timed(self, body, reps: int) -> list[float]:
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, None)
            body()
            self.ck("cuEventRecord", b, None)
            self.ck("cuEventSynchronize", b)
            self.ck("cuEventElapsedTime", C.byref(ms), a, b)
            self.cu.cuEventDestroy_v2(a)
            self.cu.cuEventDestroy_v2(b)
            out.append(ms.value)
        return out


def commit_blocks(rows: int) -> int:
    """commit_rows' grid: one 32-rank warp per CTA over the ranks padded to a power of two."""
    return max(1, (1 << max(rows - 1, 0).bit_length()) // 32)


def buffer_sizes(fmt: str, m: int, n: int, k: int) -> dict[str, int]:
    blk = FMTS[fmt][1]
    return {
        "xa": 2 * m * k, "xb": 2 * n * k, "seed_a": 32, "seed_b": 32, "salt": 32, "dom_a": 32, "dom_b": 32,
        "root_a": 32, "root_b": 32, "ticket_a": 4, "ticket_b": 4,
        "scratch_a": 32 * commit_blocks(m) + 192, "scratch_b": 32 * commit_blocks(n) + 192,
        "lines_ea": 16 * m, "lines_eb": 16 * n, "lines_fa": 16 * k, "lines_fb": 16 * k, "fat": 16 * k, "fbt": 16 * k,
        "a_codes": m * k // 2, "a_scales": m * k // blk, "a_alpha": 4 * m, "a_inv": 4 * m, "a_ae": m,
        "b_codes": n * k // 2, "b_scales": n * k // blk, "b_alpha": 4 * n, "b_inv": 4 * n, "b_ae": n,
        "pa": 128 * m, "pb": 128 * n, "g": 4096, "bf": 128 * n,
        "y": 2 * m * n, "digests": m * n, "leaves": (m // 64) * (n // 64) * 32,
    }


def form_split(rows: int, k: int, sms: int) -> int:
    """Forming's k-split S (a power of two): enough (rows/16) x S CTAs to reach every SM, at least 8 k64 chunks each."""
    s = 1
    while (rows // 16) * s < sms and k // 64 // (2 * s) >= 8:
        s *= 2
    return s


def domain(m: int, n: int, k: int) -> str | None:
    if not (1024 <= k <= 1 << 16 and k % 512 == 0):
        return "k: 1,024 <= k <= 2^16 with 512 | k (the scheme's k_min and 128 | k; commit_rows' 1 KB row multiple)"
    if m % 64 or n % 128:
        return "m a multiple of 64 (the tile; pad with zero filler rows) and n of 128 (the CTA width)"
    if max(m, n) > 1 << 16:
        return "at most 2^16 rows per commitment (commit_rows' two-byte ranks)"
    return None


class Pipeline:
    """One (format, shape) instance: its buffers and its kernels in order.  `external` maps buffer names (xa, xb) to
    device pointers the caller owns (the harness's operand sets); the rest are allocated here.  `index` is the matmul's
    index in A's seed."""

    def __init__(self, dev: Dev, fmt: str, m: int, n: int, k: int, bm: int | None = None, store_cu: bool = False,
                 external: dict[str, int] | None = None, split: int | None = None, chain_h: int | None = None,
                 bn: int | None = None, index: int = 0):
        why = domain(m, n, k)
        if why:
            raise ValueError(f"{m}x{n}x{k} outside pearl_c4.cu's domain: {why}")
        self.dev, self.fmt, self.m, self.n, self.k, self.index = dev, fmt, m, n, k, index
        self.split_a = split or form_split(m, k, dev.sms)
        self.split_b = split or form_split(n, k, dev.sms)
        self.chain_h = chain_h
        self.tag, self.blk = FMTS[fmt]
        self.bm = bm or (128 if m % 128 == 0 else 64)
        self.bn = bn or (64 if self.bm == 64 and (m // 64) * (n // 128) < dev.sms else 128)
        if (self.bm, self.bn) not in ((128, 128), (64, 128), (64, 64)):
            raise ValueError(f"no {self.bm} x {self.bn} GEMM in pearl_c4.cu")
        self.sizes = buffer_sizes(fmt, m, n, k)
        self.external = dict(external or {})
        if store_cu:
            self.sizes.update(c=4 * m * n, u=4 * m * n)
        self.buf = {name: self.external.get(name) or dev.alloc(size) for name, size in self.sizes.items()}
        self.threads = 32 * (self.bm // 32) * (self.bn // 64)
        self.form = f"form_{self.tag}"
        self.form_smem = dev.dyn_smem(self.form)
        if self.form_smem:
            dev.ck("cuFuncSetAttribute", dev.fn(self.form), CARVEOUT, CARVEOUT_MAX_SHARED)

    def gemm(self, mode: str) -> str:
        return f"gemm_{self.tag}_{mode}" + ("" if self.bm == 128 else "_m64") + ("" if self.bn == 128 else "n64")

    def chain32(self, rows: int, p: list[int], i: list[int]) -> tuple[str, tuple, dict]:
        """The launch of a rows x 32 chain: two 16-row blocks per CTA where that still fills every SM."""
        h = self.chain_h or (32 if rows % 32 == 0 and rows >= 32 * self.dev.sms else 16)
        name = f"chain32_{self.tag}" + ("_r2" if h == 32 else "")
        return name, (rows // h, 8 * h, self.dev.dyn_smem(name)), dict(p=p, i=i)

    def commit(self, side: str) -> tuple[str, tuple, dict]:
        """The launch of A's (side "a") or B's commitment and seed: the root, then seed_a (reading root_b) or seed_b."""
        b, k = self.buf, self.k
        rows, xs, extra = (self.m, "xa", b["root_b"]) if side == "a" else (self.n, "xb", 0)
        return "commit_rows", (commit_blocks(rows), 32), dict(
            p=[b[xs], b[f"dom_{side}"], b[f"scratch_{side}"], b[f"ticket_{side}"], b[f"root_{side}"], b["salt"], extra,
               b[f"seed_{side}"]], i=[rows, 2 * k, self.index])

    def weight_side(self, stream: int = 0, hashing: bool = True) -> None:
        """Per (weight, epoch): B's commitment and seed_b, F_A and F_B's lines, F's transposes, B's forming (with E_B's
        lines), G, B~F_A, P_B.  hashing=False leaves the commitment out (seed_b as it stands)."""
        d, b, n, k = self.dev, self.buf, self.n, self.k
        L = lambda *a, **kw: d.launch(*a, stream=stream, **kw)  # noqa: E731
        if hashing:
            name, geo, kw = self.commit("b")
            L(name, *geo, **kw)
        L("lines", cdiv(k, 512), 128, p=[b["seed_b"], b["lines_fa"]], i=[ROLE_FA, k])
        L("lines", cdiv(k, 512), 128, p=[b["seed_b"], b["lines_fb"]], i=[ROLE_FB, k])
        if not hashing:
            L("lines", cdiv(n, 512), 128, p=[b["seed_b"], b["lines_eb"]], i=[ROLE_EB, n])
        L("transpose", cdiv(16 * k, 256), 256, p=[b["lines_fa"], b["fat"]], i=[k])
        L("transpose", cdiv(16 * k, 256), 256, p=[b["lines_fb"], b["fbt"]], i=[k])
        L(self.form, (n // 16, self.split_b), 256, self.form_smem, p=[
            b["xb"], b["lines_eb"], b["lines_fb"], b["b_codes"], b["b_scales"], b["b_alpha"], b["b_inv"], b["b_ae"], b["pb"],
            b["seed_b"] if hashing else 0], i=[n, k, 32, 1, ROLE_EB])
        for name, geo, kw in (self.chain32(32, [b["fbt"], 0, b["fat"], b["g"]], [32, k, 0]),
                              self.chain32(n, [b["b_codes"], b["b_scales"], b["fat"], b["bf"]], [n, k, 0])):
            L(name, *geo, **kw)
        L("pb", cdiv(32 * n, 256), 256, p=[b["pb"], b["g"], b["bf"]], i=[n])

    def prime(self, stream: int = 0) -> None:
        """A's commitment and seed_a, then E_A's lines once, so that the nohash call (which leaves the XOF out) forms from
        real lines."""
        name, geo, kw = self.commit("a")
        self.dev.launch(name, *geo, stream=stream, **kw)
        self.dev.launch("lines", cdiv(self.m, 512), 128, p=[self.buf["seed_a"], self.buf["lines_ea"]], i=[ROLE_EA, self.m],
                        stream=stream)

    def steps(self, mode: str = "hash") -> list[tuple[str, tuple, dict]]:
        """One call's launches: (hash) A's commitment and seed_a, A's forming (with E_A's lines from seed_a in the hash
        mode; the others read the primed ones), A'F_B into P_A, the chain with the peel and (hash) the epilogue's digests,
        then (hash) the tile leaves."""
        b, m, n, k = self.buf, self.m, self.n, self.k
        out = []
        if mode == "hash":
            out.append(self.commit("a"))
        out.append((self.form, ((m // 16, self.split_a), 256, self.form_smem), dict(
            p=[b["xa"], b["lines_ea"], b["lines_fa"], b["a_codes"], b["a_scales"], b["a_alpha"], b["a_inv"], b["a_ae"], b["pa"],
               b["seed_a"] if mode == "hash" else 0], i=[m, k, 0, 0, ROLE_EA])))
        out.append(self.chain32(m, [b["a_codes"], b["a_scales"], b["fbt"], b["pa"]], [m, k, 1]))
        g = self.gemm(mode)
        out.append((g, ((m // self.bm) * (n // self.bn), self.threads, self.dev.dyn_smem(g)), dict(
            p=[b["a_codes"], b["a_scales"], b["b_codes"], b["b_scales"], b["pa"], b["pb"], b["a_inv"], b["b_inv"], b["y"],
               b["digests"], b.get("c", 0), b.get("u", 0)], i=[m, n, k])))
        if mode == "hash":
            out.append(("leaves", (cdiv((m // 64) * (n // 64), 64), 64), dict(p=[b["digests"], b["leaves"]], i=[m, n])))
        return out

    def launch(self, stream: int = 0, mode: str = "hash") -> None:
        for name, geo, kw in self.steps(mode):
            self.dev.launch(name, *geo, stream=stream, **kw)

    def free(self) -> None:
        for name, p in self.buf.items():
            if name not in self.external:
                self.dev.free(p)


# ---------------------------------------------------------------- the check

WRITTEN = {
    "weight": ("root_b", "seed_b", "lines_fa", "lines_fb", "lines_eb", "fat", "fbt", "b_codes", "b_scales", "b_alpha", "b_inv",
               "b_ae", "pb", "g", "bf"),
    "hash": ("root_a", "seed_a", "lines_ea", "a_codes", "a_scales", "a_alpha", "a_inv", "a_ae", "pa", "c", "u", "y", "digests",
             "leaves"),
    "nohash": ("a_codes", "a_scales", "a_alpha", "a_inv", "a_ae", "pa", "y"),
    "plain": ("a_codes", "a_scales", "y_plain"),
}
WORD = {"a_alpha": 4, "a_inv": 4, "b_alpha": 4, "b_inv": 4, "g": 4, "bf": 4, "c": 4, "u": 4, "pa": 2, "pb": 2, "y": 2,
        "y_plain": 2, "digests": 32, "leaves": 32, "root_a": 32, "root_b": 32, "seed_a": 32, "seed_b": 32}


def diff(name: str, got: bytes, want: bytes) -> dict:
    w = WORD.get(name, 1)
    g, e = np.frombuffer(got, np.uint8).reshape(-1, w), np.frombuffer(want, np.uint8).reshape(-1, w)
    bad = np.nonzero((g != e).any(axis=1))[0]
    out = {"buffer": name, "words": len(e), "mismatches": int(len(bad)), "sha256": hashlib.sha256(got).hexdigest()}
    if len(bad):
        out["first"] = [{"index": int(j), "got": g[j].tobytes().hex(), "want": e[j].tobytes().hex()} for j in bad[:4]]
    return out


def check_fmt(dev: Dev, d: Path) -> bool:
    mf = json.loads((d / "manifest.json").read_text())
    m, n, k = mf["shape"]["m"], mf["shape"]["n"], mf["shape"]["k"]
    want = {name: (d / f"{name}.bin").read_bytes() for name in mf["files"]}
    for name, meta in mf["files"].items():
        if hashlib.sha256(want[name]).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"{d}/{name}.bin does not match its manifest")
    ok = True
    for bm, bn in ((128, 128), (64, 128), (64, 64)):
        if m % bm:
            continue
        wide = bm == 64 and m % 32 == 0 and n % 32 == 0
        pl = Pipeline(dev, mf["fmt"], m, n, k, bm=bm, bn=bn, store_cu=True, split=k // 64 // 8 if bm == 64 else 1,
                      chain_h=32 if wide else 16)
        for name in ("xa", "xb", "salt", "dom_a", "dom_b"):
            dev.write(pl.buf[name], want[name])
        pl.weight_side()
        dev.sync()
        rows = [diff(name, dev.read(pl.buf[name], len(want[name])), want[name]) for name in WRITTEN["weight"]]
        for mode in MODES:
            pl.launch(mode=mode)
            dev.sync()
            for name in WRITTEN[mode]:
                src = "y" if name == "y_plain" else name
                rows.append(diff(name, dev.read(pl.buf[src], len(want[name])), want[name]) | {"mode": mode})
        pl.free()
        for r in rows:
            r.update(fmt=mf["fmt"], bm=bm, bn=bn, split=[pl.split_a, pl.split_b], chain_h=pl.chain_h, shape=[m, n, k])
            emit(r)
            ok &= r["mismatches"] == 0
    return ok


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "time"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--expect-uuid")
    ap.add_argument("--fmt", default="nvfp4", choices=tuple(FMTS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--cubin", default=str(HERE / "pearl_c4.cubin"))
    a = ap.parse_args(argv)
    try:
        dev = Dev(Path(a.cubin).read_bytes(), a.expect_uuid)
    except PermissionError as e:
        emit({"refused": str(e)})
        return 3
    emit({"identity": dev.identity, "cubin_sha256": hashlib.sha256(Path(a.cubin).read_bytes()).hexdigest()})
    if a.what == "check":
        root = Path(a.args[0] if a.args else HERE / "check")
        ok = all([check_fmt(dev, root / fmt) for fmt in FMTS if (root / fmt).is_dir()])
        emit({"check": "pass" if ok else "FAIL"})
        return 0 if ok else 4
    rng = np.random.default_rng(20260930)
    for spec in a.args or ["8192x8192x8192"]:
        m, n, k = (int(v) for v in spec.split("x"))
        pl = Pipeline(dev, a.fmt, m, n, k)
        for name, rows in (("xa", m), ("xb", n)):
            x = rng.integers(-8, 9, (rows, k)).astype(np.float32).view(np.uint32) >> 16
            dev.write(pl.buf[name], x.astype("<u2").tobytes())
        for name in ("salt", "dom_a", "dom_b"):
            dev.write(pl.buf[name], hashlib.sha256(f"verity/pouw/pearl-c4-dev/{name}".encode()).digest())
        pl.weight_side()
        pl.prime()
        dev.sync()
        row = {"shape": [m, n, k], "fmt": a.fmt, "bm": pl.bm, "bn": pl.bn,
               "label": "self-timing for hillclimbing, not a panel number; clocks as the machine sets them"}
        row["weight_side_ms"] = statistics.median(dev.timed(pl.weight_side, a.reps))
        for mode in MODES:
            pl.launch(mode=mode)
            ms = statistics.median(dev.timed(lambda mode=mode: pl.launch(mode=mode), a.reps))
            row[f"{mode}_ms"] = ms
            row[f"{mode}_chain_pflops"] = round(2 * m * n * k / ms / 1e12, 3)
            row[f"{mode}_steps_ms"] = {name: round(statistics.median(dev.timed(
                lambda name=name, geo=geo, kw=kw: dev.launch(name, *geo, **kw), a.reps)), 4) for name, geo, kw in pl.steps(mode)}
        emit(row)
        pl.free()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
