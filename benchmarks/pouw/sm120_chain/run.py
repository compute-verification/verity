#!/usr/bin/env python3
"""Pearl-C's ticket chain on one RTX PRO 6000 (sm_120): the honest chain timed against the unpromoted cheat, and two rate
probes, each gated before it is timed (chain.cu; the attacker's measurement, a check on γ, not a bench arm).

1. Identity: exactly one visible device, `CUDA_VISIBLE_DEVICES` equal to --gpu with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, an
   RTX PRO 6000 at cc 12.0 whose UUID is --expect-uuid (`internal/pouw/rtx-pro/server.md`'s table); NVML by that UUID.
   Exit 3 otherwise.
2. Build and SASS gate: chain.cu with nvcc >= 12.8 for sm_120a, no spills or local memory, and per k-step of 128 exactly
   64 `QMMA.16832.F32.E4M3.E4M3` (+ 64 FADD, the promotions, in chain_honest only), 64 IMMA in chain_int8, 64 F16 QMMA in
   chain_f16acc.  Exit 5 otherwise.
3. Gates (exit 4 on any mismatch, nothing timed): on formed rows (the census's families under §5's forming on the sm_120
   atom, 128 x 128 words at k = 8192) every word of chain_honest equals the census twin's G4 ticket word and every word of
   chain_unpromoted its pure chain (`pearlc_census.chain` on `sm120-e4m3-k32`, ±0 equal), and chain_int8 equals the exact
   integer product; then at each timed shape on 256 sampled words.  chain_f16acc has no reference (a rate probe only).
4. Timing at each shape (stand-in operands: E4M3 codes of Gaussians at the formed rows' median magnitude): REPS reps, each
   running every variant once in a seeded random order, CALLS launches between two events after a read-only L2 flush,
   NVML clocks, power, temperature and throttle reasons per item (clocks are whatever the run line locked; recorded).
5. Report: per variant the median ms per call and TFLOPS; the promotion's time share 1 - t(unpromoted)/t(honest) beside
   its W1 price (p_fadd/128 per MAC); t(int8)/t(honest) and t(f16acc)/t(honest), this kernel's rate ratios (prices for
   pearlc_attacks.py --prices); and the unpromoted attack's success on the full output (words, 64 x 64 tiles).
   --reps 0 stops after the gates and the unpromoted chain's success: a correctness run on the GPU alone.

  CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID python benchmarks/pouw/sm120_chain/run.py --expect-uuid GPU-...
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402

ATOM = "sm120-e4m3-k32"
BM = BN = BK = 128
SMEM = 3 * 2 * BM * BK
VARIANTS = {"honest": "chain_honest", "unpromoted": "chain_unpromoted", "int8": "chain_int8", "f16acc": "chain_f16acc"}
SASS = {"chain_honest": {"QMMA.16832.F32.E4M3.E4M3": 64, "FADD": 64},
        "chain_unpromoted": {"QMMA.16832.F32.E4M3.E4M3": 64, "FADD": 0},
        "chain_int8": {"IMMA.16832.S8.S8": 64}, "chain_f16acc": {"QMMA.16832.F16.E4M3.E4M3": 64}}
GATE_FAMILIES = ("gaussian", "outliers-groupstart", "aligned-spikes-pm1-r444")
FORMED_MEDIAN = 52.0            # the median |code| of s5-formed Gaussian rows (pearlc_attacks on gaussian, k = 8192)
L2_BYTES = 128 << 20

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDriverGetVersion": [P(i32)], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32],
    "cuDeviceGetName": [C.c_char_p, i32, i32], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuEventDestroy_v2": [vp], "cuGetErrorString": [i32, P(C.c_char_p)],
}
NVML = {"nvmlInit_v2": [], "nvmlDeviceGetHandleByUUID": [C.c_char_p, P(vp)], "nvmlSystemGetDriverVersion": [C.c_char_p, u],
        "nvmlDeviceGetClockInfo": [vp, i32, P(u)], "nvmlDeviceGetCurrentClocksThrottleReasons": [vp, P(C.c_ulonglong)],
        "nvmlDeviceGetPowerUsage": [vp, P(u)], "nvmlDeviceGetTemperature": [vp, i32, P(u)]}


def bind(lib, sigs):
    for f, a in sigs.items():
        getattr(lib, f).argtypes, getattr(lib, f).restype = a, i32
    return lib


def emit(fh, obj: dict) -> None:
    line = json.dumps(obj)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


class Dev:
    def __init__(self):
        self.cu = bind(C.CDLL("libcuda.so.1"), CUDA)
        self.ck("cuInit", 0)
        self.count = self.get("cuDeviceGetCount")
        self.dev = self.get("cuDeviceGet", 0)
        b = C.create_string_buffer(256)
        self.ck("cuDeviceGetName", b, 256, self.dev)
        self.name = b.value.decode()
        uu = C.create_string_buffer(16)
        self.ck("cuDeviceGetUuid_v2", uu, self.dev)
        h = uu.raw.hex()
        self.uuid = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        self.cc = (self.get("cuDeviceGetAttribute", 75, self.dev), self.get("cuDeviceGetAttribute", 76, self.dev))
        self.sms = self.get("cuDeviceGetAttribute", 16, self.dev)
        self.driver = self.get("cuDriverGetVersion")
        ctx = vp()
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), self.dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.fns: dict[str, vp] = {}

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f}: CUresult {r} ({s.value.decode() if s.value else '?'})")

    def get(self, f: str, *a) -> int:
        v = i32()
        self.ck(f, C.byref(v), *a)
        return v.value

    def load(self, cubin: bytes) -> None:
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), cubin)
        for name in [*VARIANTS.values(), "l2_flush"]:
            f = vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
            self.fns[name] = f
            if name != "l2_flush":
                self.ck("cuFuncSetAttribute", f, 8, SMEM)      # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

    def alloc(self, n: int) -> int:
        p = u64()
        self.ck("cuMemAlloc_v2", C.byref(p), n)
        self.ck("cuMemsetD8_v2", p, 0, n)
        return p.value

    def put(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        p = self.alloc(arr.nbytes)
        self.ck("cuMemcpyHtoD_v2", u64(p), arr.ctypes.data_as(vp), arr.nbytes)
        return p

    def get_words(self, p: int, n: int) -> np.ndarray:
        out = np.empty(n, np.uint32)
        self.ck("cuMemcpyDtoH_v2", out.ctypes.data_as(vp), u64(p), out.nbytes)
        return out

    def free(self, p: int) -> None:
        self.ck("cuMemFree_v2", u64(p))

    def launch(self, name: str, grid: tuple[int, int, int], block: int, smem: int, args: list) -> None:
        held = [a for a in args]
        params = (vp * len(held))(*[C.cast(C.pointer(a), vp) for a in held])
        self.ck("cuLaunchKernel", self.fns[name], *grid, block, 1, 1, smem, None, params, None)

    def chain(self, variant: str, a: int, b: int, out: int, m: int, n: int, k: int) -> None:
        self.launch(VARIANTS[variant], (n // BN, m // BM, 1), 256, SMEM,
                    [u64(a), u64(b), u64(out), i32(m), i32(n), i32(k)])

    def sync(self) -> None:
        self.ck("cuCtxSynchronize")

    def elapsed(self, body) -> float:
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
        return ms.value


class Nvml:
    def __init__(self, uuid: str):
        self.lib, self.h, self.driver = None, vp(), ""
        try:
            lib = bind(C.CDLL("libnvidia-ml.so.1"), NVML)
        except (OSError, AttributeError):
            return
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)) == 0:
            self.lib = lib
            b = C.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(b, 96) == 0:
                self.driver = b.value.decode()

    def read(self) -> dict:
        if not self.lib:
            return {}
        sm, mem, pw, t, thr = u(), u(), u(), u(), C.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, C.byref(sm))
        self.lib.nvmlDeviceGetClockInfo(self.h, 2, C.byref(mem))
        self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.h, C.byref(thr))
        self.lib.nvmlDeviceGetPowerUsage(self.h, C.byref(pw))
        self.lib.nvmlDeviceGetTemperature(self.h, 0, C.byref(t))
        return {"sm_mhz": sm.value, "mem_mhz": mem.value, "throttle": hex(thr.value), "power_w": round(pw.value / 1000, 1),
                "temp_c": t.value}


def build(out: Path, nvcc: str) -> tuple[bytes, dict]:
    """chain.cu -> chain.cubin for sm_120a, and the SASS gate; returns the cubin and its build record."""
    cuobjdump = str(Path(nvcc).with_name("cuobjdump"))
    cubin = out / "chain.cubin"
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xptxas", "-v", "-cubin",
           "-o", str(cubin), str(HERE / "chain.cu")]
    ptx = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    sass = subprocess.run([cuobjdump, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    problems = [ln for ln in ptx.splitlines() if re.search(r"warning|error|serializ", ln, re.I)]
    problems += [f"spill or stack: {ln.strip()}" for ln in ptx.splitlines()
                 if re.search(r"bytes stack frame", ln) and not re.search(r"0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads", ln)]
    kernels: dict[str, dict[str, int]] = {}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            kernels[cur] = {}
            continue
        ins = re.search(r"\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if cur and ins:
            op = ins.group(1)
            for key in ("QMMA.16832.F32.E4M3.E4M3", "QMMA.16832.F16.E4M3.E4M3", "IMMA.16832.S8.S8", "FADD", "LDL", "STL",
                        "LDSM", "LDGSTS", "HMMA", "OMMA"):
                if op == key or op.startswith(key + "."):
                    kernels[cur][key] = kernels[cur].get(key, 0) + 1
    for name, want in SASS.items():
        got = kernels.get(name, {})
        for key, n in want.items():
            if got.get(key, 0) != n:
                problems.append(f"{name}: {got.get(key, 0)} {key}, want {n}")
        if got.get("LDL") or got.get("STL") or got.get("HMMA") or got.get("OMMA"):
            problems.append(f"{name}: unexpected {got}")
    data = cubin.read_bytes()
    return data, {"nvcc": version, "cmd": " ".join(cmd), "cubin_sha256": hashlib.sha256(data).hexdigest(),
                  "source_sha256": hashlib.sha256((HERE / "chain.cu").read_bytes()).hexdigest(), "sass": kernels,
                  "problems": problems}


def same_words(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x == y) | (((x & 0x7FFFFFFF) == 0) & ((y & 0x7FFFFFFF) == 0))


def twin(a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> dict[str, np.ndarray]:
    """The census twin's words for the pairs (ii, jj): the G4 ticket word, the pure chain, the exact int8 product."""
    A, B = a[ii], b[jj]
    words = {}
    for name, G in (("honest", 4), ("unpromoted", 0)):
        final = census.chain(A, B, G)[4]
        words[name] = np.asarray(final, np.float32).view(np.uint32)
    ints = np.concatenate([(A[s:s + 1024].view(np.int8).astype(np.int32) * B[s:s + 1024].view(np.int8).astype(np.int32)).sum(1)
                           for s in range(0, len(A), 1024)])
    words["int8"] = ints.astype(np.int32).view(np.uint32)
    return words


def formed(fam: str, k: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    XA, XB, _, _ = census.family_rows(fam, k, 1.0, rows, rows, 1)
    FcA, FcB = census.s5_lines(k, 1, 16.0)
    a = census.form_s5(XA, FcA, 1.0, 16.0, census._rng(1, fam, k, 1.0, "unit-A"))[0]
    b = census.form_s5(XB, FcB, 1.0, 16.0, census._rng(1, fam, k, 1.0, "unit-B"))[0]
    return a, b


def standin(rows: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((rows, k), np.uint8)
    sigma = FORMED_MEDIAN / 0.6745
    for r0 in range(0, rows, 256):
        out[r0:r0 + 256] = census.enc(rng.standard_normal((min(256, rows - r0), k)) * sigma)
    return out


def gate(dev: Dev, a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray, what: str) -> dict:
    m, k = a.shape
    n = b.shape[0]
    da, db, out = dev.put(a), dev.put(b), dev.alloc(4 * m * n)
    ref = twin(a, b, ii, jj)
    res = {"gate": what, "m": m, "n": n, "k": k, "words": int(len(ii))}
    for v in ("honest", "unpromoted", "int8"):
        dev.chain(v, da, db, out, m, n, k)
        dev.sync()
        got = dev.get_words(out, m * n)[ii * n + jj]
        bad = int((~same_words(got, ref[v])).sum())
        res[f"{v}_mismatches"] = bad
    res["twin_unpromoted_equal_words"] = round(float(same_words(ref["honest"], ref["unpromoted"]).mean()), 6)
    for p in (da, db, out):
        dev.free(p)
    res["ok"] = all(res[f"{v}_mismatches"] == 0 for v in ("honest", "unpromoted", "int8"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--shapes", default="8192x8192x8192", help="MxNxK, comma-separated; multiples of 128")
    ap.add_argument("--reps", type=int, default=20, help="timed reps per shape; 0: the gates and the unpromoted chain only")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "sm120_chain.jsonl", "w")
    census.set_atom(ATOM)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu) or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        emit(fh, {"refused": f"need CUDA_VISIBLE_DEVICES={args.gpu} and CUDA_DEVICE_ORDER=PCI_BUS_ID"})
        sys.exit(3)
    dev = Dev()
    ident = {"name": dev.name, "uuid": dev.uuid, "cc": list(dev.cc), "sms": dev.sms, "cuda_driver_api": dev.driver,
             "visible_devices": dev.count}
    if dev.count != 1 or dev.uuid != args.expect_uuid or dev.cc != (12, 0) or "RTX PRO 6000" not in dev.name:
        emit(fh, {"refused": "identity", **ident, "expect_uuid": args.expect_uuid})
        sys.exit(3)
    nvml = Nvml(dev.uuid)
    emit(fh, {"identity": {**ident, "nvml_driver": nvml.driver, "numpy": np.__version__, "atom": ATOM}})

    cubin, rec = build(out_dir, args.nvcc)
    emit(fh, {"build": rec})
    if rec["problems"]:
        sys.exit(5)
    dev.load(cubin)
    flush = dev.alloc(4 * L2_BYTES)
    sink = dev.alloc(16)

    ok = True
    for fam in GATE_FAMILIES:
        a, b = formed(fam, 8192, BM)
        ii, jj = (x.ravel() for x in np.meshgrid(np.arange(BM), np.arange(BN), indexing="ij"))
        r = gate(dev, a, b, ii, jj, f"formed/{fam}")
        emit(fh, r)
        ok &= r["ok"]
    if not ok:
        sys.exit(4)

    summary = []
    rng = random.Random(args.seed)
    for shape in args.shapes.split(","):
        m, n, k = map(int, shape.split("x"))
        a, b = standin(m, k, args.seed), standin(n, k, args.seed + 1)
        srng = np.random.default_rng(args.seed + 2)
        ii, jj = srng.integers(0, m, 256), srng.integers(0, n, 256)
        g = gate(dev, a, b, ii, jj, f"standin/{shape}")
        emit(fh, g)
        if not g["ok"]:
            sys.exit(4)
        da, db = dev.put(a), dev.put(b)
        outs = {v: dev.alloc(4 * m * n) for v in VARIANTS}
        for v in VARIANTS:
            dev.chain(v, da, db, outs[v], m, n, k)
        dev.sync()
        hon, unp = dev.get_words(outs["honest"], m * n), dev.get_words(outs["unpromoted"], m * n)
        eq = same_words(hon, unp).reshape(m // 64, 64, n // 64, 64)
        attack = {"shape": shape, "unpromoted_equal_words": round(float(eq.mean()), 6),
                  "unpromoted_equal_tiles_64": round(float(eq.all(axis=(1, 3)).mean()), 6)}
        emit(fh, attack)
        del hon, unp
        if not args.reps:
            summary.append(attack)
            for p in (da, db, *outs.values()):
                dev.free(p)
            continue

        times: dict[str, list[float]] = {v: [] for v in VARIANTS}
        for rep in range(args.reps):
            order = list(VARIANTS)
            rng.shuffle(order)
            for v in order:
                dev.launch("l2_flush", (4 * dev.sms, 1, 1), 512, 0, [u64(flush), sz(4 * L2_BYTES // 16), u64(sink)])
                ms = dev.elapsed(lambda v=v: [dev.chain(v, da, db, outs[v], m, n, k) for _ in range(args.calls)]) / args.calls
                times[v].append(ms)
                emit(fh, {"shape": shape, "rep": rep, "variant": v, "ms": round(ms, 5), **nvml.read()})
        med = {v: float(np.median(t)) for v, t in times.items()}
        row = {"shape": shape, "median_ms": {v: round(t, 5) for v, t in med.items()},
               "tflops": {v: round(2 * m * n * k / (t * 1e-3) / 1e12, 1) for v, t in med.items()},
               "promotion_time_share": round(1 - med["unpromoted"] / med["honest"], 5),
               "promotion_w1_share_public": round((8 / 128) / (1 + 8 / 128), 5),
               "int8_over_honest": round(med["int8"] / med["honest"], 5),
               "f16acc_over_honest": round(med["f16acc"] / med["honest"], 5), **attack,
               "clocks": "recorded per item (NVML); locked only if the run line locked them"}
        emit(fh, {"summary": row})
        summary.append(row)
        for p in (da, db, *outs.values()):
            dev.free(p)
    doc = {"tool": "pouw_sm120_chain", "identity": ident, "build": rec, "summary": summary}
    path = out_dir / "sm120_chain.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    s0 = summary[0]
    if not args.reps:
        run_result.write(path, True, "sm120-chain", f"gates passed (untimed); unpromoted equal words {s0['unpromoted_equal_words']}",
                         [("unpromoted_equal_words", s0["unpromoted_equal_words"], "fraction"),
                          ("unpromoted_equal_tiles_64", s0["unpromoted_equal_tiles_64"], "fraction")])
        return
    run_result.write(path, True, "sm120-chain", f"gates passed; promotion time share {s0['promotion_time_share']}",
                     [("promotion_time_share", s0["promotion_time_share"], "fraction"),
                      ("honest_tflops", s0["tflops"]["honest"], "TFLOPS"),
                      ("unpromoted_equal_words", s0["unpromoted_equal_words"], "fraction"),
                      ("int8_over_honest", s0["int8_over_honest"], "ratio"),
                      ("f16acc_over_honest", s0["f16acc_over_honest"], "ratio")])


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(json.dumps({"seconds": round(time.time() - t0, 1)}), file=sys.stderr)
