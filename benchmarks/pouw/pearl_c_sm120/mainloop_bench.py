#!/usr/bin/env python3
"""Bench for mainloop_sm120.cuh on one RTX PRO 6000 (sm_120), through run.py's ctypes driver (stdlib only).

mainloop_bench.cubin holds GPU 1's kernels (pearl_c_sm120.cu, the gated references) and the mainloop's C~ kernels (ml_*), built
from mainloop_bench.cu:

  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o mainloop_bench.cubin \
    mainloop_bench.cu

0. Identity as run.py's (PEARLC_EXPECT_UUID).
1. Twin gate: at the fixture's shape (fixture.py's check.json), every ml_* kernel and GPU 1's gemm_ct / gemm_ct_u run on the
   fixture's A' and B~ and their C~ must equal the fixture's `c` (the CPU twin) byte for byte, on the full grid, 1 CTA and 3.
2. Big gate: at SHAPE (default 8192^3) on random non-NaN E4M3 codes, every ml_* kernel's C~ must equal gemm_ct's (G = 4) or
   gemm_ct_u's (G = 0) in every word; the two references must differ (so the gate tells G apart).
3. Diagnostic timing (not the headline; GPU 6's harness times arms): each kernel that passed both gates, REPS launches after a
   warm-up, the median, TFLOPS, NVML clocks before and after; with LT_LIB (the harness's lt.cu built with -DPH_LT_ONLY),
   cuBLASLt's FP8 GEMM on the same operands in the same round-robin.
4. STEADY=N: each kernel N times back to back (the power cap settles), round-robin, NVML sampled during each batch: ms, SM
   clock, power and cycles per GEMM.

  python3 mainloop_bench.py CUBIN CHECK_JSON    (env: PEARLC_EXPECT_UUID, PEARLC_DIR (run.py's directory), SHAPE=8192, REPS=20,
                                          ML="ml_ct ml_ct_u ...", RASTER=8, GATE_REPEATS=5, LT_LIB, STEADY,
                                          ONLY_CHECK=1, PROFILE=1: the big gate and one gemm_fast launch only, for ncu)
One JSON object per line on stdout, a summary on stderr.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import re
import os
import statistics
import struct
import sys
import time

sys.path.insert(0, os.environ.get("PEARLC_DIR", "benchmarks/pouw/pearl_c_sm120"))
import run as R  # noqa: E402

u32, u64, i32, vp, P = C.c_uint32, C.c_uint64, C.c_int, C.c_void_p, C.POINTER
ENCODE = [vp, i32, u32, vp, P(u64), P(u64), P(u32), P(u32), i32, i32, i32, i32]
UINT8, SWIZZLE_128B, L2_128B = 0, 3, 2
G_OF = {"gemm_ct": 4, "gemm_ct_u": 0}  # an ml_ kernel is unpromoted (G = 0) iff its name has "_u"
VARIANT_OF = {4: "sm120", 0: "sm120-unpromoted"}


def g_of(name: str) -> int:
    name = name.partition("@")[0]
    return G_OF.get(name, 0 if "_u" in name else 4)


def ml_tile(name: str) -> tuple[int, int, int, int]:
    """An ml_ kernel's tile, blocks per SM and warp-tile rows, from its name (ml_ct_128x64_2: 128 x 64, 2 per SM; _w16: warp
    tiles of 16 x 32, else 64 x 32); 128 x 128, 1 otherwise."""
    name = name.partition("@")[0]
    t = re.search(r"_(\d+)x(\d+)(?:_(\d+))?$", name)
    wm = 16 if "_w16" in name else 64
    return (int(t.group(1)), int(t.group(2)), int(t.group(3) or 1), wm) if t else (128, 128, 1, wm)


class LtCandidate(C.Structure):
    _fields_ = [("algo", C.c_uint64 * 8), ("workspace", C.c_size_t), ("waves", C.c_float), ("algo_id", i32), ("tile", i32),
                ("stages", i32), ("splitk", i32), ("reduction", i32), ("swizzle", i32), ("custom", i32),
                ("inner_shape", i32), ("cluster_shape", i32)]


class Lt:
    """cuBLASLt's plain FP8 GEMM (E4M3, tensor scales 1, BF16 out) through the harness's lt.cu built alone (-DPH_LT_ONLY):
    the heuristic's first CANDS candidates, each timed briefly, the fastest kept as the reference `lt_fp8`."""

    def __init__(self, dev: R.Dev, path: str, m: int, n: int, k: int, a: int, b: int):
        L = self.lib = C.CDLL(path)
        L.ph_last_error.restype = C.c_char_p
        L.ph_lt_plan_create.argtypes = [i32, i32, i32, i32, C.c_size_t, i32, P(vp)]
        L.ph_lt_candidates.argtypes = [vp, vp, vp, i32, P(LtCandidate), P(i32)]
        L.ph_lt_run.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp]
        L.ph_lt_version.restype = C.c_ulonglong
        assert C.sizeof(LtCandidate) == L.ph_lt_candidate_size()
        self.dev, self.a, self.b = dev, a, b
        self.ws_bytes = 32 << 20
        self.ws, self.d = dev.alloc(self.ws_bytes), dev.alloc(m * n * 2)
        self.sa, self.sb = dev.put(struct.pack("<f", 1.0)), dev.put(struct.pack("<f", 1.0))
        self.plan = vp()
        self.ck(L.ph_lt_plan_create(1, m, n, k, self.ws_bytes, 0, C.byref(self.plan)))
        want = int(os.environ.get("LT_CANDS", "8"))
        self.cands, got = (LtCandidate * want)(), i32()
        self.ck(L.ph_lt_candidates(self.plan, vp(self.sa), vp(self.sb), want, self.cands, C.byref(got)))
        self.n = got.value
        self.best, self.tried = None, []
        for j in range(self.n):
            c = self.cands[j]
            if c.workspace > self.ws_bytes:
                continue
            t = statistics.median(dev.timed(lambda j=j: self.run(j), 5))
            self.tried.append({"i": j, "ms": round(t, 4), **{f: getattr(c, f) for f in ("algo_id", "tile", "stages", "splitk",
                               "swizzle", "custom", "inner_shape", "cluster_shape")}})
            if self.best is None or t < self.tried[self.best]["ms"]:
                self.best = len(self.tried) - 1
        self.version = int(L.ph_lt_version())

    def ck(self, r: int) -> None:
        if r:
            raise RuntimeError(f"cuBLASLt {r}: {self.lib.ph_last_error().decode()}")

    def run(self, j: int) -> None:
        self.ck(self.lib.ph_lt_run(self.plan, C.byref(self.cands[j]), vp(self.a), vp(self.b), vp(self.d), vp(self.sa),
                                   vp(self.sb), vp(self.ws), None))


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Bench:
    def __init__(self, dev: R.Dev):
        self.dev = dev
        dev.cu.cuTensorMapEncodeTiled.argtypes, dev.cu.cuTensorMapEncodeTiled.restype = ENCODE, i32
        self.smem: dict[str, int] = {}
        self._maps: dict[tuple, int] = {}
        self.raster = int(os.environ.get("RASTER", "8"))

    def tmap(self, ptr: int, depth: int, rows: int, box_rows: int = 128) -> bytes:
        """A CUtensorMap over a row-major (rows x depth) byte matrix: box_rows x 128-byte boxes, SWIZZLE_128B."""
        raw = C.create_string_buffer(256)
        at = (C.addressof(raw) + 63) & ~63
        dims, strides = (u64 * 2)(depth, rows), (u64 * 1)(depth)
        box, estr = (u32 * 2)(128, box_rows), (u32 * 2)(1, 1)
        self.dev.ck("cuTensorMapEncodeTiled", vp(at), UINT8, 2, vp(ptr), dims, strides, box, estr, 0, SWIZZLE_128B, L2_128B, 0)
        return C.string_at(at, 128)

    def maps(self, a: int, b: int, m: int, n: int, k: int, tm: int = 128, bn: int = 128) -> int:
        key = (a, b, m, n, k, tm, bn)
        if key not in self._maps:
            self._maps[key] = self.dev.put(self.tmap(a, k, m, tm) + self.tmap(b, k, n, bn) + bytes(256))
        return self._maps[key]

    def prep(self, name: str) -> int:
        if name not in self.smem:
            probe = self.dev.alloc(16)
            self.dev.launch(name + "_smem", 1, 1, p=(probe,))
            self.dev.sync()
            self.smem[name] = int.from_bytes(self.dev.read(probe, 4), "little")
            self.dev.free(probe)
            self.dev.ck("cuFuncSetAttribute", self.dev.fn(name), R.MAX_DYN, self.smem[name])
        return self.smem[name]

    def gemm(self, name: str, a: int, b: int, c: int, m: int, n: int, k: int, maps: int = 0, grid: int | None = None,
             y: int = 0, rs: int = 0, cs: int = 0) -> None:
        name, _, skew = name.partition("@s")   # ml_ct@s512: warp rows of the second half start 512 cycles late
        smem = self.prep(name)
        if name.startswith("ml_"):
            tm, bn, ctas, wm = ml_tile(name)
            ncw = (tm // wm) * (bn // 32)
            g = min((m // tm) * (n // bn), grid or self.dev.sms * ctas)
            p = [a, b, c] + [0] * 7 + [self.maps(a, b, m, n, k, tm, bn)]
            self.dev.launch(name, g, ncw * 32 + (128 if ncw % 4 == 0 else 32), smem, p=p, i=(m, n, k, self.raster, int(skew or 0)))
            return
        g = min((m // 128) * (n // 128), grid or self.dev.sms)
        self.dev.launch(name, g, R.NT, smem, p=(a, b, c, 0, y, 0, 0, rs, cs), i=(m, n, k, self.raster))

    def diff(self, x: int, y: int, words: int, counter: int) -> int:
        self.dev.ck("cuMemsetD8_v2", u64(counter), 0, 8)
        self.dev.launch("diff_words", R.cdiv(words, 256), 256, p=(x, y, counter), i=(words,))
        self.dev.sync()
        return int.from_bytes(self.dev.read(counter, 8), "little")


def twin_gate(bench: Bench, man: dict, base: str, kernels: list[str]) -> dict:
    dev = bench.dev
    m, n, k = man["shape"]
    load = lambda sub, f: open(os.path.join(base, sub, f["file"]), "rb").read()  # noqa: E731
    a, b = dev.put(load("", man["common"]["at"])), dev.put(load("", man["common"]["bt"]))
    maps = bench.maps(a, b, m, n, k)
    c = dev.alloc(m * n * 4)
    out = {}
    for name in kernels:
        want = load(VARIANT_OF[g_of(name)], man["variants"][VARIANT_OF[g_of(name)]]["files"]["c"])
        for grid in (None, 1, 3):
            dev.ck("cuMemsetD8_v2", u64(c), 0xAB, m * n * 4)
            bench.gemm(name, a, b, c, m, n, k, maps=maps, grid=grid)
            dev.sync()
            got = dev.read(c, len(want))
            bad = sum(got[i:i + 4] != want[i:i + 4] for i in range(0, len(want), 4))
            out[f"{name}@grid{grid or 'full'}"] = bad
    for p in (a, b, maps, c):
        dev.free(p)
    return out


def where(got: bytes, want: bytes, n: int) -> dict:
    """Where two C~ buffers differ: the first words (row, column, got, want) and the 128 x 128 tiles, warps (64 x 32) and
    m16 x n8 blocks touched."""
    first, tiles, warps, blocks = [], set(), set(), set()
    step = 1 << 16
    for off in range(0, len(want), step):
        if got[off:off + step] == want[off:off + step]:
            continue
        for i in range(off, min(off + step, len(want)), 4):
            if got[i:i + 4] != want[i:i + 4]:
                r, c = divmod(i // 4, n)
                tiles.add((r // 128, c // 128))
                warps.add((r // 128, c // 128, (r % 128) // 64, (c % 128) // 32))
                blocks.add((r // 16, c // 8))
                if len(first) < 8:
                    first.append([r, c, got[i:i + 4].hex(), want[i:i + 4].hex()])
    return {"first": first, "tiles": sorted(tiles)[:16], "n_tiles": len(tiles), "n_warps": len(warps), "n_blocks": len(blocks)}


def big_gate(bench: Bench, m: int, n: int, k: int, kernels: list[str]) -> tuple[dict, dict]:
    dev = bench.dev
    a, b = dev.alloc(m * k), dev.alloc(n * k)
    for x, rows, seed in ((a, m, 21), (b, n, 22)):
        words = rows * k // 4
        dev.launch("ml_gen_e4m3", R.cdiv(words, 256), 256, p=(x,), i=(words, seed))
    maps = bench.maps(a, b, m, n, k)
    ref = {4: dev.alloc(m * n * 4), 0: dev.alloc(m * n * 4)}
    got, counter = dev.alloc(m * n * 4), dev.alloc(8)
    bench.gemm("gemm_ct", a, b, ref[4], m, n, k)
    bench.gemm("gemm_ct_u", a, b, ref[0], m, n, k)
    dev.sync()
    out = {"refs_differ_words": bench.diff(ref[4], ref[0], m * n, counter)}
    repeats = int(os.environ.get("GATE_REPEATS", "5"))
    detail = {}
    for name in kernels:
        worst = 0
        for rep in range(repeats):
            dev.ck("cuMemsetD8_v2", u64(got), 0xAB, m * n * 4)
            bench.gemm(name, a, b, got, m, n, k, maps=maps)
            dev.sync()
            bad = bench.diff(got, ref[g_of(name)], m * n, counter)
            if bad and name not in detail:
                detail[name] = {"rep": rep, **where(dev.read(got, m * n * 4), dev.read(ref[g_of(name)], m * n * 4), n)}
            worst = max(worst, bad)
            out.setdefault(name + "@reps", []).append(bad)
        out[name] = worst
    if detail:
        out["detail"] = detail
    bufs = {"a": a, "b": b, "maps": maps, "got": got, "ref4": ref[4], "ref0": ref[0], "counter": counter}
    return out, bufs


def timing(bench: Bench, nv: R.Nvml, m: int, n: int, k: int, bufs: dict, kernels: list[str], reps: int) -> dict:
    dev = bench.dev
    y, rs, cs = dev.alloc(m * n * 2), dev.alloc(4 * m), dev.alloc(4 * n)
    bodies = {}
    lt = None
    if os.environ.get("LT_LIB"):
        lt = Lt(dev, os.environ["LT_LIB"], m, n, k, bufs["a"], bufs["b"])
        best = lt.tried[lt.best]["i"]
        bodies["lt_fp8"] = lambda: lt.run(best)  # noqa: E731
        kernels = ["lt_fp8"] + kernels
    for name in kernels:
        if name == "lt_fp8":
            continue
        if name == "gemm_fast":
            bodies[name] = lambda: bench.gemm("gemm_fast", bufs["a"], bufs["b"], 0, m, n, k, y=y, rs=rs, cs=cs)  # noqa: E731
        else:
            bodies[name] = lambda name=name: bench.gemm(name, bufs["a"], bufs["b"], bufs["got"], m, n, k, maps=bufs["maps"])  # noqa: E731
    before = nv.read()
    runs = {name: [] for name in kernels}
    for _ in range(reps):                       # round-robin, so drift in clock or power lands on every kernel alike
        for name in kernels:
            runs[name] += dev.timed(bodies[name], 1)
    after = nv.read()
    med = {nm: statistics.median(t) for nm, t in runs.items()}
    for p in (y, rs, cs):
        dev.free(p)
    return {"kind": "timing", "label": "diagnostic (one GPU, events around one launch, kernels round-robin, clocks as recorded)",
            "shape": [m, n, k], "kernels": kernels,
            "lt": None if lt is None else {"version": lt.version, "candidates": lt.n, "tried": lt.tried, "best": lt.best},
            "reps": reps, "raster": bench.raster, "ms": {nm: round(v, 4) for nm, v in med.items()},
            "tflops": {nm: round(2 * m * n * k / v / 1e9, 1) for nm, v in med.items()},
            "runs": {nm: [round(x, 4) for x in t] for nm, t in runs.items()}, "clk_before": before, "clk_after": after}


def steady(bench: Bench, nv: R.Nvml, m: int, n: int, k: int, bufs: dict, kernels: list[str], rounds: int, batch: int) -> dict:
    """Each kernel `batch` times back to back (so the power cap settles), round-robin, NVML sampled while the batch runs:
    ms per GEMM, the SM clock and power over the batch's second half, and their product (cycles per GEMM)."""
    dev = bench.dev
    dev.cu.cuEventQuery.argtypes, dev.cu.cuEventQuery.restype = [C.c_void_p], C.c_int
    bodies = {}
    if os.environ.get("LT_LIB"):
        lt = Lt(dev, os.environ["LT_LIB"], m, n, k, bufs["a"], bufs["b"])
        best = lt.tried[lt.best]["i"]
        bodies["lt_fp8"] = lambda: lt.run(best)  # noqa: E731
    for name in kernels:
        bodies[name] = lambda name=name: bench.gemm(name, bufs["a"], bufs["b"], bufs["got"], m, n, k, maps=bufs["maps"])  # noqa: E731
    rows = {nm: [] for nm in bodies}
    for _ in range(rounds):
        for nm, body in bodies.items():
            a, b = C.c_void_p(), C.c_void_p()
            dev.ck("cuEventCreate", C.byref(a), 0)
            dev.ck("cuEventCreate", C.byref(b), 0)
            dev.ck("cuEventRecord", a, None)
            for _ in range(batch):
                body()
            dev.ck("cuEventRecord", b, None)
            samples = []
            while dev.cu.cuEventQuery(b) == 600:     # CUDA_ERROR_NOT_READY
                samples.append(nv.read())
                time.sleep(0.01)
            dev.ck("cuEventSynchronize", b)
            ms = C.c_float()
            dev.ck("cuEventElapsedTime", C.byref(ms), a, b)
            dev.cu.cuEventDestroy_v2(a)
            dev.cu.cuEventDestroy_v2(b)
            late = [s for s in samples[len(samples) // 2:] if "sm_mhz" in s] or [{"sm_mhz": 0, "power_w": 0, "throttle": "?"}]
            mhz = statistics.mean(s["sm_mhz"] for s in late)
            per = ms.value / batch
            rows[nm].append({"ms": round(per, 4), "sm_mhz": round(mhz, 1), "power_w": round(statistics.mean(s["power_w"] for s in late), 1),
                             "throttle": sorted({s["throttle"] for s in late}), "samples": len(late), "mcycles": round(per * mhz / 1e3, 4)})
    summary = {nm: {key: round(statistics.median(r[key] for r in rs), 4) for key in ("ms", "sm_mhz", "power_w", "mcycles")}
               for nm, rs in rows.items()}
    return {"kind": "steady", "label": "diagnostic (one GPU, batches back to back, round-robin, NVML during the batch's second half)",
            "shape": [m, n, k], "rounds": rounds, "batch": batch, "summary": summary, "rows": rows}


def main(argv: list[str]) -> int:
    cubin_path, man_path = argv[1], argv[2]
    image = open(cubin_path, "rb").read()
    man = json.load(open(man_path))
    dev = R.Dev(image)
    expect = os.environ.get("PEARLC_EXPECT_UUID")
    refused = dev.identity(expect)
    nv = R.Nvml(dev.uuid)
    emit({"kind": "env", "device": dev.name, "uuid": dev.uuid, "expect_uuid": expect, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "count": dev.count, "cc": list(dev.cc), "sms": dev.sms, "cuda_driver_api": dev.driver, "nvml_driver": nv.driver,
          "cubin_sha256": hashlib.sha256(image).hexdigest(), "check_json_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest(),
          "clk": nv.read(), "refused": refused, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if refused:
        print("REFUSED: " + "; ".join(refused), file=sys.stderr)
        return 3
    dev.open()
    bench = Bench(dev)
    ml = os.environ.get("ML", "ml_ct ml_ct_u ml_ct_e ml_ct_u_e ml_ct_x2 ml_ct_x2e ml_ct_u_x2 ml_cts ml_cts_u").split()
    s = int(os.environ.get("SHAPE", "8192"))
    if os.environ.get("PROFILE"):
        big, bufs = big_gate(bench, s, s, s, ml)
        y, rs, cs = dev.alloc(s * s * 2), dev.alloc(4 * s), dev.alloc(4 * s)
        bench.gemm("gemm_fast", bufs["a"], bufs["b"], 0, s, s, s, y=y, rs=rs, cs=cs)
        dev.sync()
        emit({"kind": "profile_pass", "shape": [s, s, s], "words_differing": big})
        return 0
    t0 = time.time()
    twin = twin_gate(bench, man, os.path.dirname(man_path), ["gemm_ct", "gemm_ct_u"] + ml)
    emit({"kind": "twin_gate", "shape": man["shape"], "words_differing": twin, "seconds": round(time.time() - t0, 2)})
    t0 = time.time()
    big, bufs = big_gate(bench, s, s, s, ml)
    emit({"kind": "big_gate", "shape": [s, s, s], "words_differing": big, "seconds": round(time.time() - t0, 2)})
    refs_ok = all(twin[f"{r}@grid{g}"] == 0 for r in ("gemm_ct", "gemm_ct_u") for g in ("full", 1, 3)) and big["refs_differ_words"] > 0
    passed = [nm for nm in ml if refs_ok and big[nm] == 0 and all(twin[f"{nm}@grid{g}"] == 0 for g in ("full", 1, 3))]
    print(f"twin gate {man['shape']}: {twin}\nbig gate {s}^3: {big}\npassed: {passed}", file=sys.stderr)
    if os.environ.get("ONLY_CHECK"):
        return 0 if len(passed) == len(ml) else 4
    kernels = ["gemm_fast", "gemm_ct", "gemm_ct_u"] + passed
    row = timing(bench, nv, s, s, s, bufs, kernels, int(os.environ.get("REPS", "20")))
    row["gated"] = passed
    emit(row)
    print("timing (diagnostic): " + ", ".join(f"{nm} {row['ms'][nm]:.4f} ms {row['tflops'][nm]} TFLOPS" for nm in row["kernels"]),
          file=sys.stderr)
    if row["lt"]:
        print(f"cuBLASLt {row['lt']['version']}: {row['lt']['tried']}", file=sys.stderr)
    if os.environ.get("STEADY"):
        st = steady(bench, nv, s, s, s, bufs, passed, int(os.environ.get("STEADY_ROUNDS", "4")), int(os.environ["STEADY"]))
        emit(st)
        print(f"steady (diagnostic): {st['summary']}", file=sys.stderr)
    return 0 if len(passed) == len(ml) else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
