#!/usr/bin/env python3
"""NCP-FP8's checked chain on one RTX PRO 6000 Blackwell (sm_120): where a bound NCP's time goes, by arm (ncp_sm120.cu).

0. Build: ncp_sm120.cu with the node's nvcc (CUDA, default /usr/local/cuda-13.0) into OUT, refused unless ptxas reports no
   spill or stack frame and the SASS has 128 QMMA.16832.F32.E4M3.E4M3 in each prefill kernel and 32 in each decode kernel
   (the first stage and the loop body) and no local memory.  ptxas.txt, sass.json, toolchain.txt land in OUT.
1. The check: at small shapes (one 2 x 2 tile grid at prefill's tile, one 64 x 128 decode tile with 32 rows), every arm's
   C and lane digests are written whole to OUT/check/ for check.py (off the card, against verity's sm_120 atom and BLAKE3).
2. The arms, at each SHAPES entry (p: m = n = k = 8,192; d: m = 32, n = k = 8,192), each REPS times after a warm-up, the
   median: plain at K = k (this kernel's plain FP8 GEMM) and at K = 3k (the stacked MMA work alone); core (the D-3s chain,
   C = R_(T-1), nothing bound); store_r / store_ir at S = 1 (every bound word written to a ring: an unfused binding's
   traffic); hash_r at each S of S_R and hash_ir at each S of S_IR (BLAKE3 over the bound words from registers); chash at
   each S of S_C (the plain chained GEMM over K, every S-th accumulator bound); and hash_r and chash at each S of S_STAGGER
   with the bound atoms staggered across warps (`stagger`).  A sample of
   lanes (their C cells and digests) at every arm, and sha256 of the whole C and digest buffers, go to OUT/arms.json for
   check.py.  NVML's clocks before and after every arm.

Operands are stand-ins for the formed X~ and Y~ (finite E4M3 codes from gen_codes, seeds 1 and 2): forming is not timed
here.  Diagnostic timing, untimed-node rules apply: the run says which node and lease it ran under.

  python3 run.py OUT   (env: CUDA, SHAPES="p d", REPS=5, S_R="1 2 4 8 16 32 64 768", S_IR="1 4 16", S_C="1 2 4 8 16 32",
                       S_STAGGER="4 8", NCP_LIBCUDA)
"""

from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "ncp_sm120.cu")
FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-fmad=false", "-Xptxas", "-v", "-cubin"]
ARMS = ("plain", "core", "store_r", "store_ir", "hash_r", "hash_ir", "chash")
QMMA = {"p": 128, "d": 32}
NEVER = 1 << 30            # an S no atom reaches: core's guard and nothing bound
RASTER = 8
SEED_X, SEED_Y = 1, 2
SAMPLE_LANES = (0, 101, 255)

_spec = importlib.util.spec_from_file_location("pearl_c_sm120_run", os.path.join(HERE, "..", "pearl_c_sm120", "run.py"))
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def hashed(arm: str) -> bool:
    return arm in ("hash_r", "hash_ir", "chash")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def build(out: str) -> bytes:
    cuda = os.environ.get("CUDA", "/usr/local/cuda-13.0")
    nvcc = os.path.join(cuda, "bin", "nvcc")
    cubin = os.path.join(out, "ncp_sm120.cubin")
    r = subprocess.run([nvcc, *FLAGS, "-o", cubin, KERNEL], capture_output=True, text=True)
    with open(os.path.join(out, "ptxas.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"nvcc failed ({r.returncode}): see ptxas.txt")
    log = r.stdout + r.stderr
    bad = [ln.strip() for ln in log.splitlines() if re.search(r"[1-9]\d* bytes (stack frame|spill)", ln)]
    if bad:
        raise SystemExit(f"ptxas spilled: {bad[:4]}")
    env = dict(os.environ, PATH=os.path.join(cuda, "bin") + os.pathsep + os.environ.get("PATH", ""))
    sass = subprocess.run([os.path.join(cuda, "bin", "cuobjdump"), "-sass", cubin], capture_output=True, text=True, env=env)
    if sass.returncode:
        raise SystemExit(f"cuobjdump failed: {sass.stderr[:300]}")
    parts = re.split(r"\n\s*Function : (\S+)\n", sass.stdout)
    counts = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        mma = re.findall(r"\b[A-Z]MMA\.[0-9A-Z.]+", body)
        counts[name] = {"qmma": sum(m.startswith("QMMA.16832.F32.E4M3.E4M3") for m in mma), "other_mma": len(mma) -
                        sum(m.startswith("QMMA.16832.F32.E4M3.E4M3") for m in mma),
                        "local": len(re.findall(r"\b(LDL|STL)\b", body))}
    for name, c in counts.items():
        want = QMMA.get(name[0], 0) if name[:2] in ("p_", "d_") and not name.endswith("_cfg") else 0
        if c["qmma"] != want or c["other_mma"] or c["local"]:
            raise SystemExit(f"SASS gate: {name} {c}, want {want} QMMA, no other MMA, no local memory")
    with open(os.path.join(out, "sass.json"), "w") as f:
        json.dump(counts, f, indent=1, sort_keys=True)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    with open(os.path.join(out, "toolchain.txt"), "w") as f:
        f.write(ver + "nvcc " + " ".join(FLAGS) + " -o ncp_sm120.cubin ncp_sm120.cu\n")
    with open(cubin, "rb") as f:
        return f.read()


class Ncp:
    """One shape's buffers on the device: X (M x K), Y (N x K), C, digests, ring, maps; `run(arm, S)` launches one arm."""

    def __init__(self, d: pc.Dev, kind: str, m: int, n: int, k: int):
        self.d, self.kind = d, kind
        self.cfg = {}
        for name in self.kernels():
            out = d.alloc(32)
            arg = C.c_uint64(out)
            d.ck("cuLaunchKernel", d.fn(name + "_cfg"), 1, 1, 1, 1, 1, 1, 0, None, (pc.vp * 1)(C.addressof(arg)), None)
            d.sync()
            raw = d.read(out, 20)
            smem, nt, nct, tm, bn = (int.from_bytes(raw[4 * j:4 * j + 4], "little") for j in range(5))
            d.free(out)
            d.ck("cuFuncSetAttribute", d.fn(name), pc.MAX_DYN, smem)
            self.cfg[name] = (smem, nt, nct, tm, bn)
        smem, self.nt, self.nct, self.tm, self.bn = self.cfg[self.kernels()[0]]
        self.rows, self.n, self.k = m, n, k
        self.m = -(-m // self.tm) * self.tm
        self.K = 3 * k
        self.tiles = (self.m // self.tm) * (n // self.bn)
        self.grid = min(self.tiles, d.sms)
        self.x = d.alloc(self.m * self.K)
        self.y = d.alloc(n * self.K)
        self.gen(self.x, m * self.K, SEED_X)
        self.gen(self.y, n * self.K, SEED_Y)
        self.c = d.alloc(self.m * n * 4)
        self.dg = d.alloc(self.tiles * self.nct * 32)
        self.ring = d.alloc(self.grid << (22 + 2))
        self.maps = {}

    def kernels(self) -> list[str]:
        return [f"{self.kind}_{a}" for a in ARMS]

    def gen(self, ptr: int, count: int, seed: int) -> None:
        self.d.launch("gen_codes", 4 * self.d.sms, 256, 0, p=(ptr,), i=(count & 0xFFFFFFFF, seed, count >> 32))

    def tmaps(self, K: int) -> int:
        """The maps over the operands' first K columns (rows of 3k bytes): plain at K = k reads X~ and Y~'s first block."""
        if K not in self.maps:
            raw = tmap(self.d, self.x, K, self.m, self.K, self.tm) + tmap(self.d, self.y, K, self.n, self.K, self.bn)
            self.maps[K] = self.d.put(raw + raw)
        return self.maps[K]

    def launch(self, arm: str, S: int, K: int, stagger: int = 0) -> None:
        name = f"{self.kind}_{arm}"
        self.d.launch(name, self.grid, self.nt, self.cfg[name][0], p=(self.tmaps(K), self.c, self.dg, self.ring),
                      i=(self.m, self.n, K, RASTER, 0, S, self.rows, stagger))

    def lanes(self, tiles: list[int], digests: bool) -> list[dict]:
        """The sampled lanes: tile coordinates, their C cells in fragment order (mi, ni, e) as FP32 bit patterns (hex, big
        endian), and their digests (32 bytes, the eight words little endian)."""
        out, tiles_n, lay = [], self.n // self.bn, layout(self.kind)
        for t in tiles:
            tm, tn = t // tiles_n, t % tiles_n
            for tid in SAMPLE_LANES:
                warp, lane = tid // 32, tid % 32
                wm, wn, g, q = warp // lay["warps_n"], warp % lay["warps_n"], lane >> 2, lane & 3
                cells = []
                for mi in range(lay["mi"]):
                    for ni in range(lay["ni"]):
                        for e in range(4):
                            row = tm * self.tm + wm * lay["wm"] + 16 * mi + g + 8 * (e >> 1)
                            col = tn * self.bn + wn * 32 + 8 * ni + 2 * q + (e & 1)
                            cells.append(self.d.read(self.c + (row * self.n + col) * 4, 4)[::-1].hex())
                ln = {"tile": t, "tm": tm, "tn": tn, "tid": tid, "cells": cells}
                if digests:
                    ln["digest"] = self.d.read(self.dg + (t * self.nct + tid) * 32, 32).hex()
                out.append(ln)
        return out


def tmap(d: pc.Dev, ptr: int, depth: int, rows: int, pitch: int, box_rows: int) -> bytes:
    """pearl_c_sm120's map (UINT8, rank 2, 128-byte x box_rows boxes, SWIZZLE_128B, L2 128B) over the first `depth` bytes of
    rows `pitch` bytes apart."""
    f = d.cu.cuTensorMapEncodeTiled
    f.argtypes, f.restype = [pc.vp, pc.i32, pc.u, pc.vp, pc.P(pc.u64), pc.P(pc.u64), pc.P(pc.u), pc.P(pc.u), pc.i32, pc.i32,
                             pc.i32, pc.i32], pc.i32
    raw = C.create_string_buffer(pc.TMAP_BYTES + 64)
    at = (C.addressof(raw) + 63) & ~63
    dims, strides = (pc.u64 * 2)(depth, rows), (pc.u64 * 1)(pitch)
    box, estr = (pc.u * 2)(pc.TMAP_BYTES, box_rows), (pc.u * 2)(1, 1)
    d.ck("cuTensorMapEncodeTiled", pc.vp(at), pc.TMAP_UINT8, 2, pc.vp(ptr), dims, strides, box, estr, 0,
         pc.TMAP_SWIZZLE_128B, pc.TMAP_L2_128B, 0)
    return C.string_at(at, pc.TMAP_BYTES)


def layout(kind: str) -> dict:
    return {"p": {"mi": 4, "ni": 4, "wm": 64, "warps_n": 4}, "d": {"mi": 1, "ni": 4, "wm": 16, "warps_n": 2}}[kind]


SHAPES = {"p": (8192, 8192, 8192), "d": (32, 8192, 8192)}
CHECK = {"p": (256, 256, 256), "d": (32, 128, 256)}


def arm_list(kind: str, k: int) -> list[tuple[str, int, int, int]]:
    """(arm, S, K, stagger) in the order run."""
    K, T = 3 * k, 3 * k // 32
    env = lambda key, default: [int(s) for s in os.environ.get(key, default).split()]
    arms = [("plain", 0, k, 0), ("plain", 0, K, 0), ("core", NEVER, K, 0), ("store_r", 1, K, 0), ("store_ir", 1, K, 0)]
    arms += [("hash_r", s, K, 0) for s in env("S_R", f"1 2 4 8 16 32 64 {T}")]
    arms += [("hash_ir", s, K, 0) for s in env("S_IR", "1 4 16")]
    arms += [("chash", s, K, 0) for s in env("S_C", "1 2 4 8 16 32")]
    arms += [(a, s, K, 1) for s in env("S_STAGGER", "4 8") for a in ("hash_r", "chash")]
    return arms


def check(d: pc.Dev, out: str) -> None:
    os.makedirs(os.path.join(out, "check"), exist_ok=True)
    for kind, (m, n, k) in CHECK.items():
        x = Ncp(d, kind, m, n, k)
        T = x.K // 32
        runs = [("plain", 0, k, 0), ("plain", 0, x.K, 0), ("core", NEVER, x.K, 0), ("store_r", 1, x.K, 0),
                ("store_ir", 1, x.K, 0)]
        runs += [("hash_r", s, x.K, 0) for s in (1, 3, 4, T)] + [("hash_ir", s, x.K, 0) for s in (1, 3, T)]
        runs += [("chash", s, x.K, 0) for s in (1, 3, T)]
        runs += [("hash_r", 3, x.K, 1), ("hash_ir", 3, x.K, 1), ("chash", 3, x.K, 1)]
        meta = {"kind": kind, "m": x.m, "rows": m, "n": n, "k": k, "K": x.K, "tm": x.tm, "bn": x.bn, "nct": x.nct,
                "tiles": x.tiles, "grid": x.grid, "seeds": [SEED_X, SEED_Y], "layout": layout(kind), "runs": []}
        for arm, S, K, stagger in runs:
            d.ck("cuMemsetD8_v2", pc.u64(x.c), 0, x.m * n * 4)
            d.ck("cuMemsetD8_v2", pc.u64(x.dg), 0, x.tiles * x.nct * 32)
            x.launch(arm, S, K, stagger)
            d.sync()
            tag = f"{kind}_{arm}_S{S}_K{K}" + ("_stagger" if stagger else "")
            with open(os.path.join(out, "check", tag + ".c.bin"), "wb") as f:
                f.write(d.read(x.c, x.m * n * 4))
            if hashed(arm):
                with open(os.path.join(out, "check", tag + ".dg.bin"), "wb") as f:
                    f.write(d.read(x.dg, x.tiles * x.nct * 32))
            meta["runs"].append({"arm": arm, "S": S, "K": K, "stagger": stagger, "tag": tag})
        with open(os.path.join(out, "check", f"{kind}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        emit({"part": "check", "kind": kind, "runs": len(runs)})


def arms(d: pc.Dev, nv, out: str, kind: str, reps: int) -> dict:
    m, n, k = SHAPES[kind]
    x = Ncp(d, kind, m, n, k)
    tiles = sorted({0, x.tiles // 2 + 1, x.tiles - 1})
    res = {"kind": kind, "m": x.m, "rows": m, "n": n, "k": k, "K": x.K, "T": x.K // 32, "tm": x.tm, "bn": x.bn, "nct": x.nct,
           "tiles": x.tiles, "grid": x.grid, "seeds": [SEED_X, SEED_Y], "layout": layout(kind), "arms": []}
    for arm, S, K, stagger in arm_list(kind, k):
        before = nv.read()
        times = d.timed(lambda: x.launch(arm, S, K, stagger), reps)
        after = nv.read()
        row = {"arm": arm, "S": S, "K": K, "stagger": stagger, "ms": statistics.median(times), "reps": times,
               "nvml_before": before, "nvml_after": after}
        row["c_sha256"] = hashlib.sha256(d.read(x.c, x.m * n * 4)).hexdigest()
        if hashed(arm):
            row["dg_sha256"] = hashlib.sha256(d.read(x.dg, x.tiles * x.nct * 32)).hexdigest()
        row["lanes"] = x.lanes(tiles, hashed(arm))
        res["arms"].append(row)
        emit({"part": "arm", "kind": kind, "arm": arm, "S": S, "K": K, "stagger": stagger, "ms": round(row["ms"], 4),
              "sm_mhz": [before.get("sm_mhz"), after.get("sm_mhz")]})
    return res


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out")
    os.makedirs(out, exist_ok=True)
    cubin = build(out)
    if os.environ.get("NCP_LIBCUDA"):
        os.environ["PEARLC_LIBCUDA"] = os.environ["NCP_LIBCUDA"]
    d = pc.Dev(cubin)
    ident = {"name": d.name, "uuid": d.uuid, "cc": d.cc, "sms": d.sms, "driver": d.driver, "count": d.count,
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "cubin_sha256": hashlib.sha256(cubin).hexdigest()}
    emit({"part": "identity", **ident})
    if d.cc != (12, 0) or not d.name.startswith(pc.NAME_PREFIX) or d.count != 1:
        emit({"part": "refused", "why": "not exactly one RTX PRO 6000 Blackwell (sm_120) visible"})
        return 3
    d.open()
    nv = pc.Nvml(d.uuid)
    ident["nvml_driver"] = nv.driver
    check(d, out)
    reps = int(os.environ.get("REPS", "5"))
    result = {"identity": ident, "shapes": []}
    for kind in os.environ.get("SHAPES", "p d").split():
        result["shapes"].append(arms(d, nv, out, kind, reps))
        with open(os.path.join(out, "arms.json"), "w") as f:
            json.dump(result, f, indent=1)
    emit({"part": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
