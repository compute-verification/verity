"""W1 price microbenchmarks for the FP4 path on sm_120a (lane pouw rtx-pro GPU 4): what one MMA of each row costs on this card.

    python tools/tc_probe_fp4/w1_price.py --out $RESEARCH_RUN_DIR --expect-uuid GPU-... [--rows nvf4,mxf4,...] [--reps 10]
    python tools/tc_probe_fp4/w1_price.py --out /tmp/w1 --sass-only          # build + SASS gate, no device
    python tools/tc_probe_fp4/w1_price.py --out /tmp/w1 --fake-silicon       # CPU dry run of the driver (never under a research run)

Each row is one ``mma.sync`` of ``rows_fp4.cuh`` (the mnemonic the probe's capture gates), timed by ``w1_price.cu``:

  throughput  NSETS x NACC register-resident MMAs per loop iteration from every warp of every SM (``warps_per_sm`` warps each);
              for the scaled rows a second variant rotates every scale register a byte per iteration (``scale_update``), so
              the scale values change as a GEMM's do
  latency     one warp, one accumulator, a dependent chain (cycles per MMA, from clock64)

Every launch's SM clock is its thread-0 clock64 span over its event time, so each rate is also stated per SM per clock, which
does not depend on the clock.  The rows:

  nvf4                 kind::mxf4nvf4 4X ue4m3 m16n8k64      OMMA  the honest dense NVFP4 rate (GPU 7's fp4_attack ``dense`` too)
  mxf4                 kind::mxf4nvf4 2X ue8m0 m16n8k64      OMMA
  f8f6f4               kind::f8f6f4 e2m1 x e2m1 m16n8k32     QMMA  unscaled E2M1 in 8-bit containers
  f8f6f4_e2m1_e4m3     kind::f8f6f4 e2m1 x e4m3 m16n8k32     QMMA
  mxf8f6f4_e2m1        kind::mxf8f6f4 1X ue8m0 e2m1 x e2m1   QMMA
  mxf8f6f4_e2m1_e4m3   kind::mxf8f6f4 1X ue8m0 e2m1 x e4m3   QMMA
  f8f6f4_e4m3          kind::f8f6f4 e4m3 x e4m3 m16n8k32     QMMA  the FP8 reference (the FP8 stream owns its capture)

Gates, all before any timing (a failed gate writes no measurements): the device UUID (``--expect-uuid``) and compute capability
12.0; cuobjdump counts exactly NSETS x NACC of the row's opcode in each throughput kernel and LAT_UNROLL in each latency kernel;
every kernel's outputs are finite and bit-identical across two launches; every launch is one wave (all its blocks resident, so
thread 0's clock64 span is the launch's).  ``--corrupt-word`` flips one output word of the second
gate launch: the run must then fail.  Nothing here checks the arithmetic; ``probe.py`` does.

Clocks: each rep samples ``nvidia-smi -i <uuid>`` (SM and memory clock, power, temperature, throttle reasons).  The run is
labelled ``locked at F MHz`` only with ``--locked-mhz F`` and every sample at F with every launch's SM clock within 1% of it,
else ``unlocked``; a container can't lock clocks, so RunPod runs are ``unlocked``.

Honest notes.  Operands are uniform random codes (every E2M1 code, finite E4M3 bytes, UE4M3 scales 0x30..0x47, UE8M0 124..131),
a GEMM's data rather than zeros; an attacker's zeros could draw less power, which is not measured here.  The K=32 rows take
E2M1 in 8-bit containers, so a GEMM that stores packed nibbles pays an unpack this instruction-level rate leaves out.  The
throughput is an upper bound on any kernel built from the instruction, not a GEMM rate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402

SOURCE = "w1_price.cu"
ARCH = "sm_120a"
SCHEMA = "pouw-rtxpro/w1-price-fp4/v1"
NSETS, NACC, WORDS, LAT_UNROLL = 4, 8, 8, 8
BODY = NSETS * NACC

#: row -> (RowKind in rows_fp4.cuh, K, A format, B format, scale format or None, SASS opcode)
ROWS = {
    "nvf4": (0, 64, "e2m1", "e2m1", "ue4m3", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"),
    "mxf4": (1, 64, "e2m1", "e2m1", "ue8m0", "OMMA.SF.16864.F32.E2M1.E2M1.E8"),
    "f8f6f4": (2, 32, "e2m1", "e2m1", None, "QMMA.16832.F32.E2M1.E2M1"),
    "f8f6f4_e2m1_e4m3": (3, 32, "e2m1", "e4m3", None, "QMMA.16832.F32.E2M1.E4M3"),
    "mxf8f6f4_e2m1": (5, 32, "e2m1", "e2m1", "ue8m0", "QMMA.SF.16832.F32.E2M1.E2M1.E8"),
    "mxf8f6f4_e2m1_e4m3": (6, 32, "e2m1", "e4m3", "ue8m0", "QMMA.SF.16832.F32.E2M1.E4M3.E8"),
    "f8f6f4_e4m3": (8, 32, "e4m3", "e4m3", None, "QMMA.16832.F32.E4M3.E4M3"),
}
PTX = {
    "nvf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
    "mxf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
    "f8f6f4": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32",
    "f8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32",
    "mxf8f6f4_e2m1": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0",
    "mxf8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0",
    "f8f6f4_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32",
}
MODES = {0: "const_scales", 1: "scale_update"}
#: the named comparisons (numerator row/mode, denominator row/mode), each a ratio of logical flop rates
COMPARISONS = {
    "mxf4_over_nvf4": (("mxf4", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_over_nvf4": (("f8f6f4", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_e4m3_over_nvf4": (("f8f6f4_e2m1_e4m3", 0), ("nvf4", 0)),
    "mxf8f6f4_e2m1_over_nvf4": (("mxf8f6f4_e2m1", 0), ("nvf4", 0)),
    "f8f6f4_e2m1_over_f8f6f4_e4m3": (("f8f6f4", 0), ("f8f6f4_e4m3", 0)),
    "mxf8f6f4_e2m1_over_f8f6f4_e2m1": (("mxf8f6f4_e2m1", 0), ("f8f6f4", 0)),
    "nvf4_scale_update_over_const": (("nvf4", 1), ("nvf4", 0)),
    "mxf4_scale_update_over_const": (("mxf4", 1), ("mxf4", 0)),
    "mxf8f6f4_e2m1_scale_update_over_const": (("mxf8f6f4_e2m1", 1), ("mxf8f6f4_e2m1", 0)),
    "mxf8f6f4_e2m1_e4m3_scale_update_over_const": (("mxf8f6f4_e2m1_e4m3", 1), ("mxf8f6f4_e2m1_e4m3", 0)),
}
_E4M3_FINITE = np.array([c for c in range(256) if c & 0x7F != 0x7F], np.uint32)


def flop_per_mma(row: str) -> int:
    return 2 * 16 * 8 * ROWS[row][1]


def _bytes(rng, fmt: str, shape) -> np.ndarray:
    if fmt == "e2m1":  # a code at bits [2,6) of its 8-bit container
        return rng.integers(0, 16, size=shape, dtype=np.uint32) << 2
    if fmt == "e4m3":
        return _E4M3_FINITE[rng.integers(0, len(_E4M3_FINITE), size=shape)]
    if fmt == "ue4m3":  # 2^-3 .. 2^4 x mantissa: accumulators stay finite over any run length used here
        return rng.integers(0x30, 0x48, size=shape, dtype=np.uint32)
    if fmt == "ue8m0":
        return rng.integers(124, 132, size=shape, dtype=np.uint32)
    raise ValueError(fmt)


def _pack(b: np.ndarray) -> np.ndarray:
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def fragments(row: str, nwarps: int, seed: int) -> np.ndarray:
    """u32[nwarps*32][NSETS][WORDS]: a[4] b[2] sfa sfb per operand set per lane, uniform random codes of the row's formats."""
    kind, k, fa, fb, sf, _ = ROWS[row]
    rng = np.random.default_rng([seed, kind])
    lanes = nwarps * 32
    f = np.zeros((lanes, NSETS, WORDS), np.uint32)
    if k == 64:  # E2M1 nibbles, every code of both signs
        f[:, :, :6] = rng.integers(0, 2**32, size=(lanes, NSETS, 6), dtype=np.uint64).astype(np.uint32)
    else:
        f[:, :, 0:4] = _pack(_bytes(rng, fa, (lanes, NSETS, 4, 4)))
        f[:, :, 4:6] = _pack(_bytes(rng, fb, (lanes, NSETS, 2, 4)))
    if sf:
        f[:, :, 6:8] = _pack(_bytes(rng, sf, (lanes, NSETS, 2, 4)))
    return f


def variants(rows: list[str], scale_update: bool) -> list[tuple[str, int]]:
    return [(r, m) for r in rows for m in ((0, 1) if (scale_update and ROWS[r][4]) else (0,))]


def kernel_name(row: str, mode: int | None) -> str:
    kind = ROWS[row][0]
    return f"price_latency_kernelILi{kind}EE" if mode is None else f"price_tput_kernelILi{kind}ELi{mode}EE"


def sass_gate(so: Path, rows: list[str], scale_update: bool) -> dict:
    per_fn = P.sass_check(so)
    out, ok = {}, True
    for row, mode in variants(rows, scale_update) + [(r, None) for r in rows]:
        name = kernel_name(row, mode)
        recs = [v for fn, v in per_fn.items() if name in fn]
        want = BODY if mode is not None else LAT_UNROLL
        got = recs[0] if len(recs) == 1 else {"mma_count": 0, "opcodes": []}
        good = len(recs) == 1 and got["mma_count"] == want and set(got["opcodes"]) == {ROWS[row][5]}
        ok &= good
        out[name] = {"mma_count": got["mma_count"], "expected": want, "opcodes": sorted(set(got["opcodes"])),
                     "expected_opcode": ROWS[row][5], "ok": good}
    return {"ok": ok, "kernels": out}


class W1Device:
    def __init__(self, so: Path):
        self.lib = lib = ctypes.CDLL(str(so))
        V, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.w1_device.argtypes = [ctypes.POINTER(I)] * 4
        lib.w1_setup.argtypes = [V, I, I, I]
        lib.w1_run.argtypes = [I, I, I, I, I, L, ctypes.POINTER(ctypes.c_float), V, V]
        lib.w1_latency.argtypes = [I, I, L, V, V]
        lib.w1_occupancy.argtypes = [I, I, I, ctypes.POINTER(I)]
        lib.device_uuid.restype = I
        lib.device_uuid.argtypes = [ctypes.c_char_p, I]
        lib.has_mxf4.restype = I
        lib.device_identity.restype = I
        lib.device_identity.argtypes = [ctypes.POINTER(I), ctypes.POINTER(I), ctypes.c_char_p, I]
        self.tcgen05 = False

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def device(self) -> dict:
        v = [ctypes.c_int() for _ in range(4)]
        rc = self.lib.w1_device(*v)
        if rc:
            raise RuntimeError(f"w1_device rc={rc}")
        return {"sms": v[0].value, "max_threads_per_sm": v[1].value, "cc": f"{v[2].value}.{v[3].value}"}

    def occupancy(self, kind: int, mode: int, wpb: int) -> int:
        n = ctypes.c_int()
        rc = self.lib.w1_occupancy(kind, mode, wpb, ctypes.byref(n))
        if rc:
            raise RuntimeError(f"w1_occupancy(kind={kind}, mode={mode}) rc={rc}")
        return n.value

    def setup(self, frags: np.ndarray, nwarps: int, nslots: int, nblocks: int) -> None:
        self._frags = np.ascontiguousarray(frags, np.uint32)
        rc = self.lib.w1_setup(self._frags.ctypes.data, nwarps, nslots, nblocks)
        if rc:
            raise RuntimeError(f"w1_setup rc={rc}")

    def run(self, kind: int, mode: int, slot: int, nblocks: int, wpb: int, iters: int, want: bool = False):
        ms = ctypes.c_float()
        sink = np.zeros(nblocks * wpb * 32 * 4, np.float32) if want else None
        cyc = np.zeros(nblocks, np.int64)
        rc = self.lib.w1_run(kind, mode, slot, nblocks, wpb, iters, ctypes.byref(ms), sink.ctypes.data if want else None, cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"w1_run(kind={kind}, mode={mode}) rc={rc}")
        return ms.value, sink, cyc

    def latency(self, kind: int, slot: int, iters: int):
        sink = np.zeros(32 * 4, np.float32)
        cyc = np.zeros(1, np.int64)
        rc = self.lib.w1_latency(kind, slot, iters, sink.ctypes.data, cyc.ctypes.data)
        if rc:
            raise RuntimeError(f"w1_latency(kind={kind}) rc={rc}")
        return sink, int(cyc[0])

    def teardown(self) -> None:
        self.lib.w1_teardown()


class FakeW1:
    """CPU stand-in for the driver's dry run: a 188-SM part at 2,000 MHz issuing one MMA per SM per 8 clocks (K=64) or 4 (K=32).
    Its numbers are invented; ``--fake-silicon`` never runs under a research run."""

    tcgen05 = False
    SMS, MHZ = 188, 2000.0

    def identity(self) -> dict:
        return {"device": "FAKE SILICON", "compute_capability": "12.0", "driver": None, "gpu_uuid": "GPU-fake", "backend": "fake",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def clocks(self, uuid):
        return None

    def device(self) -> dict:
        return {"sms": self.SMS, "max_threads_per_sm": 1536, "cc": "12.0"}

    def occupancy(self, kind, mode, wpb) -> int:
        return 4

    def setup(self, frags, nwarps, nslots, nblocks) -> None:
        self._frags = frags

    def _cpm(self, kind: int) -> int:
        return 8 if kind in (0, 1) else 4

    def run(self, kind, mode, slot, nblocks, wpb, iters, want=False):
        warps_per_sm = nblocks * wpb / self.SMS
        cycles = iters * BODY * warps_per_sm * self._cpm(kind) * (1.01 if mode else 1.0)
        ms = cycles / (self.MHZ * 1e3)
        sink = np.full(nblocks * wpb * 32 * 4, float(kind + slot), np.float32) if want else None
        return ms, sink, np.full(nblocks, int(cycles), np.int64)

    def latency(self, kind, slot, iters):
        return np.zeros(128, np.float32), iters * LAT_UNROLL * (20 if kind in (0, 1) else 16)

    def teardown(self) -> None:
        pass


def _median(xs):
    return statistics.median(xs) if xs else None


def clock_label(samples: list[dict | None], launch_mhz: list[float], locked_mhz: float | None) -> str:
    if locked_mhz is None or not samples or any(s is None for s in samples):
        return "unlocked"
    try:
        at = all(float(s["clocks.sm"]) == float(locked_mhz) for s in samples)
    except (KeyError, TypeError, ValueError):
        return "unlocked"
    near = all(abs(m - locked_mhz) <= 0.01 * locked_mhz for m in launch_mhz)
    return f"locked at {locked_mhz:g} MHz" if (at and near) else "unlocked"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ROWS), help="comma list of " + ", ".join(ROWS))
    ap.add_argument("--reps", type=int, default=10, help="timed launches per variant, interleaved round-robin")
    ap.add_argument("--target-ms", type=float, default=100.0, help="calibrated length of one timed launch")
    ap.add_argument("--sustain-s", type=float, default=3.0, help="back-to-back nvf4 launches before the first rep")
    ap.add_argument("--warps-per-sm", type=int, default=16)
    ap.add_argument("--latency-iters", type=int, default=20000, help="dependent chain length / LAT_UNROLL")
    ap.add_argument("--no-scale-update", action="store_true")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=900.0, help="refuse a plan longer than this (quiet windows are 20 min)")
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--locked-mhz", type=float, default=None, help="the clock the GPU was locked at outside this run, if it was")
    ap.add_argument("--corrupt-word", action="store_true", help="gate self-test: one flipped output word must fail the run")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--sass-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in args.rows.split(",") if r]
    bad = [r for r in rows if r not in ROWS]
    if bad:
        ap.error(f"unknown rows {bad}")
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        ap.error("--fake-silicon is a dry run and never publishes: not under a research run")
    scale_update = not args.no_scale_update
    plan = variants(rows, scale_update)
    planned_s = args.sustain_s + len(plan) * (args.reps + 3) * args.target_ms / 1e3 + args.reps * 0.2 * 2
    if planned_s > args.max_seconds:
        ap.error(f"the plan takes about {planned_s:.0f} s > --max-seconds {args.max_seconds:g}")
    t_all = time.time()

    have_nvcc = Path(P.nvcc_bin()).is_file()
    so, nvcc = (P.build(ARCH, out, SOURCE) if (have_nvcc or not args.fake_silicon) else (None, "not built (fake silicon)"))
    sass = sass_gate(so, rows, scale_update) if so is not None else {"ok": None, "kernels": {}, "note": "no nvcc"}
    dev = FakeW1() if args.fake_silicon else (None if args.sass_only else W1Device(so))
    identity = dev.identity() if dev is not None else {"device": None, "gpu_uuid": None, "backend": "none (--sass-only)"}
    identity["nvcc"] = nvcc
    record: dict = {"schema": SCHEMA, "identity": identity, "rows": rows, "plan": [f"{r}/{MODES[m]}" for r, m in plan],
                    "sass": sass, "fake_silicon": args.fake_silicon, "corrupt_word": args.corrupt_word, "seed": args.seed}
    failures: list[str] = []
    if sass["ok"] is False:
        failures.append("SASS gate: " + ", ".join(k for k, v in sass["kernels"].items() if not v["ok"]))
    if args.expect_uuid and dev is not None and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
    if dev is None or failures:
        return finish(out, record, failures, args, None, t_all)

    d = dev.device()
    record["device"] = d
    if d["cc"] != "12.0":
        failures.append(f"compute capability {d['cc']} != 12.0")
        return finish(out, record, failures, args, None, t_all)
    wpb = 4
    nblocks = d["sms"] * max(1, args.warps_per_sm // wpb)
    nwarps = nblocks * wpb
    grid = {"sms": d["sms"], "blocks": nblocks, "warps_per_block": wpb, "warps": nwarps, "warps_per_sm": nblocks * wpb // d["sms"],
            "operand_sets": NSETS, "accumulator_chains": NACC, "mmas_per_iteration": BODY, "latency_unroll": LAT_UNROLL}
    record["grid"] = grid
    slot = {r: i for i, r in enumerate(rows)}
    frags = np.stack([fragments(r, nwarps, args.seed) for r in rows])
    record["operands_sha256"] = hashlib.sha256(frags.tobytes()).hexdigest()
    dev.setup(frags, nwarps, len(rows), nblocks)

    gate: dict = {}
    for row, mode in plan + [(r, None) for r in rows]:
        kind = ROWS[row][0]
        if mode is None:
            s1, _ = dev.latency(kind, slot[row], 64)
            s2, _ = dev.latency(kind, slot[row], 64)
        else:
            _, s1, _ = dev.run(kind, mode, slot[row], nblocks, wpb, 64, want=True)
            _, s2, _ = dev.run(kind, mode, slot[row], nblocks, wpb, 64, want=True)
        if args.corrupt_word and not gate:
            s2 = s2.copy()
            s2.view(np.uint32)[0] ^= 1
        finite = bool(np.isfinite(s1).all() and np.isfinite(s2).all())
        same = bool(np.array_equal(s1.view(np.uint32), s2.view(np.uint32)))
        gate[kernel_name(row, mode)] = {"finite": finite, "bit_identical": same, "ok": finite and same}
    record["output_gate"] = gate
    occ = {f"{r}/{MODES[m]}": dev.occupancy(ROWS[r][0], m, wpb) for r, m in plan}
    record["occupancy_blocks_per_sm"] = occ
    failures += [f"occupancy {k}: {v} blocks/SM < {nblocks // d['sms']} (not one wave)" for k, v in occ.items() if v < nblocks // d["sms"]]
    failures += [f"output gate {k}: finite={v['finite']} bit_identical={v['bit_identical']}" for k, v in gate.items() if not v["ok"]]
    if failures:
        dev.teardown()
        return finish(out, record, failures, args, None, t_all)

    iters = {}
    for row, mode in plan:
        ms, _, _ = dev.run(ROWS[row][0], mode, slot[row], nblocks, wpb, 2000)
        iters[(row, mode)] = max(2000, int(2000 * args.target_ms / max(ms, 1e-3)))
    t0 = time.time()
    first = plan[0]
    while time.time() - t0 < args.sustain_s:
        dev.run(ROWS[first[0]][0], first[1], slot[first[0]], nblocks, wpb, iters[first])
    launches: dict = {f"{r}/{MODES[m]}": [] for r, m in plan}
    samples = []
    for rep in range(args.reps):
        order = plan[rep % len(plan):] + plan[:rep % len(plan)]
        for row, mode in order:
            ms, _, cyc = dev.run(ROWS[row][0], mode, slot[row], nblocks, wpb, iters[(row, mode)])
            launches[f"{row}/{MODES[mode]}"].append({"rep": rep, "ms": ms, "cycles_median": float(np.median(cyc)),
                                                    "cycles_max": int(cyc.max()), "sm_mhz": float(np.median(cyc)) / (ms * 1e3)})
        samples.append(dev.clocks(identity.get("gpu_uuid")))
    latency = {}
    for row in rows:
        per = [dev.latency(ROWS[row][0], slot[row], args.latency_iters)[1] / (args.latency_iters * LAT_UNROLL) for _ in range(5)]
        latency[row] = {"cycles_per_mma": per, "cycles_per_mma_median": _median(per)}
    dev.teardown()

    per_variant = {}
    for row, mode in plan:
        key = f"{row}/{MODES[mode]}"
        ls = launches[key]
        n_mma = float(iters[(row, mode)]) * BODY * nwarps
        ms_med = _median([x["ms"] for x in ls])
        mhz_med = _median([x["sm_mhz"] for x in ls])
        mma_s = n_mma / (ms_med / 1e3)
        per_variant[key] = {"row": row, "mode": MODES[mode], "instruction": PTX[row], "sass_opcode": ROWS[row][5], "k": ROWS[row][1],
                            "iterations": iters[(row, mode)], "ms_median": ms_med, "ms_all": [x["ms"] for x in ls],
                            "mma_per_second": mma_s, "flop_per_second": mma_s * flop_per_mma(row), "sm_mhz_median": mhz_med,
                            "sm_mhz_all": [x["sm_mhz"] for x in ls],
                            "mma_per_sm_per_clock": mma_s / (d["sms"] * mhz_med * 1e6),
                            "flop_per_sm_per_clock": mma_s * flop_per_mma(row) / (d["sms"] * mhz_med * 1e6)}
    comparisons = {}
    for name, (num, den) in COMPARISONS.items():
        a, b = (per_variant.get(f"{r}/{MODES[m]}") for r, m in (num, den))
        if a and b:
            comparisons[name] = {"flop_rate_ratio": a["flop_per_second"] / b["flop_per_second"],
                                 "per_clock_ratio": a["flop_per_sm_per_clock"] / b["flop_per_sm_per_clock"]}
    nv = per_variant.get("nvf4/const_scales")
    prices = ({key: nv["flop_per_sm_per_clock"] / v["flop_per_sm_per_clock"] for key, v in per_variant.items()} if nv else {})
    record.update(per_variant=per_variant, latency=latency, comparisons=comparisons, clock_samples=samples, launches=launches,
                  price_per_mac_in_dense_nvf4_units=prices,
                  clocks=clock_label(samples, [x["sm_mhz"] for ls in launches.values() for x in ls], args.locked_mhz))
    return finish(out, record, failures, args, per_variant, t_all)


def finish(out: Path, record: dict, failures: list[str], args, per_variant: dict | None, t_all: float) -> int:
    record["failures"] = failures
    record["elapsed_s"] = time.time() - t_all
    (out / "w1_price.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    ok = not failures and (per_variant is not None or args.sass_only)
    detail = ("; ".join(failures) if failures else
              ("SASS gate only (--sass-only)" if per_variant is None else
               f"{len(record['output_gate'])} kernels finite and bit-identical across two launches; SASS {len(record['sass']['kernels'])} "
               f"kernels each exactly the row's opcode x the body"))
    meas = []
    if per_variant and not failures:
        for key, v in per_variant.items():
            stem = key.replace("/", "_")
            meas += [{"name": f"{stem}_flop_per_second", "value": v["flop_per_second"], "unit": "flop/s"},
                     {"name": f"{stem}_flop_per_sm_per_clock", "value": v["flop_per_sm_per_clock"], "unit": "flop/(SM*cycle)"},
                     {"name": f"{stem}_sm_mhz_median", "value": v["sm_mhz_median"], "unit": "MHz"}]
        for row, v in record["latency"].items():
            meas.append({"name": f"{row}_latency_cycles_per_mma", "value": v["cycles_per_mma_median"], "unit": "cycles"})
        for name, v in record["comparisons"].items():
            meas.append({"name": name, "value": v["per_clock_ratio"], "unit": "ratio"})
        meas.append({"name": "elapsed", "value": record["elapsed_s"], "unit": "s"})
    ident = record["identity"]
    doc = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": os.environ.get("RESEARCH_RUN_ID"),
           "device": ident.get("device"), "gpu_uuid": ident.get("gpu_uuid"), "driver": ident.get("driver"), "nvcc": ident.get("nvcc"),
           "cuda_visible_devices": ident.get("cuda_visible_devices"), "clocks": record.get("clocks", "unlocked"),
           "label": "Measured" + ("" if record.get("clocks", "unlocked") != "unlocked" else ", unlocked clocks"),
           "method": (f"instruction microbenchmark: each row's mma.sync issued back to back from {record.get('grid', {}).get('warps')} "
                      f"warps, {NACC} accumulator chains x {NSETS} register-resident operand sets, median of {args.reps} interleaved "
                      f"event-timed launches after {args.sustain_s:g} s sustained; SM clock per launch from clock64; latency from one "
                      f"warp's dependent chain"),
           "grid": record.get("grid"), "comparisons": record.get("comparisons"),
           "price_per_mac_in_dense_nvf4_units": record.get("price_per_mac_in_dense_nvf4_units"),
           "workload_fingerprint": {"tool": "tc_price_fp4", "version": "1", "rows": record["rows"], "plan": record["plan"],
                                    "reps": args.reps, "target_ms": args.target_ms, "sustain_s": args.sustain_s,
                                    "warps_per_sm": args.warps_per_sm, "latency_iters": args.latency_iters, "seed": args.seed,
                                    "operands_sha256": record.get("operands_sha256"), "commit": os.environ.get("RESEARCH_SOURCE_COMMIT")},
           "validation": {"status": "passed" if ok else "failed",
                          "checker": "device UUID + cc 12.0; cuobjdump opcode counts per kernel; outputs finite and bit-identical "
                                     "across two launches (no arithmetic check: probe.py's)", "detail": detail,
                          "evidence": ["w1_price.json"]},
           "measurements": meas, "measurement_files": ["w1_price.json"], "artifacts": []}
    if not args.fake_silicon:
        (out / "result.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    else:
        (out / "result.fake.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if failures:
        print("FAILURES", failures, flush=True)
        return 2 if any("UUID" in f for f in failures) else 1
    print("W1 PRICE", "SASS ONLY" if per_variant is None else "RECORDED", flush=True)
    for k, v in (per_variant or {}).items():
        print(f"  {k:40s} {v['flop_per_second'] / 1e12:9.1f} TFLOP/s  {v['flop_per_sm_per_clock']:8.1f} flop/SM/clk  {v['sm_mhz_median']:7.1f} MHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
