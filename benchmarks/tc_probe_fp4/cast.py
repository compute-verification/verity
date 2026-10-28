"""The FP4-path conversions on sm_120, bit for bit against exact references (``probe.py --instruction cvt_*``; kernels in
``cvt_fp4.cu``, one F2FP per conversion in SASS).

References (the hypotheses under test; every mismatch is reported by input class, nothing is tuned):

* ``cvt_e2m1x2_f32`` (``cvt.rn.satfinite.e2m1x2.f32``): round to nearest on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to
  the even code, |x| > 6 saturates to 6, the sign kept (a negative that rounds to zero is code 0x8).  Gated: every finite input.
* ``cvt_f16x2_e2m1x2``: the exact value of each nibble as FP16 (-0 kept).  Gated: all 256 bytes.
* ``cvt_ue8m0x2_f32_rz`` / ``_rp``: 2^floor(log2 x) / 2^ceil(log2 x) as a UE8M0 code (2^(c-127)), saturating at 254 (2^127).
  Gated: positive normal inputs; zero, negatives, subnormals, inf and NaN are recorded with the reference's prediction.
* ``cvt_bf16x2_ue8m0x2``: 2^(c-127) as BF16 (c = 0: the BF16 subnormal 2^-127).  Gated: codes 0..254 in both halves; 0xFF recorded.
* ``cvt_e4m3x2_f32_ue4m3`` (``cvt.rn.satfinite.e4m3x2.f32``) on the UE4M3 domain the NVFP4 recipe feeds it (positive inputs):
  round to nearest even on the E4M3 grid, saturating to 448.  Gated: positive finite inputs; the signed and special inputs are
  GPU 0's cast (recorded here).

The packing order (``a`` into the upper half) is part of what is checked: each output packs two inputs.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

import k32
import probe as P
from verity_config.settings import SHARED

HERE = Path(__file__).resolve().parent
OPS = {"cvt_e2m1x2_f32": 0, "cvt_f16x2_e2m1x2": 1, "cvt_ue8m0x2_f32_rz": 2, "cvt_ue8m0x2_f32_rp": 3, "cvt_bf16x2_ue8m0x2": 4,
       "cvt_e4m3x2_f32_ue4m3": 5}
PER_OUT = {0: 8, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2}
PTX = {"cvt_e2m1x2_f32": "cvt.rn.satfinite.e2m1x2.f32", "cvt_f16x2_e2m1x2": "cvt.rn.f16x2.e2m1x2",
       "cvt_ue8m0x2_f32_rz": "cvt.rz.satfinite.ue8m0x2.f32", "cvt_ue8m0x2_f32_rp": "cvt.rp.satfinite.ue8m0x2.f32",
       "cvt_bf16x2_ue8m0x2": "cvt.rn.bf16x2.ue8m0x2", "cvt_e4m3x2_f32_ue4m3": "cvt.rn.satfinite.e4m3x2.f32 (UE4M3 domain: x > 0)"}
INSTRUCTION_ID = {k: "sm120." + v.split(" ")[0] for k, v in PTX.items()}

E2M1_GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
E2M1_MIDS = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2


# --- references ------------------------------------------------------------------------------------------------------------------


def _f64(words: np.ndarray) -> np.ndarray:
    return np.asarray(words, np.uint32).view(np.float32).astype(np.float64)


def ref_e2m1(words: np.ndarray) -> np.ndarray:
    """E2M1 codes of FP32 words (non-finite inputs: 0xFF marks "no reference")."""
    x = _f64(words)
    a = np.abs(x)
    idx = np.searchsorted(E2M1_MIDS, a, side="left")          # number of midpoints below |x|
    tie = (idx < len(E2M1_MIDS)) & (a == E2M1_MIDS[np.minimum(idx, len(E2M1_MIDS) - 1)])
    idx = np.where(tie & (idx % 2 == 1), idx + 1, idx)         # a tie goes to the even code
    idx = np.minimum(idx, 7)
    code = idx.astype(np.uint32) | np.where(np.signbit(x), 8, 0).astype(np.uint32)
    return np.where(np.isfinite(x), code, 0xFF).astype(np.uint32)


def ref_ue8m0(words: np.ndarray, mode: str) -> np.ndarray:
    """UE8M0 codes (0xFFFF: no reference) for positive normal inputs; the prediction elsewhere is ``_ue8m0_edges``."""
    w = np.asarray(words, np.uint32)
    e = (w >> 23) & 0xFF
    m = w & 0x7FFFFF
    c = e.astype(np.int64) + (1 if mode == "rp" else 0) * (m != 0)
    c = np.minimum(c, 254)
    ok = (w >> 31 == 0) & (e >= 1) & (e <= 254)
    return np.where(ok, c, 0xFFFF).astype(np.uint32)


def _ue8m0_edges(words: np.ndarray, mode: str) -> np.ndarray:
    """Conjectured codes outside the gated domain (recorded, never gated): NaN -> 0xFF, +inf -> 254 (satfinite), positive
    subnormals rz -> 0 when >= 2^-127 else 0 (the smallest code), rp -> 0 when <= 2^-127 else 1, zero and negatives -> 0."""
    w = np.asarray(words, np.uint32)
    e, m = (w >> 23) & 0xFF, w & 0x7FFFFF
    out = np.zeros(w.shape, np.uint32)
    out[(e == 0xFF) & (m != 0)] = 0xFF
    out[w == 0x7F800000] = 254
    sub = (w >> 31 == 0) & (e == 0) & (m != 0)
    if mode == "rp":
        out[sub & (m > (1 << 22))] = 1
    return out


def ref_e4m3_pos(words: np.ndarray) -> np.ndarray:
    """E4M3 bytes of positive finite FP32 inputs, round to nearest even, saturating at 448 (0x7E); 0xFFFF: no reference."""
    x = _f64(words)
    ok = np.isfinite(x) & (x > 0)
    xs = np.where(ok, x, 1.0)
    e = np.floor(np.log2(xs)).astype(np.int64)
    q = np.exp2((np.maximum(e, -6) - 3).astype(np.float64))
    v = np.rint(xs / q) * q                                    # exact: xs / q has at most 24 significant bits
    v = np.minimum(v, 448.0)
    lut = {float(k32.E4M3_VALUE[c]): c for c in range(0x7F) if c != 0x7F}
    code = np.array([lut[float(t)] for t in v.ravel()], np.uint32).reshape(v.shape)
    return np.where(ok, code, 0xFFFF).astype(np.uint32)


def _f16_bits(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, np.float16).view(np.uint16).astype(np.uint32)


def reference(op: int, inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(predicted output words, gated mask) for the op's input words (grouped per output as the kernel reads them)."""
    if op == 0:
        x = inp.reshape(-1, 4, 2)
        ca, cb = ref_e2m1(x[:, :, 0]), ref_e2m1(x[:, :, 1])
        byte = (ca << 4) | cb
        gated = (ca != 0xFF).all(1) & (cb != 0xFF).all(1)
        word = (byte[:, 0] | (byte[:, 1] << 8) | (byte[:, 2] << 16) | (byte[:, 3] << 24)).astype(np.uint32)
        return word, gated
    if op == 1:
        b = inp & 0xFF
        lo = _f16_bits(k32.E2M1_VALUE[b & 0xF])
        hi = _f16_bits(k32.E2M1_VALUE[b >> 4])
        return (lo | (hi << 16)).astype(np.uint32), np.ones(b.shape, bool)
    if op in (2, 3):
        mode = "rz" if op == 2 else "rp"
        x = inp.reshape(-1, 2)
        ra, rb = ref_ue8m0(x[:, 0], mode), ref_ue8m0(x[:, 1], mode)
        gated = (ra != 0xFFFF) & (rb != 0xFFFF)
        ea, eb = _ue8m0_edges(x[:, 0], mode), _ue8m0_edges(x[:, 1], mode)
        ra, rb = np.where(ra == 0xFFFF, ea, ra), np.where(rb == 0xFFFF, eb, rb)
        return ((ra << 8) | rb).astype(np.uint32), gated
    if op == 4:
        h = inp & 0xFFFF
        def bf(c):
            v = np.exp2(c.astype(np.float64) - 127).astype(np.float32).view(np.uint32) >> 16
            return np.where(c == 0, 0x0040, v).astype(np.uint32)
        lo, hi = h & 0xFF, h >> 8
        gated = (lo != 0xFF) & (hi != 0xFF)
        return (bf(lo) | (bf(hi) << 16)).astype(np.uint32), gated
    x = inp.reshape(-1, 2)
    ra, rb = ref_e4m3_pos(x[:, 0]), ref_e4m3_pos(x[:, 1])
    gated = (ra != 0xFFFF) & (rb != 0xFFFF)
    return ((np.where(gated, ra, 0) << 8) | np.where(gated, rb, 0)).astype(np.uint32), gated


# --- inputs ----------------------------------------------------------------------------------------------------------------------


def _f32w(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


def _neighbours(values) -> np.ndarray:
    """Every value, its FP32 neighbours one ulp either side, both signs."""
    v = np.asarray(values, np.float32)
    out = np.concatenate([v, np.nextafter(v, np.float32(np.inf)), np.nextafter(v, np.float32(-np.inf))])
    return _f32w(np.concatenate([out, -out]))


def inputs(op: int, rng, n: int) -> dict[str, np.ndarray]:
    """Named input sets, each a flat array of input words (a multiple of the op's inputs per output)."""
    per = PER_OUT[op]
    sets: dict[str, np.ndarray] = {}
    if op in (1, 4):
        sets["exhaustive"] = np.arange(256 if op == 1 else 65536, dtype=np.uint32)
        return sets
    exp_sweep = []
    for e in range(256):
        for m in (0, 1, 1 << 22, (1 << 23) - 1, *rng.integers(0, 1 << 23, 4).tolist()):
            exp_sweep.append((e << 23) | m)
    exp_sweep = np.array(exp_sweep, np.uint32)
    sets["exponent_sweep"] = np.concatenate([exp_sweep, exp_sweep | np.uint32(0x80000000)])
    if op == 0:
        grid = np.concatenate([E2M1_GRID, E2M1_MIDS, [6.5, 7, 8, 12, 1e30, 0.125, 0.25 - 2**-30]])
        sets["grid_and_midpoints"] = _neighbours(grid)
        sets["in_range"] = _f32w(rng.uniform(-8, 8, n * per))
        sets["pack_order"] = _f32w(np.tile(np.array([6, 0.5, 1, -2, 3, -0.5, 4, 1.5], np.float32), max(1, n // 64)))
    elif op in (2, 3):
        p2 = np.exp2(np.arange(-127, 128, dtype=np.float64)).astype(np.float32)
        sets["powers_and_neighbours"] = _neighbours(p2)
        sets["in_range"] = _f32w(np.exp2(rng.uniform(-126, 128, n * per)))
        sets["pack_order"] = _f32w(np.tile(np.array([2.0**10, 2.0**-10], np.float32), max(1, n // 16)))
    else:
        grid = k32.E4M3_VALUE[[c for c in range(0x7F)]]
        mids = (grid[1:] + grid[:-1]) / 2
        sets["grid_and_midpoints"] = _neighbours(np.concatenate([grid, mids, [464, 480, 1e4, 2.0**-10, 2.0**-11]]))
        sets["in_range"] = _f32w(np.exp2(rng.uniform(-12, 9.5, n * per)))
        sets["pack_order"] = _f32w(np.tile(np.array([448.0, 2.0**-9], np.float32), max(1, n // 16)))
    sets["uniform_bits"] = rng.integers(0, 1 << 32, n * per, dtype=np.uint64).astype(np.uint32)
    sets["specials"] = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 1, 0x80000001, 0x007FFFFF],
                                np.uint32)
    for name, a in sets.items():
        pad = (-len(a)) % per
        if pad:
            sets[name] = np.concatenate([a, a[:pad] if len(a) >= pad else np.zeros(pad, np.uint32)])
    return sets


# --- driver ----------------------------------------------------------------------------------------------------------------------


class CastDevice:
    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        self.lib.cvt_run.restype = ctypes.c_int
        self.lib.cvt_run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.device_uuid.restype = ctypes.c_int
        self.lib.device_uuid.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.has_mxf4.restype = ctypes.c_int
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
        self.tcgen05 = False

    identity = P.Probe.identity
    clocks = P.Probe.clocks

    def run(self, op: int, inp: np.ndarray) -> np.ndarray:
        inp = np.ascontiguousarray(inp, np.uint32)
        n = len(inp) // PER_OUT[op]
        out = np.zeros(n, np.uint32)
        rc = self.lib.cvt_run(op, inp.ctypes.data, out.ctypes.data, n)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cvt op {op}")
        return out


class FakeCast:
    """CPU stand-in: the reference's own words (``--fake-silicon reference``)."""

    tcgen05 = False

    def identity(self) -> dict:
        return {"device": "FAKE SILICON (reference)", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def run(self, op: int, inp: np.ndarray) -> np.ndarray:
        return reference(op, inp)[0]


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    instr, op = args.instruction, OPS[args.instruction]
    arch = args.arch or "sm_120a"
    if args.fake_silicon and SHARED.RESEARCH_RUN_ID:
        raise SystemExit("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    so, nvcc = (None, "not built") if (args.fake_silicon and not P.shutil.which(P.nvcc_bin())) else P.build(arch, out, "cvt_fp4.cu")
    dev = FakeCast() if args.fake_silicon else (None if args.sass_only else CastDevice(so))
    identity = dev.identity() if dev is not None else {"device": None, "gpu_uuid": None, "driver": None, "compute_capability": None,
                                                       "backend": "none (--sass-only)"}
    identity.update(nvcc=nvcc, arch=arch)
    failures: list[str] = []
    if args.expect_uuid and dev is not None and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        msg = f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement"
        (out / "probe_results.json").write_text(json.dumps({"identity": identity, "failures": [msg]}, indent=1))
        print("FAILURES", [msg], flush=True)
        return 2
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-cast/v1", "instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr],
                     "identity": identity, "reference": "cast.py (" + instr + ")", "fake_silicon": args.fake_silicon,
                     "corrupt_word": bool(args.corrupt_word)}
    if so is not None and args.sass_check:
        text = P.subprocess.run([str(Path(P.nvcc_bin()).parent / "cuobjdump"), "-sass", str(so)], capture_output=True, text=True,
                                check=True).stdout
        sass, fn = {}, None
        for line in text.splitlines():
            if "Function :" in line:
                fn = line.split("Function :")[1].strip()
                sass[fn] = []
            elif fn and "F2FP" in line:
                sass[fn].append(line.split("*/")[1].split()[0] if "*/" in line else line.strip())
        results["sass"] = sass
        print("sass", json.dumps(sass), flush=True)
    if args.sass_only:
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        return 0
    if dev is not None:
        results["clocks_start"] = dev.clocks(identity.get("gpu_uuid"))
    rng = np.random.default_rng(args.seed)
    sets = inputs(op, rng, args.n_random * 128)
    fams, total_gated, total_mism, recorded = {}, 0, 0, {}
    corrupted = False
    for name, inp in sets.items():
        got = dev.run(op, inp)
        want, gated = reference(op, inp)
        if args.corrupt_word and not corrupted and gated.any():  # a gated word: a record-only one would leave the gate untested
            got = got.copy()
            got[int(np.argmax(gated))] ^= 1
            corrupted = True
        mism = gated & (got != want)
        rec_only = ~gated
        fam = {"outputs": int(len(got)), "gated": int(gated.sum()), "mismatches": int(mism.sum()),
               "record_only": int(rec_only.sum()), "record_only_differs_from_prediction": int((rec_only & (got != want)).sum()),
               "first_mismatches": [], "record_only_sample": []}
        per = PER_OUT[op]
        grouped = inp.reshape(-1, per)
        for i in np.nonzero(mism)[0][:20]:
            fam["first_mismatches"].append({"in": [f"{int(w):08x}" for w in grouped[i]], "hw": f"{int(got[i]):08x}", "ref": f"{int(want[i]):08x}"})
        for i in np.nonzero(rec_only)[0][:40]:
            fam["record_only_sample"].append({"in": [f"{int(w):08x}" for w in grouped[i]], "hw": f"{int(got[i]):08x}",
                                              "conjectured": f"{int(want[i]):08x}"})
        np.savez_compressed(out / f"cast_{name}.npz", inp=inp, out=got, ref=want, gated=gated)
        fams[name] = fam
        total_gated += fam["gated"]
        total_mism += fam["mismatches"]
        recorded[name] = fam["record_only"]
        print(name, json.dumps({k: v for k, v in fam.items() if k not in ("first_mismatches", "record_only_sample")}), flush=True)
    if total_mism:
        failures.append(f"{total_mism} gated outputs differ from the reference")
    if dev is not None:
        results["clocks_end"] = dev.clocks(identity.get("gpu_uuid"))
    results.update(families=fams, gated_outputs=total_gated, mismatches=total_mism, seconds=round(time.time() - t_all, 1), failures=failures)
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": INSTRUCTION_ID[instr],
            "ptx": PTX[instr], "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "families": {k: {kk: vv for kk, vv in f.items() if kk not in ("first_mismatches",)} for k, f in fams.items()},
            "counts": {"elements": sum(f["outputs"] for f in fams.values()), "gated_elements": total_gated, "mismatches": {"reference": total_mism}},
            "model": results["reference"], "primary_model": "reference", "model_agrees": total_mism == 0, "agreement": {"reference": total_mism == 0},
            "first_mismatches": [m for f in fams.values() for m in f["first_mismatches"]][:20], "seed": args.seed, "n": args.n_random,
            "sass_opcodes": {k: sorted(set(v)) for k, v in results.get("sass", {}).items()},
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"), "note": "recorded, not locked"},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/probe.py", "instruction": INSTRUCTION_ID[instr], "arch": arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": args.n_random},
            "validation": {"status": "passed" if not failures else "failed", "checker": "tc_probe_fp4 cast.py vs its exact references",
                           "detail": "; ".join(failures) or "every gated output reproduced", "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"], "artifacts": sorted(p.name for p in out.glob("cast_*.npz")),
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    print("FAILURES" if failures else "ALL GATED OUTPUTS REPRODUCED", failures, f"{results['seconds']}s", flush=True)
    return 1 if failures else 0
