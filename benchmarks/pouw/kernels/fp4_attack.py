"""fp4_attack: build, SASS-gate, and (on the card) time the FP4 attacker's three tensor-core routes (``fp4_attack.cu``).

    python benchmarks/pouw/kernels/fp4_attack.py build              # nvcc + cuobjdump gate only (no GPU): the SASS opcodes + the price table
    python benchmarks/pouw/kernels/fp4_attack.py peak [--expect-uuid GPU-...] [--reps 5] [--sustain-s 2.0]

The three routes, each an instruction-peak loop (the ``mma_peak_fp4.cu`` pattern: NSETS operand sets x NACC accumulator
chains, register-resident, every warp of every SM):

  dense   OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X       the honest block-scaled FP4 rate (m16n8k64)
  sparse  OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X   2:4-sparse A (m16n8k128): the 0.5-unit route the design must exclude
  int8    IMMA.16832.S8.S8                            the exact-integer route (2 x E2M1 fits int8; m16n8k32)

``build`` compiles for sm_120a (ptxas refuses ``.block_scale`` / ``.kind::mxf4nvf4`` on plain sm_120), then cuobjdump
counts the opcodes: the gate is exactly NSETS x NACC of each in its kernel (the ``#pragma unroll 1`` body once).  This runs
on a CPU box with nvcc 12.9 -- it is the record-only ptxas/SASS proof that 2:4 sparse ``mma.sp`` kind::mxf4nvf4 exists on
sm_120a (GPU 4's recheck item 6).  ``peak`` additionally times each loop on the card and reports TFLOP/s and the ratio to
the dense rate; ``sparse_over_dense`` (logical flop) is the measured 2:4 speed-up the design's 0.5-unit price assumes.

Under ``research run`` (``$RESEARCH_RUN_DIR`` set) it writes a ``research/result/v0.1`` document.  Honest note: this is a
rate and a SASS gate, not an arithmetic check -- ``fp4_census`` and ``tc_probe`` check the words; the gate here is that the
outputs are finite and bit-identical across two launches and the SASS opcodes are the expected ones.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_result  # noqa: E402  benchmarks/pouw/run_result.py

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fp4_attack.cu"
ARCH = "sm_120a"
NSETS, NACC = 4, 8
BODY = NSETS * NACC                      # MMAs per loop body / expected SASS count per opcode

KINDS = ("dense", "sparse", "int8")
OPCODES = {
    "dense": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
    "sparse": "OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X",
    "int8": "IMMA.16832.S8.S8",
}
FLOP_PER_MMA = {"dense": 2 * 16 * 8 * 64, "sparse": 2 * 16 * 8 * 128, "int8": 2 * 16 * 8 * 32}
WORDS_PER_SET = {"dense": 8, "sparse": 12, "int8": 6}

# The design's placeholder W1 prices (FP4 units per dense FP4 MAC), from public peaks; `peak` replaces them with measured
# ratios.  (fp4-design.md §5 / fp4_census.py PRICES.)
PUBLIC_PRICE = {"dense": 1.0, "sparse": 0.5, "int8": 2.0}


def nvcc_bin() -> str | None:
    for c in (shutil.which("nvcc"), "/usr/local/cuda-12.9/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).is_file():
            return c
    return None


def tool_beside_nvcc(name: str) -> str | None:
    nvcc = nvcc_bin()
    if nvcc:
        p = Path(nvcc).parent / name
        if p.is_file():
            return str(p)
    return shutil.which(name)


def build(out_dir: Path) -> dict:
    """nvcc the three loops for sm_120a; return the artifact path, the nvcc version, and the cuobjdump SASS gate."""
    nvcc = nvcc_bin()
    if nvcc is None:
        raise FileNotFoundError("nvcc not found (PATH, /usr/local/cuda-12.9/bin or /usr/local/cuda/bin)")
    so = out_dir / f"libfp4_attack_{ARCH}.so"
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    cmd = [nvcc, "-O2", f"-gencode=arch=compute_{ARCH[3:]},code={ARCH}", "-Xcompiler", "-fPIC", "-shared",
           "-o", str(so), str(SOURCE)]
    subprocess.run(cmd, check=True)
    sass = sass_gate(so)
    return {"so": str(so), "nvcc": " / ".join(l.strip() for l in ver[-2:]), "nvcc_cmd": " ".join(cmd),
            "sha256": _sha256(so), "sass": sass}


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sass_gate(so: Path) -> dict:
    """cuobjdump -sass: count each MMA opcode; the gate is exactly BODY (=NSETS x NACC) of each expected opcode."""
    cuobjdump = tool_beside_nvcc("cuobjdump")
    if not cuobjdump:
        return {"ok": False, "error": "cuobjdump not found", "counts": {}}
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    counts: dict[str, int] = {}
    for line in text.splitlines():
        for m in re.finditer(r"\b((?:H|Q|I|O|D|B)?G?MMA(?:\.[A-Z0-9_]+)*)\b", line):
            if "/*" in line:                                # a real SASS instruction line has an address comment
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    per_kind = {k: counts.get(op, 0) for k, op in OPCODES.items()}
    ok = all(per_kind[k] == BODY for k in KINDS)
    return {"ok": ok, "expected_per_kind": BODY, "per_kind": per_kind, "counts": counts,
            "opcodes": {k: OPCODES[k] for k in KINDS}}


# ---- fragments -------------------------------------------------------------------------------------------------------------

def _e2m1_words(rng, n) -> np.ndarray:
    return rng.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)


def _ue4m3_scales(rng, n) -> np.ndarray:
    """UE4M3 scale bytes drawn from the finite codes around 1 (0x30..0x47), packed 4 to a u32 (as the peak kernel does)."""
    b = rng.integers(0x30, 0x48, size=(n, 4), dtype=np.uint32)
    return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)


def fragments(kind: str, nwarps: int, seed: int = 20260930) -> np.ndarray:
    """Per-lane register fragments for `kind`: nwarps*32 lanes x NSETS sets x WORDS_PER_SET[kind] u32, host order."""
    rng = np.random.default_rng(seed)
    lanes = nwarps * 32
    w = WORDS_PER_SET[kind]
    f = np.zeros((lanes, NSETS, w), dtype=np.uint32)
    if kind == "dense":
        f[:, :, :6] = _e2m1_words(rng, lanes * NSETS * 6).reshape(lanes, NSETS, 6)          # a[4] b[2]
        f[:, :, 6] = _ue4m3_scales(rng, lanes * NSETS).reshape(lanes, NSETS)                # sfa
        f[:, :, 7] = _ue4m3_scales(rng, lanes * NSETS).reshape(lanes, NSETS)                # sfb
    elif kind == "sparse":
        f[:, :, :8] = _e2m1_words(rng, lanes * NSETS * 8).reshape(lanes, NSETS, 8)          # a[4] b[4]
        f[:, :, 8] = np.uint32(0x44444444)                                                 # 2:4 ordered metadata: {0,1} per group
        f[:, :, 9] = _ue4m3_scales(rng, lanes * NSETS).reshape(lanes, NSETS)               # sfa
        f[:, :, 10] = _ue4m3_scales(rng, lanes * NSETS).reshape(lanes, NSETS)              # sfb
    elif kind == "int8":
        f[:, :, :6] = _e2m1_words(rng, lanes * NSETS * 6).reshape(lanes, NSETS, 6)          # a[4] b[2] (random s8 bytes)
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(f.reshape(-1))


# ---- timing on the card ----------------------------------------------------------------------------------------------------

def _lib(so: Path):
    lib = ctypes.CDLL(str(so))
    lib.fp4_attack_device.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
    lib.fp4_attack_words_per_lane.argtypes = [ctypes.c_int]
    lib.fp4_attack_flop_per_mma.argtypes = [ctypes.c_int]
    lib.fp4_attack_flop_per_mma.restype = ctypes.c_longlong
    lib.fp4_attack_setup.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.fp4_attack_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
                                   ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
    return lib


def _check_identity(expect_uuid: str | None) -> dict:
    """One visible device; its UUID equals --expect-uuid (from server.md) and gpu-lease's GPU_LEASE_UUID when set.  A guard
    never fails open (AGENTS.md).  nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists every GPU on the node, so it is asked
    for the visible device only (``-i``; gpu-lease sets CUDA_DEVICE_ORDER=PCI_BUS_ID, nvidia-smi's own order)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    lease = os.environ.get("GPU_LEASE_UUID") or None
    smi = shutil.which("nvidia-smi")
    uuid = None
    if smi:
        sel = ["-i", vis] if vis and "," not in vis else []
        r = subprocess.run([smi, *sel, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        uuid = lines[0] if len(lines) == 1 else f"{len(lines)} devices visible"
    want = expect_uuid or lease
    ok = uuid is not None and uuid.startswith("GPU-") and (want is None or uuid == want) and (lease is None or uuid == lease)
    return {"ok": ok, "cuda_visible_devices": vis, "device_uuid": uuid, "expect_uuid": expect_uuid, "gpu_lease_uuid": lease}


def measure(kind: str, so: Path, warmup: int, reps: int, sustain_s: float, warps_per_sm: int = 16,
            target_rep_s: float = 0.25, seed: int = 20260930) -> dict:
    lib = _lib(so)
    ki = KINDS.index(kind)
    sms, maxthr, major, minor = (ctypes.c_int() for _ in range(4))
    if lib.fp4_attack_device(sms, maxthr, major, minor):
        raise RuntimeError("fp4_attack_device failed")
    warps_per_block = 4
    nblocks = sms.value * (warps_per_sm // warps_per_block)
    nwarps = nblocks * warps_per_block
    words = lib.fp4_attack_words_per_lane(ki)
    frags = fragments(kind, nwarps, seed)
    if lib.fp4_attack_setup(frags.ctypes.data, nwarps, words):
        raise RuntimeError("fp4_attack_setup failed (cuda error)")
    ms = ctypes.c_float()
    sink = np.zeros(nwarps * 32 * 4, dtype=np.float32)

    def run(iters, want_sink=False):
        if lib.fp4_attack_run(ki, nblocks, warps_per_block, iters, ms, sink.ctypes.data if want_sink else None):
            raise RuntimeError("fp4_attack_run failed (cuda error)")
        return ms.value / 1e3

    flop_per_iter = float(nwarps) * BODY * FLOP_PER_MMA[kind]
    t_cal = run(2000)
    iters = max(2000, int(2000 * target_rep_s / max(t_cal, 1e-6)))
    for _ in range(warmup):
        run(iters)
    t0 = time.time()
    k, sus = 0, 0.0
    while True:
        sus += run(iters); k += 1
        if time.time() - t0 >= sustain_s:
            break
    times = [run(iters) for _ in range(reps)]
    med = statistics.median(times)
    run(iters, want_sink=True); first = sink.copy()
    run(iters, want_sink=True)
    finite = bool(np.isfinite(first).all())
    same = bool(np.array_equal(first.view(np.uint32), sink.view(np.uint32)))
    lib.fp4_attack_teardown()
    flop = flop_per_iter * iters
    return {"kind": kind, "opcode": OPCODES[kind], "iters": iters, "reps": reps, "warmup": warmup,
            "grid": {"sms": sms.value, "blocks": nblocks, "warps": nwarps, "mmas_per_iter": BODY, "sets": NSETS, "chains": NACC},
            "seconds_median": med, "seconds_all": times, "flop_per_second": flop / med,
            "sustain_flop_per_second": flop * k / sus, "finite": finite, "deterministic": same,
            "cc": f"{major.value}.{minor.value}"}


# ---- CLI -------------------------------------------------------------------------------------------------------------------

def cmd_build(args) -> int:
    out_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or args.out or HERE / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = build(out_dir)
    g = rec["sass"]
    print(f"nvcc: {rec['nvcc']}")
    print(f"built {Path(rec['so']).name}  sha256 {rec['sha256'][:16]}")
    print("SASS gate (expected {} of each):".format(g["expected_per_kind"]))
    for k in KINDS:
        print(f"  {k:7s} {OPCODES[k]:44s} x{g['per_kind'][k]}  {'OK' if g['per_kind'][k] == g['expected_per_kind'] else 'FAIL'}")
    print(f"gate: {'PASS' if g['ok'] else 'FAIL'}")
    print("\nPublic-peak price (FP4 units per dense MAC; peak replaces with measured):")
    for k in KINDS:
        print(f"  {k:7s} {PUBLIC_PRICE[k]}")
    detail = {"build": rec, "public_price": PUBLIC_PRICE, "note": "SASS gate only; no GPU timing (run `peak` on the card)"}
    out = out_dir / "fp4_attack_build.json"
    out.write_text(json.dumps(detail, indent=1) + "\n")
    meas = [(f"sass_{k}", float(g["per_kind"][k]), "count") for k in KINDS]
    run_result.write(out, g["ok"], "cuobjdump -sass: NSETS x NACC of each expected opcode",
                     "; ".join(f"{k} {OPCODES[k]} x{g['per_kind'][k]}" for k in KINDS), meas)
    return 0 if g["ok"] else 1


def cmd_peak(args) -> int:
    out_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or args.out or HERE / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    ident = _check_identity(args.expect_uuid)
    if not ident["ok"]:
        print(f"identity gate FAILED: {ident}", file=sys.stderr)
        detail = {"identity": ident}
        (out_dir / "fp4_attack_peak.json").write_text(json.dumps(detail, indent=1) + "\n")
        run_result.write(out_dir / "fp4_attack_peak.json", False, "device UUID == --expect-uuid",
                         f"identity gate failed: {ident}", [])
        return 5
    rec = build(out_dir)
    if not rec["sass"]["ok"]:
        print(f"SASS gate FAILED: {rec['sass']['per_kind']}", file=sys.stderr)
        return 1
    so = Path(rec["so"])
    results = {}
    for k in KINDS:
        r = measure(k, so, args.warmup, args.reps, args.sustain_s)
        results[k] = r
        print(f"{k:7s} {r['flop_per_second']/1e12:8.1f} TFLOP/s (median {r['seconds_median']*1e3:.2f} ms)  "
              f"finite={r['finite']} det={r['deterministic']}")
    dense_fps = results["dense"]["flop_per_second"]
    ratios = {k: results[k]["flop_per_second"] / dense_fps for k in KINDS}
    print("\nlogical TFLOP/s ratio to dense (sparse>1 => 2:4 pays; the 0.5-unit price is 1/ratio):")
    for k in KINDS:
        print(f"  {k:7s} {ratios[k]:.3f}   measured price {1.0/ratios[k]:.3f} vs public {PUBLIC_PRICE[k]}")
    ok = all(results[k]["finite"] and results[k]["deterministic"] for k in KINDS)
    detail = {"identity": ident, "build": rec, "results": results, "ratios_to_dense": ratios,
              "measured_price": {k: 1.0 / ratios[k] for k in KINDS}, "public_price": PUBLIC_PRICE}
    out = out_dir / "fp4_attack_peak.json"
    out.write_text(json.dumps(detail, indent=1) + "\n")
    meas = []
    for k in KINDS:
        meas.append((f"{k}_tflops", results[k]["flop_per_second"] / 1e12, "TFLOP/s"))
    meas.append(("sparse_over_dense", ratios["sparse"], "ratio"))
    meas.append(("int8_over_dense", ratios["int8"], "ratio"))
    run_result.write(out, ok, "outputs finite and bit-identical across two launches; SASS opcodes gated",
                     "; ".join(f"{k} {results[k]['flop_per_second']/1e12:.1f} TFLOP/s" for k in KINDS), meas)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="nvcc + cuobjdump SASS gate (no GPU)")
    b.add_argument("--out", default=None)
    b.set_defaults(fn=cmd_build)
    p = sub.add_parser("peak", help="time the three routes on the card")
    p.add_argument("--expect-uuid", default=None, help="device UUID to require (server.md)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--sustain-s", type=float, default=2.0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_peak)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
