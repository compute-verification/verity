"""Hardware probe for the Ampere BF16 transition unit: one ``mma.sync.m16n8k16`` per tile, arbitrary accumulators, captures in the
golden-record format, compared bit for bit against ``verity.ml.tc.silicon.tc_dot``.

Provenance: ported for lane WP1 from veritor ``gpu/tensor-core-semantics/`` -- ``mma_tiles.cu`` (the kernel, BF16 half),
``tcs_gpu.py`` (the ctypes driver and the BF16 encodings) and ``validate_tiles.py`` (the finite-sweep families, the per-element
records, the multiprocess model comparison and the golden sampling).  The E4M3 path, the torch dependency and the Hawkeye
``matmul`` cross-check (now done CPU-side by ``backends/numerical/tests/reference``) were left behind.  Only the GPU host runs
this; it needs numpy and nvcc.

    python tools/tc_probe/tc_probe.py --arch sm_89 --out $RESEARCH_RUN_DIR \
        --replay-golden packages/verity/tests/ml/fixtures/golden/ada_bf16_m16n8k16.json \
        --sweep --n-random 100000

``--replay-golden`` re-executes every record of a golden file on the device (each at its own tile position) and requires the
hardware word to equal the record's ``d``.  ``--sweep`` runs the finite-sweep families (random, random nonzero accumulators, uniform
encodings, subnormal, cancellation, mixed magnitude, near overflow, signed zero, tiny accumulators) and compares every output
element with ``silicon.tc_dot``.  Outputs under ``--out``: ``probe_results.json`` (counts, device identity, first mismatches),
``capture_sample.json`` (golden-format records sampled per family), ``tiles_<family>.npz`` (the full capture: A, Bt, C, D per
tile) and ``result.json`` (the research envelope; validation passes iff every mismatch count is zero).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from verity.ml.tc import silicon as S

HERE = Path(__file__).resolve().parent
K = 16
MAX_TILES_PER_CALL = 1 << 16


# --- the kernel ----------------------------------------------------------------------------------------------------------------


def build(arch: str, out: Path) -> tuple[Path, str]:
    so = out / f"libmma_tiles_{arch}.so"
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    subprocess.run(
        ["nvcc", "-O2", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / "mma_tiles.cu")],
        check=True,
    )
    return so, " / ".join(line.strip() for line in nvcc)


class Probe:
    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        fn = self.lib.mma_bf16_tiles
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cudaGetDeviceProperties")
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return {
            "device": name.value.decode(),
            "compute_capability": f"{major.value}.{minor.value}",
            "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
            "hostname": platform.node(),
        }

    def mma_tiles(self, A: np.ndarray, Bt: np.ndarray, C: np.ndarray) -> np.ndarray:
        """One ``mma.sync`` per tile.  ``A``: (n,16,16), ``Bt``: (n,8,16) BF16 words, ``C``: (n,16,8) FP32 words -> ``D`` (n,16,8)."""

        n = A.shape[0]
        A = np.ascontiguousarray(A, dtype=np.uint16).reshape(n, 16, K)
        Bt = np.ascontiguousarray(Bt, dtype=np.uint16).reshape(n, 8, K)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
        D = np.zeros((n, 16, 8), dtype=np.uint32)
        for start in range(0, n, MAX_TILES_PER_CALL):
            stop = min(n, start + MAX_TILES_PER_CALL)
            a, b, c = A[start:stop], Bt[start:stop], C[start:stop]
            d = np.zeros((stop - start, 16, 8), dtype=np.uint32)
            rc = self.lib.mma_bf16_tiles(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, stop - start)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the tile kernel")
            D[start:stop] = d
        return D


# --- encodings (tcs_gpu.py) ------------------------------------------------------------------------------------------------------


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 (torch's cast)."""

    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def random_finite_fp32_bits(rng, n, exp_lo=1, exp_hi=254):
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return (s << 31) | (e << 23) | m


def random_finite_bf16_bits(rng, shape, exp_lo=1, exp_hi=254):
    n = int(np.prod(shape))
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint16)
    m = rng.integers(0, 128, size=n).astype(np.uint16)
    s = rng.integers(0, 2, size=n).astype(np.uint16)
    return ((s << 15) | (e << 7) | m).reshape(shape)


# --- the finite-sweep families (validate_tiles.py, BF16) -----------------------------------------------------------------------------


def families(rng, n_random):
    """Yield (name, A (n,16,16), Bt (n,8,16), C (n,16,8)) -- the nine families of the Ada validation, unchanged."""

    cast = lambda x: to_bf16_bits(x.astype(np.float32))
    zeros_acc = lambda n: np.zeros((n, 16, 8), np.uint32)

    # 1. Hawkeye-style: standard normal operands, zero accumulator.
    n = n_random
    yield "randn_zero_acc", cast(rng.standard_normal((n, 16, K))), cast(rng.standard_normal((n, 8, K))), zeros_acc(n)

    # 2. standard normal operands, random nonzero accumulators over a wide range.
    n = n_random // 4
    acc = rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-20, 21, (n, 16, 8)))
    yield "randn_random_acc", cast(rng.standard_normal((n, 16, K))), cast(rng.standard_normal((n, 8, K))), f32_bits(acc.astype(np.float32))

    # 3. uniformly random finite encodings for everything (all exponents).
    n = n_random // 4
    yield (
        "uniform_bits",
        random_finite_bf16_bits(rng, (n, 16, K)),
        random_finite_bf16_bits(rng, (n, 8, K)),
        random_finite_fp32_bits(rng, n * 128).reshape(n, 16, 8),
    )

    # 4. subnormal-heavy operands, tiny/subnormal accumulators.
    n = n_random // 8
    yield (
        "subnormal",
        random_finite_bf16_bits(rng, (n, 16, K), 0, 3),
        random_finite_bf16_bits(rng, (n, 8, K), 0, 3),
        random_finite_fp32_bits(rng, n * 128, 0, 3).reshape(n, 16, 8),
    )

    # 5. exact cancellation: pairs of equal-magnitude opposite products, small residuals, random acc.
    n = n_random // 8
    a = rng.standard_normal((n, 16, K)) * 2.0**8
    b = rng.standard_normal((n, 8, K)) * 2.0**8
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    small = rng.standard_normal((n, 16, K)) * 2.0**-6
    mask = rng.random((n, 16, K)) < 0.15
    a = np.where(mask, small, a)
    acc = rng.standard_normal((n, 16, 8)) * 2.0**-4
    yield "cancellation", cast(a), cast(b), f32_bits(acc.astype(np.float32))

    # 6. mixed magnitudes: per-element random exponents over the whole operand range.
    n = n_random // 8
    yield (
        "mixed_magnitude",
        random_finite_bf16_bits(rng, (n, 16, K)),
        random_finite_bf16_bits(rng, (n, 8, K)),
        f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-60, 61, (n, 16, 8)))).astype(np.float32)),
    )

    # 7. near FP32 overflow: accumulators near max, large operands.
    n = n_random // 16
    yield (
        "near_overflow",
        random_finite_bf16_bits(rng, (n, 16, K), 245, 254),
        random_finite_bf16_bits(rng, (n, 8, K), 245, 254),
        random_finite_fp32_bits(rng, n * 128, 250, 254).reshape(n, 16, 8),
    )

    # 8. zero handling: -0 accumulators, all-zero products, cancellations to zero.
    n = 256
    a = np.zeros((n, 16, K), np.float32)
    b = np.zeros((n, 8, K), np.float32)
    a[n // 2 :, :, 0] = 1.0
    a[n // 2 :, :, 1] = 1.0
    b[n // 2 :, :, 0] = 3.0
    b[n // 2 :, :, 1] = -3.0
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", cast(a), cast(b), acc

    # 9. tiny accumulators with zero products (exponent floor probe).
    n = 512
    acc = random_finite_fp32_bits(rng, n * 128, 0, 30).reshape(n, 16, 8)
    yield "tiny_acc_zero_products", np.zeros((n, 16, K), np.uint16), np.zeros((n, 8, K), np.uint16), acc


# --- the model, per tile, in processes -----------------------------------------------------------------------------------------------

_PIPE = None


def _init(pipeline_name):
    global _PIPE
    _PIPE = S.PIPELINES[pipeline_name]


def _eval_tiles(args):
    A, Bt, C = args
    n = A.shape[0]
    out = np.zeros((n, 16, 8), np.uint32)
    err = np.zeros((n, 16, 8), np.bool_)
    for t in range(n):
        a_rows = [A[t, i].tolist() for i in range(16)]
        b_rows = [Bt[t, j].tolist() for j in range(8)]
        for i in range(16):
            for j in range(8):
                try:
                    out[t, i, j] = S.tc_dot(_PIPE, int(C[t, i, j]), a_rows[i], b_rows[j])
                except Exception:  # noqa: BLE001 - domain errors are counted, not raised
                    err[t, i, j] = True
    return out, err


def model_eval(pool, A, Bt, C, procs):
    n = A.shape[0]
    chunks = max(1, min(n // 64 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(_eval_tiles, [(A[i], Bt[i], C[i]) for i in idx])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# --- records ----------------------------------------------------------------------------------------------------------------------


def hexrec(acc, a, b, d) -> dict:
    return {
        "acc": f"{int(acc):08x}",
        "a": "".join(f"{int(x):04x}" for x in a),
        "b": "".join(f"{int(x):04x}" for x in b),
        "d": f"{int(d):08x}",
    }


def words(hex_text: str, width: int = 4) -> list[int]:
    return [int(hex_text[i : i + width], 16) for i in range(0, len(hex_text), width)]


def layout_check(probe: Probe) -> None:
    """Exact small-integer tiles: D must be C + A.Bt computed exactly (all sums below 2^24, so FP32 is exact)."""

    rng = np.random.default_rng(1)
    n = 64
    a = rng.integers(-8, 9, (n, 16, K)).astype(np.float32)
    b = rng.integers(-8, 9, (n, 8, K)).astype(np.float32)
    c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float32)
    want = f32_bits((c + np.einsum("nik,njk->nij", a, b)).astype(np.float32))
    got = probe.mma_tiles(to_bf16_bits(a), to_bf16_bits(b), f32_bits(c))
    if not np.array_equal(want, got):
        raise RuntimeError("fragment layout check failed: the kernel does not compute C + A.Bt on exact integer tiles")


def replay_golden(probe: Probe, path: Path) -> dict:
    """Every record at its own tile position (r % 16, (r // 16) % 8), the rest of the tile zero."""

    data = json.loads(path.read_text())
    records = data["records"]
    n = len(records)
    A = np.zeros((n, 16, K), np.uint16)
    Bt = np.zeros((n, 8, K), np.uint16)
    C = np.zeros((n, 16, 8), np.uint32)
    for r, rec in enumerate(records):
        i, j = r % 16, (r // 16) % 8
        A[r, i] = words(rec["a"])
        Bt[r, j] = words(rec["b"])
        C[r, i, j] = int(rec["acc"], 16)
    D = probe.mma_tiles(A, Bt, C)
    mism = []
    other_nonzero = 0
    for r, rec in enumerate(records):
        i, j = r % 16, (r // 16) % 8
        if int(D[r, i, j]) != int(rec["d"], 16):
            mism.append({**rec, "hardware": f"{int(D[r, i, j]):08x}"})
        rest = D[r].copy()
        rest[i, j] = 0
        other_nonzero += int(np.count_nonzero(rest))  # zero rows times anything, zero accumulator: every other output is +0
    return {
        "file": str(path),
        "pipeline": data["pipeline"],
        "records": n,
        "mismatches": len(mism),
        "first_mismatches": mism[:5],
        "other_positions_nonzero": other_nonzero,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="sm_80 (A100) or sm_89 (RTX 4090)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", default="ampere_bf16_m16n8k16")
    ap.add_argument("--replay-golden", default=None, metavar="JSON")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None, help="comma list (default all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    so, nvcc = build(args.arch, out)
    probe = Probe(so)
    identity = probe.identity()
    identity["nvcc"] = nvcc
    identity["arch"] = args.arch
    layout_check(probe)
    results: dict = {"schema": "wp1/tc-probe/v0", "identity": identity, "pipeline": args.pipeline, "layout_check": "passed"}
    print("device", json.dumps(identity), flush=True)

    if args.replay_golden:
        results["golden_replay"] = replay_golden(probe, Path(args.replay_golden))
        print("golden_replay", json.dumps({k: v for k, v in results["golden_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.sweep:
        only = set(args.families.split(",")) if args.families else None
        rng = np.random.default_rng(args.seed)
        pipe = S.PIPELINES[args.pipeline]
        capture = {
            "pipeline": args.pipeline,
            "dtype": "bf16",
            "k": pipe.k,
            "operand_bits": 16,
            "note": "acc/d are FP32 bit patterns (hex); a/b are 16 BF16 words (hex) as measured on the device in identity",
            "identity": identity,
            "records": [],
        }
        fams: dict = {}
        with mp.Pool(args.procs, initializer=_init, initargs=(args.pipeline,)) as pool:
            for name, A, Bt, C in families(rng, args.n_random):
                if only is not None and name not in only:
                    continue
                t0 = time.time()
                D = probe.mma_tiles(A, Bt, C)
                t_gpu = time.time() - t0
                t0 = time.time()
                model, err = model_eval(pool, A, Bt, C, args.procs)
                t_model = time.time() - t0
                mism = (model != D) | err
                fam = {
                    "tiles": int(A.shape[0]),
                    "elements": int(A.shape[0]) * 128,
                    "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                    "mismatches_vs_model": int(mism.sum()),
                    "model_domain_errors": int(err.sum()),
                    "gpu_seconds": round(t_gpu, 2),
                    "model_seconds": round(t_model, 1),
                    "first_mismatches": [],
                }
                for t, i, j in list(zip(*np.nonzero(mism)))[:5]:
                    rec = hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j])
                    rec["model"] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                    fam["first_mismatches"].append(rec)
                np.savez_compressed(out / f"tiles_{name}.npz", A=A, Bt=Bt, C=C, D=D)
                flat = rng.choice(A.shape[0] * 128, size=min(args.golden_per_family, A.shape[0] * 128), replace=False)
                for f in sorted(flat.tolist()):
                    t, i, j = f // 128, (f % 128) // 8, f % 8
                    capture["records"].append({**hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j]), "family": name})
                fams[name] = fam
                print(name, json.dumps({k: v for k, v in fam.items() if k != "first_mismatches"}), flush=True)
        results["sweep"] = {
            "seed": args.seed,
            "n_random": args.n_random,
            "families": fams,
            "total_elements": sum(f["elements"] for f in fams.values()),
            "total_nonzero_accumulators": sum(f["nonzero_accumulators"] for f in fams.values()),
            "total_mismatches_vs_model": sum(f["mismatches_vs_model"] for f in fams.values()),
        }
        (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
        print("sweep TOTAL", results["sweep"]["total_elements"], "elements,", results["sweep"]["total_mismatches_vs_model"], "mismatches", flush=True)

    results["seconds"] = round(time.time() - t_all, 1)
    failures = []
    if "golden_replay" in results and (results["golden_replay"]["mismatches"] or results["golden_replay"]["other_positions_nonzero"]):
        failures.append(f"golden_replay mismatches={results['golden_replay']['mismatches']}")
    if "sweep" in results and results["sweep"]["total_mismatches_vs_model"]:
        failures.append(f"sweep mismatches={results['sweep']['total_mismatches_vs_model']}")
    results["failures"] = failures
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        result = {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "workload_fingerprint": {"probe": "tools/tc_probe/tc_probe.py", "arch": args.arch, "device": identity["device"],
                                     "seed": args.seed, "n_random": args.n_random, "pipeline": args.pipeline},
            "validation": {"status": "passed" if not failures else "failed", "checker": "tc_probe vs silicon.tc_dot",
                           "detail": "; ".join(failures) or "every hardware word reproduced", "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": ["capture_sample.json"] if args.sweep else [],
        }
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out / "result.json")
    print("FAILURES" if failures else "ALL HARDWARE WORDS REPRODUCED", failures, f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
