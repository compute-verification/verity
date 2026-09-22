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
hardware word to equal the record's ``d``.  ``--replay-instances DIR`` replays ``bench-instances/v1`` (the ``tu-k16`` tier as one
launch, ``vu-k1536`` as 96 dependent launches with the hardware's accumulators carried) against the dataset's expected words, after
checking the arrays' sha256 against the pinned manifest (``--instances-manifest``).  ``--sweep`` runs the finite-sweep families (random, random nonzero accumulators, uniform
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
import shutil
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
    nvcc_bin = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"  # the CUDA images do not put it on a non-login PATH
    nvcc = subprocess.run([nvcc_bin, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    subprocess.run(
        [nvcc_bin, "-O2", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / "mma_tiles.cu")],
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

    # 10. (WP1 addition) +-0 accumulators with cancelling products below 2^-145: BF16 subnormal a (30% of lanes), b in
    # [2^-27, 2^-5].  Decides whether a zero accumulator's exponent enters the alignment maximum (silicon and the simulator's
    # E_ZERO zero: no; a zero carried at -126/-127 would truncate these sums differently -- see the log).
    n = max(256, n_random // 16)
    a = ((rng.integers(0, 2, (n, 16, K)) << 15) | rng.integers(1, 128, (n, 16, K))).astype(np.uint16)
    a = np.where(rng.random((n, 16, K)) < 0.7, np.uint16(0), a)
    b = ((rng.integers(0, 2, (n, 8, K)) << 15) | (rng.integers(100, 123, (n, 8, K)) << 7) | rng.integers(0, 128, (n, 8, K))).astype(np.uint16)
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "zero_acc_tiny_cancellation", a, b, acc


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


def model_eval(pool, A, Bt, C, procs, worker=_eval_tiles):
    n = A.shape[0]
    chunks = max(1, min(n // 64 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(worker, [(A[i], Bt[i], C[i]) for i in idx])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _eval_zero_alternatives(args):
    """For zero accumulators: the word the source-close evaluator gives when the zero is carried at exponent -126 / -127 instead of
    E_ZERO (the two arrays returned).  Elements with a nonzero accumulator get the ordinary result in both."""

    from verity_numerical.reference import hawkeye_close as H

    A, Bt, C = args
    n = A.shape[0]
    out = [np.zeros((n, 16, 8), np.uint32) for _ in range(2)]
    for t in range(n):
        for i in range(16):
            for j in range(8):
                acc_bits = int(C[t, i, j])
                for k, zero_exp in enumerate((-126, -127)):
                    if acc_bits & 0x7FFFFFFF:
                        g = H.fp32_to_gfloat(acc_bits)
                    else:
                        g = H.Gfloat(bool(acc_bits >> 31), zero_exp, 0)
                    for start in (0, 8):
                        g = H.group_sum([g] + [H.multiply_bfloat16_to_gfloat(int(A[t, i, l]), int(Bt[t, j, l])) for l in range(start, start + 8)])
                    out[k][t, i, j] = H.gfloat_to_float_bits(g)
    return out[0], out[1]


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


def special_all_finite_records(path: Path) -> list[dict]:
    """The all-finite records (finite acc, finite a and b words) of a ``tc-total-2026-09-07/<device>/mma_special.reduced.json.gz``
    file, in the golden-record shape: the ``cases`` list and the ``pos`` list (one measured word each)."""

    import gzip

    data = json.loads(gzip.open(path).read())
    assert data["instruction"] == "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    out = []
    for rec in data["cases"]:
        out.append({"acc": rec["acc"], "a": rec["a"], "b": rec["b"], "d": rec["d"], "family": f"cases/{rec['family']}"})
    for rec in data["pos"]:
        assert len(rec["d"]) == 1, rec["name"]
        out.append({"acc": rec["acc"], "a": rec["a"], "b": rec["b"], "d": rec["d"][0], "family": "pos"})
    finite = []
    for rec in out:
        if (int(rec["acc"], 16) >> 23) & 0xFF == 0xFF:
            continue
        if any((w >> 7) & 0xFF == 0xFF for w in (*words(rec["a"]), *words(rec["b"]))):
            continue
        finite.append(rec)
    return finite


def replay_golden(probe: Probe, path: Path, records: list[dict] | None = None, pipeline: str | None = None) -> dict:
    """Every record at its own tile position (r % 16, (r // 16) % 8), the rest of the tile zero."""

    if records is None:
        data = json.loads(path.read_text())
        records, pipeline = data["records"], data["pipeline"]
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
        "pipeline": pipeline,
        "records": n,
        "nonzero_accumulators": int(sum(1 for rec in records if int(rec["acc"], 16) & 0x7FFFFFFF)),
        "mismatches": len(mism),
        "first_mismatches": mism[:5],
        "other_positions_nonzero": other_nonzero,
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_instances(probe: Probe, root: Path, manifest_path: Path | None = None) -> dict:
    """Replay ``bench-instances/v1`` on the device: the ``tu-k16`` tier as one launch (instance ``r`` at tile ``r``, position
    ``(r % 16, (r // 16) % 8)``, the rest of the tile zero) and the ``vu-k1536`` tier as 96 dependent launches (all instances per
    launch; the hardware's D of step ``s`` is the C of step ``s + 1``).  Every outgoing accumulator word is compared with the
    dataset's ``tu-k16.out.u32`` / ``vu-k1536.acc.u32``; the final accumulator, cast FP32 -> BF16 RNE on the CPU (the epilogue is
    not a tensor-core operation), with ``vu-k1536.y.u16``.  Before anything runs, every array's sha256 is checked against
    ``manifest_path`` (default ``<root>/manifest.json``): a replay of arrays other than the pinned ones is refused."""

    from verity.ml.tc.cast import f32_to_bf16_word
    from verity_numerical.bench import instances as I

    manifest_path = manifest_path or root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digests = {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "checked": 0, "mismatched": []}
    for tier in ("tu-k16", "vu-k1536"):
        for name, rec in manifest["tiers"][tier]["files"].items():
            digests["checked"] += 1
            if _sha256(root / name) != rec["sha256"]:
                digests["mismatched"].append(name)
    if digests["mismatched"]:
        raise RuntimeError(f"instance arrays under {root} differ from {manifest_path}: {digests['mismatched']}")

    report: dict = {"dataset": manifest["dataset"], "root": str(root), "digests": digests}

    # tu-k16: one launch
    tu = I.load("tu-k16", root)
    n = tu.n
    A = np.zeros((n, 16, K), np.uint16)
    Bt = np.zeros((n, 8, K), np.uint16)
    C = np.zeros((n, 16, 8), np.uint32)
    a = np.frombuffer(tu.a.tobytes(), dtype=np.uint16).reshape(n, K)
    b = np.frombuffer(tu.b.tobytes(), dtype=np.uint16).reshape(n, K)
    c = np.frombuffer(tu.c.tobytes(), dtype=np.uint32)
    want = np.frombuffer(tu.out.tobytes(), dtype=np.uint32)
    rows = np.arange(n) % 16
    cols = (np.arange(n) // 16) % 8
    A[np.arange(n), rows] = a
    Bt[np.arange(n), cols] = b
    C[np.arange(n), rows, cols] = c
    t0 = time.time()
    D = probe.mma_tiles(A, Bt, C)
    got = D[np.arange(n), rows, cols]
    rest = D.copy()
    rest[np.arange(n), rows, cols] = 0
    bad = np.nonzero(got != want)[0]
    report["tu-k16"] = {
        "instances": int(n),
        "mismatches": int(bad.size),
        "other_positions_nonzero": int(np.count_nonzero(rest)),
        "first_mismatches": [{"id": int(r), **hexrec(c[r], a[r], b[r], got[r]), "expected": f"{int(want[r]):08x}"} for r in bad[:5]],
        "gpu_seconds": round(time.time() - t0, 2),
    }
    print("replay tu-k16", json.dumps({k: v for k, v in report["tu-k16"].items() if k != "first_mismatches"}), flush=True)

    # vu-k1536: 96 dependent launches
    vu = I.load("vu-k1536", root)
    n, steps = vu.n, vu.K // K
    x = np.frombuffer(vu.x.tobytes(), dtype=np.uint16).reshape(n, steps, K)
    w = np.frombuffer(vu.w.tobytes(), dtype=np.uint16).reshape(n, steps, K)
    acc = np.frombuffer(vu.acc.tobytes(), dtype=np.uint32).reshape(n, steps)
    y = np.frombuffer(vu.y.tobytes(), dtype=np.uint16)
    rows = np.arange(n) % 16
    cols = (np.arange(n) // 16) % 8
    carried = np.zeros(n, np.uint32)  # c_0 = +0
    per_step = []
    first = []
    other_nonzero = 0
    t0 = time.time()
    for s in range(steps):
        A = np.zeros((n, 16, K), np.uint16)
        Bt = np.zeros((n, 8, K), np.uint16)
        C = np.zeros((n, 16, 8), np.uint32)
        A[np.arange(n), rows] = x[:, s]
        Bt[np.arange(n), cols] = w[:, s]
        C[np.arange(n), rows, cols] = carried
        D = probe.mma_tiles(A, Bt, C)
        got = D[np.arange(n), rows, cols]
        rest = D.copy()
        rest[np.arange(n), rows, cols] = 0
        other_nonzero += int(np.count_nonzero(rest))
        bad = np.nonzero(got != acc[:, s])[0]
        per_step.append(int(bad.size))
        for r in bad[: max(0, 5 - len(first))]:
            first.append({"id": int(r), "step": s, **hexrec(carried[r], x[r, s], w[r, s], got[r]), "expected": f"{int(acc[r, s]):08x}"})
        carried = got.copy()  # the hardware's own accumulator carries on (mismatches propagate, as they would in a real chain)
    y_hw = np.array([f32_to_bf16_word(int(v)) for v in carried], dtype=np.uint16)
    y_bad = np.nonzero(y_hw != y)[0]
    report["vu-k1536"] = {
        "instances": int(n),
        "steps": int(steps),
        "accumulator_words_compared": int(n * steps),
        "accumulator_mismatches": int(sum(per_step)),
        "steps_with_mismatches": {str(s): m for s, m in enumerate(per_step) if m},
        "y_compared": int(n),
        "y_mismatches": int(y_bad.size),
        "first_y_mismatches": [{"id": int(r), "hardware": f"{int(y_hw[r]):04x}", "expected": f"{int(y[r]):04x}"} for r in y_bad[:5]],
        "other_positions_nonzero": other_nonzero,
        "first_mismatches": first,
        "epilogue": "f32_to_bf16_word (RNE) applied on the CPU to the hardware's final accumulator",
        "gpu_seconds": round(time.time() - t0, 2),
    }
    print("replay vu-k1536", json.dumps({k: v for k, v in report["vu-k1536"].items() if k not in ("first_mismatches", "first_y_mismatches")}), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="sm_80 (A100) or sm_89 (RTX 4090)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", default="ampere_bf16_m16n8k16")
    ap.add_argument("--replay-golden", default=None, metavar="JSON")
    ap.add_argument("--replay-special", default=None, metavar="JSON.GZ", help="mma_special.reduced.json.gz: its all-finite records")
    ap.add_argument("--replay-instances", default=None, metavar="DIR", help="bench-instances/v1 directory with the built arrays: replay tu-k16 and vu-k1536")
    ap.add_argument("--instances-manifest", default=None, metavar="JSON", help="the pinned manifest to check the arrays against (default DIR/manifest.json)")
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

    if args.replay_special:
        path = Path(args.replay_special)
        results["special_replay"] = replay_golden(probe, path, special_all_finite_records(path), args.pipeline)
        print("special_replay", json.dumps({k: v for k, v in results["special_replay"].items() if k != "first_mismatches"}), flush=True)

    if args.replay_instances:
        results["instances_replay"] = replay_instances(probe, Path(args.replay_instances),
                                                       Path(args.instances_manifest) if args.instances_manifest else None)

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
                if name == "zero_acc_tiny_cancellation":
                    alt126, alt127 = model_eval(pool, A, Bt, C, args.procs, worker=_eval_zero_alternatives)
                    fam["zero_exponent_probe"] = {
                        "elements_where_a_zero_at_-126_differs_from_model": int(np.count_nonzero(alt126 != model)),
                        "hardware_eq_zero_at_-126_where_it_differs": int(np.count_nonzero((alt126 != model) & (alt126 == D))),
                        "elements_where_a_zero_at_-127_differs_from_model": int(np.count_nonzero(alt127 != model)),
                        "hardware_eq_zero_at_-127_where_it_differs": int(np.count_nonzero((alt127 != model) & (alt127 == D))),
                    }
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
    for replay in ("golden_replay", "special_replay"):
        if replay in results and (results[replay]["mismatches"] or results[replay]["other_positions_nonzero"]):
            failures.append(f"{replay} mismatches={results[replay]['mismatches']} other_positions_nonzero={results[replay]['other_positions_nonzero']}")
    if "sweep" in results and results["sweep"]["total_mismatches_vs_model"]:
        failures.append(f"sweep mismatches={results['sweep']['total_mismatches_vs_model']}")
    if "instances_replay" in results:
        ir = results["instances_replay"]
        tu, vu = ir["tu-k16"], ir["vu-k1536"]
        if tu["mismatches"] or tu["other_positions_nonzero"]:
            failures.append(f"instances tu-k16 mismatches={tu['mismatches']} other_positions_nonzero={tu['other_positions_nonzero']}")
        if vu["accumulator_mismatches"] or vu["y_mismatches"] or vu["other_positions_nonzero"]:
            failures.append(f"instances vu-k1536 accumulator_mismatches={vu['accumulator_mismatches']} y_mismatches={vu['y_mismatches']} "
                            f"other_positions_nonzero={vu['other_positions_nonzero']}")
    results["failures"] = failures
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        fingerprint = {"probe": "tools/tc_probe/tc_probe.py", "arch": args.arch, "device": identity["device"],
                       "seed": args.seed, "n_random": args.n_random, "pipeline": args.pipeline}
        if "instances_replay" in results:
            fingerprint["instances"] = {"dataset": results["instances_replay"]["dataset"], "manifest_sha256": results["instances_replay"]["digests"]["manifest_sha256"]}
        result = {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "workload_fingerprint": fingerprint,
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
