"""PoUW alone on fixed matmul shapes, on one RTX 4090 (W1): `ncp-v2`'s fused prover against the fastest plain GEMM of the
same shape, with hashing included and reported separately; γ_0 and Ω* from the scheme's certificate; samples per bit with
the verifier's cost per draw. No served model.

    python benchmarks/pouw/pouw_bench.py --dry-run --out DRY.json    (CPU: everything below that needs no GPU)
    python benchmarks/pouw/pouw_bench.py --out RESULT.json           (the pod: gates, then timings; honest calls only)
    python benchmarks/pouw/pouw_bench.py --tile-variants 'rows=8,unroll=2,sha=1;rows=0' --out RESULT.json

**Shapes** (`pouw_bench_shapes.json`): ``baseline``, Qwen2.5-0.5B's four linears at 16, 32, 256 and 2,048 rows, the model
#389 and #435 measured through the served pipeline; ``census``, the GEMMs of census workloads (`census/workloads.json`, by
id) at 16 and 2,048 rows; ``square``, 4,096³ and 8,192³.

**Variants** are builds of the fused kernel by `ncp2_tiles.cuh`'s macros, in `ncp2_gpu_bench --tile-variants`'s form (S2's
list runs as given there); a variant's `-D` flags go on both its nvcc build and the CPU twin. The default is
``rows=0,sha=0``, the tile layout before change 3 (every running sum in shared memory, nvcc's SHA-256 round), which is the
baseline, then change 3's default. Change 2's tree pass is in every build since #464, and change 1 is the ``async`` arm.

**Gates**, each stopping the run at its first difference:
- per variant, the CPU twin built with the variant's flags against `pouw_native`, on `pouw_device.gate_calls` with the
  baseline linears and one 16-row call per other (k, n) timed; on the pod, the device's hashes against hashlib, the fused
  kernel against `pouw_native` on the same calls, and change 1's async path;
- before each shape is timed, every prover arm's flags at that shape, on the first of the operand sets it is timed on
  (`gate_shape`): the fused kernel's root, every leaf, the weight side's Y leaves and y against the independent device build
  (`ncp2_kernels.cu`, by `ncp2_gpu_bench.gate_timed`), with units drawn from the root against `pouw_native`, twice for the
  cached Y leaves (a miss, then a hit); the timed entry point's root; ``nohash``'s y; and change 1's served y and each root
  after the drain. Every operand set of every shape has its own weight id, so no Y-leaf cache entry is shared;
- after a shape is timed, a consistency gate (`consistency`): where the plain int8 GEMM is compute-bound, a ``nohash`` call,
  which does W_ref/mkn plain products' MACs, takes at least that many plain int8 GEMMs (`EFFICIENCY_FLOOR`), or work was
  skipped.

**Arms, per shape and variant** (honest calls only; no tamper arm, and nothing whose γ is unproved):
- the plain GEMM of the same (m, k, n): ``int8``, cuBLASLt ``torch._int_mm`` (m padded to 32 rows where it needs more than
  16); ``serve``, the int8 serving path (the same product and `ncp2_dequant`, as change 1 serves y); ``fp8``, cuBLASLt
  ``torch._scaled_mm`` (e4m3 in, FP32 accumulate, BF16 out), which runs at half the int8 rate on a GeForce RTX 4090;
  ``bf16``, ``torch.mm``. Each is also autotuned (``*_tuned``, `--autotune inductor`: Inductor's max-autotune, ATen's
  cuBLASLt call against its Triton templates, timed on the device, its pick recorded). The plain path is the fastest of
  serve, FP8 and BF16, each at the better of its default and its autotuned call.
- ``nohash``: the fused kernel with every hash and XOF left out and a stand-in noise, the arithmetic alone.
- ``sync``: the fused kernel committing every leaf (`pouw_device.FusedCommitter`), the prover's time per call.
- ``async``: change 1 (`pouw_device.AsyncCommitter`): y served from the exact int8 product, the commit on side streams.

**Timing**, every arm the same way (`Clock`): bursts of the same number of calls between CUDA events, cycling through the
shape's operand sets (distinct weights and activations, past twice the L2 in all), each burst after a read of 4× the L2
(the buffer zeroed once, so the flush leaves no dirty lines; `flush_control` also times int8 and serve after a write
flush), the arms' order rotated each rep. ``async``'s burst ends when its last commit on the side streams does. An arm's
figure is its median per-call time over the reps. Every rep is saved, with the device's clocks, temperature, power and
throttle reasons before and after it, beside the versions (torch, CUDA, cuBLASLt, Triton, driver, nvcc).

**Reported:** the prover's overhead, ``sync`` over the plain path; hashing and XOF (incremental), reported separately,
``sync`` less ``nohash``; ``async`` over the plain path; and each over int8, serve and FP8. Beside them, W1's W_ref/mkn,
the slowdown the certificate charges against the plain product, and Ω = W_ref / the credited 3·m·k·n with γ = 1 − (1 − γ_0)/Ω
where the shape is in the scheme's domain, beside the certificate's Ω* and γ over the domain (`verity_pouw.schemes.ncp.NCP`);
those are derived from the certificate, not measured. **Samples per bit**, our definition until the measurement plan is
readable: work-weighted sampling gives δ_s = (1 − ε_s)^t (Lean `workWeightedSampling`), so a bit of soundness costs
1/−log₂(1 − ε_s) draws; with the verifier's cost per draw at each shape (`verifier_draw`, `opened_bytes`), each bit's CPU
seconds and opened bytes. The ``forwards`` are synthetic: the baseline linears' per-call medians summed over 24 layers.
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import ncp2_gpu_bench as G  # noqa: E402
from ncp2_gpu_bench import TILE_MACROS, absorbed, call_accounting  # noqa: E402
from verity_pouw.protocol import DomainError, Matmul, Shape  # noqa: E402
from verity_pouw.schemes import ncp  # noqa: E402
from verity_vllm.protocol_options import pouw_circuit as PC  # noqa: E402
from verity_vllm.protocol_options import pouw_device as PD  # noqa: E402
from verity_vllm.protocol_options import pouw_native as PN  # noqa: E402

HERE = Path(__file__).parent
SHAPES = HERE / "pouw_bench_shapes.json"
CENSUS = HERE.parents[1] / "census" / "workloads.json"
SCHEME = "ncp-v2"
SALT = bytes(range(32))
D = 16

#: the tile layout before change 3 (every running sum in shared memory, nvcc's SHA-256 round), then change 3's default, in
#: `ncp2_gpu_bench --tile-variants`'s form; S2's list runs as given there
VARIANTS = "rows=0,sha=0;rows=8,unroll=2,sha=1"
#: detection fractions ε_s the samples-per-bit table is given at, and the soundness it is sized for
EPSILONS = (0.001, 0.005, 0.01, 0.05)
LAMBDA = 128

# ---- shapes ------------------------------------------------------------------------------------------------------------------


def census_gemms(workloads: list[dict], wid: str) -> list[tuple[int, int]]:
    """A census workload's distinct GEMM (K, N), from its Gemm_v1 and Gemm_v2 templates, in the census's order."""
    rows = [w for w in workloads if w["id"] == wid]
    if not rows:
        raise KeyError(f"no census workload {wid!r}")
    out: list[tuple[int, int]] = []
    for t in rows[0]["template_mix"]:
        if t["vu_family"] in ("Gemm_v1", "Gemm_v2"):
            kn = (int(t["parameters"]["K"]), int(t["parameters"]["N"]))
            if kn not in out:
                out.append(kn)
    return out


def bench_shapes(sets: list[str], spec: dict, workloads: list[dict] | None) -> list[dict]:
    """The selected sets' shapes: each with its set, (m, k, n) and where it comes from."""
    out: list[dict] = []
    for name in sets:
        s = spec["sets"][name]
        if name == "census":
            for wid in s["workloads"]:
                for k, n in census_gemms(workloads or [], wid):
                    if n <= s["max_n"]:
                        out += [{"set": name, "m": m, "k": k, "n": n, "source": wid} for m in s["rows"]]
        elif "mkn" in s:
            out += [{"set": name, "m": m, "k": k, "n": n, "source": s["why"]} for m, k, n in s["mkn"]]
        else:
            out += [{"set": name, "m": m, "k": k, "n": n, "source": s["source"]} for m in s["rows"] for k, n in s["shapes"]]
    return out


def gate_shapes(shapes: list[dict], spec: dict) -> list[tuple[int, int]]:
    """The (k, n) timed beyond the baseline linears, which `pouw_device.gate_calls` already covers: one 16-row gate call each."""
    covered = {tuple(kn) for kn in spec["sets"]["baseline"]["shapes"]}
    return sorted({(s["k"], s["n"]) for s in shapes} - covered)


def gate_calls(spec: dict, extra: list[tuple[int, int]]) -> list[tuple[str, object]]:
    calls = PD.gate_calls(tuple(tuple(kn) for kn in spec["sets"]["baseline"]["shapes"]))
    calls += [(f"bench k={k} n={n}", PD.gate_call(f"bench k={k} n={n}", D, k, n, layer=7, site=i, weight=500 + i))
              for i, (k, n) in enumerate(extra)]
    return calls


# ---- W1, the certificate and sampling ---------------------------------------------------------------------------------------


def certificate() -> dict:
    c = ncp.NCP().certificate
    return {"conjecture": c.conjecture, "gamma0": str(c.gamma0), "omega_star": str(c.omega), "gamma": float(c.gamma),
            "parameters": dict(c.parameters), "hypotheses": list(c.hypotheses)}


def w1(m: int, k: int, n: int) -> dict:
    """W_ref / m·k·n, the slowdown W1 (route U on the 4090) charges against the plain product; Ω = W_ref / the credited
    3·m·k·n; and γ = 1 − (1 − γ_0)/Ω where the shape is in the domain. Derived from the certificate, not measured."""
    s = ncp.NCP()
    shape = Shape(m, k, n)
    wref = Fraction(s.wref(shape), shape.wmm)
    omega = wref / 3
    out = {"w1_x_mkn": float(wref), "omega": float(omega), "hashed_bytes_per_mac": hashed_bytes_per_mac(m, k, n)}
    try:
        s.check(Matmul(1, shape, 2))
    except DomainError as e:
        return out | {"gamma": None, "outside": str(e)}
    return out | {"gamma": float(1 - (1 - s.certificate.gamma0) / omega)}


#: a call's own messages (`ncp2_gpu_bench.call_accounting`); the Y strips are the weight side's, committed once per run
CALL_MESSAGES = ("tile words", "dequant words", "form_x strips", "units tree")


def hashed_bytes_per_mac(m: int, k: int, n: int) -> float:
    """Bytes SHA-256 absorbs for one call's leaves and nodes, per useful MAC."""
    acc = call_accounting(m, k, n)
    return sum(acc["messages"][p][0] * absorbed("sha256", acc["messages"][p][1]) for p in CALL_MESSAGES) / acc["macs"]


def samples_per_bit(eps: float) -> float:
    """Work-weighted draws per bit of soundness at detection fraction ``eps``: δ_s = (1 − ε_s)^t."""
    return 1 / -math.log2(1 - eps)


SAMPLING_DEFINITION = (
    "ours, until the measurement plan (docs/protocol-measurement-plan.md) is readable: draws per bit of soundness under "
    "work-weighted draws with replacement, δ_s = (1 − ε_s)^t, so 1/−log₂(1 − ε_s); a draw's cost is `pouw_native` "
    "recomputing one drawn tile on the CPU (its X and Y strips, checkpoints and leaf, a fresh verifier each draw) and the "
    "bytes it opens (the tile's 16 A rows and 16 B rows with their scales and ids, its leaf and its path in the units tree; "
    "the inputs' own openings not counted); per bit is draws per bit times either")


def draws() -> list[dict]:
    return [{"eps_s": e, "samples_per_bit": samples_per_bit(e), f"draws_for_{LAMBDA}_bits": math.ceil(LAMBDA * samples_per_bit(e))}
            for e in EPSILONS]


def verifier_draw(k: int, n: int, count: int = 3) -> dict:
    """The verifier's CPU time for one drawn tile of a 16-row call at (k, n) (a tile's cost does not depend on m), each draw
    by a fresh `pouw_native` verifier, so no strip it formed for another draw is reused: the median of ``count``."""
    call = PD.gate_call(f"draw k={k} n={n}", D, k, n, layer=5, site=0, weight=700)
    times = []
    for j in range(count):
        t0 = time.perf_counter()
        nat = PN.Native(call, SCHEME)
        PC.unit_leaf(nat.outputs(nat.index("tile", 0, (7919 * j) % (n // D))))
        times.append(time.perf_counter() - t0)
    return {"k": k, "n": n, "tile_s": sorted(times)[len(times) // 2], "raw_s": times}


def opened_bytes(m: int, k: int, n: int) -> int:
    """A drawn tile's opening: its 16 A rows and 16 B rows (k bytes each) with their scales and ids (4 bytes each), its leaf,
    and its path in the call's units tree."""
    ms, ns = m // D, n // D
    population = 3 * ms + ms * ns + ms * ns * D * D
    return 2 * D * k + 4 * D * 4 + 32 + 32 * math.ceil(math.log2(population))


def sampling(shapes: list[dict]) -> dict:
    """Draws per bit at each ε_s, and at each shape the verifier's cost per draw and per bit (`SAMPLING_DEFINITION`)."""
    cost = {kn: verifier_draw(*kn) for kn in sorted({(s["k"], s["n"]) for s in shapes})}
    per_shape = []
    for s in shapes:
        c, opened = cost[(s["k"], s["n"])], opened_bytes(s["m"], s["k"], s["n"])
        per_shape.append({"m": s["m"], "k": s["k"], "n": s["n"], "tile_s": c["tile_s"], "opened_bytes": opened,
                          "per_bit": [{"eps_s": e, "verifier_s": samples_per_bit(e) * c["tile_s"],
                                       "opened_bytes": samples_per_bit(e) * opened} for e in EPSILONS]})
    return {"definition": SAMPLING_DEFINITION, "cpu": _cpu_name(), "draws": draws(), "verifier_draws": list(cost.values()),
            "shapes": per_shape}


# ---- variants -------------------------------------------------------------------------------------------------------------


def parse_variants(spec: str) -> list[tuple[str, tuple[str, ...]]]:
    """``ncp2_gpu_bench --tile-variants``'s form: builds separated by ``;``, each like ``rows=8,unroll=2,sha=1`` (a macro left
    out keeps the header's default), each named by its own text, with its `-D` flags."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        fields = {}
        for kv in [kv for kv in part.split(",") if kv]:
            key, _, value = kv.partition("=")
            if key not in TILE_MACROS or not value.isdigit():
                raise KeyError(f"{kv!r}: a variant sets {', '.join(TILE_MACROS)} to integers")
            fields[key] = int(value)
        name = ",".join(f"{k}={v}" for k, v in fields.items())
        if name not in (n for n, _f in out):
            out.append((name, tuple(f"-D{TILE_MACROS[k]}={v}" for k, v in sorted(fields.items()))))
    return out


BASE_CFLAGS, BASE_NVCC_FLAGS = PD.CFLAGS, PD.NVCC_FLAGS


def use(flags: tuple[str, ...]) -> None:
    """Build every later host, device and fused library with ``flags``: the flags are part of each build's key."""
    PD.CFLAGS, PD.NVCC_FLAGS = BASE_CFLAGS + flags, BASE_NVCC_FLAGS + flags
    PD.configure()


def flag(flags: tuple[str, ...], name: str, default: int) -> int:
    for f in flags:
        if f.startswith(f"-D{name}="):
            return int(f.split("=", 1)[1])
    return default


def tile_block(flags: tuple[str, ...]) -> dict:
    """The committed tile kernel's block (`ncp2_tiles.cuh` `TileShape`): 64 lanes with half the running sums in registers,
    else 32; 32 KB of shared memory either way."""
    reg = flag(flags, "NCP2_TILE_REG_ROWS", 8)
    threads = 64 if reg else 32
    return {"reg_rows": reg, "threads": threads, "smem": threads * (D - reg) * D * 4}


#: sm_89 (RTX 4090): registers, their allocation unit per warp, shared memory and the 1 KB each block reserves, warps and
#: blocks per SM
SM89 = {"regs": 65536, "reg_unit": 256, "smem": 100 * 1024, "smem_reserved": 1024, "warps": 48, "blocks": 24}


def occupancy(registers: int, threads: int, smem: int, sm: dict = SM89) -> dict:
    """Resident blocks and warps per SM from the kernel's registers, block size and dynamic shared memory."""
    warps = -(-threads // 32)
    per_warp = -(-registers * 32 // sm["reg_unit"]) * sm["reg_unit"]
    limits = {"registers": sm["regs"] // per_warp // warps, "shared memory": sm["smem"] // (smem + sm["smem_reserved"]),
              "warps": sm["warps"] // warps, "blocks": sm["blocks"]}
    blocks = min(limits.values())
    return {"blocks": blocks, "warps": blocks * warps, "chains": blocks * threads,
            "limited_by": [k for k, v in limits.items() if v == blocks]}


PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
PTXAS_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
PTXAS_REGS = re.compile(r"Used (\d+) registers")
TILES = re.compile(r"_ZN4ncp28kf_tilesILi(\d)EE")


def ptxas_usage(text: str) -> dict:
    """`nvcc --resource-usage`'s report for each tile kernel (``kf_tiles<LEAF>``): registers, stack and spills."""
    out, name = {}, None
    for line in text.splitlines():
        if m := PTXAS_ENTRY.search(line):
            t = TILES.match(m.group(1))
            name = f"kf_tiles<{t.group(1)}>" if t else None
        elif name and (m := PTXAS_FRAME.search(line)):
            out[name] = {"stack": int(m.group(1)), "spill_stores": int(m.group(2)), "spill_loads": int(m.group(3))}
        elif name and (m := PTXAS_REGS.search(line)):
            out[name]["registers"] = int(m.group(1))
            name = None
    return out


def resource_usage(flags: tuple[str, ...]) -> dict | None:
    """The fused build's tile kernels under ``flags``, from ptxas (a cubin, not the library); None without nvcc."""
    if PD.nvcc() is None:
        return None
    cuda = [f for f in PD.cuda_flags() if f not in ("-shared", "-Xcompiler", "-fPIC")]
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([PD.nvcc(), *cuda, *flags, "--resource-usage", "-cubin", "-o", str(Path(tmp) / "fused.cubin"),
                            str(PD.HERE / "ncp2_fused.cu")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"nvcc {' '.join(flags)}: {r.stderr[-2000:]}")
    usage = ptxas_usage(r.stdout + r.stderr)
    block = tile_block(flags)
    for leaf in ("kf_tiles<0>", "kf_tiles<1>"):
        if leaf in usage:
            usage[leaf] |= {"block": block["threads"], "smem": block["smem"]} | occupancy(usage[leaf]["registers"], block["threads"],
                                                                                          block["smem"])
    if usage.get("kf_tiles<1>", {}).get("spill_stores"):
        raise SystemExit(f"variant {' '.join(flags)}: the committed tile kernel spills ({usage['kf_tiles<1>']})")
    return usage


# ---- gates ------------------------------------------------------------------------------------------------------------------


def twin_gate(calls, refs: dict) -> list[dict]:
    """The CPU twin (the fused kernel's functions on the host, built with the current flags) against `pouw_native`."""
    out = []
    for name, call in calls:
        diff = PD.first_difference(refs[name], PD.commit_call(0, SCHEME, call, on="twin"))
        out.append({"call": name, "difference": diff})
        if diff:
            raise SystemExit(f"twin gate: {name}: {diff}")
    return out


def device_gate(calls, refs: dict) -> list[dict]:
    """The fused kernel against `pouw_native`, leaf for leaf, root and served y."""
    out = []
    for name, call in calls:
        got, ms = PD.commit_call_timed(0, SCHEME, call, on="fused")
        diff = PD.first_difference(refs[name], got)
        out.append({"call": name, "difference": diff, "device_ms": round(ms, 3)})
        if diff:
            raise SystemExit(f"fused gate: {name}: {diff}")
    return out


# ---- the #389 path on the CPU -------------------------------------------------------------------------------------------------


def cpu_commit(spec: dict, rows: int = D) -> dict:
    """`pouw_native`'s commit of each baseline linear at ``rows`` rows (the #389 path: the committer its Commit ran), its
    weight side committed first as a run's first call does; and the C++ twin's. Per call and per forward."""
    layers = spec["sets"]["baseline"]["layers"]
    out: dict = {"cpu": _cpu_name(), "threads": os.cpu_count(), "rows": rows, "calls": []}
    for i, (k, n) in enumerate(spec["sets"]["baseline"]["shapes"]):
        call = PD.gate_call(f"cpu k={k} n={n}", rows, k, n, layer=3, site=i, weight=900 + i)
        t0 = time.perf_counter()
        side = PN.commit_weight(SCHEME, call)
        t1 = time.perf_counter()
        PN.commit_call(0, SCHEME, call, side=side)
        t2 = time.perf_counter()
        PD.commit_call(0, SCHEME, call, on="twin")
        t3 = time.perf_counter()
        out["calls"].append({"k": k, "n": n, "native_weight_s": t1 - t0, "native_call_s": t2 - t1, "twin_call_s": t3 - t2})
    out["forward_native_s"] = layers * sum(c["native_call_s"] for c in out["calls"])
    # #389 measured before the port, when each call formed and hashed its weight's Y: that is the call and its side together
    out["forward_native_with_weight_s"] = layers * sum(c["native_call_s"] + c["native_weight_s"] for c in out["calls"])
    out["forward_twin_s"] = layers * sum(c["twin_call_s"] for c in out["calls"])
    return out


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ---- derived metrics ----------------------------------------------------------------------------------------------------------


PLAIN = ("int8", "serve", "fp8", "bf16")
#: the plain paths a deployment would serve the product by; int8 alone lacks the dequantization the prover's y includes
SERVED = ("serve", "fp8", "bf16")


def derive(row: dict) -> dict:
    """From one shape's medians: each plain GEMM at the better of its default and autotuned call; the plain path, the fastest
    of `SERVED`; the prover (``sync``), the same kernel with nothing hashed (``nohash``), hashing and XOF alone (``sync``
    less ``nohash``) and change 1 (``async``), each over the plain path and over int8, serve and FP8."""
    out: dict = {}
    for kind in PLAIN:
        got = {k: row[f"{k}_ms"] for k in (kind, f"{kind}_tuned") if row.get(f"{k}_ms")}
        if got:
            out[f"{kind}_best"] = min(got, key=got.get)
            out[f"{kind}_best_ms"] = got[out[f"{kind}_best"]]
    served = {k: out[f"{k}_best_ms"] for k in SERVED if f"{k}_best_ms" in out}
    out["plain_path"] = min(served, key=served.get)
    out["plain_ms"] = served[out["plain_path"]]
    out["hashing_ms"] = row["sync_ms"] - row["nohash_ms"]
    out["hashing_share"] = out["hashing_ms"] / row["sync_ms"]
    for base in ("plain", "int8", "serve", "fp8"):
        ms = out.get("plain_ms") if base == "plain" else out.get(f"{base}_best_ms")
        if not ms:
            continue
        out[f"prover_x_{base}"] = row["sync_ms"] / ms
        out[f"arithmetic_x_{base}"] = row["nohash_ms"] / ms
        out[f"hashing_x_{base}"] = out["hashing_ms"] / ms
        if row.get("async_ms"):
            out[f"async_x_{base}"] = row["async_ms"] / ms
    if row.get("sync_tile_ms"):
        acc = call_accounting(row["m"], row["k"], row["n"])
        count, length = acc["messages"]["tile words"]
        out["tile_word_rate_bytes_per_s"] = count * absorbed("sha256", length) / (row["sync_tile_ms"] * 1e-3)
    return out


#: the consistency gate's kernel-efficiency floor e: a ``nohash`` call does W_ref/mkn plain products' MACs, so where the plain
#: int8 GEMM is compute-bound (m·k·n at least `COMPUTE_BOUND_MACS`) it takes at least e·W_ref/mkn plain int8 GEMMs
EFFICIENCY_FLOOR = 1.0
COMPUTE_BOUND_MACS = 1 << 32


def consistency(row: dict) -> dict | None:
    """The consistency gate at a compute-bound shape (None elsewhere): ``nohash`` against e·W_ref/mkn plain int8 GEMMs."""
    if row["m"] * row["k"] * row["n"] < COMPUTE_BOUND_MACS:
        return None
    floor = EFFICIENCY_FLOOR * row["w1_x_mkn"] * row["int8_best_ms"]
    return {"nohash_ms": row["nohash_ms"], "floor_ms": floor, "efficiency_floor": EFFICIENCY_FLOOR,
            "ok": row["nohash_ms"] >= floor}


def forwards(rows: list[dict], spec: dict) -> dict:
    """The baseline set per variant and m, synthetic: the four linears' per-call medians summed over 24 layers, per arm (not
    a served forward: no attention, norms or launches between them), beside #435's served figures."""
    base = spec["sets"]["baseline"]
    want = {tuple(kn) for kn in base["shapes"]}
    out: dict = {}
    for r in rows:
        if r.get("set") != "baseline" or "error" in r or (r["k"], r["n"]) not in want:
            continue
        f = out.setdefault(r["variant"], {}).setdefault(str(r["m"]), {"calls": 0, "synthetic": True})
        f["calls"] += 1
        for arm in ("plain_ms", "int8_best_ms", "serve_best_ms", "fp8_best_ms", "nohash_ms", "sync_ms", "async_ms"):
            if arm in r:
                f[arm] = f.get(arm, 0.0) + base["layers"] * r[arm]
    for per in out.values():
        for f in per.values():
            if f["calls"] == len(want) and f.get("plain_ms"):
                f["prover_x_plain"] = f["sync_ms"] / f["plain_ms"]
    return out


# ---- on the pod --------------------------------------------------------------------------------------------------------------


#: each prover arm's flags; `gate_shape` gates every one at every timed shape
ARM_FLAGS = {"nohash": PD.TIME_TILES, "sync": PD.HASH | PD.TIME_TILES | PD.CACHE_Y, "async": PD.HASH | PD.CACHE_Y}
#: a shape's operand sets: enough that their int8 bytes pass twice the L2, at least 2 and at most this many
MAX_SETS = 512
#: calls in a burst, at least; with more operand sets, a burst is each of them once
MIN_BURST = 4
#: Inductor's GEMM autotuning, for the ``*_tuned`` arms
AUTOTUNE = "max-autotune-no-cudagraphs"


def set_count(m: int, k: int, n: int, l2: int) -> int:
    return min(MAX_SETS, max(2, -(-2 * l2 // ((m + n) * k))))


def rotation(names: list[str], rep: int) -> list[str]:
    """The arms' order in rep ``rep``: rotated by one each rep, so over len(names) reps each arm runs once in each place."""
    return names[rep % len(names):] + names[:rep % len(names)]


def weight_id(shape_index: int, i: int) -> int:
    """Operand set ``i`` of the ``shape_index``-th shape: a weight id no other set of any shape has."""
    return ((shape_index + 1) << 32) | i


def sensors():
    """A reader of the device's SM and memory clocks, temperature, power and throttle reasons: NVML's through pynvml where it
    is installed, else nvidia-smi's, else None."""
    dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    dev = int(dev) if dev.isdigit() else 0
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(dev)
        reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
        return lambda: {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                        "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                        "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1e3, "throttle": hex(reasons(h))}
    except Exception:  # noqa: BLE001 -- no pynvml, or no NVML behind it: nvidia-smi below
        pass
    if not shutil.which("nvidia-smi"):
        return lambda: None
    keys = ("sm_mhz", "mem_mhz", "temp_c", "power_w", "throttle")
    for reasons in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        query = ["nvidia-smi", "-i", str(dev), f"--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,{reasons}",
                 "--format=csv,noheader,nounits"]
        if subprocess.run(query, capture_output=True).returncode == 0:
            def smi(query=query) -> dict:
                values = [v.strip() for v in subprocess.run(query, capture_output=True, text=True).stdout.split(",")]
                return {k: (v if k == "throttle" else float(v)) for k, v in zip(keys, values)}
            return smi
    return lambda: None


def versions(torch) -> dict:
    """What a number here depends on beyond the source: the device, torch, CUDA, cuBLASLt, Triton, the driver, nvcc, and the
    power limit and clocks nvidia-smi reports."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    out: dict = {"device": props.name, "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
                 "l2_bytes": props.L2_cache_size, "memory_bytes": props.total_memory, "torch": torch.__version__,
                 "cuda": torch.version.cuda, "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import triton
        out["triton"] = triton.__version__
    except ImportError:
        out["triton"] = None
    try:
        lt = ctypes.CDLL("libcublasLt.so.12")
        lt.cublasLtGetVersion.restype = ctypes.c_size_t
        out["cublaslt"] = int(lt.cublasLtGetVersion())
    except (OSError, AttributeError):
        out["cublaslt"] = None
    nvcc = PD.nvcc()
    out["nvcc"] = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if nvcc else None
    if shutil.which("nvidia-smi"):
        keys = ("driver_version", "power.limit", "clocks.applications.graphics", "clocks.applications.memory", "clocks.max.sm",
                "clocks.max.mem", "persistence_mode", "pstate")
        r = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={','.join(keys)}", "--format=csv,noheader"], capture_output=True,
                           text=True)
        out["nvidia_smi"] = dict(zip(keys, (v.strip() for v in r.stdout.split(",")))) if r.returncode == 0 else r.stderr[-300:]
    return out


class Clock:
    """Every arm's clock: CUDA events around a burst of calls, after a read of 4× the L2 (the buffer zeroed once, so the
    flush leaves no dirty lines in the L2; ``write`` flushes by writing it instead, for the control), with the sensors read
    before and after."""

    def __init__(self, torch) -> None:
        self.torch = torch
        self.l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        self.buf = torch.zeros(self.l2, dtype=torch.float32, device="cuda")
        self.sink = torch.zeros((), dtype=torch.float32, device="cuda")
        self.sample = sensors()

    def burst(self, call, count: int, end=None, write: bool = False) -> dict:
        torch = self.torch
        torch.cuda.synchronize()
        before = self.sample()
        if write:
            self.buf.zero_()
        else:
            torch.sum(self.buf, 0, out=self.sink)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(count):
            call(i)
        if end is not None:
            end()
        e1.record()
        e1.synchronize()
        return {"ms": e0.elapsed_time(e1) / count, "before": before, "after": self.sample()}


def operand_sets(torch, m: int, k: int, n: int, l2: int, shape_index: int) -> list[dict]:
    """A shape's operand sets (`set_count`), each with its own weight id and its operands for every arm, from its own seed."""
    out = []
    for i in range(set_count(m, k, n, l2)):
        weight = weight_id(shape_index, i)
        g = torch.Generator(device="cuda").manual_seed(weight)
        a = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="cuda", generator=g)
        abf = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=g)
        bbf = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=g)
        out.append({"weight": weight, "a": a, "a32": a if m > 16 else torch.cat([a, a.new_zeros((32 - m, k))]),
                    "b": torch.randint(-64, 64, (n, k), dtype=torch.int8, device="cuda", generator=g),
                    "s": torch.full((m,), 1e-3, dtype=torch.float32, device="cuda").view(torch.int32),
                    "t": torch.full((n,), 1e-2, dtype=torch.float32, device="cuda").view(torch.int32),
                    "bias": torch.zeros(n, dtype=torch.int16, device="cuda"), "abf": abf, "bbf": bbf,
                    "af8": abf.to(torch.float8_e4m3fn), "bf8": bbf.to(torch.float8_e4m3fn)})
    return out


def gate_shape(torch, o: dict, rowids, colids, streams: int) -> list[dict]:
    """Every prover arm's flags at a timed shape, on operand set ``o`` (the first it is timed on), against the independent
    device build (`ncp2_gpu_bench.reference`, `ncp2_kernels.cu`): ``sync``'s by `ncp2_gpu_bench.gate_timed` (root, every leaf,
    the weight side's Y leaves, y, and units drawn from the root against `pouw_native`), a Y-leaf cache miss and then a hit,
    and its root through the timed entry point (`FusedCommitter`); ``nohash``'s y; and change 1's served y and each root
    after the drain, a miss's and a hit's. Stops the run at the first difference."""
    args = (o["a"], o["s"], o["b"], o["t"], o["bias"], 1, 2, o["weight"], rowids, colids)
    call = G.host_call(*args, salt=SALT)
    ref = G.reference(call)
    y_of = lambda y: bool(np.array_equal(y.cpu().numpy().view(np.uint16), ref["y"]))  # noqa: E731
    out = [{"arm": "sync", "cache": c} | G.gate_timed(call, ARM_FLAGS["sync"]) for c in ("miss", "hit")]
    root = np.zeros(32, np.uint8)
    y = PD.FusedCommitter(SCHEME, SALT, ARM_FLAGS["sync"])(*args, root=root)
    torch.cuda.synchronize()
    out.append({"arm": "sync", "entry": "FusedCommitter", "root": root.tobytes() == ref["root"], "y": y_of(y)})
    out.append({"arm": "nohash", "flags": ARM_FLAGS["nohash"], "y": y_of(PD.FusedCommitter(SCHEME, SALT, ARM_FLAGS["nohash"])(*args))})
    c = PD.AsyncCommitter(SCHEME, SALT, ARM_FLAGS["async"], streams=streams)
    served = [c(*args) for _ in range(2)]
    drained = c.drain()
    c.close()
    out.append({"arm": "async", "flags": ARM_FLAGS["async"], "y": all(y_of(y) for y in served),
                "roots": drained["roots"] == [ref["root"]] * 2, "mismatched_words": drained["mismatched_words"]})
    for g in out[2:]:
        if not (g["y"] and g.get("root", True) and g.get("roots", True) and not g.get("mismatched_words")):
            raise SystemExit(f"timed gate {tuple(o['a'].shape)}x{o['b'].shape[0]} {g['arm']}: {g}")
    return out


def _autotune_log() -> list:
    """Inductor's pick for each GEMM it autotunes, appended as it tunes (empty where this torch reports none)."""
    log: list = []
    try:
        from torch._inductor import select_algorithm

        def saver(timings, name, *_rest) -> None:
            best = min(timings, key=timings.get)
            log.append({"op": name, "best": getattr(best, "name", str(best)), "best_ms": float(timings[best]),
                        "choices": len(timings)})

        select_algorithm.add_feedback_saver(saver)
    except (ImportError, AttributeError):
        pass
    return log


def time_shape(torch, clock: Clock, shape: dict, shape_index: int, args, tune_log: list) -> dict:
    """One shape: its gate (`gate_shape`), then every arm timed the same way (`Clock`), the arms' order rotated each rep,
    reps as many as ``args.budget_s`` fits after the first (at least 3, at most ``args.reps``), then the flush control and the
    consistency gate."""
    m, k, n = shape["m"], shape["k"], shape["n"]
    ops = operand_sets(torch, m, k, n, clock.l2, shape_index)
    sets, count = len(ops), max(len(ops), MIN_BURST)
    rowids = torch.arange(m, dtype=torch.int32, device="cuda")
    colids = torch.arange(n, dtype=torch.int32, device="cuda")
    row: dict = dict(shape) | {"weights": [o["weight"] for o in ops], "sets": sets, "burst": count, "int8_m": ops[0]["a32"].shape[0],
                               "operand_bytes": sets * (m + n) * k, "l2_bytes": clock.l2}
    row["gates"] = gate_shape(torch, ops[0], rowids, colids, args.streams)

    lib, one, forward = PD.fused(), torch.ones((), device="cuda"), itertools.count(10)
    gemm = {"int8": lambda a, b: torch._int_mm(a, b.t()), "bf16": lambda a, b: torch.mm(a, b.t()),
            "fp8": lambda a, b: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)}
    operands = {"int8": ("a32", "b"), "fp8": ("af8", "bf8"), "bf16": ("abf", "bbf")}

    def plain(kind: str, fn):
        x, w = operands[kind]
        return lambda i: fn(ops[i % sets][x], ops[i % sets][w])

    def serve(fn):
        def call(i: int) -> None:
            o = ops[i % sets]
            z = fn(o["a32"], o["b"])[:m].contiguous()
            y = torch.empty((m, n), dtype=torch.int16, device="cuda")
            if lib.ncp2_dequant(m, n, z.data_ptr(), o["s"].data_ptr(), o["t"].data_ptr(), o["bias"].data_ptr(), y.data_ptr(),
                                torch.cuda.current_stream().cuda_stream):
                raise RuntimeError(f"ncp2_dequant {m}x{n}")
        return call

    def fused(c):
        def call(i: int) -> None:
            o = ops[i % sets]
            c(o["a"], o["s"], o["b"], o["t"], o["bias"], next(forward), 2, o["weight"], rowids, colids)
        return call

    exact = PD.AsyncCommitter(SCHEME, SALT, ARM_FLAGS["async"], streams=args.streams)
    arms: dict = {kind: (plain(kind, gemm[kind]), None) for kind in operands}
    arms["serve"] = (serve(gemm["int8"]), None)
    for arm in ("nohash", "sync"):
        arms[arm] = (fused(PD.FusedCommitter(SCHEME, SALT, ARM_FLAGS[arm])), None)
    arms["async"] = (fused(exact), lambda: [torch.cuda.current_stream().wait_stream(st) for st in exact.streams])
    row["autotune"] = {}
    if args.autotune == "inductor":
        for kind in operands:
            fn, seen = torch.compile(gemm[kind], mode=AUTOTUNE, dynamic=False), len(tune_log)
            try:
                for i in range(sets):
                    plain(kind, fn)(i)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 -- an arm Inductor can't build is left out, and why is recorded
                row["autotune"][kind] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                continue
            arms[f"{kind}_tuned"] = (plain(kind, fn), None)
            if kind == "int8":
                arms["serve_tuned"] = (serve(fn), None)
            row["autotune"][kind] = {"picks": tune_log[seen:]}

    for _name, (call, end) in arms.items():                    # warm: Y leaves cached, workspaces grown, kernels loaded
        for i in range(count):
            call(i)
        if end is not None:
            end()
    torch.cuda.synchronize()
    exact.drain()
    exact.tile_ms()

    names, raw = list(arms), {a: [] for a in arms}
    reps, rep = None, 0
    while reps is None or rep < reps:
        for a in rotation(names, rep):
            call, end = arms[a]
            r = clock.burst(call, count, end)
            if a in ("nohash", "sync"):
                tile, calls = exact.tile_ms()                   # the library's tile events, whichever committer timed them
                r["tile_ms"] = tile / max(1, calls)
            if a == "async":
                drained = exact.drain()
                if drained["mismatched_words"] or drained["calls"] != count:
                    raise SystemExit(f"async {m}x{k}x{n}: {drained['mismatched_words']} served words differ from the committed y "
                                     f"over {drained['calls']} of {count} calls")
            raw[a].append(r)
        if reps is None:
            slowest = max(raw[a][0]["ms"] for a in names) * count / 1e3
            reps = max(3, min(args.reps, 1 + int(args.budget_s / max(slowest, 1e-6))))
        rep += 1
    exact.close()
    median = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    for a in names:
        row[f"{a}_ms"] = median([r["ms"] for r in raw[a]])
        if a in ("nohash", "sync"):
            row[f"{a}_tile_ms"] = median([r["tile_ms"] for r in raw[a]])
    row["reps"], row["raw"] = reps, raw
    row["flush_control"] = {}
    for a in ("int8", "serve"):
        both = {"read": [], "write": []}
        for _ in range(3):
            for kind in both:
                both[kind].append(clock.burst(arms[a][0], count, write=kind == "write")["ms"])
        row["flush_control"][a] = {f"{kind}_ms": median(v) for kind, v in both.items()} | {"raw": both}
    row |= w1(m, k, n) | derive(row)
    row["consistency"] = consistency(row)
    if row["consistency"] and not row["consistency"]["ok"]:
        raise SystemExit(f"consistency gate {m}x{k}x{n}: nohash {row['nohash_ms']:.4f} ms is under "
                         f"{row['consistency']['floor_ms']:.4f} ms, e·W_ref/mkn plain int8 GEMMs: work was skipped")
    torch.cuda.synchronize()
    lib.ncp2_fused_reset_cache()
    del ops, arms
    torch.cuda.empty_cache()
    return row


# ---- the run ------------------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="CPU only: shapes, W1, sampling, the twin's gates, resource usage "
                                                             "when nvcc is on the path, and the #389 path on the CPU")
    ap.add_argument("--tile-variants", default=VARIANTS, help="builds as ncp2_gpu_bench takes them, ';'-separated, each like "
                                                              "rows=8,unroll=2,sha=1 (default: before change 3, then its default)")
    ap.add_argument("--sets", default="baseline,census,square")
    ap.add_argument("--reps", type=int, default=20, help="at most, per arm and shape; at least 3")
    ap.add_argument("--budget-s", type=float, default=2.0, help="per arm and shape: reps after the first while the slowest "
                                                                  "arm's bursts fit")
    ap.add_argument("--streams", type=int, default=4, help="change 1's side streams")
    ap.add_argument("--autotune", choices=("inductor", "none"), default="inductor",
                    help="the plain GEMMs' autotuned arms (Inductor's max-autotune), or none")
    ap.add_argument("--cpu-rows", type=int, default=D, help="rows of the #389 CPU commit (0: skip it)")
    ap.add_argument("--shapes", type=Path, default=SHAPES)
    ap.add_argument("--census", type=Path, default=CENSUS)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "pouw_bench.json")
    args = ap.parse_args()
    PD.configure(build_dir=os.environ.get("VERITY_POUW_CUDA_BUILD"), arch=os.environ.get("VERITY_POUW_CUDA_ARCH"))
    spec = json.loads(args.shapes.read_text())
    sets = [s for s in args.sets.split(",") if s]
    workloads = json.loads(args.census.read_text()) if "census" in sets else None
    shapes = bench_shapes(sets, spec, workloads)
    variants = parse_variants(args.tile_variants)
    res: dict = {
        "schema": "verity-pouw/pouw-bench/v2", "scheme": SCHEME, "dry_run": args.dry_run, "autotune": args.autotune,
        "baseline": "the plain path of the same (m, k, n): the fastest of the int8 serving path (cuBLASLt torch._int_mm, m "
                    "padded to 32 where it needs more than 16 rows, and ncp2_dequant), FP8 (torch._scaled_mm: e4m3, FP32 "
                    "accumulate, BF16 out; half the int8 rate on a GeForce RTX 4090) and BF16 (torch.mm), each at the better "
                    "of its default and its autotuned call",
        "timing": "every arm the same way: bursts of the same number of calls between CUDA events, cycling operand sets "
                  "past twice the L2, each burst after a read of 4x the L2 (no dirty lines), the arms' order rotated each "
                  "rep; the median per-call time over reps, every rep saved with the sensors before and after",
        "derived": "w1_x_mkn, omega and gamma are derived from the certificate, not measured",
        "hashing": "hashing and XOF (incremental): sync less nohash, which leaves out every hash and the XOF noise; the "
                   "weight side's Y leaves are cached per weight (CACHE_Y) and not in any arm",
        "certificate": certificate(), "sampling": sampling(shapes), "figures": spec["figures"],
        "shapes": [s | w1(s["m"], s["k"], s["n"]) for s in shapes], "variants": {}}

    def save(key: str | None = None, value=None) -> None:
        if key is not None:
            res[key] = value
            print(f"{key.upper()} " + json.dumps(value, default=str)[:3000], flush=True)
        args.out.write_text(json.dumps(res, indent=1, default=str))

    save()
    calls = gate_calls(spec, gate_shapes(shapes, spec))
    refs = {name: PN.commit_call(0, SCHEME, call) for name, call in calls}
    torch = clock = None
    tune_log: list = []
    if not args.dry_run:
        import torch

        torch._dynamo.config.cache_size_limit = 1 << 12              # a compile per shape and plain GEMM, none evicted
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = 1 << 14
        tune_log = _autotune_log()
        clock = Clock(torch)
        res["device"] = torch.cuda.get_device_name()
        res["versions"] = versions(torch)
        res["sensors_idle"] = clock.sample()
        res["hash_gate"] = G.gate_hashes()
    rows: list[dict] = []
    for name, flags in variants:
        use(flags)
        v: dict = {"flags": list(flags), "resource_usage": resource_usage(flags), "twin_gate": twin_gate(calls, refs)}
        res["variants"][name] = v
        save()
        if torch is None:
            continue
        v["fused_gate"] = device_gate(calls, refs)
        v["async_gate"] = G.gate_async(torch)
        save()
        for index, shape in enumerate(shapes):
            try:
                rows.append({"variant": name} | time_shape(torch, clock, shape, index, args, tune_log))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                rows.append({"variant": name, **shape, "error": str(e)[:300]})
                PD.fused().ncp2_fused_reset_cache()
                torch.cuda.empty_cache()
            res["timings"] = rows
            save()
    use(())
    if rows:
        save("forwards", forwards(rows, spec))
    if args.cpu_rows:
        save("cpu_389", cpu_commit(spec, args.cpu_rows))
    _report(res)
    import run_result

    ms = [(f"{r['variant']}.{r['set']}.{r['m']}x{r['k']}x{r['n']}.{key}", r[key], "ratio") for r in rows if "error" not in r
          for key in ("prover_x_plain", "hashing_x_plain", "async_x_plain") if key in r]
    run_result.write(args.out, not any("error" in r for r in rows), "pouw_bench: every variant's gates passed before its timings",
                     f"{len(variants)} variants, {len(shapes)} shapes, {len(rows)} timed", ms)


def _report(res: dict) -> None:
    c = res["certificate"]
    print(f"certificate: {c['conjecture']}, γ_0 = {c['gamma0']}, Ω* = {c['omega_star']}, γ = {c['gamma']:.4%} over the domain")
    print("samples per bit (our definition): " + "; ".join(f"ε_s {s['eps_s']:.1%}: {s['samples_per_bit']:.1f}"
                                                           for s in res["sampling"]["draws"]))
    for d in res["sampling"]["verifier_draws"]:
        print(f"  a drawn tile at k={d['k']} n={d['n']}: {d['tile_s'] * 1e3:.1f} ms on the verifier's CPU")
    for name, v in res["variants"].items():
        ru = (v.get("resource_usage") or {}).get("kf_tiles<1>")
        print(f"{name}: twin gate {len(v['twin_gate'])} calls pass" + (f"; kf_tiles<1> {ru['registers']} registers, "
              f"{ru['spill_stores']} B spills, {ru['warps']} warps an SM" if ru else ""))
    for r in res.get("timings", []):
        if "error" in r:
            print(f"{r['variant']} {r['m']}x{r['k']}x{r['n']}: {r['error']}")
        else:
            print(f"{r['variant']} {r['m']}x{r['k']}x{r['n']}: prover {r['prover_x_plain']:.1f}x the plain path "
                  f"({r['plain_path']}), {r['prover_x_int8']:.1f}x int8; hashing and XOF {r['hashing_x_plain']:.1f}x "
                  f"({r['hashing_share']:.0%} of the prover); async {r.get('async_x_plain', float('nan')):.1f}x; "
                  f"W1 charges {r['w1_x_mkn']:.2f}x (Ω {r['omega']:.4f}); {r['reps']} reps of {r['burst']} calls")
    if "cpu_389" in res:
        cpu = res["cpu_389"]
        print(f"#389 path on {cpu['cpu']} ({cpu['threads']} threads): a forward at {cpu['rows']} rows {cpu['forward_native_s']:.1f} s "
              f"(pouw_native), {cpu['forward_native_with_weight_s']:.1f} s with each weight side as #389 committed it per call, "
              f"{cpu['forward_twin_s']:.2f} s (twin); #389 measured {res['figures']['#389']['cpu_forward_16_rows_s']}")


if __name__ == "__main__":
    main()
