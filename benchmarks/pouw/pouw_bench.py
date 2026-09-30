"""PoUW alone on fixed matmul shapes, on one RTX 4090 (W1): `ncp-v2`'s fused prover against the plain int8 and FP8 GEMM of
the same shape, with hashing included and reported separately; γ_0 and Ω* from the scheme's certificate; samples per bit.
No served model.

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

**Gates before timing**, per variant, each stopping the run at its first difference:
- the CPU twin built with the variant's flags against `pouw_native`, on `pouw_device.gate_calls` with the baseline linears
  and one 16-row call per other (k, n) timed (every arm below commits exactly these leaves);
- on the pod, the device's hashes against hashlib, the fused kernel against `pouw_native` on the same calls, and change 1's
  async path (every served y against its committed y, and each root after the drain).

**Arms, per shape and variant** (honest calls only; no tamper arm, and nothing whose γ is unproved):
- ``int8``: cuBLASLt ``torch._int_mm`` (m padded to 32 rows where it needs more than 16), ``fp8``: cuBLASLt
  ``torch._scaled_mm`` (e4m3 in, FP32 accumulate, BF16 out), ``bf16``: ``torch.mm``. These are the baseline: the plain GEMM
  of the same (m, k, n).
- ``nohash``: the fused kernel with every hash and XOF left out and a stand-in noise, the arithmetic alone.
- ``sync``: the fused kernel committing every leaf (`pouw_device.FusedCommitter`), the prover's time per call.
- ``async``: change 1 (`pouw_device.AsyncCommitter`): y served from the exact int8 product, the commit on side streams;
  a run of calls then the drain, per call.

Every arm is the median of CUDA-event repetitions with the L2 flushed before each. The prover's overhead is ``sync`` over
the plain GEMM; hashing, reported separately, is ``sync`` less ``nohash``.

**W1:** each shape's Ω = W_ref / the credited 3·m·k·n and its γ = 1 − (1 − γ_0)/Ω where the shape is in the scheme's domain,
beside the certificate's Ω* and γ over the domain (`verity_pouw.schemes.ncp.NCP`). **Samples per bit:** work-weighted
sampling gives δ_s = (1 − ε_s)^t (Lean `workWeightedSampling`), so a bit of soundness costs 1/−log₂(1 − ε_s) draws, whatever
the shape.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ncp2_gpu_bench import TILE_MACROS, absorbed, call_accounting  # noqa: E402
from verity_pouw.protocol import DomainError, Matmul, Shape  # noqa: E402
from verity_pouw.schemes import ncp  # noqa: E402
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
    """Ω = W_ref / 3·m·k·n (route U, W1 on the 4090) and γ = 1 − (1 − γ_0)/Ω where the shape is in the domain."""
    s = ncp.NCP()
    shape = Shape(m, k, n)
    omega = Fraction(s.wref(shape), 3 * shape.wmm)
    out = {"omega": float(omega), "hashed_bytes_per_mac": hashed_bytes_per_mac(m, k, n)}
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


def sampling() -> list[dict]:
    return [{"eps_s": e, "samples_per_bit": samples_per_bit(e), f"draws_for_{LAMBDA}_bits": math.ceil(LAMBDA * samples_per_bit(e))}
            for e in EPSILONS]


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


def derive(row: dict) -> dict:
    """From one shape's timings: the prover's time over each plain GEMM (hashing included), the same kernel with nothing
    hashed, and hashing alone (``sync`` less ``nohash``), reported separately; change 1's per-call time likewise."""
    out: dict = {"hashing_ms": row["sync_ms"] - row["nohash_ms"]}
    out["hashing_share"] = out["hashing_ms"] / row["sync_ms"]
    for plain in ("int8", "fp8", "bf16"):
        base = row.get(f"{plain}_ms")
        if not base:
            continue
        out[f"prover_x_{plain}"] = row["sync_ms"] / base
        out[f"arithmetic_x_{plain}"] = row["nohash_ms"] / base
        out[f"hashing_x_{plain}"] = out["hashing_ms"] / base
        if "async_ms_per_call" in row:
            out[f"async_x_{plain}"] = row["async_ms_per_call"] / base
    if row.get("sync_tile_ms"):
        acc = call_accounting(row["m"], row["k"], row["n"])
        count, length = acc["messages"]["tile words"]
        out["tile_word_rate_bytes_per_s"] = count * absorbed("sha256", length) / (row["sync_tile_ms"] * 1e-3)
    return out


def forwards(rows: list[dict], spec: dict) -> dict:
    """The baseline set per variant and m: a 24-layer forward of the four linears, per arm, beside #435's figures."""
    base = spec["sets"]["baseline"]
    want = {tuple(kn) for kn in base["shapes"]}
    out: dict = {}
    for r in rows:
        if r.get("set") != "baseline" or "error" in r or (r["k"], r["n"]) not in want:
            continue
        f = out.setdefault(r["variant"], {}).setdefault(str(r["m"]), {"calls": 0})
        f["calls"] += 1
        for arm in ("int8_ms", "fp8_ms", "bf16_ms", "nohash_ms", "sync_ms", "async_ms_per_call"):
            if arm in r:
                f[arm] = f.get(arm, 0.0) + base["layers"] * r[arm]
    for per in out.values():
        for f in per.values():
            if f["calls"] == len(want) and f.get("int8_ms"):
                f["prover_x_int8"] = f["sync_ms"] / f["int8_ms"]
    return out


# ---- on the pod --------------------------------------------------------------------------------------------------------------


def _timer(torch, fn, reps: int, budget_s: float) -> tuple[float, int]:
    """The median of CUDA-event repetitions, the L2 flushed (a 256 MB write) before each: at least 3, at most ``reps``, and
    as many as fit ``budget_s`` after a first measure."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def once() -> float:
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b)

    once()
    first = once()
    n = max(3, min(reps, int(budget_s * 1e3 / max(first, 1e-3))))
    times = sorted([first] + [once() for _ in range(n - 1)])
    return times[len(times) // 2], n


def time_shape(torch, shape: dict, reps: int, budget_s: float, streams: int) -> dict:
    m, k, n = shape["m"], shape["k"], shape["n"]
    row: dict = dict(shape)
    g = torch.Generator(device="cuda").manual_seed(m * 1_000_003 + k * 1009 + n)
    a = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-64, 64, (n, k), dtype=torch.int8, device="cuda", generator=g)
    s = torch.full((m,), 1e-3, dtype=torch.float32, device="cuda").view(torch.int32)
    t = torch.full((n,), 1e-2, dtype=torch.float32, device="cuda").view(torch.int32)
    bias = torch.zeros(n, dtype=torch.int16, device="cuda")
    rowids = torch.arange(m, dtype=torch.int32, device="cuda")
    colids = torch.arange(n, dtype=torch.int32, device="cuda")

    wide = a if m > 16 else torch.cat([a, a.new_zeros((32 - m, k))])
    row["int8_m"] = wide.shape[0]
    row["int8_ms"], row["int8_reps"] = _timer(torch, lambda: torch._int_mm(wide, b.T), reps, budget_s)
    abf = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=g)
    bbf = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=g)
    row["bf16_ms"], row["bf16_reps"] = _timer(torch, lambda: abf @ bbf.T, reps, budget_s)
    af8, bf8, one = abf.to(torch.float8_e4m3fn), bbf.to(torch.float8_e4m3fn), torch.ones((), device="cuda")
    try:
        row["fp8_ms"], row["fp8_reps"] = _timer(
            torch, lambda: torch._scaled_mm(af8, bf8.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), reps, budget_s)
    except RuntimeError as e:
        row["fp8_error"] = str(e)[:200]
    del wide, abf, bbf, af8, bf8

    ys = {}
    for arm, flags in (("nohash", PD.TIME_TILES), ("sync", PD.HASH | PD.TIME_TILES | PD.CACHE_Y)):
        c = PD.FusedCommitter(SCHEME, SALT, flags)
        ys[arm] = c(a, s, b, t, bias, 1, 2, n, rowids, colids)
        torch.cuda.synchronize()
        c.tile_ms()
        forward = [2]

        def call(c=c, forward=forward) -> None:
            c(a, s, b, t, bias, forward[0], 2, n, rowids, colids)
            forward[0] += 1

        row[f"{arm}_ms"], row[f"{arm}_reps"] = _timer(torch, call, reps, budget_s)
        tile, calls = c.tile_ms()
        row[f"{arm}_tile_ms"] = tile / max(1, calls)

    c = PD.AsyncCommitter(SCHEME, SALT, PD.HASH | PD.CACHE_Y, streams=streams)
    served = c(a, s, b, t, bias, 1, 2, n, rowids, colids)
    c.drain()
    runs = max(4, min(64, int(budget_s * 1e3 / max(row["sync_ms"], 1e-3))))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(runs):
        c(a, s, b, t, bias, 100 + i, 2, n, rowids, colids)
    drained = c.drain()
    row["async_ms_per_call"], row["async_calls"] = (time.perf_counter() - t0) * 1e3 / runs, runs
    row["serve_ms"], row["serve_reps"] = _timer(torch, lambda: c.serve(a, s, b, t, bias), reps, budget_s)
    c.close()
    if drained["mismatched_words"] or drained["calls"] != runs:
        raise SystemExit(f"async {m}x{k}x{n}: {drained['mismatched_words']} served words differ from the committed y "
                         f"over {drained['calls']} of {runs} calls")
    row["y_equal"] = bool(torch.equal(ys["sync"], ys["nohash"]) and torch.equal(ys["sync"], served))
    if not row["y_equal"]:
        raise SystemExit(f"{m}x{k}x{n}: the arms serve different y")
    row |= derive(row)
    del a, b
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
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--budget-s", type=float, default=1.0, help="per arm and shape, after a first measure")
    ap.add_argument("--streams", type=int, default=4, help="change 1's side streams")
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
        "schema": "verity-pouw/pouw-bench/v1", "scheme": SCHEME, "dry_run": args.dry_run,
        "baseline": "the plain GEMM of the same (m, k, n): cuBLASLt int8 (torch._int_mm, m padded to 32 where it needs more "
                    "than 16 rows) and FP8 (torch._scaled_mm: e4m3, FP32 accumulate, BF16 out), and BF16 (torch.mm); each the "
                    "median of CUDA-event repetitions with the L2 flushed before each",
        "certificate": certificate(), "sampling": sampling(), "figures": spec["figures"],
        "shapes": [s | w1(s["m"], s["k"], s["n"]) for s in shapes], "variants": {}}

    def save(key: str | None = None, value=None) -> None:
        if key is not None:
            res[key] = value
            print(f"{key.upper()} " + json.dumps(value, default=str)[:3000], flush=True)
        args.out.write_text(json.dumps(res, indent=1, default=str))

    save()
    calls = gate_calls(spec, gate_shapes(shapes, spec))
    refs = {name: PN.commit_call(0, SCHEME, call) for name, call in calls}
    torch = None
    if not args.dry_run:
        import torch

        from ncp2_gpu_bench import gate_async, gate_hashes
        res["device"] = torch.cuda.get_device_name()
        res["hash_gate"] = gate_hashes()
    rows: list[dict] = []
    for name, flags in variants:
        use(flags)
        v: dict = {"flags": list(flags), "resource_usage": resource_usage(flags), "twin_gate": twin_gate(calls, refs)}
        res["variants"][name] = v
        save()
        if torch is None:
            continue
        v["fused_gate"] = device_gate(calls, refs)
        v["async_gate"] = gate_async(torch)
        save()
        for shape in shapes:
            try:
                rows.append({"variant": name} | time_shape(torch, shape, args.reps, args.budget_s, args.streams))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                rows.append({"variant": name, **shape, "error": str(e)[:300]})
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
          for key in ("prover_x_int8", "prover_x_fp8", "hashing_x_int8") if key in r]
    run_result.write(args.out, not any("error" in r for r in rows), "pouw_bench: every variant's gates passed before its timings",
                     f"{len(variants)} variants, {len(shapes)} shapes, {len(rows)} timed", ms)


def _report(res: dict) -> None:
    c = res["certificate"]
    print(f"certificate: {c['conjecture']}, γ_0 = {c['gamma0']}, Ω* = {c['omega_star']}, γ = {c['gamma']:.4%} over the domain")
    print("samples per bit: " + "; ".join(f"ε_s {s['eps_s']:.1%}: {s['samples_per_bit']:.1f}" for s in res["sampling"]))
    for name, v in res["variants"].items():
        ru = (v.get("resource_usage") or {}).get("kf_tiles<1>")
        print(f"{name}: twin gate {len(v['twin_gate'])} calls pass" + (f"; kf_tiles<1> {ru['registers']} registers, "
              f"{ru['spill_stores']} B spills, {ru['warps']} warps an SM" if ru else ""))
    for r in res.get("timings", []):
        if "error" in r:
            print(f"{r['variant']} {r['m']}x{r['k']}x{r['n']}: {r['error']}")
        else:
            print(f"{r['variant']} {r['m']}x{r['k']}x{r['n']}: prover {r['prover_x_int8']:.1f}x int8, "
                  f"{r.get('prover_x_fp8', float('nan')):.1f}x FP8; hashing {r['hashing_x_int8']:.1f}x int8 "
                  f"({r['hashing_share']:.0%} of the prover); async {r.get('async_x_int8', float('nan')):.1f}x int8")
    if "cpu_389" in res:
        cpu = res["cpu_389"]
        print(f"#389 path on {cpu['cpu']} ({cpu['threads']} threads): a forward at {cpu['rows']} rows {cpu['forward_native_s']:.1f} s "
              f"(pouw_native), {cpu['forward_twin_s']:.2f} s (twin); #389 measured {res['figures']['#389']['cpu_forward_16_rows_s']}")


if __name__ == "__main__":
    main()
