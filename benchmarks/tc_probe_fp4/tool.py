"""Store declaration for the Blackwell FP4 tensor-core probe (research.store README §6.1), lane fp4-5090.

    research run --on vy-5090 --project verity --campaign r21-silicon --tool tc_probe_fp4 --source <worktree> \
        --send autoproof/probes/results/probe3_1783788526.json -- \
        python benchmarks/tc_probe_fp4/probe.py --instruction nvf4 --out $RESEARCH_RUN_DIR --sass-check --sweep --n-random 4096 \
            --replay-autoproof $RESEARCH_RUN_DIR/inputs/probe3_1783788526.json

Key params follow probe.py's argparse; `--out` and `--procs` never enter the derivation.  Replay files are content-resolved.

`tc_price_fp4` is the W1 price microbenchmark beside it (w1_price.py, lane pouw rtx-pro GPU 4); `--out`, `--expect-uuid` and
`--max-seconds` are guards and paths, not part of the answer:

    research run --on vy-nebius-2 --project verity --campaign pouw --tool tc_price_fp4 --source . --cwd source -- \
        gpu-lease 1 -- uv run --no-project --with numpy python benchmarks/tc_probe_fp4/w1_price.py --out $RESEARCH_RUN_DIR \
            --expect-uuid GPU-...,GPU-... --locked-mhz 2100

`tc_rates_fp4` is F1 (f1_rates.py): its CPU steps (``build``, ``collect``) run as research runs, its timed families as fill jobs
(``benchmarks.tc_probe_fp4.tool:TC_RATES_FP4``; paths, ``--jobs`` and guards are not part of the answer).  `tc_lut_fp4` is F3
(f3_lut.py), run the same way with one width per fill chunk, and `tc_strassen_fp4` is F2 (f2_strassen.py), one level per chunk.
`tc_recheck_fp4` (recheck.py) and `tc_sparse_e4m3` (sp_e4m3.py, the 2:4 sparse E4M3 row against two dense k32 atoms) capture in
fill chunks and verify on the CPU (``benchmarks.tc_probe_fp4.tool:TC_SPARSE_E4M3``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)

SCRIPT = "probe.py"


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == SCRIPT), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--instruction", default="nvf4")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay-autoproof", action="append", default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None)
    ap.add_argument("--sass-check", action="store_true")
    ap.add_argument("--library", default=None)                  # a prebuilt library; its sha256 is in the result's nvcc identity
    ap.add_argument("--chain2", action="store_true")            # lane tcgen05: K=128 chains on the TMEM accumulator
    ap.add_argument("--smem-strides", default=None)             # lane tcgen05: shared-memory descriptor strides
    ap.add_argument("--chain", type=int, default=1)             # mma.sync: in-register chains of L dependent MMAs
    ap.add_argument("--expect-model", default=None)             # the model whose mismatches gate the run
    ap.add_argument("--corrupt-word", action="store_true")      # the gate's self-test: one flipped word must fail the run
    ap.add_argument("--expect-uuid", default=None)              # a guard (stops a run on the wrong GPU), not part of the answer
    ap.add_argument("--fake-silicon", default=None)             # dry run; probe.py refuses it under a research run
    ap.add_argument("--sass-only", action="store_true")
    ap.add_argument("--hot-bits", default="14,15,16")            # sp_nvf4_t1: the hot accumulators' h
    ap.add_argument("--model-sample", type=int, default=4096)   # sp_nvf4_t1: words checked against the pinned models
    ap.add_argument("--prebuilt", default=None)                 # a path to the same source's library (its sha256 is recorded)
    ap.add_argument("--vectors", default=None)                  # nvf4_lean_vectors: the Lean vector file (content-resolved)
    ap.add_argument("--vector-list", default="nvf4P7")
    ap.add_argument("--vector-ids", default=None)
    ap.add_argument("--chain-ids", default=None)
    ap.add_argument("--special-ids", default=None)
    ap.add_argument("--collect", default=None)                  # a fill chunk's capture directory: a path, so only its presence
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    out.pop("prebuilt")
    out.pop("library")
    out["collect"] = out["collect"] is not None
    if out["arch"] is None:  # probe.py's default: sm_120a for the mma.sync instructions, sm_100a for the tcgen05 ones
        out["arch"] = "sm_100a" if str(out["instruction"]).startswith("tcgen05") else "sm_120a"
    out["n"] = out.pop("n_random")
    out.pop("expect_uuid")
    out["replay_autoproof"] = sorted(out["replay_autoproof"])
    if unknown:
        out["_unknown"] = unknown
    if out.get("families"):
        out["families"] = ",".join(sorted(f for f in out["families"].split(",") if f))
    return out


def parse_price(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "w1_price.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--target-ms", type=float, default=100.0)
    ap.add_argument("--sustain-s", type=float, default=3.0)
    ap.add_argument("--warps-per-sm", type=int, default=16)
    ap.add_argument("--ops", default=None)
    ap.add_argument("--op-warps-per-sm", type=int, default=None)
    ap.add_argument("--latency-iters", type=int, default=20000)
    ap.add_argument("--no-scale-update", action="store_true")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=900.0)   # a guard on the plan's length, not part of the answer
    ap.add_argument("--expect-uuid", default=None)                # a guard (stops a run on the wrong GPU)
    ap.add_argument("--locked-mhz", type=float, default=None)
    ap.add_argument("--corrupt-word", action="store_true")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--sass-only", action="store_true")
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    out.pop("out")
    out.pop("max_seconds")
    out.pop("expect_uuid")
    for k in ("rows", "ops"):
        if out[k]:
            out[k] = ",".join(sorted(r for r in out[k].split(",") if r))
    if unknown:
        out["_unknown"] = unknown
    return out


def parse_f1(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "f1_rates.py"), 0)
    rest = argv[idx + 1:]
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", default=None)
    for flag in ("--out", "--build", "--runs", "--libscan", "--expect-uuid", "--forms", "--family", "--libs"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=100.0)
    ap.add_argument("--sustain-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--locked-mhz", type=float, default=None)
    ap.add_argument("--corrupt-word", action="store_true")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ns, unknown = ap.parse_known_args(rest)
    out = vars(ns)
    for k in ("out", "build", "runs", "libscan", "expect_uuid", "jobs", "timeout_s"):  # paths and guards, not the answer
        out.pop(k)
    if out["forms"]:
        out["forms"] = ",".join(sorted(f for f in out["forms"].split(",") if f))
    if unknown:
        out["_unknown"] = unknown
    return out


TC_RATES_FP4 = Tool(
    name="tc_rates_fp4", version="1",
    command=["python", "benchmarks/tc_probe_fp4/f1_rates.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["cmd", "forms", "family", "libs", "reps", "target_ms", "sustain_s", "seed", "locked_mhz", "corrupt_word", "fake_silicon",
                "tolerance"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_f1,
)


def parse_f3(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "f3_lut.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", default=None)
    for flag in ("--out", "--build", "--runs", "--expect-uuid", "--width"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=100.0)
    ap.add_argument("--sustain-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--sample", type=int, default=2048)
    ap.add_argument("--locked-mhz", type=float, default=None)
    ap.add_argument("--corrupt-word", action="store_true")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    for k in ("out", "build", "runs", "expect_uuid"):  # paths and guards, not the answer
        out.pop(k)
    if unknown:
        out["_unknown"] = unknown
    return out


TC_LUT_FP4 = Tool(
    name="tc_lut_fp4", version="1",
    command=["python", "benchmarks/tc_probe_fp4/f3_lut.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["cmd", "width", "reps", "target_ms", "sustain_s", "seed", "sample", "locked_mhz", "corrupt_word", "fake_silicon",
                "tolerance"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_f3,
)


def parse_f2(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "f2_strassen.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", default=None)
    for flag in ("--out", "--build", "--runs", "--expect-uuid", "--level"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=100.0)
    ap.add_argument("--sustain-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--sample", type=int, default=2048)
    ap.add_argument("--locked-mhz", type=float, default=None)
    ap.add_argument("--corrupt-word", action="store_true")
    ap.add_argument("--fake-silicon", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    for k in ("out", "build", "runs", "expect_uuid"):  # paths and guards, not the answer
        out.pop(k)
    if unknown:
        out["_unknown"] = unknown
    return out


TC_STRASSEN_FP4 = Tool(
    name="tc_strassen_fp4", version="1",
    command=["python", "benchmarks/tc_probe_fp4/f2_strassen.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["cmd", "level", "reps", "target_ms", "sustain_s", "seed", "sample", "locked_mhz", "corrupt_word", "fake_silicon",
                "tolerance"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_f2,
)


def parse_recheck(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "recheck.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", default=None)
    for flag in ("--out", "--capture", "--prebuilt", "--expect-uuid", "--instruction", "--seeds", "--expect-model", "--fake-silicon"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=None)
    ap.add_argument("--fixed-mult", type=int, default=1)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    for k in ("out", "capture", "prebuilt", "expect_uuid", "procs", "budget_s"):  # paths and guards, not the answer
        out.pop(k)
    if unknown:
        out["_unknown"] = unknown
    return out


TC_RECHECK_FP4 = Tool(
    name="tc_recheck_fp4", version="1",
    command=["python", "benchmarks/tc_probe_fp4/recheck.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["cmd", "instruction", "seed", "seeds", "n_random", "fixed_mult", "expect_model", "fake_silicon"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse_recheck,
)


def parse_sparse_e4m3(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if Path(t).name == "sp_e4m3.py"), 0)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", default=None)
    for flag in ("--out", "--capture", "--prebuilt", "--expect-uuid", "--seeds", "--fake-silicon"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=None)
    ap.add_argument("--fixed-mult", type=int, default=1)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    for k in ("out", "capture", "prebuilt", "expect_uuid", "procs", "budget_s"):  # paths and guards, not the answer
        out.pop(k)
    if unknown:
        out["_unknown"] = unknown
    return out


TC_SPARSE_E4M3 = Tool(
    name="tc_sparse_e4m3", version="1",
    command=["python", "benchmarks/tc_probe_fp4/sp_e4m3.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["cmd", "arch", "seed", "seeds", "n_random", "fixed_mult", "fake_silicon"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse_sparse_e4m3,
)


TC_PRICE_FP4 = Tool(
    name="tc_price_fp4", version="2",
    command=["python", "benchmarks/tc_probe_fp4/w1_price.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["rows", "ops", "reps", "target_ms", "sustain_s", "warps_per_sm", "op_warps_per_sm", "latency_iters", "no_scale_update",
                "seed", "locked_mhz", "corrupt_word", "fake_silicon", "sass_only"],
    key_conditions=["hw.family"],
    caches=["nvcc"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse_price,
)


TC_PROBE_FP4 = Tool(
    name="tc_probe_fp4", version="1",
    command=["python", "benchmarks/tc_probe_fp4/probe.py"],
    closure=["benchmarks/tc_probe_fp4/**", *package_paths(TREE, "verity.ml.circuits", "silicon/**")],
    key_params=["instruction", "arch", "families", "seed", "n", "group", "sweep", "golden_per_family", "replay_autoproof", "sass_check",
                "chain2", "smem_strides", "chain", "expect_model", "corrupt_word", "fake_silicon", "sass_only", "hot_bits",
                "model_sample", "vectors", "vector_list", "vector_ids", "chain_ids", "special_ids", "collect"],
    key_conditions=["hw.family"],
    param_artifacts=["replay_autoproof", "vectors"],
    caches=["nvcc"],
    produces=["tc-evidence/v1", "run-files/v1"],
    parse=parse,
)
