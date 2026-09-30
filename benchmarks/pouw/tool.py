"""Store declarations for the PoUW harness (research.store README §6.1).

    research run --on vy-pouw-mvp --project verity --campaign pouw --tool pouw_gemm -- python benchmarks/pouw/gemm_bench.py --out $D/gemm.json
    research run ... --tool pouw_route_u -- python benchmarks/pouw/route_u_bench.py --out $D/route_u.json
    research run ... --tool pouw_vllm -- python benchmarks/pouw/vllm_bench.py --model /root/models/qwen2.5-3b-instruct --out $D/vllm.json
    research run ... --tool pouw_e2e -- python benchmarks/pouw/e2e_audit.py --out $D/e2e.json

Key params: the workload file (gemm), the model and the modes (vllm), the draw count (e2e).  ``--out`` says where and does not
enter the derivation.  Imports nothing but ``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.store.tool import Tool

CLOSURE = ["benchmarks/pouw/**", "protocols/pouw/verity_pouw/**", "packages/verity/src/verity/randomness/**",
           "packages/verity/src/verity/commitments/**", "packages/verity/src/verity/ml/tc/**", "packages/verity/src/verity/errors.py"]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", default=None)
    return ap


def parse_gemm(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--workload", default="benchmarks/pouw/workloads.json")
    ns, _ = ap.parse_known_args(argv)
    return {"workload": ns.workload}


def parse_vllm(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--modes", default="bf16,fp8,w7a7,ncp,pearl")
    ns, _ = ap.parse_known_args(argv)
    return {"model": (ns.model or "").rstrip("/").rsplit("/", 1)[-1], "modes": ns.modes}


def parse_e2e(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--t", type=int, default=8)
    ns, _ = ap.parse_known_args(argv)
    return {"t": ns.t}


def parse_route_u(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--reps", type=int, default=30)
    ns, _ = ap.parse_known_args(argv)
    return {"reps": ns.reps}


POUW_GEMM = Tool(name="pouw_gemm", version="1", command=["python", "benchmarks/pouw/gemm_bench.py"], closure=CLOSURE,
                 key_params=["workload"], produces=["bench-result/v1", "run-files/v1"], parse=parse_gemm)
POUW_ROUTE_U = Tool(name="pouw_route_u", version="1", command=["python", "benchmarks/pouw/route_u_bench.py"], closure=CLOSURE,
                    key_params=["reps"], produces=["bench-result/v1", "run-files/v1"], parse=parse_route_u)
POUW_VLLM = Tool(name="pouw_vllm", version="1", command=["python", "benchmarks/pouw/vllm_bench.py"], closure=CLOSURE,
                 key_params=["model", "modes"], produces=["bench-result/v1", "run-files/v1"], parse=parse_vllm)
POUW_E2E = Tool(name="pouw_e2e", version="1", command=["python", "benchmarks/pouw/e2e_audit.py"], closure=CLOSURE,
                key_params=["t"], produces=["bench-result/v1", "run-files/v1"], parse=parse_e2e)
POUW_PEARL_CAL = Tool(name="pouw_pearl_cal", version="1", command=["python", "benchmarks/pouw/pearl_calibration.py"],
                      closure=["benchmarks/pouw/pearl_calibration.py", "benchmarks/pouw/pearl_b200_setup.sh",
                               "benchmarks/pouw/run_result.py"],
                      key_params=["model"], produces=["bench-result/v1", "run-files/v1"], parse=parse_vllm)


def parse_pearlc_census(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--ks", default="8192,16384,65536")
    ap.add_argument("--deltas", default="1")
    ap.add_argument("--tile", default="64x16")
    ap.add_argument("--families", default=None)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--forming", default="census")
    ap.add_argument("--line-norm", type=float, default=256.0)
    ns, _ = ap.parse_known_args(argv)
    return {"ks": ns.ks, "deltas": ns.deltas, "tile": ns.tile, "families": ns.families, "run": ns.run,
            "forming": ns.forming, "line_norm": ns.line_norm}


def parse_pearlc_quality(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--line-norm", type=float, default=256.0)
    ns, _ = ap.parse_known_args(argv)
    return {"model": ns.model, "variant": ns.variant, "delta": ns.delta, "seqlen": ns.seqlen, "max_windows": ns.max_windows,
            "seed": ns.seed, "line_norm": ns.line_norm}


def parse_pearlc_activations(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260930)
    ns, _ = ap.parse_known_args(argv)
    return {"model": ns.model, "window": ns.window, "delta": ns.delta, "line_norm": ns.line_norm, "tiles": ns.tiles, "seed": ns.seed}


def parse_pearlc_headline(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--line-norm", type=float, default=16.0)
    ap.add_argument("--cap", type=float, default=1 / 400)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seed", type=int, default=20260930)
    ns, _ = ap.parse_known_args(argv)
    return {"model": ns.model, "tokens": ns.tokens, "window": ns.window, "delta": ns.delta, "line_norm": ns.line_norm,
            "cap": ns.cap, "layers": ns.layers, "seed": ns.seed}


def parse_pearlc_split7b(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--budget-share", type=float, default=0.01)
    ns, _ = ap.parse_known_args(argv)
    return {"model": ns.model, "seqlen": ns.seqlen, "eval_windows": ns.eval_windows, "calib_tokens": ns.calib_tokens,
            "delta": ns.delta, "budget_share": ns.budget_share}


# Pearl-C (docs/pouw/cheap-binding.md in the research store): TT_OUT's falsification census, and the quality of U.
POUW_PEARLC_CENSUS = Tool(name="pouw_pearlc_census", version="1", command=["python", "benchmarks/pouw/pearlc_census.py"],
                          closure=["benchmarks/pouw/pearlc_census.py", "benchmarks/pouw/run_result.py",
                                   "packages/verity/src/verity/ml/tc/**"],
                          key_params=["ks", "deltas", "tile", "families", "run", "forming", "line_norm", "atom"],
                          produces=["bench-result/v1", "run-files/v1"], parse=parse_pearlc_census)
POUW_PEARLC_ACTIVATIONS = Tool(name="pouw_pearlc_activations", version="1", command=["python", "benchmarks/pouw/pearlc_activations.py"],
                               closure=["benchmarks/pouw/pearlc_activations.py", "benchmarks/pouw/pearlc_census.py",
                                        "benchmarks/pouw/run_result.py", "packages/verity/src/verity/ml/tc/**"],
                               key_params=["model", "window", "delta", "line_norm", "tiles", "seed", "atom", "G"],
                               produces=["bench-result/v1", "run-files/v1"], parse=parse_pearlc_activations)
POUW_PEARLC_HEADLINE = Tool(name="pouw_pearlc_headline", version="1", command=["python", "benchmarks/pouw/pearlc_headline.py"],
                            closure=["benchmarks/pouw/pearlc_headline.py", "benchmarks/pouw/pearlc_census.py",
                                     "benchmarks/pouw/run_result.py", "packages/verity/src/verity/ml/tc/**"],
                            key_params=["model", "tokens", "window", "delta", "line_norm", "cap", "layers", "seed"],
                            produces=["bench-result/v1", "run-files/v1"], parse=parse_pearlc_headline)
POUW_PEARLC_SPLIT7B = Tool(name="pouw_pearlc_split7b", version="1", command=["python", "benchmarks/pouw/pearlc_split7b.py"],
                           closure=["benchmarks/pouw/pearlc_split7b.py", "benchmarks/pouw/pearlc_headline.py",
                                    "benchmarks/pouw/pearlc_census.py", "benchmarks/pouw/run_result.py"],
                           key_params=["model", "seqlen", "eval_windows", "calib_tokens", "delta", "budget_share"],
                           produces=["bench-result/v1", "run-files/v1"], parse=parse_pearlc_split7b)
POUW_PEARLC_QUALITY = Tool(name="pouw_pearlc_quality", version="1", command=["python", "benchmarks/pouw/pearlc_quality.py"],
                           closure=["benchmarks/pouw/pearlc_quality.py", "benchmarks/pouw/run_result.py"],
                           key_params=["model", "variant", "delta", "seqlen", "max_windows", "seed", "line_norm"],
                           produces=["bench-result/v1", "run-files/v1"], parse=parse_pearlc_quality)
