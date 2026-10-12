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

from research.pythonpath import package_paths, workspace_root
from research.store.tool import Tool

TREE = workspace_root(__file__)

CLOSURE = ["benchmarks/pouw/**", "verity/core/protocols/pouw/**",
           *package_paths(TREE, "verity.ml.randomness", "**"), *package_paths(TREE, "verity.ml.commitments", "**"),
           *package_paths(TREE, "verity", "errors.py"),
           *package_paths(TREE, "verity.ml.circuits", "silicon/**")]


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
