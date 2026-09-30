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


def parse_fp4_emulation(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--family", default="gaussian")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--check-atoms", default="4,32")
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=5)
    ns, _ = ap.parse_known_args(argv)
    return {"cmd": ns.cmd, "family": ns.family, "seed": ns.seed, "check_atoms": ns.check_atoms, "seconds": ns.seconds, "reps": ns.reps}


def parse_fp4_routes(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("cmd", nargs="?", default="routes")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--families", default="gaussian,t4,massive,spread,spread-both")
    ap.add_argument("--hot-bits", default="8,10,12")
    ap.add_argument("--hot-rule", default="word")
    ap.add_argument("--seed", type=int, default=20260930)
    ns, _ = ap.parse_known_args(argv)
    return {"cmd": ns.cmd, "k": ns.k, "rows": ns.rows, "cols": ns.cols, "families": ns.families, "hot_bits": ns.hot_bits,
            "hot_rule": ns.hot_rule, "seed": ns.seed}


FP4_EMULATION_CLOSURE = ["benchmarks/pouw/fp4_emulation.py", "benchmarks/pouw/kernels/fp4_emulation.cu",
                         "benchmarks/pouw/kernels/fp4_emulation_gpu.py", "benchmarks/pouw/run_result.py",
                         "packages/verity/src/verity/ml/tc/**", "packages/verity/src/verity/errors.py"]


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
#: The NVFP4 atom's emulation routes on the card (sm_120a): words checked against the pinned model, rates against native.
POUW_FP4_EMULATION = Tool(name="pouw_fp4_emulation", version="1", command=["python", "benchmarks/pouw/kernels/fp4_emulation_gpu.py"],
                          closure=FP4_EMULATION_CLOSURE,
                          key_params=["cmd", "family", "seed", "check_atoms", "seconds", "reps"],
                          produces=["bench-result/v1", "run-files/v1"], parse=parse_fp4_emulation)
#: The same routes' bit-exactness on real-length chains, on the CPU (the pinned atom and BF16 step, vectorised).
POUW_FP4_ROUTES = Tool(name="pouw_fp4_routes", version="1", command=["python", "benchmarks/pouw/fp4_emulation.py"],
                       closure=FP4_EMULATION_CLOSURE, key_params=["cmd", "k", "rows", "cols", "families", "hot_bits", "hot_rule", "seed"],
                       produces=["bench-result/v1", "run-files/v1"], parse=parse_fp4_routes)
