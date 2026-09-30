"""Store declarations for the FP4 stream's attacker and quality tools on the RTX PRO 6000 (research.store README §6.1).

    research run --on <server> --project verity --campaign pouw --env CUDA_VISIBLE_DEVICES=7 \
        --tool benchmarks.pouw.fp4_redteam_tool:POUW_FP4_QUALITY -- python benchmarks/pouw/fp4_quality.py --device cuda ...
    research run ... --tool benchmarks.pouw.fp4_redteam_tool:POUW_FP4_CENSUS -- python benchmarks/pouw/fp4_census.py synth ...
    research run ... --tool benchmarks.pouw.fp4_redteam_tool:POUW_FP4_ATTACK -- python benchmarks/pouw/kernels/fp4_attack.py peak ...

Imports nothing but ``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.store.tool import Tool


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", default=None)
    return ap


def parse_census(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("cmd", nargs="?", default="synth")
    ap.add_argument("--atom", action="append", default=None)
    ap.add_argument("--frac-bits", type=int, default=None)
    ap.add_argument("--acc-window", type=int, default=None)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--families", default="all")
    ap.add_argument("--noise", default="none,row,block")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260930)
    ns, _ = ap.parse_known_args(argv)
    return {"cmd": ns.cmd, "atom": ",".join(ns.atom or []), "frac_bits": ns.frac_bits, "acc_window": ns.acc_window, "k": ns.k,
            "pairs": ns.pairs, "families": ns.families, "noise": ns.noise, "delta": ns.delta, "max_k": ns.max_k, "seed": ns.seed}


def parse_quality(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--sigma-c", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260930)
    ns, _ = ap.parse_known_args(argv)
    return {"model": ns.model.rstrip("/").rsplit("/", 1)[-1], "seqlen": ns.seqlen, "eval_windows": ns.eval_windows,
            "variants": ns.variants, "sigma_c": ns.sigma_c, "delta": ns.delta, "device": ns.device, "seed": ns.seed}


def parse_attack(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("cmd", nargs="?", default="peak")
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sustain-s", type=float, default=2.0)
    ns, _ = ap.parse_known_args(argv)
    return {"cmd": ns.cmd, "expect_uuid": ns.expect_uuid or "", "warmup": ns.warmup, "reps": ns.reps, "sustain_s": ns.sustain_s}


POUW_FP4_CENSUS = Tool(name="pouw_fp4_census", version="1", command=["python", "benchmarks/pouw/fp4_census.py"],
                       closure=["benchmarks/pouw/fp4_census.py", "benchmarks/pouw/run_result.py", "packages/verity/src/verity/ml/tc/**",
                                "packages/verity/src/verity/errors.py"],
                       key_params=["cmd", "atom", "frac_bits", "acc_window", "k", "pairs", "families", "noise", "delta", "max_k", "seed"],
                       produces=["bench-result/v1", "run-files/v1"], parse=parse_census)
POUW_FP4_QUALITY = Tool(name="pouw_fp4_quality", version="1", command=["python", "benchmarks/pouw/fp4_quality.py"],
                        closure=["benchmarks/pouw/fp4_quality.py", "benchmarks/pouw/run_result.py"],
                        key_params=["model", "seqlen", "eval_windows", "variants", "sigma_c", "delta", "device", "seed"],
                        produces=["bench-result/v1", "run-files/v1"], parse=parse_quality)
POUW_FP4_ATTACK = Tool(name="pouw_fp4_attack", version="1", command=["python", "benchmarks/pouw/kernels/fp4_attack.py"],
                       closure=["benchmarks/pouw/kernels/fp4_attack.py", "benchmarks/pouw/kernels/fp4_attack.cu",
                                "benchmarks/pouw/run_result.py"],
                       key_params=["cmd", "expect_uuid", "warmup", "reps", "sustain_s"],
                       produces=["bench-result/v1", "run-files/v1"], parse=parse_attack)
