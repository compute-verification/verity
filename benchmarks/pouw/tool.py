"""Store declarations for the PoUW harness (research.store README §6.1).

    research run ... --tool pouw_vllm -- python benchmarks/pouw/vllm_bench.py --model /root/models/qwen2.5-3b-instruct --out $D/vllm.json

Key params: the model and the modes.  ``--out`` says where and does not enter the derivation.  Imports nothing but
``research.store.tool``.
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


def parse_vllm(argv: list[str]) -> dict:
    ap = _parser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--modes", default="bf16,fp8,w7a7,ncp,pearl")
    ns, _ = ap.parse_known_args(argv)
    return {"model": (ns.model or "").rstrip("/").rsplit("/", 1)[-1], "modes": ns.modes}


POUW_VLLM = Tool(name="pouw_vllm", version="1", command=["python", "benchmarks/pouw/vllm_bench.py"], closure=CLOSURE,
                 key_params=["model", "modes"], produces=["bench-result/v1", "run-files/v1"], parse=parse_vllm)
