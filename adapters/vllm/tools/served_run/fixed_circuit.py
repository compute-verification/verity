"""The digest of a served window's Program at its class, before the serve's first request: `window_circuit.fixed_program`
under the registered model the serve's close reads (`close_plan.ClosePlan`, from ``serve.json``). It prints
``{"program": DIGEST}``, and exits 3 with the refusal for a class whose window's circuit the serve chooses.

    fixed_circuit.py --serve-run D --class S,P,O[,LANES,EVERY]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from verity_vllm.inputs.serves import window_class_parts
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.close_plan import ClosePlan


def program_of(serve: Path, cls: str) -> str:
    c = WC.WindowClass(*window_class_parts(cls))
    plan = ClosePlan(serve, c)
    return WC.fixed_program(plan._registered(), c, plan.grain)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--serve-run", type=Path, required=True, help="the serve's --out, holding serve.json")
    p.add_argument("--class", dest="cls", required=True, help="the window class S,P,O[,LANES,EVERY] the serve declared")
    a = p.parse_args(argv)
    try:
        print(json.dumps({"program": program_of(a.serve_run, a.cls)}))
    except Unexpressible as e:
        print(f"FIXED-CIRCUIT refused: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
