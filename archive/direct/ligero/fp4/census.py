"""Rows per unit by class and the constraint counts of the compiled NVFP4 system, per gadget scope, next to FP8's and BF16's.

    python -m archive.direct.ligero.fp4.census
"""
from __future__ import annotations

import json

from ..compile import System
from ..fp8.census import by_scope, census

__all__ = ["census", "by_scope", "census_fp4"]


def census_fp4(sys: System) -> dict:
    """``fp8.census.census`` plus the relation's own counts: bits per VU, shifter count, the digit-vector width."""
    from .relation import POSITIONS, RADIX, STEPS, shifter_plan

    c = census(sys)
    c.update({"steps_per_vu": STEPS, "rows_per_vu": sys.m * STEPS, "bit_rows_per_vu": c["bit_rows"] * STEPS,
              "shifters_per_unit": len(shifter_plan()), "digit_positions": POSITIONS, "digit_radix": RADIX,
              "lookups_per_unit": sum(1 for r in sys.rows if r.kind == "sel" and r.name.endswith(".sel0"))})
    return c


def main() -> None:
    from verity_numerical.checker.params import REAL

    from ..fp8.relation import ADA, compile_fp8_unit
    from ..unit import compile_unit
    from .relation import compile_fp4_unit

    for name, s in (("nvfp4-sm120 unit", compile_fp4_unit(chain=False)), ("nvfp4-sm120 chain", compile_fp4_unit(chain=True)),
                    ("fp8-ada chain", compile_fp8_unit(ADA, chain=True)), ("bf16 chain", compile_unit(REAL, chain=True))):
        print(name, json.dumps(census(s)))
    print("nvfp4-sm120 chain:", json.dumps(census_fp4(compile_fp4_unit(chain=True))))
    print("nvfp4-sm120 chain by scope:")
    for k, v in by_scope(compile_fp4_unit(chain=True)).items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
