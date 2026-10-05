"""Rows per unit by class and the constraint counts of a compiled system, per gadget scope.

    python -m archive.direct.ligero.fp8.census            # FP8 (unit and chain) next to the BF16 chain unit
"""
from __future__ import annotations

import json

from ..compile import System

__all__ = ["census", "by_scope"]


def census(sys: System) -> dict:
    kinds = sys.summary()["by_kind"]
    return {"rows_per_unit": sys.m, "bit_rows": kinds.get("bit", 0), "selector_rows": kinds.get("sel", 0),
            "hint_rows": kinds.get("hint", 0), "pin_rows": kinds.get("pin", 0), "product_rows": kinds.get("prod", 0),
            "inverse_rows": kinds.get("inv", 0), "linear_constraints": len(sys.linear), "quadratic_constraints": len(sys.quadratic),
            "chain": sys.chain is not None}


def by_scope(sys: System, depth: int = 2) -> dict[str, dict[str, int]]:
    """Rows per (scope prefix of ``depth`` components, kind)."""
    out: dict[str, dict[str, int]] = {}
    for r in sys.rows:
        parts = r.name.split(".")
        key = ".".join(parts[:depth]) if r.kind != "pin" else "pins"
        d = out.setdefault(key, {})
        d[r.kind] = d.get(r.kind, 0) + 1
        d["total"] = d.get("total", 0) + 1
    return out


def main() -> None:
    from verity_numerical.checker.params import REAL

    from ..unit import compile_unit
    from .relation import ADA, compile_fp8_unit

    for name, s in (("fp8-ada unit", compile_fp8_unit(ADA, chain=False)), ("fp8-ada chain", compile_fp8_unit(ADA, chain=True)),
                    ("bf16 chain", compile_unit(REAL, chain=True))):
        print(name, json.dumps(census(s)))
    print("fp8-ada chain by scope:")
    for k, v in by_scope(compile_fp8_unit(ADA, chain=True)).items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
