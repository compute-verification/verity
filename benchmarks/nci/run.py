"""`verity bench nci`: the slowdown statement's first figure (`statement.slide`), the remote slowdown at tier T1 under the
best plausible PoUW design, computed from `tiers.DEFAULTS` and written as the Metrics tab's number (modeled)."""
from __future__ import annotations

import argparse

import statement as S
import tiers as T
from verity_bench import kit

METRIC = "compute-accounting/remote-slowdown-best-pouw"
UNIT = "times"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    first, *rest = S.slide()
    x = first["value"]
    head = f"nci: {S.num(x)}x, {first['quantity']} (modeled at tiers.py's defaults, MFU_0 = {S.num(T.DEFAULTS['mfu'])})"
    text = "\n".join([head, "", *S.bullets(rest)]).rstrip()
    m = {"name": METRIC, "value": x, "unit": UNIT,
         "inputs": [f"statement.slowdown(tiers.DEFAULTS | {S.BEST}, 'T1')"]}
    return kit.report("nci", text, {"value": x, "unit": UNIT, "slide": [first, *rest], "overrides": S.BEST}, [m],
                      mark="modeled")


if __name__ == "__main__":
    raise SystemExit(main())
