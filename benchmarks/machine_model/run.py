"""`verity bench machine_model`: the RTX PRO 6000's headline ε, `machine-model headline`'s sm_120 row, computed from
`data/` and written as the Metrics tab's number: the data-sheet reading, with the measured reading as its member."""
from __future__ import annotations

import argparse
from dataclasses import asdict

from verity_bench import kit
from verity_machine_model import headline as H

METRIC = "compute-accounting/rtx-pro-6000-headline-eps"
UNIT = "share of credit"
FAMILY = "sm_120"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    rows = {r: H.row(FAMILY, r) for r in ("measured", "sheet")}
    sheet, measured = rows["sheet"], rows["measured"]
    head = (f"machine_model: ε {100 * sheet.core:.3f}% for the {sheet.gpu} ({sheet.design}) at the data sheet's reading "
            f"({sheet.vector}, q* = {sheet.q:g}), {100 * measured.core:.3f}% at the measured reading "
            f"({measured.vector}, q* = {measured.q:g}); modeled from data/, not measured")
    doc = {"value": sheet.core, "unit": UNIT, "family": FAMILY, "readings": {r: asdict(x) for r, x in rows.items()}}
    # the metric's own value is written last: the tab reads the last measurement that names no member
    out = [{"name": METRIC, "value": x.core, "unit": UNIT, "member": None if r == "sheet" else r,
            "inputs": [f"{x.design} on {x.vector} at q* = {x.q:g}, operands {H.OPERANDS}, credit {H.CREDIT}"]}
           for r, x in rows.items()]
    return kit.report("machine_model", head + "\n\n" + H.text(), doc, out, mark="modeled")


if __name__ == "__main__":
    raise SystemExit(main())
