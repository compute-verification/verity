"""The PoUW design of record's ε on the Metrics tab, from `machine-model headline`'s rows.

    research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && .venv/bin/python calibration/machine_model/run.py --out "$OUT"'

Computes `headline.record` from the data directory, writes it to OUT/machine_model.json and prints the headline's
tables, then the record as one line of JSON (`kit.report`). Each family's core ε at each reading is written as `METRIC`,
modeled, its member the card and the reading (`h200-measured`), and last `OWN`'s with no member, the metric's own
point. It reads no GPU and measures nothing on the machine.
"""

from __future__ import annotations

import argparse
import re

from verity_bench import kit
from verity_machine_model import headline as H

METRIC = "compute-accounting/pouw-design-epsilon"
#: the row the metric's own point is: the RTX PRO 6000's design at the data sheet's reading, Sm120SheetHeadline's
OWN = ("sm_120", "sheet")


def measured(rec: dict) -> list[dict]:
    rows = {(x["family"], r): x for r, v in rec["readings"].items() for x in v["rows"]}

    def one(x: dict, member: str | None) -> dict:
        m = {"name": METRIC, "value": round(100 * x["core"], 6), "unit": "percent", "mark": "modeled",
             "inputs": [f"modeled: machine-model headline, {x['design']} on {x['vector']} at q* = {x['q']:g}"]}
        return m | ({"member": member} if member else {})

    slug = {(f, r): f"{re.sub(r'[^a-z0-9]+', '-', x['gpu'].lower()).strip('-')}-{r}" for (f, r), x in rows.items()}
    return [one(x, slug[k]) for k, x in rows.items()] + [one(rows[OWN], None)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out")
    a = p.parse_args(argv)
    rec = H.record()
    return kit.report("machine_model", H.text(), rec, measured(rec), out=a.out)


if __name__ == "__main__":
    raise SystemExit(main())
