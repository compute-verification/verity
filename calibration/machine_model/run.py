"""The machine model's Dither design ε on the Metrics tab: each GPU family's design of record at each reading.

    research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && .venv/bin/python calibration/machine_model/run.py --out "$OUT"'

Computes `headline.record` from the data directory, writes it to OUT/machine_model.json and prints its tables, then the
record as one line of JSON (`kit.report`). Each family's core ε at each reading is written as `METRIC`, modeled, its
member the card and the reading (`h200-measured`), and last `POINT`'s with no member, the metric's point on the tab. It
reads no GPU and measures nothing on the machine.
"""

from __future__ import annotations

import argparse
import re

from verity_bench import kit
from verity_machine_model import headline as H

METRIC = "compute-accounting/machine-model-design-epsilon"
#: the row written with no member: the H200's design at the data sheet's reading, the card the other entries run on
POINT = ("sm_90", "sheet")


def measured(rec: dict) -> list[dict]:
    rows = {(x["family"], r): x for r, v in rec["readings"].items() for x in v["rows"]}

    def one(x: dict, member: str | None) -> dict:
        m = {"name": METRIC, "value": round(100 * x["core"], 6), "unit": "percent", "mark": "modeled",
             "inputs": [f"modeled: machine-model, {x['design']} on {x['vector']} at q* = {x['q']:g}"]}
        return m | ({"member": member} if member else {})

    slug = {k: f"{re.sub(r'[^a-z0-9]+', '-', x['gpu'].lower()).strip('-')}-{k[1]}" for k, x in rows.items()}
    return [one(x, slug[k]) for k, x in rows.items()] + [one(rows[POINT], None)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out")
    a = p.parse_args(argv)
    rec = H.record()
    return kit.report("machine_model", H.text(), rec, measured(rec), out=a.out)


if __name__ == "__main__":
    raise SystemExit(main())
