"""Writes census_programs.json, test_lean_headline.py's census programs, from the census runs' records.

Fetch each run the fixture lists (`research fetch RUN --all`) into one directory, then
`uv run benchmarks/pouw/tests/fixtures/census_programs.py DIR`. A program is kept when facet_break/loads.py puts it
above 0.98 on a row of VF3r-16 or VF3s-16; the rest are too far below 1 for the test to need them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from verity_machine_model.model import census, model, vector

HERE = Path(__file__).resolve().parent
OUT = HERE / "census_programs.json"
VECTORS = ("VF3r-16", "VF3s-16")
KEEP = 0.98


def _loads():
    spec = importlib.util.spec_from_file_location("facet_break_loads", HERE.parents[1] / "facet_break" / "loads.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="a directory holding each fetched run's directory (r2026…)")
    a = ap.parse_args(argv)
    L, m = _loads(), model()
    runs = sorted(p.name for p in a.runs.glob("r2026*"))
    use = []
    for r in runs:
        chunks = sorted((a.runs / r).glob("out/chunks/*/rows.jsonl"))
        use += chunks or [a.runs / r / "out" / "rows.jsonl"]
    rows = L.load_rows(use)
    vs = {n: (set(vector(n).rows), census(m, vector(n).epoch)) for n in VECTORS}
    keep = []
    for r in rows:
        top = max((u for rs, c in vs.values() for k, u in L.program(r, c, m.cycle_u)["load"].items() if k in rs),
                  default=0)
        if top > KEEP:
            x = L.program(r, vs["VF3s-16"][1], m.cycle_u)["x"]
            keep.append([r["run"], r["probe"], {k: x[k] for k in sorted(x)}])
    keep.sort(key=lambda e: (e[0], e[1], json.dumps(e[2], sort_keys=True)))
    about = ("Census programs for the headline vector's test (test_lean_headline.py): each program's x, the warp "
             "instructions of each census class it issues a SMSP-cycle (facet_break/loads.py: load_rows of its run's "
             "out/rows.jsonl, or out/chunks/*/rows.jsonl, then program's x), from the runs VF3s-16's joint issue rows "
             "were fitted on (machine_model/data/census/runs.toml JI3t: JI3s's 17 and RT-FACET's 6), kept when "
             "facet_break/loads.py facet_loads puts it above 0.98 on a row of VF3r-16 or VF3s-16: "
             f"{len(keep)} of {len(rows)} programs.")
    lines = ["{", f'  "about": {json.dumps(about, ensure_ascii=False)},', f'  "runs": {json.dumps(runs)},',
             '  "programs": [']
    lines += [f"    {json.dumps(e, separators=(', ', ': '))}{',' * (j + 1 < len(keep))}" for j, e in enumerate(keep)]
    lines += ["  ]", "}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT.name}: {len(keep)} of {len(rows)} programs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
