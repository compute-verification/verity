"""Writes census_cover_sass.json, test_census_cover.py's SASS witnesses, from the red-team runs that timed them.

Each witness is a bit-exact fold or checkpoint program as the run's own SASS shows its timed loop: opcode counts, guard
predicates dropped. Fetch each run (`research fetch RUN --all --runs-dir DIR`) into one directory, then
`uv run benchmarks/pouw/tests/fixtures/census_cover_sass.py DIR`.

- cpw_routes (`benchmarks/pouw/theory/cpw_routes.cu`, RT3-120): every mode's loop from `out/sass/counts.json`, kept
  when `out/device.txt` says `all_bitexact=1`.
- rt3_dc's lane loops (`benchmarks/pouw/rt3_dc/cpatom.cuh`, RT3-DC): each `lr_*` and `lr8w_*` loop of the SASS gate's
  verdicts (`out/gate-*.json`, the loop's `top`), kept when `out/exact.jsonl` records its form with 0 mismatches.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "census_cover_sass.json"

CPW = {"r20261019-192718-5d46": ("sm_120", "benchmarks/pouw/theory/cpw_routes.cu @ 99015d82e9 (cursor/ca-rt3-120-ea24)")}
DC = {"r20261019-183610-89bd": ("sm_90", "benchmarks/pouw/rt3_dc/cpatom.cuh @ 0c9b5adfac (cursor/ca-rt3-dc-ea24)"),
      "r20261019-192247-8771": ("sm_100", "benchmarks/pouw/rt3_dc/cpatom.cuh @ 0c9b5adfac (cursor/ca-rt3-dc-ea24)")}
LANE_LOOP = re.compile(r"lr8?w?_(\w+)")


def cpw(run: Path, dev: str, src: str) -> list[dict]:
    if "all_bitexact=1" not in (run / "out" / "device.txt").read_text():
        raise SystemExit(f"{run.name}: device.txt has no all_bitexact=1")
    counts = json.loads((run / "out" / "sass" / "counts.json").read_text())
    return [{"device": dev, "program": f"cpw_routes {mode}", "source": src, "run": run.name,
             "counts": dict(sorted(v["counts"].items()))} for mode, v in sorted(counts.items(), key=lambda kv: int(kv[0][4:]))]


def dc(run: Path, dev: str, src: str) -> list[dict]:
    exact = {m[1] for line in (run / "out" / "exact.jsonl").read_text().splitlines() if line.strip()
             for d in [json.loads(line)] if d.get("probe") == "exact" and d["mismatches"] == 0
             for m in [re.fullmatch(r"li_exact<(\w+)>", d["kernel"])] if m}
    (gate,) = (run / "out").glob("gate-*.json")
    out = []
    for v in json.loads(gate.read_text())["verdicts"]:
        m = LANE_LOOP.fullmatch(v["probe"])
        if m and m[1] in exact and v["passed"] and not v["probe"].startswith("lr8_"):
            (loop,) = v["verdict"]["loops"]
            out.append({"device": dev, "program": f"rt3_dc {v['probe']}", "source": src, "run": run.name,
                        "counts": dict(sorted(loop["top"].items()))})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="a directory holding each fetched run's directory (r2026…)")
    a = ap.parse_args(argv)
    witnesses = [w for r, (dev, src) in CPW.items() for w in cpw(a.runs / r, dev, src)]
    witnesses += [w for r, (dev, src) in DC.items() for w in dc(a.runs / r, dev, src)]
    about = ("Bit-exact fold and checkpoint programs as their runs' SASS shows each timed loop (opcode counts a trip), "
             "for test_census_cover.py: cpw_routes' every mode (RT3-120; out/sass/counts.json, all_bitexact=1) and "
             "rt3_dc's lane loops on the H200 and B200 (RT3-DC; the SASS gate's loop, each form 0 mismatches in "
             f"exact.jsonl). Written by census_cover_sass.py: {len(witnesses)} programs.")
    lines = ["{", f'  "about": {json.dumps(about, ensure_ascii=False)},', '  "programs": [']
    lines += [f"    {json.dumps(w, ensure_ascii=False)}{',' * (j + 1 < len(witnesses))}" for j, w in enumerate(witnesses)]
    lines += ["  ]", "}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT.name}: {len(witnesses)} programs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
