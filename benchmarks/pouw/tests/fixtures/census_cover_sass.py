"""Writes census_cover_sass.json, test_census_cover.py's SASS witnesses, from the red-team runs that timed them.

Each witness is a bit-exact fold or checkpoint program as the run's own SASS shows its timed loop: opcode counts, guard
predicates dropped. Fetch each run (`research fetch RUN --all --runs-dir DIR`) into one directory, and each artifact
holding a run's SASS beside it (`research data fetch ART --to DIR/ART`, the colon a hyphen), then
`uv run benchmarks/pouw/tests/fixtures/census_cover_sass.py DIR`.

- cpw_routes (`benchmarks/pouw/theory/cpw_routes.cu`): every mode's loop from `out/sass/counts.json`, kept
  when `out/device.txt` says `all_bitexact=1`.
- rt3_dc's lane loops (`benchmarks/pouw/rt3_dc/cpatom.cuh`): each `lr_*` and `lr8w_*` loop of the SASS gate's
  verdicts (`out/gate-*.json`, the loop's `top`), kept when `out/exact.jsonl` records its form with 0 mismatches.
- red_generic (`benchmarks/pouw/rt4_sheet/red_generic.cu`): each `li_loop<CP, generic>` (the checkpoint's
  XOR into L2 through a generic address, ATOM.E) in the artifact's SASS for sm_90a and sm_100a, its longest backward
  branch, kept when the run's `out/device.txt` records that form with 0 mismatches (on the run's own device).
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
RED = {"r20261020-001850-def6": ("art:9463456f9f", "benchmarks/pouw/rt4_sheet/red_generic.cu @ 4a028ebe51 (cursor/ca-rt4-sheet-ea24)")}
RED_ARCH = {"sm_90": "sm_90a", "sm_100": "sm_100a"}
RED_CP = ("red", "red2", "red8")
LI_LOOP = re.compile(r"li_loopILi(\d)ELi1E")
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


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


def red_generic(run: Path, art: Path, src: str) -> list[dict]:
    exact = {cp for cp, gen, n in re.findall(r"li_exact<(\w+), (\w+)> .*mismatches=(\d+)", (run / "out" / "device.txt")
                                                .read_text()) if gen == "generic" and n == "0"}
    out = []
    for dev, arch in RED_ARCH.items():
        parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + (art / f"red_generic_{arch}.sass").read_text())
        for name, body in sorted(zip(parts[1::2], parts[2::2]), key=lambda nb: nb[0]):
            m = LI_LOOP.search(name)
            if not m or RED_CP[int(m[1])] not in exact:
                continue
            ins = [(int(a, 16), op, rest) for a, op, rest in INS.findall(body)]
            back = [(int(x, 16), a) for a, op, rest in ins if op.startswith("BRA")
                    for x in re.findall(r"0x([0-9a-f]+)", rest) if int(x, 16) < a]
            head, tail = max(back, key=lambda lp: lp[1] - lp[0])
            counts: dict[str, int] = {}
            for a, op, _ in ins:
                if head <= a <= tail:
                    counts[op] = counts.get(op, 0) + 1
            out.append({"device": dev, "program": f"red_generic li_loop<{RED_CP[int(m[1])]}, generic>", "source": src,
                        "run": run.name, "counts": dict(sorted(counts.items()))})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="a directory holding each fetched run's directory (r2026…)")
    a = ap.parse_args(argv)
    witnesses = [w for r, (dev, src) in CPW.items() for w in cpw(a.runs / r, dev, src)]
    witnesses += [w for r, (dev, src) in DC.items() for w in dc(a.runs / r, dev, src)]
    witnesses += [w for r, (art, src) in RED.items() for w in red_generic(a.runs / r, a.runs / art.replace(":", "-"), src)]
    about = ("Bit-exact fold and checkpoint programs as their runs' SASS shows each timed loop (opcode counts a trip), "
             "for test_census_cover.py: cpw_routes' every mode (RT3-120; out/sass/counts.json, all_bitexact=1), "
             "rt3_dc's lane loops on the H200 and B200 (RT3-DC; the SASS gate's loop, each form 0 mismatches in "
             "exact.jsonl) and red_generic's generic-address checkpoint loops in the H200's and B200's SASS (RT4-SHEET-7; "
             "art:9463456f9f, each form 0 mismatches on the RTX PRO 6000 in device.txt, and on the H200 and the B200: "
             "r20261020-031119-3113, r20261020-031119-013a, all_bitexact=1). "
             f"Written by census_cover_sass.py: {len(witnesses)} programs.")
    lines = ["{", f'  "about": {json.dumps(about, ensure_ascii=False)},', '  "programs": [']
    lines += [f"    {json.dumps(w, ensure_ascii=False)}{',' * (j + 1 < len(witnesses))}" for j, w in enumerate(witnesses)]
    lines += ["  ]", "}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT.name}: {len(witnesses)} programs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
