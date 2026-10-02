"""A row's q-word counts (`class_statement --counts-only`) cut in parallel: one process per chunk of its Call Definitions, merged.

    cut_parallel.py PROGRAM_JSON OUT_DIR [--jobs 8] [--python PY]

Q_word cuts each Call on its own, so the counts of disjoint sets of Call Definitions merge exactly: their `calls` and
`per_definition` are disjoint, and a shape's `units` and `row_units` add. The Definitions are dealt to the chunks by an estimated
cost (an attention Call's key count T, which its cut time follows; any other Definition is cheap), so a row whose one-process cut
holds every Definition in one Program (hours, and tens of GB, for 1,151 attention key counts) takes minutes. Writes OUT_DIR/
counts.json (the token-select Calls left out, as `--select` does), counts-all.json (with them) and shapes.tsv (as
73-sweep-shape.sh MODE=cut writes it).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def cost(fid: str) -> int:
    m = re.search(r"(?<=[{,])T=(\d+)", fid)
    return int(m[1]) if fid.startswith("Attention") and m else 50


def deal(calls: dict[str, int], jobs: int) -> list[dict[str, int]]:
    bins: list[tuple[int, dict[str, int]]] = [(0, {}) for _ in range(jobs)]
    for fid in sorted(calls, key=lambda f: (-cost(f), f)):
        i = min(range(jobs), key=lambda k: bins[k][0])
        bins[i] = (bins[i][0] + cost(fid), bins[i][1] | {fid: calls[fid]})
    return [b for _, b in bins if b]


def merge(parts: list[dict]) -> dict:
    out = {"calls": {}, "per_definition": {}, "shapes": {}}
    for p in parts:
        assert not set(out["calls"]) & set(p["calls"])
        out["calls"] |= p["calls"]
        out["per_definition"] |= p["per_definition"]
        for sh, s in p["shapes"].items():
            o = out["shapes"].setdefault(sh, {"definition": s["definition"], "units": 0, "row_units": 0, "word_gates": s["word_gates"]})
            assert (o["definition"], o["word_gates"]) == (s["definition"], s["word_gates"]), sh
            o["units"] += s["units"]
            o["row_units"] += s["row_units"] or 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("program_json")
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    from verity_flock import class_sweep as CW
    calls = CW.graph_calls(json.loads(Path(a.program_json).read_text()))
    kept = {f: n for f, n in calls.items() if not CW.SAMPLER.match(f)}
    left = {f: n for f, n in calls.items() if CW.SAMPLER.match(f)}
    work = a.out / "cut-parallel"
    work.mkdir(parents=True, exist_ok=True)
    chunks = [(f"c{i}", c, True) for i, c in enumerate(deal(kept, a.jobs))] + ([("sampler", left, False)] if left else [])

    def run(item) -> dict:
        name, defs, select = item
        (work / f"{name}.defs.json").write_text(json.dumps(defs, indent=1))
        cmd = [a.python, "-m", "verity_flock.class_statement", "--definitions", str(work / f"{name}.defs.json"), "--partition",
               "q-word", "--out", str(work / name), "--counts-only"] + (["--select", "units"] if select else [])
        with open(work / f"{name}.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: {len(defs)} Call Definitions", flush=True)
        return json.loads((work / name / "counts.json").read_text())

    with ThreadPoolExecutor(a.jobs) as ex:
        got = dict(zip([n for n, _, _ in chunks], ex.map(run, chunks)))
    counts = merge([got[n] for n, _, s in chunks if s])
    assert counts["calls"] == kept
    every = merge(list(got.values()))
    for name, c in (("counts.json", counts), ("counts-all.json", every)):
        tmp = a.out / f".{name}.tmp"
        tmp.write_text(json.dumps(c, sort_keys=True))
        os.replace(tmp, a.out / name)
    defs: dict[str, list[str]] = {}
    for fid, per in counts["per_definition"].items():
        for sh in per:
            defs.setdefault(sh, []).append(fid)
    rows = sorted(counts["shapes"].items(), key=lambda kv: (kv[1]["word_gates"], kv[0]))
    (a.out / "shapes.tsv").write_text("".join(f"{sh}\t{s['word_gates']}\t{s['row_units']}\t{json.dumps(sorted(defs.get(sh, [])))}\n"
                                              for sh, s in rows))
    g = sum(s["row_units"] * s["word_gates"] for s in counts["shapes"].values())
    ga = sum(s["row_units"] * s["word_gates"] for s in every["shapes"].values())
    print(f"{len(rows)} shapes, {sum(s['row_units'] for _, s in rows)} row units, {g} gates under --select; "
          f"{sum(s['row_units'] for s in every['shapes'].values())} units, {ga} gates with every Call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
