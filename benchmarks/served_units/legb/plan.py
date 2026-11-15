"""plan.py: leg B's waves from wave 2, the units files wave.sh stages and proves on node A, from the cheap read's 128 types.

Run from this directory: `python3 plan.py` writes wave2.units ... waveN.units and prints a line a wave.

- The types: form128.units, the cheap read's 128 block types (its form128.read.tsv: no type refused).
- Their AND gates: the census in benchmarks/spot_figures/cards/workload.glm-5.3.json (MoeRouter's, absent there, as staged).
- Wave 1 (wave1.units): a type whose prover exited 0 (wave1.seconds.tsv) is done and planned no more. The head block the
  GPU prover refused (exit 2: its A past GPU_NNZ_MAX) produced no proof; it is planned again, and routed to the CPU prover.
- Sizes and staging time a statement, by family, from the form pass and wave 1's stagings: DsaMqaLogits 290 bytes an AND;
  the head and split blocks a line, 1,766 bytes an AND less 0.53 GB (wave 1's NVIS=58 head block: 61.3 GB at 34.3M);
  the singles at their own form-pass rates.
- Route, predicted: the head and split blocks' A nonzeros at wave 1's NVIS=58 rate (6,091,870,711 at 34,336,169 ANDs);
  past window.GPU_NNZ_MAX is the CPU prover's. Every other family is under a tenth of that rate. stage_prove.sh routes
  each statement from its own staged nets; the prediction only labels the plan.
- Waves: the T=9 split block first; then each wave takes the largest remaining type of every family and fills largest
  first to CAP GB of circuits (node A's 600 GB of /dev/shm, less a margin). Lines are largest first, with window.py's
  estimates: peak RSS x 1.15 + 2 GB, disk x 1.05 + 1 GB, staging seconds x 1.2, device GiB min(135, max(4, GB)).
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import window  # noqa: E402

CAP = 540
FIRST = "SparseMlaSplit_v1{T=9,"
HEAD_NNZ_A_PER_AND = 6_091_870_711 / 34_336_169
SINGLES = {"AllReduceMnnvl_v1": (246, 43, 2.1), "SiluMul_v2": (248, 44, 1.2), "Fp8GroupQuantUe8m0_v1": (158, 25, 1.05),
           "MoeRouterSigmoidBiasSlots_v2": (528, 94, 2.0), "RMSNormCudaVector8_v1": (248, 44, 1.2)}
MOE = 'MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}'


def types(lines):
    return [ln.split("\t")[0].rsplit("@", 1)[0].removeprefix("census:") for ln in lines
            if ln.strip() and not ln.startswith("#")]


def size(t: str, ands: int) -> dict:
    f, m = t.split("{")[0], ands / 1e6
    if f == "DsaMqaLogits_v3":
        b, s, pk, cpu = 290 * ands, (25 if m > 8 else 35) * m, 1.24 * m, False
    elif f in ("SparseMlaHeadBlock_v1", "SparseMlaSplit_v1"):
        b, s, pk = max(0.88e9, 1766 * ands - 0.53e9), 84 * m, 1.93 * m
        cpu = ands * HEAD_NNZ_A_PER_AND > window.GPU_NNZ_MAX
    else:
        bb, ss, pp = SINGLES[f]
        b, s, pk, cpu = bb * ands, ss * m, max(1.4, pp * m), False
    return dict(t=t, f=f, ands=ands, gb=b / 1e9, s=s, peak=pk, cpu=cpu)


def waves(rows: list[dict]) -> list[list[dict]]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["f"]].append(r)
    for f in by:
        by[f].sort(key=lambda r: -r["gb"])
    fams = sorted(by, key=lambda f: -by[f][0]["gb"])
    out = []
    while any(by.values()):
        w, g = [], 0.0
        first = [r for r in by["SparseMlaSplit_v1"] if r["t"].startswith(FIRST)]
        for r in first + [by[f][0] for f in fams if by[f]]:
            if r in by[r["f"]] and g + r["gb"] <= CAP and r["f"] not in {x["f"] for x in w}:
                w.append(r)
                by[r["f"]].remove(r)
                g += r["gb"]
        for r in sorted((r for f in fams for r in by[f]), key=lambda r: -r["gb"]):
            if g + r["gb"] <= CAP:
                w.append(r)
                by[r["f"]].remove(r)
                g += r["gb"]
        out.append(w)
    return out


def main() -> int:
    wl = json.loads((HERE.parents[2] / "benchmarks/spot_figures/cards/workload.glm-5.3.json").read_text())
    ands: dict[str, int] = {}
    for s in wl["value"]["strata"]:
        ands.setdefault(s["block"], s["ands"])
    ands.setdefault(MOE, 8_313_579)
    with open(HERE / "wave1.seconds.tsv") as fh:
        done = {r["name"] for r in csv.DictReader(fh, delimiter="\t") if r["exit"] == "0"}
    every = types((HERE / "form128.units").read_text().splitlines())
    left = [t for t in every if window.name(f"census:{t}", "1") not in done]
    assert len(left) == len(every) - len(done), (len(left), len(every), len(done))
    plan = waves([size(t, ands[t]) for t in left])
    for i, w in enumerate(plan, 2):
        w.sort(key=lambda r: -r["gb"])
        cpu = sum(r["cpu"] for r in w)
        fam = collections.Counter(r["f"].split("_")[0] for r in w)
        head = (f"# leg B, wave {i} of {len(plan) + 1}: {len(w)} types, about {sum(r['gb'] for r in w):.0f} GB of circuits, "
                f"{cpu} predicted past the GPU prover's bound (the CPU prover's); largest first")
        lines = [head] + [f"census:{r['t']}@1\t{r['peak'] * 1.15 + 2:.0f}\t{r['gb'] * 1.05 + 1:.0f}\t{r['s'] * 1.2:.0f}\t"
                          f"{min(135, max(4, r['gb'])):.0f}" for r in w]
        (HERE / f"wave{i}.units").write_text("\n".join(lines) + "\n")
        print(f"wave {i}: {len(w)} types, {sum(r['gb'] for r in w):.0f} GB, {cpu} to the CPU prover "
              f"({sum(r['gb'] for r in w if r['cpu']):.0f} GB), staging {sum(r['s'] for r in w) / 3600:.1f} statement-h, "
              f"longest {max(r['s'] for r in w) / 60:.0f} min, {dict(fam)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
