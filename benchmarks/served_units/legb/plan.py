"""plan.py: leg B's waves from wave 2, the units files wave.sh stages and proves on node A's cards, and cpu.units, the types
past the GPU prover's bound for CPU nodes, from the cheap read's 128 types.

Run from this directory: `python3 plan.py` writes wave2.units ... waveN.units and cpu.units and prints a line a file.

- The types: form128.units, the cheap read's 128 block types (its form128.read.tsv: no type refused).
- Their AND gates: the census in benchmarks/spot_figures/cards/workload.glm-5.3.json (MoeRouter's, absent there, as staged).
- Wave 1 (wave1.units): a type whose prover exited 0 (wave1.seconds.tsv) is done and planned no more. The head block the
  GPU prover refused (exit 2: its A past GPU_NNZ_MAX) produced no proof; it is planned again, and routed to the CPU prover.
- Sizes and staging time a statement, by family, from the form pass and wave 1's stagings: DsaMqaLogits 290 bytes an AND;
  the head and split blocks a line, 1,766 bytes an AND less 0.53 GB (wave 1's NVIS=58 head block: 61.3 GB at 34.3M);
  the singles at their own form-pass rates.
- Route, predicted: the head and split blocks' A nonzeros at wave 1's NVIS=58 rate (6,091,870,711 at 34,336,169 ANDs);
  past window.GPU_NNZ_MAX is the CPU prover's. Every other family is under a tenth of that rate. The predicted past-bound
  types are cpu.units, for CPU nodes (stage_prove.sh DEV=cpu), and none is on node A's waves. stage_prove.sh still routes
  each statement from its own staged nets: on node A (CPU_PROVE=0) one the prediction put under the bound and its nets put
  past it is listed in the job's out/cpu.units for the CPU nodes, not proved there.
- Waves, the GPU-predicted types only: each takes the largest remaining type of every family and fills largest first to
  CAP GB of circuits (node A's 600 GB of /dev/shm, less a margin). Lines are largest first, with window.py's estimates:
  peak RSS x 1.15 + 2 GB, disk x 1.05 + 1 GB, staging seconds x 1.2, device GiB min(135, max(4, GB)). cpu.units is
  largest first, in the same columns.
- A wave's printed hours, each a guess from wave 1: on node A, SETUP_H (the sync and the builds), staging (the longest
  statement's or STAGE_SLOTS at a time, whichever is longer) and the cards (CARD_S_PER_MAND seconds a million ANDs a
  statement, WARM=1 RUNS=4, its build included, over three cards); then on CPU nodes, lean_shard.sh's restaging and
  its Lean check, LEAN_S_PER_GB seconds a GB of circuit a session (wave 1's N=60), every KEPT session of a statement
  whose one check takes under LEAN_ALL_S seconds and the last alone of the rest, serial in one shard.
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
CARDS, STAGE_SLOTS, CARD_S_PER_MAND, LEAN_S_PER_GB, LEAN_ALL_S, KEPT, SETUP_H = 3, 12, 7.0, 13.7, 120, 5, 0.3
HEAD_NNZ_A_PER_AND = 6_091_870_711 / 34_336_169
SINGLES = {"AllReduceMnnvl_v1": (246, 43, 2.1), "SiluMul_v2": (248, 44, 1.2), "Fp8GroupQuantUe8m0_v1": (158, 25, 1.05),
           "MoeRouterSigmoidBiasSlots_v2": (528, 94, 2.0), "RMSNormCudaVector8_v1": (248, 44, 1.2)}
MOE = 'MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}'


def lean_s(r: dict) -> float:
    one = LEAN_S_PER_GB * r["gb"]
    return one * (KEPT if one < LEAN_ALL_S else 1)


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
        for r in [by[f][0] for f in fams if by[f]]:
            if g + r["gb"] <= CAP:
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
    rows = [size(t, ands[t]) for t in left]
    gpu, cpu = [r for r in rows if not r["cpu"]], sorted((r for r in rows if r["cpu"]), key=lambda r: -r["gb"])
    plan = waves(gpu)
    for old in HERE.glob("wave*.units"):
        if old.stem[4:].isdigit() and old.name != "wave1.units":
            old.unlink()
    for i, w in enumerate(plan, 2):
        w.sort(key=lambda r: -r["gb"])
        gb, fam = sum(r["gb"] for r in w), collections.Counter(r["f"].split("_")[0] for r in w)
        stage_h = max(max(r["s"] for r in w), sum(r["s"] for r in w) / STAGE_SLOTS) / 3600
        card_h = max(sum(CARD_S_PER_MAND * r["ands"] / 1e6 for r in w) / CARDS,
                     max(CARD_S_PER_MAND * r["ands"] / 1e6 for r in w)) / 3600
        write(HERE / f"wave{i}.units", f"# leg B, wave {i} of {len(plan) + 1}, node A's cards: {len(w)} types, about {gb:.0f} GB "
              "of circuits, every one predicted under the GPU prover's bound; largest first", w)
        print(f"wave {i}: {len(w)} types, {gb:.0f} GB, {sum(r['ands'] for r in w) / 1e6:.0f}M ANDs; node A "
              f"{SETUP_H + stage_h + card_h:.1f} h (staging {stage_h:.1f}, longest {max(r['s'] for r in w) / 60:.0f} min; "
              f"cards {card_h:.1f}); CPU nodes' Lean check {sum(map(lean_s, w)) / 3600:.1f} h in one shard, after "
              f"{stage_h:.1f} h restaging ({sum(LEAN_S_PER_GB * r['gb'] < LEAN_ALL_S for r in w)} with every session); "
              f"{dict(fam)}")
    write(HERE / "cpu.units", f"# leg B, the CPU nodes': {len(cpu)} types predicted past the GPU prover's bound, about "
          f"{sum(r['gb'] for r in cpu):.0f} GB of circuits; largest first", cpu)
    print(f"cpu.units: {len(cpu)} types, {sum(r['gb'] for r in cpu):.0f} GB, {sum(r['ands'] for r in cpu) / 1e6:.0f}M ANDs, "
          f"largest {cpu[0]['gb']:.0f} GB ({cpu[0]['ands'] / 1e6:.1f}M ANDs), smallest {cpu[-1]['gb']:.0f} GB; staging "
          f"{sum(r['s'] for r in cpu) / 3600:.1f} statement-h, peak RSS {max(r['peak'] for r in cpu):.0f} GB at most")
    return 0


def write(path: Path, head: str, rows: list[dict]) -> None:
    path.write_text("\n".join([head] + [f"census:{r['t']}@1\t{r['peak'] * 1.15 + 2:.0f}\t{r['gb'] * 1.05 + 1:.0f}\t"
                                         f"{r['s'] * 1.2:.0f}\t{min(135, max(4, r['gb'])):.0f}" for r in rows]) + "\n")


if __name__ == "__main__":
    sys.exit(main())
