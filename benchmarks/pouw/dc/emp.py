"""RT-H200-EMP: every timed loop of a probe run against the data-sheet machine's facets, DS-90's rows (sm_90).

  uv run python emp.py RUN_OUT SASS_DIR [--json OUT]

RUN_OUT is `emp.sh time`'s output (each binary's rows.jsonl and gate.json under RUN_OUT/<binary>/); SASS_DIR is
`emp.sh build`'s sass/ (cuobjdump -sass of each binary). A row's rates are the gated patterns' warp instructions a SM
a clock; a loop's trip rate is one of them over its count, and every opcode in the loop's SASS (head to tail, the
gate's addresses) runs at its count times that. Each facet's occupancy is its SMSP-cycles over SMSP-cycles:

  issue   1 a warp instruction, every opcode, uniform included (A2, A2u)
  tensor  MACs over a quarter of the type's dense peak a SM-clock (the H200 card's rate.tensor over the format's weight,
          verity/hardware/cards/h200-sxm-141gb.toml); a 2:4 sparse form at its dense equivalent (A3); BGMMA/BMMA none
          (DS-90 prices b1 on issue alone)
  alu     2 (logic, shift, LEA, IADD3, compares: 64 results a SM-clock, A4); imad 2 (IMAD*); shfl 4 (SHFL)

The sheet holds a loop iff every facet is at most 1. Any facet above 1 is a measured rate above the sheet's ceiling.
Also printed: the SM clock each probe ran at (A1: fmax, the card's clock.sm) and the tensor slots a SM-clock (wgmma s8
n256 = 128).
The clock is read two ways: NVML's sample during the item, and the event clock, the longest block's cycles over the
item's event time a call (times.jsonl), a lower bound on the clock the loop ran at. Under a saturated wgmma the two
disagree (NVML reads up to 28% high), so a rate a second is the MAC/clk times the event clock, not NVML's.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from verity.hardware import load, number

CARD = load("h200-sxm-141gb")
FMAX_MHZ = CARD["clock"]["sm"]["value"]
PEAK = {f.upper(): int(number(CARD["rate"]["tensor"]) / number(CARD["weight"][f]))
        for f in ("s8", "u8", "e4m3", "e5m2", "f16", "bf16", "tf32")}
ALU = ("LOP3", "IADD3", "SHF", "LEA", "ISETP", "IMNMX", "SEL", "PRMT", "FLO", "POPC", "BREV", "IABS", "VIADD", "BMSK",
       "SGXT", "PLOP3", "VIMNMX")
LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][\w.]*)")
FUNC = re.compile(r"Function : (\S+)")
GMMA = re.compile(r"^([IQHB])GMMA\.(?:SP\.)?(\d+)x(\d+)x(\d+)")
MMA = re.compile(r"^([IHB])MMA\.(?:SP\.)?(\d{2})(\d)(\d+)")


def read_sass(d: Path) -> dict[str, list[tuple[int, str]]]:
    """Every function of every dump: its (address, dotted opcode) list."""
    out: dict[str, list[tuple[int, str]]] = {}
    for f in sorted(d.glob("*.sass")):
        cur = None
        for ln in f.read_text(errors="replace").splitlines():
            m = FUNC.search(ln)
            if m:
                cur = out.setdefault(m.group(1), [])
                continue
            m = LINE.search(ln)
            if m and cur is not None:
                cur.append((int(m.group(1), 16), m.group(2)))
    return out


def dtype(op: str, probe: str) -> str:
    for t in ("E4M3", "E5M2", "BF16", "TF32", "U8", "S8"):
        if f".{t}" in op:
            return t
    if op.startswith(("HGMMA", "HMMA")):
        return "F16"
    return "S8" if "s8" in probe else "F16" if "f16" in probe else "E4M3" if "e4m3" in probe else "S8"


def tensor_cycles(op: str, probe: str) -> tuple[float, float]:
    """(SMSP-cycles a warp instruction on DS_tensor, dense-equivalent MACs a warp instruction); (0, 0) off the pipe."""
    sparse = ".SP" in op or probe.startswith("wgsp_")
    m = GMMA.match(op)
    if m:
        if m.group(1) == "B":
            return 0.0, 0.0
        macs = int(m.group(2)) * int(m.group(3)) * int(m.group(4)) / 4  # a warpgroup's 4 warps issue it once each
    else:
        m = MMA.match(op)
        if not m or m.group(1) == "B":
            return 0.0, 0.0
        k = int(m.group(4))
        macs = int(m.group(2)) * int(m.group(3)) * k
    if sparse:
        macs /= 2
    return macs / (PEAK[dtype(op, probe)] / 4), macs


def facets(op: str, probe: str) -> dict[str, float]:
    c = {"issue": 1.0}
    t, _ = tensor_cycles(op, probe)
    if t:
        c["tensor"] = t
    if op.startswith(ALU) and not op.startswith("SHFL"):
        c["alu"] = 2.0
    if op.startswith(("IMAD", "IMUL")):
        c["imad"] = 2.0
    if op.startswith("SHFL"):
        c["shfl"] = 4.0
    return c


def loops_of(rec: dict, rates: dict) -> list[tuple[int, dict]]:
    """(expect index, gate loop) pairs: the loop whose gated counts are nonzero on the index's patterns."""
    idx: dict[int, list[str]] = {}
    for key in rates:
        i, pat = key.split(":", 1)
        idx.setdefault(int(i), []).append(pat)
    out = []
    for i, pats in sorted(idx.items()):
        best = max(rec["loops"], key=lambda L: sum(1 for p in pats if L["counts"].get(p, 0) > 0))
        out.append((i, best))
    return out


def event_mhz(times: Path) -> dict[str, float]:
    """Each probe's event clock: the median over its items of the longest block's cycles over the event time a call."""
    per: dict[str, list[float]] = {}
    if times.exists():
        for ln in times.read_text().splitlines():
            t = json.loads(ln) if ln.strip() else {}
            if t.get("op") == "time" and t.get("cyc") and t.get("ms"):
                per.setdefault(t["probe"], []).append(max(t["cyc"]) / (t["ms"] / t["calls"]) / 1e3)
    return {p: statistics.median(v) for p, v in per.items()}


def analyze(run: Path, sass: dict) -> list[dict]:
    res = []
    for rows in sorted(run.rglob("rows.jsonl")):
        gate = json.loads((rows.parent / "gate.json").read_text())["verdicts"]
        recs = {v["probe"]: v["verdict"] for v in gate}
        ev = event_mhz(rows.parent / "times.jsonl")
        for ln in rows.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            rec = recs.get(r["probe"])
            if not rec or rec["kernel"] not in sass:
                continue
            fn = sass[rec["kernel"]]
            per_op: Counter = Counter()
            for i, L in loops_of(rec, r["rates"]):
                pat = next((p for k, v in r["rates"].items() for p in [k.split(":", 1)[1]]
                            if int(k.split(":")[0]) == i and L["counts"].get(p, 0) > 0), None)
                if pat is None:
                    continue
                trip = r["rates"][f"{i}:{pat}"] / L["counts"][pat]
                lo, hi = int(L["head"], 16), int(L["tail"], 16)
                for a, op in fn:
                    if lo <= a <= hi:
                        per_op[op] += trip
            occ: dict[str, float] = {}
            macs = 0.0
            for op, rate in per_op.items():
                for f, c in facets(op, r["probe"]).items():
                    occ[f] = occ.get(f, 0.0) + c * rate / 4
                macs += tensor_cycles(op, r["probe"])[1] * rate
            uni = sum(v for op, v in per_op.items() if op.startswith("U")) / 4
            res.append({"binary": r["binary"], "probe": r["probe"], "run": r["run"], "sm_mhz": r["sm_mhz"],
                        "event_mhz": ev.get(r["probe"]), "spread": r["spread"], "tags": r["tags"], "counted": r["counted"], "rate": r["rate"],
                        "occupancy": occ, "uniform_issue": uni, "macs_sm_clk": macs,
                        "worst": max(occ, key=occ.get) if occ else None,
                        "worst_occ": max(occ.values()) if occ else 0.0})
    return res


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("sass", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = analyze(a.run, read_sass(a.sass))
    over = [x for x in res if x["worst_occ"] > 1.0]
    print(f"{'probe':<28} {'MHz':>5} {'evMHz':>6} {'issue':>6} {'uni':>6} {'tensor':>6} {'alu':>6} {'imad':>6} "
          f"{'shfl':>6} {'MAC/clk':>8}  worst")
    for x in sorted(res, key=lambda x: -x["worst_occ"]):
        o = x["occupancy"]
        print(f"{x['probe']:<28} {x['sm_mhz'] or 0:>5.0f} {x['event_mhz'] or 0:>6.1f} {o.get('issue', 0):>6.3f} "
              f"{x['uniform_issue']:>6.3f} "
              f"{o.get('tensor', 0):>6.3f} {o.get('alu', 0):>6.3f} {o.get('imad', 0):>6.3f} {o.get('shfl', 0):>6.3f} "
              f"{x['macs_sm_clk']:>8.1f}  {x['worst']} {x['worst_occ']:.4f}{' OVER' if x['worst_occ'] > 1 else ''}"
              f"{' [' + ','.join(x['tags']) + ']' if x['tags'] else ''}")
    mhz = [x["sm_mhz"] for x in res if x["sm_mhz"]]
    if mhz:
        print(f"A1: SM clock during the probes median {statistics.median(mhz):.0f} MHz, max {max(mhz):.0f} "
              f"(fmax {FMAX_MHZ}); {'OVER' if max(mhz) > FMAX_MHZ else 'held'}")
    ev = [x["event_mhz"] for x in res if x["event_mhz"]]
    if ev:
        print(f"   event clock median {statistics.median(ev):.1f} MHz, max {max(ev):.1f}")
    print(f"{len(res)} loops priced, {len(over)} above a DS-90 facet")
    if a.json:
        a.json.write_text(json.dumps({"rows": res, "over": [x["probe"] for x in over]}, indent=1))
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
