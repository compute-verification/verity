"""uni.sh's `a2u` table (probes/sm100_a2u.cu) into A2u's test: does an SMSP ever issue more than one warp instruction a
cycle, with the uniform datapath and a saturated tcgen05.mma beside saturated lanes?

    python3 a2u.py OUT [--json F]

Each probe has two timed loops: warp 0's fixed one (`iters` trips) and the OP_UNTIL one warps 1-15 run until warp 0
stops (the one that polls the stop flag, an LDS). Both are counted in full from the run's SASS between the loop bounds
the gate recorded (gate-build.json), so a block's warp instructions are fixed x iters + until x trips, and its issue a
SMSP-cycle is that over 4 x the stop's cycles. A2u holds at a probe when that is at most 1 (NOP excluded, since ptxas
pads with NOPs no warp may issue). `uniform_price` is what each added uniform instruction cost the lanes, in issue
slots, against a2_tc_i0: 1 where the ceiling is tight, 0 where the uniform datapath issued for free.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

SASS_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
UTC_CYCLES = 128  # SM clocks one UTCIMMA 128x256x32 holds the tensor pipe at the sheet's 8192 MACs an SM clock
BASE = "a2_tc_i0"


def loops(out: Path) -> dict[str, list[dict[str, int]]]:
    """probe -> its timed loops' full opcode counts, in address order, from the gate's bounds and the SASS."""
    bounds: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for v in json.loads((out / "gate-build.json").read_text())["verdicts"]:
        bounds[(v["binary"], v["probe"])] = [(int(lp["head"], 16), int(lp["tail"], 16)) for lp in v["verdict"]["loops"]]
    res: dict[str, list[dict[str, int]]] = {}
    for f in (out / "sass").glob("*.sass"):
        for chunk in f.read_text().split("Function : ")[1:]:
            name = chunk.split(None, 1)[0]
            bs = bounds.get((f.stem, name))
            if not bs:
                continue
            res[name] = []
            for lo, hi in bs:
                c: dict[str, int] = {}
                for m in SASS_LINE.finditer(chunk):
                    if lo <= int(m[1], 16) <= hi:
                        c[m[2]] = c.get(m[2], 0) + 1
                res[name].append(c)
    return res


def roles(ls: list[dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    """(fixed, until): the until loop is the one that polls the stop flag (an LDS)."""
    if len(ls) != 2:
        raise ValueError(f"{len(ls)} timed loops, not 2")
    polls = [any(k.startswith("LDS") for k in c) for c in ls]
    if polls.count(True) != 1:
        raise ValueError(f"the stop poll (LDS) is in {polls.count(True)} loops, not 1")
    return (ls[1], ls[0]) if polls[0] else (ls[0], ls[1])


def issued(c: dict[str, int]) -> tuple[int, int, int]:
    """(every instruction, NOP excluded, uniform-datapath ones) of one trip."""
    n = sum(c.values())
    return n, n - c.get("NOP", 0), sum(v for k, v in c.items() if k.startswith("U"))


def analyze(out: Path) -> dict:
    full = loops(out)
    times: dict[str, list[dict]] = {}
    for line in (out / "probes" / "times.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t.get("op") == "time" and t["probe"] in full:
            times.setdefault(t["probe"], []).append(t)
    res: dict[str, dict] = {}
    for name, ts in sorted(times.items()):
        fixed, until = roles(full[name])
        (f_all, f_nonop, f_uni), (u_all, u_nonop, _) = issued(fixed), issued(until)
        reps = []
        for t in ts:
            blk = [(tr, st) for tr, st in zip(t["trips"], t["stop"]) if st > 0]
            it = t["iters"]
            reps.append({
                "issue": statistics.median((f_all * it + u_all * tr) / (4 * st) for tr, st in blk),
                "issue_nonop": statistics.median((f_nonop * it + u_nonop * tr) / (4 * st) for tr, st in blk),
                "lanes": statistics.median(u_nonop * tr / (4 * st) for tr, st in blk),
                "uniform": statistics.median(f_uni * it / (4 * st) for tr, st in blk),
                "tensor": statistics.median(fixed.get("UTCIMMA", 0) * it * UTC_CYCLES / st for tr, st in blk),
                "stop": statistics.median(st for tr, st in blk),
            })
        med = {k: statistics.median(r[k] for r in reps) for k in reps[0]}
        res[name] = med | {
            "issue_nonop_max": max(r["issue_nonop"] for r in reps), "reps": len(reps),
            "fixed": fixed, "until": until, "fixed_uniform_a_trip": f_uni,
            "sm_mhz": statistics.median(m for t in ts if (m := (t["sm_mhz"]["during"] or t["sm_mhz"]["after"]))) if any(
                (t["sm_mhz"]["during"] or t["sm_mhz"]["after"]) for t in ts) else None,
            "throttled": any(int(t["throttle"], 16) & 0xEC for t in ts),
        }
    base = res.get(BASE)
    for name, r in res.items():
        if base is None or name == BASE or not name.startswith("a2_tc_"):
            continue
        added = r["uniform"] - base["uniform"]
        if added > 0:
            r["uniform_price"] = (base["lanes"] - r["lanes"]) / added
    worst = max(res.values(), key=lambda r: r["issue_nonop_max"], default=None)
    return {"probes": res, "max_issue_nonop": worst["issue_nonop_max"] if worst else None,
            "a2u_holds": all(r["issue_nonop_max"] <= 1 for r in res.values())}


def text(a: dict) -> str:
    lines = [f"{'probe':<12} {'issue':>7} {'no-NOP':>7} {'max':>7} {'lanes':>7} {'uniform':>8} {'tensor':>7} "
             f"{'u price':>8} {'MHz':>6}"]
    for name, r in a["probes"].items():
        price = f"{r['uniform_price']:.3f}" if "uniform_price" in r else "-"
        lines.append(f"{name:<12} {r['issue']:7.4f} {r['issue_nonop']:7.4f} {r['issue_nonop_max']:7.4f} "
                     f"{r['lanes']:7.4f} {r['uniform']:8.4f} {r['tensor']:7.4f} {price:>8} "
                     f"{r['sm_mhz'] or 0:6.0f}{' THROTTLED' if r['throttled'] else ''}")
    lines.append(f"A2u {'holds' if a['a2u_holds'] else 'FAILS'}: the most warp instructions an SMSP-cycle, NOP "
                 f"excluded, is {a['max_issue_nonop']:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    res = analyze(a.out)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    print(text(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
