"""uni.sh's run of probes/sm100_ldtm.cu into the issue price of tcgen05.ld.32x32b.x128 (LDTM.x128, 4096 words a warp
instruction) on sm_100.

    python3 ldtm.py analyze OUT... [--json F]   each stream's baseline issue, each density's slowdown a load, the price
    python3 ldtm.py tables GATE_JSON            rewrite sm100_ldtm.cu's table from probe.py gate's JSON

A point's time T is SMSP-cycles a warp-iteration of the stream (4 n / rate, as uniform.py's), its loop counted in full from
the run's SASS between the bounds the gate recorded. Each load adds its LDTM and what ptxas puts beside it (the wait's NOP,
the R2UR and LOP3 of the next load's address; probes/sm100_ldtm.cu): `all` charges the whole slowdown to the load,
`ldtm` first charges every other added instruction a whole issue slot, the most one costs, so it is the least the load is
left with. The price is the least `ldtm` over the priced points of a read stream (issue ≥ 0.95 without the loads), less 2%
(as uniform.py prices a uniform instruction); it passes at 0.98 of an issue slot.

A priced point has at least two loads and still issues ≥ 0.95 of its slots. The single-load point is dropped: at one load
ptxas restructures the loop (an extra LDC and LOP3 appear, then vanish at two or more), so its added-opcode count against
the no-load loop is inflated and `ldtm` there is spuriously low. Past about four loads an iteration the TMEM load path, not
the issue slot, binds (the point's own issue falls below 0.95), so those points are dropped too. The finding: the least
`ldtm` is near zero (negative on the FFMA stream that reads its accumulators), so one LDTM.32x32b.x128 costs well under an
issue slot and fails 0.98. It issues its address on the uniform datapath (the R2UR) and reads over the load path, both idle
while the vector issue port is saturated, so a load hides under saturated vector work. The repeats agree in sign and size
but not to r5's 0.002: the saturated load density is capped near four loads an iteration, so the least rests on a low-density
marginal, not r5's 16-128-instruction streams.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_FILE = HERE / "probes" / "sm100_ldtm.cu"
TABLE = re.compile(r'P_([LMN])\((\w+), "([^"]*)", "([^"]*)"\)')
SASS_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]*)")
MARGIN = 0.02
SAT_MIN = 0.95
PASS = 0.98
TC_WORDS = 32768  # a UTCIMMA 128x256x32's accumulator words


def write_table(gate: Path) -> Path:
    """Every probe expects its stream loop's eight most frequent opcodes, the branch aside (P_M adds the MMA warp's loop,
    four UTCIMMA). The counted opcode is the stream's IMMA (l) or its FFMA / FADD (n, m)."""
    rows = []
    for v in json.loads(gate.read_text())["verdicts"]:
        if v["binary"] != "sm100_ldtm":
            continue
        loops = v["verdict"]["loops"]
        stream = max(loops, key=lambda lp: sum(lp["top"].values()))
        top = [(k, n) for k, n in stream["top"].items() if not k.startswith("BRA")][:8]
        cnt = next(k for k, _ in top if k.startswith("IMMA.16832") or k in ("FFMA", "FADD")) if v["probe"][0] != "l" \
            else next(k for k, _ in top if k.startswith("IMMA.16832"))
        ex = ",".join(f"{k}={n}" for k, n in top)
        rows.append(f'    P_{v["probe"][0].upper()}({v["probe"]}, "{ex}", "{cnt}"),')
    TABLE_FILE.write_text(re.sub(r"(PROBES\[\] = \{\n).*?(\n\};)", lambda m: m[1] + "\n".join(rows) + m[2],
                                 TABLE_FILE.read_text(), count=1, flags=re.S))
    return TABLE_FILE


def table() -> dict[str, dict]:
    return {name: {"counts": {k: int(n) for k, n in (p.split("=") for p in ex.split(","))}, "counted": cnt, "mma": kind == "M"}
            for kind, name, ex, cnt in TABLE.findall(TABLE_FILE.read_text())}


def loop_counts(out: Path) -> dict[str, dict[str, int]]:
    """probe → every opcode of its stream loop (the larger of an m probe's two) and its count, from sass/sm100_ldtm.sass."""
    bounds = {}
    for v in json.loads((out / "gate-build.json").read_text())["verdicts"]:
        if v["binary"] == "sm100_ldtm" and v["verdict"]["loops"]:
            lp = max(v["verdict"]["loops"], key=lambda lp: sum(lp["top"].values()))
            bounds[v["probe"]] = (int(lp["head"], 16), int(lp["tail"], 16))
    counts: dict[str, dict[str, int]] = {}
    for chunk in (out / "sass" / "sm100_ldtm.sass").read_text().split("Function : ")[1:]:
        name = chunk.split(None, 1)[0]
        b = bounds.get(name)
        if b is None:
            continue
        c: dict[str, int] = {}
        for m in SASS_LINE.finditer(chunk):
            if b[0] <= int(m[1], 16) <= b[1]:
                c[m[2]] = c.get(m[2], 0) + 1
        counts[name] = c
    return counts


def points(out: Path) -> dict[str, dict]:
    tab, full = table(), loop_counts(out)
    pts = {}
    for line in (out / "probes" / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = tab.get(r["probe"])
        if t is None or r["probe"] not in full:
            continue
        T = 4 * t["counts"][t["counted"]] / r["rates"][f"0:{t['counted']}"]
        p = {"T": T, "counts": full[r["probe"]], "n": sum(full[r["probe"]].values()), "run": r.get("run"),
             "sm_mhz": r.get("sm_mhz"), "throttled": r.get("throttled"), "spread": r.get("spread")}
        if t["mma"]:
            mma = r["rates"].get("1:UTCIMMA*")
            p["tc_words_per_clk"] = mma * TC_WORDS if mma else None
        pts[r["probe"]] = p
    return pts


def streams(out: Path) -> dict[str, dict]:
    """Each stream (family and F): its baseline issue and each density's slowdown a load."""
    pts = points(out)
    res = {}
    for name, b in sorted(pts.items()):
        fam, k = name.rsplit("_", 1)
        if k != "0":
            continue
        s = {"T0": b["T"], "issue0": b["n"] / b["T"], "read": b["n"] / b["T"] >= SAT_MIN, "points": {}}
        if "tc_words_per_clk" in b:
            s["tc0"] = b["tc_words_per_clk"]
        for other, p in pts.items():
            pf, pk = other.rsplit("_", 1)
            if pf != fam or pk == "0":
                continue
            added = {op: p["counts"].get(op, 0) - b["counts"].get(op, 0)
                     for op in set(p["counts"]) | set(b["counts"]) if p["counts"].get(op, 0) != b["counts"].get(op, 0)}
            loads = sum(n for op, n in added.items() if op.startswith("LDTM"))
            if loads <= 0:
                continue
            others = sum(n for op, n in added.items() if not op.startswith("LDTM"))
            dT = p["T"] - b["T"]
            q = {"K": int(pk), "T": p["T"], "loads": loads, "others": others, "added": added, "issue": p["n"] / p["T"],
                 "all": dT / loads, "ldtm": (dT - others) / loads}
            if "tc_words_per_clk" in p:
                q["tc"] = p["tc_words_per_clk"]
            s["points"][other] = q
        res[fam] = s
    return res


def priced(s: dict):
    """A read stream's points that set the price: at least two loads, the point still issue-saturated (the single-load
    restructure and the load-path-bound points above it are dropped; the docstring says why)."""
    if not s["read"]:
        return
    for name, q in s["points"].items():
        if q["K"] >= 2 and q["issue"] >= SAT_MIN:
            yield name, q


def analyze(outs: list[Path]) -> dict:
    runs = {}
    for out in outs:
        rid = out.parent.name if out.name == "out" else out.name
        runs[rid] = streams(out)
    least = None
    for rid, ss in runs.items():
        for fam, s in ss.items():
            for name, q in priced(s):
                if least is None or q["ldtm"] < least[0]:
                    least = (q["ldtm"], name, rid)
    res = {"runs": runs, "min_issue": SAT_MIN, "margin": MARGIN, "pass_at": PASS}
    if least:
        price = least[0] / (1 + MARGIN)
        res |= {"least": least[0], "set_by": least[1], "run": least[2], "price": price, "pass": price >= PASS}
    if len(runs) > 1:
        per = {}
        for rid, ss in runs.items():
            for fam, s in ss.items():
                for name, q in priced(s):
                    per.setdefault(name, []).append(q["ldtm"])
        res["repeat_spread"] = max((max(v) - min(v) for v in per.values() if len(v) > 1), default=None)
        lows = []
        for rid, ss in runs.items():
            vals = [q["ldtm"] for s in ss.values() for _, q in priced(s)]
            if vals:
                lows.append(min(vals) / (1 + MARGIN))
        res["price_by_run"] = lows
        res["price_spread"] = max(lows) - min(lows) if lows else None
    return res


def text(res: dict) -> str:
    L = []
    for rid, ss in res["runs"].items():
        L.append(f"run {rid}: streams (baseline issue; then K loads: ΔT a load all-in / less the other added at a slot "
                 "each, the loop's issue; * a priced point)")
        for fam, s in ss.items():
            tc = f", tensor {s['tc0']:.1f} words/clk" if s.get("tc0") else ""
            L.append(f"  {fam}: issue {s['issue0']:.3f}{'' if s['read'] else ' (below 0.95, not read)'}, T0 {s['T0']:.1f}{tc}")
            pr = {name for name, _ in priced(s)}
            for name, q in sorted(s["points"].items(), key=lambda kv: kv[1]["K"]):
                tcq = f" tensor {q['tc']:.1f}" if q.get("tc") else ""
                L.append(f"    {'*' if name in pr else ' '}K {q['K']:2d}: ΔT/load {q['all']:+.3f} / {q['ldtm']:+.3f} "
                         f"(issue {q['issue']:.3f}){tcq}  added {q['added']}")
    if "price" in res:
        L.append(f"LDTM.x128 issue price: {res['price']:.4f} of an issue slot (least {res['least']:.4f} at {res['set_by']}, "
                 f"run {res['run']}, less 2%): {'PASS' if res['pass'] else 'FAIL'} at {PASS} "
                 "(a load hides under saturated vector work: it issues on the uniform/load path, idle while the vector "
                 "port is full)")
    else:
        L.append("LDTM.x128 issue price: no stream issues at 0.95")
    if "price_by_run" in res:
        L.append(f"repeats: price by run {', '.join(f'{p:.4f}' for p in res['price_by_run'])} (spread "
                 f"{res['price_spread']:.4f}); largest spread of a point's ΔT/load across runs {res['repeat_spread']:.4f}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["analyze", "tables"])
    ap.add_argument("out", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "tables":
        print(write_table(a.out[0]))
        return 0
    res = analyze(a.out)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    print(text(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
