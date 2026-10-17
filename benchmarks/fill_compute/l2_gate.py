"""The protocol-kernel gate's L2 half for the filler: each `fill_*` launch of `bench.py --l2-probe`, measured by ncu, writes no
more L2 sectors than its outputs declare (`lts__t_sectors_op_write.sum`) and makes no more atomic sectors than its fold and
queue declare (`lts__t_sectors_op_atom.sum` + `lts__t_sectors_op_red.sum`); there are no partials to declare.

The verdict is per kernel (`by_kernel`, what an arm is judged by; `spread` the least and most write sectors over the repeats) and per store mode (`by_store`): PoUW's y store writes 16 bytes of a 32-byte sector a request, so `all` and `y`
can be refused for the store pattern alone, which the fold (the filler that writes no tile words) is not. A `control` launch
(no tiles) is measured and reported, not judged: its sectors are what a launch writes whatever its outputs (13-15 a launch on
sm_120, r20261012-231033-73f3). Each judged launch is allowed the run's floor beside its declared sectors (a persistent
filler is one launch a window, so the floor doesn't grow with its tiles). The floor is the least dry launch in the same run
(ruling 03:10Z): a dry launch runs a judged launch's tiles and declares nothing, so it writes what running tiles writes
beyond the outputs (15 on sm_120 in every run, r20261013-025156-e585), where a zero-tile control writes 13-15. A run
without dry launches takes each kernel's least zero-tile control, the smaller allowance. Each kernel's `detail` reports
its zero-tile controls' range beside the floor.

Which repeats must pass is rule 4 (memory accounting, 13 Oct): every one (`strict`) unless this run shows the excess some
launches carry is external, and then the least of a kernel's repeats (`min`). The test: a launch is hit when its write
sectors exceed its allowance (declared + floor; a control's is its floor) by more than HIT sectors, above the launch's own
noise (a few sectors; the chunked kernels' constant 2) where the intermittent excess comes in multiples of 64. The excess is external when, against controls interleaved
with the judged launches in the same run, (a) some control is hit, (b) a one-sided Fisher exact test of "judged launches
are hit more often than controls" gives p >= ALPHA, and (c) a one-sided Mann-Whitney test of "judged excesses are larger
than controls'" gives p >= ALPHA (ruling 03:10Z; the largest of many judged launches against the largest of a few controls
favours the larger sample).
The controls that decide are the dry launches (`external_dry`: a fold kernel's tiles with the log off, as long as a judged
launch and declaring nothing), the duration-matched control (ruling 02:30Z); the zero-tile controls' test (`external`) is
reported beside it. A run without dry launches, or whose dry launches fail the test, keeps the strict rule: the guard does
not fail open.

  python3 l2_gate.py DECLARED_JSON NCU_CSV OUT_JSON      exit 0 passed, 7 refused, 3 unmeasured (no ncu rows)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

WRITE, ATOM, RED, DRAM = ("lts__t_sectors_op_write.sum", "lts__t_sectors_op_atom.sum", "lts__t_sectors_op_red.sum",
                          "dram__bytes_write.sum")
HIT, ALPHA = 64, 0.05


def measured(text: str) -> list[dict]:
    """ncu's CSV (details page): one dict of metric values per profiled kernel, in launch order."""
    lines = text.splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith('"ID"')), None)
    if start is None:
        return []
    by_id: dict[str, dict] = {}
    for r in csv.DictReader(lines[start:]):
        k = by_id.setdefault(r["ID"], {"kernel": r["Kernel Name"].split("(")[0]})
        try:
            k[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except (KeyError, ValueError):
            continue
    return [by_id[i] for i in sorted(by_id, key=int)]


def ratio(x: float, declared: int) -> float:
    return x / declared if declared else (0.0 if x == 0 else float("inf"))


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p for the 2 x 2 table [[a, b], [c, d]] (rows judged, controls; columns hit, not): the chance,
    margins fixed, of a judged hit count >= a."""
    n1, k, n = a + b, a + c, a + b + c + d
    total = math.comb(n, k)
    return sum(math.comb(n1, x) * math.comb(n - n1, k - x) for x in range(a, min(n1, k) + 1)) / total


def external(judged: list[dict], controls: list[dict]) -> dict:
    """Rule 4's test on the run's launches (each with `excess`, write sectors over its allowance)."""
    jh = [r["excess"] for r in judged if r["excess"] > HIT]
    ch = [r["excess"] for r in controls if r["excess"] > HIT]
    p = fisher_greater(len(jh), len(judged) - len(jh), len(ch), len(controls) - len(ch))
    mj, mc = max((r["excess"] for r in judged), default=0.0), max((r["excess"] for r in controls), default=0.0)
    mw = mann_whitney_greater([r["excess"] for r in judged], [r["excess"] for r in controls])
    return {"hit_sectors": HIT, "alpha": ALPHA, "judged": [len(judged), len(jh)], "controls": [len(controls), len(ch)],
            "fisher_p_judged_hit_more": p, "mann_whitney_p_judged_larger": mw, "max_judged_excess": mj,
            "max_control_excess": mc, "external": bool(ch) and p >= ALPHA and mw >= ALPHA}


def mann_whitney_greater(x: list[float], y: list[float]) -> float:
    """One-sided Mann-Whitney U p (normal approximation, tie-corrected) that x is stochastically larger than y."""
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        return 1.0
    pooled = sorted((v, i < n1) for i, v in enumerate(list(x) + list(y)))
    ranks, ties, i = [0.0] * len(pooled), 0.0, 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    u = sum(r for r, (_, first) in zip(ranks, pooled) if first) - n1 * (n1 + 1) / 2
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u - n1 * n2 / 2 - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2))


def unit(d: dict) -> str:
    """What a verdict and a floor belong to: the kernel, or the kernel under a launch variant (`kernel/variant`)."""
    return f"{d['kernel']}/{d['variant']}" if d.get("variant") else d["kernel"]


def judge(declared: dict, rows: list[dict]) -> dict:
    launches = declared["launches"]
    if len(rows) != len(launches):
        return {"status": "unmeasured", "why": f"{len(rows)} profiled launches, {len(launches)} declared"}
    ctl: dict[str, list[float]] = {}
    for d, m in zip(launches, rows):
        if d.get("control"):
            ctl.setdefault(unit(d), []).append(m.get(WRITE) or 0.0)
    dry_writes = [m.get(WRITE) or 0.0 for d, m in zip(launches, rows) if d.get("dry")]
    dry_floor = min(dry_writes) if dry_writes else None
    floors = {k: min(ws) if dry_floor is None else dry_floor for k, ws in ctl.items()}
    out, bad = [], []
    for d, m in zip(launches, rows):
        w, a = m.get(WRITE), (m.get(ATOM) or 0.0) + (m.get(RED) or 0.0)
        if m["kernel"] != d["kernel"] or w is None:
            return {"status": "unmeasured", "why": f"profiled {m['kernel']} with {sorted(m)}, declared {d['kernel']}"}
        row = {**d, "unit": unit(d), "write_sectors": w, "atom_sectors": a, "dram_write_bytes": m.get(DRAM),
               "read_sectors": m.get("lts__t_sectors_op_read.sum"), "dram_read_bytes": m.get("dram__bytes_read.sum"),
               "write_ratio": ratio(w, d["declared_write_sectors"]), "atom_ratio": ratio(a, d["declared_atom_sectors"])}
        floor = floors.get(unit(d), 0.0 if dry_floor is None else dry_floor)
        row["floor"] = 0.0 if d.get("control") else floor
        row["excess"] = w - floor if d.get("control") else w - d["declared_write_sectors"] - floor
        row["pass"] = w <= d["declared_write_sectors"] + row["floor"] and a <= d["declared_atom_sectors"]
        out.append(row)
    judged = [r for r in out if not r.get("control") and not r.get("dry")]
    test = external(judged, [r for r in out if r.get("control")])
    dry = [r for r in out if r.get("dry")]
    test_dry = external(judged, dry) if dry else None
    rule = "min" if test_dry is not None and test_dry["external"] else "strict"
    kernels = list(dict.fromkeys(r["unit"] for r in judged))
    def verdicts(key: str) -> tuple[dict, dict]:
        strict = {k: all(r[key] for r in judged if r["unit"] == k) for k in kernels}
        least = {k: min((r for r in judged if r["unit"] == k), key=lambda r: r["write_sectors"])[key]
                 and all(r["atom_sectors"] <= r["declared_atom_sectors"] for r in judged if r["unit"] == k) for k in kernels}
        return strict, least

    strict, least = verdicts("pass")
    verdict = least if rule == "min" else strict
    for r in judged:
        if not r["pass"] and not verdict[r["unit"]]:
            bad.append(f"{r['unit']}: {r['write_sectors']:.0f} write / {r['declared_write_sectors']} declared + "
                       f"{r['floor']:.0f} floor, {r['atom_sectors']:.0f} atomic / {r['declared_atom_sectors']} declared")
    by_store = {}
    for r in judged:
        by_store[r["store"]] = by_store.get(r["store"], True) and verdict[r["unit"]]
    word = {True: "passed", False: "REFUSED"}
    return {"status": "REFUSED" if bad else "passed", "rule": rule, "external": test, "external_dry": test_dry,
            "floor": {"source": "zero-tile" if dry_floor is None else "dry", "sectors": dry_floor},
            "by_store": {s: word[p] for s, p in by_store.items()},
            "by_kernel": {k: word[verdict[k]] for k in kernels},
            "by_kernel_strict": {k: word[strict[k]] for k in kernels}, "by_kernel_min": {k: word[least[k]] for k in kernels},
            "detail": {k: {"controls": [min(ctl.get(k, [0.0])), max(ctl.get(k, [0.0]))],
                           "floor": floors.get(k, 0.0 if dry_floor is None else dry_floor), "verdict": word[verdict[k]],
                           "strict": word[strict[k]], "min": word[least[k]]}
                       for k in kernels},
            "spread": {k: [min(ws), max(ws)] for k in kernels
                       for ws in [[r["write_sectors"] for r in judged if r["unit"] == k]]},
            "refusals": bad, "launches": out,
            "controls": [{k: r[k] for k in ("kernel", "write_sectors", "atom_sectors", "declared_atom_sectors")} for r in out
                         if r.get("control")],
            "max_write_ratio": max(r["write_ratio"] for r in judged), "max_atom_ratio": max(r["atom_ratio"] for r in judged)}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    dec, csv_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not dec.exists() or not csv_path.exists():
        res = {"status": "unmeasured", "why": f"missing {dec if not dec.exists() else csv_path}"}
    else:
        res = judge(json.loads(dec.read_text()), measured(csv_path.read_text()))
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
    print(f"L2 write gate: {res['status']} (rule {res.get('rule')}, dry test {res.get('external_dry')})",
          *res.get("refusals", []), res.get("why", ""))
    sys.exit({"passed": 0, "REFUSED": 7}.get(res["status"], 3))
