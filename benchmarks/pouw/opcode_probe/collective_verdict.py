"""match.cu and collective.cu's verdict from a run's rows: whether the card runs MATCH.ANY and REDUX on one unit of their
own (the coverage table's `collective` pipe, machine_model/data/cover/rows.toml, CMP-166) and spends at most 16 alu
lanes on a VOTE.

    python3 collective_verdict.py ROWS.jsonl...

Rates are warp instructions a SM-clock. The unit's rate u is the faster of MATCH.ANY (1 class) and REDUX.SUM alone.
Beside each other the two hold u (on two units they would reach 2u), a .U64 MATCH is 2 units (twice its rate at most u),
and beside LOP3 and FFMA MATCH.ANY keeps at most u. A VOTE is 16 alu lanes, half a LOP3: beside LOP3, LOP3's rate plus
half VOTE's holds at most alu's 2. REDUX beside LOP3 is only reported: its chain's adds and moves run on alu or fma as
ptxas places them (collective.cu's CHAIN_*).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SLACK = 1.001
ALU = 2.0
BINARIES = ("match", "collective")
QS = (16, 32, 64)


def verdict(rows: dict[str, dict]) -> tuple[bool, list[str]]:
    failed, said = [], []

    def rate(name: str, pattern: str) -> float | None:
        if name not in rows:
            if f"no row for {name}" not in failed:
                failed.append(f"no row for {name}")
            return None
        return rows[name]["rates"].get("0:" + pattern, 0.0)

    m, r = rate("match_any_u32", "MATCH.ANY"), rate("redux_add", "REDUX.SUM")
    if m is None or r is None:
        return False, failed
    u = max(m, r)
    said.append(f"the unit {u:.4f} (MATCH.ANY {m:.4f}, REDUX.SUM {r:.4f} alone)")

    rm, mm = rate("mix_redux_match", "REDUX.SUM"), rate("mix_redux_match", "MATCH.ANY")
    if rm is not None:
        said.append(f"REDUX.SUM + MATCH.ANY = {rm:.4f} + {mm:.4f} = {rm + mm:.4f} beside each other")
        if rm + mm > u * SLACK:
            failed.append(f"REDUX.SUM + MATCH.ANY {rm + mm:.4f} > the unit's {u:.4f} beside each other: two units")

    w = rate("match_any_u64", "MATCH.ANY.U64")
    if w is not None:
        said.append(f"MATCH.ANY.U64 {w:.4f} alone, 2 units {2 * w:.4f}")
        if 2 * w > u * SLACK:
            failed.append(f"MATCH.ANY.U64 as 2 units {2 * w:.4f} > the unit's {u:.4f}")

    beside = [(2 * row["rates"].get("0:MATCH.ANY.U64", 0.0) + row["rates"].get("0:MATCH.ANY", 0.0), name)
              for name, row in rows.items() if row.get("binary") == "match" and name.startswith("mix_")]
    if beside:
        load, name = max(beside)
        said.append(f"MATCH beside LOP3 and FFMA at most {load:.4f} units ({name}, of {len(beside)})")
        if load > u * SLACK:
            failed.append(f"MATCH beside LOP3 and FFMA {load:.4f} > the unit's {u:.4f} ({name})")

    v = rate("vote_ballot", "VOTE.ANY")
    loads = []
    for q in QS:
        name = f"mix_vote_lop3_{q}"
        vq, lq = rate(name, "VOTE.ANY"), rate(name, "LOP3.LUT")
        if vq is None:
            continue
        loads.append(f"{lq:.4f} + {vq:.4f}/2 = {lq + vq / 2:.4f}")
        if lq + vq / 2 > ALU * SLACK:
            failed.append(f"LOP3 + VOTE.ANY/2 {lq + vq / 2:.4f} > alu's {ALU:g} ({name})")
    if v is not None:
        said.append(f"VOTE.ANY {v:.4f} alone; LOP3 + VOTE/2 = {', '.join(loads)} at Q {', '.join(map(str, QS))}")

    reds = [(rows[n]["rates"].get("0:REDUX.SUM", 0.0), rows[n]["rates"].get("0:LOP3.LUT", 0.0))
            for n in (f"mix_redux_lop3_{q}" for q in QS) if n in rows]
    if reds:
        said.append(f"REDUX.SUM beside {', '.join(map(str, QS[:len(reds)]))} LOP3 {', '.join(f'{x:.4f}' for x, _ in reds)} "
                    f"(LOP3 {', '.join(f'{y:.4f}' for _, y in reds)})")

    tagged = sorted(n for n, row in rows.items() if row.get("throttled") or row.get("tags"))
    if tagged:
        said.append("tagged: " + ", ".join(f"{n} {rows[n].get('tags') or ['throttled']}" for n in tagged))
    return not failed, failed + said


def main(argv: list[str]) -> int:
    rows = {}
    for path in map(Path, argv or ["rows.jsonl"]):
        rows |= {r["probe"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("binary") in BINARIES}
    held, lines = verdict(rows)
    print(("verdict: hypothesis held: " if held else "verdict: hypothesis failed: ") + "; ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
