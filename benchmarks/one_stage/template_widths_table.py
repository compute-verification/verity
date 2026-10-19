"""CIR-57 (measurement only, scratch): the table of `template_widths.py`'s and `pouw_template_widths.py`'s outputs.

    python benchmarks/one_stage/template_widths_table.py OUT.md ROW.json... [--pouw POUW.json]

One line per over-16 template: its committed bits (the held rule's count), the rows that offer it, its units per rank
(every part of a rank summed: request, step, workload), the 16-bit re-cut per instance (``units16``: measured where
`template_widths.recut` ran, else the lower bound from its returned widths, marked ``≥``) and the AND cost of the cheaper
re-cut against today's, the class it needs under ASK-61's option A (16, 1,024, or the next power of two), and its owner.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SHORT = {"glm53__fp8__h200__tp8__b1__": "GLM B1", "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy": "GLM B8",
         "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch": "GLM B8-stoch", "glm53__fp8__h200__tp8__b32": "GLM B32",
         "dsv4-pro": "DSV4-Pro B1", "llama32-1b": "Llama-3.2-1B", "qwen3-06b": "Qwen3-0.6B", "qwen3-14b": "Qwen3-14B"}


def short(row: str) -> str:
    return next((v for k, v in SHORT.items() if row.startswith(k)), row)


def units16(rec: dict) -> tuple[int, bool]:
    r = rec.get("recut") or {}
    if "units16" in r:
        return r["units16"], True
    wide = sum(math.ceil(int(w) / 16) * n for w, n in rec.get("returned_widths", {}).items() if int(w) > 16)
    narrow = sum(int(w) * n for w, n in rec.get("returned_widths", {}).items() if int(w) <= 16)
    return wide + math.ceil(narrow / 16), False


def klass(bits: int) -> str:
    if bits <= 16:
        return "16"
    if bits <= 1024:
        return "1,024"
    return f"{2 ** math.ceil(math.log2(bits)):,} (over 1,024)"


def ands(rec: dict) -> str:
    r = rec.get("recut") or {}
    if "ands_naive" in r and r.get("ands_today_pinned"):
        best = min(r["ands_naive"], r["ands_2level"])
        cov = "" if r.get("ands_coverage") == 1.0 else f", pins cover {r['ands_coverage']:.0%} of gates"
        return f"{r['ands_today_pinned']:,} → {best:,} ({best / r['ands_today_pinned']:.2f}×{cov})"
    if rec.get("pinned_ands"):
        return f"{rec['pinned_ands']:,} today; re-cut not run"
    return "not run" if not r.get("error") else f"re-cut failed: {r['error']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("rows", nargs="*", type=Path)
    ap.add_argument("--pouw", type=Path)
    a = ap.parse_args()
    recs: dict[str, dict] = {}
    rows_of: dict[str, set] = defaultdict(set)
    per_rank: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    progs: dict[str, list] = defaultdict(list)
    for p in a.rows:
        d = json.loads(p.read_text())
        row = short(d["row"])
        progs[row] += d["programs"]
        for tid, rec in d["templates"].items():
            old = recs.get(tid, {})
            recs[tid] = rec if len(json.dumps(rec)) >= len(json.dumps(old)) else old
        for tid, by in d["units"].items():
            for key, n in by.items():
                rank = int(key.split("/")[0][1:])
                per_rank[tid][row][rank] += n
                rows_of[tid].add(row)
    over = sorted((r for r in recs.values() if (r["bits"] or 0) > 16), key=lambda r: (-r["bits"], r["id"]))
    unmeasured = sorted(r["id"] for r in recs.values() if r["bits"] is None)
    lines = ["| Definition | committed bits | rows | units per rank | re-cut at 16 bits: units per instance | ANDs per instance, today → re-cut | class under A | owner |",
             "|---|---|---|---|---|---|---|---|"]
    for r in over:
        u, measured = units16(r)
        ranks = []
        for row in sorted(rows_of[r["id"]]):
            v = per_rank[r["id"]][row]
            lo, hi = min(v.values()), max(v.values())
            ranks.append(f"{row}: {lo:,}" if lo == hi else f"{row}: {lo:,}–{hi:,} ({len(v)} ranks)")
        verdict = "" if "verdict" not in r else (" (rule run)" if r["verdict"].startswith("refused") else " (rule ACCEPTED?)")
        lines.append(f"| `{r['id']}` | {r['bits']:,}{verdict} | {', '.join(sorted(rows_of[r['id']]))} | {'; '.join(ranks)} | "
                     f"{'' if measured else '≥'}{u:,} | {ands(r)} | {klass(r['bits'])} | @circuits |")
    under = sorted(r["id"] for r in recs.values() if r["bits"] is not None and r["bits"] <= 16)
    out = ["## vLLM rows", "", *lines, "", f"{len(over)} templates over 16 bits; {len(under)} at or under 16; "
           f"{len(unmeasured)} not measured{': ' + ', '.join(unmeasured) if unmeasured else ''}.", ""]
    out += ["Programs read: " + "; ".join(f"{row} {len(ps)} ({sum(1 for p in ps if p['digest_ok'])} digests match records.json)"
                                       for row, ps in sorted(progs.items())), ""]
    if a.pouw:
        d = json.loads(a.pouw.read_text())
        pl = ["| Definition | committed bits | role | used by | re-cut at 16 bits: units per instance | ANDs per instance, today → re-cut | class under A | owner |",
              "|---|---|---|---|---|---|---|---|"]
        prs = sorted((r for r in d["templates"].values() if (r.get("bits") or 0) > 16 or r.get("unbound")),
                     key=lambda r: (-(r.get("bits") or 0), r["id"]))
        for r in prs:
            if r.get("unbound"):
                pl.append(f"| `{r['id']}` | not measured: {r['unbound']} | {'; '.join(r['roles'])} | | | | | @compute-accounting |")
                continue
            u, measured = units16(r)
            used = ", ".join(f"{k} ×{n:,}" for k, n in list(r["used_by"].items())[:3]) + (" …" if len(r["used_by"]) > 3 else "")
            pl.append(f"| `{r['id']}` | {r['bits']:,} | {'; '.join(r['roles'])} | {used} | {'' if measured else '≥'}{u:,} | "
                      f"{ands(r)} | {klass(r['bits'])} | @compute-accounting |")
        out += ["## PoUW (main's units; owner @compute-accounting)", "", *pl, ""]
    a.out.write_text("\n".join(out))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
