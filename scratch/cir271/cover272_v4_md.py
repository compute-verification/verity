"""cover272_v4_md.py COVER.json OUT.md: CIR-272's second-pass tables from the v4 record: the card's 254 types and the 18
extras by outcome; then every second-pass type refused, not run (with its estimate, measured or inferred, and its basis) or
running (with its run id), largest first."""
import json
import re
import sys
from collections import Counter

c = json.load(open(sys.argv[1]))
T = c["bound_runs"]["types"]
P2 = {t: r for t, r in T.items() if "plan" in r}


def short(n: str) -> str:
    n = re.sub(r'=\{"fn":"([^"]*)"\}', r"=\1", n)
    n = re.sub(r'=\{"f64":"[^"]*"\}', "=…", n)
    return n if len(n) < 110 else n[:107] + "…"


def estimate(r: dict) -> str:
    e = r["estimate"]
    parts = []
    if "stage_s" in e or "stage_gb" in e:
        rng = e.get("stage_gb_range")
        gb = f"{rng[0]:.0f}–{rng[1]:.0f} GB" if rng else f"{e['stage_gb']} GB" if "stage_gb" in e else None
        parts.append("stager " + ", ".join(x for x in (f"≈{e['stage_s']:,} s" if "stage_s" in e else None, gb) if x))
    if "circuit_gib" in e:
        parts.append(f"circuit ≈{e['circuit_gib']} GiB, Lean ≈{e['lean_gb']} GB at 4.42 × GiB")
    if "total_s" in e and not parts:
        parts.append(f"≈{e['total_s']:,} s in all")
    return "; ".join(parts) or "—"


def bl(r):
    return "—" if r["blocks"] is None else format(r["blocks"], ",")


def kl(r):
    return "—" if r["k_log"] is None else str(r["k_log"])


key = lambda r: (-(r["blocks"] or 0), r["tag"])  # noqa: E731
out = []
on = [b for b in c["blocks"] if b["on_card"]]
out += ["| outcome over the card's 254 types | types | blocks | share |", "|---|---:|---:|---:|"]
for s, (n, k) in c["bound_totals"].items():
    out.append(f"| {s} | {n} | {k:,} | {100 * k / sum(b['blocks'] for b in on):.4f}% |")
out += ["", "| the 18 extras | types |", "|---|---:|"] + [f"| {s} | {n} |" for s, n in c["bound_extras"].items()]
acc = [r for r in P2.values() if r["outcome"] == "accepted"]
fam = Counter(r["block"].split("{")[0] for r in acc)
out += ["", f"Accepted in the second pass ({len(acc)} types, {sum(r['blocks'] or 0 for r in acc):,} blocks): "
        + ", ".join(f"{f} ×{n}" for f, n in fam.most_common()) + "."]
ref = sorted((r for r in P2.values() if r["outcome"] in ("refused", "held-control-off")), key=key)
if ref:
    out += ["", "| tag | type | blocks | k_log | run | outcome | its own text |", "|---|---|---:|---:|---|---|---|"]
    for r in ref:
        out.append(f"| {r['tag']} | `{short(r['block'])}` | {bl(r)} | {kl(r)} | {r['run']} | {r['outcome']} | "
                   + "; ".join(t.strip().rstrip("|").replace("|", "/")[:400] for t in r["text"]) + " |")
nr = sorted((r for r in P2.values() if r["outcome"] == "not-run"), key=key)
out += ["", "| tag | type | blocks | k_log | why | estimate | measured or inferred | basis |", "|---|---|---:|---:|---|---|---|---|"]
for r in nr:
    why = r["why"] + (f" ({r['run']}: {r['text'][:200]})" if r.get("run") and r["why"] != "not-reached" else f" ({r['run']})" if r.get("run") else "")
    out.append(f"| {r['tag']} | `{short(r['block'])}` | {bl(r)} | {kl(r)} | {why.replace('|', '/')} | {estimate(r)} | {r['estimate']['kind']} | "
               f"{r['estimate']['basis'][:300].replace('|', '/')} |")
run = sorted((r for r in P2.values() if r["outcome"] == "running"), key=key)
if run:
    out += ["", "| tag | type | blocks | k_log | running in |", "|---|---|---:|---:|---|"]
    out += [f"| {r['tag']} | `{short(r['block'])}` | {bl(r)} | {kl(r)} | {r['run']} ({r['text']}) |" for r in run]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(len(out), "lines")
