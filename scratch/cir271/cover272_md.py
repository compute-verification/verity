"""cover272_md.py COVER.json OUT.md: CIR-272's tables from the v3 record: every type refused or not run, largest first, with its
reason and what one more run of it would need (the record's `needs`; [inferred] where it is a sibling's or a fit's)."""
import json
import re
import sys

c = json.load(open(sys.argv[1]))
T = c["bound_runs"]["types"]


def short(n: str) -> str:
    n = re.sub(r'=\{"fn":"([^"]*)"\}', r"=\1", n)
    n = re.sub(r'=\{"f64":"[^"]*"\}', "=…", n)
    return n if len(n) < 110 else n[:107] + "…"


def need(r: dict) -> str:
    nd = r.get("needs") or {}
    if r.get("why") == "lean-past-64GB":
        gib = nd["circuit_bytes"] / 2 ** 30
        return (f"staged in {nd['stage_s']:.0f} s at {nd['stage_gb']} GB [run]; Lean {nd['lean_gb_by_rule']} GB by the rule [run], "
                f"≈{gib * LR:.0f} GB at the measured ratio [inferred]")
    if r.get("why") == "stage-past-62GB":
        s = nd["sibling"]
        return f"stager stopped at 62 GB after {nd['stage_s']:.0f} s [run]: over 62 GB to stage; K=2048 staged at {s['stage_gb']} GB and passed in {s['seconds']} s [run]"
    if r.get("why") == "past-2700s":
        s = nd["sibling"]
        return f"not staged at 2,725 s [run]; K=2048 staged in {s['seconds']} s to a {s['circuit_bytes'] / 2 ** 30:.1f} GiB circuit, Lean {s['lean_gb_by_rule']} GB by the rule [run]: more than both [inferred]"
    if "fit" in nd:
        f = nd["fit"]
        return (f"≈{f['stage_s']} s staging at ≈{f['stage_gb']} GB, circuit ≈{f['circuit_gib']} GiB, Lean ≈{f['lean_gb_by_rule']} GB by the rule, "
                f"≈{f['lean_gb_by_measured_ratio']} GB at the measured ratio [inferred: fit over NVIS {f['from'][0]}–{f['from'][-1]}]")
    if nd.get("sibling"):
        s = nd["sibling"]
        return f"sibling {s['tag']} ({s['outcome']}): {s['seconds']} s, stage {s['stage_gb']} GB, Lean {s['lean_gb_by_rule']} GB by the rule [inferred]"
    if nd.get("earlier"):
        e = nd["earlier"][0]
        return f"only point: {short(e['block'])} staged at {e['stage_gb']} GiB in {e['run']} [inferred]"
    return "no measured point"


LR = max(max((v or {}).get("maxrss_gb") or 0 for v in m["verdicts"].values()) / (m["stage"]["circuit_bytes"] / 2 ** 30)
         for m in T.values() if m["outcome"] == "accepted" and m["stage"]["circuit_bytes"] > 5 * 2 ** 30)
key = lambda r: (-(r["blocks"] or 0), r["tag"])  # noqa: E731
out = ["| tag | type | blocks | k_log | text |", "|---|---|---:|---:|---|"]
for r in sorted((r for r in T.values() if r["outcome"] == "refused"), key=key):
    out.append(f"| {r['tag']} | `{short(r['block'])}` | {r['blocks']:,} | {r['k_log']} | " + "; ".join(t.strip().rstrip("|").replace("|", "/") for t in r["text"])
               + (f"; not run: its Boolean version `{short(r['needs']['stage'])}`, which its census lookup names ({r['zk_refusal'].split(': ', 1)[1].split(';')[0]}) [run]" if r.get("zk_refusal") else "") + " |")
out += ["", "| tag | type | blocks | k_log | why | one more run would need |", "|---|---|---:|---:|---|---|"]
for r in sorted((r for r in T.values() if r["outcome"] == "not-run"), key=key):
    why = r["why"] + (f"; {r['names_refusal'].split(': ', 1)[1][:160]}" if r.get("names_refusal") else "")
    out.append(f"| {r['tag']} | `{short(r['block'])}` | {'—' if r['blocks'] is None else format(r['blocks'], ',')} | {r['k_log'] if r['k_log'] is not None else '—'} | {why} | {need(r)} |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(len(out), "lines; measured Lean ratio", round(LR, 2))
