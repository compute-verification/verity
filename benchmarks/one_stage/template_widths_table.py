"""CIR-57 (measurement only, scratch): the table of `template_widths.py`'s and `pouw_template_widths.py`'s outputs.

    python benchmarks/one_stage/template_widths_table.py OUT.md ROW.json... [--pouw POUW.json] [--known KNOWN.json...] \\
      [--json OUT.json]

One line per (Definition family, route) over 16 bits, its specs folded (the statics that vary across them
shown as ranges, e.g. an attention head's T): the committed bits (the held rule's count), the rows that offer it, its
units per rank (every part of a rank summed: request, step, workload; every spec of the line summed), the 16-bit re-cut per
instance (``units16``: measured where `template_widths.recut` ran, else the lower bound from its returned widths, marked
``≥``) and the AND cost of the cheaper re-cut against today's, the class it needs under ASK-61's option A (16, 1,024, or the
next power of two), and its owner. The route is how a row offers the template: ``row`` (the serving-rows window's "any
row's own Definition"), ``cut`` (a ``CUT`` family's batched member) or ``a11`` (the one-stage audit's unit chain).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SHORT = {"glm53__fp8__h200__tp8__b1__": "GLM B1", "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy": "GLM B8",
         "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch": "GLM B8-stoch", "glm53__fp8__h200__tp8__b32": "GLM B32",
         "dsv4-pro": "DSV4-Pro B1", "llama32-1b": "Llama-3.2-1B", "qwen3-06b": "Qwen3-0.6B", "qwen3-14b": "Qwen3-14B"}
ROUTE = {"row": "row", "cut": "cut", "a11-chain": "a11"}
#: not a template: a workload Program's root (the whole request) and a root constant (`Q_template_instances` admits none)
NOT_ROWS = re.compile(r"^(Workload|Const)")
#: what offers a ``--known`` Definition, the Programs of record not building it
OFFERED = {
    "MoeSum_v2": "`MoeSum_v1`'s Boolean form (`boolean_moe`); the records build `MoeSum_v1`",
    "MoeSumCoordinate_v2": "`MoeSum_v2`'s member",
    "ArgmaxLeaf_v1": "Boolean `TokenSelect_v2`'s stage members (`sampling_boolean`); the records build `TokenSelect_v1`",
    "ArgmaxMerge_v1": "Boolean `TokenSelect_v2`'s stage members (`sampling_boolean`); the records build `TokenSelect_v1`",
    "F32ArgmaxLeaf_v1": "`F32ArgmaxStage_v1`, `gumbel_boolean`'s select",
    "F32ArgmaxMerge_v1": "`F32ArgmaxStage_v1`, `gumbel_boolean`'s select",
    "GatherE4m3x": "the static pair form's weight gather, one per weight (`a9.cut`, `a11.LAZY`)",
    "GatherF32x": "the static pair form's scale gather, one per scale of `MoeExpertFp8ScaleRow_v2` (`a11.LAZY`)",
    "RoPEHead_v1": "Llama's `RoPE_v1` cut (also in the vLLM table)",
    "AllReduceMultimemVector_v2": "`AllReduceMultimem_v1`'s body on `cursor/multimem-batch-21f3` (no PR), one per 16-byte vector",
    "AllReduceMultimemCoordinate_v2": "`AllReduceMultimemVector_v2`'s member",
}


def offered(family: str) -> str:
    return next((v for k, v in OFFERED.items() if family.startswith(k)), "")


def short(row: str) -> str:
    return next((v for k, v in SHORT.items() if row.startswith(k)), row)


def statics(tid: str) -> dict[str, str]:
    i = tid.find("{")
    if i < 0:
        return {}
    body, out, depth, cur = tid[i + 1:-1], {}, 0, ""
    for ch in body + ",":
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            k, _, v = cur.partition("=")
            out[k] = v
            cur = ""
        else:
            cur += ch
    return out


def fold(ids: list[str]) -> str:
    fam = ids[0].split("{")[0]
    if len(ids) == 1:
        return ids[0]
    ss = [statics(t) for t in ids]
    keys = list(dict.fromkeys(k for s in ss for k in s))
    parts = []
    for k in keys:
        vs = sorted({s.get(k, "∅") for s in ss}, key=lambda v: (len(v), v))
        if len(vs) == 1:
            parts.append(f"{k}={vs[0]}")
        elif all(v.lstrip("-").isdigit() for v in vs):
            n = sorted(int(v) for v in vs)
            parts.append(f"{k}={n[0]}–{n[-1]}" if len(n) == n[-1] - n[0] + 1 else f"{k}∈{{{','.join(map(str, n[:6]))}{',…' if len(n) > 6 else ''}}}")
        else:
            parts.append(f"{k}∈{{{','.join(vs[:4])}{',…' if len(vs) > 4 else ''}}}")
    return f"{fam}{{{','.join(parts)}}} ({len(ids)} specs)"


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
        cov = "" if r.get("ands_coverage") == 1.0 else f"; pins cover {r['ands_coverage']:.0%} of gates"
        return f"{r['ands_today_pinned']:,} → {best:,} ({best / r['ands_today_pinned']:.2f}×{cov})"
    if "ands_naive" in r:
        return f"no pinned ANDs (pins cover {r.get('ands_coverage', 0):.0%} of gates)"
    if rec.get("pinned_ands"):
        return f"{rec['pinned_ands']:,} today; re-cut not run"
    if r.get("error"):
        return f"re-cut not finished ({r['error'][:40]})"
    return "re-cut not run"


def pick(recs: list[dict]) -> dict:
    """The line's representative: a spec whose re-cut has ANDs, else one with a re-cut, else the largest."""
    for test in (lambda r: (r.get("recut") or {}).get("ands_today_pinned"), lambda r: "units16" in (r.get("recut") or {}),
                 lambda r: True):
        c = [r for r in recs if test(r)]
        if c:
            return max(c, key=lambda r: r.get("gates") or 0)
    return recs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("rows", nargs="*", type=Path)
    ap.add_argument("--pouw", type=Path)
    ap.add_argument("--known", type=Path, nargs="*", default=[])
    ap.add_argument("--json", type=Path)
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
            recs[tid]["hows"] = sorted(set(old.get("hows", [])) | set(rec.get("hows", [])))
        for tid, by in d["units"].items():
            for key, n in by.items():
                per_rank[tid][row][int(key.split("/")[0][1:])] += n
                rows_of[tid].add(row)
    lines_by: dict[tuple, list[str]] = defaultdict(list)
    for tid, r in recs.items():
        if (r.get("bits") or 0) > 16 and not NOT_ROWS.match(r["family"]):
            route = "+".join(ROUTE[h] for h in r["hows"])
            lines_by[(r["family"], route)].append(tid)
    head = ["| Definition | committed bits | route | rows | units per rank | re-cut: 16-bit units per instance | "
            "ANDs per instance, today → re-cut | class under A | owner |", "|---|---|---|---|---|---|---|---|---|"]
    top = {k: max(recs[t]["bits"] for t in v) for k, v in lines_by.items()}
    order = sorted(lines_by, key=lambda k: ({"cut": 0, "a11": 1}.get(k[1].split("+")[0], 2), -top[k], k[0]))
    lines, jl = [], []
    for key in order:
        ids = lines_by[key]
        rs = [recs[t] for t in ids]
        rep = pick(rs)
        rws = sorted(set().union(*(rows_of[t] for t in ids)))
        ranks = []
        for row in rws:
            tot: dict[int, int] = defaultdict(int)
            for t in ids:
                for rk, n in per_rank[t][row].items():
                    tot[rk] += n
            lo, hi = min(tot.values()), max(tot.values())
            ranks.append(f"{row} {lo:,}" if lo == hi else f"{row} {lo:,}–{hi:,} over {len(tot)} ranks")
        u, measured = units16(rep)
        ruled = [r for r in rs if "verdict" in r]
        ref = sum(r["verdict"].startswith("refused") for r in ruled)
        rule = f" (rule refused {ref}/{len(ruled)} run)" if ruled else ""
        lo_b, hi_b = min(r["bits"] for r in rs), max(r["bits"] for r in rs)
        bits = f"{hi_b:,}" if lo_b == hi_b else f"{lo_b:,}–{hi_b:,}"
        at = "" if len(ids) == 1 else f" (at `{rep['id'][len(key[0]):]}`, {rep['bits']:,} bits)"
        lines.append(f"| `{fold(sorted(ids))}` | {bits}{rule} | {key[1]} | {', '.join(rws)} | {'; '.join(ranks)} | "
                     f"{'' if measured else '≥'}{u:,}{at} | {ands(rep)} | {klass(hi_b)} | @circuits |")
        jl.append({"definition": fold(sorted(ids)), "specs": sorted(ids), "bits": [lo_b, hi_b], "route": key[1], "rows": rws,
                   "units_per_rank": {row: dict(sorted({rk: sum(per_rank[t][row].get(rk, 0) for t in ids)
                                                        for rk in set().union(*(per_rank[t][row] for t in ids))}.items()))
                                      for row in rws},
                   "units16_per_instance": u, "units16_measured": measured, "recut": rep.get("recut"),
                   "representative": rep["id"], "rule_runs": len(ruled), "rule_refused": ref, "class_A": klass(hi_b),
                   "owner": "@circuits"})
    dropped = sorted(t for t, r in recs.items() if NOT_ROWS.match(r["family"]))
    under = sorted(r["id"] for r in recs.values() if r.get("bits") is not None and r["bits"] <= 16 and not NOT_ROWS.match(r["family"]))
    unmeasured = sorted(r["id"] for r in recs.values() if r.get("bits") is None)
    dis = sorted(r["id"] for r in recs.values() if r.get("disagree"))
    out = ["## vLLM rows", "", *head, *lines, "",
           f"{sum(len(v) for v in lines_by.values())} template specs over 16 bits in {len(lines)} lines; {len(under)} at or "
           f"under 16 ({', '.join(sorted({recs[t]['family'] for t in under}))}); {len(unmeasured)} not measured; "
           f"{len(dis)} disagreements between the rule's walk and the descriptor count; {len(dropped)} root constants and workload "
           f"roots left out (not templates).", "",
           "Programs read: " + "; ".join(f"{row}: {len(ps)} ({sum(1 for p in ps if p['digest_ok'])} digests match records.json)"
                                         for row, ps in sorted(progs.items())), ""]
    kj = []
    if a.known:
        kl = ["| Definition | committed bits | rule | offered by | measured at | re-cut: 16-bit units per instance | "
              "ANDs per instance, today → re-cut | class under A | owner |", "|---|---|---|---|---|---|---|---|---|"]
        for p in a.known:
            d = json.loads(p.read_text())
            for r in sorted(d["templates"].values(), key=lambda r: (-(r.get("bits") or 0), r["id"])):
                verdict = (r.get("verdict") or "descriptor count only").split(":")[0]
                u, measured = units16(r) if (r.get("bits") or 0) > 16 else (1, True)
                kl.append(f"| `{r['id']}` | {r.get('bits') or 0:,} | {verdict} | {offered(r['family'])} | {d['where']} | "
                          f"{'' if measured else '≥'}{u:,} | {ands(r) if (r.get('bits') or 0) > 16 else '—'} | "
                          f"{klass(r.get('bits') or 0)} | @circuits |")
                kj.append({"definition": r["id"], "bits": r.get("bits"), "verdict": r.get("verdict"), "offered_by": offered(r["family"]),
                           "where": d["where"], "tree": d["tree"], "units16_per_instance": u, "recut": r.get("recut"),
                           "class_A": klass(r.get("bits") or 0), "owner": "@circuits"})
        out += ["## Named Definitions outside the Programs of record", "", *kl, ""]
    pj = []
    if a.pouw:
        d = json.loads(a.pouw.read_text())
        pl = ["| Definition | committed bits | role | used by | re-cut: 16-bit units per instance | ANDs per instance, today → re-cut | "
              "class under A | owner |", "|---|---|---|---|---|---|---|---|"]
        prs = sorted((r for r in d["templates"].values() if (r.get("bits") or 0) > 16 or r.get("unbound")),
                     key=lambda r: (-(r.get("bits") or 0), r["id"]))
        for r in prs:
            if r.get("unbound"):
                pl.append(f"| `{r['id']}` | not measured: {r['unbound']} | {'; '.join(r['roles'])} | | | | | @compute-accounting |")
                continue
            u, measured = units16(r)
            used = ", ".join(f"{k.split('{')[0]} ×{n:,}" for k, n in list(r["used_by"].items())[:2]) + (" …" if len(r["used_by"]) > 2 else "")
            pl.append(f"| `{r['id']}` | {r['bits']:,} | {'; '.join(r['roles'])} | {used} | {'' if measured else '≥'}{u:,} | "
                      f"{ands(r)} | {klass(r['bits'])} | @compute-accounting |")
            pj.append({"definition": r["id"], "bits": r["bits"], "roles": r["roles"], "used_by": r["used_by"],
                       "units16_per_instance": u, "units16_measured": measured, "recut": r.get("recut"),
                       "verdict": r.get("verdict"), "class_A": klass(r["bits"]), "owner": "@compute-accounting"})
        out += ["## PoUW (main's units at circuit-check's statics, and the verifier-pinned tiles; owner @compute-accounting)", "",
                *pl, ""]
    a.out.write_text("\n".join(out))
    if a.json:
        a.json.write_text(json.dumps({"vllm": jl, "known": kj, "pouw": pj}, indent=1))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
