#!/usr/bin/env python3
"""make_late.py WORK LATE_ROWS.json READ_Z MAIN > section: a second read of the queue, for the heads the sweep had not run.

LATE_ROWS.json has one row for each pull request open on GitHub at the second read: its fetched head, the queue's row for
it if it has one, and whether the sweep had already run that very head.  WORK/out-late/ and WORK/out-late-forced/ hold the
harness's results for the heads it had not, WORK/out/ and WORK/out-forced/ the first sweep's.
"""
import collections
import json
import sys
from pathlib import Path

work, rows_file, read_z, main = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = json.loads(rows_file.read_text())
opened = set((work / "opened.txt").read_text().split())


def load(d: str) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((work / d).glob("*.json"))}


first, forced, late, late_forced = load("out"), load("out-forced"), load("out-late"), load("out-late-forced")


def passed(r: dict) -> bool:
    return (r.get("pytest", "").startswith("3 passed") and not r.get("pytest_failed") and not r.get("detail_error")
            and not r.get("new") and not r.get("gone") and not r.get("moved") and not r.get("twice"))


def verdict(name: str, a: dict, b: dict) -> str:
    r = a[name]
    if not r.get("conflicts"):
        return "pass" if passed(r) else "fail"
    f = b.get(name)
    if f is None or "pytest" not in f or f.get("detail_error"):
        return "stops-unresolved"
    return "stops-pass" if passed(f) else "stops-fail"


def group(r: dict) -> str:
    if r["state"] is None:
        return "open, with no row in the queue's table"
    if not r["marked"]:
        return "not marked"
    return {"ready": "marked, admitted", "refused": "marked, refused", "waiting": "marked, waiting"}[r["state"]]


V, where = {}, {}
for r in rows:
    n = f"pr-{r['pr']}"
    if r["ran_before"]:
        V[n], where[n] = verdict(n, first, forced), "first"
    else:
        assert n in late, f"not run: {n}"
        V[n], where[n] = verdict(n, late, late_forced), "late"
GROUPS = ["marked, admitted", "marked, refused", "marked, waiting", "not marked", "open, with no row in the queue's table"]
p = print
marked = [r for r in rows if r["marked"]]
new = [r for r in rows if not r["ran_before"]]
p()
p(f"## A second read, at {read_z}")
p()
p(f"By {read_z} `main` was `{main[:12]}` (one of the candidates run above), {len(rows)} pull requests besides #8463 were open on")
p(f"GitHub and {len(marked)} were marked. The sweep had already run {sum(1 for r in rows if r['ran_before'])} of them at the very head they still had;")
p(f"the other {len(new)} (opened or pushed since, or open with no row in the queue's table) were run the same way, at the same base,")
p("with every candidate that then held `main`'s tip. This table is of every pull request open at that read, each at the")
p("head it then had:")
p()
p("| heads | count | of them run in this pass | merged with the base, the test passes | the merge stops; passes with its side | would fail | the merge stops, no verdict |")
p("|---|---|---|---|---|---|---|")
for g in GROUPS:
    rs = [r for r in rows if group(r) == g]
    if rs:
        c = collections.Counter(V[f"pr-{r['pr']}"] for r in rs)
        p(f"| {g} | {len(rs)} | {sum(1 for r in rs if not r['ran_before'])} | {c['pass']} | {c['stops-pass']} | "
          f"{c['fail'] + c['stops-fail']} | {c['stops-unresolved']} |")
cands = [n for n in late if not n.startswith("pr-")]
c = collections.Counter(verdict(n, late, late_forced) for n in cands)
p(f"| candidates holding `main`'s tip, and `main` | {len(cands)} | {len(cands)} | {c['pass']} | {c['stops-pass']} | {c['fail'] + c['stops-fail']} | {c['stops-unresolved']} |")
p()
fails = [r for r in rows if V[f"pr-{r['pr']}"] in ("fail", "stops-fail")] + [n for n in cands if verdict(n, late, late_forced) in ("fail", "stops-fail")]
if fails:
    p("**Would fail:**")
    p()
    for r in fails:
        n = r if isinstance(r, str) else f"pr-{r['pr']}"
        src = (late_forced if n in late_forced else late) if n in late else (forced if n in forced else first)
        f = src[n]
        p(f"- {'#' + n[3:] if n.startswith('pr-') else n} ({'' if isinstance(r, str) else str(r['lane']) + ', ' + group(r)}): "
          f"`{f.get('pytest', '')}`; new {[x['id'] for x in f.get('new', [])]}; gone {f.get('gone')}; moved {len(f.get('moved', []))}")
    p()
else:
    p(f"**Still none would fail**: 0 of the {len(marked)} marked, 0 of the rest.")
    p()
mstops = [r for r in marked if V[f"pr-{r['pr']}"] != "pass"]
if mstops:
    p(f"The {len(mstops)} marked ones whose merge stops:")
    p()
    for r in mstops:
        n = f"pr-{r['pr']}"
        src = late if where[n] == "late" else first
        files = [f for fs in src[n]["conflicts"].values() for f in fs]
        p(f"- #{r['pr']} ({r['lane']}, {r['state']}): {len(files)} files, {sum(1 for f in files if f in opened)} read by the test; "
          + ("run in this pass" if where[n] == "late" else "as in the table above") + f"; {V[n].replace('stops-', '')} with its side")
    p()
base_ids = set(first["main"]["ids"])
changed = []
for n, r in late.items():
    f = late_forced.get(n, r) if r.get("conflicts") else r
    if "ids" in f:
        add, cut = set(f["ids"]) - base_ids, base_ids - set(f["ids"])
        if add or cut:
            changed.append((n, add, cut))
if changed:
    by = {f"pr-{r['pr']}": r for r in rows}
    p("Heads of this pass that register other ids than the base, none leaving a version superseded:")
    p()
    for n, add, cut in changed:
        r = by.get(n)
        high = sorted(i for i in add if not i.endswith("_v1"))
        p(f"- {'#' + n[3:] if n.startswith('pr-') else n}" + (f" ({r['lane']}, {group(r)})" if r else "") + f": {len(add)} added, {len(cut)} removed"
          + (f"; above `_v1`: {', '.join('`' + i + '`' for i in high[:6])}" + (f" and {len(high) - 6} more" if len(high) > 6 else "") if high else ""))
    p()
unres = [r for r in rows if V[f"pr-{r['pr']}"] == "stops-unresolved"]
if unres:
    p("No verdict (the head's side of its conflicts gives no tree whose registries load): "
      + ", ".join(f"#{r['pr']}" for r in unres) + ".")
