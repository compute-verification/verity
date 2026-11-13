#!/usr/bin/env python3
"""make_inflight.py WORK ROWS.json HEAD MAIN STATUS_Z > inflight.md: the sweep's results as the document.

WORK holds the harness's results: out/ (each head merged with the base), out-forced/ (each head whose merge stopped, again
with the head's side of every conflict), out-controls/ (two heads made to fail), jobs.txt (NAME SHA a line) and
opened.txt (the tracked files the test opens at the base).  ROWS.json is queue_rows.py's reading of `research queue
status` with each pull request's fetched head (`sha`).  Everything the document counts is counted here from those files.
"""
import collections
import json
import sys
from pathlib import Path

work, rows_file, head, main, status_z = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
rows = {r["pr"]: r for r in json.loads(rows_file.read_text())}
jobs = [line.split()[:2] for line in (work / "jobs.txt").read_text().splitlines() if line.strip()]
opened = set((work / "opened.txt").read_text().split())


def load(d: str) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((work / d).glob("*.json"))}


first, forced, controls = load("out"), load("out-forced"), load("out-controls")
missing = [n for n, _ in jobs if n not in first]
assert not missing, f"not run: {missing}"
base_ids = set(first["main"]["ids"])


def passed(r: dict) -> bool:
    return (r.get("pytest", "").startswith("3 passed") and not r.get("pytest_failed") and not r.get("detail_error")
            and not r.get("new") and not r.get("gone") and not r.get("moved") and not r.get("twice"))


def verdict(name: str) -> str:
    """pass | stops-pass | stops-fail | stops-unresolved | fail

    stops-unresolved: the head's side of every conflict gives no tree (the merge can't be committed) or a tree whose
    registries don't load (the test's own functions stop on an import), so there is nothing to hold to the list."""
    r = first[name]
    if not r.get("conflicts"):
        return "pass" if passed(r) else "fail"
    f = forced.get(name)
    if f is None or "pytest" not in f or f.get("detail_error"):
        return "stops-unresolved"
    return "stops-pass" if passed(f) else "stops-fail"


def final(name: str) -> dict:
    return forced[name] if first[name].get("conflicts") and name in forced else first[name]


def group(name: str) -> str:
    if name.startswith("pr-"):
        r = rows[name[3:]]
        if not r["marked"]:
            return "not marked"
        return {"ready": "marked, admitted", "refused": "marked, refused", "waiting": "marked, waiting"}[r["state"]]
    return {"queue": "live candidates", "top": "top's staged move branch", "main": "`main`"}[name.split("-")[0]]


def lane(name: str) -> str:
    if name.startswith("pr-"):
        return rows[name[3:]]["lane"] if rows[name[3:]]["lane"] != "-" else "no lane"
    return "top" if name.startswith("top") else "the queue"


def pr(name: str) -> str:
    return f"#{name[3:]}" if name.startswith("pr-") else name


GROUPS = ["marked, admitted", "marked, refused", "marked, waiting", "live candidates", "top's staged move branch", "`main`",
          "not marked"]
by_group = collections.defaultdict(list)
for n, _ in jobs:
    by_group[group(n)].append(n)
V = {n: verdict(n) for n, _ in jobs}
marked = [n for n, _ in jobs if group(n).startswith("marked")]
unmarked = by_group["not marked"]
fails = [n for n, _ in jobs if V[n] in ("fail", "stops-fail")]
unresolved = [n for n, _ in jobs if V[n] == "stops-unresolved"]
moved = sorted((r["pr"] for r in rows.values() if r.get("moved_since_status")), key=int)
p = print

p("# CIR-208: the superseded-versions check against the queue, Fri 13 Nov")
p()
p(f"The base is #8463's head `{head}`, which holds `main` `{main[:12]}` (and so #9375, on `main` since 03:01Z). The queue is")
p(f"`research queue status` as read at {status_z}: {len(rows)} open pull requests, {sum(1 for r in rows.values() if r['marked'])} of them")
p("marked (" + ", ".join(f"{sum(1 for r in rows.values() if r['marked'] and r['state'] == s)} {w}" for s, w in
                         (("ready", "admitted"), ("refused", "refused"), ("waiting", "waiting on another"))) + "). Each pull request's")
p("head is `refs/pull/N/head` as fetched right after" + (f"; {len(moved)} had moved between the two reads (" +
  ", ".join(f"#{m}" for m in moved) + "), and the fetched head is the one run" if moved else "") + ". A live candidate is a")
p("`refs/heads/queue/*` ref that holds `main`'s tip. #8463 itself is left out.")
p()
p("## Result")
p()
nm_fail = [n for n in fails if n in marked]
p(f"**{len(nm_fail)} of the {len(marked)} marked pull requests would fail the check, and no live candidate does.** "
  f"Of the {len(unmarked)} not marked, {sum(1 for n in unmarked if n in fails)} would.")
p()
p("| heads | count | merged with the base, the test passes | the merge stops; with the head's side of each conflict the test passes | would fail | the merge stops, no verdict |")
p("|---|---|---|---|---|---|")
for g in GROUPS:
    ns = by_group[g]
    if ns:
        c = collections.Counter(V[n] for n in ns)
        p(f"| {g} | {len(ns)} | {c['pass']} | {c['stops-pass']} | {c['fail'] + c['stops-fail']} | {c['stops-unresolved']} |")
c = collections.Counter(V.values())
p(f"| all | {len(jobs)} | {c['pass']} | {c['stops-pass']} | {c['fail'] + c['stops-fail']} | {c['stops-unresolved']} |")
p()
if fails:
    p("## The heads that would fail")
    p()
    p("| head | lane | state | versions it leaves superseded, or what else the test finds | way through |")
    p("|---|---|---|---|---|")
    for n in fails:
        r = final(n)
        new = ", ".join(f"`{x['id']}` (under `{x['by']}`, `{x['file']}`)" for x in r.get("new", []))
        other = "; ".join(f"{k}: {r[k]}" for k in ("gone", "moved", "twice") if r.get(k)) or r.get("pytest", "")
        p(f"| {pr(n)} | {lane(n)} | {group(n)} | {new or other} | to be read by hand |")
    p()
else:
    p("So the table the brief asks for, one row for each pull request that would fail with the versions it leaves superseded")
    p("and its way through, has no row.")
    p()

p("## How a head was run, and the controls")
p()
p("One scratch worktree a head (`check.py`): the base, `git merge` of the head, then `pytest` on the test file alone with")
p("every workspace package taken from that worktree, then the test's own functions (`registered`, `superseded`, `listed`,")
p("`findings`) for what the merged tree registers and what the list lacks. \"Passes\" is `3 passed` and no superseded")
p("version without an entry, no entry for a version that is gone, none filed under another file and none twice.")
p()
for n, r in controls.items():
    what = {"controlA": "a Definition change (`F32Sqrt_v2` registered without the word it computes, so it supersedes `F32Sqrt_v1`)",
            "controlB": "the list with `NvLogf_v2`'s entry taken out"}.get(n, n)
    p(f"- Control, merged the same way: {what}: `{r.get('pytest', '')}`, naming {', '.join('`' + x['id'] + '`' for x in r.get('new', []))}.")
p(f"- `main` itself merged with the base: `{first['main'].get('pytest', '')}`, {first['main']['superseded']} superseded, "
  f"{first['main']['entries']} entries, {len(first['main']['ids'])} registered.")
p()
changed = []
for n, _ in jobs:
    r = final(n)
    if "ids" in r and n != "main":
        add, cut = set(r["ids"]) - base_ids, base_ids - set(r["ids"])
        if add or cut:
            changed.append((n, add, cut, r))
p(f"The test was not idle on these heads: {len(changed)} of them register other ids than the base does "
  f"({sum(1 for n, *_ in changed if n in marked)} marked, {sum(1 for n, *_ in changed if n in unmarked)} not marked, "
  f"{sum(1 for n, *_ in changed if not n.startswith('pr-'))} of the candidates and branches), and none leaves a version superseded:")
p()
p("| head | lane | state | ids added | ids removed | versioned above `_v1` among the added |")
p("|---|---|---|---|---|---|")
for n, add, cut, r in changed:
    high = sorted(i for i in add if not i.endswith("_v1"))
    p(f"| {pr(n)} | {lane(n)} | {group(n)} | {len(add)} | {len(cut)} | "
      f"{', '.join('`' + i + '`' for i in high[:6]) + (f' and {len(high) - 6} more' if len(high) > 6 else '') if high else 'none'} |")
p()
extra = [(n, final(n).get("extra", {})) for n, _ in jobs if final(n).get("extra")]
if extra:
    p("Definitions a head's changed modules register only when something imports them, which the registries' loader does")
    p("not (so the check, like circuit-check, does not see them):")
    p()
    for n, x in extra:
        p(f"- {pr(n)} ({lane(n)}): {len(x)}, " + ", ".join(f"`{i}`" for i in sorted(x)[:5]) + (" ..." if len(x) > 5 else ""))
    p()

p("## Marked pull requests whose merge with `main` stops")
p()
stops = [n for n in marked if first[n].get("conflicts")]
p(f"{len(stops)} of the {len(marked)}. None stops in a file of this change: each stops against `main` alone on the same files")
p("(`git merge-tree` of `main` and the head; `sweep/conflicts-main-alone.json`), which is the queue's own reason for most of")
p("the refused. For these the test was run on a scratch resolution, the pull request's side of every")
p("conflict, so its verdict is of that tree and not of the merge its owner will make. \"Read by the test\" counts the")
p(f"conflicted files among the {len(opened)} tracked files the test opens at the base: where it is 0, no file the registries")
p("load is in conflict, and the verdict doesn't depend on the side taken.")
p()
p("| pull request | lane | state | files in conflict | read by the test | with the pull request's side |")
p("|---|---|---|---|---|---|")
for n in stops:
    files = [f for fs in first[n]["conflicts"].values() for f in fs]
    f = forced.get(n, {})
    res = (f"`{f['pytest']}`, {f.get('superseded')} superseded, {f.get('entries')} entries" if "pytest" in f else
           "not resolved: " + (f.get("force_error") or f.get("merge_error") or "no result")[-80:].replace("\n", " "))
    shown = ", ".join(f"`{x}`" for x in files[:3]) + (f" and {len(files) - 3} more" if len(files) > 3 else "")
    p(f"| {pr(n)} | {lane(n)} | {rows[n[3:]]['state']} | {len(files)}: {shown} | {sum(1 for x in files if x in opened)} | {res} |")
p()
by_lane = collections.Counter(lane(n) for n in stops)
p("By lane: " + ", ".join(f"{k} {v}" for k, v in sorted(by_lane.items(), key=lambda kv: -kv[1])) + ".")
p()

p("## The pull requests not marked (a forecast: nobody asked the queue to land these)")
p()
p("| lane | heads | pass | merge stops, passes with its side | would fail | merge stops, no verdict |")
p("|---|---|---|---|---|---|")
for k, ns in sorted(collections.Counter(lane(n) for n in unmarked).items(), key=lambda kv: -kv[1]):
    c = collections.Counter(V[n] for n in unmarked if lane(n) == k)
    p(f"| {k} | {ns} | {c['pass']} | {c['stops-pass']} | {c['fail'] + c['stops-fail']} | {c['stops-unresolved']} |")
p()
ust = [n for n in unmarked if first[n].get("conflicts")]
read = [n for n in ust if any(f in opened for fs in first[n]["conflicts"].values() for f in fs)]
p(f"{len(ust)} of them stop against `main`; {len(read)} of those stop in a file the test reads"
  + (" (" + ", ".join(pr(n) for n in read[:40]) + ("..." if len(read) > 40 else "") + "), so for them the scratch resolution's verdict"
     " is weak: the owner's merge decides what the registries hold" if read else "") + ".")
if unresolved:
    p()
    p("No verdict, since the head's side of its conflicts gives no tree whose registries load (each is an older tree than")
    p("the layout's move, and its side of a conflicted module imports a package that is gone):")
    p()
    for n in unresolved:
        f = forced.get(n, {})
        why = (f.get("detail_error") or f.get("force_error") or f.get("merge_error") or "no result").strip().splitlines()[-1][:140]
        files = sum(len(v) for v in first[n]["conflicts"].values())
        p(f"- {pr(n)} ({lane(n)}, {group(n)}): {files} files in conflict; `{why}`")
