#!/usr/bin/env python3
"""VLLM-289 scratch: the rehearsal's one table, from `meta.py`'s JSON and the runs' out/results -> Markdown on stdout.

    table.py META.json RUN... [--stand-in RUN...] [--norow NOROW.json]

RUNs before `--stand-in` are the rehearsal on the staging; those after it are the rehearsal on the stand-in staging over the
candidate (`drive.py --stand-in`), which has a column of its own. NOROW.json is `norow.py`'s: the heads whose own new file the
candidate's map has no row for, which the script refuses.
A run's result for a head the pull request no longer has is not its row's outcome; a later run's result for the same label
replaces an earlier one's. A mend's own result is the one labelled `mPR`.
"""
import glob
import json
import os
import subprocess
import sys

meta = json.load(open(sys.argv[1]))
rest = sys.argv[2:]
norow = {}
if "--norow" in rest:
    i = rest.index("--norow")
    norow = json.load(open(rest[i + 1]))
    del rest[i:i + 2]
cut = rest.index("--stand-in") if "--stand-in" in rest else len(rest)
runs, runs2 = rest[:cut], rest[cut + 1:]
PRIORITY = [8378, 8448, 8529, 8567, 8566, 8585]
#: unmarked since the loop put it out of a candidate
EJECTED = {8378: "- (ejected 17:03Z; its mark and verdict are gone)"}
#: the candidate the loop dropped at 17:03Z: a mend that merged it holds two pull requests that are in no candidate now
DROPPED = "034a1702664bff0ac1f8b416b4715f00f84b15d0"
LEFT_OUT = {8027: "left out on the lead's word (its fix crosses under VLLM-249)",
            8656: "no restack: its commit is in the staging; closed when the layout lands"}
#: pr -> what the mend on cursor/vllm-restack-PR-7c1e resolves, or why there is none
MENDS = {8378: "the fix alone, to cherry-pick after the crossing: 13 calls of `map_names` drop `capsys`",
         8384: "`cli.py` drops both sides' deleted commands",
         8395: "`pyproject.toml`'s test list",
         8401: "one test file, both sides kept",
         8529: "on #8401's mend; `served_units`, both sides kept",
         8501: "`circuit_check`'s pyproject, `test_interface.py`",
         8567: "merges #8501's mend",
         8580: "`windows/interface.py`: code both sides changed",
         8584: "merges #8580's mend",
         8402: "`cli.py` drops both sides' deleted commands; needed only once #8378 is under it",
         8048: "4 files: code both sides changed",
         8060: "text only"}
NO_MEND = {7958: "none: `main` took the other design since; the author's, or dead",
           7874: "none: 11 files with a Lean definition and fixtures to derive again; the author's, or dead"}



def load(runs):
    res: dict = {}
    for r in runs:
        for f in glob.glob(os.path.expanduser(f"~/.research/runs/{r}/out/results/pr-*.json")):
            d = json.load(open(f))
            d["run"] = r
            res[str(d["pr"])] = d
    return res


res, res2 = load(runs), load(runs2)


def short(files, n=2):
    names = [os.path.basename(f) for f in files]
    return ", ".join(f"`{f}`" for f in names[:n]) + (f" +{len(names) - n}" if len(names) > n else "")


def stale_staging(d):
    """A stop that is the staging's and not the branch's: `af72` is staged under `main`'s #8404, so a head that holds it stops
    at the second move's base on the one line both changed."""
    return d.get("step") == "base" and len(d.get("crossed", [])) == 1 and d.get("files") == ["tests/test_verifier_cost.py"]


def outcome(d):
    st, step = d.get("status"), d.get("step")
    if st == "clean":
        return "`clean`"
    if stale_staging(d):
        return "not valid here †"
    if st == "conflicted":
        which = "base" if step in ("merge BASE", "base") and not d.get("crossed") else "base of move 2" if step == "base" else step
        return f"`conflicted` at {which}: {short(d.get('files', []))}"
    if st == "script-failed":
        return "`script-failed` ‡"
    return f"`{st}`" + (f" at {step}" if step else "") + (f": {(d.get('error') or '')[:70]}" if d.get("error") else "")


rows = {r["pr"]: r for r in meta["rows"]}
for m in meta["extra"].values():
    m["holds_dropped"] = subprocess.run(["git", "merge-base", "--is-ancestor", DROPPED, m["head"]]).returncode == 0


def klass(r):
    n = r["pr"]
    if n in LEFT_OUT:
        return 6
    if r["in_base"]:
        return 5
    if r["marked"]:
        return 0 if r["verdict"] == "yes" else 1
    return 2 if n in PRIORITY else 3


def depth(n, seen=()):
    under = [u for u in rows[n]["stacked_on"] if u in rows and u not in seen]
    return 1 + max(depth(u, (*seen, n)) for u in under) if under else 0


def stack_key(n):
    """A stack's rows together, bottom first: the bottom's number, then the depth."""
    r, seen = rows[n], {n}
    while (under := [u for u in r["stacked_on"] if u in rows and u not in seen]):
        seen.add(under[0])
        r = rows[under[0]]
    return r["pr"]


BASE = meta["base"][:12]
print(f"| PR | head | on | mark | restack on `41d5514d7680` | `git merge` of `{BASE}` | `Move:` commits it lacks | the day's form on a stand-in over `034a1702664b` | mend, or whose move |")
print("|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows.values(), key=lambda r: (klass(r), stack_key(r["pr"]) if klass(r) != 2 else 0, depth(r["pr"]), r["pr"])):
    n, h = r["pr"], r["head"]
    d = res.get(str(n))
    on = ", ".join(f"#{u}" for u in r["stacked_on"])
    if r["base"] != "main" and not r["base_pr"]:
        on = (on + "; " if on else "") + f"base `{r['base'][-11:]}`"
    mark = ("mark + land verdict" if r["verdict"] == "yes" else "mark, no verdict") if r["marked"] else EJECTED.get(n, "-")
    if d and d["head"] == h:
        dry = outcome(d)
    elif d:
        dry = f"head moved since ({outcome(d)} at `{d['head'][:10]}`)"
    else:
        dry = "not run"
    move, d2 = "", res2.get(str(n))
    if n in LEFT_OUT or r["in_base"]:
        two = ""
    elif d2 and d2["head"] == h:
        two = "stops at the merge" if d2.get("step") == "merge BASE" else outcome(d2)
    elif f"#{n}" in norow:
        two = "`script-failed` ‡ (read from the map)"
    else:
        two = "in the pool"
    if n in LEFT_OUT:
        merge, lacks, move = "", "", LEFT_OUT[n]
    elif r["in_base"]:
        merge, lacks = "in it: crosses by landing", ""
    else:
        merge = f"stops: {short(r['merge_base'])}" if r["merge_base"] else "clean"
        lacks = str(len(r["lacks_moves"])) if r["lacks_moves"] else "none"
        if n in MENDS:
            m = meta["extra"].get(f"m{n}", {})
            md = res2.get(f"m{n}")
            state = ("merge clean" if not m.get("merge_base") else f"merge stops: {short(m['merge_base'])}") if m else "?"
            ran = f"; crossed {outcome(md)} on the stand-in" if md and md["head"] == m.get("head") else ""
            redo = "; **to redo on the cut's base** (it merged the dropped `034a`)" if m.get("holds_dropped") else ""
            bad = "" if state == "merge clean" else f"; {state}"
            move = f"mend `{m.get('head', '')[:10]}`: {MENDS[n]}{bad}{ran}{redo}"
        elif n in NO_MEND:
            move = NO_MEND[n]
        elif r["merge_base"] or (d and d["head"] == h and d.get("status") != "clean" and not stale_staging(d)):
            move = "the author's"
        if f"#{n}" in norow:
            files = ", ".join(f"`{os.path.basename(x['file'])}`" for x in norow[f"#{n}"])
            move = (move + "; " if move else "") + f"no mend of mine: the map needs a row for {files} ‡"
    print(f"| #{n} | `{h[:10]}` | {on} | {mark} | {dry} | {merge} | {lacks} | {two} | {move} |")
