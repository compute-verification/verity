"""The open PRs that are carried or superseded, one row each, with the carrier and the evidence: from contain.py's output and
`research queue status --json`.

    python3 carried_csv.py C_DIR QSTATUS_JSON OUT_CSV REPO LABELS_JSON

Categories (a PR takes the first that applies):

- applied: its change (every file it changes, at its head's blob) is already on `main`, `checked` or `next`, its head not.
- carried-marked: its head is an ancestor of the head of a marked PR of its own lane that isn't on the demo line; withdrawing
  it loses nothing, and GitHub closes it as merged when the carrier reaches `main`.
- carried-unmarked-same-lane: as above, but no carrier of its lane is marked yet.
- carried-cross-lane: only another lane's PR contains it (that PR waits for it, or carries it once its mark is withdrawn).
- blob-carried: another open PR's head has its change at the same blobs, without the ancestry (a squash or a cherry-pick).
- demo-line: its head contains the demo branches' old `next` snapshot (`DEMO_BASE`), so it contains about 115 other open PRs
  and lands through the queue only after all of them.
- declared: its body or a carrier's says it carries, folds or supersedes it, with no git evidence (listed for a human look).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

#: Tue 3 Nov's merge into the PoUW full-width run's tree, below `remote`, `cursor/onsite-demo-061a` and the soundness branches,
#: with 651 of `next`'s merge commits not on `checked`
DEMO_BASE = "ea3f9c563560b66af50fb81e969e7a818e672586"
MARKED = {"ready", "on_next", "landed_checked"}
CARRY_RE = re.compile(r"(carr(?:y|ies|ied)|supersed\w*|replac\w*|fold\w*|subsum\w*|absorb\w*)[^.\n]{0,80}?#(\d{4,5})", re.I)


def state(q: dict | None) -> str:
    if not q:
        return "not-queued"
    if q["state"] == "queued":
        return "landed_checked" if "landed on `checked`" in (q.get("why") or "") else "on_next"
    return q["state"]


def main(c_dir: str, qstatus: str, out_csv: str, repo: str, labels_json: str) -> None:
    recs = {r["pr"]: r for r in json.load(open(Path(c_dir) / "prs.json"))}
    labs = {int(k): v for k, v in json.load(open(labels_json)).items()}
    verdicts = {n for n, r in recs.items()
                if any(x["key"] == "reviewed" and x["value"] == "land" and x["target"].endswith("@" + r["head"]) for x in labs.get(n, ()))}
    pairs = json.load(open(Path(c_dir) / "pairs.json"))["pairs"]
    q = {x["pr"]: x for x in json.load(open(qstatus))}
    st = {n: state(q.get(n)) for n in recs}
    demo = {n for n, r in recs.items()
            if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", DEMO_BASE, r["head"]]).returncode == 0}
    anc, blob = defaultdict(set), defaultdict(set)
    for p in pairs:
        (anc if p["how"] == "ancestry" else blob)[p["carried"]].add(p["carrier"])
    bodies = {n: r.get("title", "") for n, r in recs.items()}
    declared = defaultdict(set)
    for p in json.load(open(Path(c_dir).parent / "open.json")):
        for m in CARRY_RE.finditer(p.get("body") or ""):
            if (c := int(m.group(2))) in recs and c != p["number"]:
                declared[c].add(p["number"])

    def nearest(n: int, cs: set[int]) -> list[int]:
        """Carriers of ``n`` no other carrier of ``n`` is contained in: the ones closest to it, marked and on `next` first."""
        cs = {c for c in cs if c not in demo} or cs
        near = [c for c in cs if not any(c in anc.get(d, ()) for d in cs if d != c)]
        rank = {"on_next": 0, "landed_checked": 0, "ready": 1}
        return sorted(near, key=lambda c: (rank.get(st[c], 2), c))

    rows = []
    for n, r in sorted(recs.items()):
        if r["ancestor_checked"]:
            continue
        lane, cat, carriers, ev, act = r["lane"], None, [], "", ""
        if r["applied_main"] or r["applied_checked"] or r["applied_next"]:
            where = [b for b in ("main", "checked", "next") if r[f"applied_{b}"]]
            cat, ev = "applied", f"all {len(r['files'])} changed files at its head's blobs on {', '.join(where)}"
            act = "close: its change is already there" if "next" not in where or len(where) > 1 else "withdraw: next carries it"
        elif n in demo:
            cat = "demo-line"
            ev = (f"head contains {DEMO_BASE[:10]} (the demo branches' old next, 651 queue merges); contains "
                  f"{len(anc_of := [c for c, s in anc.items() if n in s])} open heads, {r['own_commits']} commits above checked")
            act = "restack its own commits onto main, or retarget to its demo branch and leave the queue"
        elif anc.get(n):
            same = {c for c in anc[n] if recs[c]["lane"] == lane}
            marked_same = {c for c in same if st[c] in MARKED}
            if marked_same:
                cat, carriers = "carried-marked", nearest(n, marked_same)
                c0 = carriers[0]
                if st[n] not in MARKED:
                    act = f"nothing: unmarked, same owner, so #{c0} carries it; GitHub closes it as merged when #{c0} lands"
                elif n in verdicts:
                    act = f"keep: it has its land verdict, so it lands first; #{c0} follows"
                elif st[c0] == "on_next" or st[n] != "on_next":
                    act = (f"fold: eject it 'carried by #{c0}' (both {'on next' if st[n] == 'on_next' else 'admitted'}): one merge, "
                           f"one preflight, one verdict; GitHub closes it as merged when #{c0} lands")
                else:
                    act = f"keep: it is on next and #{c0} isn't yet, so it lands first; fold only if #{c0} is the lane's priority"
            elif same:
                cat, carriers = "carried-unmarked-same-lane", nearest(n, same)
                act = f"leave unmarked or close into #{carriers[0]}; it closes when #{carriers[0]} lands"
            else:
                cat, carriers = "carried-cross-lane", nearest(n, anc[n])
                act = "land it (its carrier waits for it), or its owner withdraws it so the carrier carries it"
            ev = f"head {r['head'][:10]} is an ancestor of " + ", ".join(f"#{c} ({recs[c]['head'][:10]}, {st[c]})" for c in carriers[:4])
        elif blob.get(n):
            cat, carriers = "blob-carried", sorted(blob[n])
            ev = f"its {len(r['files'])} changed files have their head's blobs at " + ", ".join(f"#{c}" for c in carriers[:4])
            act = "check by hand: same content, no ancestry (a lockfile both regenerated, or a squash)"
        elif declared.get(n):
            cat, carriers = "declared", sorted(declared[n])
            ev = "a carrier's body says it carries, folds or supersedes it; no ancestry or blob evidence"
            act = "ask the carrier's lane"
        if not cat:
            continue
        rows.append({"pr": n, "lane": lane, "state": st[n], "head": r["head"][:12], "category": cat,
                     "carrier": carriers[0] if carriers else "", "carrier_state": st[carriers[0]] if carriers else "",
                     "carrier_lane": recs[carriers[0]]["lane"] if carriers else "", "n_carriers": len(anc.get(n, ())),
                     "evidence": ev, "action": act, "title": bodies[n][:90]})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r["category"] for r in rows))
    print(Counter((r["category"], r["state"]) for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:6])
