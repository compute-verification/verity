"""landing-trace/v2: v1 plus per-PR cause hours, change class, re-heads, red tells on checked, and daily flow."""
import collections
import gzip
import json
import re
from datetime import datetime, timedelta, timezone

v1 = json.load(open("/tmp/trace/landing-trace-v1.json"))
pp = json.load(open("/tmp/trace/per_pr_causes.json"))
pf = json.load(open("/tmp/trace/pr_files.json"))
rh = json.load(open("/tmp/trace/reheads.json"))

rows, buf = [], []
for line in open("/tmp/trace/drive.log", encoding="utf-8", errors="replace"):
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick rc=", line)
    if m:
        rows += [(m[1], b) for b in buf]
        buf = []
    else:
        buf.append(line.rstrip("\n"))

reds = []
for t, l in rows:
    m = re.search(r"told (@[\w @-]+): (suite|step) (\S+) is red on `checked` (\w{12}) \(run ([^)]*)\)", l)
    if not m:
        continue
    if "log has no error line" in l:
        kind = "no error line (machine or step failure)"
    elif "read outside its declared inputs" in l:
        kind = "read outside declared inputs"
    elif "tests/lint/" in l or "ratchet" in l or "retired_term" in l or "test_repo_" in l:
        kind = "lint, ratchet or prose invariant"
    else:
        kind = "test failure"
    cul = re.search(r"Culprit: #(\d+)", l)
    rev = "Reverted nothing" not in l and "Reverted" in l
    reds.append(dict(tick_end=t, told=m[1], part=f"{m[2]} {m[3]}", checked=m[4], runs=m[5].split(), kind=kind,
                     culprit=int(cul[1]) if cul else None, reverted=rev, text=l[l.find("):") + 3:][:400]))


def cls(n):
    f = pf.get(str(n))
    if not f:
        return None
    return "lean" if f["lean"] else ("lowrisk" if f["lowrisk"] else "nonlean")


for p in v1["prs"]:
    n = str(p["pr"])
    p["cause_hours"] = {k: round(v, 3) for k, v in sorted(pp.get(n, {}).items()) if v > 0}
    p["change_class"] = cls(n)
    p["reheads"] = [dict(frm=r["frm"], to=r["to"], kind=r["kind"], frm_marked=r["frm_marked"], to_marked=r["to_marked"])
                    for r in rh if str(r["pr"]) == n]


def pst_day(s):
    return (datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=8)).strftime("%Y-%m-%d")


flow = collections.defaultdict(collections.Counter)
for e in v1["events"]:
    if e["ts"] >= "2026-10-21":
        flow[pst_day(e["ts"])][e["stage"]] += 1

v2 = dict(v1)
v2["schema"] = "landing-trace/v2"
v2["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
v2["supersedes"] = "art:436e9968048fe4d6c4b25d39c6a7f0c2b5a0bc74c6a32e26744186e0105c09de"
v2["stage_definitions"] = dict(v1["stage_definitions"],
    cause_hours="per PR: hours from its final mark to main, each drive pass's interval charged to the one cause that held "
                "the PR at that pass (drive stopped; admission queued, waiting for a preflight turn, or refused; on next "
                "behind a running Lean check, a member without a land verdict, a hold, a rebuild, or tick cadence; on "
                "checked with an after-landing run going, going with a red named, red with no run, or ci's hand run that "
                "drive can't see); a stage's hours sum to its stage time up to one pass of rounding",
    change_class="lean: the PR's merge onto checked touches a Lean input (queue.lean_change approximated by path); lowrisk: "
                 "touches nothing drive.toml's lowrisk_excludes name (fnmatch approximation); nonlean: neither; null: no "
                 "merge commit on checked (open PRs, and PRs that reached main before checked existed)",
    reheads="each change of marked head: kind from `git rev-list --first-parent --no-merges frm..to` (merge-only: the PR "
            "added no commit of its own, only merges of main or other heads)",
    red_tells="drive's `told @X: suite|step S is red on checked SHA (run R)` lines, classified by their text")
v2["red_tells_on_checked"] = reds
v2["daily_flow_pst"] = {d: dict(c) for d, c in sorted(flow.items())}
out = "/tmp/trace/landing-trace-v2.json"
json.dump(v2, open(out, "w"), separators=(",", ":"))
print(out, len(v2["prs"]), "prs", len(reds), "red tells", sum(1 for p in v2["prs"] if p["cause_hours"]), "with causes")
