"""Assemble the landing trace (landing-trace/v1) from the parsed sources."""
import json
import re
from datetime import datetime, timedelta, timezone


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


raw = json.load(open("/tmp/trace/trace_v1_raw.json"))
T = json.load(open("/tmp/trace/ticks.json"))
C = json.load(open("/tmp/trace/causes.json"))
stages = {r["pr"]: r for r in json.load(open("/tmp/trace/stages.json"))}

# main moves: GitHub mergedAt clusters (exact push time), with drive's line saying how
ff = []
cur_end = None
for line in open("/tmp/trace/drive.log", encoding="utf-8", errors="replace"):
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick rc=", line)
    if m:
        for f in ff:
            f.setdefault("tick_end", m[1])
        continue
    m = re.match(r"^research queue drive: `main` fast-forwarded to ([0-9a-f]{12}), which its after-landing run (.*?) passed", line)
    if m:
        ff.append(dict(sha=m[1], run=m[2], how="ci baseline (hand run)" if "ci baseline" in m[2] else "drive's after-landing run"))
    m = re.match(r"^research queue drive: landed ((?:#\d+ ?)+) on `main` at ([0-9a-f]{12}) \(check ([^)]*)\)", line)
    if m:
        ff.append(dict(sha=m[2], run=m[3] or None, how="drive landed on main directly (pre-checked era)"))

mains = sorted({r["main"] for r in raw["records"] if r["main"]})
clusters = []
for m in mains:
    t = ts(m)
    if clusters and (t - clusters[-1]["_last"]).total_seconds() < 600:
        clusters[-1]["_last"] = t
    else:
        clusters.append(dict(_first=t, _last=t))
for c in clusters:
    c["prs"] = sorted(r["pr"] for r in raw["records"] if r["main"] and c["_first"] <= ts(r["main"]) <= c["_last"])
moves = []
prev = None
for c in clusters:
    near = [f for f in ff if f.get("tick_end") and abs((ts(f["tick_end"]) - c["_first"]).total_seconds()) < 3 * 3600]
    best = min(near, key=lambda f: abs((ts(f["tick_end"]) - c["_first"]).total_seconds()), default=None)
    moves.append(dict(ts=c["_first"].strftime("%Y-%m-%dT%H:%M:%SZ"), n_prs=len(c["prs"]), prs=c["prs"],
                      gap_before_h=round((c["_first"] - prev).total_seconds() / 3600, 2) if prev else None,
                      how=best["how"] if best else "unmatched in drive's log (hand `research merge` or GitHub-only)",
                      sha=best["sha"] if best else None, check=best["run"] if best else None))
    prev = c["_last"]

prs = []
for r in raw["records"]:
    s = stages.get(r["pr"], {})
    r = dict(r)
    for k in ("rework", "admission", "review", "to_checked", "after_landing", "r2m", "r2m_first"):
        if k in s:
            r[k + "_h"] = round(s[k], 3) if s[k] is not None else None
    prs.append(r)

ticks = []
for t, st in zip(T["ticks"], C["tick_state"]):
    ticks.append(dict(end=t["end"], start=t["start"], took_s=t.get("took_s"), main_at_start=t.get("main"), rc=t.get("rc"),
                      next_size=t.get("next_size"), waiting_admission=t.get("waiting"), stopped=st["stopped"],
                      bottom_block=st["block"][0], bottom_block_pr=st["block"][1], after_landing_state=st["al"],
                      landed_checked=t["landed_checked"], landed_main=t["landed_main"], lean_checks=t["lean_checks"],
                      postland=t["postland"]))

doc = dict(
    schema="landing-trace/v1",
    generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    window=dict(start="2026-10-22T07:00:00Z", end=T["ticks"][-1]["end"], note="PRs on main with mergedAt >= start, plus PRs open now whose current head carries ready=true"),
    sources=dict(
        drive_log="control pod /data/drive/drive.log (19 Oct to the window's end; tick end lines carry main's SHA at tick start)",
        labels="store labels/pr_* (ready, reviewed) per pr:N@HEAD, every assertion with its ts",
        github="REST pulls (closed since 14 Oct, open now): createdAt, mergedAt (main's push time), head SHA",
        git="origin/main, origin/checked, origin/next reachability of each head"),
    stage_definitions=dict(
        opened="GitHub createdAt",
        marked="each ready=true label (head, by); unmarked: ready=false",
        admitted="first tick end at or after the final mark whose sync logged `preflight: #N at HEAD merged onto next` (logged only since 1 Nov; null before)",
        verdict="first reviewed=land at the final head by someone other than its marker",
        hold="each reviewed=hold at the final head",
        refused="drive told the owner the PR isn't admitted (reason in why)",
        ejected="drive told the owner the PR is out of next (reason in why)",
        checked="tick end of drive's `landed #N on checked` (low-risk lane, reviewed, or direct to main before 29 Oct)",
        main="GitHub mergedAt: when main's push first contained the head"),
    timing_resolution="drive events carry their tick's end time; a tick took a median of about 40 min since 1 Nov (took_s); labels and mergedAt are exact",
    main_moves=moves,
    prs=prs,
    events=sorted(raw["events"], key=lambda e: (e["pr"], e["ts"] or "")),
    ticks=ticks,
    pareto_all=C["pareto"], pareto_last7d=C["week"], incidents=C["incidents"][:200],
)
json.dump(doc, open("/tmp/trace/landing-trace-v1.json", "w"), default=str)
print(len(prs), "prs", len(doc["events"]), "events", len(moves), "main moves", len(ticks), "ticks")
from collections import Counter
print(Counter(m["how"] for m in moves))
for m in moves[-22:]:
    print(m["ts"], m["n_prs"], m["gap_before_h"], m["how"], (m["check"] or "")[:40])
