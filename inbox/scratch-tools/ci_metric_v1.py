"""ci's landing metrics (CI-485): ready-to-main time per PR, and main's lag, per day in PST.

Push times come from GitHub's repository activity for refs/heads/checked and refs/heads/main. A PR's landing is its
`Queue: #N` merge commit on checked's first-parent chain, whose second parent is the PR's head; it reaches main at the
first push of main whose tip holds that commit. Its clock starts at the ready mark at that head (drive's `ready` label on
pr:N@HEAD), else at its landing on checked when no mark is found. A PR that a `Revert "Queue: #N` commit on checked undid
counts as a miss. main's lag at a moment is the age of the oldest landing on checked that main doesn't hold yet.

usage: ci_metric.py [SINCE_ISO] [--json OUT]   (run inside a clone of the repository with checked and main fetched)
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

REPO = "compute-verification/verity"
PST = timezone(timedelta(hours=-8))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SINCE = datetime.fromisoformat(args[0]) if args else datetime.now(timezone.utc) - timedelta(days=3)
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
FLOOR_H, LAG_H = 2.0, 3.0


def pushes(ref: str) -> list[tuple[datetime, str]]:
    url = f"repos/{REPO}/activity?ref=refs/heads/{ref}&per_page=100&activity_type=push&time_period=month"
    r = subprocess.run(["gh", "api", "--paginate", url, "--jq", ".[] | [.timestamp, .after] | @tsv"],
                       capture_output=True, text=True, check=True).stdout
    out = [(datetime.fromisoformat(ts.replace("Z", "+00:00")), sha) for ts, sha in (ln.split("\t") for ln in r.split("\n") if ln)]
    return sorted(o for o in out if o[0] >= SINCE - timedelta(days=4))


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def first_seen(ps: list[tuple[datetime, str]]) -> dict[str, datetime]:
    seen: dict[str, datetime] = {}
    prev = None
    for t, sha in ps:
        if git("cat-file", "-t", sha).strip() != "commit":
            continue
        rng = [sha] + ([f"^{prev}"] if prev else [f"--since={(SINCE - timedelta(days=6)).isoformat()}"])
        for c in git("rev-list", "--first-parent", *rng).split():
            seen.setdefault(c, t)
        prev = sha
    return seen


def ready_at(pr: str, head: str) -> datetime | None:
    r = subprocess.run([sys.executable, "-m", "research", "data", "labels", f"pr:{pr}@{head}", "--key", "ready", "--json"],
                       capture_output=True, text=True, env={**os.environ, "RESEARCH_AT_CHECKOUT": "1"})
    try:
        rows = [x for x in json.loads(r.stdout or "[]") if x.get("value") is True]
    except json.JSONDecodeError:
        return None
    return min((datetime.fromisoformat(x["ts"].replace("Z", "+00:00")) for x in rows), default=None)


checked, main = pushes("checked"), pushes("main")
landed, on_main = first_seen(checked), first_seen(main)
now = datetime.now(timezone.utc)
info = {}
for line in git("log", "--first-parent", f"--since={(SINCE - timedelta(days=6)).isoformat()}", "--format=%H %P%x09%s",
                checked[-1][1]).splitlines():
    shas, _, subject = line.partition("\t")
    c, *parents = shas.split()
    info[c] = (parents, subject)
reverted = {m.group(1) for _, s in info.values() if (m := re.match(r'Revert "Queue: #(\d+)', s))}
prs = []
for c, t in landed.items():
    parents, subject = info.get(c, ([], ""))
    m = re.match(r"Queue: #(\d+)", subject)
    if m and len(parents) > 1 and t >= SINCE - timedelta(days=1):
        prs.append({"pr": m.group(1), "head": parents[1], "landing": c, "checked": t, "main": on_main.get(c)})
with ThreadPoolExecutor(12) as ex:
    for p, r in zip(prs, ex.map(lambda p: ready_at(p["pr"], p["head"]), prs)):
        p["ready"], p["marked"] = (r, True) if r and r <= p["checked"] else (p["checked"], False)
prs = [p for p in prs if p["ready"] >= SINCE]


def day(t: datetime) -> str:
    return t.astimezone(PST).strftime("%a %d %b")


def q(xs: list[float], f: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


rows = {}
for p in prs:
    done = p["main"] is not None and p["pr"] not in reverted
    hours = ((p["main"] if done else now) - p["ready"]).total_seconds() / 3600
    if not done and hours < FLOOR_H:
        continue
    r = rows.setdefault(day(p["ready"]), {"prs": 0, "hours": [], "within": 0, "reverted": 0, "unmarked": 0})
    r["prs"] += 1
    r["hours"].append(hours)
    r["within"] += done and hours <= FLOOR_H
    r["reverted"] += p["pr"] in reverted
    r["unmarked"] += not p["marked"]
lag: dict[str, list[float]] = {}
at = SINCE
while at < now:
    ages = [(at - p["checked"]).total_seconds() / 3600 for p in prs if p["checked"] <= at < (p["main"] or now)]
    lag.setdefault(day(at), []).append(max(ages, default=0.0))
    at += timedelta(minutes=10)

order = lambda kv: datetime.strptime(kv[0] + " 2026", "%a %d %b %Y")
print(f"ready to main, by the PST day of the ready mark (a PR still off main counts once {FLOOR_H:g} h have passed)")
print("day           PRs  median h  p90 h  on main within 2 h  reverted  no mark found")
for d, r in sorted(rows.items(), key=order):
    print(f"{d:12s}  {r['prs']:4d}  {q(r['hours'], .5):8.1f}  {q(r['hours'], .9):5.1f}  {r['within'] / r['prs']:18.0%}  "
          f"{r['reverted']:8d}  {r['unmarked']:13d}")
print("\nmain's lag (the oldest landing on checked not yet on main), sampled every 10 min")
print("day           worst h  share of the day at most 3 h")
for d, xs in sorted(lag.items(), key=order):
    print(f"{d:12s}  {max(xs):7.1f}  {sum(x <= LAG_H for x in xs) / len(xs):6.0%}")
print(f"\nas of {now:%Y-%m-%d %H:%MZ}; main at {main[-1][1][:10]} (pushed {main[-1][0]:%d %b %H:%MZ}), checked at {checked[-1][1][:10]}")
if OUT:
    json.dump({"as_of": now.isoformat(), "since": SINCE.isoformat(),
               "ready_to_main": {d: {**r, "median_h": q(r["hours"], .5), "p90_h": q(r["hours"], .9)} for d, r in rows.items()},
               "lag": {d: {"worst_h": max(xs), "share_le_3h": sum(x <= LAG_H for x in xs) / len(xs)} for d, xs in lag.items()}},
              open(OUT, "w"), indent=1, default=str)
