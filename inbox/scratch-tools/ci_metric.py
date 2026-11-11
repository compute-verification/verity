"""ci's landing metric (CI-485): ready-to-main time per PR, per day in PST. Scorer v2, for the verified queue.

A PR's landing is its `Queue: #N` merge commit on main's first-parent chain, whose second parent is the PR's head; it
reaches main at the first push of main whose tip holds that commit (GitHub's repository activity for refs/heads/main).
Its clock starts at the ready mark at that head (the `ready` label on pr:N@HEAD: the first `true` after the last
`false`, from before the push), else at the merge commit's own date when no mark is found. A PR that a
`Revert "Queue: #N` commit on main undid counts as a miss. A PR that is marked ready now and not on main (the queue's
admitted list) counts on its mark's day once 2 h have passed, timed to now.

v1 (ci_metric_v1.py) followed `checked`, which stopped moving on Mon 9 Nov 15:25Z when the verified queue took over:
its landings were built again onto main, so v1 read them as never reaching main. main's lag behind `checked` is printed
only for days `checked` still moved.

usage: ci_metric.py [SINCE_ISO] [--json OUT] [--status FILE]   (in a clone with main and checked fetched; FILE is
       `research queue status`'s output, default /tmp/qstatus.txt, read for the PRs still waiting)
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
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--json", "--status")]
SINCE = datetime.fromisoformat(args[0]) if args else datetime.now(timezone.utc) - timedelta(days=3)
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
STATUS = sys.argv[sys.argv.index("--status") + 1] if "--status" in sys.argv else "/tmp/qstatus.txt"
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


def ready_at(pr: str, head: str, before: datetime) -> datetime | None:
    r = subprocess.run([sys.executable, "-m", "research", "data", "labels", f"pr:{pr}@{head}", "--key", "ready",
                        "--history", "--json"], capture_output=True, text=True, env={**os.environ, "RESEARCH_AT_CHECKOUT": "1"})
    try:
        rows = sorted((datetime.fromisoformat(x["ts"].replace("Z", "+00:00")), x.get("value")) for x in json.loads(r.stdout or "[]"))
    except (json.JSONDecodeError, KeyError):
        return None
    mark = None
    for ts, value in rows:
        if ts > before:
            break
        mark = (mark or ts) if value is True else None
    return mark


def chain(tip: str) -> dict[str, tuple[list[str], str, datetime]]:
    out = {}
    for line in git("log", "--first-parent", f"--since={(SINCE - timedelta(days=6)).isoformat()}",
                    "--format=%H %P%x09%cI%x09%s", tip).splitlines():
        shas, when, subject = line.split("\t", 2)
        c, *parents = shas.split()
        out[c] = (parents, subject, datetime.fromisoformat(when))
    return out


checked, main = pushes("checked"), pushes("main")
on_checked, on_main = first_seen(checked), first_seen(main)
now = datetime.now(timezone.utc)
info = chain(main[-1][1])
reverted = {m.group(1) for _, s, _ in info.values() if (m := re.match(r'Revert "Queue: #(\d+)', s))}
prs = []
for c, t in on_main.items():
    parents, subject, built = info.get(c, ([], "", None))
    m = re.match(r"Queue: #(\d+)", subject)
    if m and len(parents) > 1 and t >= SINCE - timedelta(days=1):
        prs.append({"pr": m.group(1), "head": parents[1], "landing": c, "built": built, "main": t})
with ThreadPoolExecutor(12) as ex:
    for p, r in zip(prs, ex.map(lambda p: ready_at(p["pr"], p["head"], p["main"]), prs)):
        p["ready"], p["marked"] = (r, True) if r else (min(p["built"] or p["main"], p["main"]), False)

landed_heads = {(p["pr"], p["head"]) for p in prs}
waiting = []
if os.path.exists(STATUS):
    admitted = re.findall(r"^\s+-\s+#(\d+)\s+([0-9a-f]{8})\s+ready\s", open(STATUS).read(), re.M)
    heads = json.loads(subprocess.run(["gh", "pr", "list", "--repo", REPO, "--state", "open", "--limit", "800", "--json",
                                       "number,headRefOid"], capture_output=True, text=True, check=True).stdout)
    full = {str(h["number"]): h["headRefOid"] for h in heads}
    todo = [(n, full[n]) for n, short in admitted if full.get(n, "").startswith(short) and (n, full[n]) not in landed_heads]
    with ThreadPoolExecutor(12) as ex:
        for (n, head), r in zip(todo, ex.map(lambda x: ready_at(x[0], x[1], now), todo)):
            if r:
                waiting.append({"pr": n, "head": head, "landing": None, "built": None, "main": None, "ready": r, "marked": True})
everything = [p for p in prs + waiting if p["ready"] >= SINCE]


def day(t: datetime) -> str:
    return t.astimezone(PST).strftime("%a %d %b")


def q(xs: list[float], f: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


rows = {}
for p in everything:
    done = p["main"] is not None and p["pr"] not in reverted
    hours = ((p["main"] if done else now) - p["ready"]).total_seconds() / 3600
    if not done and hours < FLOOR_H:
        continue
    r = rows.setdefault(day(p["ready"]), {"prs": 0, "hours": [], "within": 0, "reverted": 0, "unmarked": 0, "waiting": 0})
    r["prs"] += 1
    r["hours"].append(hours)
    r["within"] += done and hours <= FLOOR_H
    r["reverted"] += p["pr"] in reverted
    r["unmarked"] += not p["marked"]
    r["waiting"] += p["main"] is None

# main's lag behind checked, while checked moved: a landing on checked is on main once main's chain holds that commit.
old = chain(checked[-1][1]) if checked else {}
stopped = checked[-1][0] if checked else SINCE
lag: dict[str, list[float]] = {}
at = SINCE
day_end = (stopped.astimezone(PST).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(timezone.utc)
while at < min(now, day_end):
    ages = [(at - t).total_seconds() / 3600 for c, t in on_checked.items()
            if re.match(r"Queue: #", old.get(c, ([], "", None))[1]) and t <= at < (on_main.get(c) or now)]
    lag.setdefault(day(at), []).append(max(ages, default=0.0))
    at += timedelta(minutes=10)

order = lambda kv: datetime.strptime(kv[0] + " 2026", "%a %d %b %Y")
print(f"ready to main, by the PST day of the ready mark (a PR still off main counts once {FLOOR_H:g} h have passed, timed to now)")
print("day           PRs  median h  p90 h  on main within 2 h  still waiting  reverted  no mark found")
for d, r in sorted(rows.items(), key=order):
    print(f"{d:12s}  {r['prs']:4d}  {q(r['hours'], .5):8.1f}  {q(r['hours'], .9):5.1f}  {r['within'] / r['prs']:18.0%}  "
          f"{r['waiting']:13d}  {r['reverted']:8d}  {r['unmarked']:13d}")
print(f"\nmain's lag behind checked (the oldest landing on checked not yet on main), every 10 min, to the end of the PST "
      f"day checked stopped ({stopped:%d %b %H:%MZ})")
print("day           worst h  share of the day at most 3 h")
for d, xs in sorted(lag.items(), key=order):
    print(f"{d:12s}  {max(xs):7.1f}  {sum(x <= LAG_H for x in xs) / len(xs):6.0%}  ({len(xs) / 6:.1f} h sampled)")
print(f"\nas of {now:%Y-%m-%d %H:%MZ}; main at {main[-1][1][:10]} (pushed {main[-1][0]:%d %b %H:%MZ}); "
      f"{len(prs)} landings read, {len(waiting)} marked PRs waiting")
if OUT:
    json.dump({"scorer": "v2", "as_of": now.isoformat(), "since": SINCE.isoformat(),
               "ready_to_main": {d: {**r, "median_h": q(r["hours"], .5), "p90_h": q(r["hours"], .9)} for d, r in rows.items()},
               "lag_behind_checked": {d: {"worst_h": max(xs), "share_le_3h": sum(x <= LAG_H for x in xs) / len(xs),
                                          "hours_sampled": len(xs) / 6} for d, xs in lag.items()},
               "landings": [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in p.items()} for p in everything]},
              open(OUT, "w"), indent=1, default=str)
