"""Parse drive's log into ticks with the landing events each holds."""
import json
import re
from datetime import datetime, timedelta, timezone

TS = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)"
TICK_END = re.compile(rf"^{TS} tick rc=(\d+)\s*([0-9a-f]*)")
STOPPED = re.compile(rf"^{TS} stopped: ")
LOCKED = re.compile(rf"^{TS} another copy holds")
TIMING = re.compile(r"^ci timing: tick (\d+) s")
LANDED_CHECKED = re.compile(r"^research queue drive: landed ((?:#\d+ ?)+) on `checked` at ([0-9a-f]{12}) \((.*)")
LANDED_MAIN = re.compile(r"^research queue drive: landed ((?:#\d+ ?)+) on `main` at ([0-9a-f]{12}) \(check ([^)]*)\)")
PREFLIGHT = re.compile(r"^research queue: preflight: #(\d+) at ([0-9a-f]{12}) merged onto `next`")
NEXT_SIZE = re.compile(r"^research queue: `next` at ([0-9a-f]{12}): (\d+) PR\(s\) above `(checked|main)`(.*)")
WAITING = re.compile(r"(\d+) waiting off it for their preflight \(([^)]*)\)")
NO_VERDICT = re.compile(r"^research queue drive: nothing reviewed lands: #(\d+) at ([0-9a-f]{8}): (.*)")
NOTHING_LANDS = re.compile(r"^research queue drive: nothing lands: #(\d+) at ([0-9a-f]{8}) waits")
LEAN_WAIT = re.compile(r"^research queue drive: #(\d+)'s Lean change waits for the Lean check")
POSTLAND = re.compile(r"^research queue drive: post-landing run of `(checked|main)` ([0-9a-f]{12}).*?: (.*)$")
STALE = re.compile(r"^research queue drive: rang (@[^:]+): `(checked|main)`'s last verified commit ([0-9a-f]{12}) is ([0-9.]+) h behind its tip ([0-9a-f]{12}) \((\d+) landing\(s\) unverified\); (.*)")
NOPOSTLAND = re.compile(r"^ci shim: no after-landing run started")
RANG_READY = re.compile(r"^research queue drive: rang (@\S+): marked ready (\d+) h ago or more and not landed on `checked`: (.*)")
OUT_NEXT = re.compile(r"^research queue drive: told (@[^:]+): #(\d+) at ([0-9a-f]{8}) (is out of `next`|isn't admitted)(.*)")
LEAN_CHECK = re.compile(r"^research queue drive: the Lean check (r\S+) of ([0-9a-f]{12}) (passed|failed)")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def prs(s):
    return [int(x) for x in re.findall(r"#(\d+)", s)]


def parse(path):
    ticks = []
    cur = dict(lines=0, landed_checked=[], landed_main=[], preflight=[], no_verdict=[], nothing_lands=[], lean_wait=[],
               postland=[], stale=[], next_size=None, waiting=None, waiting_prs=[], rang_ready=[], out_next=[], lean_checks=[])
    stopped = []
    last_end = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        if m := STOPPED.match(line):
            stopped.append(m[1])
            continue
        if LOCKED.match(line):
            continue
        if m := TICK_END.match(line):
            cur.update(end=m[1], rc=int(m[2]), main=m[3] or None)
            ticks.append(cur)
            cur = dict(lines=0, landed_checked=[], landed_main=[], preflight=[], no_verdict=[], nothing_lands=[], lean_wait=[],
                       postland=[], stale=[], next_size=None, waiting=None, waiting_prs=[], rang_ready=[], out_next=[], lean_checks=[])
            continue
        if m := TIMING.match(line):
            if ticks:
                ticks[-1]["took_s"] = int(m[1])
            continue
        if NOPOSTLAND.match(line):
            if ticks:
                ticks[-1]["no_postland"] = True
            continue
        cur["lines"] += 1
        if m := LANDED_CHECKED.match(line):
            cur["landed_checked"].append(dict(prs=prs(m[1]), sha=m[2], how=m[3][:120]))
        elif m := LANDED_MAIN.match(line):
            cur["landed_main"].append(dict(prs=prs(m[1]), sha=m[2], check=m[3]))
        elif m := PREFLIGHT.match(line):
            cur["preflight"].append((int(m[1]), m[2]))
        elif m := NEXT_SIZE.match(line):
            cur["next_size"] = int(m[2])
            cur["next_base"] = m[3]
            if w := WAITING.search(m[4]):
                cur["waiting"] = int(w[1])
                cur["waiting_prs"] = prs(w[2])
        elif m := NO_VERDICT.match(line):
            cur["no_verdict"].append(dict(pr=int(m[1]), head=m[2], why=m[3][:160]))
        elif m := NOTHING_LANDS.match(line):
            cur["nothing_lands"].append(dict(pr=int(m[1]), head=m[2]))
        elif m := LEAN_WAIT.match(line):
            cur["lean_wait"].append(int(m[1]))
        elif m := POSTLAND.match(line):
            cur["postland"].append(dict(branch=m[1], sha=m[2], rid=m[3][:200]))
        elif m := STALE.match(line):
            cur["stale"].append(dict(branch=m[2], verified=m[3], behind_h=float(m[4]), tip=m[5], n=int(m[6]), why=m[7][:600]))
        elif m := RANG_READY.match(line):
            cur["rang_ready"].append(dict(who=m[1], hours=int(m[2]), text=m[3][:2000]))
        elif m := OUT_NEXT.match(line):
            cur["out_next"].append(dict(pr=int(m[2]), head=m[3], what=m[4], why=m[5][:300]))
        elif m := LEAN_CHECK.match(line):
            cur["lean_checks"].append(dict(rid=m[1], sha=m[2], verdict=m[3]))
    for i, t in enumerate(ticks):
        end = ts(t["end"])
        prev = ts(ticks[i - 1]["end"]) if i else end - timedelta(minutes=10)
        t["prev_end"] = prev.strftime("%Y-%m-%dT%H:%M:%SZ")
        start = end - timedelta(seconds=t["took_s"]) if t.get("took_s") else prev
        t["start"] = max(start, prev).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ticks, stopped


if __name__ == "__main__":
    ticks, stopped = parse("/tmp/trace/drive.log")
    json.dump(dict(ticks=ticks, stopped=stopped), open("/tmp/trace/ticks.json", "w"))
    print(len(ticks), "ticks", ticks[0]["end"], ticks[-1]["end"], len(stopped), "stopped lines")
    print(sum(len(t["landed_checked"]) for t in ticks), "landed_checked lines", sum(len(t["landed_main"]) for t in ticks), "landed_main")
    mains = []
    for t in ticks:
        if t.get("main") and (not mains or mains[-1][1] != t["main"]):
            mains.append((t["end"], t["main"]))
    print(len(mains), "main changes at tick ends")
    for m in mains[-12:]:
        print(m)
