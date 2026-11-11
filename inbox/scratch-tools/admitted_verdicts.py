"""For each admitted PR, and each one `status` refuses for an old `next` that is marked by label (CI-501), in `research queue status` (stdin or /tmp/qstatus.txt), print its `reviewed` and `ready` labels at head
(remote), newest first.  Run from /tmp/rchk with PYTHONPATH=/tmp/rchk/tools/research/src RESEARCH_AT_CHECKOUT=1."""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

rows = [m.groups() for line in open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qstatus.txt")
        if ("admitted" in line or "old `next`" in line)
        and (m := re.search(r"#(\d+)\s+([0-9a-f]{8})\s+(?:ready|refused)\s+(\S+)", line))]
heads = {}
for pr, short, owner in rows:
    r = subprocess.run(["git", "-C", "/workspace", "ls-remote", "origin", f"refs/pull/{pr}/head"], capture_output=True, text=True)
    heads[pr] = r.stdout.split()[0] if r.stdout.split() else short

def one(row):
    pr, short, owner = row
    out = subprocess.run([sys.executable, "-m", "research", "data", "labels", f"pr:{pr}@{heads[pr]}", "--remote", "--json", "--history"],
                         capture_output=True, text=True)
    try:
        labs = json.loads(out.stdout)
    except Exception:
        return f"#{pr} {short} {owner}: labels unreadable: {out.stderr.strip()[-120:]}"
    labs = [l.get("label", l) for l in (labs if isinstance(labs, list) else labs.get("labels", []))]
    rev = sorted((l for l in labs if l.get("key") == "reviewed"), key=lambda l: l.get("ts") or "")
    rdy = sorted((l for l in labs if l.get("key") == "ready"), key=lambda l: l.get("ts") or "")
    last = rev[-1] if rev else None
    r0 = rdy[0] if rdy else {}
    if not rdy:
        return None     # a head `status` refuses for an old `next` (CI-501) and nobody marked by label
    return (f"#{pr} {heads[pr][:8]} {owner:20s} ready {str(r0.get('ts'))[:16]}  "
            + (f"reviewed {last.get('value')} by {last.get('by')} {str(last.get('ts'))[:16]}" if last else "NO VERDICT"))

with ThreadPoolExecutor(8) as ex:
    for line in ex.map(one, rows):
        if line:
            print(line)
