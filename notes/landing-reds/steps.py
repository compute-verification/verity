"""Fetch each attempt's result record (steps, reuse) by id; cache in /tmp/reds/results/."""
import json, os, sqlite3, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

OUT = "/tmp/reds/results"
os.makedirs(OUT, exist_ok=True)
db = sqlite3.connect(os.path.expanduser("~/.research/store/catalog.sqlite"), check_same_thread=False)


def result_art(rid):
    row = db.execute("select manifest_json from attempts where id=?", (rid,)).fetchone()
    if not row:
        return None
    return (json.loads(row[0]).get("outputs") or {}).get("result")


def fetch(rid):
    p = f"{OUT}/{rid}.json"
    if os.path.exists(p):
        return rid, "cached"
    art = result_art(rid)
    if not art:
        return rid, "no result"
    r = subprocess.run(["research", "data", "show", art, "--json"], capture_output=True, text=True, timeout=300)
    if r.returncode:
        return rid, f"rc={r.returncode} {r.stderr[-200:]}"
    meta = json.loads(r.stdout)["manifest"]["meta"]
    json.dump({k: meta.get(k) for k in ("steps", "reuse", "validation", "verdicts", "deferred")}, open(p, "w"))
    return rid, "ok"


if __name__ == "__main__":
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with ThreadPoolExecutor(16) as pool:
        for rid, st in pool.map(fetch, ids):
            if st not in ("ok", "cached"):
                print(rid, st)
    print("done", len(ids))
