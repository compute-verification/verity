"""batch_summary.py BATCH RUN RUN_DIR: one bound.sh run's kinds as scratch/cir289/batch-results/BATCH.json, and its BATCH line.

A kind is accepted when B.verify.held exits 0 with accepted true and B.verify.no-program and B.verify.other-def both
exit 1 with accepted false. Anything else is a stop, named by the last step it reached."""
import glob
import json
import os
import re
import sys
from pathlib import Path

batch, run, rd = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out = rd / "out"
listed = [l.split("\t") for l in Path(f"scratch/cir272/types-cir289-{batch}.tsv").read_text().splitlines() if l.strip()]


def rss_gb(f):
    try:
        t = Path(f).read_text()
    except FileNotFoundError:
        return None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    if m:
        return int(m.group(1)) / 2**20
    try:
        return json.loads(t.splitlines()[-1])["maxrss_kb"] / 2**20
    except Exception:  # noqa: BLE001
        return None


def verdict(o, name):
    p = o / "lean" / f"{name}.out"
    rc = o / "lean" / f"{name}.rc"
    if not rc.exists():
        return None
    acc = None
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith("VERDICT "):
            acc = json.loads(line[len("VERDICT "):]).get("accepted")
    return {"exit": int(rc.read_text().strip()), "accepted": acc}


not_reached = {l.split("\t")[0] for l in (out / "not-reached.tsv").read_text().splitlines() if l.strip()} \
    if (out / "not-reached.tsv").exists() else set()
kinds = []
for tag, unit, *_ in listed:
    o = out / tag
    rec = {"tag": tag, "kind": unit.removeprefix("census:")}
    if tag in not_reached or not o.exists():
        rec["outcome"] = "not-reached"
        kinds.append(rec)
        continue
    done = json.loads((o / "done").read_text()) if (o / "done").exists() else None
    st = json.loads((o / "B" / "stage.json").read_text()) if (o / "B" / "stage.json").exists() else {}
    v = {n: verdict(o, n) for n in ("B.verify.held", "B.verify.no-program", "B.verify.other-def")}
    peaks = {"stage": rss_gb(o / "B" / "stage.time"), "prove": rss_gb(o / "B" / "prove.time"),
             "serve": rss_gb(o / "B" / "serve.time"),
             "lean": max([g for g in (rss_gb(f) for f in glob.glob(str(o / "lean" / "*.time"))) if g] or [0]) or None}
    rec.update(staged=st.get("word") or st.get("definition"), ands=st.get("ands"), k_log=st.get("k_log"),
               circuit_bytes=st.get("circuit_bytes"), seconds=(done or {}).get("seconds"), verdicts=v,
               peak_gb={k: round(x, 2) for k, x in peaks.items() if x})
    ok = (v["B.verify.held"] == {"exit": 0, "accepted": True}
          and v["B.verify.no-program"] == {"exit": 1, "accepted": False}
          and v["B.verify.other-def"] == {"exit": 1, "accepted": False})
    if ok:
        rec["outcome"] = "accepted"
    elif (o / "timeout").exists():
        rec["outcome"] = "stopped-timeout"
    elif (out / tag / "lean-not-run").exists():
        rec["outcome"] = f"stopped-lean-gate-{(out / tag / 'lean-not-run').read_text().strip()}GB"
    elif not st:
        err = (o / "B" / "stage.err")
        rec["outcome"] = "stopped-stage"
        rec["why"] = err.read_text().strip().splitlines()[-1][:300] if err.exists() and err.read_text().strip() else None
    elif v["B.verify.held"] is None:
        rec["outcome"] = "stopped-session-or-statement"
    else:
        rec["outcome"] = "not-accepted"
    kinds.append(rec)

acc = sum(k["outcome"] == "accepted" for k in kinds)
peak = max((max(k.get("peak_gb", {}).values() or [0]) for k in kinds), default=0)
log = (out / "job.log").read_text().splitlines()
ts = [re.match(r"(\d\d):(\d\d):(\d\d)Z", l) for l in log]
ts = [int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) for m in ts if m]
span = (ts[-1] - ts[0]) % 86400 if ts else None
res = {"batch": batch, "run": run, "kinds": len(kinds), "accepted": acc,
       "stopped": [{"kind": k["kind"], "outcome": k["outcome"], "why": k.get("why")} for k in kinds if k["outcome"] != "accepted"],
       "hours": round(span / 3600, 2) if span is not None else None, "peak_gb": round(peak, 2), "records": kinds}
Path("scratch/cir289/batch-results").mkdir(parents=True, exist_ok=True)
Path(f"scratch/cir289/batch-results/{batch}.json").write_text(json.dumps(res, indent=1) + "\n")
print(f"BATCH {batch} done: accepted {acc} of {len(kinds)}")
print(json.dumps({k: res[k] for k in ("run", "hours", "peak_gb", "stopped")})[:1500])
