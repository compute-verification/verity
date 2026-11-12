"""cover_accepted.py RUNS_DIR OUT.json RUN...: each staged type of CIR-259's jobs (`bound.sh`'s job.log, after `research fetch RUN
--all`) as what the Lean verifier said of it: its STAGED record (the Definition staged, the word it computes, ANDs, k_log) and
its VERDICTS lines, read into one verdict:

  accepted   `B.verify.held` accepted a session (the statement handed its partition object and Program)
  refused    `B.ks-form` refused the statement's form
  no-verdict staged, with no `B.verify.held` line
  stopped    a stager the job stopped (no STAGED line; the job's `stopped:` line)
"""
import json
import re
import sys
from pathlib import Path

runs_dir, out, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = []
for run in runs:
    log = runs_dir / run / "out" / "job.log"
    if not log.exists():
        rows.append({"run": run, "error": f"no {log}"})
        continue
    staged, verdicts, stopped, named = {}, {}, {}, {}
    for line in log.read_text(errors="replace").splitlines():
        if m := re.match(r"^\S+Z ([^\s:/]+): (\S+) N=(\d+)", line):
            named[m[1]] = m[2]
        if line.startswith("STAGED "):
            tag, js = line[len("STAGED "):].split(" ", 1)
            staged[tag] = json.loads(js)
        elif line.startswith("VERDICTS "):
            tag, rest = line[len("VERDICTS "):].split(" ", 1)
            arm, _, said = rest.partition(": ")
            verdicts.setdefault(tag, {})[arm] = said
        elif m := re.match(r"^\S+Z (\S+): stage\.py (\S+) (\S+)(?: --bind)?: rc (\d+);.*?(stopped: .*)$", line):
            stopped[m[1]] = {"unit": m[2], "n": m[3], "rc": int(m[4]), "why": m[5][:300]}
    for tag in sorted(set(staged) | set(verdicts) | set(stopped)):
        st, v = staged.get(tag), verdicts.get(tag, {})
        if st is None:
            verdict = "stopped" if tag in stopped else "no-verdict"
        elif "KS-FORM refused" in v.get("B.ks-form", ""):
            verdict = "refused"
        elif "accepted=True" in v.get("B.verify.held", ""):
            verdict = "accepted"
        else:
            verdict = "no-verdict"
        rows.append({"run": run, "tag": tag, "verdict": verdict, "unit": named.get(tag),
                     "definition": (st or {}).get("definition"), "word": (st or {}).get("word"),
                     "census": (st or {}).get("census"), "n": (st or {}).get("n"), "ands": (st or {}).get("ands"),
                     "k_log": (st or {}).get("k_log"), "stopped": stopped.get(tag),
                     "verdicts": {k: v[k][:240] for k in sorted(v)}})
out.write_text(json.dumps(rows, indent=1) + "\n")
for r in rows:
    print(r["run"][-4:], r.get("tag"), r.get("verdict"), r.get("word") or (r.get("stopped") or {}).get("unit"), "|", r.get("definition"))
