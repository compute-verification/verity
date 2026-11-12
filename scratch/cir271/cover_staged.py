"""cover_staged.py RUNS_DIR OUT.json RUN...: every staging step of CIR-259's `bound.sh` runs, from each run's out/job.log.

A step is a ``TAG: stage.py NAME N ...: rc RC; ... Maximum resident set size (kbytes): K`` line; a completed one is followed by
``STAGED TAG {json}``, which names its Definition and word. One record per step: run, tag, the name staged, units, rc, peak GiB,
and the Definition and word when it completed.
"""
import json
import re
import sys
from pathlib import Path

RUNS, OUT, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
STEP = re.compile(r"^\S+ (\S+): stage\.py (\S+) (\d+)\b.*?: rc (\S+);.*?Maximum resident set size \(kbytes\): (\d+)")
out = []
for run in runs:
    log = RUNS / run / "out" / "job.log"
    if not log.exists():
        continue
    open_steps = {}
    for line in log.read_text().split("\n"):
        m = STEP.match(line)
        if m:
            tag, name, n, rc, kb = m.groups()
            rec = {"run": run, "tag": tag, "name": name, "n": int(n), "rc": int(rc) if rc.isdigit() else rc,
                   "peak_gib": round(int(kb) / 2**20, 2), "definition": None, "word": None}
            out.append(rec)
            open_steps[tag] = rec
        elif line.startswith("STAGED "):
            tag, _, js = line[len("STAGED "):].partition(" ")
            try:
                d = json.loads(js)
            except json.JSONDecodeError:
                continue
            if tag in open_steps:
                open_steps[tag].update(definition=d.get("definition"), word=d.get("word"))
OUT.write_text(json.dumps(out, indent=1) + "\n")
for r in out:
    print(r["run"][-4:], r["tag"], r["rc"], r["peak_gib"], r["n"], (r["word"] or r["name"])[:90])
