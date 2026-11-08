"""apply_row.py ROW NEUTRAL_RUN [AOT_RUN ART]: put a re-pin into this tree's manifests/records.json (and re-switch a heavy one).

The row's entry is the one `neutrality ROW --tier T --update` wrote in NEUTRAL_RUN's clone (its out/records.json). For a heavy
row, its `pinned_from.build` becomes the traced Build's art: id (the precedent's), then `circuit_db.switch` with AOT_RUN's
comparison.json, which refuses unless the comparison is equal on every rank and the pins are its Programs. A fast row (no AOT_RUN)
goes in as neutrality wrote it. Rows this tree already re-pinned at NEUTRAL_RUN's head may differ. Run with
PYTHONPATH=integrations/vllm:. from the tree's root."""
import json
import sys
from pathlib import Path

from verity_vllm.pipeline import circuit_db as D
from verity_vllm.pipeline import config_record as CR

row, nrun = sys.argv[1:3]
arun, art = (sys.argv[3:5] + [None, None])[:2]
runs = Path.home() / ".research/runs"
theirs_doc = json.loads((runs / nrun / "out/records.json").read_text())
theirs = theirs_doc["rows"]
head = theirs[row]["pinned_from"]["tree"]
doc = CR.load_index(D.INDEX)
ours = doc["rows"]
moved = sorted(k for k in set(theirs) | set(ours) if k != row and theirs.get(k) != ours.get(k)
               and (ours.get(k) or {}).get("pinned_from", {}).get("tree") != head)
if moved:
    sys.exit(f"{nrun}'s records.json differs from this tree's beyond {row} and the rows re-pinned at {head}: {moved}")
new = theirs[row]
if new["pinned_from"]["run"] != nrun or new.get("tier") != ours[row].get("tier") or new.get("record") != ours[row].get("record"):
    sys.exit(f"{row}: {nrun}'s entry is not a re-pin of this tree's row: {({k: v for k, v in new.items() if k != 'programs'})}")
if "aot" in new:
    sys.exit(f"{row}: {nrun}'s entry is still switched; neutrality --update drops aot when the pins move")
if (arun is None) != ("aot" not in ours[row]):
    sys.exit(f"{row}: a switched row needs AOT_RUN and ART, an unswitched one neither")
old = ours[row]["programs"]
doc["rows"][row] = {**new, "pinned_from": {**new["pinned_from"], "build": art}} if art else new
CR.write_index(doc, D.INDEX)
if arun:
    comparison = json.loads((runs / arun / "out/comparison.json").read_text())
    D.switch(row, comparison, run=arun, trace=art)
ranks = sorted(new["programs"], key=int)
parts = {r: sorted(p for p in ("step", "workload") if new["programs"][r].get(p) != old.get(r, {}).get(p))
         + (["requests"] if new["programs"][r].get("requests") != old.get(r, {}).get("requests") else []) for r in ranks}
print(json.dumps({"row": row, "moved": parts}, indent=1))
