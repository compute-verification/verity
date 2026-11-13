"""form.py TYPES.tsv NAMES.jsonl FORM.tsv: the list as bound.sh runs it, on stdout. A census row whose lookup (names.py's
line for its tag) gives a ``zk_refusal`` is staged as ``census:`` + the lookup's ``boolean_version``, the form the refusal
names; every other row as listed. FORM.tsv: TAG, the listed unit, the staged unit, and the lookup's refusal or ``-``."""
import json
import sys
from pathlib import Path

names = {n["tag"]: n for n in map(json.loads, Path(sys.argv[2]).read_text().splitlines()) if n.get("tag")}
form = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    tag, unit, *rest = line.split("\t")
    census = names[tag].get("census") or {}
    staged, why = unit, "-"
    if census.get("zk_refusal"):
        if not census.get("boolean_version"):
            raise SystemExit(f"form.py: {tag}'s lookup refuses its word form but names no Boolean version: {census}")
        staged, why = f"census:{census['boolean_version']}", census["zk_refusal"].replace("\t", " ")
    print("\t".join([tag, staged, *rest]))
    form.append(f"{tag}\t{unit}\t{staged}\t{why}\n")
Path(sys.argv[3]).write_text("".join(form))
