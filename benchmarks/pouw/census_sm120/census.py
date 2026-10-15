"""Collect the census sections' JSON lines into census.json (stdlib only; analysis to follow)."""

import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
raw = {}
for f in sorted(out.glob("*.jsonl")):
    rows = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    raw[f.stem] = rows
(out / "census.json").write_text(json.dumps({"raw": raw}, indent=1))
