"""verdicts.py OUT/TAG: each Lean command's exit code and what it said, one line each and OUT/TAG/verdicts.json."""
import json
import re
import sys
from pathlib import Path

o = Path(sys.argv[1])
rows = {}
for rc in sorted((o / "lean").glob("*.rc")):
    name = rc.name[:-3]
    out = (o / "lean" / f"{name}.out").read_text(errors="replace")
    err = (o / "lean" / f"{name}.err").read_text(errors="replace")
    time = (o / "lean" / f"{name}.time").read_text(errors="replace")
    row: dict = {"exit": int(rc.read_text())}
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time)
    e = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", time)
    row["maxrss_kb"], row["wall"] = (int(m[1]) if m else None), (e[1] if e else None)
    for line in out.splitlines():
        if line.startswith("VERDICT "):
            v = json.loads(line[8:])
            row["verdict"] = {"accepted": v.get("accepted"), "why": v.get("why"), "partition": v.get("partition")}
        elif line.startswith("KS-FORM"):
            row["ks_form"] = line[8:]
        elif line.startswith("setup:"):
            row["setup"] = line[:600]
        elif line.startswith("{"):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            row["statement"] = {k: j.get(k) for k in ("statement", "circuit_sha512", "class_sha512", "program_sha512", "partition",
                                                      "partition_check", "m", "regions")}
    if len(row) == 3:
        row["said"] = (out.strip().splitlines() or err.strip().splitlines() or [""])[-1][:600]
    rows[name] = row
(o / "verdicts.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
for name, row in rows.items():
    s = row.get("statement") or {}
    v = row.get("verdict") or {}
    pc = s.get("partition_check") or v.get("partition") or {}
    say = (f"accepted={v['accepted']} why={v.get('why')!r}" if v else row.get("setup") or (f"KS-FORM {row['ks_form']}" if "ks_form" in row else None)
           or ("statement ok" if s else row.get("said")))
    print(f"VERDICTS {o.name} {name}: exit {row['exit']}; {say}; units={pc.get('units') if isinstance(pc, dict) else None}; "
          f"note={(pc.get('note') if isinstance(pc, dict) else '')!s:.260}")
