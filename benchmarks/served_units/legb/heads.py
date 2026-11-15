"""heads.py FLOCK_WORK: each staged statement's circuit head as the form mirror reads it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

print("dir\tformat\tout_net\tunits_per_vu\tout_leaves\tunit_head\tcircuit_bytes\tstatement")
for d in sorted(Path(sys.argv[1], "data").iterdir()):
    c = d / "circuit.txt"
    if not (d / ".done").exists() or not c.exists():
        continue
    with open(c, "rb") as f:
        first = f.readline().decode().strip()[:80]
        meta = f.readline().decode()
    m = json.loads(meta[len("META "):]) if meta.startswith("META ") else {}
    h = PF._unit_head(c)
    lv, upv = m.get("leaves_out", []), m.get("units_per_vu")
    leaves = sum(len(x) for x in lv if isinstance(x, list)) if isinstance(lv, list) else "-"
    print("\t".join(map(str, [d.name, first, m.get("out_net", "unit(default)"), upv, leaves,
                              " ".join(h[1][:6]) if h else "no unit head (templated or other form)", c.stat().st_size,
                              m.get("statement", "-")])), flush=True)
