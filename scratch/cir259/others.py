"""others.py B/stage.json T UNIT: other Programs and partition objects than the bound statement's, written by stage.py's own
`binding`, those $CB_OTHERS names (default all three): T/other-def (another Definition, the same count), T/other-count (the
same Definition, one more input) and T/larger (the same Definition at twice the count, a population the statement's proof
blocks are a part of)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

SRC = Path.cwd()
spec = importlib.util.spec_from_file_location("su_stage", SRC / "benchmarks/served_units/stage.py")
stage = importlib.util.module_from_spec(spec)
sys.modules["su_stage"] = stage
spec.loader.exec_module(stage)

from verity.circuits.codec import descriptor_id
from verity_vllm.pipeline.serving_rows import template

rec, T, unit = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
n, name = rec["n"], rec["definition"]
_kind, u, S = stage.parse(unit)
if isinstance(u, stage.Census):
    d = template(u.name(S)).fn
elif isinstance(u, stage.Read):
    from verity.commitments.gates import embedding as E

    d = E.read(S["V"], S["H"])
else:
    d = u.definition(S)
assert descriptor_id(d) == name or d.id == name, (descriptor_id(d), d.id, name)
other = template("RoPEHeadGptJ_v2{D=64}" if "SiluMul" in name else "SiluMul_v4{I=256}").fn
want = os.environ.get("CB_OTHERS", "other-def other-count larger").split()
out = {}
for tag, (fn, k) in {"other-def": (other, n), "other-count": (d, n + 1), "larger": (d, 2 * n)}.items():
    if tag not in want:
        continue
    digests, r = stage.binding(fn, k, T / tag)
    out[tag] = {"definition": descriptor_id(fn), "population": k, "partition": digests["partition"][:16], "program": digests["program"][:16],
                "program_bytes": r["program_bytes"]}
print(json.dumps(out))
