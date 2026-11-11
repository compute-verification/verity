"""Scratch, never for landing: the benchmark's report (`report.main`, as `run.py` calls it) with the class circuit
lowered once, the Program the count is given kept, and the built class's whole Program kept beside it.
From the repository's root:  uv run python benchmarks/inference_only/own_probe.py DIR"""
import json
import sys
from pathlib import Path

import report

from verity.circuits import codec
from verity_vllm.program.predict.window_circuit import class_circuit

B, P = report.B, report.P
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
stated = P.stated(P.SERVED)
circuit = class_circuit(B.registered(B.MODEL), P.class_of(stated.cls))


def check(inst, *a, **k):
    (out / "count-instance.json").write_text(json.dumps({"program": inst["program"]}))
    return B.lean_count(inst, *a, **k)


report.main(["--out", str(out)], check=check, circuit=circuit)
program, _ = circuit.program()
text = codec.canonical_json(codec.encode_program(program)).decode()
(out / "class-program.json").write_text(json.dumps({"program": text}))
print(f"[own_probe] the built class {circuit.name}: its whole Program is {len(text):,} bytes; "
      f"{len(circuit.templates)} templates, {sum(circuit.units.values()):,} units, {len(circuit.wires):,} wires",
      file=sys.stderr)
