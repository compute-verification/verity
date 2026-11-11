"""Scratch, never for landing: the Program of the small served phase's circuit the vllm suite builds
(tests/program/predict/test_window_circuit.py), written for the own-rows probe.
From adapters/vllm:  uv run python own_small.py DIR"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from tests.program.predict import test_window_circuit as T  # noqa: E402

from verity.circuits import codec  # noqa: E402

c = T._circuit(T.WINDOW)
program, _ = c.program()
text = codec.canonical_json(codec.encode_program(program)).decode()
Path(sys.argv[1], "small-program.json").write_text(json.dumps({"program": text}))
print(f"[own_small] {c.name}: its Program is {len(text):,} bytes; {len(c.templates)} templates, "
      f"{sum(n for _, n in c.templates):,} units, {len(c.wires)} wires")
