"""The P2 v4 setup difftest against `PousCheck` (`verity/experimental/core/protocols/pous`): the Python half.

`scripts/setup_difftest.py` there writes generated instances and this package's answers on each (P2's maps, its v4 keys,
the commitment to `W`, `decode_segment` and the public-encoder `verifier_key`), and copies the spec's §7 vectors. The
Lean audit runs the Lean half on them (`runs` in that package's `lean-audit.json`, which pins their SHA-256), and caches
its pass on the Lean package's files alone, so this test is what says the data is still what this package computes: a
change to an answer changes the data, fails this test, and `check.py --update` records the new data.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "protocols" / "pous"
SCRIPT = LEAN / "scripts" / "setup_difftest.py"
_spec = importlib.util.spec_from_file_location("setup_difftest", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["PousCheck.SetupDifftestMain"]


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins_and_covers_what_it_says(tmp_path):
    """Every file byte for byte as the record pins it (about 14 s: four full-width encodes), and the instances reach each
    kind of answer the generator's docstring lists."""
    paths = vectors.write(tmp_path)
    assert {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()} == RUN["data"]
    insts = [json.loads(x) for x in paths["difftest/instances.jsonl"].read_text().splitlines()]
    answers = [json.loads(x) for x in paths["difftest/python.jsonl"].read_text().splitlines()]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", "{difftest/p2-vectors.json}",
                           str(len(insts))]
    seen = collections.Counter((i["kind"], a.get("error", a.get("verdict", "ok"))) for i, a in zip(insts, answers))
    assert {("setup", k) for k in ("ok", "segments", "shape", "salt", "noncanonical", "payload", "commitment")} <= set(seen)
    assert {("decode", k) for k in ("ok", "segments", "shape", "salt", "noncanonical", "payload", "width")} <= set(seen)
    assert {("maps", "ok"), ("keys", "ok"), ("keys", "salt"), ("commit", "ok"), ("tagged", "ok")} <= set(seen)
    assert sum(1 for i, a in zip(insts, answers) if i["kind"] == "keys" and a.get("t_index", 0) > 0) >= 3
    assert {i["params"]["w"] for i in insts if i["kind"] == "setup"} >= {64, 1024, 16448}
    rejected = [i for i in insts if i["kind"] == "maps" and int(i["r"], 16) ^ int(i["k"], 16) >= (1 << i["w"]) - i["c0"]]
    assert len(rejected) >= 20
