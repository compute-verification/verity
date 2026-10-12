"""The difftest of `verity.ml.crypto.lms`'s verifier against the Lean package `lms`: the Python half.

`verity/core/primitives/lms/lean/scripts/difftest_vectors.py` writes 600 instances (keys at every parameter
set `Params` admits, their signatures and tampered ones) and this module's results on each (`Params`, `Signature.decode`,
`digits`, `root_of`, `verify`). `check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`):
`Lms/DifftestMain.lean` computes every result with `Lms.Verify` and fails on any difference. The audit checks the data
against the record's SHA-256, and its pass is cached on the Lean package's files alone, so the test here is what says the
data is still what `lms.py` computes: a change to the scheme changes the data, fails this test, and `check.py --update`
records the new data, which makes `check` run the Lean half on it. The other test checks that the generator reaches what
its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from verity.ml.crypto import lms

LEAN = Path(__file__).resolve().parents[3] / "core" / "primitives" / "lms" / "lean"
SCRIPT = LEAN / "scripts" / "difftest_vectors.py"
_spec = importlib.util.spec_from_file_location("lms_difftest_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["Lms.DifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 600 instances and `lms.py`'s results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    insts = [vectors.instance(s) for s in range(vectors.COUNT)]
    results = [vectors.python_results(i) for i in insts]
    accepted = Counter((n, w, toy) for i, r in zip(insts, results, strict=True) if r["verify"]
                       for n, w, h, toy in [i["params"]])
    assert set(accepted) == {s for s, _ in vectors.SETS}
    d = lms.DEPLOYED
    deployed = [i for i, r in zip(insts, results, strict=True) if r["verify"] and i["params"] == [d.N, d.W, d.H, False]]
    assert len(deployed) >= 5
    heights = {i["params"][2] for i, r in zip(insts, results, strict=True) if r["verify"]}
    assert heights == {1, 2, 3, 5, 11, 20, 30}
    verdicts = Counter((i["tamper"], r["verify"]) for i, r in zip(insts, results, strict=True))
    assert {t for t, v in verdicts if v is True} == {"none"} and verdicts[("none", False)] == 0
    rejected = {"C", "y", "path", "message", "q", "q-past", "other-key", "other-message", "pk-bit"}
    assert rejected == {t for t, v in verdicts if v is False}
    assert {t for t, v in verdicts if v is None} == {"short", "long", "pk-length", "params"}
    assert any(r["root"] is not None and r["verify"] is None for r in results)  # root_of reads I from a short key
    assert any(r["decode"] and r["root"] is None for r in results)  # q past 2^H
    assert {len(bytes.fromhex(i["message"])) for i in insts} >= {0, 1, 64, 100}
