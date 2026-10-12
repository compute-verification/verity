"""The difftest of `verity.ml.crypto.lms`'s key derivation and signing against the Lean package `lms`: the Python half.

`verity/core/primitives/lms/lean/scripts/sign_vectors.py` writes 80 instances (keys from seeds at N = 64 and
at `TOY_N`, the leaves and messages to sign at) and this module's results on each (`Params`, `public_key` of `keygen`'s
or `key`'s key, each `sign_at`'s signature and `verify`'s verdict on it). `check` runs the Lean half in its Lean audit
(`runs` in that package's `lean-audit.json`): `Lms/SignDifftestMain.lean` computes every result with `Lms.Sign` and fails
on any byte that differs. It is the only check of the Lean signer, which the network certifier device's issuer signs
with; the audit runs it interpreted, about a minute of the audit's time. The audit checks the data against the record's
SHA-256, and its pass is cached on the Lean package's files alone, so the test here is what says the data is still what
`lms.py` computes: a change to the scheme changes the data, fails this test, and `check.py --update` records the new data,
which makes `check` run the Lean half on it. The other test checks that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

LEAN = Path(__file__).resolve().parents[3] / "core" / "primitives" / "lms" / "lean"
SCRIPT = LEAN / "scripts" / "sign_vectors.py"
_spec = importlib.util.spec_from_file_location("lms_sign_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["Lms.SignDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 80 instances and `lms.py`'s results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{sign/instances.jsonl}", "{sign/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    insts = [vectors.instance(s) for s in range(vectors.COUNT)]
    results = [vectors.python_results(i) for i in insts]
    admitted = [(i, r) for i, r in zip(insts, results, strict=True) if r["params"]]
    assert len(admitted) < len(insts)
    assert {(i["params"][0], i["params"][1]) for i, _ in admitted} == {(64, 1), (64, 2), (64, 4), (8, 1), (8, 2), (8, 4),
                                                                        (8, 8)}
    assert {i["params"][2] for i, _ in admitted} == {1, 2, 3, 4}
    made = {(i["make"], r["public"] is None) for i, r in admitted}
    assert made == {("keygen", False), ("keygen", True), ("key", False), ("key", True)}
    signatures = [s for _, r in admitted for s in r["signatures"] or []]
    assert None in signatures and sum(s is not None for s in signatures) > 200
    assert all(s[1] for s in signatures if s is not None)
    assert any(len(i["signs"]) >= 2 ** i["params"][2] for i, r in admitted if r["public"])  # every leaf of a key
