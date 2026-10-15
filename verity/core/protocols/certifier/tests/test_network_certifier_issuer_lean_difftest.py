"""The difftest of `certificate.Issuer` against the Lean package `certifier_device`'s `Issuer.certify`: the Python half.

`verity/core/devices/certifier/lean/scripts/issuer_vectors.py` writes 160 LMS keys with calls to one issuer
on each in a certifier session (its id 32 bytes, or 31 or 33, which the issuer refuses), and what this package's real `certificate.Issuer` does with them, its signer at the leaf the host reserves for
each call (the parameters digest, the certified record or none, `verify` under the key and after a tamper, and the
issuer's `refused`). `check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`, about a
minute interpreted, since every key and signature is hashed by the Lean side's SHA-512):
`CertifierDevice/IssuerDifftestMain.lean` threads one `Issuer` through each instance's calls and fails on any result
that differs. The audit checks the data against the record's SHA-256, and its pass is cached on the Lean package's files
alone, so the test here is what says the data is still what this package does: a change to the issuer, the record head,
the certificate or LMS signing changes the data, fails this test, and `check.py --update` records the new data. The
other test checks that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

LEAN = Path(__file__).resolve().parents[3] / "devices" / "certifier" / "lean"
SCRIPT = LEAN / "scripts" / "issuer_vectors.py"
_spec = importlib.util.spec_from_file_location("certifier_issuer_vectors", SCRIPT)
vectors = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["CertifierDevice.IssuerDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 160 instances and the issuer's results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{issuer/instances.jsonl}", "{issuer/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    insts = [vectors.instance(s) for s in range(vectors.COUNT)]
    results = [vectors.python_results(i) for i in insts]
    keys = Counter(tuple(i["key"]["params"]) for i in insts)
    assert set(keys) == {k for k, _ in vectors.KEYS}
    calls = Counter()
    malformed = Counter()
    tampers = Counter()
    after_refusal = 0
    sessions = Counter()
    for inst, res in zip(insts, results, strict=True):
        session = len(bytes.fromhex(inst["session"]))
        sessions[session] += 1
        refused = [[c["record"]["link"], c["direction"], c["record"]["window"]] for c, r in zip(inst["calls"], res["calls"])
                   if r["certify"] is not None and r["certify"]["certified"] is None]
        assert res["refused"] == refused
        seen_refusal = False
        for c, r in zip(inst["calls"], res["calls"], strict=True):
            if c["malformed"] is not None:
                malformed[c["malformed"], r["certify"] is None] += 1
            if r["certify"] is None:
                calls["record refused"] += 1
                assert c["malformed"] is not None or session != 32
                continue
            assert session == 32
            certified = r["certify"]["certified"]
            if certified is None:
                calls["leaf refused"] += 1
                assert c["leaf"] is None
                seen_refusal = True
                continue
            calls["certified", c["direction"], c["sync_set"] is not None, c["deadline"]] += 1
            after_refusal += seen_refusal
            assert certified["verify"]
            if c["tamper"] is not None:
                tampers[c["tamper"][0]] += 1
                assert certified["tampered"] is False, c["tamper"]
    assert set(tampers) == set(vectors.TAMPERS)
    assert {k for k, _ in malformed} == set(vectors.MALFORMED)
    for kind in vectors.MALFORMED:
        assert malformed[kind, True] > malformed[kind, False], kind
    assert calls["record refused"] > 20 and calls["leaf refused"] > 20 and after_refusal > 5
    for direction in ["egress", "ingress"]:
        for deadline in [True, False]:
            assert calls["certified", direction, False, deadline] > 0
    assert calls["certified", "egress", True, True] > 0
    assert set(sessions) == {31, 32, 33} and sessions[32] > 100
