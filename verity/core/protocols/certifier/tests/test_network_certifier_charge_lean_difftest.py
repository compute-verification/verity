"""The difftest of `arith_encode.reference_charge` against the certifier's Lean program: the Python half.

`lean/scripts/charge_vectors.py` writes 4,000 random instances, a book, a carry and windows of slots that ask or not
with their decisions, and `reference_charge`'s results on each: window by window from the carry, and chained from the
book's priors. `check` runs the Lean half in its Lean audit (`runs` in `lean/lean-audit.json`):
`lean/NetworkCertifier/ChargeDifftestMain.lean` evaluates every instance with the program's `AdviceV1.window` and
`AdviceV1.charges` and fails on any difference, and `security_proofs` proves `charges` equal to `NetTiming.AdviceV1`'s
(`CodeChargesV1`). circuit-check holds `ChargeCount_v1{N}`'s circuit to `reference_charge`, so this is what holds the
circuit's charge to the Lean definition's. The audit checks that data against the record's SHA-256, and its pass is
cached on the Lean package's files alone, so the test here is what says the data is still what this package computes:
a change to `reference_charge` changes the data, fails this test, and `check.py --update` records the new data, which
makes `check` run the Lean half on it. The other test checks that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from verity.core.protocols.certifier import arith_encode

LEAN = Path(__file__).resolve().parents[1] / "lean"
SCRIPT = LEAN / "scripts" / "charge_vectors.py"
_spec = importlib.util.spec_from_file_location("charge_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCertifier.ChargeDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 4,000 instances and `reference_charge`'s results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{charge/instances.jsonl}", "{charge/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says(monkeypatch):
    scalings = Counter()
    scale = arith_encode._scale

    def counted(low, high):
        out = scale(low, high)
        scalings[out[2]] += 1
        return out

    monkeypatch.setattr(arith_encode, "_scale", counted)
    seen = Counter()
    for seed in range(vectors.COUNT):
        inst = vectors.instance(seed)
        res = vectors.python_results(inst)
        assert len(res["steps"]) <= len(inst["windows"])
        assert all(s is not None for s in res["steps"][:-1])
        seen["from priors" if inst["carry"] == arith_encode.Carry(inst["priors"], 0) else "from a carry"] += 1
        seen["a refused window" if None in res["steps"] else "every window accepted"] += 1
        seen["refused" if res["charges"] is None else "charged"] += 1
        seen["a session of 3 or more charged"] += res["charges"] is not None and len(res["charges"]) >= 3
        seen["no window"] += not inst["windows"]
        seen["a window with no asks"] += any(not any(en) for en, _ in inst["windows"])
        seen["run 3"] += inst["carry"].run == 3
        seen["a total past MAX_TOTAL"] += inst["book"].total > arith_encode.MAX_TOTAL
        seen["a window charged 0"] += any(s is not None and s[0] == 0 for s in res["steps"])
        seen["counts at 2^KW - 1"] += any(sum(c) == (1 << vectors.KW) - 1 for c in inst["carry"].counts)
    for case in ("from priors", "from a carry", "a refused window", "every window accepted", "refused", "charged"):
        assert seen[case] > 1000, (case, seen)
    for case in ("a session of 3 or more charged", "no window", "a window with no asks", "run 3",
                 "a total past MAX_TOTAL", "a window charged 0", "counts at 2^KW - 1"):
        assert seen[case] > 20, (case, seen)
    assert max(scalings) == 17 and scalings[17] > 100, scalings
