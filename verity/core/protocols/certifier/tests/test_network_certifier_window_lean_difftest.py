"""The difftest of `window_charge.reference_window` against the certifier's Lean program: the Python half.

`lean/scripts/window_vectors.py` writes 4,000 random instances, a book, a carry and windows of decision, hit and order
slots, and `reference_window`'s results on each: window by window from the carry, and chained from the book's first
carry. `check` runs the Lean half in its Lean audit (`runs` in `lean/lean-audit.json`):
`lean/NetworkCertifier/WindowDifftestMain.lean` evaluates every instance with the program's `WindowCharge.window` and
`WindowCharge.charges` and fails on any difference, and `security_proofs` proves `charges` equal to
`NetTiming.WindowCharge`'s (`CodeChargesWhole`), which `EncardChargedLeWhole` is proved of. circuit-check holds
`ChargeCount_v2{N, H, O}`'s circuit to `reference_window`, so this is what holds the circuit's charge to the one the
Lean proofs count. The audit checks that data against the record's SHA-256, and its pass is cached on the Lean package's
files alone, so the test here is what says the data is still what this package computes: a change to `reference_window`
changes the data, fails this test, and `check.py --update` records the new data, which makes `check` run the Lean half
on it. The other test checks that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from verity.core.protocols.certifier import arith_encode
from verity.ml.protocols.certifier import window_charge as WC

LEAN = Path(__file__).resolve().parents[1] / "lean"
SCRIPT = LEAN / "scripts" / "window_vectors.py"
_spec = importlib.util.spec_from_file_location("window_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCertifier.WindowDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 4,000 instances and `reference_window`'s results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{window/instances.jsonl}", "{window/python.jsonl}", str(vectors.COUNT)]


def _refusal(carry, slots) -> str:
    """Why `reference_window` refuses a window: its carry, a hit, an order entry, or else the coder (a decision's KT
    counts, a symbol the book's table can't code)."""
    en, bit, hen, short, oen, p, r = slots
    if carry.run >= arith_encode.RUNS or carry.rank > WC.INDEX_CAP + 1:
        return "the carry"
    if any(e and s > WC.HIT_CAP for e, s in zip(hen, short)):
        return "a hit past HIT_CAP"
    pos = carry.pos
    for e, pp, rr in zip(oen, p, r):
        if not e:
            continue
        if pp == (1 << 32) - 1:
            return "an entry at 2^32 - 1"
        if pp < pos:
            return "an entry before pos"
        if pp - pos > WC.GAP_CAP:
            return "a gap past GAP_CAP"
        if rr > WC.INDEX_CAP:
            return "a rank past INDEX_CAP"
        pos = pp + 1
    return "the coder"


def test_the_generator_covers_what_it_says(monkeypatch):
    scalings, coded = Counter(), Counter()
    scale, binned, uniform = arith_encode._scale, WC.binned_symbols, WC._uniform

    def counted(low, high):
        out = scale(low, high)
        scalings[out[2]] += 1
        return out

    def seen_value(v, freqs, total, hi):
        coded[(hi, v)] += v == hi
        coded["bins at top"] += v.bit_length() == len(freqs) - 1
        return binned(v, freqs, total, hi)

    def seen_uniform(u, n):
        coded["two uniform steps"] += n > arith_encode.MAX_TOTAL
        return uniform(u, n)

    monkeypatch.setattr(arith_encode, "_scale", counted)
    monkeypatch.setattr(WC, "binned_symbols", seen_value)
    monkeypatch.setattr(WC, "_uniform", seen_uniform)
    seen = Counter()
    for seed in range(vectors.COUNT):
        inst = vectors.instance(seed)
        res = vectors.python_results(inst)
        assert len(res["steps"]) <= len(inst["windows"])
        assert all(s is not None for s in res["steps"][:-1])
        first = inst["carry"] == WC.WindowCarry(inst["priors"])
        seen["from the first carry" if first else "from a carry"] += 1
        seen["a refused window" if None in res["steps"] else "every window accepted"] += 1
        seen["refused" if res["charges"] is None else "charged"] += 1
        seen["a session of 3 or more charged"] += res["charges"] is not None and len(res["charges"]) >= 3
        seen["no window"] += not inst["windows"]
        seen["a window with no slot that holds a value"] += any(not any(w[0] + w[2] + w[4]) for w in inst["windows"])
        seen["hits and order entries"] += any(any(w[2]) and any(w[4]) for w in inst["windows"])
        seen["a window charged 0"] += any(s is not None and s[0] == 0 for s in res["steps"])
        seen["counts at 2^KW - 1"] += any(sum(c) == (1 << vectors.KW) - 1 for c in inst["carry"].counts)
        if None in res["steps"]:
            carry = inst["carry"]
            for s in res["steps"][:-1]:
                carry = WC.WindowCarry(tuple(map(tuple, s[1])), *s[2:])
            seen[_refusal(carry, inst["windows"][len(res["steps"]) - 1])] += 1
    for case in ("from the first carry", "from a carry", "a refused window", "every window accepted", "refused",
                 "charged"):
        assert seen[case] > 1000, (case, seen)
    for case in ("a session of 3 or more charged", "no window", "a window with no slot that holds a value",
                 "hits and order entries", "a window charged 0", "counts at 2^KW - 1", "the carry",
                 "a hit past HIT_CAP", "an entry at 2^32 - 1", "an entry before pos", "a gap past GAP_CAP",
                 "a rank past INDEX_CAP", "the coder"):
        assert seen[case] > 20, (case, seen)
    for case in ((WC.HIT_CAP, WC.HIT_CAP), (WC.GAP_CAP, WC.GAP_CAP), (WC.ZIG_CAP, WC.ZIG_CAP), "bins at top",
                 "two uniform steps"):
        assert coded[case] > 20, (case, coded)
    assert max(scalings) == 17 and scalings[17] > 100, scalings
