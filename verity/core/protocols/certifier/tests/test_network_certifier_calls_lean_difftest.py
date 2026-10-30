"""The difftest of `calls.Calls` against the certifier's Lean program: the Python half.

`lean/scripts/calls_vectors.py` writes 1,000 random windows of the lead's calls, each an egress cap, the node's release
bits, the certifier's entropy and the window's events, and `Calls`'s answers on each. `check` runs the Lean half in its
Lean audit (`runs` in `lean/lean-audit.json`): `lean/NetworkCertifier/CallsDifftestMain.lean` runs the program's
`Calls.run` on every window and fails on any difference: each A root's release byte for byte or its refusal, each
verdict and the record's forwards. The audit checks that data against the record's SHA-256, and its pass is cached on the Lean package's files
alone, so the test here is what says the data is still what this package computes: a change to `calls` changes the
data, fails this test, and `check.py --update` records the new data, which makes `check` run the Lean half on it. The
other test checks that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from collections import Counter
from pathlib import Path

from verity.core.protocols.certifier import calls as C

LEAN = Path(__file__).resolve().parents[1] / "lean"
SCRIPT = LEAN / "scripts" / "calls_vectors.py"
_spec = importlib.util.spec_from_file_location("calls_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCertifier.CallsDifftestMain"]

#: What a `CallRefusal` refuses, by a part of its message.
REFUSALS = {"the window is closed": "after the close", "is registered already": "a second A root",
            "is a 64-byte SHA-512 root": "a root not of 64 bytes", "is a positive integer": "a dimension 0",
            "more than 2^40 MMA tiles": "past 2^40 tiles", "holds at most": "frames past the allocation",
            "is not open": "a words root of a forward not open", "the forwards are serial": "another forward open",
            " is closed": "a closed forward"}


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 1,000 windows and `Calls`'s answers on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{calls/instances.jsonl}", "{calls/python.jsonl}", str(vectors.COUNT)]


def _replay(inst: dict) -> tuple[list[str], list[dict], bool]:
    """Each event's outcome on a fresh `Calls` (its kind and `answered` or `no coins`, or what it refuses), the record's
    forwards, and whether entropy is left over."""
    entropy = vectors.Entropy(inst["entropy"])
    calls, out = C.Calls(inst["cap"], entropy, inst["release_bits"]), []
    for kind, *args in inst["events"]:
        try:
            if kind == "a":
                calls.a_root(args[0], args[1], bytes.fromhex(args[2]), C.CallShape(*args[3:]))
            elif kind == "b":
                calls.bucket(*args)
            elif kind == "w":
                calls.words_root(args[0], bytes.fromhex(args[1]))
            else:
                calls.close()
        except C.CallRefusal as e:
            out.append(next(why for key, why in REFUSALS.items() if key in str(e)))
            continue
        except ValueError:
            out.append(f"{kind}: no coins")
            continue
        out.append(f"{kind}: answered")
    return out, calls.summary(), "a: no coins" not in out and entropy.at < len(entropy.data)


def test_the_generator_covers_what_it_says(monkeypatch):
    seen, g, bits = Counter(), Counter(), Counter()
    weights, passed = C._weights, C.EgressCap.passed

    def counted_weights(entropy, n):
        read = []

        def reading(k):
            out = entropy(k)
            read.append(out)
            return out

        made = weights(reading, n)
        words = struct.unpack(f"<{sum(map(len, read)) // 4}I", b"".join(read))
        g["rejected"] += len(words) - n
        for name, w in (("G_BOUND", C.G_BOUND), ("2^32 - 1", 2**32 - 1), ("0", 0), ("G_BOUND - 1", C.G_BOUND - 1)):
            g[name] += words.count(w)
        return made

    def counted_passed(cap, macs, count):
        limit = cap.cap * macs // cap.fwd_macs
        bits["at its bits"] += count == limit
        bits["one past"] += count == limit + 1
        return passed(cap, macs, count)

    monkeypatch.setattr(C, "_weights", counted_weights)
    monkeypatch.setattr(C.EgressCap, "passed", counted_passed)
    for seed in range(vectors.COUNT):
        inst = vectors.instance(seed)
        outcomes, summary, left = _replay(inst)
        seen.update(outcomes)
        flips = [(C.CallShape(*ev[4:]), o) for ev, o in zip(inst["events"], outcomes) if o.startswith("a: ")]
        seen["a flip of two row words a row"] += any(s.H == 2 for s, o in flips if o == "a: answered")
        seen["a flip of three or four coin tiles"] += any(s.tiles >= 3 for s, o in flips if o == "a: answered")
        seen["a flip of one step"] += any(s.S == 1 for s, o in flips if o == "a: answered")
        seen["no coins at 2^40 tiles"] += any(s.S > 2**39 for s, o in flips if o == "a: no coins")
        seen["no coins after the entropy ended"] += outcomes.count("a: no coins") > 1
        seen.update(f"verdict: {f['verdict']}" for f in summary)
        seen["a forward of two calls or more"] += any(len(f["calls"]) > 1 for f in summary)
        seen["release bits"] += inst["release_bits"] > 0 and any(f["verdict"] for f in summary)
        seen["a window left open"] += "c: answered" not in outcomes
        seen["entropy left over"] += left
    for case in ("a: answered", "b: answered", "w: answered", "c: answered", "a: no coins", "verdict: accepted",
                 "verdict: past its bits", "verdict: open when the window closed",
                 "verdict: a call's coins were not flipped", "a forward of two calls or more", "release bits"):
        assert seen[case] > 150, (case, seen)
    for case in (*REFUSALS.values(), "a flip of two row words a row", "a flip of three or four coin tiles",
                 "a flip of one step", "no coins at 2^40 tiles", "no coins after the entropy ended", "verdict: None",
                 "a window left open", "entropy left over"):
        assert seen[case] > 10, (case, seen)
    assert g["rejected"] > 300 and all(g[k] > 10 for k in ("G_BOUND", "2^32 - 1", "0", "G_BOUND - 1")), g
    assert bits["at its bits"] > 200 and bits["one past"] > 200, bits
