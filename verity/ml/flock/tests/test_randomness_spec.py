"""`verity.ml.randomness` against its Lean spec (`verity/core/primitives/randomness/Randomness.lean`),
which the keyed-draw theorems read (`KeyedDraw.lean`), through vectors the spec generates (`fixtures/randomness_lean.json`, `lean/generate.sh`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.randomness import derive

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "randomness_lean.json").read_text())


def _part(p):
    return bytes.fromhex(p["bytes"]) if isinstance(p, dict) else p


def _key(v):
    return derive(bytes.fromhex(v["source"]), v["domain"], {k: _part(x) for k, x in v["context"].items()})


def test_the_lean_spec_reproduces_the_pinned_vectors():
    """The spec, run on `verity/ml/randomness/tests/vectors.json`'s inputs, returns every frame, key, stream, shard
    key, uniform value, Bernoulli bit and subset the file pins."""
    assert FIXTURE["kat"] == json.loads((ROOT / "verity/ml/randomness/tests/vectors.json").read_text())


def test_python_matches_every_lean_vector():
    """`derive`, `Key.stream`, `Key.uniform` and `Key.subset` return what the spec does on its own inputs: PoUW's
    closure-draw domain, and `plan.draw`'s `("call", forward, position, "stratum", name)` contexts, whose swap `i` reads
    the stream at `("s", *context, i)`."""
    assert len(FIXTURE["extra"]) == 3
    for v in FIXTURE["extra"]:
        key = _key(v)
        assert key.value.hex() == v["key"]
        for s in v["streams"]:
            assert key.stream(tuple(_part(p) for p in s["index"]), 100).hex() == s["bytes"]
        for n, idx, want in v["uniform"]:
            assert key.uniform(n, tuple(_part(p) for p in idx)) == want
        for k, n, idx, want in v["subset"]:
            assert list(key.subset(k, n, tuple(_part(p) for p in idx))) == want


def test_the_window_draw_is_plan_draw_per_call_under_one_key():
    """`PlanDraw.window`: each call's `plan.draw` loop (each stratum's `Key.subset(k, len(units), context)` mapped through
    its units), the calls in order under the one key, and a window with a repeated call index refused. Restated here from
    #364's `plan.draw`; the test against `verity.core.protocols.pouw.circuit.plan` itself lands with #364 and its admission fix."""
    for w in FIXTURE["window"]:
        key, seen, drawn = _key(w), set(), []
        for c in w["calls"]:
            index = tuple(_part(p) for p in c["index"])
            if index in seen:
                drawn = None
                break
            seen.add(index)
            for s in c["strata"]:
                context = ("call", *index, "stratum", s["name"]) if index else ("stratum", s["name"])
                drawn += [s["units"][i] for i in key.subset(s["k"], len(s["units"]), context)]
        assert drawn == w["drawn"]
    assert [w["drawn"] is None for w in FIXTURE["window"]] == [False, True]


def test_the_lean_vectors_were_generated_from_this_tree():
    """The fixture records the sha256 of the exporter, the pinned vectors, the spec's modules, the SHA-256 and uniform
    sampler they run on, and the toolchain pins. A change to any of them fails here until the vectors are regenerated."""
    stale = [p for p, h in FIXTURE["inputs"].items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not stale, f"changed since the vectors were generated, rerun verity/ml/flock/tests/lean/generate.sh: {stale}"
