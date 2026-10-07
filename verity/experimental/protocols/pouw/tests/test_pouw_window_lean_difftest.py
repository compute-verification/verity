"""The difftest of `pouw.window` against its executable Lean (`pouw_window`): the Python half.

`verity/experimental/core/protocols/pouw/window/scripts/difftest_vectors.py` writes 12,000 random instances and this
module's answers on each. `check` runs the Lean half in its Lean check (`runs` in that package's `lean-audit.json`):
`PouwWindow/DifftestMain.lean` evaluates every instance with the package's definitions and fails on any difference,
refusals compared by their reason. The Lean check holds that data to the SHA-256 its record pins, and its pass is cached
on the Lean package's files alone, so the test here is what says the data is still what `window.py` computes: a change to
an answer changes the data and fails this test, and `check.py --update` records the new data, which makes `check` run the
Lean half on it. The tests here also check that the generator reaches every refusal its docstring promises.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

LEAN = Path(__file__).resolve().parents[3] / "core" / "protocols" / "pouw" / "window"
_spec = importlib.util.spec_from_file_location("pouw_window_difftest_vectors", LEAN / "scripts" / "difftest_vectors.py")
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["PouwWindow.DifftestMain"]


def test_the_data_is_what_the_lean_check_pins(tmp_path):
    """All 12,000 instances and window.py's answers on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_reaches_every_refusal():
    pairs = [vectors.instance(s) for s in range(3000)]
    by = {}
    for inst, ans in pairs:
        by.setdefault(inst["op"], []).append((inst, ans))
    assert set(by) == {"constants", "grid", "levels", "k_prime", "sampling", "window"}
    reasons = " | ".join(a["problem"] or "accepted" for _, a in by["window"])
    for needle in ("accepted", "not the bound", "kind is none of", "the calls, then one filler", "kept rows are not",
                   "does not start where", "not its", "the filler's range", "not the padded count",
                   "not the registered level"):
        assert needle in reasons, needle
    assert sum(len(a["tile_at"]) for _, a in by["window"]) > 500
    assert any("refused" in r for _, a in by["window"] for r in a["tile_at"] + a["tile_in"])
    ks = [(i, a["k_prime"]) for i, a in by["k_prime"]]
    assert any("refused" in k for _, k in ks)
    assert any(k.get("ok") == i["n"] > 1 for i, k in ks)
    assert any(k.get("ok", 0) > 10_000 for _, k in ks) and any(i["n"] > 1 << 50 for i, _ in ks)
    grid = [a for _, a in by["grid"]]
    for f in ("floor_point", "level", "point", "bucket"):
        assert any("refused" in a[f] for a in grid) and any("ok" in a[f] for a in grid), f
    assert any("refused" in a["levels"] for _, a in by["levels"]) and any(a["levels"].get("ok", 1) <= 0 for _, a in by["levels"])
    assert any("refused" in a["sampling"] for _, a in by["sampling"])