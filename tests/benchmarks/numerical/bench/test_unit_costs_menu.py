"""The pinned menu's Lean: `lean` writes the menu as the Lean module `VStar.Menu.Pinned` and refuses a bound that is no
whole number; the module in the tree is `lean` of the pinned menu JSON, byte for byte; and V*'s staging of the reps'
algebra refuses a shape past the lone part's bound."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from verity_flock import rec_residuals as RR
from verity_flock.rec_algebra import Shape

from verity_numerical.bench import unit_costs_menu as M


def test_the_lean_menu_lists_each_model_s_entries():
    mn = {"format": "verity/f6-menu/v1", "K": 50, "models": [
        {"workload": "w1", "entries": [{"slots": 16, "bound": 3.0, "inner_m": 30, "k_log": 22, "claims": 8, "positions": 3,
                                        "unit_types": ["a", "b"]},
                                       {"slots": 4, "bound": 7.0, "inner_m": 28, "k_log": 26, "claims": 6, "positions": 1,
                                        "unit_types": ["c"]}]},
        {"workload": "w2", "entries": [{"slots": 1, "bound": 9.0, "inner_m": 26, "k_log": 24, "claims": 16, "positions": 2,
                                        "unit_types": ["d"]}]}]}
    text = M.lean(mn, "SRC")
    assert text.startswith("module\n\npublic import VStar.Menu\n") and "\nSRC\n(verity/f6-menu/v1, K = 50); do not edit. -/" in text
    assert ('def pinned : List (String × List Entry) := [\n  ("w1",\n    [⟨16, 3, 30, 22, 8, 3, 2⟩,\n     ⟨4, 7, 28, 26, 6, 1, 1⟩]),\n'
            '  ("w2",\n    [⟨1, 9, 26, 24, 16, 2, 1⟩])]\n') in text
    mn["models"][1]["entries"][0]["bound"] = 9.5
    with pytest.raises(ValueError, match="not a whole number"):
        M.lean(mn, "SRC")


def test_the_pinned_lean_menu_is_generated_from_the_pinned_menu_json():
    """`VStar.Menu.Pinned` is `lean` of F6's pinned `menu.json` (`data/f6-menu.json`), byte for byte."""
    here = Path(__file__).resolve()
    pinned = here.parents[4] / "verity/protocols/proof/flock/verifier/VStar/Menu/Pinned.lean"
    mn = json.loads((here.parent / "data/f6-menu.json").read_text())
    assert M.lean(mn, "`tests/benchmarks/numerical/bench/data/f6-menu.json`") == pinned.read_text()


def test_v_star_stages_the_reps_algebra_at_16_claims_and_refuses_past_the_lone_part_bound():
    """At m 35, k_log 27 and 16 claims the batched-T residual alone needs more sha512x3 slots than a part holds, so it
    takes a part of its own (at most ``MAX_LONE_SLOTS``) and V* stages the algebra in 3 parts. At m 26, k_log 24 and 28
    claims it needs 300 alone, past that bound, and V* refuses (the control of `test_vstar_menu.py`)."""
    products, why = M.rep_algebra(35, 27, 16)
    assert products > 4900 and why is None
    assert len(RR.parts(RR.spec(Shape(35, 27, 16)))) == 3
    _, why = M.rep_algebra(26, 24, 28)
    assert why == f"V* parts: residual 32 alone reads more than {RR.MAX_REGIONS} regions or {RR.MAX_LONE_SLOTS} sha512x3 slots"
