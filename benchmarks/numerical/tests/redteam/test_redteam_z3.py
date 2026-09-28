"""Task 1 (z3): the transcribed predicates.  Over the integers every attack query is UNSAT; over F_p the gadget
queries are SAT exactly where the wrap analysis says, and a whole-group model found by z3 replays through the
relation's own ``check_step``."""
from __future__ import annotations

import pytest

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP
from verity_numerical.redteam import cases as C
from verity_numerical.redteam import forge as F
from verity_numerical.redteam import relation_target as RT
from verity_numerical.redteam import z3_relation as Z

GOLDEN, MEASURED = C.golden_cases()
PIPES = {"ampere": AMP, "hopper": HOP}
TO = 30_000


def _witness(pipe, x):
    return R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]))


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
def test_gadgets_over_the_integers_are_pinned(pname: str) -> None:
    pipe = PIPES[pname]
    x = GOLDEN[0][1] if pipe is AMP else dict(C.structured_cases(pipe))["random[0]:kind0"]
    w = _witness(pipe, x)
    g = w["groups"][0]
    t = next(t for t in g["terms"] if t["aligned"] and t["d"])
    scaled = t["aligned"] * t["pow"] + t["r"]
    exps = [e for _, e, _ in R._terms(R.params(pipe), w["in"], w["prod"][:pipe.groups[0]])]
    assert Z.align_query(pipe, None, t["d"], scaled, timeout_ms=TO)[0] == "UNSAT"
    assert Z.final_query(pipe, None, g["magW"], g["e_out"], timeout_ms=TO)[0] == "UNSAT"
    assert Z.norm_query(pipe, None, g["mag_t"], timeout_ms=TO)[0] == "UNSAT"
    assert Z.estar_query(pipe, None, exps, mode="above", timeout_ms=TO)[0] == "UNSAT"
    assert Z.estar_query(pipe, None, exps, mode="nonmax", timeout_ms=TO)[0] == "UNSAT"
    assert Z.estar_query(pipe, None, exps, timeout_ms=TO)[0] == "UNSAT"
    assert Z.sign_query(pipe, None, g["total"], timeout_ms=TO)[0] == "UNSAT"
    st, model, _ = Z.sign_query(pipe, None, 0, timeout_ms=TO)          # a zero total: the sign column is free
    assert st == "SAT" and model["mag_t"] == 0


def test_gadgets_over_a_40_bit_field_wrap() -> None:
    x = GOLDEN[0][1]
    w = _witness(AMP, x)
    g = w["groups"][0]
    t = g["terms"][1]
    scaled = t["aligned"] * t["pow"] + t["r"]
    st, model, _ = Z.align_query(AMP, RT.P40, t["d"], scaled, timeout_ms=TO)
    assert st == "SAT" and model["aligned"] != model["honest_aligned"] and 0 <= model["rinv"] < 1 << 26
    st, model, _ = Z.final_query(AMP, RT.P40, g["magW"], g["e_out"], timeout_ms=TO)
    assert st == "SAT" and model["mag24"] != model["honest_mag24"] and 0 <= model["rsinv"] < 1 << 35
    # above the thresholds the same gadgets are pinned
    assert Z.align_query(AMP, RT.GOLDILOCKS, t["d"], scaled, timeout_ms=TO)[0] == "UNSAT"
    assert Z.final_query(AMP, RT.GOLDILOCKS, g["magW"], g["e_out"], timeout_ms=TO)[0] == "UNSAT"
    assert Z.norm_query(AMP, RT.GOLDILOCKS, g["mag_t"], timeout_ms=TO)[0] == "UNSAT"


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
@pytest.mark.slow
def test_whole_group_and_step_over_the_integers_are_pinned(pname: str) -> None:
    pipe = PIPES[pname]
    pool = GOLDEN[:4] if pipe is AMP else C.structured_cases(pipe)[:4]
    for name, x in pool:
        w = _witness(pipe, x)
        for gi in range(len(pipe.groups)):
            assert Z.group_query(pipe, None, w, gi, timeout_ms=TO)[0] == "UNSAT", (name, gi)
        assert Z.step_query(pipe, None, w, timeout_ms=60_000)[0] == "UNSAT", name


@pytest.mark.slow
def test_whole_group_over_goldilocks_is_forgeable_and_the_model_replays() -> None:
    """z3 finds the unranged-magW forgery on its own; the model, replayed through ``check_step`` under
    ``FieldCensus``, is accepted with an output one ulp below the measured word."""
    name, x = GOLDEN[0]
    w = _witness(AMP, x)
    st, model, _ = Z.group_query(AMP, RT.GOLDILOCKS, w, 0, timeout_ms=120_000)
    if st == "UNKNOWN":
        pytest.skip("z3 timed out; the closed-form forgery covers this instance (test_redteam_forgeries)")
    assert st == "SAT"
    w2 = Z.witness_from_model(AMP, w, model, RT.GOLDILOCKS)
    F._regroup_following(AMP, w2, 0)
    y2 = R.pack_state(w2["out"])
    ok, viol = RT.relation_check(RT.RelationSemantics(AMP, R.ALIGN_DIV, RT.GOLDILOCKS), x, y2, w2)
    assert ok, viol[:3]
    assert y2 != MEASURED[name]
    assert not 0 <= w2["groups"][0]["magW"] < 1 << AMP.width                        # the field-sized magW


def test_repair_range_magW_pins_the_group_over_goldilocks() -> None:
    name, x = GOLDEN[0]
    w = _witness(AMP, x)
    st, _, _ = Z.group_query(AMP, RT.GOLDILOCKS, w, 0, timeout_ms=120_000, extra_ranges={"magW": AMP.width})
    assert st in ("UNSAT", "UNKNOWN")
    if st == "UNKNOWN":
        pytest.skip("z3 timed out on the repaired group")
