"""Task 2: the relation's witness generator satisfies every predicate on every all-finite golden record and on the
structured cases, and the witness's own output is the measured / reference word."""
from __future__ import annotations

import pytest

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP, tc_dot
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP
from verity_numerical.redteam import cases as C
from verity_numerical.redteam import harness as H
from verity_numerical.redteam import relation_target as RT

GOLDEN, MEASURED = C.golden_cases()
PIPES = {"ampere": AMP, "hopper": HOP}


def test_golden_fixture_shape() -> None:
    assert len(GOLDEN) == 360
    for name, (acc, a, b) in GOLDEN:
        assert len(a) == len(b) == 16 and 0 <= acc < 1 << 32
        assert (acc >> 23) & 0xFF != 0xFF, name                       # all-finite records


@pytest.mark.parametrize("align", [R.ALIGN_DIV, R.ALIGN_TABLE])
def test_completeness_on_golden_records(align: str) -> None:
    t = RT.relation_target(AMP, align)
    f = H.run_completeness(t, GOLDEN)
    assert f.status == "PASS", f.detail["failures"][:3]
    for name, x in GOLDEN:                                          # the relation's output is the measured word
        assert t.output_of(t.gen(x)) == MEASURED[name] == tc_dot(AMP, x[0], list(x[1]), list(x[2])), name


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
@pytest.mark.parametrize("align", [R.ALIGN_DIV, R.ALIGN_TABLE])
def test_completeness_on_structured_cases(pname: str, align: str) -> None:
    pipe = PIPES[pname]
    t = RT.relation_target(pipe, align)
    cs = C.structured_cases(pipe)
    f = H.run_completeness(t, cs)
    assert f.status == "PASS", f.detail["failures"][:3]
    finite = [(n, x) for n, x in cs if (x[0] >> 23) & 0xFF != 0xFF]
    for name, x in finite:                                          # finite domain: silicon and the relation agree
        assert t.output_of(t.gen(x)) == tc_dot(pipe, x[0], list(x[1]), list(x[2])), name


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
def test_structured_cases_reach_the_corners(pname: str) -> None:
    pipe = PIPES[pname]
    cs = dict(C.structured_cases(pipe))
    W = pipe.width

    def wit(name):
        acc, a, b = cs[name]
        return R.step_witness(pipe, R.decode_state(acc, pipe), list(a), list(b))

    assert max(t["d"] for t in wit("max_shift_d506")["groups"][0]["terms"]) == 506
    g = wit("flush_total_1")["groups"][0]
    assert g["total"] == 1 and g["L"] == 1 and g["mag24"] == 0 and g["z_out"] == 1        # truncation to zero
    g = wit("cancel_huge_ovf_flag")["groups"][0]
    assert g["total"] == 0 and g["ovf"] == 1 and g["z_t"] == 1 and g["ovf_eff"] == 0     # cancellation under ovf
    assert wit("L_above_W_all_max")["groups"][0]["L"] > W
    assert wit("overflow_group0")["groups"][0]["out"].inf == 1
    assert wit("overflow_group0")["out"].inf == 1                                          # sticky through the step
    g = wit("subnormal_acc_min")["groups"][0]
    assert g["e_fin"] == R.FP32_MIN_E and g["mag24"] == 1
    w = wit("subnormal_acc_cancel")
    assert w["out"].M == 0 and w["out"].e == pipe.zero_exponent and w["out"].neg == 0     # canonical zero state
