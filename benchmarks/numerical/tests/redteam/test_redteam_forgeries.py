"""Task 1 (field semantics): forged witnesses the relation accepts over a prime field with the wrong output, judged
by ``check_step`` itself under ``FieldCensus``; and the same forgeries rejected where the analysis says they must be."""
from __future__ import annotations

import pytest

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP
from verity_numerical.redteam import cases as C
from verity_numerical.redteam import forge as F
from verity_numerical.redteam import relation_target as RT

GOLDEN, MEASURED = C.golden_cases()
PIPES = {"ampere": AMP, "hopper": HOP}


def _cases(pipe, n_random: int = 48):
    return (GOLDEN[:60] if pipe is AMP else []) + C.structured_cases(pipe, n_random=n_random)


def _accepted(pipe, field, x, fg) -> bool:
    ok, _ = RT.relation_check(RT.RelationSemantics(pipe, R.ALIGN_DIV, field), x, fg.forged_output, fg.witness)
    return ok


def _sweep(pipe, field, gadget, n_random: int = 48):
    fn = F.FORGERIES[gadget]
    wrong, same = [], 0
    for name, x in _cases(pipe, n_random):
        w = R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]))
        y = R.pack_state(w["out"])
        for gi in range(len(pipe.groups)):
            fg = fn(pipe, w, field, gi)
            if fg is None:
                continue
            assert _accepted(pipe, field, x, fg), (gadget, name, gi)      # every forgery we build must be accepted
            assert fg.honest_output == y
            if fg.forged_output != y:
                wrong.append((name, gi, fg))
            else:
                same += 1
    return wrong, same


def test_honest_witness_is_accepted_over_every_field() -> None:
    for pipe in PIPES.values():
        for field in (RT.P40, RT.M61, RT.GOLDILOCKS):
            t = RT.relation_target(pipe, field=field)
            for name, x in _cases(pipe)[:40]:
                assert t.check(x, t.reference(x), t.gen(x)), (pipe.name, field, name)


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
@pytest.mark.parametrize("gadget", ["align", "final", "norm"])
def test_gadget_forgeries_accepted_at_40_bits_rejected_above_threshold(pname: str, gadget: str) -> None:
    """The docstring's 'field of at least 40 bits': all three wrap gadgets forge at p ~ 2^40 (the leading-bit one on
    a fraction of instances); none exists at Goldilocks, which lies above every threshold."""
    pipe = PIPES[pname]
    n_random = 400 if gadget == "norm" else 48       # the leading-bit wrap lands in the window on ~1-5% of groups
    wrong, _ = _sweep(pipe, RT.P40, gadget, n_random)
    assert wrong, gadget
    name, gi, fg = wrong[0]
    assert fg.forged_output != fg.honest_output and "missing_bound" in fg.detail
    for field in (RT.GOLDILOCKS,):
        assert field > F.max_products(pipe)[gadget]
        wrong_big, same_big = _sweep(pipe, field, gadget, n_random)
        assert wrong_big == [] and same_big == 0


@pytest.mark.parametrize("pname", ["ampere", "hopper"])
@pytest.mark.parametrize("field", [RT.P40, RT.M61, RT.GOLDILOCKS, (1 << 89) - 1, (1 << 127) - 1])
def test_split_forgery_is_accepted_over_every_prime_field(pname: str, field: int) -> None:
    """``magW`` is never range-checked: mis-splitting ``normB`` gives a field-sized ``magW`` and the final shift then
    accepts another ``mag24``.  Above 2^59 the accepted output is exactly one ulp low whenever the honest ``r_s`` is 0."""
    pipe = PIPES[pname]
    wrong, same = _sweep(pipe, field, "split")
    assert len(wrong) >= 0.25 * sum(len(pipe.groups) for _ in _cases(pipe))
    if field > 1 << 60:
        for name, gi, fg in wrong:
            assert fg.detail["mag24_forged"] == fg.detail["mag24_honest"] - 1, (name, gi)


def test_split_forgery_concrete_record_0() -> None:
    """The first golden record, group 0, over Goldilocks: honest 0x3fd82158, accepted 0x3fd82156."""
    name, x = GOLDEN[0]
    w = R.step_witness(AMP, R.decode_state(x[0], AMP), list(x[1]), list(x[2]))
    fg = F.forge_split(AMP, w, RT.GOLDILOCKS, 0)
    assert fg is not None and fg.honest_output == 0x3FD82158 == MEASURED[name]
    assert fg.forged_output == 0x3FD82156
    assert fg.detail["r_n_honest"] == 0 and fg.detail["r_n_forged"] == 1
    assert fg.detail["mag24_forged"] == fg.detail["mag24_honest"] - 1
    assert _accepted(AMP, RT.GOLDILOCKS, x, fg)
    # over the integers the same witness is rejected by the split identity (magW is a field element)
    ok, viol = RT.relation_check(RT.RelationSemantics(AMP, R.ALIGN_DIV, None), x, fg.forged_output, fg.witness)
    assert not ok and any("normW" in v or "final" in v or "rsinv" in v for v in viol), viol[:3]
    # the repair: a W-bit range check on magW rejects it
    p = R.params(AMP)
    assert not 0 <= fg.witness["groups"][0]["magW"] < 1 << p.W


def test_forgeries_only_touch_the_forged_gadget() -> None:
    """Everything downstream of the forged value is the honest derivation from it: with the forged column pinned,
    re-deriving the tail reproduces the forged witness exactly."""
    name, x = GOLDEN[3]
    w = R.step_witness(AMP, R.decode_state(x[0], AMP), list(x[1]), list(x[2]))
    fg = F.forge_align(AMP, w, RT.P40, 0)
    assert fg is not None
    g = fg.witness["groups"][0]
    total = sum(t["aligned"] - 2 * t["signed"] for t in g["terms"])
    assert g["total"] == total and g["mag_t"] == abs(total) and g["L"] == abs(total).bit_length()
    assert fg.witness["groups"][1] == R.step_witness(
        R.Pipeline(name="g", arch=AMP.arch, dtype="bf16", operand_bits=16, groups=(8,), width=AMP.width,
                   zero_exponent=AMP.zero_exponent, validated=""), g["out"], list(x[1][8:]), list(x[2][8:]))["groups"][0]
