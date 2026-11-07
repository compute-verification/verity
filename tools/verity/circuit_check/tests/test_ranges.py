"""`circuit_check.ranges`: a sidecar's ``RANGES`` rules, which ids each covers, the report's ``ranges`` and how a report registers
an id; and that each rule in the tree has its ends bound in its own module's sidecar, so ``--all`` checks them."""
from __future__ import annotations

import pytest

from circuit_check import ranges as RG

RULE = RG.parse("Fam_v1{T=1..64,NS=1..4,D=16,DOT=Dot_v1,SCALE=0.0625}", "m.circuit-check.py")


def _id(t: int, ns: int, d: int = 16, scale: str = "0.0625") -> str:
    return f"Fam_v1{{T={t},NS={ns},D={d},DOT=Dot_v1,SCALE={scale}}}"


def test_a_rule_spans_the_statics_written_as_ranges_and_fixes_the_rest_by_their_spelling():
    assert RULE.family == "Fam_v1" and RULE.over == (("T", 1, 64), ("NS", 1, 4))
    assert RULE.fixed == (("D", "16"), ("DOT", "Dot_v1"), ("SCALE", "0.0625"))
    assert RULE.at(_id(64, 1)) == {"T": 64, "NS": 1}
    assert [RULE.covers(i) for i in (_id(1, 4), _id(65, 1), _id(0, 1), _id(8, 5), _id(8, 1, d=32), _id(8, 1, scale="0.125"),
                                     "Fam_v2{T=8,NS=1,D=16,DOT=Dot_v1,SCALE=0.0625}", "Fam_v1{T=8,NS=1,D=16,DOT=Dot_v1}",
                                     "Fam_v1{T=8,NS=1,D=16,DOT=Dot_v1,SCALE=0.0625,X=1}", "Fam_v1")] == [True] + [False] * 9


@pytest.mark.parametrize("text, match", [("Fam_v1", "is no id"), ("Fam_v1{T=4,D=16}", "spans no static"),
                                         ("Fam_v1{T=9..2}", "is empty")])
def test_a_malformed_rule_is_refused_with_how_to_write_it(text, match):
    with pytest.raises(ValueError, match=match):
        RG.parse(text, "m.circuit-check.py")


def test_a_rule_holds_with_a_passing_spot_at_both_ends_of_each_range():
    ok = {"failures": []}
    targets = [{"target": _id(1, 1), **ok}, {"target": _id(64, 4), **ok}, {"target": _id(30, 2), **ok},
               {"target": "Other_v1{N=4}", "failures": [{"name": "x"}]}]
    (got,) = RG.section(targets, [RULE])
    assert got == {"rule": RULE.text, "sidecar": "m.circuit-check.py", "spots": [_id(1, 1), _id(64, 4), _id(30, 2)],
                   "failed": [], "unreached": {}, "holds": True}
    (short,) = RG.section(targets[1:], [RULE])
    assert short["unreached"] == {"T": [1], "NS": [1]} and not short["holds"]
    (bad,) = RG.section([*targets[:2], {"target": _id(30, 2), "failures": [{"name": "y"}]}], [RULE])
    assert bad["failed"] == [_id(30, 2)] and not bad["holds"]
    assert RG.section(targets[3:], [RULE]) == []


def test_a_report_registers_an_id_by_its_own_target_else_by_a_holding_rule():
    report = {"targets": [{"target": _id(1, 1), "failures": []}, {"target": _id(64, 4), "failures": []},
                          {"target": _id(2, 1), "failures": [{"name": "z"}]}],
              "ranges": [{"rule": RULE.text, "holds": True}, {"rule": "Gam_v1{N=1..8}", "holds": False}]}
    ids = [_id(1, 1), _id(2, 1), _id(33, 3), "Gam_v1{N=4}", "Gam_v1{N=9}"]
    assert RG.statuses(report, ids) == {_id(1, 1): "checked", _id(2, 1): "failed", _id(33, 3): "by range",
                                        "Gam_v1{N=4}": "range fails", "Gam_v1{N=9}": "unregistered"}


def test_each_rule_in_the_tree_binds_its_ends_in_its_own_modules_sidecar():
    """A rule holds only in a run that checks a spot at each end of each range, so the sidecar declaring it binds those spots
    (`_model_roots` entries, made here, never built), and its family is that module's."""
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG

    TG.load_registries()
    loaded = {SC.rel(SC.discover()[mod]["py"]): (mod, m) for mod, m in SC.binding_modules()}
    for rule in RG.declared():
        mod, m = loaded[rule.sidecar]
        made = [make() for _label, make in (getattr(m, "_model_roots", None) or list)()]
        spots = [f for f in made if rule.covers(f.id)]
        assert spots and rule.unreached(f.id for f in spots) == {}, (
            f"{rule.sidecar}: RANGES' {rule.text} has no _model_roots spot at {rule.unreached(f.id for f in spots)}: bind one "
            f"at each end of each range")
        assert {SC.owner(f) for f in spots} == {mod}, f"{rule.text} is no Definition of {mod}: its own module's sidecar holds it"
