"""`circuit_check.ranges`: a sidecar's ``RANGES`` rules and their spots, which ids a rule covers, ``--ranges``' targets, the report's
``ranges`` and how a report registers an id; and that each rule in the tree names spots that bind, in its own module."""
from __future__ import annotations

import pytest

from circuit_check import ranges as RG

TEXT = "Fam_v1{T=1..64,NS=1..4,D=16,DOT=Dot_v1,SCALE=0.0625}"
RULE = RG.parse(TEXT, ["T=1,NS=1", "T=64,NS=4", "T=30,NS=2"], "m.circuit-check.py")


def _id(t: int, ns: int, d: int = 16, scale: str = "0.0625") -> str:
    return f"Fam_v1{{T={t},NS={ns},D={d},DOT=Dot_v1,SCALE={scale}}}"


def test_a_rule_spans_the_statics_written_as_ranges_and_fixes_the_rest_by_their_spelling():
    assert RULE.family == "Fam_v1" and RULE.over == (("T", 1, 64, 1), ("NS", 1, 4, 1))
    assert RULE.fixed == (("D", "16"), ("DOT", "Dot_v1"), ("SCALE", "0.0625"))
    assert RULE.spots == (_id(1, 1), _id(64, 4), _id(30, 2))
    assert RULE.at(_id(64, 1)) == {"T": 64, "NS": 1}
    assert [RULE.covers(i) for i in (_id(1, 4), _id(65, 1), _id(0, 1), _id(8, 5), _id(8, 1, d=32), _id(8, 1, scale="0.125"),
                                     "Fam_v2{T=8,NS=1,D=16,DOT=Dot_v1,SCALE=0.0625}", "Fam_v1{T=8,NS=1,D=16,DOT=Dot_v1}",
                                     "Fam_v1{T=8,NS=1,D=16,DOT=Dot_v1,SCALE=0.0625,X=1}",
                                     "Fam_v1{NS=1,T=8,D=16,DOT=Dot_v1,SCALE=0.0625}", "Fam_v1",
                                     "Fam_v1{T=x,NS=1,D=16,DOT=Dot_v1,SCALE=0.0625}")] == [True] + [False] * 11


def test_a_stepped_range_spans_its_grid_from_lo_to_hi_and_nothing_between():
    grid = RG.parse("Cls_v1{N=4096..131072 step 2048,K=2048}", ["N=4096", "N=131072", "N=67584"])
    assert grid.over == (("N", 4096, 131072, 2048),) and grid.spots[2] == "Cls_v1{N=67584,K=2048}"
    assert [grid.covers(f"Cls_v1{{N={n},K=2048}}") for n in (4096, 6144, 129024, 131072, 2048, 4097, 6143, 133120)] == \
        [True] * 4 + [False] * 4
    assert RG.statuses({"targets": [], "ranges": [{"rule": grid.text, "holds": True}]},
                       ["Cls_v1{N=8192,K=2048}", "Cls_v1{N=8193,K=2048}"]) == \
        {"Cls_v1{N=8192,K=2048}": "by range", "Cls_v1{N=8193,K=2048}": "unregistered"}


@pytest.mark.parametrize("text, spots, match", [
    ("Fam_v1", [], "is no id"),
    ("Fam_v1{T=4,D=16}", [], "spans no static"),
    ("Fam_v1{T=9..2}", [], "is empty"),
    ("Fam_v1{T=1..64 step 2}", [], "doesn't reach hi from lo"),
    ("Fam_v1{T=1..64 step 0}", [], "doesn't reach hi from lo"),
    ("Fam_v1{T=0..64 step 2}", ["T=0", "T=64", "T=33"], "outside its ranges"),
    (TEXT, ["T=1,NS=1", "T=64,NS=4", "T=65,NS=1"], "outside its ranges"),
    (TEXT, ["T=1,NS=1", "T=64,NS=4", "T=8"], "not an integer for each of"),
    (TEXT, ["T=1,NS=1", "T=64,NS=4", "T=1,NS=1"], "listed twice"),
    (TEXT, ["T=1,NS=1", "T=30,NS=4"], "no spot at T=64: add one at each end of each range"),
])
def test_a_malformed_rule_or_spot_is_refused_with_how_to_write_it(text, spots, match):
    with pytest.raises(ValueError, match=match):
        RG.parse(text, spots, "m.circuit-check.py")


def test_a_rule_holds_where_the_run_checked_each_of_its_spots_and_each_passed():
    ok = {"failures": []}
    targets = [*({"target": s, **ok} for s in RULE.spots), {"target": _id(9, 3), **ok},
               {"target": "Other_v1{N=4}", "failures": [{"name": "x"}]}]
    (got,) = RG.section(targets, [RULE])
    assert got == {"rule": TEXT, "sidecar": "m.circuit-check.py", "spots": list(RULE.spots), "missing": [], "failed": [],
                   "holds": True, "unchecked": {}}
    (short,) = RG.section(targets[1:], [RULE])
    assert short["missing"] == [_id(1, 1)] and not short["holds"]
    (bad,) = RG.section([*targets[:2], {"target": _id(30, 2), "failures": [{"name": "y"}]}], [RULE])
    assert bad["failed"] == [_id(30, 2)] and not bad["holds"]
    assert RG.section(targets[3:], [RULE]) == []


def test_a_spot_that_passed_without_running_a_check_is_named_beside_the_verdict():
    """The partition over its leaf budget of a spot that is no Call is a warning, which fails nothing (circuit-check's own
    verdict); the rule still holds, and the section names the spot and the check it skipped, so a reader doesn't take it as
    checked."""
    skip = {"check": "partition", "code": "unchecked"}
    style = {"check": "redundant-gates", "code": "boolean"}
    targets = [{"target": RULE.spots[0], "failures": [], "warnings": [style]},
               {"target": RULE.spots[1], "failures": [], "warnings": [skip, style]},
               {"target": RULE.spots[2], "failures": []}]
    (got,) = RG.section(targets, [RULE])
    assert got["holds"] and got["unchecked"] == {RULE.spots[1]: ["partition"]}


def test_a_report_registers_an_id_by_its_own_target_else_by_a_holding_rule():
    report = {"targets": [{"target": _id(1, 1), "failures": []}, {"target": _id(64, 4), "failures": []},
                          {"target": _id(2, 1), "failures": [{"name": "z"}]}],
              "ranges": [{"rule": TEXT, "holds": True}, {"rule": "Gam_v1{N=1..8}", "holds": False}]}
    ids = [_id(1, 1), _id(2, 1), _id(33, 3), "Gam_v1{N=4}", "Gam_v1{N=9}"]
    assert RG.statuses(report, ids) == {_id(1, 1): "checked", _id(2, 1): "failed", _id(33, 3): "by range",
                                        "Gam_v1{N=4}": "range fails", "Gam_v1{N=9}": "unregistered"}


def test_ranges_targets_are_the_named_families_spots_and_an_unknown_family_is_refused(monkeypatch):
    other = RG.parse("Gam_v1{N=1..8}", ["N=1", "N=8"])
    monkeypatch.setattr(RG, "declared", lambda: (RULE, other))
    assert RG.spots() == [*RULE.spots, "Gam_v1{N=1}", "Gam_v1{N=8}"]
    assert RG.spots(["Gam_v1"]) == ["Gam_v1{N=1}", "Gam_v1{N=8}"]
    with pytest.raises(LookupError, match="no RANGES rule for Hum_v1"):
        RG.spots(["Hum_v1"])


def test_each_rule_in_the_tree_names_spots_that_bind_to_its_own_modules_definitions():
    """A spot is a named target, so it must bind as the id the rule spells (a report's target is that string, and a reader looks
    served ids up by it); and the family is the module's whose sidecar holds the rule.  Bound, never built."""
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG

    TG.load_registries()
    by_file = {SC.rel(files["py"]): mod for mod, files in SC.discover().items() if "py" in files}
    for rule in RG.declared():
        for spot in rule.spots:
            fn = TG.definition(spot)
            assert fn.id == spot, f"{rule.sidecar}: spot {spot} binds as {fn.id}: write the rule in the id's own spelling"
            assert SC.owner(fn) == by_file[rule.sidecar], f"{rule.text} is no Definition of {by_file[rule.sidecar]}"
