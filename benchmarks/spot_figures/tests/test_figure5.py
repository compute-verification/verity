import figure1 as f1
import figure5 as f
import pytest

FP4 = 3.0e12


def _row(case, scope, tax, beta, at_one):
    return {"case": case, "scope": scope, "tax": tax, "h": 1e8, "floor": 1e12, "beta": {"FP8": 2 * beta, "FP4": beta},
            "bound": {"0.01": at_one, "0.05": at_one / 5, "0.1": at_one / 10}}


SURFACE = {"meta": {"weights_bits": {"FP8": 2 * FP4, "FP4": FP4}},
           "rows": [_row(c, s, 0.02, beta, one) for c, beta, one in (("A", 11.0, 3e15), ("F", 1.8, 5e14))
                    for s in (f1.PRIMARY, f1.SECONDARY)]}
LATER = {"meta": {"weights_bits": {"FP4": FP4}}, "rows": [_row("F", f1.PRIMARY, 0.02, 1.5, 4e14)]}
PATH = {"design": "the cut isn't served",
        "steps": [{"label": "today", "status": "measured", "basis": "served-card-priced", "cases": {"two-stage": "A"},
                   "note": "as served"},
                  {"label": "a cache", "status": "assumed", "basis": "served-card-priced",
                   "cases": {"two-stage": None}, "note": "pending"},
                  {"label": "the cut", "status": "modeled", "basis": "design", "cases": {"two-stage": "F"},
                   "note": "fused"}],
        "off": [{"label": "a hash cut", "status": "assumed", "basis": "design", "owner": None, "after": "today",
                 "cases": {"two-stage": "F"}, "note": "off"}],
        "unpriced": [{"label": "a slice (modeled)", "status": "modeled", "note": "adds to every step"},
                     {"label": "the advice charge", "status": "charged", "note": "until a rule lands"}]}


@pytest.fixture(scope="module")
def out():
    return f.run([SURFACE, LATER], PATH)


def test_a_step_totals_its_tax_and_beta_and_a_later_surface_replaces_its_row(out):
    today, cache, cut = out["steps"]
    a = today["at"]["two-stage"][f1.PRIMARY]
    assert a["total"] == pytest.approx(11.02) and not today["pending"]
    assert cut["at"]["two-stage"][f1.PRIMARY]["beta"] == 1.5
    assert cut["at"]["two-stage"][f1.SECONDARY]["beta"] == 1.8
    assert cache["pending"] and cache["at"]["two-stage"][f1.PRIMARY] is None


def test_a_design_whose_bound_never_reaches_the_weights_has_no_total_and_the_caption_says_so():
    never = _row("O", f1.PRIMARY, 5.7, 1.0, 1e18) | {"beta": {"FP4": None}}
    path = PATH | {"steps": [PATH["steps"][0] | {"cases": {"two-stage": "A", "one-stage": "O"}}]}
    out = f.run([SURFACE, {"meta": SURFACE["meta"], "rows": [never]}], path)
    assert out["steps"][0]["at"]["one-stage"][f1.PRIMARY]["total"] is None
    assert "The one-stage design never breaks even: its serve tax alone is 570%" in f.caption(out)


def test_the_bound_at_the_goal_is_interpolated_on_log_log_axes(out):
    a = out["steps"][0]["at"]["two-stage"][f1.PRIMARY]
    assert a["bound_at_goal"] == pytest.approx(3e15 * 0.01 / 0.08)
    assert a["times_target"] == pytest.approx(a["bound_at_goal"] / FP4)
    assert f.bound_at(_row("x", "s", 0, 1, 1e15), 0.005) == pytest.approx(2e15)


def test_a_status_outside_the_three_is_refused():
    bad = {"steps": [PATH["steps"][0] | {"status": "projected"}]}
    with pytest.raises(ValueError, match="not one of"):
        f.run([SURFACE], bad)


def test_a_basis_outside_the_three_or_a_design_without_its_reason_is_refused():
    with pytest.raises(ValueError, match="basis 'served' is not one of"):
        f.run([SURFACE], {"steps": [PATH["steps"][0] | {"basis": "served"}]})
    with pytest.raises(ValueError, match="basis None is not one of"):
        f.run([SURFACE], {"steps": [{k: v for k, v in PATH["steps"][0].items() if k != "basis"}]})
    with pytest.raises(ValueError, match="doesn't say what makes it one"):
        f.run([SURFACE], {"steps": PATH["steps"]})


def test_the_caption_calls_no_design_today_and_names_where_the_path_turns_to_one(out):
    text = f.caption(out)
    assert "Today, the served cut at card prices, breaks even at 1,102%" in text
    assert "From the cut on, every total is a design not served: the cut isn't served." in text
    measured = f.run([SURFACE, LATER], PATH | {"steps": [PATH["steps"][0] | {"basis": "served-measured"}]})
    assert "Today's measured system breaks even at 1,102%" in f.caption(measured)
    design = f.run([SURFACE, LATER], PATH | {"steps": [PATH["steps"][0] | {"basis": "design"}]})
    assert "Today's measured system" not in f.caption(design)
    assert "Today, a design not served, breaks even" in f.caption(design)


def test_the_headline_cards_rows_are_stated_with_their_basis():
    import cards
    cells = {"trace S": {"without m = 37": {f1.PRIMARY: 0.1806}, "with m = 37": {f1.PRIMARY: 0.1347}}}
    row = {"row": "(ii)", "label": "(ii) a fused cut", "pricing": "on its own premises", "basis": "design",
           "basis_from": ["a cut no serving path commits"], "rests_on": [], "cells": cells,
           "runs": ["r20261109-192926-0dd9"]}
    card = {"schema": cards.SCHEMA, "slot": "headline", "model": "GLM-5.3", "status": "modeled", "provisional": True,
            "source": "a run", "runs": ["r20261109-192926-0dd9"], "owner": "@proofs",
            "value": {"bases": list(cards.BASES), "rows": [row], "quote": "none"}}
    out = f.run([SURFACE, LATER], PATH, card)
    assert out["headline"][0]["basis"] == "design"
    assert ("(ii) a fused cut, on its own premises (a design not served): trace S 18.1% without m = 37 and "
            "13.5% with m = 37. No row is served and measured.") in f.caption(out)
    with pytest.raises(ValueError, match="a measured headline card holds only served-measured rows"):
        f.run([SURFACE], PATH, card | {"status": "measured"})


def test_an_off_path_lever_is_priced_beside_its_step_and_a_bad_branch_is_refused(out):
    (hash_cut,) = out["off"]
    assert hash_cut["after_index"] == 0 and hash_cut["at"]["two-stage"][f1.PRIMARY]["beta"] == 1.5
    with pytest.raises(ValueError, match="no step"):
        f.run([SURFACE], PATH | {"off": [PATH["off"][0] | {"after": "nowhere"}]})


def test_the_figure_renders(out, tmp_path):
    text = f.caption(out)
    assert "breaks even at 1,102%, 110x the goal" in text
    assert "A cache (assumed, pending: no number yet, and the steps after it leave it out)" in text
    assert "Off the path, grey: a hash cut (unowned, assumed): off; after today it would break even at 152%" in text
    assert "Not in any step, a slice (modeled): adds to every step." in text
    assert "The advice charge is charged, in no total: until a rule lands." in text
    assert [p.name for p in f.render(out, tmp_path)] == ["figure5.png", "figure5.svg"]


def test_the_default_path_names_a_case_and_a_status_for_every_step():
    assert f.PATH["steps"][0]["cases"] == {"two-stage": "R0"}
    assert {st["status"] for st in f.PATH["steps"] + f.PATH["off"]} <= set(f.STATUSES)
    assert {st["basis"] for st in f.PATH["steps"] + f.PATH["off"]} == {"design"} and f.PATH["design"]
    assert not any("hash" in c for st in f.PATH["steps"] for c in st["cases"].values() if c)
    assert [a["status"] for a in f.PATH["unpriced"]] == ["modeled", "charged"]
