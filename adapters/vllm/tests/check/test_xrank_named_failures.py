"""verity_vllm.check.match.xrank_collectives: what the cross-rank pass names when the ranks' dumps are not what it expects, and
the key `spec_key` gives a static that is no number.

The dumps are written out by hand: two ranks, one site, and picks that are honest, so that each case differs from a passing
pass by one thing.  A routed pick is not evaluable, by name, when its spec names a family that is no routed all-reduce, when
its rows reach past the call's `SIZE`, and when its partials are not rows.  An unrouted pick list is named when the ranks' lists
differ in length, when a pick's (site, step, kind) differs across ranks, and when its kind is neither collective.  A rank whose
dump states no sites of the unreplayable families makes the coverage incomplete.  Each is held as the pass answers today, the
`why` it writes included: a pick the ranks name differently is counted nowhere, so a pass of that pick alone says it drew no
picks; and a family that is no routed all-reduce is refused only after the call's `SIZE` is read."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from verity_vllm.check.match import xrank_collectives as XC

torch = pytest.importorskip("torch")

W, M, N = 2, 6, 8                                        # two ranks, a call of 6 rows of 8 words
ROWS = [1, 4, 5]
SITE = "model.layers.2.mlp.down_proj"
FAMILY = "AllReduce2_v1"                                 # the received-output family of an all-reduce between two ranks
ROUTE = "AllReduceCustom_v1"
STATICS = {"WORLD": W, "SIZE": M * N, "STAGE": 2}
NOT_EQUAL = "0 pick(s) where a committed received output != the definition over the committed partials, 1 not evaluable -- first: "
NAMED_DIFFERENTLY = "pick 0 names different (site, step, kind) across ranks"


def _tensor(words: np.ndarray) -> Any:
    return torch.from_numpy(np.ascontiguousarray(words).view(np.int16)).view(torch.bfloat16)


def _spec(family: str, statics: dict[str, int]) -> str:
    return family + "{" + ",".join(f"{k}={v}" for k, v in statics.items()) + "}"


def _routed(family: str = ROUTE, statics: dict[str, int] = STATICS, *, rows_of_one_dimension: bool = False) -> dict[int, dict[str, Any]]:
    """Two ranks' dumps of one routed pick: each rank's partial rows `ROWS` of the call, and the rows the route's Definition gives."""
    parts = (np.random.default_rng(1).standard_normal((W, len(ROWS), N)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    (received,) = XC.routed_expected(ROUTE, STATICS, parts, ROWS)
    pick = {"site": SITE, "step": 3, "kind": "all_reduce", "family": ROUTE, "stratum": f"{SITE}|all_reduce|decode", "rows": ROWS, "rows_total": M,
            "spec": _spec(family, statics), "statics": statics, "program_binds": True, "why": None}
    return {r: {"rank": r, "picks": [dict(pick, part=_tensor(parts[r, 0] if rows_of_one_dimension else parts[r]), recv=_tensor(received))]}
            for r in range(W)}


def _unrouted(kind: str = "all_reduce", family: str = FAMILY, n: tuple[int, int] = (1, 1)) -> dict[int, dict[str, Any]]:
    """Two ranks' dumps of `n[rank]` unrouted picks, each the same honest sum of the two partials."""
    generator = torch.Generator().manual_seed(0)
    a, b = (torch.randn(3, N, generator=generator).to(torch.bfloat16) for _ in range(W))
    received = (a.float() + b.float()).to(torch.bfloat16)
    pick = {"step": 2, "site": SITE, "kind": kind, "family": family, "stratum": f"{SITE}|{kind}|decode", "rows": [0, 1, 2], "why": None, "emb": None}
    return {r: {"rank": r, "picks": [dict(pick, part=part, recv=received.clone()) for _ in range(n[r])]} for r, part in enumerate((a, b))}


def _counts(res: dict[str, Any]) -> dict[str, Any]:
    return {k: res[k] for k in ("result", "picks", "equal", "mismatch_n", "missing_n")}


# --- spec_key ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec,statics", [
    ("Fam_v1{MODE=fast,N=4}", (("MODE", "fast"), ("N", 4))),
    ("Fam_v1{X=a b, Y= [1, 2 ,Z=3}", (("X", "a b"), ("Y", "[1, 2 ,Z=3"))),
], ids=["a word", "a bracket left open"])
def test_spec_key_keeps_a_static_that_is_neither_json_nor_a_literal_as_its_text(spec: str, statics: tuple) -> None:
    assert XC.spec_key(spec) == ("Fam_v1", statics)


def test_spec_key_of_a_static_that_is_a_map_or_a_set_is_hashable_and_ignores_its_order() -> None:
    literal = XC.spec_key("Fam_v1{A={'y':(2,1),'x':[3]},B={3,1,2}}")
    assert literal == ("Fam_v1", (("A", (("x", (3,)), ("y", (2, 1)))), ("B", (1, 2, 3))))
    assert XC.spec_key("Fam_v1{B={2,3,1},A={'x':[3],'y':(2,1)}}") in {literal}
    assert XC.spec_key("Fam_v1{A={'y':(2,1),'x':[4]},B={3,1,2}}") not in {literal}
    assert XC.spec_key("Fam_v1{A={'y':(2,1),'x':[3]},B={3,1}}") not in {literal}
    # as a Program's rows spell a map: JSON, its floats as hex
    described = XC.spec_key('Fam_v1{MAP={"dict":{"b":[2,1],"a":true}},K={"b":1,"a":{"f64":"0x1p-1"}}}')
    assert described == ("Fam_v1", (("K", (("a", 0.5), ("b", 1))), ("MAP", (("a", True), ("b", (2, 1))))))
    assert XC.spec_key('Fam_v1{K={"a":{"f64":"0x1p-1"},"b":1},MAP={"dict":{"a":true,"b":[2,1]}}}') in {described}


# --- a routed pick ----------------------------------------------------------------------------------------------------------------

def test_routed_expected_refuses_a_family_that_is_no_routed_all_reduce_once_it_has_read_the_calls_size() -> None:
    parts: np.ndarray = np.zeros((W, len(ROWS), N), np.uint16)
    with pytest.raises(ValueError, match=f"^{FAMILY}: not a routed all-reduce$"):
        XC.routed_expected(FAMILY, {"SIZE": M * N}, parts, ROWS)
    with pytest.raises(ValueError, match=f"^rows 5 x N={N} past SIZE=40$"):
        XC.routed_expected(FAMILY, {"SIZE": 40}, parts, ROWS)
    with pytest.raises(KeyError, match="SIZE"):                      # the family is asked last: statics without the size are not named
        XC.routed_expected(FAMILY, {"N": M * N}, parts, ROWS)


def test_the_honest_routed_pick_passes() -> None:
    res = XC.check_picks(_routed())
    assert _counts(res) == {"result": True, "picks": 1, "equal": 1, "mismatch_n": 0, "missing_n": 0}
    assert res["by_family"] == {ROUTE: {"picks": 1, "equal": 1, "mismatch": 0, "missing": 0}}


@pytest.mark.parametrize("dumps,why", [
    ({"family": "Other_v1"}, "Other_v1: Other_v1: not a routed all-reduce"),
    ({"statics": {**STATICS, "SIZE": 40}}, f"{ROUTE}: rows 5 x N={N} past SIZE=40"),
    ({"rows_of_one_dimension": True}, f"partials of shape ({N},); expected [rows, N]"),
], ids=["a family that is no route", "rows past the size", "partials that are not rows"])
def test_a_routed_pick_that_cannot_be_evaluated_is_missing_by_name(dumps: dict[str, Any], why: str) -> None:
    res = XC.check_picks(_routed(**dumps))
    assert _counts(res) == {"result": False, "picks": 1, "equal": 0, "mismatch_n": 0, "missing_n": 1}
    assert res["missing"] == [{"pick": 0, "site": SITE, "step": 3, "kind": "all_reduce", "family": ROUTE, "why": why}]
    assert res["mismatches"] == [] and res["by_family"] == {ROUTE: {"picks": 1, "equal": 0, "mismatch": 0, "missing": 1}}
    assert res["by_stratum"] == {f"{SITE}|all_reduce|decode": {"picks": 1, "equal": 0, "families": [ROUTE]}}
    assert res["why"] == f"{NOT_EQUAL}{SITE} step 3 all_reduce {why}"


# --- the ranks' pick lists --------------------------------------------------------------------------------------------------------

def test_pick_lists_of_unlike_length_are_named_and_the_picks_both_ranks_have_are_still_evaluated() -> None:
    res = XC.check_picks(_unrouted(n=(2, 1)))
    assert _counts(res) == {"result": False, "picks": 1, "equal": 1, "mismatch_n": 0, "missing_n": 1}
    assert res["missing"] == [{"why": "pick lists differ across ranks: [2, 1]"}]
    assert res["by_family"] == {FAMILY: {"picks": 1, "equal": 1, "mismatch": 0, "missing": 0}}
    assert res["why"] == f"{NOT_EQUAL}None step None  pick lists differ across ranks: [2, 1]"
    assert XC.check_picks(_unrouted(n=(1, 1)))["result"] is True


DIFFERENTLY = pytest.mark.parametrize("field,other", [("site", "model.layers.3.mlp.down_proj"), ("step", 9), ("kind", "all_gather")])


@DIFFERENTLY
def test_a_pick_the_ranks_name_differently_is_named_and_counted_nowhere(field: str, other: Any) -> None:
    dumps = _unrouted()
    dumps[1]["picks"][0][field] = other
    res = XC.check_picks(dumps)
    assert _counts(res) == {"result": False, "picks": 0, "equal": 0, "mismatch_n": 0, "missing_n": 1}
    named = {"site": SITE, "step": 2, "kind": "all_reduce"}
    assert res["missing"] == [{"pick": 0, "why": NAMED_DIFFERENTLY, "keys": [tuple(named.values()), tuple({**named, field: other}.values())]}]
    assert res["by_family"] == {} and res["by_stratum"] == {}
    assert res["why"] == "no collective picks (no rank partials committed?)"       # the pass counted none, and says so instead


@DIFFERENTLY
def test_the_pick_after_one_the_ranks_name_differently_is_still_evaluated(field: str, other: Any) -> None:
    dumps = _unrouted(n=(2, 2))
    dumps[1]["picks"][0][field] = other
    res = XC.check_picks(dumps)
    assert _counts(res) == {"result": False, "picks": 1, "equal": 1, "mismatch_n": 0, "missing_n": 1}
    assert [m["pick"] for m in res["missing"]] == [0] and res["by_family"] == {FAMILY: {"picks": 1, "equal": 1, "mismatch": 0, "missing": 0}}
    assert res["why"] == f"{NOT_EQUAL}None step None  {NAMED_DIFFERENTLY}"


def test_a_kind_that_is_neither_collective_is_named_unbound() -> None:
    res = XC.check_picks(_unrouted(kind="all_to_all", family="all_to_all"))
    assert _counts(res) == {"result": False, "picks": 1, "equal": 0, "mismatch_n": 0, "missing_n": 1}
    assert res["missing"] == [{"pick": 0, "site": SITE, "step": 2, "why": "unbound collective kind all_to_all"}]
    assert res["by_family"] == {"all_to_all": {"picks": 1, "equal": 0, "mismatch": 0, "missing": 1}}
    assert res["by_stratum"] == {f"{SITE}|all_to_all|decode": {"picks": 1, "equal": 0, "families": ["all_to_all"]}}
    assert res["why"] == f"{NOT_EQUAL}{SITE} step 2  unbound collective kind all_to_all"


# --- coverage of the unreplayable sites -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sites", [{}, {"unreplayable_sites": None}, {"unreplayable_sites": [SITE]}], ids=["absent", "none", "a list"])
def test_a_rank_that_states_no_sites_makes_the_coverage_incomplete_by_name(sites: dict[str, Any]) -> None:
    stated = {"rank": 0, "unreplayable_sites": {FAMILY: [SITE]}}
    res = XC.check_picks(_unrouted())
    assert XC.coverage_of_unreplayable(res, [stated, {"rank": 1, "unreplayable_sites": {FAMILY: [SITE]}}])["complete"] is True and res["result"] is True
    res = XC.check_picks(_unrouted())
    coverage = XC.coverage_of_unreplayable(res, [stated, {"rank": 1, **sites}])
    assert {k: v for k, v in coverage.items() if k != "rule"} == {
        "complete": False, "families": {FAMILY: {"sites": 1, "covered": 1, "missing": [], "picks": 1, "equal": 1}}, "strata_sites_n": 1,
        "errors": ["rank 1: no unreplayable_sites"], "why": "rank 1: no unreplayable_sites"}
    assert res["result"] is False and res["coverage_of_unreplayable"] is coverage
    assert res["why"] == "coverage of the unreplayable collective sites incomplete -- rank 1: no unreplayable_sites"
