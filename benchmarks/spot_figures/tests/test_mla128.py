"""The 128k workload's sparse-MLA lever (`breakeven/mla128.py`): a row's units as serving cuts it, the rule held to
the `workload` card's own request, and each block's factor from the serve plan's requests."""
from collections import Counter

import cards
import pytest

from breakeven import mla128 as M


def card() -> dict:
    return cards.load("workload", "GLM-5.3")["value"]


def test_a_row_is_counted_as_serving_cuts_it():
    """At one split a row is each head's key blocks and its end; at more, each split's blocks, its end in the column
    blocks and a combine of each column block a head; a row the card holds whole stays whole. The units are
    `sparse_mla_split.split_chain`'s, by count."""
    from verity.definitions.registry import sparse_mla, sparse_mla_split as S

    assert M.units(9, 1, 8, frozenset({9})) == Counter({("whole", 9): 1})
    assert M.units(9, 1, 8) == Counter({("first", 9): 8, ("end", 0): 8})
    assert M.units(130, 1, 8) == Counter({("first", 2): 8, ("later", 64): 16, ("end", 0): 8})
    assert M.units(287, 5, 8) == Counter({("first", 64): 32, ("first", 31): 8, ("cols", 0): 40, ("combine", 5): 64})
    assert M.units(2048, 28, 8)[("combine", 16)] == 64 and M.units(2048, 28, 8)[("later", 64)] == 8 * 16
    row = S.bind(S.SparseMlaSplit, T=287, NS=5, **sparse_mla.SPARSE_MLA_GLM53)
    assert sorted(n for _c, n in S.split_chain(row)) == sorted([32, 8, *[40] * 8, 64])


def test_a_request_s_prefill_is_one_launch_and_each_decode_step_its_own():
    """A prefill past the largest batch that splits is at one split a row; a decode step of one token splits as its
    keys give, at most the indexer's top-k of them; the prefix a request reads from the cache is not computed."""
    rows = M.rows_of(256, 31)
    assert sum(rows.values()) == 287 and {ns for (_t, ns) in rows if _t <= 256} == {1}
    assert rows[(287, 5)] == 1 and rows[(257, 5)] == 1
    long = M.rows_of(5000, 2, 4096)
    assert long == Counter({(M.TOPK, 1): 904, (M.TOPK, M.decode_splits(M.TOPK)): 2})
    assert sum(M.rows_of(3000, 0).values()) == 3000 and M.rows_of(3000, 0)[(M.TOPK, 1)] == 3000 - M.TOPK + 1


def test_the_rule_gives_the_card_its_own_request_s_blocks():
    """Over the card's own request every sparse-MLA block's factor is 1: the rule reproduces the card's count of each
    type, which `volume` checks before it counts the serve plan. A card with one count moved is refused by the
    block's name, and so is a sparse-MLA block type the lever has no unit for."""
    c = card()
    own = [(c["request"]["prefill"], c["tokens"] - c["request"]["prefill"], 0)]
    vol = M.volume(c, own)
    assert len(vol) > 80 and all(f == pytest.approx(1.0, rel=1e-12) for f in vol.values())
    end = next(x["block"] for x in c["strata"] if x["block"].startswith("SparseMlaHeadEnd_v1{"))
    moved = c | {"strata": [x | {"per_token": 2 * x["per_token"]} if x["block"] == end else x for x in c["strata"]]}
    with pytest.raises(ValueError, match="1 of 84 block types differ, the first `SparseMlaHeadEnd_v1"):
        M.volume(moved, own)
    with pytest.raises(ValueError, match="`SparseMlaHeadSplit_v9{T=3}`: a sparse-MLA block type the 128k lever has no"):
        M.volume(c | {"strata": [*c["strata"], c["strata"][0] | {"block": "SparseMlaHeadSplit_v9{T=3}"}]}, own)


def head_units() -> dict:
    """The `workload` card cut by the partition card `head-scores-values`: each head block its scores and values units."""
    import total_cost

    part = next(x for x in cards.load_all("partition", "GLM-5.3") if x["value"]["name"] == "head-scores-values")
    return card() | {"strata": total_cost.cut_strata(card()["strata"], part)}


def test_a_head_block_s_scores_and_values_units_take_its_factor():
    """Cut into scores and values units, the card's own request still gives every sparse-MLA block's count: a values
    unit and a 64-column end are one of eight column blocks of their unit. Over the serve plan each unit takes its
    key block's factor, and a 64-column end its whole end's."""
    c, u = card(), head_units()
    assert [M.columns(b) for b in ('SparseMlaHeadValues_v1{NVIS=64,FIRST=false,BN=64,G=64}',
                                   'SparseMlaHeadEnd_v1{DV=64,INV={"fn":"Fa3InvSum_v1"}}',
                                   'SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}}',
                                   'SparseMlaSplitEnd_v1{DV=64,INV={"fn":"Fa3InvSum_v1"}}',
                                   'SparseMlaSplitEndCols_v1{DV=512,G=64,C=3}',
                                   'SparseMlaHeadScores_v1{DR=64,DV=512,NVIS=64,FIRST=false}')] == [8, 8, 1, 8, 1, 1]
    own = [(c["request"]["prefill"], c["tokens"] - c["request"]["prefill"], 0)]
    vol = M.volume(u, own)
    assert len(vol) == 142 and all(f == pytest.approx(1.0, rel=1e-12) for f in vol.values())
    assert not any(b.startswith("SparseMlaHeadBlock_v1{") for b in vol)
    whole, cut = M.volume(c), M.volume(u)
    by = {(M.unit_of(b), b.split("{")[0]): f for b, f in whole.items()}
    for b, f in cut.items():
        kind = {"SparseMlaHeadScores_v1": "SparseMlaHeadBlock_v1", "SparseMlaHeadValues_v1": "SparseMlaHeadBlock_v1",
                "SparseMlaSplitEnd_v1": "SparseMlaSplitEndCols_v1"}.get(b.split("{")[0], b.split("{")[0])
        assert f == pytest.approx(by[(M.unit_of(b), kind)], rel=1e-12), b


def test_the_serve_plan_s_requests_set_each_block_s_factor():
    """The 128k workload reads many times the census request's keys a token: its head blocks a token are several
    times the card's, its whole rows a small share of them, and a combine over n splits counts as n / 5 of the card's
    five-split combine. A request served wholly from its prefix cache but for one decode step at the top-k's keys has
    that step's units alone."""
    c = card()
    vol = M.volume(c)
    reqs = M.requests()
    assert len(reqs) == 651 and all(h % M.PLAN_BLOCK == 0 and h <= p for p, _o, h in reqs)
    later = next(b for b in vol if "NVIS=64,FIRST=false" in b)
    whole = [f for b, f in vol.items() if b.startswith("SparseMlaSplit_v1{")]
    assert 5 < vol[later] < 20 and all(0 < f < 0.2 for f in whole)
    tokens = c["tokens"]
    per = {M.unit_of(x["block"]): x["per_token"] for x in c["strata"] if x["block"].startswith("SparseMla")}
    one = M.volume(c, [(M.TOPK - 1, 1, M.TOPK - 1)])
    comb = next(b for b in one if b.startswith("SparseMlaCombine_v1{NSPLIT=5,"))
    rows = per[("end", 0)] * tokens / (8 * (c["request"]["prefill"] - 9))
    assert one[comb] == pytest.approx(64 * 16 / 5 / (per[("combine", 5)] / rows), rel=1e-12)
    assert one[later] == pytest.approx(8 * 16 / (per[("later", 64)] / rows), rel=1e-12)
    assert all(one[b] == 0 for b in one if b.startswith(("SparseMlaSplit_v1{", "SparseMlaHeadEnd_v1{")))
