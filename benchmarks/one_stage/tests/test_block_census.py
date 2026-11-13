"""`block_census.py`: each row's cut (the row itself when it fits, else each slot's coarsest fitting option, else a gap with
its finest option's reason; a ``COLS`` or ``WIDE_COORDS`` row or slot at its fitting width of least leakage-weighted proving
cost; no proof block
for a row of no gate or no input; a ``CHAINS`` row's key blocks as scores and values units before whole), the slots of each
kind of split the serving rows' tables name, and the census's counts per
Definition, phase, token and request over the ranks. The sized units are stand-ins: nothing is bound or lowered."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import block_census as B  # noqa: E402


def unit(spec, n=1, *, ands=1000, out=(16,), ins=(64,), est_log=22, rows_log=21, **extra):
    """A sized unit as `m1_families.sized` gives it."""
    return {"spec": spec, "n": n, "ands": ands, "macs": 0, "reads": 0, "asserts": 0, "unpieced": [], **extra,
            "block": {"in_bits": list(ins), "out_bits": list(out), "est_log": est_log, "rows_log": rows_log}}


def family(spec, units=(), *, steps=None, gates=10, **row):
    e = unit(spec, **row)
    del e["spec"], e["n"]
    return {"rows": sum((steps or {"0": 1}).values()), "steps": steps or {"0": 1}, "word_gates": gates,
            "estimate": {**e, "units": list(units)}}


@pytest.mark.parametrize("e, why", [
    ({"error": "does not bind"}, "does not bind"),
    (unit("X_v1{}", unpieced=["Erf_v1"]), "C-Flock has no piece for Erf_v1"),
    (unit("X_v1{}", rows_log=28, est_log=28), "its rows alone need a 2^28 block"),
    (unit("X_v1{}", est_log=28, ands=2**27), "a 2^28 block at 134,217,728 ANDs"),
    (unit("X_v1{}", out=(262_144, 16)), "262,160 output bits, past the widest size class"),
    (unit("X_v1{}", out=(262_144,)), None)])
def test_a_unit_fits_or_says_why_not(e, why):
    assert B.misfit(e) == why


def test_a_unit_holding_the_top_p_keep_word_names_the_word_gate_form():
    why = B.misfit(unit("GumbelTopPTokenSelectSharedGreedy_v1{V=154880}", unpieced=["TopPMaskWordx154880_v1"]))
    assert why.startswith("C-Flock has no piece for TopPMaskWordx154880_v1: ") and "TopPKeepWord_v2{V,S}" in why


def test_a_parts_rows_slots_are_its_parts_each_whole_then_finer():
    q, g, g1 = (unit(s) for s in ("Fp8GroupQuantUe8m0_v1{K=8}", "ScaledMmFp8BlockSharedScale_v1{K=8,N=4}",
                                  "ScaledMmFp8BlockSharedScale_v1{K=8,N=1}"))
    assert B.slots("DeepGemmFp8Linear_v1{K=8,N=4}", [q, g, g1]) == [[[q]], [[g], [g1]]]


def test_a_cut_rows_units_are_each_a_slot():
    a, b = unit("A_v1{}"), unit("B_v1{}")
    assert B.slots("RoPE_v1{D=64}", [a, b]) == [[[a]], [[b]]]


#: a decode row of GLM-5.3's sparse attention: 272 keys in five splits, eight heads
CHAIN_ROW = ('SparseMlaSplit_v1{T=272,NS=5,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
             'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}}')


def _chain_units(**sized):
    """`CHAIN_ROW`'s unit templates past itself as stand-in sized units (``sized``: a family's fields), and its ways' units."""
    ways = dict(B.ST.chain_ways(CHAIN_ROW))
    by = {u: unit(u, n, **sized.get(u.split("{")[0], {})) for way in ways.values() for u, n in way}
    return list(by.values()), {g: [by[u] for u, _n in way] for g, way in ways.items()}


def test_a_chain_rows_slot_is_its_key_blocks_as_scores_and_values_units_of_64_columns_then_whole():
    units, ways = _chain_units()
    assert B.CHAIN_COLS == 64 and sorted(ways, key=str) == [128, 16, 256, 32, 64, None]
    assert B.slots(CHAIN_ROW, units) == [[ways[64], ways[None]]]
    assert [(u["spec"].split("{")[0], u["n"]) for u in ways[64]] == [
        ("SparseMlaHeadScores_v1", 32), ("SparseMlaHeadValues_v1", 256), ("SparseMlaHeadScores_v1", 8),
        ("SparseMlaHeadValues_v1", 64), ("SparseMlaSplitEnd_v1", 320), ("SparseMlaCombine_v1", 64)]
    assert B.slots(CHAIN_ROW, ways[None]) == [[ways[None]]]                         # a census sized before the units existed


def test_a_chain_row_takes_its_scores_and_values_units_and_whole_key_blocks_when_one_misfits():
    units, ways = _chain_units()
    row = family(CHAIN_ROW, units, est_log=31)
    assert B.cut(CHAIN_ROW, row) == ("split", ways[64], None)
    units, ways = _chain_units(SparseMlaHeadValues_v1={"unpieced": ["Erf_v1"]})
    assert B.cut(CHAIN_ROW, family(CHAIN_ROW, units, est_log=31)) == ("split", ways[None], None)


def test_the_served_sparse_attention_partition_names_a_decode_rows_units_at_the_census_s_width():
    """The served partition's templates are `CHAIN_ROW`'s units at `CHAIN_COLS` (a scores or values unit its Boolean form), so
    the planner's card and the served partition hold one cut of the row."""
    served = json.loads((HERE.parent / "glm53_sparse_mla_tp8_step16.json").read_text())["templates"]
    assert sorted(B.ST.template(t).word for t in served) == sorted(u for u, _n in dict(B.ST.chain_ways(CHAIN_ROW))[B.CHAIN_COLS])


def test_a_kchunks_rows_slot_is_its_coordinate_or_its_chunks_of_one_width():
    c = unit("MoeExpertFp8Gemm_v1{K=512,N=1}", 4)
    ch = [unit(f"MoeExpertFp8CoordinateChunk_v1{{K={k},FIRST={f},LAST={la}}}", n)
          for k, f, la, n in ((256, "true", "false", 4), (256, "false", "true", 4), (128, "true", "false", 4),
                              (128, "false", "false", 8), (128, "false", "true", 4))]
    assert B.slots("MoeExpertFp8Gemm_v1{K=512,N=4}", [c, *ch]) == [[[c], ch[:2], ch[2:]]]


def test_a_coords_cols_or_slices_rows_slot_is_its_units_coarse_to_fine():
    h2, h4 = unit("MoeSum_v1{H=4}", 2), unit("MoeSum_v1{H=2}", 4)
    assert B.slots("MoeSum_v1{H=8}", [h2, h4]) == [[[h2], [h4]]]
    assert B.slots("Whole_v1{N=8}", []) == []


def test_a_row_that_fits_is_its_own_proof_block():
    kind, units, why = B.cut("MoeSum_v1{H=8}", family("MoeSum_v1{H=8}", [unit("MoeSum_v1{H=4}", 2)]))
    assert (kind, [(u["spec"], u["n"]) for u in units], why) == ("row", [("MoeSum_v1{H=8}", 1)], None)


def test_a_row_that_doesnt_fit_takes_each_slots_coarsest_fitting_option():
    q, g, g1 = (unit("Fp8GroupQuantUe8m0_v1{K=8}"), unit("ScaledMmFp8BlockSharedScale_v1{K=8,N=4}", est_log=28),
                unit("ScaledMmFp8BlockSharedScale_v1{K=8,N=1}", 4))
    kind, units, why = B.cut("DeepGemmFp8Linear_v1{K=8,N=4}", family("DeepGemmFp8Linear_v1{K=8,N=4}", [q, g, g1], est_log=29))
    assert (kind, [(u["spec"], u["n"]) for u in units], why) == (
        "split", [("Fp8GroupQuantUe8m0_v1{K=8}", 1), ("ScaledMmFp8BlockSharedScale_v1{K=8,N=1}", 4)], None)


def halvings(family, axis, W, est_log, out, last=1):
    """A ``COLS`` row's units at each halving of its width W down to ``last``, each one 2^1 rows and half the output bits
    smaller than the one before (the row itself at ``est_log`` and ``out``)."""
    got, k = [], 2
    while W // k >= last:
        got.append(unit(f"{family}{{{axis}={W // k}}}", k, est_log=est_log - k.bit_length() + 1, out=(out // k,)))
        k *= 2
    return got


def test_a_cols_row_takes_its_width_of_least_leakage_weighted_cost_no_smaller_than_the_floor():
    """Each width's cost is its blocks times their outputs' size class plus the location bits times 2^k: here it falls as the
    row halves, until the floor stops it at 2^22 rows (16 blocks of ResidualAdd_v1{N=4}), though the row itself fits."""
    us = halvings("ResidualAdd_v1", "N", 64, 26, 1024)
    kind, units, why = B.cut("ResidualAdd_v1{N=64}", family("ResidualAdd_v1{N=64}", us, est_log=26, out=(1024,)))
    assert (kind, [(u["spec"], u["n"]) for u in units], why) == ("split", [("ResidualAdd_v1{N=4}", 16)], None)
    assert [u["block"]["est_log"] for u in us] == [25, 24, 23, 22, 21, 20]
    costs = [B.leak_cost([u]) for u in us]
    assert costs == sorted(costs, reverse=True) and B.leak_cost(units) > costs[4]


def test_a_cols_row_smaller_than_the_floor_takes_its_coarsest_fitting_width():
    """No width is 2^WIDTH_FLOOR rows or more: the row itself is taken, though its halvings cost less."""
    us = halvings("ResidualAdd_v1", "N", 64, 21, 1024)
    kind, units, _ = B.cut("ResidualAdd_v1{N=64}", family("ResidualAdd_v1{N=64}", us, est_log=21, out=(1024,)))
    assert (kind, [(u["spec"], u["n"]) for u in units]) == ("row", [("ResidualAdd_v1{N=64}", 1)])
    assert B.leak_cost(units) > B.leak_cost(us[:1])


def test_a_splits_cols_slot_takes_its_width_of_least_cost_and_its_other_slots_their_coarsest():
    resid = [unit("AllReduceResidualMnnvl_v1{N=64}", est_log=26, out=(1024,)),
             *halvings("AllReduceResidualMnnvl_v1", "N", 64, 26, 1024, last=16)]
    norm = unit("RmsNormMnnvl_v1{N=64}", est_log=27)
    row = "AllReduceRmsNormMnnvl_v1{N=64}"
    kind, units, why = B.cut(row, family(row, [*resid, norm], est_log=28))
    assert (kind, [(u["spec"], u["n"]) for u in units], why) == (
        "split", [("AllReduceResidualMnnvl_v1{N=16}", 4), ("RmsNormMnnvl_v1{N=64}", 1)], None)


def coords(K, N, est_logs):
    """The block-scaled FP8 GEMM's coordinates of a row N wide at each width c of ``est_logs`` (c: its block's log2 rows),
    coarse to fine, each c 16-bit outputs."""
    return [unit(f"ScaledMmFp8BlockSharedScale_v1{{K={K},N={c},G=128}}", N // c, est_log=k, out=(16 * c,))
            for c, k in sorted(est_logs.items(), reverse=True)]


def test_a_gemm_s_coordinates_take_their_width_of_least_leakage_weighted_cost():
    """``WIDE_COORDS``: a coordinate of c outputs hashes the activation row once for c of them, so 4 outputs fit the 2^25 rows
    2 need, and cost least: their blocks times their outputs' size class plus the location bits times 2^k.  So the linear's
    GEMM slot, and a row of the GEMM's own, take coordinates of 4, where the coarsest fitting option would be 16."""
    cs = coords(6144, 16, {16: 27, 8: 26, 4: 25, 2: 25, 1: 24})
    costs = {u["spec"]: B.leak_cost([u]) for u in cs}
    assert min(costs, key=costs.get) == "ScaledMmFp8BlockSharedScale_v1{K=6144,N=4,G=128}"
    q = unit("Fp8GroupQuantUe8m0_v1{K=6144}")
    whole = unit("ScaledMmFp8BlockSharedScale_v1{K=6144,N=16,G=128}", est_log=28, out=(256,))
    row = "DeepGemmFp8Linear_v1{K=6144,N=16}"
    kind, units, why = B.cut(row, family(row, [q, whole, *cs], est_log=29))
    assert (kind, [(u["spec"], u["n"]) for u in units], why) == (
        "split", [("Fp8GroupQuantUe8m0_v1{K=6144}", 1), ("ScaledMmFp8BlockSharedScale_v1{K=6144,N=4,G=128}", 4)], None)
    own = "ScaledMmFp8BlockSharedScale_v1{K=6144,N=16,G=128}"
    kind, units, _ = B.cut(own, family(own, cs, est_log=28, out=(256,)))
    assert (kind, [(u["spec"], u["n"]) for u in units]) == ("split", [("ScaledMmFp8BlockSharedScale_v1{K=6144,N=4,G=128}", 4)])


def test_a_row_with_a_slot_no_option_fits_is_a_gap_with_its_finest_options_reason():
    h2, h4 = unit("MoeSum_v1{H=4}", 2, est_log=29), unit("MoeSum_v1{H=2}", 4, est_log=28)
    kind, units, why = B.cut("MoeSum_v1{H=8}", family("MoeSum_v1{H=8}", [h2, h4], est_log=30, ands=5))
    assert (kind, units) == ("gap", [])
    assert why == "a 2^30 block at 5 ANDs; finest split MoeSum_v1{H=2}: a 2^28 block at 1,000 ANDs"
    assert B.cut("Whole_v1{N=8}", family("Whole_v1{N=8}", est_log=30, ands=5)) == ("gap", [], "a 2^30 block at 5 ANDs")


def test_a_row_of_no_gate_or_no_input_is_no_proof_block():
    """A gather has no gate; a constant has no input, a Definition's parameters or, for a primitive constant that binds to
    no Definition, its row's argument groups."""
    assert B.cut("AllGather_v1{WORLD=8}", family("AllGather_v1{WORLD=8}", gates=0))[0] == "no gates"
    assert B.cut("Const_v1{}", family("Const_v1{}", ins=()))[0] == "const"
    const = {"rows": 2, "steps": {"0": 2}, "word_gates": None, "ports": {"in": [], "members": []},
             "estimate": {"error": "does not bind", "units": []}}
    assert B.cut("Const32[0x00000001]_v1", const) == ("const", [], None)
    assert B.cut("Const32[0x00000001]_v1", {**const, "ports": {"in": [16], "members": []}})[0] == "gap"


def doc(rank, families, prompt=4, steps=3):
    return {"rank": rank, "requests": {"r0": {"prompt_len": prompt, "steps": steps, "families": families}}}


def test_the_census_counts_each_definition_per_phase_token_and_request_over_the_ranks():
    """Two ranks of one request (a prefill step of 4 tokens, 2 decode steps): rank 0's row fits whole, rank 1's splits
    into 4 coordinates of a unit shared with rank 0's other row, and a gap row counts its rows but no proof block."""
    coord = unit("Lin_v1{N=1}", 4, ands=10, out=(16,), ins=(64, 32))
    r0 = {"Lin_v1{N=4}": family("Lin_v1{N=4}", [coord], steps={"0": 2, "1": 1, "2": 1}, ands=30, macs=8, out=(64,)),
          "Lin_v1{N=1}": family("Lin_v1{N=1}", steps={"0": 1}, ands=10, out=(16,), ins=(64, 32)),
          "Big_v1{}": family("Big_v1{}", steps={"1": 3}, est_log=30, ands=7)}
    r1 = {"Lin_v1{N=4}": family("Lin_v1{N=4}", [coord], steps={"0": 1, "2": 2}, ands=30, est_log=28, macs=8)}
    c = B.census([doc(0, r0), doc(1, r1)])
    assert c["ranks"] == [0, 1] and c["tokens"] == {"prefill": 4, "decode": 2}
    d = c["definitions"]
    assert {s: (x["prefill"], x["decode"], x["of"]) for s, x in d.items()} == {
        "Lin_v1{N=4}": (2, 2, {"Lin_v1": "row"}), "Lin_v1{N=1}": (1 + 4, 4 * 2, {"Lin_v1": "split"})}
    assert d["Lin_v1{N=1}"]["rows"] == [8, 4, 2] and d["Lin_v1{N=1}"]["row_bytes"] == 14
    t = c["totals"]
    assert (t["prefill"]["blocks"], t["decode"]["blocks"]) == (2 + 5, 2 + 8)
    assert (t["prefill"]["ands"], t["decode"]["ands"]) == (2 * 30 + 5 * 10, 2 * 30 + 8 * 10)
    assert (t["prefill"]["out_bits"], t["prefill"]["class_bits"]) == (2 * 64 + 5 * 16, 2 * 64 + 5 * 16)
    assert t["decode"]["circuit_ands"] == 2 * 30 + 3 * 7 + 2 * 30 and t["decode"]["circuit_macs"] == 4 * 8
    assert (t["decode"]["rows gap"], t["decode"]["gap_ands"], t["prefill"]["rows split"]) == (3, 3 * 7, 1)
    assert c["per_token"]["prefill"]["blocks"] == 7 / 4 and c["per_token"]["decode"]["blocks"] == 10 / 2
    assert c["per_request"]["blocks"] == 17 and c["per_request"]["rows"] == 4 + 7
    assert c["per_rank"][1]["decode"]["blocks"] == 8 and c["per_rank"][0]["prefill"]["blocks"] == 3
    assert list(c["gaps"]) == ["Big_v1{}"] and c["gaps"]["Big_v1{}"]["decode"] == 3
    card = B.card(c)
    assert card["definitions"]["Lin_v1{N=1}"] == {"family": "Lin_v1", "prefill": 5, "decode": 8, "out_bits": 16, "ands": 10,
                                                  "macs": 0, "rows": [8, 4, 2], "k_log": 22, "of": ["Lin_v1"]}
    assert card["gaps"] == {"Big_v1": {"definitions": 1, "prefill": 0, "decode": 3, "ands": 21,
                                       "why": "a 2^30 block at 7 ANDs"}}
    md = B.table(c)
    assert "| proof blocks | 1.8 | 5 | 17 |" in md and "| `Big_v1` | 1 | 0 | 3 | 21 | a 2^30 block at 7 ANDs |" in md


def test_the_tables_group_gaps_and_constants_by_family():
    consts = {f"Const32[0x0000000{i}]_v1": {"family": f"Const32[0x0000000{i}]_v1", "kind": "const", "prefill": 2, "decode": 1}
              for i in range(3)}
    assert B._by_family(consts) == [("Const32_v1", list(consts.values()))]


def test_the_census_refuses_ranks_of_two_request_shapes():
    with pytest.raises(ValueError, match="one request shape"):
        B.census([doc(0, {}, prompt=4), doc(1, {}, prompt=8)])


def test_main_writes_the_census_the_table_and_the_card(tmp_path, monkeypatch):
    (tmp_path / "s0.json").write_text(json.dumps(doc(0, {"Lin_v1{N=1}": family("Lin_v1{N=1}")})))
    monkeypatch.setattr(sys, "argv", ["block_census.py", "--sized", str(tmp_path / "s0.json"), "--out", str(tmp_path / "o")])
    assert B.main() == 0
    assert {p.name for p in (tmp_path / "o").iterdir()} == {"census.json", "table.md", "card.json"}
