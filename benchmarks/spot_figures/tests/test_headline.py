"""The figure of record's driver (`breakeven/headline.py`): a row counts every sparse-MLA head, carries its own plan's
request count and split, and leaves the planner as it found it; the figure reads the 10% point off pinned rows."""
import hashlib
import json
from dataclasses import astuple
from math import isclose
from pathlib import Path

import cards
import pytest

from breakeven import headline as H

COMBINE = "SparseMlaCombine_v1{NSPLIT=5,DV=64}"
PAGES = "PageCopy[262144 bits]"
#: the row's last pricing call, and its model's strata (the copy stratum is priced after pricing, in the model)
CALLS = [(True, [{"block": COMBINE, "kind": "SparseMlaCombine_v1", "per_token": 2.0, "seconds_per_block": 1.0,
                  "fixed_seconds": 4.0, "statement_blocks": 4, "ands": 10, "k_log": 24, "priced": "measured",
                  "macs": 0}])]
STRATA = [{"block": COMBINE, "priced": "measured"}, {"block": PAGES, "priced": "modeled"}]


def plan(beta=0.4, replay_sample=2.0e10, q=1.0e-6, prove_s=(3.0, 1.0)):
    """A plan of one cell and the carried pages left out."""
    return {"beta": beta, "replay_sample": replay_sample, "q": q,
            "cells": [{"block": COMBINE, "share": 0.75, "prove_s": prove_s[0], "taken": 2.0}],
            "left_out": [{"block": PAGES, "share": 0.25, "prove_s": prove_s[1], "taken": 1.0}]}


def row(p, requests=None):
    return {"row": "(i) honest", "scope": "world year", "beta": p["beta"], "replay_sample": p["replay_sample"],
            "requests": p["replay_sample"] / p["q"] if requests is None else requests}


def test_the_sparse_mla_blocks_count_glm53_s_heads():
    """The 128k workload's sparse-MLA blocks are GLM-5.3's eight heads', not one head's: every sparse-MLA block of
    the `workload` card takes a factor, a head's units at one head a row an eighth of it and a whole row its own."""
    from verity.definitions.registry.sparse_mla import SPARSE_MLA_GLM53

    assert H.heads() == SPARSE_MLA_GLM53["NH"] == 8
    vol, one = H.volume(), H.volume(1)
    card = [x["block"] for x in cards.load("workload", "GLM-5.3")["value"]["strata"]]
    assert set(vol) == {b for b in card if b.startswith("SparseMla")} and len(vol) > 1
    for b, f in vol.items():
        assert f > 0 and isclose(one[b], f if b.startswith("SparseMlaSplit_v1{") else f / 8, rel_tol=1e-12), b
    spec = H.spec_of("(i) honest", "world year", 0.4)
    assert spec["volume"] == vol and spec["heads"] == 8
    assert H.spec_of("(i) honest", "world year", 0.4, n_heads=1)["volume"] == one


def test_the_head_count_is_read_not_written(monkeypatch):
    """The volume counts the registry's `NH` heads a row, as a head count written into the driver would not: under
    another count the card's own request no longer gives the card's blocks, and the row is refused."""
    from verity.definitions.registry import sparse_mla

    monkeypatch.setitem(sparse_mla.SPARSE_MLA_GLM53, "NH", 3)
    with pytest.raises(ValueError, match="not its own request's at 3 heads a row"):
        H.spec_of("(iii) honest", "datacenter week", 1.0)


def test_no_row_rescales_the_combine_and_the_warm_factor_names_the_card_s_units():
    """The lever counts a combine's splits in its volume, so no row scales the combine's seconds; the third row's warm
    factor names the card's sparse-MLA units, and each type kept at one prover a GPU is on the card."""
    assert [r["scale"] for r in H.ROWS.values()][:2] == [{}, {}]
    warm = H.ROWS["(iii) honest"]["scale"]
    card = [x["block"] for x in cards.load("workload", "GLM-5.3")["value"]["strata"]]
    assert all(H.factors({"block": b}, warm) == (0.25, 0.25) for b in card if b.startswith("SparseMla"))
    assert all(any(b.startswith(p) for b in card) for p in ("SparseMlaHeadBlock_v1{", *H.PAIR_GIB))


def test_a_row_whose_requests_are_not_its_plan_s_is_refused():
    """A row carrying another count than its plan's (the census's, say, beside a plan over the 128k line) is refused;
    its own plan's is kept."""
    p = plan()
    assert H.assemble(row(p), p, CALLS, STRATA, CALLS[0][1])["requests"] == p["replay_sample"] / p["q"]
    with pytest.raises(ValueError, match="requests are not its plan's"):
        H.assemble(row(p, requests=1.14 * p["replay_sample"] / p["q"]), p, CALLS, STRATA, CALLS[0][1])


def test_a_row_s_split_is_its_own_plan_s():
    """A row's split is its own plan's at its beta, its blocks priced as its own calls and model priced them, and
    another beta's plan is refused."""
    a, b = plan(), plan(beta=1.0, replay_sample=5.0e10, prove_s=(6.0, 2.0))
    got = H.assemble(row(a), a, CALLS, STRATA, CALLS[0][1])["split"]
    assert (got["X"], got["q"], got["replay_sample"]) == (4.0, a["q"], a["replay_sample"])
    assert got["priced"] == {"measured": 0.75, "modeled": 0.25}
    assert got["families"]["SparseMlaCombine_v1"]["priced"] == ["measured"]
    assert H.assemble(row(b), b, CALLS, STRATA, CALLS[0][1])["split"]["X"] == 8.0
    with pytest.raises(ValueError, match="not the row's"):
        H.assemble(row(a), b, CALLS, STRATA, CALLS[0][1])


def test_a_row_carries_what_the_command_selects_it_by():
    """`verity bench exfiltration` takes a run of record's rows by world, row and a step's prefix, at the pinned beta
    it reads as `b10`: a row without them is one the command can't read."""
    p = plan()
    got = H.assemble(row(p), p, CALLS, STRATA, CALLS[0][1])
    assert (got["world"], got["b10"]) == ("ours", p["beta"])
    assert H.STEP.startswith("+ the verifier as measured")


def test_the_pricing_split_is_the_row_s_last_call():
    """Each family's share of the proving seconds a token, X, from the strata the row's last call priced."""
    s = H.pricing_split(CALLS[0][1] + [{"block": "Gemm_v2{M=1}", "per_token": 1.0, "seconds_per_block": 2.0,
                                        "fixed_seconds": 0.0, "statement_blocks": 1, "k_log": 20,
                                        "priced": "modeled"}])
    assert s["X"] == 2.0 * (1.0 + 4.0 / 4) + 2.0
    assert s["families"]["SparseMlaCombine_v1"]["share"] == 4.0 / 6.0
    assert s["families"]["Gemm_v2"]["k_log"] == ["20"]


def test_the_figure_reads_the_10_percent_point():
    """On rows whose bound is inverse in beta and whose compile is in proportion to it, the 10% point is where
    fixed + beta (1 + c) = 0.1, at the bound's bits there."""
    fixed, c, k = 0.031, 0.001, 4.0e12
    rows = [{"row": "(i) honest", "scope": "world year", "step": H.STEP, "beta": b, "bound_bits": k / b,
             "compile": c * b, "serve": 0.03, "edge": 0.0004, "pool": 0.0006, "replay_sample": 1.0, "q": 1e-16,
             "requests": 1e16, "S": H.SERVED_BITS, "delta_seen": 0.01, "share": H.SHARE, "gpu_seconds": 1.0,
             "weights_bits": 3.0e12} for b in (0.05, 0.1, 0.4, 1.0)]
    fig = H.read(rows)["(i) honest | world year"]
    beta = (0.1 - fixed) / (1 + c)
    assert isclose(fig["gb10"], k / beta / 8e9, rel_tol=1e-9)
    assert isclose(fig["e"], -1.0, rel_tol=1e-12) and fig["resid"] < 1e-12
    assert isclose(fig["FP4 l1"]["beta"], k / 3.0e12, rel_tol=1e-12)
    with pytest.raises(ValueError, match="rows with errors"):
        H.read(rows + [rows[0] | {"beta": 2.0, "error": "boom"}])


@pytest.mark.slow
def test_a_planner_row_describes_itself_and_restores_the_planner():
    """One row on the planner (about 6 s): its split is its own bound's plan, its requests its line's (not the
    census's), its heads eight, its verification phase a datacenter's window with a whole count of its replay blocks
    sampled, a measured block's table the proof blocks its k_log holds, and every name and table the
    row's path set is as it was after it; a beta that pays for no sampled replay block of the phase is refused.
    Catches a planner change that renames or rebinds what the driver sets, which the plain dicts above can't."""
    P = H.planner()
    names = [(P.reprice, "apply_server"), (P.reprice, "K_MAX"), (P.sweep, "make_line"), (P.f1, "model"),
             (P.tc, "two_stage_design")]
    before = [getattr(o, n) for o, n in names]
    tables = (dict(P.reprice.MARGINAL), dict(P.reprice.CACHE))
    got, p = H.run(H.spec_of("(i) honest", "world year", 0.4), P=P)
    assert [getattr(o, n) for o, n in names] == before
    assert (P.reprice.MARGINAL, P.reprice.CACHE) == tables
    assert got["heads"] == 8 and got["volume"] == H.volume()
    assert (got["split"]["q"], got["split"]["replay_sample"]) == (p["q"], got["replay_sample"])
    assert isclose(got["requests"], p["replay_sample"] / p["q"], rel_tol=1e-9)
    assert got["requests"] < got["census_scope"]["requests"]
    assert got["total"] == got["serve"] + got["edge"] + got["beta"] + got["compile"] + got["pool"]
    phase = got["verification_phase"]
    assert phase["name"] == "datacenter" and phase["machines"] > 1 and phase["seconds"] == H.RUN_S
    assert isinstance(phase["sampled"], int) and 1 <= phase["sampled"] <= phase["replay_blocks"]
    assert p["replay_sample"] == pytest.approx(phase["sampled"] * got["requests"] / phase["replay_blocks"])
    assert phase["sessions"] > len(p["cells"]) and phase["limit"]
    measured = {b["block"]: b for b in cards.load("proof_overhead", "GLM-5.3")["value"]["blocks"]}
    held = [c for c in p["cells"] if c["block"] in measured]
    assert held and all(c["table"] == P.tc.table_blocks(measured[c["block"]]["k_log"]) for c in held)
    with pytest.raises(ValueError, match="pays for no sampled replay block"):
        H.run(H.spec_of("(i) honest", "world year", 1e-6), P=P)
    assert [getattr(o, n) for o, n in names] == before


QUANT = "Fp8GroupQuantUe8m0_v1{K=2048,G=128}"
#: a line of two blocks as the planner hands them to its pricing, and the steady-state rows a census names them by
LINE = [{"block": b, "per_token": 1.0, "seconds_per_block": 1.0, "fixed_seconds": 8.0, "statement_blocks": 4,
         "ands": 10, "k_log": 24, "priced": "measured", "macs": 0} for b in (QUANT, COMBINE)]
MEASURED = {QUANT: {"block": QUANT, "statement_seconds": 4.0, "blocks": 8, "provers": 1, "record": "art:quant"},
            "Combine_v9{DV=8}": {"block": "Combine_v9{DV=8}", "statement_seconds": 16.0, "blocks": 64, "provers": 1,
                                 "record": "art:combine"},
            "Gone_v1{N=1}": {"block": "Gone_v1{N=1}", "statement_seconds": 1.0, "blocks": 4, "provers": 1,
                             "record": "art:gone"}}


def planner_price(strata, steady=None, verify=0.0, **_kw):
    """Stands for `reprice.apply_server`: a block with a steady-state row at its row's seconds with no fixed seconds,
    every other as it came, and the verifier's 2 s a statement where asked."""
    out = []
    for x in strata:
        if r := (steady or {}).get(x["block"]):
            x = x | {"seconds_per_block": r["statement_seconds"] / r["blocks"], "fixed_seconds": 0.0,
                     "statement_blocks": r["blocks"], "priced": "steady", "basis": f"steady: {r['record']}"}
        else:
            x = x | {"basis": "reprice.CACHE"}
        out.append(x | {"fixed_seconds": x["fixed_seconds"] + 2.0 * verify})
    return out


def test_a_steady_row_prices_the_line_s_block_its_map_names():
    """A row prices the line's block of its own name; the map says where the names differ, cuts a row of one of a
    block's n column blocks to the block, and lists a row mapped to null as unread. A stale entry, a statement that
    isn't whole blocks of its target and two rows for one block are refused."""
    by, unread = H.placed(MEASURED, {"Combine_v9{DV=8}": [COMBINE, 8], "Gone_v1{N=1}": None})
    assert list(by) == [QUANT, COMBINE] and unread == ["Gone_v1{N=1}"]
    assert by[QUANT] == MEASURED[QUANT] | {"type": QUANT, "per_block": 1}
    assert by[COMBINE] == {"block": COMBINE, "type": "Combine_v9{DV=8}", "statement_seconds": 16.0, "blocks": 8,
                           "per_block": 8, "provers": 1, "record": "art:combine"}
    assert H.placed(MEASURED, {"Gone_v1{N=1}": PAGES})[0][PAGES]["type"] == "Gone_v1{N=1}"
    with pytest.raises(ValueError, match=r"names 1 block types no steady-state row measures: \['Other_v1'\]"):
        H.placed(MEASURED, {"Other_v1": None})
    with pytest.raises(ValueError, match="a statement of 4 blocks isn't whole blocks of `PageCopy"):
        H.placed(MEASURED, {"Gone_v1{N=1}": [PAGES, 3]})
    with pytest.raises(ValueError, match="takes two steady-state rows, `Fp8GroupQuant.*` and `Gone_v1.N=1.`: map one to null"):
        H.placed(MEASURED, {"Gone_v1{N=1}": QUANT})


def test_a_block_with_a_steady_row_costs_what_the_row_measured():
    """Priced by the planner alone, the quant type takes the host walk and the combine the row's scale. With a
    steady-state row each costs its row's seconds a block and the measured verifier, and neither: the row timed the
    whole statement, and a lever on it would rescale a measurement."""
    scale, core = {"SparseMlaCombine_v1{": {"marg": 4.0, "fixed_x": 1.0}}, 2.0
    plain = H.pricing(planner_price, scale, core, [])(LINE, verify=1.0)
    assert plain[0]["seconds_per_block"] == 1.0 + H.QUANT_WALK + H.V_OTHER[1] / core
    assert plain[1]["seconds_per_block"] == 4.0 * (1.0 + H.V_OTHER[1] / core) and plain[1]["priced"] == "measured"
    rows, _ = H.placed(MEASURED, {"Combine_v9{DV=8}": [COMBINE, 8], "Gone_v1{N=1}": None})
    calls = []
    got = H.pricing(planner_price, scale, core, calls, rows)(LINE, verify=1.0)
    assert [x["seconds_per_block"] for x in got] == [0.5 + H.V_OTHER[1] / core, 2.0 + H.V_OTHER[1] / core]
    assert [x["fixed_seconds"] for x in got] == [2.0 * H.V_OTHER[0]] * 2 and [x["statement_blocks"] for x in got] == [8, 8]
    assert [x["priced"] for x in got] == ["steady", "steady"]
    assert [x["basis"] for x in calls[-1][1]] == ["steady: art:quant", "steady: art:combine"]


def test_a_steady_row_that_prices_no_block_of_the_line_is_refused_by_name():
    """A row whose block, its own name or its map's, is none of the line's is no silent drop: the pricing refuses it
    with the census type, the block it was mapped to and the fix."""
    rows, _ = H.placed(MEASURED, {"Combine_v9{DV=8}": "Combine_v1{DV=512}"})
    with pytest.raises(ValueError) as refused:
        H.pricing(planner_price, {}, 2.0, [], rows)(LINE)
    assert str(refused.value) == (
        "2 steady-state rows price no block of this line: `Combine_v9{DV=8}` (as `Combine_v1{DV=512}`); `Gone_v1{N=1}`. "
        "Map each to the line's block it measures, or to null to list it unread (--steady-map)")


#: the two fused reads' types as a census names them, and their steady-state rows
W13 = ('MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=3072}"},K=6144,N=1,G=128,KB=128,SWAP=false,'
       'DOT={"fn":"HopperE4m3MmaV2Dot32_v2"}}')
W2 = ('MoeExpertFp8ReadGemmW_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=4096}"},K=256,N=32,G=128,KB=128,SWAP=true,'
      'DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}')
FUSED_ROWS = {n: {"block": n, "statement_seconds": 2.0, "blocks": 1024, "provers": 1, "record": f"art:{t}"}
              for t, n in (("w13", W13), ("w2", W2))}


def fused_line(P):
    """The proof server's line as the planner builds it before pricing: the card's per-expert strata, reads fused."""
    case = P.sweep.served_line()["P5+poolwin600"]
    assert case.fuse
    return P.levers.fuse_reads(P.levers.priced_strata(case.line))


def test_a_fused_read_s_row_prices_the_line_s_fused_block_with_no_map():
    """A census holds a fused read as one type; the line names the block by its read and its GEMM. With no map each
    of the two rows prices its fused block of the planner's own line, none is unread, and the pricing takes them; a
    map entry still decides, and a type that isn't a fused read's keeps its own name."""
    line = fused_line(H.planner())
    fused = [x["block"] for x in line if x["kind"] == "fused"]
    by, unread = H.placed(FUSED_ROWS, {})
    assert sorted(by) == sorted(fused) and len(fused) == 2 and unread == []
    assert {r["type"]: b for b, r in by.items()} == {W13: H.fused_block(W13), W2: H.fused_block(W2)}
    assert by[H.fused_block(W13)] == FUSED_ROWS[W13] | {"block": H.fused_block(W13), "type": W13, "per_block": 1}
    got = H.pricing(planner_price, {}, 2.0, [], by)(line)
    assert sorted(x["block"] for x in got if x["priced"] == "steady") == sorted(fused)
    assert [x["basis"] for x in got if x["block"] == H.fused_block(W2)] == ["steady: art:w2"]
    assert H.placed(FUSED_ROWS, {W13: None}) == ({H.fused_block(W2): by[H.fused_block(W2)]}, [W13])
    assert H.fused_block(QUANT) is None and H.fused_block(W13.replace("H=3072", "H=8")) is None
    assert list(H.placed(MEASURED, {})[0]) == list(MEASURED)


def test_beside_the_fused_reads_a_row_that_names_nothing_is_still_refused_by_name():
    """The fused reads' rows placed with no map, a row for no block of the line still stops the pricing by its name;
    and on a line with no fused block a fused read's row is refused with the block it was read as."""
    line = fused_line(H.planner())
    rows, _ = H.placed(FUSED_ROWS | {"Gone_v1{N=1}": MEASURED["Gone_v1{N=1}"]}, {})
    with pytest.raises(ValueError) as refused:
        H.pricing(planner_price, {}, 2.0, [], rows)(line)
    assert str(refused.value) == (
        "1 steady-state rows price no block of this line: `Gone_v1{N=1}`. "
        "Map each to the line's block it measures, or to null to list it unread (--steady-map)")
    with pytest.raises(ValueError) as refused:
        H.pricing(planner_price, {}, 2.0, [], H.placed({W2: FUSED_ROWS[W2]}, {})[0])(LINE)
    assert str(refused.value) == (
        f"1 steady-state rows price no block of this line: `{W2}` (as `{H.fused_block(W2)}`). "
        "Map each to the line's block it measures, or to null to list it unread (--steady-map)")


def test_a_row_s_split_gives_each_basis_s_share():
    """The plan's proving by what priced it: every steady-state row's blocks under `steady`, the rest under the
    planner's table, a column cut's note left off, and the carried pages under the copy stratum's own model."""
    p = plan()
    p["cells"].append({"block": QUANT, "share": 0.5, "prove_s": 2.0, "taken": 1.0})
    calls = [(True, [CALLS[0][1][0] | {"priced": "steady", "basis": "steady: art:combine"},
                     LINE[0] | {"basis": "reprice.CACHE, cut into 32 column blocks"}])]
    strata = [{"block": PAGES, "priced": "modeled", "basis": H.COPY_BASIS}]
    got = H.plan_split(p, calls, strata)
    assert got["priced"] == {"steady": 0.75, "measured": 0.5, "modeled": 0.25}
    assert got["basis"] == {"steady": 0.75, "reprice.CACHE": 0.5, H.COPY_BASIS: 0.25}
    assert H.plan_split(plan(), CALLS, STRATA)["basis"] == {H.NO_BASIS: 1.0}


def test_a_table_row_s_block_is_tagged_by_its_source_and_its_verifier_seconds_apart():
    """A block a table row prices is tagged as the row's source names it (`steady`, `family`), not by the planner's
    word; its Lean verifier's seconds, its share of the cell's statements and of its blocks, are their own share under
    `verifier_priced` and the Lean verifier's source; a block the planner prices keeps the planner's word, `computed`
    included, for the rest. The split's shares still sum to the plan's."""
    p = plan()
    p["cells"][0] |= {"fixed_s": 2.0, "sessions": 0.5}
    p["cells"].append({"block": QUANT, "share": 0.5, "prove_s": 2.0, "taken": 1.0, "fixed_s": 1.0, "sessions": 1.0})
    p["cells"].append({"block": "Gemm_v2{M=1}", "share": 0.25, "prove_s": 1.0, "taken": 1.0, "fixed_s": 1.0,
                       "sessions": 1.0})
    calls = [(True, [CALLS[0][1][0] | {"table": "steady", "verifier_priced": "modeled", "verifier_fixed": 0.5,
                                       "verifier_block": 0.0},
                     LINE[0] | {"priced": "computed", "verifier_priced": "modeled", "verifier_fixed": 0.25,
                                "verifier_block": 0.1},
                     LINE[0] | {"block": "Gemm_v2{M=1}", "table": "family", "verifier_priced": "measured",
                                "verifier_fixed": 0.0, "verifier_block": 0.0}])]
    got = H.plan_split(p, calls, [{"block": PAGES, "priced": "modeled", "basis": H.COPY_BASIS}])
    combine, quant = 0.75 * 0.5 / 3.0, 0.5 * (0.25 + 0.1) / 2.0
    assert got["priced"] == pytest.approx({"steady": 0.75 - combine, "modeled": combine + quant + 0.25,
                                           "computed": 0.5 - quant, "family": 0.25})
    assert got["sources"] == pytest.approx({"own row": 0.75 - combine, H.VERIFIER_SOURCE: combine + quant,
                                            H.PLANNER: 0.5 - quant, "family row": 0.25, H.COPY_BASIS: 0.25})
    assert got["families"]["SparseMlaCombine_v1"]["priced"] == ["modeled", "steady"]
    assert got["families"]["SparseMlaCombine_v1"]["ways"] == pytest.approx({"steady": 0.75 - combine, "modeled": combine})
    assert got["families"]["Gemm_v2"]["ways"] == {"family": 0.25}
    assert isclose(sum(got["priced"].values()), 1.75) and isclose(sum(got["sources"].values()), 1.75)
    assert got["basis"] == {"steady": 0.75, H.NO_BASIS: 0.5, "family": 0.25, H.COPY_BASIS: 0.25}


def test_each_reader_says_what_its_gb_is():
    """`plan` says its GB is the bound at its own pinned budget, with the share of compute its total is there and
    lambda; `rows` and `read` say their column is lambda times the bound where the total is 10%."""
    r = row(plan()) | {"bound_bits": 8e12, "total": 0.10105001369288587}
    assert H.show_what(r) == ("1,000.0 GB: the bound at the pinned budget beta = 40% of compute, where the row's total "
                              "is 10.11% (not read at 10%), at lambda = 1; x1.05 is lambda times it")
    assert H.show({}).splitlines()[-1].startswith("GB@10% x1.05: lambda = 1.05 times the bound where the scope's total "
                                                  "is 10% of its compute")


def test_a_block_with_neither_its_own_nor_its_family_s_row_keeps_the_planner_s_price():
    """At `verity bench prove`'s table, a block whose row is a steady-state one, its own or its family's, costs the
    row's seconds a block: its price without the verifier (the row's scale in it) times one factor, the verifier's
    seconds a statement on top. A block whose row is from a single session or the k_log model, or that has no row,
    keeps the planner's price as the row scales it. A table's row for no block of the line is refused by name."""
    scale, core = {"SparseMlaCombine_v1{": {"marg": 4.0, "fixed_x": 1.0}, "Gemm_v2{": 2.0}, 2.0
    gemm = LINE[1] | {"block": "Gemm_v2{M=1}"}
    line = LINE + [gemm]
    plain = H.pricing(planner_price, scale, core, [])(line, verify=1.0)
    table = {COMBINE: {"source": "family", "seconds": 3.0, "record": "art:combine"},
             QUANT: {"source": "single", "seconds": 9.0, "record": None}}
    calls = []
    got = H.pricing(planner_price, scale, core, calls, table=table)(line, verify=1.0)
    assert got[0] == plain[0] and got[2] == plain[2] and plain[2]["seconds_per_block"] == 2.0 * (1.0 + H.V_OTHER[1] / core)
    assert got[1]["seconds_per_block"] == 0.5 * 4.0 and isclose(got[1]["fixed_seconds"], 4.0 + 2.0 * H.V_OTHER[0])
    assert isclose(got[1]["seconds_per_block"] + (got[1]["fixed_seconds"] - 2.0 * H.V_OTHER[0]) / 4, 3.0)
    assert (got[1]["priced"], got[1]["table"], got[1]["basis"]) == ("measured", "family", "family: art:combine")
    assert [x["table"] for x in calls[-1][1]] == [None, "family", None]
    # the verifier's seconds are the row's apart: all of its seconds a statement over the table's row, none a block
    assert [x["verifier_priced"] for x in calls[-1][1]] == ["modeled"] * 3
    assert isclose(got[1]["verifier_fixed"] * got[1]["fixed_seconds"], 2.0 * H.V_OTHER[0]) and got[1]["verifier_block"] == 0
    assert isclose(plain[2]["verifier_block"] * plain[2]["seconds_per_block"], 2.0 * H.V_OTHER[1] / core)
    assert isclose(plain[2]["verifier_fixed"] * plain[2]["fixed_seconds"], 2.0 * 2.0 * H.V_OTHER[0])
    with pytest.raises(ValueError) as refused:
        H.pricing(planner_price, {}, core, [], table=table | {"Gone_v1{N=1}": table[COMBINE]})(line)
    assert str(refused.value) == ("1 rows of the table price no block of this line: `Gone_v1{N=1}`. It is another "
                                  "cut's table, or the line names those blocks otherwise (`FUSED`)")
    doc = {"cores": 10, "rows": [{"block": W13, "source": "steady", "seconds": 1.0}]}
    assert list(H.prices(doc)["cells"]) == [H.fused_block(W13)] and H.prices(doc)["cores"] == 10
    with pytest.raises(ValueError, match="steady-state rows or at a table, not both"):
        H.spec_of("(ii) honest", "world year", 0.4, steady={"rows": {}}, prices=H.prices(doc))


CONTROL = Path(__file__).parent / "fixtures" / "control"
#: the control: `verity bench prove`'s table it prices row (ii) at, and the planner's inputs of the plan it must give
CONTROL_SHA256 = {"bench_prices.json": "f2badd5ecdf36530b918b34782fec63466b8586cd6df55e46c43af442af9604c",
                  "plan_inputs.json": "cb8f1733d189228657f96342c359a8b230078c8760066b443559862bd44d0c80"}
#: the control's plan's budget
PLAN_BETA = 0.05544106123672309


def test_row_ii_at_the_control_s_table_is_the_plan_s():
    """Row (ii) at the control's table and the plan's budget (about 3 s): its bound is the plan's to 1e-12, its 193
    blocks are the plan's in order, every planner input within 1e-9 of the plan's, and its plan's proving is the
    table's own rows', its family rows', the Lean verifier's and the planner's price's in their shares, each tagged as
    its source prices it, with `computed` only on the planner's price. Catches a change to the pricing rule, the
    planner or the card that moves the figure or its split, which the stand-in pricing above can't."""
    for name, digest in CONTROL_SHA256.items():
        assert hashlib.sha256((CONTROL / name).read_bytes()).hexdigest() == digest, f"fixtures/control/{name}"
    want = json.loads((CONTROL / "plan_inputs.json").read_text())
    P = H.planner()
    spec = H.spec_of("(ii) honest", "world year", PLAN_BETA,
                     prices=H.prices(json.loads((CONTROL / "bench_prices.json").read_text())))
    row, reads = H.evaluate(spec, P)
    p = H.plan(spec, row, reads, {}, P)
    got = H.assemble(row, p, reads["calls"], reads["strata"], reads["pricing"])
    assert isclose(got["bound_bits"] / 8e9, 2635.9320571029266, rel_tol=1e-12)
    assert isclose(got["total"], 0.10086052035955254, rel_tol=1e-12)
    assert len(p["cells"]) + len(p["left_out"]) == 193
    assert [x["block"] for x in reads["strata"]] == [x["block"] for x in want["strata"]]
    cells = [astuple(c) for c in reads["design"]["cells"]]
    assert len(cells) == len(want["cells"]) == 193
    off = [(want["strata"][i]["block"], f, u, v) for i, (a, b) in enumerate(zip(want["cells"], cells))
           for f, u, v in zip(want["cell_fields"], a, b) if not isclose(u, v, rel_tol=1e-9)]
    assert off == []
    assert got["prices"] == {"read": None, "cores": None, "rows": 192, "above_cores": {}}
    sources, priced = got["split"]["sources"], got["split"]["priced"]
    assert list(sources) == ["own row", H.VERIFIER_SOURCE, "family row", H.PLANNER, H.COPY_BASIS]
    assert [round(v, 6) for v in sources.values()] == [0.787037, 0.209103, 0.003741, 0.000117, 1e-06]
    assert {k: round(v, 6) for k, v in priced.items()} == {"steady": 0.787037, "modeled": 0.206741, "measured": 0.00245,
                                                           "family": 0.003741, "computed": 3.1e-05}
    assert isclose(sum(priced.values()), 1.0, rel_tol=1e-12) and isclose(sum(sources.values()), 1.0, rel_tol=1e-12)
    assert (priced["steady"], priced["family"]) == (sources["own row"], sources["family row"])
    assert priced["computed"] < sources[H.PLANNER]


@pytest.mark.slow
def test_a_planner_row_is_priced_at_a_steady_row_through_its_map(tmp_path):
    """Two rows on the planner (about 12 s): with a steady-state row mapped to the 64-key head block, the planner
    prices that block at the row (the row's seconds a block and the measured verifier's, in statements of the row's
    blocks; its plan's proving says `steady` for it and the bound falls with the cheaper block), every other block as
    without it, and a row for no block of the line stops the row. Catches the planner's own
    pricing no longer taking a row by the line's block, which the stand-in pricing above can't."""
    P = H.planner()
    part = next(x["block"] for x in P.levers.priced_strata("tokens")
                if x["block"].startswith("SparseMlaHeadBlock_v1{") and "NVIS=64,FIRST=false" in x["block"])
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": [{"block": "Head_v1{NVIS=64}", "statement_seconds": 0.128, "blocks": 128,
                                          "provers": 1, "record": "art:head"}]}))
    where = tmp_path / "map.json"
    where.write_text(json.dumps({"Head_v1{NVIS=64}": part}))
    base, _ = H.run(H.spec_of("(i) honest", "world year", 0.4), P=P)
    spec = H.spec_of("(i) honest", "world year", 0.4, steady=H.steady(rows, where, P))
    row, reads = H.evaluate(spec, P)
    p = H.plan(spec, row, reads, {}, P)
    got = H.assemble(row, p, reads["calls"], reads["strata"], reads["pricing"])
    assert "steady" not in base and "steady" not in base["split"]["priced"]
    assert got["steady"] == {"priced": [{"block": part, "type": "Head_v1{NVIS=64}", "per_block": 1, "provers": 1,
                                         "record": "art:head"}], "unread": []}
    priced = next(x for x in reads["pricing"] if x["block"] == part)
    card = next(x for x in P.levers.priced_strata("tokens") if x["block"] == part)
    assert card["statement_blocks"] != 128 and card["priced"] != "steady"
    assert (priced["priced"], priced["basis"], priced["statement_blocks"]) == ("steady", "steady: art:head", 128)
    assert isclose(priced["seconds_per_block"], 0.128 / 128 + H.V_OTHER[1] / P.reprice.CORE_PER_GPU, rel_tol=1e-12)
    cell = next(c for c in p["cells"] if c["block"] == part)
    statements = cell["sessions"] * cell["fixed_s"]
    lean = (statements * priced["verifier_fixed"] + (cell["prove_s"] - statements) * priced["verifier_block"]) / cell["prove_s"]
    assert priced["verifier_priced"] == "modeled" and 0 < lean < 1
    assert isclose(got["split"]["priced"]["steady"], cell["share"] * (1 - lean), rel_tol=1e-12)
    assert isclose(got["split"]["basis"]["steady"], cell["share"], rel_tol=1e-12)
    assert isclose(sum(got["split"]["basis"].values()), 1.0, rel_tol=1e-9) and H.NO_BASIS not in got["split"]["basis"]
    assert got["bound_bits"] < base["bound_bits"]
    with pytest.raises(ValueError, match="1 steady-state rows price no block of this line: `Head_v1.NVIS=64.`"):
        H.run(H.spec_of("(i) honest", "world year", 0.4, steady=H.steady(rows, None, P)), P=P)
