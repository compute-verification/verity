"""The figure of record's driver (`breakeven/headline.py`): a row counts every sparse-MLA head, carries its own plan's
request count and split, and leaves the planner as it found it; the figure reads the 10% point off pinned rows."""
from math import isclose

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
    """The 128k workload's per-head sparse-MLA blocks are GLM-5.3's eight heads', not one head's."""
    from verity.definitions.registry.sparse_mla import SPARSE_MLA_GLM53

    assert H.heads() == SPARSE_MLA_GLM53["NH"] == 8
    vol = H.volume()
    assert vol == {p: f * (8 if p in H.PER_HEAD else 1) for p, f in H.MLA128_VOLUME.items()}
    assert H.spec_of("(i) honest", "world year", 0.4)["volume"] == vol


def test_the_head_count_is_read_not_written(monkeypatch):
    """The volume follows the registry's `NH`, as a head count written into the driver would not."""
    from verity.definitions.registry import sparse_mla

    monkeypatch.setitem(sparse_mla.SPARSE_MLA_GLM53, "NH", 3)
    spec = H.spec_of("(iii) honest", "datacenter week", 1.0)
    assert spec["heads"] == 3
    assert spec["volume"]["SparseMlaCombine_v1{"] == 3 * H.MLA128_VOLUME["SparseMlaCombine_v1{"]
    assert spec["volume"]["SparseMlaSplit_v1{"] == H.MLA128_VOLUME["SparseMlaSplit_v1{"]


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
    census's), its heads eight, and every name and table the row's path set is as it was after it. Catches a planner
    change that renames or rebinds what the driver sets, which the plain dicts above can't."""
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
