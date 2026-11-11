"""GLM-5.3's proof-block card (`glm53.py` over `glm53_blocks.json`): it names its census and the Build by id; every
proof-block Definition is a planner candidate whose outputs fit a size class, its units one request's count times the
requests, its work its own ANDs and a statement at most `cap` of it; no served row is a gap; the totals per token and per
request are the Definitions' sums; the planner certifies a sample over the card's candidates; and `main` prints the totals and
each gap's rows, ANDs and why."""

from __future__ import annotations

import json
import re
import sys

import pytest

import glm53 as G
from verity.protocols.spot.optimize import Objective, certify

CARD = G.load()
ART = re.compile(r"art:[0-9a-f]{64}")


def test_the_card_names_its_census_and_the_build_it_was_taken_from():
    assert ART.fullmatch(CARD["source"]) and ART.fullmatch(CARD["build"])
    assert (CARD["ranks"], CARD["prompt_len"], CARD["decode_steps"]) == (8, 256, 31)


@pytest.mark.parametrize("requests", [1, 3])
def test_every_definition_is_a_candidate_over_the_requests(requests):
    cs = G.candidates(CARD, requests)
    assert [c.template for c in cs] == list(CARD["definitions"]) and len({c.op for c in cs}) == len(cs)
    for c in cs:
        d = CARD["definitions"][c.template]
        per = d["prefill"] + d["decode"]
        assert (c.units, c.n_rho, c.grain) == (requests * per, per, "request")
        assert c.work == c.ands == d["ands"] and c.out_bits == c.commit_bits == d["out_bits"]
        assert c.cap == min(2 ** (35 - d["k_log"]), 2 ** 20) and c.rows == tuple(d["rows"])


def test_no_served_row_is_a_gap():
    assert not CARD["gaps"], f"rows no cut fits a block: {sorted(CARD['gaps'])}; cut their family (its module's CUTS)"


def test_the_totals_are_the_definitions_sums():
    t, ds = G.totals(CARD), CARD["definitions"].values()
    assert t["request"]["blocks"] == sum(d["prefill"] + d["decode"] for d in ds)
    assert t["request"]["out_bits"] == sum((d["prefill"] + d["decode"]) * d["out_bits"] for d in ds)
    assert t["prefill token"]["ands"] * CARD["prompt_len"] == pytest.approx(sum(d["prefill"] * d["ands"] for d in ds))
    assert t["decode token"]["macs"] * CARD["decode_steps"] == pytest.approx(sum(d["decode"] * d["macs"] for d in ds))
    assert t["request"]["row_bytes"] == sum((d["prefill"] + d["decode"]) * sum(d["rows"]) for d in ds)


def test_the_planner_certifies_a_draw_over_the_card():
    cs = G.candidates(CARD)
    b = certify({c.template: min(c.units, 4) for c in cs}, cs, Objective(1e-6, work=1, bits=1), exact=False)
    assert 0 < b.harm < sum(c.units * (c.work + c.out_bits) for c in cs)
    assert 0 < b.free_compute <= sum(c.units * c.work for c in cs)


def test_main_prints_the_totals_and_the_gaps(capsys, monkeypatch, tmp_path):
    gap = {"definitions": 1, "prefill": 8, "decode": 248, "ands": 6898930176, "why": "its rows alone need a 2^28 block"}
    card = tmp_path / "card.json"
    card.write_text(json.dumps(CARD | {"gaps": {"TokenSelect_v1": gap}}))
    monkeypatch.setattr(sys, "argv", ["glm53.py", "--card", str(card), "--requests", "2"])
    assert G.main() == 0
    out = capsys.readouterr().out
    assert CARD["source"] in out and "decode token" in out
    assert "gap TokenSelect_v1: 1 Definitions, 8 prefill and 248 decode rows, 6.899e+09 ANDs; its rows alone need" in out
