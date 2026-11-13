"""GLM-5.3's proof-block card (`glm53.py` over `glm53_blocks.json`): it names its census and the Build by id; every
proof-block Definition is a planner candidate whose outputs fit a size class, its units one request's count times the
requests, its work its own ANDs and a statement at most `cap` of it; no served row is a gap; the totals per token and per
request are the Definitions' sums; the planner certifies a sample over the card's candidates; and `main` prints the totals and
each gap's rows, ANDs and why. A stop class adds one stop block and one response tie a response, each a candidate whose
ANDs, rows and output bits are its Definition's; every block of a population sampled leaves no block incorrect, no ε
and no bit charged at any δ, a smaller sample is charged its count bound's blocks' output bits, and the planner's
certificate over the two populations agrees; and `main --stop` prints both populations with what proving every block leaves."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import glm53 as G
from verity.commitments.windows.interface import proof_block_k_log
from verity.definitions.registry import boolean_stop_rule
from verity.protocols.certifier.prover import response_tie
from verity.protocols.spot.optimize import Objective, certify

CARD = G.load()
ART = re.compile(r"art:[0-9a-f]{64}")


def _pins(module) -> dict:
    """The ANDs circuit-check pins for a module's Definitions."""
    return json.loads(Path(module.__file__).with_suffix(".circuit-check.json").read_text())["definitions"]


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
    assert CARD["source"] in out and "decode token" in out and "stop class" not in out
    assert "gap TokenSelect_v1: 1 Definitions, 8 prefill and 248 decode rows, 6.899e+09 ANDs; its rows alone need" in out


def test_a_stop_class_adds_one_stop_block_and_one_response_tie_a_response():
    stop, tie = G.response_blocks(16, 5)
    assert (stop.template, tie.template) == ("StopRule16_v2{T=16,NSTOP=3}", "ResponseTieStop_v1{O=16}")
    for c, inputs in ((stop, 5), (tie, 3)):
        assert (c.op, c.units, c.n_rho, c.grain) == (c.template, 5, 1, "request")
        assert c.work == c.ands > 0 and c.out_bits == c.commit_bits and c.out_gates == c.out_bits // 16
        k_log = proof_block_k_log(c.rows[:inputs], c.rows[inputs:], c.ands, packed=False)
        assert c == G._candidate(c.template, 5, 1, c.ands, c.out_bits, c.rows, k_log), "a statement holds its block's count"
    assert (stop.out_bits, stop.rows) == (16, (4 * 16, 4, 4 * 3, 4, 4, 2))
    assert (tie.out_bits, tie.rows) == (16 + 32 * 17, (4 * 16, 2, 4, 2, 4 * 17))
    assert tie.ands == _pins(response_tie)[tie.template]["and"] == response_tie.Shape(16, True).ands
    pinned = G.response_blocks(4, stop_ids=1)[0]
    assert pinned.ands == _pins(boolean_stop_rule)[pinned.template]["and"]
    assert [c.units for c in G.response_blocks(16)] == [1, 1]


@pytest.mark.parametrize("delta", [1e-6, 2.0 ** -64])
def test_every_block_sampled_leaves_nothing_charged_and_a_smaller_sample_is_charged(delta):
    cs = G.response_blocks(16, 1540)
    for c in cs:
        assert G.charged(c, delta) == {"population": 1540, "sampled": 1540, "rate": "1/1", "incorrect": 0, "eps": 0.0,
                                       "bits": 0}
        x = G.charged(c, delta, 154)
        assert (x["population"], x["sampled"], x["rate"]) == (1540, 154, "1/10") and 0 < x["incorrect"] <= 1540 - 154
        assert x["eps"] == x["incorrect"] / 1540 and x["bits"] == x["incorrect"] * c.out_bits
        assert G.charged(c, delta, 1539)["incorrect"] == 1
    with pytest.raises(ValueError):
        G.charged(cs[0], delta, 1541)
    every = certify({c.template: c.units for c in cs}, cs, Objective(delta, work=1, bits=1))
    assert every.leak < 1 and every.free_compute < 1
    tenth = certify({c.template: 154 for c in cs}, cs, Objective(delta, work=1, bits=1))
    assert tenth.leak >= max(G.charged(c, delta, 154)["bits"] for c in cs)


def test_main_prints_a_stop_classs_populations_and_what_proving_every_block_leaves(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glm53.py", "--requests", "7", "--stop", "16", "--delta", "1e-9"])
    assert G.main() == 0
    out = capsys.readouterr().out
    stop, tie = G.response_blocks(16)
    assert f"stop class StopRule16_v2{{T=16,NSTOP=3}}: 7 proof block(s), one a response, {stop.ands:,} ANDs and 16 " in out
    assert f"stop class ResponseTieStop_v1{{O=16}}: 7 proof block(s), one a response, {tie.ands:,} ANDs and 560 " in out
    assert out.count("rate:1/1 leaves at most 0 incorrect at delta 1e-09 (eps 0), 0 bits charged") == 2
