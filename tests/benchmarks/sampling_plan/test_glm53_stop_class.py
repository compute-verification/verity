"""GLM-5.3's stop class (`glm53.response_blocks`, `glm53.main --stop`) with the vLLM adapter's stop rule and the network
attester's response tie, which SPOT's cost takes from its caller: a stop class adds one stop block and one response tie a response, each a candidate
whose ANDs, rows and output bits are its Definition's; every block of a population sampled leaves no block incorrect, no
ε and no bit charged at any δ, a smaller sample is charged its count bound's blocks' output bits, and the planner's
certificate over the two populations agrees; and `main --stop` prints both populations with what proving every block
leaves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.commitments.periods.interface import proof_block_k_log
from verity.circuits.defs import bind
from verity.protocols.resource_accounting.network.prover import response_tie
from verity.protocols.proofs.spot.prover.cost import glm53 as G
from verity.protocols.proofs.spot.prover.optimize import Objective, certify
from verity_vllm.definitions.registry import boolean_stop_rule


def _pins(module) -> dict:
    """The ANDs circuit-check pins for a module's Definitions."""
    return json.loads(Path(module.__file__).with_suffix(".circuit-check.json").read_text())["definitions"]


def blocks_of(outputs: int, stop_ids: int = G.STOP_IDS):
    """A stop class's stop block at T = ``outputs`` and the network attester's response tie at O = T."""
    return (bind(boolean_stop_rule.StopRule16V2, T=outputs, NSTOP=stop_ids),
            response_tie.block(response_tie.Shape(outputs, True)))


def test_a_stop_class_adds_one_stop_block_and_one_response_tie_a_response():
    stop, tie = G.response_blocks(*blocks_of(16), 5)
    assert (stop.template, tie.template) == ("StopRule16_v2{T=16,NSTOP=3}", "ResponseTieStop_v1{O=16}")
    for c, inputs in ((stop, 5), (tie, 3)):
        assert (c.op, c.units, c.n_rho, c.grain) == (c.template, 5, 1, "request")
        assert c.work == c.ands > 0 and c.out_bits == c.commit_bits and c.out_gates == c.out_bits // 16
        k_log = proof_block_k_log(c.rows[:inputs], c.rows[inputs:], c.ands, packed=False)
        assert c == G._candidate(c.template, 5, 1, c.ands, c.out_bits, c.rows, k_log), "a statement holds its block's count"
    assert (stop.out_bits, stop.rows) == (16, (4 * 16, 4, 4 * 3, 4, 4, 2))
    assert (tie.out_bits, tie.rows) == (16 + 32 * 17, (4 * 16, 2, 4, 2, 4 * 17))
    assert tie.ands == _pins(response_tie)[tie.template]["and"] == response_tie.Shape(16, True).ands
    pinned = G.response_blocks(*blocks_of(4, stop_ids=1))[0]
    assert pinned.ands == _pins(boolean_stop_rule)[pinned.template]["and"]
    assert [c.units for c in G.response_blocks(*blocks_of(16))] == [1, 1]


@pytest.mark.parametrize("delta", [1e-6, 2.0 ** -64])
def test_every_block_sampled_leaves_nothing_charged_and_a_smaller_sample_is_charged(delta):
    cs = G.response_blocks(*blocks_of(16), 1540)
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
    assert G.main(blocks_of) == 0
    out = capsys.readouterr().out
    stop, tie = G.response_blocks(*blocks_of(16))
    assert f"stop class StopRule16_v2{{T=16,NSTOP=3}}: 7 proof block(s), one a response, {stop.ands:,} ANDs and 16 " in out
    assert f"stop class ResponseTieStop_v1{{O=16}}: 7 proof block(s), one a response, {tie.ands:,} ANDs and 560 " in out
    assert out.count("rate:1/1 leaves at most 0 incorrect at delta 1e-09 (eps 0), 0 bits charged") == 2
