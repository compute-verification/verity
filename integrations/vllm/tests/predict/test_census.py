"""census/v1 and the requirement (`verity.ml.circuits.census`, `.requirement`) of a real row's Programs: Llama-3.2-1B B1 256/32
greedy on rtxpro6000 (cov-k05-7), its step (the traced Build `test_predict` pins) and its request Program (a prefill of 256 rows,
then 31 decode steps), each step ending at a sampling event; and the census path held to the explicit expansion on `TINY`'s.
A change to a Definition either Program calls moves the census digests with the Program's: recompute them with `_views`.
The requirement takes the weights as registered (`Program(fn, registered=("weights",))`), whatever the Build binds them as.
"""

from __future__ import annotations

import json

import pytest

from tests.predict.test_predict import FIX, TINY, _phi
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.program import Program
from verity.ml.circuits.requirement import census_gates, census_schedule, from_census, requirement
from verity_vllm.predict.build import build_program, model_shape
from verity_vllm.query.required import SAMPLING_EVENTS

CFG = json.loads((FIX / "llama32-1b.config.json").read_text())
C = model_shape(CFG)
#: a token row's linear MACs: q/k/v, o, gate/up and down per layer, and the LM head
LINEAR = C.NL * C.H * ((C.NH + 2 * C.KVH) * C.D + C.NH * C.D + 3 * C.I) + C.H * C.V
#: one position's K and V over every layer, in bf16
KV_BITS = C.NL * 2 * C.KVH * C.D * 16
#: one position's row of the rotary cos/sin table, in bf16
ROW = C.rot * 16


def _base(rc: dict) -> int:
    """The weights every step reads: all but the rotary table (a step reads its positions' rows) and each layer's four unused
    attention scales (fp32)."""
    return rc["totals"]["inputs"]["registration"]["bits"] - C.max_pos * ROW - C.NL * 4 * 32


def _registered(r) -> tuple[Program, dict]:
    """The Build's Program with its weights registered, and its census at the sampling steps."""
    P = Program(r.program.fn, registered=("weights",))
    return P, census(P, step_ends(P, SAMPLING_EVENTS))


def _views(cfg: dict, wrapper: str, **kw) -> tuple:
    r = build_program(_phi(cfg), wrapper, **kw)
    P, c = _registered(r)
    return r, census(r.program, step_ends(r.program, SAMPLING_EVENTS), r.digest), from_census(c), c


def test_step_census_and_requirement():
    r, c, req, rc = _views(CFG, "step", M=1, pos0=8)
    assert r.digest.startswith("773b545605765fe9")
    assert c["digest"] == "461f379c3ef8f43fdd226b5f81fc44a159ddc5d7d8f5dfdfb9599b973b849a9b0223d13c5f549ec6691b511209cd65f13213cef9e6f596999c8c13ccbaa3e5ab"
    gemms = [e for e in c["classes"].values() if e["family"] == "Gemm_v2"]
    assert all(e["served_macs"] == e["statics"]["K"] * e["statics"]["N"] for e in gemms)
    assert sum(e["instances"] * e["served_macs"] for e in gemms) == LINEAR
    attn = [e for e in c["classes"].values() if e["family"] == "Attention_v5"]
    assert c["totals"]["served_macs"] == LINEAR + sum(e["instances"] * e["served_macs"] for e in attn)
    assert (c["totals"]["classes"], c["totals"]["values"], c["totals"]["tail_atoms"], c["totals"]["receipt_folds"]) == (57, 394, 0, 0)
    assert c["selects"]["gates"] == [2048] and c["selects"]["bits"] == [2048 * 16]  # the row's embedding: H bf16 gathers
    assert rc["totals"]["inputs"]["run"] == {"gates": 1, "bits": 32}  # the token id
    w = _base(rc) + ROW
    assert req.to_json() == {"steps": 1, "live_bits": [w], "read_bits": [w], "registered_bits": w}
    assert c["selects"]["values"] and req.resolved(0)  # the embedding's gathers select from the table the tied LM head reads whole


@pytest.mark.slow
def test_request_census_and_requirement():
    """The row's request Program, about 30 s: the only test of a Program with values across steps (each layer's K and V rows a
    step's attention produces and every later step's reads), which the step and the toys in `ml/circuits` cannot show at scale."""
    r, c, req, rc = _views(CFG, "request", LP=256, T=31)
    assert r.digest.startswith("2d98d37bd481d2a0")
    assert c["digest"] == "1809b72d85282d2d2552e4ba133df8ed4fd0ff1b9f99e291ca34cc9e9475da2ab59259ecd584392e22b26bd9add6bad261a51ab4798ec8d526d351ed80ab0458"
    assert len(c["steps"]["ends"]) == 32 and c["totals"]["classes"] == 761
    base, got = _base(rc), req.to_json()
    # prefill reads rotary rows 0-255 and decode step t row 255 + t; a step holds the rows of its own and later positions,
    # the K/V rows so far and the token sampled before, and reads all it holds but the rows of later positions
    assert got["live_bits"] == [base + 287 * ROW] + [base + (32 - t) * ROW + (255 + t) * KV_BITS + 32 for t in range(1, 32)]
    assert got["read_bits"] == [base + 256 * ROW] + [base + ROW + (255 + t) * KV_BITS + 32 for t in range(1, 32)]


@pytest.mark.parametrize("wrapper,kw", [("step", {"M": 1, "pos0": 8}), ("request", {"LP": 4, "T": 3})])
def test_the_census_path_is_the_expansion_on_tiny(wrapper, kw):
    """`TINY`'s step and a request of four prompt rows and three decode steps (105 k and 181 k gates): `from_census` gives the
    explicit expansion's live gates and bounded reads at every step."""
    P, c = _registered(build_program(_phi(TINY), wrapper, **kw))
    by_census, by_gates = from_census(c), requirement(P, census_schedule(c))
    for t in range(by_gates.steps):
        assert census_gates(P, c, by_census.live(t)) == by_gates.live(t)
        assert census_gates(P, c, by_census.reads_upper(t)) == by_gates.reads_upper(t)
    assert by_census.to_json() == by_gates.to_json()
