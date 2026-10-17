"""census/v1 and memory-map/v1 (`verity.ml.circuits.census`, `.memory_map`) of a real row's Programs: Llama-3.2-1B B1 256/32
greedy on rtxpro6000 (cov-k05-7), its step (the traced Build `test_predict` pins) and its request Program (a prefill of 256 rows,
then 31 decode steps), each step ending at a sampling event.  A change to a Definition either Program calls moves the census and
memory-map digests with the Program's: recompute them with `_views`.
"""

from __future__ import annotations

import json

import pytest

from tests.predict.test_predict import FIX, _phi
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.memory_map import memory_map
from verity_vllm.predict.build import build_program, model_shape
from verity_vllm.query.required import SAMPLING_EVENTS

CFG = json.loads((FIX / "llama32-1b.config.json").read_text())
C = model_shape(CFG)
#: a token row's linear MACs: q/k/v, o, gate/up and down per layer, and the LM head
LINEAR = C.NL * C.H * ((C.NH + 2 * C.KVH) * C.D + C.NH * C.D + 3 * C.I) + C.H * C.V
#: one position's K and V over every layer, in bf16
KV_BITS = C.NL * 2 * C.KVH * C.D * 16


def _views(wrapper: str, **kw) -> tuple:
    r = build_program(_phi(CFG), wrapper, **kw)
    c = census(r.program, step_ends(r.program, SAMPLING_EVENTS), r.digest)
    return r, c, memory_map(c)


def test_step_census_and_memory_map():
    r, c, m = _views("step", M=1, pos0=8)
    assert r.digest.startswith("773b545605765fe9")
    assert c["digest"] == "461f379c3ef8f43fdd226b5f81fc44a159ddc5d7d8f5dfdfb9599b973b849a9b0223d13c5f549ec6691b511209cd65f13213cef9e6f596999c8c13ccbaa3e5ab"
    assert m["digest"] == "2c2d5eb1c3ba79e62b421132f469edf20cc29c1ee9a4b82cf3b38b79b73b37f210a14d3b5dcce1d73db997e907d9c608593e67e107f31fd3f0d02a92dac4cc08"
    gemms = [e for e in c["classes"].values() if e["family"] == "Gemm_v2"]
    assert all(e["served_macs"] == e["statics"]["K"] * e["statics"]["N"] for e in gemms)
    assert sum(e["instances"] * e["served_macs"] for e in gemms) == LINEAR
    attn = [e for e in c["classes"].values() if e["family"] == "Attention_v5"]
    assert c["totals"]["served_macs"] == LINEAR + sum(e["instances"] * e["served_macs"] for e in attn)
    assert (c["totals"]["classes"], c["totals"]["values"], c["totals"]["tail_atoms"], c["totals"]["receipt_folds"]) == (57, 394, 0, 0)
    assert c["selects"]["gates"] == [2048] and c["selects"]["bits"] == [2048 * 16]  # the row's embedding: H bf16 gathers
    assert m["steps"]["count"] == 1 and m["totals"]["born_bits"] == c["totals"]["value_bits"]


@pytest.mark.slow
def test_request_census_and_memory_map():
    """The row's request Program, about 12 s: the only test of a Program with values across steps (each layer's K and V rows a
    step's attention produces and every later step's reads), which the step and the toys in `ml/circuits` cannot show at scale."""
    r, c, m = _views("request", LP=256, T=31)
    assert r.digest.startswith("2d98d37bd481d2a0")
    assert c["digest"] == "1809b72d85282d2d2552e4ba133df8ed4fd0ff1b9f99e291ca34cc9e9475da2ab59259ecd584392e22b26bd9add6bad261a51ab4798ec8d526d351ed80ab0458"
    assert m["digest"] == "83b5adbcb6e1e14857fd63bb183a3a5d3fd3d8bcfd832dc1309a54afbf286774487c80b6a921b01e5e98ebc9d825b335f42cca99172d1e8cebce1149c08c8a2f"
    assert len(c["steps"]["ends"]) == 32 and c["totals"]["classes"] == 761
    held, moved = m["steps"]["held"]["computed"], m["steps"]["moved"]["computed"]
    assert held[1:] == [(255 + s) * KV_BITS + 32 for s in range(1, 32)]  # the K/V rows so far, and the token sampled before
    assert moved == held  # full attention reads every K/V row it holds
    assert m["totals"]["peak_step"] == 31 and m["totals"]["moved_bits"] == m["totals"]["held_bit_steps"]
