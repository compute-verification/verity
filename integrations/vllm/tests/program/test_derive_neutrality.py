"""Derive-neutrality, the suite's tier: GLM-5.3's served constructions derived on vLLM-free stand-ins keep the Program digest, gate
count and node count pinned in `derive_neutrality.json` -- the three facts the neutrality proof compared across trees, here
on every change.  The TP toy's ranks at world 2 and 8, the DSA indexer and IndexShare layers on the H200, the routed-MoE FP8 chain
at vLLM's sm_90 defaults, and RMSNorm (unfused, and fused with the residual) at GLM-5.3's width under the declared non-BI H200.

A refactor that is meant to be neutral keeps every pin.  A change that moves a circuit on purpose re-pins in its own PR
(`VERITY_REPIN=1 uv run pytest tests/program/test_derive_neutrality.py`), and the real rows of record are then rebuilt on a CPU pod:
`verity-vllm neutrality --tier fast|heavy` (`pipeline/neutrality.py`)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile, vllm_ruleset
from tests.pipeline import tp_toy
from tests.program import test_dsa_derive as DSA
from tests.program import test_sigmoid_moe_fp8_derive as MOE
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)

PINS = Path(__file__).with_name("derive_neutrality.json")
REPIN = os.environ.get("VERITY_REPIN") == "1"
GLM_HIDDEN, GLM_EPS = 6144, 1e-5


class _Norm(nn.Module):
    def __init__(self, fused: bool) -> None:
        super().__init__()
        self.norm, self.fused = ex.RMSNorm(GLM_HIDDEN, GLM_EPS), fused

    def forward(self, x, res):
        return self.norm(x, res) if self.fused else self.norm(x)


def _norm(fused: bool):
    with torch.device("meta"):
        m = _Norm(fused)
    xs = tuple(torch.empty(1, GLM_HIDDEN, dtype=torch.bfloat16, device="meta") for _ in range(2))
    observe_target_profile(MOE.H200)
    try:
        return derive_report(m, xs, mode="profile", profile="b1-eager-v3", name="GlmNorm", rules=vllm_ruleset())
    finally:
        clear_observations()


def _moe():
    observe_target_profile(MOE.H200)
    try:
        return MOE._derive(MOE._ExpertsDefault(), (MOE._logits(MOE.E), MOE._bf16(np.random.default_rng(7), MOE.M, MOE.H)))
    finally:
        clear_observations()


def _dsa():
    model = DSA._model()
    return DSA._derive(model, DSA._inputs(model))


STANDINS = {
    **{f"tp-toy-w{w}-r{r}": (lambda w=w, r=r: tp_toy.derive_rank(w, r)[0]) for w, r in ((2, 0), (2, 1), (8, 0), (8, 7))},
    "dsa-indexer-indexshare-h200": _dsa,
    "moe-experts-fp8-ue8m0-h200": _moe,
    "rmsnorm-6144-h200": lambda: _norm(False),
    "rmsnorm-fused-6144-h200": lambda: _norm(True),
}


def _facts(rep) -> dict:
    assert rep.program is not None, rep.unsupported
    return {"digest": rep.digest, "gates": rep.program.gates, "nodes": len(rep.program.fn.body.nodes)}


@pytest.fixture(scope="module")
def pins():
    doc = json.loads(PINS.read_text())
    yield doc
    if REPIN:
        PINS.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def test_every_stand_in_is_pinned(pins):
    assert sorted(pins) == sorted(STANDINS), "add or drop the stand-in's pin (VERITY_REPIN=1)"


@pytest.mark.parametrize("name", sorted(STANDINS))
def test_the_stand_in_keeps_its_pinned_program(name, pins, request):
    if name.startswith("dsa-"):
        request.getfixturevalue("served")
    got = _facts(STANDINS[name]())
    if REPIN:
        pins[name] = got
    assert got == pins.get(name), (f"{name}: the derive moved ({pins.get(name)} -> {got}); a refactor meant to be neutral is not, "
                                   f"or re-pin a deliberate circuit change with VERITY_REPIN=1 and rebuild the rows of record "
                                   f"(verity-vllm neutrality)")
