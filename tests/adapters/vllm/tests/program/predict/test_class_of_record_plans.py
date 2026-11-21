"""A served window at the class of record (the BF16 Llama 3.2 1B row on the H100, four requests of 27 prompt ids and 2
outputs, all in one window) through the close's one compile (`window_circuit.served_lowering`) and its plan
(`window_plan.Plan`), on the processor: every refusal the card meets there shows here first, which no smaller class
shows, since the small templates and their rows are the class's own."""
from __future__ import annotations

import pytest

from tests.program.predict import test_steps as S
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_plan as WP
from verity_vllm.program.predict.phi import phi_of_serve
from verity_vllm.program.predict.steps import Registered

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("code_versions_stood_in")]

#: the H100 smoke's serve.json, as φ reads it
SERVE = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "execution": "bi-eager",
         "target": {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": True, "attention_backend": None,
                    "vllm_revision": "d9105ea80", "numerics_profile": "b1-eager-v3", "flash_attn_version": None}}
CLASS = WC.WindowClass(4, 27, 2)
G2 = {**S.GREEDY, "max_tokens": 2}


def test_a_served_window_at_the_class_of_record_plans():
    reg = Registered.of(phi_of_serve(SERVE), 657832114885280620, ADV.CAP_BITS)
    advice, ingress = S._session([(27, 2, G2)] * 4, [[(i, 0, 27) for i in range(4)], [(i, 27, 1) for i in range(4)]])
    compiled, low = WC.served_lowering(reg, CLASS, advice, ingress)
    plan = WP.Plan(low, [None] * len(low.programs))
    got = plan.describe()
    assert len(low.programs) == 4 and sum(got["units"].values()) == sum(k for _, k in low.circuit.templates) > 0
    assert low.unprovable["row_floor"] == {}
