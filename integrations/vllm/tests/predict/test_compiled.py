"""verity_vllm.predict.compiled: each pinned model's predicted served-compiled Build is the Build traced on the pinned machine, from the
model's config.json (`engine/profiles/hf_configs/`), its served pin (`verity_vllm/pins/served/`) and the device pin alone.

The pinned digests are the served Builds of record on vy-glm53-s1 (one H200: cc 9.0, 132 SMs) over SCHEDULE (prompts of 100, 37, 9 and
300 tokens capped at 14, 9, 8 and 6, arriving at steps 0, 0, 3 and 6): Llama-3.2-1B r20261018-062949-a32b, Qwen3-8B
r20261018-062957-da83, Qwen3-30B-A3B r20261018-084657-c7f6, each Matched and Committed at that Build.  The served Commit with the
predicted Build as Build of record passes for each: r20261018-105743-72a0, r20261018-115551-2b64 and r20261018-124350-5c89.  On
vy-nebius-2 (one RTX PRO 6000: cc 12.0, 188 SMs) over the same SCHEDULE: Llama-3.2-1B r20261018-203256-0665 and Qwen3-8B
r20261018-211943-d337, each Matched and Committed at that Build.  A library change that moves a Definition moves the traced digest too: trace the Build again (`served-match`), write
its pin (`predict compiled-pin`) and pin both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.engine.profiles.dense_generic import find_hf_config
from verity_vllm.predict import compiled as PC

H200 = "NVIDIA H200"
RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
SCHEDULE = [{"request_id": f"r{i}", "arrive": a, "prompt_tokens": n, "max_tokens": t}
            for i, (n, t, a) in enumerate(((100, 14, 0), (37, 9, 0), (9, 8, 3), (300, 6, 6)))]


@pytest.mark.parametrize("role,device,program,manifest,artifact", [
    ("LLAMA32_1B", H200, "1f2df5e1029f878debf647a562314108b4e15159e90c5a852adefe9ade1987077c2b4cbf5621209148cdafbdc1113f15cbafd7a9c3f38d11a70659c291062a6b",
     "f2862dbc43dbeb61bd7e6be2c60cefbd756e04d754a193875e4b3fa64d747071", "0a169ef67dd55fb6"),
    ("QWEN3_8B", H200, "beed60d6bb75d7d9d39b7adf154b0b458b981e76ff2a6aed29c5f73046f0194b44b221b20ff86e0e5a7cb6a41a465592beecf3ea81a8add0b3d5d165c79f960c",
     "3d1368d361d6c1a6", "3bd5dcb11c2004ca"),
    ("QWEN3_30B_A3B", H200, "763ff762378d085d78977b341aab0b38bf7047507cf9887f51a4bcedcbe36dac87faeefd68e957a914d6a70147f912c31b4ffe3dc7c44f5cb86c477f9cd662b4",
     "8564ceaa69e105bf", "970c4d99d29792e3"),
    ("LLAMA32_1B", RTX, "c98f0b6b62bf8ea5a06333f828df1c09ebae314bac4c438ca4f84d8206b3595c5f8bf1b7333a62e5f264ee2f8377bd61939ff4f13765bb9eca99e5cd145c47cf",
     "ee8896c9db33aac5", "fba557acf341db86"),
    ("QWEN3_8B", RTX, "feb1c2906ea08bddd4c38b5fe402b203ac364ef888b1b4b9664b9c2011e3a33c4adc1b965b6049af37de659457d4e0784217f7be61ae420e962a11e3a09ebbc0",
     "c28dc3bf7e0ef088", "533721fc9d37833a"),
])
def test_the_predicted_served_build_is_the_traced_one(role, device, program, manifest, artifact):
    b = PC.predict(role, device, SCHEDULE, 14)
    assert b["program_digest"] == program
    assert b["manifest_digest"].startswith(manifest) and b["artifact"]["digest"].startswith(artifact)
    assert b["complete"] and not b["routes_missing"]


def _cfg(role: str) -> dict:
    return json.loads(find_hf_config(role)[0].read_text())


@pytest.mark.parametrize("cfg,needle", [
    ({**_cfg("QWEN3_8B"), "attention_bias": True}, "attention bias"),
    ({**_cfg("QWEN3_8B"), "hidden_act": "gelu"}, "hidden_act"),
    ({**_cfg("QWEN3_30B_A3B"), "mlp_only_layers": [0]}, "dense layers"),
    (json.loads((Path(__file__).parent / "fixtures" / "olmoe-1b-7b.config.json").read_text()), "layout"),
])
def test_a_config_outside_the_compiled_layouts_is_refused(cfg, needle):
    with pytest.raises(Unmodelled, match=needle):
        PC.model_of(cfg)


def test_a_step_at_a_fused_moe_row_count_without_a_recorded_launch_is_refused():
    with pytest.raises(Unmodelled, match="no fused_moe_kernel launch at 50 rows"):
        PC.predict("QWEN3_30B_A3B", H200, [{"request_id": "r0", "arrive": 0, "prompt_tokens": 50, "max_tokens": 2}], 2)
