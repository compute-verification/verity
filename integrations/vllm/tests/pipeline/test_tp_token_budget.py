"""A tensor-parallel row's engine token budget never exceeds ``max_num_seqs * max_model_len``: vLLM's warm-up dummy forward packs
the whole budget into ``max_num_seqs`` requests, and one longer than ``max_model_len`` faults FlashAttention's varlen kernel at TP2."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import row_records as R


def _wl(tmp_path, batch, I, O, mnbt=None):
    p = tmp_path / "wl.json"
    req = {"max_model_len": I + O, **({"max_num_batched_tokens": mnbt} if mnbt is not None else {})}
    p.write_text(json.dumps({"sweep": {"batch": batch, "max_input": I, "max_output": O, "input_lengths": [I] * batch, "output_caps": [O] * batch},
                             "engine_args_required": req, "model": "m", "tokenizer_revision": "r"}))
    return str(p)


@pytest.mark.parametrize("batch,I,O,mnbt,expect", [
    (1, 256, 32, 2048, 288),       # the crashing B1 coverage rows: max(2048, prompt + B) for one request of max_model_len 288
    (1, 256, 32, 257, 257),        # a budget under the cap is the workload's
    (8, 1024, 128, 8256, 8256),    # B8 rows: 8 x 1152 = 9216 >= the workload's budget, unchanged
    (2, 256, 32, None, 288),       # no budget declared: max_model_len, as before
])
def test_the_tp_token_budget_is_capped_at_max_num_seqs_times_max_model_len(tmp_path, monkeypatch, batch, I, O, mnbt, expect):
    monkeypatch.setattr(R, "checkpoint_dir", lambda *a, **k: "")
    s = R.tp_shape(_wl(tmp_path, batch, I, O, mnbt), str(tmp_path / "checkpoints.json"), str(tmp_path))
    assert (s["B"], s["MML"], s["MNBT"]) == (batch, I + O, expect)
