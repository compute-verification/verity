"""serve-openai's ``--boundary-rows``: off unless given, history tables refused, and given, the first stage on the serve's
scheduler: the smallest window (one request, one token) committed under OUT/boundary/ with its line in boundary.jsonl."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array
from verity.circuits.types import Value as Bits

from verity_vllm.config import UsageError
from verity_vllm.engine import handle
from verity_vllm.pipeline import boundary_rows as BR
from verity_vllm.pipeline import boundary_windows as BW
from verity_vllm.pipeline import serve_openai as SO

XOR8 = PrimitiveDefinition("ServeBoundaryTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("ServeBoundaryTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 1, "min_tokens": 0, "ignore_eos": True,
          "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}


class Scheduler:
    def __init__(self):
        self.cache_config, self.max_model_len, self.outputs = NS(enable_prefix_caching=False), 64, {}

    def add_request(self, request):
        pass

    def schedule(self):
        return NS()

    def update_from_output(self, so, mro):
        out, self.outputs = self.outputs, {}
        return out

    def finish_requests(self, request_ids, finished_status):
        return []


def test_the_first_stage_is_off_unless_given_and_commits_no_history_tables():
    assert SO.boundary_config(NS(boundary_rows=None)) is None
    assert SO.boundary_config(NS(boundary_rows='{"check": "t"}')) == {"check": "t"}
    with pytest.raises(UsageError, match="no history tables"):
        SO.boundary_config(NS(boundary_rows='{"check": "t", "tables": {"layers": 1}}'))
    with pytest.raises(UsageError, match="naming its check template"):
        SO.boundary_config(NS(boundary_rows="{}"))


def test_the_smallest_window_is_committed_on_the_serves_scheduler(tmp_path, monkeypatch):
    sched = Scheduler()
    monkeypatch.setattr(handle, "scheduler", lambda engine: sched)
    monkeypatch.setattr(BW, "check_template", lambda t: PAIR)
    stop = BW.install_boundary({"check": "pair", "per_prompt": 0, "per_output": 1}, NS(), NS(), tmp_path, [])
    try:
        sched.add_request(NS(request_id="a", prompt_token_ids=[1], num_cached_tokens=0, sampling_params=NS(**GREEDY)))
        sched.outputs = {0: NS(outputs=[NS(request_id="a", new_token_ids=[7], finish_reason=NS(name="LENGTH"))])}
        sched.schedule()
        sched.update_from_output(None, None)
    finally:
        stop()
    (line,) = [json.loads(x) for x in (tmp_path / "boundary.jsonl").read_text().splitlines()]
    assert "error" not in line and (line["window"], line["prompt"], line["outputs"], line["spans"]) == (0, 1, 2, 1)
    (span,) = BR.reopen(tmp_path / "boundary" / "w0" / "p1-o2", PAIR).spans
    assert (span.engine_id, span.served, span.finish) == ("a", (7,), "length")
