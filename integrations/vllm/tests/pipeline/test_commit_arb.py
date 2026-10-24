"""`pipeline.commit_arb`'s traffic and verdict, without an engine: CERT-VARLEN's lengths and caps, the sampling mix, seeded
and in arrival order; a run passes only when every request and every window did."""
from __future__ import annotations

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import commit_arb as CA


def encode(text):
    return [0] + [ord(c) for c in text]


def test_traffic_is_seeded_varied_and_in_arrival_order():
    reqs = CA.traffic(encode, 400, seed=3, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    assert reqs == CA.traffic(encode, 400, seed=3, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    assert reqs != CA.traffic(encode, 400, seed=4, prompt_tokens=(16, 512), max_tokens=32, rate=0.5)
    steps = [q["arrive_step"] for q in reqs]
    assert steps == sorted(steps) and len({q["request_id"] for q in reqs}) == 400
    lengths = [len(q["prompt"]) for q in reqs]
    assert min(lengths) >= 16 and max(lengths) <= 512 and max(lengths) > 4 * min(lengths)
    tail = encode(CA.CHAT_TAIL)[1:]
    assert all(q["prompt"][-len(tail):] == tail for q in reqs if len(q["prompt"]) >= len(tail))
    s = [q["sampling"] for q in reqs]
    assert all(1 <= x["max_tokens"] <= 32 for x in s)
    temps = {x["temperature"] for x in s}
    assert temps == {0.0, 0.7, 1.0}
    assert any("seed" in x for x in s) and any(x["temperature"] and "seed" not in x for x in s)
    assert not any("seed" in x for x in s if x["temperature"] == 0.0)
    assert any("stop" in x for x in s) and any("min_tokens" in x for x in s)
    assert all(x.get("min_tokens", 0) <= x["max_tokens"] for x in s)


def test_a_template_frame_wraps_the_same_turns():
    class Tok:
        def apply_chat_template(self, turns, *, tokenize, add_generation_prompt, enable_thinking):
            assert not tokenize and add_generation_prompt and not enable_thinking
            return f"<u>{turns[0]['content']}</u><a>"

    frame = CA.template_frame(Tok())
    assert frame == ("<u>", "</u><a>")
    reqs = CA.traffic(encode, 20, seed=1, prompt_tokens=(64, 128), max_tokens=8, rate=1.0, frame=frame)
    assert all(q["prompt"][1:4] == encode("<u>")[1:] and q["prompt"][-7:] == encode("</u><a>")[1:] for q in reqs)


@pytest.mark.parametrize("text", ["16", "a:b", "0:4", "9:3"])
def test_prompt_range_refuses(text):
    with pytest.raises(UsageError):
        CA.prompt_range(text)


def test_summary_passes_only_when_every_request_and_window_did():
    q = {"request_id": "r", "sampling": {"temperature": 0.0}, "prompt": [1, 2]}
    ok = {"passed": True, "streamed_ids": 3, "discarded_samples": 1, "preemptions": 1}
    win = {"window": 0, "passed": True, "commit": {"steps": 4}}
    fin = {"reason": "stop", "stop_reason": "."}
    got = CA.summarize([(q, fin, ok)], [win], engine_steps=5, idle_steps=1)
    assert got["passed"] and got["finishes"] == {"stop-string": 1} and got["chunked_requests"] == 1 and got["committed_steps"] == 4
    assert not CA.summarize([(q, fin, {**ok, "passed": False})], [win], engine_steps=5, idle_steps=1)["passed"]
    assert not CA.summarize([(q, fin, ok)], [{**win, "passed": False}], engine_steps=5, idle_steps=1)["passed"]
    assert not CA.summarize([], [win], engine_steps=0, idle_steps=0)["passed"]
