"""The adapter side of the static schedule: a workload's `sampling.ignore_eos` reaches vLLM's SamplingParams, a static run is
refused under the host prefix cache before any request is added, and `run_requests` records `timing["static_schedule"]`.  Over a stub
engine whose tokens depend on the prompt, so an EOS stop makes the schedule depend on contents and ignore_eos removes that."""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from verity_vllm.engine import static_schedule

EOS = 0


def _va():
    return pytest.importorskip("verity_vllm.engine.vllm_adapter")


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", mod)


class _Core:
    """Each step() decodes one token per live request, token = (first prompt token + position) % 5; a request stops at EOS unless its
    params say ignore_eos, else at max_tokens."""

    def __init__(self) -> None:
        self.live: dict[str, dict] = {}
        self.adds: list[str] = []

    def add_request(self, request_id, prompt, params) -> None:
        self.adds.append(request_id)
        self.live[request_id] = {"p0": prompt["prompt_token_ids"][0], "params": params, "toks": []}

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def step(self):
        outs = []
        for rid, st in list(self.live.items()):
            tok = (st["p0"] + len(st["toks"])) % 5
            st["toks"].append(tok)
            p = st["params"]
            reason = "length" if len(st["toks"]) == p.max_tokens else ("stop" if tok == EOS and not p.ignore_eos else None)
            outs.append(SimpleNamespace(request_id=rid, finished=reason is not None, num_cached_tokens=0,
                                        outputs=[SimpleNamespace(index=0, token_ids=list(st["toks"]), finish_reason=reason)]))
            if reason is not None:
                del self.live[rid]
        return outs


def _engine(va, prefix_caching=False):
    core = _Core()
    eng = va.Engine(llm=SimpleNamespace(llm_engine=core), runner=None, model=None, cp={}, kwargs={"enable_prefix_caching": False},
                    env_flags={}, build_s=0.0, effective={"enable_prefix_caching": prefix_caching})
    return eng, core


def _reqs(p0s, cap=6):
    return [{"request_id": f"r{i}", "prompt_token_ids": [p, 9, 9], "max_tokens": cap, "arrive_step": i} for i, p in enumerate(p0s)]


def test_load_workload_carries_ignore_eos_only_when_declared(tmp_path):
    va = _va()
    base = {"schema": "verity-vllm/workload/v2", "requests": [{"request_id": "a", "prompt_token_ids": [1, 2], "max_tokens": 4}],
            "variants": {"greedy": {"temperature": 0.0}}}
    for sampling, want in (({"variant": "greedy"}, None), ({"variant": "greedy", "ignore_eos": True}, True)):
        p = tmp_path / "wl.json"
        p.write_text(json.dumps({**base, "sampling": sampling}))
        assert va.load_workload(p)["sampling"].get("ignore_eos") is want


def test_sampling_params_hands_ignore_eos_to_vllm(fake_vllm):
    va = _va()
    assert va.sampling_params({"max_tokens": 4}, {"ignore_eos": True}).ignore_eos is True
    assert va.sampling_params({"max_tokens": 4}, {}).ignore_eos is False


def test_a_static_run_has_a_content_independent_schedule(fake_vllm):
    va = _va()
    runs = []
    for p0s in ([1, 3], [4, 2]):
        eng, _ = _engine(va)
        tokens, timing = va.run_requests(eng, _reqs(p0s), {"ignore_eos": True})
        assert timing["static_schedule"]["ok"], timing["static_schedule"]["problems"]
        assert all(len(t) == 6 for t in tokens.values())
        runs.append(timing["static_schedule"]["schedule"])
    assert static_schedule.compare(runs[0], runs[1])["ok"]


def test_without_ignore_eos_the_eos_decides_the_schedule(fake_vllm):
    va = _va()
    scheds = []
    for p0s in ([1, 3], [4, 2]):
        eng, _ = _engine(va)
        _, timing = va.run_requests(eng, _reqs(p0s), {})
        assert "static_schedule" not in timing
        scheds.append(static_schedule.schedule_of(["r0", "r1"], timing))
        held = static_schedule.check(_reqs(p0s), timing)
        assert not held["ok"] and "finished by 'stop'" in held["problems"][0]
    assert not static_schedule.compare(scheds[0], scheds[1])["ok"]


def test_a_static_run_under_the_host_prefix_cache_is_refused_before_any_request(fake_vllm):
    va = _va()
    eng, core = _engine(va, prefix_caching=True)
    with pytest.raises(RuntimeError, match="enable_prefix_caching=false"):
        va.run_requests(eng, _reqs([1]), {"ignore_eos": True})
    assert core.adds == []
