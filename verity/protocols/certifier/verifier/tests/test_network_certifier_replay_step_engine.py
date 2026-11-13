"""replay_step.statics on the engine it is handed. An engine the step models binds to what the binding's formula gives
from the keys it reads: the served fixtures' engines, what a serving integration records of its scheduler, the replay's
engine. An engine with a draft, a policy or a scheduler the step does not model, or a key the binding does not read,
is refused by name."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.protocols.certifier.verifier import compile_slice as CS
from verity.protocols.certifier.verifier import replay_step as RS

FIXTURES = Path(__file__).parent / "fixtures"
SLICES = ("compile_slice_smallkv", "compile_slice_w1", "compile_slice_w1_class", "compile_slice_w212",
          "compile_slice_w212_class", "compile_slice_w212_class_async")

#: what a serving integration records of vLLM's `Scheduler`, and of its `AsyncScheduler`
SERVED = {"scheduler": "Scheduler", "max_num_batched_tokens": 2048, "max_num_seqs": 8, "max_model_len": 512,
          "enable_chunked_prefill": True, "long_prefill_token_threshold": 0, "policy": "fcfs", "async_scheduling": False,
          "block_size": 16, "enable_prefix_caching": True, "num_gpu_blocks": 27536}
SERVED_ASYNC = {**SERVED, "scheduler": "AsyncScheduler", "async_scheduling": True}
#: the replay's engine, field for field
REPLAY = {"max_num_seqs": 4, "max_num_batched_tokens": 256, "max_model_len": 512, "num_gpu_blocks": 1024, "block_size": 16,
          "long_prefill_token_threshold": 0, "enable_chunked_prefill": True, "enable_prefix_caching": False,
          "async_scheduling": False, "watermark": 0.0, "scheduler_reserve_full_isl": True, "policy": "fcfs"}
MODELED = {"sync serve": SERVED, "async serve": SERVED_ASYNC, "replay": REPLAY, "demo": CS.DEMO_ENGINE,
           "partial prefills": {**SERVED, "max_num_partial_prefills": 4}, "kv format": {**SERVED_ASYNC, "kv_cache_dtype": "fp8"},
           "watermark": {**REPLAY, "watermark": 0.25, "scheduler_reserve_full_isl": False, "long_prefill_token_threshold": 32}}

#: a set value of each draft key
DRAFTS = [("num_speculative_tokens", 5), ("num_speculative_tokens", 1),
          ("speculative", {"method": "mtp", "num_speculative_tokens": 5}), ("num_lookahead_tokens", 5),
          ("num_prefill_lookahead", 1), ("use_eagle_block_drop", True), ("draft_slots", 2)]
#: another value of each key the binding reads, at which `_engine()`'s binding differs
OTHER = {"max_num_seqs": 5, "max_num_batched_tokens": 65, "max_model_len": 129, "num_gpu_blocks": 25, "block_size": 8,
         "long_prefill_token_threshold": 3, "enable_chunked_prefill": False, "watermark": 0.5, "scheduler_reserve_full_isl": False,
         "enable_prefix_caching": True, "async_scheduling": True}


def _formula(R, W, A, e, IW, EW=0):
    """The binding as a function of the keys it reads alone, which is all `statics` was before it refused anything."""
    return {"R": R, "W": W, "A": A, "M": int(e["max_num_seqs"]), "T": int(e["max_num_batched_tokens"]),
            "L": int(e["max_model_len"]), "B": int(e.get("block_size", 16)),
            "H": int(e.get("long_prefill_token_threshold") or 0), "CH": int(bool(e.get("enable_chunked_prefill", True))),
            "WM": int(float(e.get("watermark", 0.0) or 0.0) * int(e["num_gpu_blocks"])),
            "FULL": int(bool(e.get("scheduler_reserve_full_isl", True))), "PC": int(bool(e.get("enable_prefix_caching"))),
            "NB": int(e["num_gpu_blocks"]), "IW": IW, "AS": int(bool(e.get("async_scheduling"))), "EW": EW}


def _engine(**kw):
    e = dict(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=128, num_gpu_blocks=24, block_size=16,
             enable_chunked_prefill=True, enable_prefix_caching=False, long_prefill_token_threshold=0, watermark=0.0,
             scheduler_reserve_full_isl=True)
    e.update(kw)
    return e


def _refused(engine) -> str:
    with pytest.raises(ValueError) as err:
        RS.statics(2, 1, 2, engine, 7)
    return str(err.value)


class _Reads(dict):
    """A dict that notes the keys read from it by name."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set[str] = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)


def test_the_served_fixtures_engines_bind_as_before():
    seen = set()
    for name in SLICES:
        fx = json.loads((FIXTURES / f"{name}.json").read_text())
        args = (fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"])
        st = RS.statics(*args)
        assert st == _formula(*args), name
        seen.add((st["AS"], st["PC"]))
    assert seen == {(0, 0), (1, 0), (0, 1)}


@pytest.mark.parametrize("name", sorted(MODELED))
def test_an_engine_the_step_models_binds_as_before(name):
    for EW in (0, 9):
        assert RS.statics(8, 1, 9, MODELED[name], 9, EW) == _formula(8, 1, 9, MODELED[name], 9, EW)
    assert RS.statics(8, 1, 9, SERVED_ASYNC, 9)["AS"] == 1 and RS.statics(8, 1, 9, MODELED["watermark"], 9)["WM"] == 256


def test_the_binding_reads_the_keys_it_lists_and_each_one_changes_it():
    e = _Reads(_engine(async_scheduling=False))
    base = RS.statics(2, 1, 2, e, 7)
    assert e.read - set(RS.ENGINE_DRAFT) == set(RS.ENGINE_READ) == set(OTHER)
    for k, v in OTHER.items():
        assert RS.statics(2, 1, 2, _engine(**{k: v}), 7) != base, k
    lists = (RS.ENGINE_READ, RS.ENGINE_DRAFT, tuple(RS.ENGINE_HELD), tuple(RS.ENGINE_INERT))
    assert sum(map(len, lists)) == len({k for ks in lists for k in ks})


@pytest.mark.parametrize("key, value", DRAFTS)
def test_an_engine_with_a_draft_is_refused_by_name(key, value):
    assert {k for k, _ in DRAFTS} == set(RS.ENGINE_DRAFT)
    for engine in (_engine(**{key: value}), {**SERVED_ASYNC, key: value}, {**REPLAY, key: value}):
        msg = _refused(engine)
        assert f"engine {key}={value!r}" in msg and "draft" in msg and "Serve with the draft off" in msg


@pytest.mark.parametrize("unset", [0, None, False, {}])
def test_an_engine_with_the_draft_off_or_absent_binds(unset):
    assert not any(k in _engine() for k in RS.ENGINE_DRAFT)
    base = RS.statics(2, 1, 2, _engine(), 7)
    assert base == _formula(2, 1, 2, _engine(), 7)
    for k in RS.ENGINE_DRAFT:
        assert RS.statics(2, 1, 2, _engine(**{k: unset}), 7) == base, k


@pytest.mark.parametrize("served", [SERVED, SERVED_ASYNC], ids=["sync", "async"])
def test_a_serves_speculative_entry_binds_absent_or_empty_and_is_refused_with_any_setting(served):
    """The three shapes of the entry a serve records of its speculative configuration: no key where the engine has none,
    an empty dict, and a dict of the settings that are set."""
    assert "speculative" not in served
    base = RS.statics(8, 1, 9, served, 9)
    assert base == _formula(8, 1, 9, served, 9)
    assert RS.statics(8, 1, 9, {**served, "speculative": {}}, 9) == base
    for settings in ({"method": "mtp"}, {"num_speculative_tokens": 0}, {"disable_padded_drafter_batch": False},
                     {"method": "mtp", "num_speculative_tokens": 5, "rejection_sample_method": "strict"}):
        msg = _refused({**served, "speculative": settings})
        assert f"engine speculative={settings!r}" in msg and all(name in msg for name in settings)
        assert "draft" in msg and "Serve with the draft off" in msg


@pytest.mark.parametrize("key, value", [("policy", "priority"), ("policy", None), ("scheduler", "PriorityScheduler"),
                                        ("scheduler", "SimpleNamespace")])
def test_a_policy_or_a_scheduler_the_step_does_not_model_is_refused_by_name(key, value):
    msg = _refused(_engine(**{key: value}))
    assert f"engine {key}={value!r}" in msg and all(repr(v) in msg for v in RS.ENGINE_HELD[key])


@pytest.mark.parametrize("scheduler, asy", [("AsyncScheduler", False), ("AsyncScheduler", None), ("Scheduler", True)])
def test_a_scheduler_that_disagrees_with_async_scheduling_is_refused_by_name(scheduler, asy):
    engine = _engine(scheduler=scheduler) if asy is None else _engine(scheduler=scheduler, async_scheduling=asy)
    msg = _refused(engine)
    assert f"engine scheduler={scheduler!r} with async_scheduling={asy!r}" in msg and "disagree" in msg
    assert RS.statics(2, 1, 2, _engine(scheduler=scheduler, async_scheduling=scheduler == "AsyncScheduler"), 7)["AS"] == int(
        scheduler == "AsyncScheduler")


def test_a_key_the_binding_does_not_read_is_refused_by_name():
    msg = _refused(_engine(max_num_scheduled_tokens=32))
    assert "engine max_num_scheduled_tokens=32:" in msg and "does not read it" in msg and "ENGINE_INERT" in msg
    msg = _refused({**SERVED, "stream_interval": 4, "kv_cache_groups": 2})
    assert "engine kv_cache_groups=2, stream_interval=4:" in msg and "does not read them" in msg
    assert "draft" in _refused(_engine(num_speculative_tokens=5, max_num_scheduled_tokens=32))


def test_a_key_that_leaves_the_steps_as_they_are_binds_at_any_value():
    base = RS.statics(2, 1, 2, _engine(), 7)
    for k, why in RS.ENGINE_INERT.items():
        assert why
        for v in (0, 1, 7, None, "x"):
            assert RS.statics(2, 1, 2, _engine(**{k: v}), 7) == base, k


@pytest.mark.parametrize("served", [SERVED, SERVED_ASYNC], ids=["sync", "async"])
def test_a_serve_whose_launch_left_the_context_limit_to_vllm_binds_as_one_launched_at_the_fitted_limit(served):
    """A serving integration records ``max_model_len_auto`` beside the limit vLLM fitted. The scheduler's steps read
    the fitted limit alone, so the two launches bind the same step, and another fitted limit binds another."""
    fitted = {**served, "max_model_len": 1008}
    auto = RS.statics(8, 1, 9, {**fitted, "max_model_len_auto": True}, 9)
    assert auto == RS.statics(8, 1, 9, fitted, 9) == _formula(8, 1, 9, fitted, 9) and auto["L"] == 1008
    assert RS.statics(8, 1, 9, {**served, "max_model_len_auto": True}, 9)["L"] == served["max_model_len"] != 1008


def test_the_compile_slice_refuses_what_its_step_refuses():
    size = dict(Q=3, D=5, P=4, K=3)
    st = CS.statics(2, 1, 2, CS.DEMO_ENGINE, 3, **size)
    assert {k: st[k] for k in RS.STATICS} == _formula(2, 1, 2, CS.DEMO_ENGINE, 3)
    with pytest.raises(ValueError, match="engine num_speculative_tokens=5: .*draft"):
        CS.statics(2, 1, 2, {**CS.DEMO_ENGINE, "num_speculative_tokens": 5}, 3, **size)
    with pytest.raises(ValueError, match="engine max_num_scheduled_tokens=8: "):
        CS.statics(2, 1, 2, {**CS.DEMO_ENGINE, "max_num_scheduled_tokens": 8}, 3, **size)
