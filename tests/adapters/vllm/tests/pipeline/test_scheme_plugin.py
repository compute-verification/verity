"""`scheme_plugin`'s engine side on stand-ins for vLLM's engine core and configuration: the schemes attached by name with
the engine's context, the exchange answered from the configuration, a refused input aborted, and what a scheme holds
released at shutdown."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity_vllm import messages
from verity_vllm.engine import admission, kv_blocks
from verity_vllm.pipeline import scheme_plugin as SP

FEED = "/tmp/feed"
ENV = {"VERITY_PROTOCOLS": "stand-in", "VERITY_PROTOCOLS_CONFIG": '{"feed": "%s", "stand-in": {"k": 1}}' % FEED}


def vllm_config(tp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="m", enforce_eager=True, dtype="torch.bfloat16", quantization=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False, num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(async_scheduling=False))


class Held:
    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


@pytest.fixture
def stand_in(monkeypatch):
    got: list[tuple[Any, dict]] = []
    held = Held()

    def attach(exchange, ctx):
        got.append((exchange, ctx))
        exchange.request("stand-in", H.FixRanges(64))
        return held
    monkeypatch.setitem(S.SCHEMES, "stand-in", SimpleNamespace(attach=attach))
    yield got, held
    messages.close()


def test_the_settings_name_the_schemes_and_their_shared_feed():
    assert SP.config({}) == ([], {})
    assert SP.config(ENV) == (["stand-in"], {"feed": FEED, "stand-in": {"k": 1}})
    with pytest.raises(ValueError, match="with feed, the directory the server's front and its engine share"):
        SP.config({"VERITY_PROTOCOLS": "stand-in"})


def test_the_engine_attaches_each_scheme_with_its_context_and_answers_what_vllms_configuration_declares(stand_in):
    got, held = stand_in
    ex, cfg = messages.Exchange(), vllm_config()
    assert SP.attach(ex, cfg, *SP.config(ENV)) == [held]
    assert got == [(ex, {"process": H.ENGINE, "feed": FEED, "config": {"k": 1}})]
    assert cfg.cache_config.num_gpu_blocks_override == 64
    assert ex.declared.properties == {"model": "m", "execution": None, "eager": True, "world": 1, "dtype": "bfloat16",
                                      "quantization": None, "prefix_cache": False, "kernels": "profile",
                                      "async_steps": False}
    with pytest.raises(NotImplementedError, match="vllm/scheme-on-in-vllm-server: schemes inside vLLM's server on one GPU"):
        SP.attach(messages.Exchange(), vllm_config(tp=2), *SP.config(ENV))


def test_register_attaches_before_the_kv_cache_and_releases_at_shutdown(stand_in, monkeypatch):
    got, held = stand_in
    seams: list = []
    monkeypatch.setattr(kv_blocks, "attach", lambda: (lambda: None))
    monkeypatch.setattr(admission, "register", lambda attach, refuse=None: seams.append(refuse))
    order: list[str] = []

    class Core:
        def _initialize_kv_caches(self, vllm_config):
            order.append(f"kv after {len(got)} attach")

        def shutdown(self):
            order.append(f"shutdown after {held.released} release")

    monkeypatch.setattr(SP, "_PLUGIN", [])
    SP.register(ENV, core=Core)
    try:
        SP.register(ENV, core=Core)
        assert len(SP._PLUGIN) == 1 and seams == [SP.aborts]
        core = Core()
        core._initialize_kv_caches(vllm_config())
        core.shutdown()
        assert order == ["kv after 1 attach", "shutdown after 1 release"] and got[0][0] is messages.current()
    finally:
        SP._PLUGIN[0][0].uninstall()


def test_a_refused_input_is_aborted_in_the_scheduler_and_its_reason_logged():
    finished, logged = [], []
    hook = SimpleNamespace(held={"cmpl-a-0-1": "a"}, live={"cmpl-b-0-1": "b", "cmpl-a-1-1": "a"},
                           sched=SimpleNamespace(finish_requests=lambda ids, status: finished.append((ids, status))))
    SP.aborts(hook, logged.append, status="ABORTED")("a", "its release was lost")
    assert finished == [(["cmpl-a-0-1", "cmpl-a-1-1"], "ABORTED")]
    assert logged == ["verity: input a refused (2 request(s) aborted): its release was lost"]
