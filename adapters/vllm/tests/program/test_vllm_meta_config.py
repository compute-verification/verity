"""`vllm_meta.make_vllm_config` makes the eager meta config a Build traces, whatever execution its target declares. A compiled target's
`enforce_eager` and `compilation_config` (its served engine's, which the Build declares through `modelled_engine()`) don't reach
`EngineArgs` beside the config's own `enforce_eager=True`; the target's other engine args do."""
from __future__ import annotations

from tests import fake_vllm
from verity_vllm.program.frontend import vllm_meta as vm
from verity_vllm.program.frontend.target_profile import TargetProfile


def _engine_args_seen(monkeypatch, **kw) -> dict:
    """The keyword arguments `make_vllm_config` hands `EngineArgs`, here stood in."""
    seen: dict = {}

    class EngineArgs:
        def __init__(self, **args):
            seen.update(args)

        def create_engine_config(self):
            return seen

    fake_vllm.install(monkeypatch).set("vllm.engine.arg_utils:EngineArgs", EngineArgs)
    vm.make_vllm_config(**kw)
    return seen


def test_a_compiled_class_g_target_reaches_the_meta_config_with_enforce_eager_once(monkeypatch):
    target = TargetProfile(batch_invariant=False, compilation="breakable", async_scheduling=True)
    args = target.engine_args()
    assert args["enforce_eager"] is False and "compilation_config" in args
    seen = _engine_args_seen(monkeypatch, **args)
    assert seen["enforce_eager"] is True and "compilation_config" not in seen
    assert seen["async_scheduling"] is True


def test_an_eager_target_s_engine_args_reach_the_meta_config_unchanged(monkeypatch):
    args = TargetProfile(async_scheduling=False).engine_args()
    assert "enforce_eager" not in args
    seen = _engine_args_seen(monkeypatch, **args)
    assert seen["enforce_eager"] is True and seen["async_scheduling"] is False
