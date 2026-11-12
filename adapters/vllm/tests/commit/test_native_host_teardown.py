"""`NativeHostCommitter.uninstall` takes its sources off newest first: two sources that wrap one attribute (on a TP rank,
`MoeBlockSource` and then `LinearQuantSource` both wrap ``torch.ops._C.per_token_group_fp8_quant``) unwind to the
original, where taking them off in install order is refused (`engine.hooks`) and leaves the committer half installed."""
from __future__ import annotations

import types

import torch

from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.engine import hooks


class WrapSource:
    """A source that wraps ``ops.quant`` through its own `hooks.Hooks`, as the real sources do."""

    def __init__(self, ops: types.SimpleNamespace, tag: str) -> None:
        self.ops, self.tag, self._hooks = ops, tag, hooks.Hooks()

    def install(self) -> None:
        self._hooks.wrap(self.ops, "quant", lambda orig: (lambda x: (self.tag, orig(x))))

    def uninstall(self) -> None:
        self._hooks.uninstall()


def test_two_sources_on_one_attribute_come_off_to_the_original() -> None:
    def quant(x):
        return ("quant", x)

    ops = types.SimpleNamespace(quant=quant)
    def execute_model(scheduler_output, *a, **kw):
        return None

    runner = types.SimpleNamespace(execute_model=execute_model)
    com = NativeHostCommitter(types.SimpleNamespace(model=torch.nn.Module(), runner=runner), run_id="teardown", arena_bytes=None,
                              geo_digest=b"g" * 32)
    com._start_worker = lambda: None
    com.gc_pauses = types.SimpleNamespace(install=lambda: None, uninstall=lambda: None)
    for tag in ("moe", "linear_quant"):
        com.attach_source(WrapSource(ops, tag))
    before = hooks.live()
    com.install()
    assert ops.quant(1) == ("linear_quant", ("moe", ("quant", 1)))
    assert len(hooks.live()) == len(before) + 3
    com.uninstall()
    assert ops.quant is quant and runner.execute_model is execute_model
    assert hooks.live() == before
    assert not com.installed
