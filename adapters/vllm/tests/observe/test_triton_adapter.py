"""`verity_vllm.observe.triton_adapter`'s wrapper of a Triton launch, with the engine's patch of `JITFunction.run` stood in
(no CPU machine has Triton): `install` patches once and binds the recorder, a second `install` and `set_recorder` only
rebind it, `uninstall` takes the patch off once, and the wrapper hands a launch straight to the run it replaced while no
recorder is bound or the bound one is not recording."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.observe import triton_adapter as T


class _Patch:
    """What `hooks.triton_launch` returns, as far as the adapter reads it."""

    def __init__(self) -> None:
        self.launches: list[tuple] = []
        self.uninstalled = 0

    def original(self, jit_fn, *args, **kwargs):
        self.launches.append((jit_fn, args, kwargs))
        return "the kernel"

    def uninstall(self) -> None:
        self.uninstalled += 1


@pytest.fixture
def patched(monkeypatch) -> tuple[_Patch, list]:
    """The adapter with nothing installed and the engine's patch stood in: the patch, and what was installed with it."""
    for key in ("orig_run", "rec", "patch"):
        monkeypatch.setitem(T._STATE, key, None)
    patch, made = _Patch(), []

    def triton_launch(make):
        made.append(make("the run Triton has"))
        return patch

    monkeypatch.setattr(T.hooks, "triton_launch", triton_launch)
    return patch, made


def test_install_patches_the_launch_once_and_uninstall_takes_it_off_once(patched) -> None:
    patch, made = patched
    first: Any = SimpleNamespace(active=False)
    second: Any = SimpleNamespace(active=False)
    assert not T.is_installed()
    T.install(first)
    assert T.is_installed() and made == [T._run]
    assert dict(T._STATE) == {"orig_run": patch.original, "rec": first, "patch": patch}
    T.install(second)
    assert made == [T._run] and T._STATE["rec"] is second and T._STATE["patch"] is patch
    T.set_recorder(None)
    assert T._STATE["rec"] is None and T.is_installed()
    T.set_recorder(first)
    T.uninstall()
    assert patch.uninstalled == 1 and not T.is_installed() and T._STATE["rec"] is None
    T.uninstall()
    assert patch.uninstalled == 1


@pytest.mark.parametrize("recorder", [None, SimpleNamespace(active=False)], ids=["no recorder", "a recorder that is not recording"])
def test_the_wrapper_hands_a_launch_to_the_run_it_replaced_when_nothing_records(patched, recorder) -> None:
    patch, _ = patched
    T.install(recorder)
    assert T._run("the jit function", 1, 2, grid=(4,), warmup=False, num_warps=8) == "the kernel"
    assert patch.launches == [("the jit function", (1, 2), {"grid": (4,), "warmup": False, "num_warps": 8})]
