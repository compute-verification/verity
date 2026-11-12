"""``slots_gpu.driver_bindings``: the driver bindings an ``SmPartition`` is made through are torch's own, and a Python
without them is refused by name, on any machine (no device is touched)."""
from __future__ import annotations

import pytest

import slots_gpu as G

torch = pytest.importorskip("torch")
from torch.cuda import _utils  # noqa: E402


def test_the_bindings_are_torch_s(monkeypatch: pytest.MonkeyPatch) -> None:
    drv, ok = object(), object()
    monkeypatch.setattr(_utils, "_cuda_bindings_driver", drv, raising=False)
    monkeypatch.setattr(_utils, "_check_cuda_bindings", ok, raising=False)
    assert G.driver_bindings() == (drv, ok)


@pytest.mark.parametrize("state", ["missing", "None"])
def test_a_python_without_them_is_refused_by_name(monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    """A torch from before its green contexts has no such name; a later one without cuda-bindings holds None."""
    if state == "missing":
        monkeypatch.delattr(_utils, "_cuda_bindings_driver", raising=False)
    else:
        monkeypatch.setattr(_utils, "_cuda_bindings_driver", None, raising=False)
    with pytest.raises(RuntimeError, match=rf"no CUDA driver bindings .*_cuda_bindings_driver is {state}\), so no green context"):
        G.driver_bindings()
