"""`verity_vllm.target_family.device_cc`: the compute capability of the GPU this process would use, read with no GPU.

`nvidia-smi` is asked first and torch only when it gives no answer. Both are stood in: `nvidia-smi` by the module's
`subprocess.run`, torch by a module under its name.
"""
from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verity_vllm import target_family as tf

SMI = ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"]


def _smi(monkeypatch, stdout: str = "", returncode: int = 0, raises: Exception | None = None) -> list[list[str]]:
    """`nvidia-smi` answers `stdout` with `returncode`, or cannot be run. Returns the command lines it is asked."""
    asked: list[list[str]] = []

    def run(cmd, **kwargs):
        asked.append(list(cmd))
        if raises is not None:
            raise raises
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(tf.subprocess, "run", run)
    return asked


def _torch(monkeypatch, capability: tuple[int, int] | None, raises: Exception | None = None) -> None:
    """A torch whose CUDA sees device 0 with `capability`, sees no device (None), or fails with `raises` when asked."""

    def is_available() -> bool:
        if raises is not None:
            raise raises
        return capability is not None

    torch = ModuleType("torch")
    vars(torch)["cuda"] = SimpleNamespace(is_available=is_available, get_device_capability=lambda index: {0: capability}[index])
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_nvidia_smi_answers_for_the_first_device_the_process_may_use(monkeypatch):
    asked = _smi(monkeypatch, "0, 8.9\n1, 9.0\n")
    _torch(monkeypatch, None, raises=AssertionError("torch asked though nvidia-smi answered"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert tf.device_cc() == ((8, 9), "nvidia-smi compute_cap") and asked == [SMI]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert tf.device_cc() == ((9, 0), "nvidia-smi compute_cap")
    # an index nvidia-smi does not list (or a device's UUID): its first row
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert tf.device_cc() == ((8, 9), "nvidia-smi compute_cap")


@pytest.mark.parametrize("smi", [
    {"raises": FileNotFoundError("nvidia-smi")},
    {"raises": subprocess.TimeoutExpired("nvidia-smi", 20)},
    {"returncode": 9, "stdout": "0, 8.9\n"},
    {"stdout": "\n"},
    {"stdout": "0, [N/A]\n"},
], ids=["absent", "hung", "failed", "no-rows", "no-capability"])
def test_torch_is_asked_only_when_nvidia_smi_gives_no_capability(monkeypatch, smi):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _smi(monkeypatch, **smi)
    _torch(monkeypatch, (9, 0))
    assert tf.device_cc() == ((9, 0), "torch.cuda.get_device_capability")
    _torch(monkeypatch, None)
    assert tf.device_cc() == (None, "torch.cuda.is_available() False")


def test_a_torch_that_fails_is_no_device_by_the_errors_name(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _smi(monkeypatch, raises=FileNotFoundError("nvidia-smi"))
    _torch(monkeypatch, (9, 0), raises=RuntimeError("CUDA driver initialization failed"))
    assert tf.device_cc() == (None, "no device: RuntimeError")


@pytest.mark.parametrize(("answer", "cc"), [([8, 9], (8, 9)), ((9, 0, 1), (9, 0, 1)), ((9,), (9,)), ("90", ("9", "0"))],
                         ids=["a-list", "three", "one", "a-text"])
def test_torchs_answer_is_given_as_the_tuple_of_whatever_it_holds(monkeypatch, answer, cc):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _smi(monkeypatch, raises=FileNotFoundError("nvidia-smi"))
    _torch(monkeypatch, answer)
    assert tf.device_cc() == (cc, "torch.cuda.get_device_capability")
