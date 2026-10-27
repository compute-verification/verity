"""compiled_relations.warm_vml: each process's first MKL VML call is made on one thread, once, before the compiled relations run.

torch's statically linked MKL races on its first VML call when several intra-op threads make it at once (one chunk at 1771 ulp);
the helper's single-threaded `exp` initialises every VML function.  CPU: without torch it does nothing; with torch it makes one
`exp` call at one intra-op thread and restores the thread count, and a second call makes none.  `commit-replay` and both compiled
checks call it before their first torch op."""
from __future__ import annotations

import sys

import pytest

from verity_vllm.program import compiled_relations as CR


def test_without_torch_it_does_nothing(monkeypatch):
    monkeypatch.setattr(CR, "_VML_WARM", False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert CR.warm_vml() is False
    assert CR._VML_WARM is False


def test_with_torch_it_makes_one_single_threaded_exp_call_restores_the_threads_and_is_idempotent(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(CR, "_VML_WARM", False)
    before = torch.get_num_threads()
    real_exp, real_set = torch.exp, torch.set_num_threads
    exp_threads: list[int] = []
    set_calls: list[int] = []

    def exp(x, *a, **k):
        exp_threads.append(torch.get_num_threads())
        return real_exp(x, *a, **k)

    def set_num_threads(n):
        set_calls.append(n)
        real_set(n)

    monkeypatch.setattr(torch, "exp", exp)
    monkeypatch.setattr(torch, "set_num_threads", set_num_threads)
    assert CR.warm_vml() is True
    assert exp_threads == [1]
    assert set_calls == [1, before]
    assert torch.get_num_threads() == before
    assert CR.warm_vml() is True
    assert exp_threads == [1] and set_calls == [1, before]


def test_the_thread_count_is_restored_when_exp_raises(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(CR, "_VML_WARM", False)
    before = torch.get_num_threads()

    def exp(_x):
        raise RuntimeError("exp")

    monkeypatch.setattr(torch, "exp", exp)
    with pytest.raises(RuntimeError, match="exp"):
        CR.warm_vml()
    assert torch.get_num_threads() == before
    assert CR._VML_WARM is False


def test_commit_replay_warms_before_it_replays(monkeypatch):
    from verity_vllm.pipeline import cpu_replay

    calls: list[str] = []
    monkeypatch.setattr(cpu_replay, "warm_vml", lambda: calls.append("warm"))
    monkeypatch.setattr(cpu_replay, "replay", lambda o: calls.append("replay") or True)
    assert cpu_replay.main(object()) == 0
    assert calls == ["warm", "replay"]


@pytest.mark.parametrize("module, cls", [("verity_vllm.check.replay.compiled_kernel_check", "CompiledKernelCheck"),
                                         ("verity_vllm.check.compiled_value_check", "CompiledValueCheck")])
def test_each_compiled_check_warms_when_it_is_made(monkeypatch, module, cls):
    torch = pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module(module)
    calls: list[str] = []
    monkeypatch.setattr(mod, "warm_vml", lambda: calls.append("warm"))
    model = torch.nn.Linear(1, 1)
    if cls == "CompiledKernelCheck":
        mod.CompiledKernelCheck(None, model, None, seed=0, seed_source=None)
    else:
        mod.CompiledValueCheck(None, model, None)
    assert calls == ["warm"]
