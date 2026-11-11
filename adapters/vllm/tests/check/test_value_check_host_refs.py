"""`ValueChecker`'s references are host copies taken at hook time: a module that rewrites its inputs in place (the fused add-RMSNorm)
re-executes on the pre-hook copies, and a buffer overwritten after its hook keeps its hook-time value in the reference, so a
committer that reads it late fails the tap compare by name."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check.value_check import ValueChecker  # noqa: E402


class RMSNorm(torch.nn.Module):
    """vLLM's fused add-RMSNorm contract: `residual += x`, then `x` is overwritten with the normed sum; both are returned."""

    def forward(self, x, residual):
        residual.add_(x)
        x.copy_(residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + 1e-6))
        return x, residual


class Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.norm = RMSNorm()

    def forward(self, x, residual):
        h, residual = self.norm(self.lin(x), residual)
        out = h * 2
        h.mul_(3)                     # the norm's output buffer reused after its hook ran
        return out, residual


class Committer:
    """Acquires every hooked module's outputs as metas of its step; `late` keeps the tensors and reads their bytes at step end."""

    def __init__(self, model: Toy, *, late: bool = False) -> None:
        self.model, self.runner, self.late = model, SimpleNamespace(), late
        self._modules = [("lin", model.lin), ("norm", model.norm)]
        self._layouts: dict[int, list] = {}
        self.step, self._held = -1, []
        for name, m in self._modules:
            m.register_forward_hook(self._acquire(name))

    def _acquire(self, name):
        def hook(module, args, output):
            outs = [output] if isinstance(output, torch.Tensor) else list(output)
            for k, o in enumerate(outs):
                key = name if isinstance(output, torch.Tensor) else f"{name}/{k}"
                meta = SimpleNamespace(name=key, ordinal=len(self._layouts[self.step]), nbytes=o.numel() * o.element_size(), value=None)
                self._layouts[self.step].append(meta)
                if self.late:
                    self._held.append((meta, o))
                else:
                    meta.value = o.detach().contiguous().view(torch.uint8).numpy().tobytes()
        return hook

    def _begin_step(self) -> None:
        self.step += 1
        self._layouts[self.step] = []

    def _end_step(self) -> None:
        for meta, o in self._held:
            meta.value = o.detach().contiguous().view(torch.uint8).numpy().tobytes()
        self._held = []


class Reader:
    def read_meta(self, step, m):
        return m.value

    def record(self):
        return {"reader": "test"}


def _run(late: bool) -> tuple[ValueChecker, dict]:
    torch.manual_seed(0)
    model = Toy()
    com = Committer(model, late=late)
    vc = ValueChecker(com)
    vc.install()
    with torch.inference_mode():
        for _ in range(2):
            com._begin_step()
            model(torch.randn(3, 8), torch.randn(3, 8))
            com._end_step()
    vc.uninstall()
    refs = [t for st in vc.steps for d in st.values() for t in [*(a for c in d["in"] for a in c), *(o for c in d["out"] for o in c)]]
    assert refs and all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in refs)
    return vc, vc.compare(Reader())


def test_host_references_pass_with_in_place_inputs_and_a_reused_buffer() -> None:
    _, res = _run(late=False)
    assert res["verdict"] == "PASS"
    assert (res["tensors"], res["tap_equal"]) == (6, 6)
    assert res["reexec_tensors"] == res["reexec_equal"] == 6


def test_a_late_read_of_an_overwritten_buffer_fails_the_tap_compare() -> None:
    """The norm overwrites the Linear's output in place and the model then scales the norm's output: read at step end, both differ
    from their hook-time references; the residual, untouched after its hook, still agrees."""
    _, res = _run(late=True)
    assert res["verdict"] == "FAIL"
    assert sorted({key for _step, key, _n in res["tap_mismatch"]}) == ["lin", "norm/0"]
    assert (res["tensors"], res["tap_mismatch_n"]) == (6, 4)
