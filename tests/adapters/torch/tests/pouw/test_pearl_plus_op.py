"""Pearl+'s capture op (`verity_torch.pouw.pearl_plus_op`): one call as one dispatcher op whose body is the live run's
``gemm``, with a fake that gives the output's shape without running it."""
from __future__ import annotations

import pytest

from verity_torch.pouw import pearl_plus_op as P

torch = pytest.importorskip("torch")


class _Run:
    def __init__(self) -> None:
        self.calls: list = []

    def gemm(self, w, x, weight, bias):
        self.calls.append((w, x.shape, weight.shape, bias))
        return x @ weight.t()


def test_the_op_computes_with_the_runs_gemm_and_its_fake_has_the_outputs_shape(monkeypatch) -> None:
    run = _Run()
    monkeypatch.setitem(P.RUNS, "build-1", run)
    op = P.linear_op()
    x, weight = torch.ones(2, 3), torch.full((5, 3), 2.0)
    assert torch.equal(op(x, weight, None, "build-1", 4), torch.full((2, 5), 6.0))
    assert run.calls == [(4, (2, 3), (5, 3), None)]
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        y = op(mode.from_tensor(x), mode.from_tensor(weight), None, "build-1", 4)
    assert tuple(y.shape) == (2, 5) and len(run.calls) == 1
