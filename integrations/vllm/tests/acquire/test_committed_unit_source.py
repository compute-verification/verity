from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verity_vllm.acquire.sources import committed_unit_source as S
from verity_vllm.pipeline import committed_units as C


class _Committer:
    name = "fake"

    def __init__(self):
        self._cur = SimpleNamespace(step=0)
        self.errors = []
        self.sources = []
        self.acquired = []

    def attach_source(self, source):
        self.sources.append(source)

    def _acquire(self, name, tensor, job):
        self.acquired.append((name, tensor.detach().clone(), job.step))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.ReLU()

    def forward(self, x):
        return self.block(x)


def _manifest():
    return {
        "identities": [{
            "family": C.FAMILY, "step": 0, "op_path": "block", "op": "aten.relu.default",
            "output_member": "committed_0",
        }],
        "committed_units": C.record([{
            "input": "committed_0", "op": "aten.relu.default", "module": "block",
            "reason": "test", "list_entry": {"op": "aten.relu.default", "module": "block", "reason": "test"},
        }]),
    }


def test_dispatch_tap_commits_the_listed_op_output():
    com, model = _Committer(), _Model()
    classes, source = S.attach(com, _manifest(), (), model=model)
    assert classes == (C.FAMILY,) and source is not None
    source.install()
    source.on_step_begin()
    got = model(torch.tensor([-2.0, 3.0]))
    source.uninstall()
    assert torch.equal(got, torch.tensor([0.0, 3.0]))
    assert len(com.acquired) == 1
    assert com.acquired[0][0] == "block/committed_0"
    assert torch.equal(com.acquired[0][1], got)
    assert source.stats["tensors"] == 1 and not com.errors


def test_nonempty_record_is_fail_closed_for_of_record_use():
    rec = _manifest()
    assert rec["committed_units"]["integrity"] == {"linked": False, "unverified": ["committed_0"]}
    with pytest.raises(RuntimeError, match="unverified and not approved"):
        C.require_of_record(rec, where="test Program of record")
