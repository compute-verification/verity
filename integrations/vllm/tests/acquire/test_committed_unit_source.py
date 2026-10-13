from __future__ import annotations

import json
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


class _InPlace(torch.nn.Module):
    def forward(self, x):
        x = x.clone()
        x.add_(2)
        return x


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


def test_tuple_output_is_committed_as_one_input_group_with_members():
    com = _Committer()
    sites = [
        {"family": C.FAMILY, "step": 0, "op_path": "block", "op": "custom.tuple",
         "output_member": f"committed_0/{member}"}
        for member in ("0", "1")
    ]
    source = S.CommittedUnitSource(com, _Model(), sites)
    values = (torch.tensor([1.0]), torch.tensor([2]))
    source._after("block", "custom.tuple", values)
    assert [name for name, _tensor, _step in com.acquired] == ["block/committed_0/0", "block/committed_0/1"]


def test_mutable_op_tap_commits_the_return_and_post_write_operand():
    com, model = _Committer(), _InPlace()
    sites = [
        {"family": C.FAMILY, "step": 0, "op_path": "", "op": "aten.add_.Tensor",
         "output_member": f"committed_0/{member}"}
        for member in ("0", "1")
    ]
    source = S.CommittedUnitSource(com, model, sites)
    source.install()
    source.on_step_begin()
    got = model(torch.tensor([1.0]))
    source.uninstall()
    assert torch.equal(got, torch.tensor([3.0]))
    assert [name for name, _tensor, _step in com.acquired] == ["committed_0/0", "committed_0/1"]
    assert all(torch.equal(tensor, got) for _name, tensor, _step in com.acquired)


def test_tensor_list_tap_selects_only_named_writes():
    class _Func:
        _schema = SimpleNamespace(arguments=[
            SimpleNamespace(name="kernel", alias_info=None),
            SimpleNamespace(name="tensors", alias_info=SimpleNamespace(is_write=True)),
            SimpleNamespace(name="meta_json", alias_info=None),
        ])

        def __str__(self):
            return "verity_cba.triton_launch.default"

    a, b, c = (torch.tensor([v]) for v in (1, 2, 3))
    meta = json.dumps({"tensor_args": ["a_ptr", "b_ptr", "c_ptr"]})
    assert S._written_values(_Func(), ("kernel", [a, b, c], meta), {}, selected=["c_ptr"])[0] is c
