"""Consecutive training steps with torch's own optimizers: `export_steps` traces them as one graph on a copy of the model, and
`steps_program` derives that graph as one Program, which computes what torch's steps compute, bit for bit."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from verity.circuits.boundary import gate_width
from verity_vllm.definitions.numerics.serve3_reference import materialize

from verity_torch.frontend import train_trace
from verity_torch.frontend.outcomes import Unsupported
from verity_torch.frontend.train_step import export_steps, steps_program
from verity_torch.frontend.train_trace import ATTENTION, BACKWARD, STEP, CaptureRefused

BATCH = (torch.zeros(2, 4), torch.zeros(2, 1))


def _model(*between: torch.nn.Module) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), *between, torch.nn.Linear(3, 1))


def _step(m, batch, opts):
    for o in opts:
        o.zero_grad(set_to_none=True)
    x, y = batch
    ((m(x) - y) ** 2).mean().backward()
    for o in opts:
        o.step()


def _sgd(m: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(m.parameters(), lr=0.1, foreach=False)


def _bits(x: torch.Tensor) -> list[int]:
    return x.detach().contiguous().reshape(-1).view(torch.int32).numpy().astype(np.uint32).tolist()


def test_the_steps_are_traced_on_a_copy_each_node_with_its_step():
    m = _model()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    gm, metas, layout = export_steps([_step, _step], m, BATCH, measured=1, optimizers=[_sgd(m)])
    assert layout.params == ["0.weight", "0.bias", "1.weight", "1.bias"] and layout.buffers == []
    assert sorted(v[1] for v in layout.attrs.values()) == sorted(layout.params)
    assert (layout.measured, layout.mid_run, len(layout.inputs), len(metas)) == (1, False, 4, 4)
    calls = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert {n.meta["te_step"] for n in calls} == {0, 1}
    assert any(str(n.args[0]) == f"{STEP}1" for n in calls if "record_function_enter" in str(n.target))
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())


def test_integer_batch_tensors_of_marked_steps_are_token_inputs():
    torch.manual_seed(0)
    m = torch.nn.Embedding(8, 2)

    def step(model, batch, opts):
        model(batch).sum().backward()
        opts[0].step()

    _, _, layout = export_steps([step], m, torch.zeros(3, dtype=torch.long), measured=0, optimizers=[_sgd(m)])
    assert layout.inputs == ["tokens_0"]


def test_a_step_that_reads_a_value_into_python_is_refused_at_its_line():
    m = _model()

    def step(model, batch, opts):
        if model(batch[0]).sum() > 0:
            pass

    with pytest.raises(CaptureRefused) as e:
        export_steps([step], m, BATCH, measured=None)
    assert e.value.reason == "dynamic_control_flow" and "test_train_steps.py" in (e.value.where or "")


def test_a_step_that_deep_copies_the_model_is_refused():
    m = _model()
    with pytest.raises(CaptureRefused) as e:
        export_steps([lambda model, batch, opts: copy.deepcopy(model)], m, BATCH, measured=None)
    assert e.value.reason == "deep_copy"


class _Attending(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.proj(x).view(x.shape[0], 1, x.shape[1], 8)
        return torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True).sum()


def _attend(m, batch, opts):
    m(batch).backward()


def test_an_attention_calls_forward_and_backward_are_marked_with_the_dtype_it_computes_in():
    gm, _, _ = export_steps([_attend], _Attending(), torch.zeros(2, 4, 8), measured=None)
    marks = [str(n.args[0]) for n in gm.graph.nodes if "record_function_enter" in str(n.target)]
    assert marks.count(f"{ATTENTION}float32") >= 2 and BACKWARD in marks


def test_an_attention_call_that_returns_in_another_dtype_than_it_is_marked_is_refused(monkeypatch):
    monkeypatch.setattr(train_trace, "_computes_in", lambda query: torch.float16)
    with pytest.raises(CaptureRefused) as e:
        export_steps([_attend], _Attending(), torch.zeros(2, 4, 8), measured=None)
    assert (e.value.reason, e.value.op) == ("trace", "scaled_dot_product_attention")
    assert "an attention call marked torch.float16 returned torch.float32" in str(e.value)


def test_measured_names_one_of_the_steps():
    with pytest.raises(ValueError, match="measured is 2"):
        export_steps([_step, _step], _model(), BATCH, measured=2)


def test_the_program_computes_torchs_steps_bit_for_bit():
    m = _model()
    prog = steps_program([_step, _step], m, BATCH, measured=1, optimizers=[_sgd(m)])
    assert not isinstance(prog, Unsupported), prog
    ann = prog.annotations["train_step"]
    assert ann["outputs"] == ["param:0.weight", "param:0.bias", "param:1.weight", "param:1.bias"]
    assert list(prog.theta) == ann["inputs"][:4] and (ann["measured"], ann["mid_run"]) == (1, False)

    g = torch.Generator().manual_seed(1)
    batches = [(torch.randn(2, 4, generator=g), torch.randn(2, 1, generator=g)) for _ in range(2)]
    run = copy.deepcopy(m)
    opt = _sgd(run)
    for b in batches:
        _step(run, b, [opt])
    want = torch.cat([run.get_parameter(n.split(":", 1)[1]).detach().reshape(-1) for n in ann["outputs"]])

    params = dict(m.named_parameters())
    tab = materialize(prog, [_bits(params[n]) for n in ("0.weight", "0.bias", "1.weight", "1.bias")]
                      + [_bits(t) for b in batches for t in b])
    got = np.array([tab[o] << 32 - gate_width(prog, o) for o in prog.circuit.root_outputs()], dtype=np.uint32).view(np.float32)
    assert np.array_equal(got.view(np.uint32), want.numpy().view(np.uint32))


def test_steps_that_start_mid_run_read_no_theta():
    m = _model()
    prog = steps_program([_step], m, BATCH, measured=0, optimizers=[_sgd(m)], mid_run=True)
    assert not isinstance(prog, Unsupported), prog
    assert list(prog.theta) == [] and prog.annotations["train_step"]["mid_run"] is True


def test_an_op_no_training_rule_lowers_is_unsupported():
    m = _model(torch.nn.ReLU())
    got = steps_program([_step], m, BATCH, measured=None, optimizers=[_sgd(m)])
    assert isinstance(got, Unsupported) and "threshold_backward" in str(got)
