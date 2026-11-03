"""`capture_run` / `schedule_updated`: a schedule that trains the top block, then everything, is two captures and four
classifications, and |C| is Σ steps · MACs exactly."""
from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from models import TinyLM, lm_loss
from te_core.capture import capture_run, capture_step, schedule_updated

B, S = 2, 4


def batch():
    return torch.empty(B, S, dtype=torch.long), torch.empty(B, S, dtype=torch.long)


def test_schedule_updated():
    sched = [{"a"}] * 3 + [{"a", "b"}] * 2 + [{"b"}]
    assert schedule_updated(sched) == [set(), {"a"}, {"a"}, {"a"}, {"a", "b"}, {"a", "b"}]
    assert schedule_updated(sched, registered={"a"}) == [set(), set(), set(), set(), {"b"}, {"b"}]


def test_top_then_full_run():
    model = TinyLM()
    names = [n for n, _ in model.named_parameters()]
    top_names = {n for n in names if n.startswith("blocks.1.")}
    calls = []

    def top():
        calls.append("top")
        m = TinyLM()
        m.requires_grad_(False)
        m.blocks[-1].requires_grad_(True)
        return capture_step(m, torch.optim.AdamW(m.blocks[-1].parameters()), lm_loss, batch())

    def full():
        calls.append("full")
        return capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, batch())

    n_top, n_full = 10, 20
    sched = [top_names] * n_top + [set(names)] * n_full
    upd = schedule_updated(sched)
    g_top, g_full = top(), full()
    segments = [(1, g_top.with_updated(upd[0])), (n_top - 1, g_top.with_updated(upd[1])),
                (1, g_full.with_updated(upd[n_top])), (n_full - 1, g_full.with_updated(upd[n_top + 1]))]
    run = capture_run(segments)
    assert calls == ["top", "full"]
    assert run.n_steps == n_top + n_full and len(run.steps) == 4
    assert run.macs == n_top * g_top.macs + n_full * g_full.macs
    assert run.tokens == (n_top + n_full) * B * S
    # step 0 reads θ: nothing is updated; the first full step reads only the top block as new
    assert run.steps[0][1].role_macs("updated") == 0
    T, H, I, V = B * S, 16, 32, 32
    top_fwd = 4 * T * H * H + 2 * T * H * I + 2 * T * S * H + T * H * V   # the top block's forward GEMMs and the head
    assert run.steps[2][1].role_macs("updated") == g_top.role_macs("updated") == top_fwd
    assert run.steps[3][1].role_macs("updated") == g_full.role_macs("updated")
    assert run.role_macs("producing") == n_top * g_top.role_macs("producing") + n_full * g_full.role_macs("producing")
    want = (n_top - 1) * g_top.deep_macs(1, "matmuls") + run.steps[2][1].deep_macs(1, "matmuls") \
        + (n_full - 1) * g_full.deep_macs(1, "matmuls")
    assert run.phi(1, "matmuls") == Fraction(want, run.c_macs)


def test_thunks_capture_once_and_identical_graphs_merge():
    model = TinyLM()
    calls = []

    def step():
        calls.append(1)
        return capture_step(model, torch.optim.AdamW(model.parameters()), lm_loss, batch())

    run = capture_run([(3, step), (4, step), (5, step)])
    assert len(calls) == 1 and run.steps[0][0] == 12 and len(run.steps) == 1
    again = capture_run([(3, step()), (4, step())])   # two captures of one structure merge by digest
    assert len(again.steps) == 1 and again.n_steps == 7
    with pytest.raises(ValueError):
        capture_run([(-1, step)])
