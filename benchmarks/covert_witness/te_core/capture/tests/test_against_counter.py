"""The capture's static record against te_core.flops.Counter's runtime record of the same steps: they count from one op
list (te_core.flops's *_NAMES) and must agree on every step's MACs. The structure te_core 4.0 scores (updated MACs, depth,
the cut) is the capture's alone: the Counter's rows are the weight-operand rule it had before."""
from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import models
from te_core.capture import capture_run, capture_step
from te_core.flops import Counter

N = 3


def counted(model, opt, new: set[str]) -> dict:
    theta = {k: v.detach().clone() for k, v in model.state_dict().items() if k not in new}
    c = Counter(theta, digests=False)
    c.register(model.state_dict(keep_vars=True))
    with sdpa_kernel([SDPBackend.MATH]), c:
        for _ in range(N):
            c.begin_step()
            models.lm_loss(model, BATCH).backward()
            opt.step()
            opt.zero_grad()
            c.end_step()
    c.finish()
    return c.record()


def captured(model, opt, new: set[str]) -> dict:
    g0 = capture_step(model, opt, models.lm_loss, BATCH, updated=new)
    return capture_run([(1, g0), (N - 1, g0.with_updated(g0.trained | new))]).record()


_g = torch.Generator().manual_seed(1)
BATCH = (torch.randint(0, 32, (2, 4), generator=_g), torch.randint(0, 32, (2, 4), generator=_g))


def full():
    return models.TinyLM(), set()


def top():
    m = models.TinyLM()
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.blocks[-1].parameters():
        p.requires_grad_(True)
    return m, set()


def lora_new():
    m = models.lora(models.TinyLM(), 2)
    return m, {n for n, p in m.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("make", [full, top, lora_new])
def test_the_capture_and_the_counter_agree_step_for_step(make):
    torch.manual_seed(0)
    model, new = make()
    adamw = lambda: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])  # noqa: E731
    cap = captured(model, adamw(), new)
    cnt = counted(model, adamw(), new)
    keys = ("tokens", "matmul_macs", "elementwise_macs", "macs")
    for i in range(N):
        assert [cnt["steps"][i][k] for k in keys] == [cap["steps"][i][k] for k in keys], i
    assert cnt["totals"]["macs"] == cap["totals"]["macs"]
