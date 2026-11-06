"""The capture's static record against training_slowdown.flops.Counter's runtime record of the same steps: they count from one op
list (training_slowdown.flops's *_NAMES) and must agree on every step's MACs. The structure training_slowdown 4.0 scores (updated MACs, depth,
the cut) is the capture's alone: the Counter's rows are the weight-operand rule it had before."""
from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import models
from training_slowdown.capture import capture_run, capture_step
from training_slowdown.capture.script import recorded_forward_gemm_macs
from training_slowdown.flops import Counter, gather_macs
from training_slowdown.opnames import gemm_like

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


def test_a_trace_is_of_its_runs_program_where_its_product_shapes_are_the_counters():
    """training_slowdown.formalism.other_program on the counter's op log as a ledger row keeps it (training_slowdown.ledger's macs_log)
    and the capture's product rows of the same program (coordinator 20:42Z): LoRA at rank 2, trained from scratch,
    and the whole model trained run the same matrix-product shapes in both, so their terms are priced, while a capture
    at rank 3 runs shapes the rank-2 run never ran (each adapter's factors and their gradients), so its term isn't."""
    from training_slowdown.capture.products import rows
    from training_slowdown.formalism import other_program
    from training_slowdown.ledger import _macs_log

    def both(make, rank=None):
        torch.manual_seed(0)
        model, new = make()
        opt = lambda: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])  # noqa: E731
        g0 = capture_step(model, opt(), models.lm_loss, BATCH, updated=new)
        return _macs_log(counted(model, opt(), new), None), rows(g0.with_updated(g0.trained | new))["rows"]

    for make in (full, lora_new):
        log, got = both(make)
        assert got and other_program(got, log) is None, make
    log = both(lora_new)[0]
    torch.manual_seed(0)
    model = models.lora(models.TinyLM(), 3)
    new = {n for n, p in model.named_parameters() if p.requires_grad}
    g0 = capture_step(model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]), models.lm_loss,
                      BATCH, updated=new)
    o = other_program(rows(g0.with_updated(g0.trained | new))["rows"], log)
    assert o is not None and o["unmatched"] > 0 and all(3 in s for s in o["largest"]), o


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


def test_the_counters_forward_is_the_traces_forward_gemms_and_a_trained_lookup_and_elementwise_work():
    """Once the embedding trains, the counter's forward holds its lookup by token ids at the gather Definition's
    multiplexers, which |C| counts and no product is, so a trace's forward GEMMs fall short of the counter's whole
    forward by it and the forward's elementwise work; they are the counter's forward GEMMs exactly
    (`recorded_forward_gemm_macs`). Before the first update the table is θ's and its lookup computes nothing."""
    torch.manual_seed(0)
    model, new = full()
    adamw = lambda: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])  # noqa: E731
    g0 = capture_step(model, adamw(), models.lm_loss, BATCH, updated=new)
    g1 = g0.with_updated(g0.trained | new)
    cnt = counted(model, adamw(), new)
    lookup = gather_macs("embedding", tuple(model.emb.weight.shape), None, BATCH[0].numel() * model.emb.weight.shape[1])
    for i, g in ((0, g0), (1, g1)):
        step = cnt["steps"][i]
        log = [r for r in cnt["structures"][step["struct"]]["log"] if r[0] == "fwd"]
        forward = sum(v for k, v in step["by"].items() if k.startswith("fwd/"))
        assert recorded_forward_gemm_macs(cnt, i) == g.phase_macs()["forward"] > 0, i
        assert [r[5] for r in log if r[2] == "embedding"] == ([lookup] if i else []), i
        assert forward == g.phase_macs()["forward"] + (lookup if i else 0) + sum(
            r[5] for r in log if not gemm_like(r[2]) and r[2] != "embedding"), i
    assert lookup > 0
