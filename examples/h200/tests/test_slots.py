"""The H200 demo's sequence slots and KV cache (``slots.py``): admission, finish and reuse, and rows bit for bit against a
per-request reference, written a step at once or a micro-batch of rows at a time."""
from __future__ import annotations

import pytest
import torch

from check_slots import mix, reference, requests, run
from e2e_h200.slots import K, V, Request, SlotKV, SlotShape, StepPlan, StepRows

SHAPE = SlotShape(slots=4, layers=3, ctx=8, head_dim=4, prompt=3, generate=4)


def test_requests_are_admitted_fifo_into_the_lowest_free_slots_and_their_slots_reused():
    reqs = requests(SHAPE, 11)
    kv = SlotKV(SHAPE, admit=2)
    plans: list[StepPlan] = []
    done = run(kv, reqs, on_begin=lambda _kv, plan: plans.append(plan))
    seqs = [r.seq for r in reqs]
    assert [done[s].admitted for s in seqs] == [0, 0, 1, 1, 6, 6, 7, 7, 12, 12, 13]
    assert [done[s].slot for s in seqs] == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert [done[s].finished - done[s].admitted for s in seqs] == [SHAPE.steps - 1] * 11
    assert kv.step == 18 and kv.idle
    for plan in plans:
        assert plan.slots.tolist() == sorted(set(plan.slots.tolist())) and len(plan.admitted) <= 2
        for j, seq, p in zip(plan.slots.tolist(), plan.seqs, plan.positions.tolist()):
            assert done[seq].slot == j and plan.step - done[seq].admitted == p
    assert plans[6].admitted == (0, 1) and plans[6].positions.tolist() == [0, 0, 5, 5]
    assert all(len(done[s].generated) == SHAPE.generate for s in seqs)


def test_kv_rows_and_outputs_equal_each_request_run_alone():
    reqs = requests(SHAPE, 11, seed=1)
    want = {r.seq: reference(SHAPE, r) for r in reqs}
    kv = SlotKV(SHAPE, admit=2)
    got: dict[tuple[int, int], torch.Tensor] = {}

    def on_begin(kv: SlotKV, plan: StepPlan) -> None:
        for layer in range(SHAPE.layers):
            k, v, mask = kv.read(layer)
            assert mask.tolist() == [[j <= p for j in range(SHAPE.ctx)] for p in plan.positions.tolist()]
            for i, (seq, p) in enumerate(zip(plan.seqs, plan.positions.tolist())):
                rows = want[seq][0]
                assert torch.equal(k[i, :p].view(torch.int16), rows[:p, layer, K])
                assert torch.equal(v[i, :p].view(torch.int16), rows[:p, layer, V])
                assert not k[i, p:].view(torch.int16).any() and not v[i, p:].view(torch.int16).any()

    def on_step(plan: StepPlan, rows: StepRows) -> None:
        for i, (seq, p) in enumerate(zip(rows.seqs, rows.positions.tolist())):
            got[seq, p] = rows.words[i]

    done = run(kv, reqs, on_step, on_begin)
    for r in reqs:
        rows, generated = want[r.seq]
        assert torch.equal(torch.stack([got[r.seq, p] for p in range(SHAPE.steps)]), rows)
        assert done[r.seq].generated == generated
    assert len({done[r.seq].slot for r in reqs}) < len(reqs)


def test_micro_batches_of_rows_write_and_read_what_the_whole_step_does():
    """The driver writes and reads a step's rows a micro-batch at a time (``rows``): the cache ends each step as a whole-step
    write leaves it, and a layer written for only some rows is refused."""
    reqs = requests(SHAPE, 7, seed=3)
    whole, parts = SlotKV(SHAPE, admit=2), SlotKV(SHAPE, admit=2)
    whole.submit(reqs)
    parts.submit(reqs)
    while not whole.idle:
        plan, same = whole.begin(), parts.begin()
        assert plan.slots.tolist() == same.slots.tolist()
        h = plan.tokens * 1000003 + plan.positions * 7919 + 1
        for layer in range(SHAPE.layers):
            k, v = (mix(h, layer, kind, SHAPE.head_dim).view(torch.bfloat16) for kind in (K, V))
            whole.write(layer, k, v)
            for rows in (slice(0, 1), slice(1, None)):
                parts.write(layer, k[rows], v[rows], rows)
                pk, pv, pm = parts.read(layer, rows)
                wk, wv, wm = whole.read(layer)
                assert torch.equal(pk.view(torch.int16), wk[rows].view(torch.int16))
                assert torch.equal(pv.view(torch.int16), wv[rows].view(torch.int16)) and torch.equal(pm, wm[rows])
        assert torch.equal(parts.rows().words, whole.rows().words)
        out = h % 256
        assert whole.end(out) == parts.end(out)
    kv = SlotKV(SHAPE, admit=2)
    kv.submit(reqs[:2])
    kv.begin()
    z = torch.zeros(1, SHAPE.head_dim, dtype=torch.bfloat16)
    for layer in range(SHAPE.layers):
        kv.write(layer, z, z, slice(0, 1))
    with pytest.raises(RuntimeError, match=r"wrote no KV rows for layers \[0, 1, 2\]"):
        kv.end([0, 0])


def test_rows_keep_every_bf16_bit_pattern():
    shape = SlotShape(slots=2, layers=1, ctx=4, head_dim=4, prompt=2, generate=2)
    kv = SlotKV(shape)
    kv.submit([Request(1, b"ab")])
    kv.begin()
    bits = torch.tensor([[0x7FC1, -1, -32768, 1]], dtype=torch.int16)
    kv.write(0, bits.view(torch.bfloat16), (bits ^ 0x5555).view(torch.bfloat16))
    assert kv.rows().words.reshape(-1).tolist() == bits.reshape(-1).tolist() + (bits ^ 0x5555).reshape(-1).tolist()


def test_refusals():
    with pytest.raises(ValueError, match="past the context"):
        SlotShape(slots=1, ctx=8, prompt=4, generate=6)
    kv = SlotKV(SHAPE, admit=2)
    with pytest.raises(ValueError, match="not the class's 3"):
        kv.submit([Request(1, b"ab")])
    kv.submit([Request(1, b"abc")])
    with pytest.raises(ValueError, match="already submitted"):
        kv.submit([Request(1, b"abc")])
    with pytest.raises(RuntimeError, match="call begin"):
        kv.read(0)
    kv.begin()
    with pytest.raises(RuntimeError, match="hasn't ended"):
        kv.begin()
    z = torch.zeros(1, SHAPE.head_dim)
    with pytest.raises(TypeError, match="no conversion is exact"):
        kv.write(0, z, z)
    with pytest.raises(TypeError, match="no conversion is exact"):
        kv.write(0, z[:, :2].bfloat16(), z[:, :2].bfloat16())
    kv.write(0, z.bfloat16(), z.bfloat16())
    with pytest.raises(RuntimeError, match=r"wrote no KV rows for layers \[1, 2\]"):
        kv.end([0])
    for layer in (1, 2):
        kv.write(layer, z.bfloat16(), z.bfloat16())
    with pytest.raises(ValueError, match="takes 1 output bytes"):
        kv.end([0, 0])
    assert kv.end([0]) == []
