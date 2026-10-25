"""The H200 demo's sequence slots and KV cache (``slots.py``): admission, finish and reuse, rows bit for bit against a
per-request reference, and the rows the replay opens through the stand-in commit call."""
from __future__ import annotations

import pytest
import torch

from check_slots import reference, requests, run
from e2e_h200.slots import K, V, Request, SlotKV, SlotShape, StepPlan, StepRows, commit_step, verify_row

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

    def on_step(plan: StepPlan, rows: StepRows) -> int:
        for i, (seq, p) in enumerate(zip(rows.seqs, rows.positions.tolist())):
            got[seq, p] = rows.words[i].reshape(SHAPE.layers, 2, SHAPE.head_dim)
        return 0

    done = run(kv, reqs, on_step, on_begin)
    for r in reqs:
        rows, generated = want[r.seq]
        assert torch.equal(torch.stack([got[r.seq, p] for p in range(SHAPE.steps)]), rows)
        assert done[r.seq].generated == generated
    assert len({done[r.seq].slot for r in reqs}) < len(reqs)


def test_refs_open_every_row_the_attention_read_from_the_steps_that_committed_it():
    shape = SlotShape(slots=4, layers=3, ctx=8, head_dim=4, prompt=3, generate=4, row_words=2)
    reqs = requests(shape, 9, seed=2)
    kv = SlotKV(shape, admit=2)
    commits = {}
    base = 5

    def on_step(plan: StepPlan, rows: StepRows) -> int:
        other = torch.full((plan.n, base, shape.row_words), 7, dtype=torch.int16)
        words = torch.cat([other, rows.words], dim=1)
        commits[plan.step] = commit_step(plan.step, (rows.slots, words), slots=shape.slots, salt_key=bytes([plan.step]) * 32)
        return base

    done = run(kv, reqs, on_step)
    assert len(done) == len(reqs)
    for r in reqs:
        rows, _ = reference(shape, r)
        for layer in range(shape.layers):
            for p in (0, shape.steps // 2, shape.steps - 1):
                refs = kv.refs(r.seq, layer, p)
                assert [(x.position, x.kind) for x in refs] == [(j, kind) for j in range(p + 1) for kind in (K, V)]
                for x in refs:
                    assert x.step == done[r.seq].admitted + x.position and x.slot == done[r.seq].slot
                    c = commits[x.step]
                    for q in range(x.rows):
                        o = c.open(x.slot, x.index + q)
                        assert verify_row(o, c.root, slots=shape.slots, row_bits=c.row_bits)
                        span = slice(q * shape.row_words, (q + 1) * shape.row_words)
                        assert o.words.tolist() == rows[x.position, layer, x.kind, span].tolist()


def test_a_tampered_opening_fails():
    words = torch.arange(12, dtype=torch.int16).reshape(2, 3, 2)
    c = commit_step(4, (torch.tensor([1, 3]), words), slots=4, salt_key=bytes(32))
    o = c.open(3, 2)
    assert verify_row(o, c.root, slots=4, row_bits=32) and o.words.tolist() == [10, 11]
    flipped = bytes([o.row[0] ^ 1]) + o.row[1:]
    assert not verify_row(type(o)(**{**o.__dict__, "row": flipped}), c.root, slots=4, row_bits=32)
    assert not verify_row(type(o)(**{**o.__dict__, "slot": 1}), c.root, slots=4, row_bits=32)
    assert not verify_row(type(o)(**{**o.__dict__, "step": 5}), c.root, slots=4, row_bits=32)
    assert not verify_row(o, c.slot_roots[3], slots=4, row_bits=32)
    other = commit_step(4, (torch.tensor([1, 3]), words), slots=4, salt_key=bytes(32))
    assert other.root == c.root
    assert commit_step(4, (torch.tensor([1, 3]), words), slots=4, salt_key=bytes([1]) * 32).root != c.root


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
    with pytest.raises(KeyError, match="hasn't been committed"):
        kv.refs(1, 0, 1)
