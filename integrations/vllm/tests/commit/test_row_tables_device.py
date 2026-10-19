"""`row_tables_device`: `row_tables.step_roots` on the GPU (`row_tables_device.step_roots`) and its CPU twin, each held bit for bit to the CPU reference
(`row_tables.step_roots`, over `verity.ml.commitments.row_table`) on random steps.  The steps cover a row's segment cases (K + 4
a multiple of 1024 and two past one, codes whose count is not a multiple of 8), one-row and non-power-of-two tables, a one-table
step, rows out of rank order, tables out of step order, and an engine step past 2^32 (the nonce's second word).  The twin needs a
C++ compiler; the device tests need CUDA and nvcc, and skip elsewhere."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from verity.ml.commitments import row_table as RT
from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as D

CTX, KEY = bytes(range(64)), bytes(range(100, 132))
STEPS = [
    {"model.layers.0.mlp.down_proj": (1, 2)},
    {"model.layers.0.self_attn.qkv_proj": (3, 1020), "model.layers.0.mlp.down_proj": (5, 1022), "lm_head": (2, 2046)},
    {"model.layers.1.self_attn.o_proj": (17, 4096), "model.layers.0.self_attn.o_proj": (9, 4100),
     "model.layers.0.mlp.down_proj": (33, 14336)},
]


def outputs(shapes: dict[str, tuple[int, int]], seed: int) -> dict:
    """A step's quantize outputs: random codes and scales, each table's (request, token row) ids shuffled."""
    rng = np.random.default_rng(seed)
    out = {}
    for op, (b, k) in shapes.items():
        ids = [(f"req{i % 3}", i) for i in range(b)]
        rng.shuffle(ids)
        out[op] = (ids, rng.integers(-128, 128, (b, k), dtype=np.int8), rng.standard_normal(b).astype(np.float32))
    return out


def test_the_table_domain_ids_are_row_tables():
    for step, index, leaves in [(0, 0, 1), (9, 3, 32), (2**33 + 1, 127, 300), (5, 1000, 2**16 + 1)]:
        assert D.table_domain_id(CTX, step, index, leaves) == RT.table_domain(CTX, step, index, leaves).domain_id


@pytest.mark.parametrize("engine_step", [7, 2**33 + 5])
@pytest.mark.parametrize("shapes", STEPS, ids=["one-row", "segment-edges", "llama-shapes"])
def test_the_cpu_twin_is_the_cpu_reference(shapes, engine_step):
    outs = outputs(shapes, engine_step)
    assert D.host_step(CTX, engine_step, outs, KEY) == RTB.step_roots(CTX, engine_step, outs, KEY)


def test_tables_that_share_one_rows_list_are_ranked_alike():
    outs = outputs({"model.layers.0.self_attn.qkv_proj": (6, 4096), "model.layers.0.mlp.down_proj": (6, 14336)}, 5)
    ids = outs["model.layers.0.mlp.down_proj"][0]
    outs = {op: (ids, c, s) for op, (_ids, c, s) in outs.items()}
    assert D.host_step(CTX, 9, outs, KEY) == RTB.step_roots(CTX, 9, outs, KEY)


def test_the_plan_refuses_what_step_roots_refuses():
    for bad in [{"x": ([("r", 0)], 1, 3)}, {"x": ([("r", 0), ("r", 0)], 2, 4)}, {"x": ([("r", 0)], 2, 4)}, {}]:
        with pytest.raises(ValueError):
            D.plan(CTX, 1, bad, KEY)
    with pytest.raises(ValueError):
        D.plan(bytes(32), 1, {"x": ([("r", 0)], 1, 4)}, KEY)


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or D.nvcc() is None:
        pytest.skip("the device path needs CUDA and nvcc")
    return torch


def _on(torch, outs: dict) -> dict:
    return {op: (ids, torch.from_numpy(c).cuda(), torch.from_numpy(s).cuda()) for op, (ids, c, s) in outs.items()}


@pytest.mark.parametrize("shapes", STEPS, ids=["one-row", "segment-edges", "llama-shapes"])
def test_the_device_path_is_the_cpu_reference(shapes):
    torch = _cuda()
    outs = outputs(shapes, 11)
    pending = D.step_roots(CTX, 2**33 + 5, _on(torch, outs), KEY)
    assert isinstance(pending, D.Pending)
    assert pending.result() == RTB.step_roots(CTX, 2**33 + 5, outs, KEY)
    timed = D.step_roots(CTX, 2**33 + 5, _on(torch, outs), KEY, timing=True)
    assert timed.kernel_ms() > 0 and timed.result() == pending.result()


def test_the_side_stream_reads_the_codes_the_serving_stream_wrote_before_the_launch():
    torch = _cuda()
    outs = outputs(STEPS[2], 12)
    dev = {op: (ids, torch.zeros_like(c), torch.zeros_like(s)) for op, (ids, c, s) in _on(torch, outs).items()}
    big = torch.ones(4096, 4096, device="cuda")
    for _ in range(8):
        big = big @ big / 4096
    for op, (_ids, c, s) in dev.items():
        c.copy_(torch.from_numpy(outs[op][1]), non_blocking=True)
        s.copy_(torch.from_numpy(outs[op][2]), non_blocking=True)
    assert D.step_roots(CTX, 3, dev, KEY).result() == RTB.step_roots(CTX, 3, outs, KEY)


def test_record_ready_hands_over_the_ready_steps_in_launch_order():
    torch = _cuda()
    steps = {s: outputs(STEPS[s % 3], 20 + s) for s in range(4)}
    queue = deque(D.step_roots(CTX, s, _on(torch, o), KEY) for s, o in steps.items())
    queue[-1].done.synchronize()
    got: list[RTB.StepRoots] = []
    assert D.record_ready(queue, got.append) == 4 and not queue
    assert got == [RTB.step_roots(CTX, s, o, KEY) for s, o in steps.items()]
    assert D.record_ready(queue, got.append) == 0
