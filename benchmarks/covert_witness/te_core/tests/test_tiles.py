"""te_core.tiles: reading A's matrix-product tiles hold the most MACs their reads allow, place every product the log
counts at its own MACs, and take the operand the log can't say is affected as free."""
from __future__ import annotations

import itertools
import json

import pytest
import torch
from torch.utils import flop_counter

from te_core import tiles
from te_core.flops import _shape_macs
from te_core.gates import per_mac


def brute(R, alpha, beta, b, m, K, n) -> float:
    """The most b'·m'·k'·n' over integer boxes within the dims with b'·k'·(α·m' + β·n') ≤ R."""
    best = 0
    for bb, mm, kk, nn in itertools.product(range(1, b + 1), range(1, m + 1), range(1, K + 1), range(1, n + 1)):
        if bb * kk * (alpha * mm + beta * nn) <= R:
            best = max(best, bb * mm * kk * nn)
    return best


@pytest.mark.parametrize("R, alpha, beta, dims", [
    (100, 1, 1, (1, 10, 50, 10)), (100, 0, 1, (1, 10, 20, 300)), (100, 1, 1, (1, 6, 5, 9)),
    (60, 2, 3, (2, 4, 6, 5)), (7, 1, 1, (1, 3, 2, 8)), (40, 0, 4, (3, 4, 2, 6)),
])
def test_a_tiles_volume_is_at_least_every_boxs_within_its_reads(R, alpha, beta, dims):
    """`volume` is the relaxation (fractional boxes) of the integer search, so it is never below it: a cap below a
    real tile's would overclaim."""
    v = tiles.volume(R, alpha, beta, *dims)
    assert v >= brute(R, alpha, beta, *dims)
    b, m, K, n = dims
    assert v <= b * m * K * n


def test_a_tile_spans_rows_and_columns_and_takes_the_better_operand_free():
    """With every row and column's one-coordinate slice fitting, V = m·n·min(bK, R/(αm + βn)); otherwise the best
    m'·n' on the budget's line; `cap` is the better free operand, at most the block's gates."""
    assert tiles.volume(100, 1, 1, 1, 10, 50, 10) == pytest.approx(500)
    assert tiles.volume(100, 1, 1, 1, 10, 5, 1000) == pytest.approx(10 * 90)
    assert tiles.volume(100, 1, 1, 1, 1000, 5, 1000) == pytest.approx(50 * 50)
    assert tiles.volume(100, 0, 4, 1, 10, 50, 1000) == pytest.approx(25 * 10)
    op = {"gates": 2e6, "macs": 1e6, "widths": [16, 16]}
    prod = (1, 2048, 64, 192)
    free_a = tiles.volume(80_000, 0, 16, *prod)
    assert free_a > tiles.volume(80_000, 16, 0, *prod)
    assert tiles.cap(op, prod, 80_000, 1e30) == pytest.approx(2 * free_a)
    assert tiles.cap(op, prod, 80_000, 5.0) == 5.0


def meta(*shapes):
    return [torch.empty(s, device="meta") for s in shapes]


@pytest.mark.parametrize("op, shapes", [
    ("mm", ([2048, 64], [64, 192])), ("addmm", ([192], [2048, 64], [64, 192])), ("bmm", ([4, 32, 16], [4, 16, 8])),
    ("baddbmm", ([4, 32, 8], [4, 32, 16], [4, 16, 8])), ("addbmm", ([32, 8], [4, 32, 16], [4, 16, 8])),
    ("mv", ([32, 16], [16])), ("addmv", ([32], [32, 16], [16])), ("dot", ([16], [16])), ("addr", ([8, 6], [8], [6])),
])
def test_products_hold_the_ops_macs_as_the_counter_counts_them(op, shapes):
    (prods,) = [tiles.products(op, [list(s) for s in shapes])]
    assert sum(b * m * K * n for b, m, K, n in prods) == _shape_macs(getattr(torch.ops.aten, op).default, meta(*shapes))


def test_attention_is_its_products_as_torchs_formula_counts_them_shared_key_heads_stacked():
    q, k, v = [2, 4, 16, 8], [2, 2, 32, 8], [2, 2, 32, 6]
    fwd = tiles.products("_scaled_dot_product_flash_attention", [q, k, v])
    assert sum(b * m * K * n for b, m, K, n in fwd) == flop_counter.sdpa_flop_count(q, k, v) // 2
    assert fwd[0] == (4, 32, 8, 32)
    out = [2, 4, 16, 6]
    bwd = tiles.products("_scaled_dot_product_flash_attention_backward", [out, q, k, v])
    assert sum(b * m * K * n for b, m, K, n in bwd) == flop_counter.sdpa_backward_flop_count(out, q, k, v) // 2


def test_a_log_places_its_gemms_and_lets_the_rest_ride_free():
    """An op's products are kept only where their MACs times its count are its MACs; an elementwise op and the MACs
    of a class no op row holds have none."""
    log = {"by": {"fwd/updated": 3 * 2048 * 64 * 192 + 1000, "bwd/grad": 64 * 2048 * 192},
           "ops": {f"fwd updated mm {json.dumps([[2048, 64], [64, 192]])} bfloat16": [3, 3 * 2048 * 64 * 192],
                   f"bwd grad mm {json.dumps([[64, 2048], [2048, 192]])}": [1, 64 * 2048 * 192],
                   f"fwd updated mul {json.dumps([[2048, 64], [64]])}": [1, 1000],
                   f"fwd updated mm {json.dumps([[2048, 64], [64, 192]])} float32": [1, 7]}}
    ops = tiles.of_log(log)
    by = {(o["class"], o["op"], o["macs"]): o for o in ops}
    mm = by[("updated", "mm", 3 * 2048 * 64 * 192)]
    assert mm["products"] == [[1, 2048, 64, 192]] and mm["widths"] == [16, 16]
    assert mm["gates"] == pytest.approx(mm["macs"] * per_mac("gemm/bf16"))
    assert by[("grad", "mm", 64 * 2048 * 192)]["products"] == [[1, 64, 2048, 192]]
    assert by[("updated", "mul", 1000)]["products"] == [] and by[("updated", "mm", 7)]["products"] == []
    assert not any(o["op"] == "unlisted" for o in ops)
    assert tiles.unplaced(ops) == pytest.approx(sum(o["gates"] for o in ops if not o["products"])
                                                / sum(o["gates"] for o in ops))
    caps = tiles.caps(ops, "fwd", 80_000, 1e30)
    assert sum(g for g, _ in caps) == pytest.approx(sum(o["gates"] for o in ops if o["phase"] == "fwd"))
