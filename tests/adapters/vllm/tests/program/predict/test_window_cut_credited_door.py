"""`benchmarks/compile_cost/credited_door.py`'s staging of Pearl+'s credited block (`window_cut.credited`) for the door's
lowering, at the test instance K=256, TM=TN=N=1, the smallest the scheme takes: the core's parameters as the lowering's
input rows are the block's one input port bit for bit, the honest tile's ok is 1 and the tile with its y word changed has
ok 0, and the outputs are 33 16-bit words with ok at bit 512."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits.evaluation import evaluate_bits
from verity.circuits.ops.boolean.versions import block_circuit, boolean_table

from tests.program.predict import test_window_cut_credited as TC
from verity_vllm.program.predict import window_cut as WK

SPEC = importlib.util.spec_from_file_location(
    "credited_door", Path(__file__).parents[6] / "benchmarks/compile_cost/credited_door.py")
CD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CD)
K, TM, TN, N = TC.K, TC.TM, TC.TN, TC.N


def _staged() -> tuple:
    core = WK.credited_product(CD.SCHEME, K, TM, TN, N)
    honest, href = CD.tile(K, TM, TN, N)
    dishonest, dref = CD.flipped(honest, href)
    return core, (honest, href), (dishonest, dref)


def test_the_door_rows_are_the_blocks_port():
    """The lowering's input rows are the core's parameters in order, their bits the block's one port: the bits
    `test_window_cut_credited` evaluates the block on, honest and with y's sign flipped; the dishonest rows differ from
    the honest ones in y's one word alone; `reference` gives ok 1 and 0."""
    core, (honest, href), (dishonest, dref) = _staged()
    params, _ret = core.signature()
    hb, db = CD.port_bits(core, honest), CD.port_bits(core, dishonest)
    assert [p for p, _v in hb] == [p for p, _t in params]
    assert [len(v) for _p, v in hb] == [sum(BL.elements(t)) for _p, t in params]
    width = 32 * (TM * K + TN * K + TM * N + 2 * TM * TN + 2 + TM + 1024) + 8 * (96 + 32 * (TN + 2 * K))
    assert sum(len(v) for _p, v in hb) == width == 181_440
    vectors = TC._vectors()
    assert sum((v for _p, v in hb), []) == TC._bits(vectors[0][0])
    assert sum((v for _p, v in db), []) == TC._bits(vectors[1][0])
    assert [p for (p, u), (_q, v) in zip(hb, db) if u != v] == ["y"]
    assert [i for i, (u, v) in enumerate(zip(dict(hb)["y"], dict(db)["y"])) if u != v] == [31]
    want = CD.reference(K, N, href)
    assert want[-1] == 1 and CD.reference(K, N, dref)[-1] == 0
    out = [(want[i] >> j) & 1 for i in range(64) for j in range(8)] + [want[64]]
    words = CD.words16(out)
    assert len(words) == 33 and words[32] == 1
    assert [(words[k // 16] >> (k % 16)) & 1 for k in range(513)] == out


def test_the_drawn_block_fixes_the_tiles_public_rows():
    """The verifier's draw over a one-call run is the run's one credited block at the test instance, and the tile staged
    on it carries the drawn unit, rows and col bit for bit (`drawn_bits`, from the block alone)."""
    core = WK.credited_product(CD.SCHEME, K, TM, TN, N)
    block, index = CD.draw(K, TM, TN, N, bytes(64))
    assert (index, block.size, block.unit, block.rows, block.col) == (0, (K, TM, TN, N), 7, (0,), 0)
    args, _ref = CD.tile(K, TM, TN, N, block=block)
    bits = dict(CD.port_bits(core, args))
    assert CD.drawn_bits(core, block) == {p: bits[p] for p in CD.DRAWN}
    other = CD.drawn_bits(core, CD.SimpleNamespace(unit=block.unit + 1, rows=block.rows, col=block.col))
    assert other["unit"] != bits["unit"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_VLLM_CREDITED_DOOR"),
                    reason="VERITY_VLLM_CREDITED_DOOR=1: restricts the credited block (about 5 minutes and 27 GiB) and "
                           "evaluates it on bits")
def test_the_block_on_the_door_rows():
    """The block `restrict` cuts, on the staged rows: the honest tile's digest and ok 1, the changed y's ok 0. It catches
    what the rows test can't: that the block computes the scheme on the port order the door's rows stage."""
    core, (honest, href), (dishonest, dref) = _staged()
    _program, _part, _c, b = TC.credited()
    rows = [sum((v for _p, v in CD.port_bits(core, a)), []) for a in (honest, dishonest)]
    got = evaluate_bits(block_circuit(b.program.fn, boolean_table(b.program)), np.asarray(rows, dtype=np.uint8))
    got = got.reshape(2, -1).tolist()
    assert [[sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + [g[512]] for g in got] == \
        [CD.reference(K, N, href), CD.reference(K, N, dref)]
