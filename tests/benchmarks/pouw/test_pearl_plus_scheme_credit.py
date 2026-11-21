"""Pearl+'s resource scheme (`verity.protocols.accounting.compute.pouw.scheme`) against the run verifier's credit
(``benchmarks/pouw/pearl_c_vllm/verify_run.py``, `Credit.lean`'s Python twin): a block's cost is a whole tile's credited
MACs, and each charge's bound is a key of the work the verifier publishes."""
from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from verity.protocols.accounting.compute.pouw import scheme as PS
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c import PearlC
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_work import UnitRecord
from verity.protocols.accounting.compute.pouw.verifier.protocol import Matmul, Shape

VERIFY_RUN = Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "pearl_c_vllm" / "verify_run.py"


def _verify_run():
    name = "pearl_c_vllm_verify_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, VERIFY_RUN)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_the_verifiers_protocol_and_assumption() -> None:
    vr = _verify_run()
    assert PS.PEARL_PLUS.protocol == vr.PROTOCOL and PS.PEARL_PLUS.assumptions == (vr.ASSUMPTION_H100,)


def test_block_cost_is_the_credits_macs() -> None:
    """A credited tile, `PearlPlusTileCoreWords_v5`, costs what the verifier credits a whole-width tile: its TR real
    rows' MACs at K, filler rows crediting none."""
    vr = _verify_run()
    scheme = PearlC()
    tm, tn = scheme.tile
    (block,) = PS.PEARL_PLUS.blocks
    for real in (tm, tm - 12):
        record = UnitRecord(Matmul(0, Shape(tm, 1024, tn), 0), real_rows=real, k_kept=1024)
        credit = vr.credit_at_assumption([SimpleNamespace(scheme=scheme, records=[record])], 3, Fraction(1, 100))
        assert block.cost({"K": 1024, "TM": tm, "TR": real, "TN": tn}) == credit["checked_macs"] == real * tn * 1024
    assert Fraction(credit["rho"]) == scheme.cap
    assert {c.bound for c in PS.PEARL_PLUS.charges} <= {*credit, *vr.grid_work(credit, credit)}


def test_its_credited_product_is_one_block_a_grid_tile_in_the_lean_verifiers_order_and_tickets() -> None:
    """`verify_run.credited_blocks` on two passes of Pearl+'s test instance (k = 128) at 32 x 16 tiles: ids per pass in `Run.lean`'s
    `tile?` order (call by call, row-major, the last row block narrower), and each ticket `ticketWeight`'s, |credited rows|
    x |credited columns| x `wref`(1, |K|, n) / n (the H100 record's `wref`, written here as `Records.lean` states it), over the
    lcm of every pass's denominators; each block's real-row count is its rows below the call's ``real_rows``. Weight 1's excluded B row 20 takes its block of 64 columns, so its calls' tiles have
    ticket 0 and `verify_run.sampled_blocks` never takes them."""
    import math

    from verity.commitments.randomness import derive
    from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes

    vr = _verify_run()
    scheme = pouw_schemes.get("pearl-c-h100-v1-h3", k_min=32, tile=(32, 16))
    k = 128

    def call(index: int, m: int, n: int, weight: int, real: int, **rows: tuple[int, ...]) -> UnitRecord:
        return UnitRecord(Matmul(index, Shape(m, k, n), weight), real, k, **rows)

    passes = [[call(0, 32, 16, 0, 20, excluded_a=(5,)), call(1, 16, 32, 1, 16, excluded_b=(20,))],
              [call(2, 48, 16, 0, 40), call(3, 16, 32, 1, 16, excluded_b=(20,))]]
    blocks = vr.credited_blocks(scheme, passes)
    r16, r32, r48 = tuple(range(16)), tuple(range(32)), tuple(range(32, 48))
    c0, c1 = tuple(range(16)), tuple(range(16, 32))
    want = [((0, 0), 0, 0, r32, c0, 0, (k, 32, 16, 16), 20, 19 * 16), ((0, 1), 1, 1, r16, c0, 0, (k, 16, 16, 32), 16, 0),
            ((0, 2), 1, 1, r16, c1, 1, (k, 16, 16, 32), 16, 0), ((1, 0), 0, 2, r32, c0, 0, (k, 32, 16, 16), 32, 32 * 16),
            ((1, 1), 0, 2, r48, c0, 0, (k, 16, 16, 16), 8, 8 * 16), ((1, 2), 1, 3, r16, c0, 0, (k, 16, 16, 32), 16, 0),
            ((1, 3), 1, 3, r16, c1, 1, (k, 16, 16, 32), 16, 0)]
    assert [(b.id, b.record, b.unit, b.rows, b.cols, b.col, b.size, b.real) for b in blocks] == [w[:8] for w in want]

    def wref(m: int, k: int, n: int) -> int:
        return m * n * (k + 32 * (k // 128)) - 32 * m * n + m * k * 32 + 2 * 64 * m * n + 68 * m * k

    weights = [Fraction(wref(1, k, b.size[3]) * cells, b.size[3]) for b, (*_, cells) in zip(blocks, want)]
    den = math.lcm(*(w.denominator for w in weights))
    assert [b.ticket for b in blocks] == [int(w * den) for w in weights]
    key = derive(b"verifier" * 4, "verity/pouw/pearl-plus-run-draw/v1", {"run": bytes(64)})
    taken = vr.sampled_blocks(scheme, passes, key, 16)
    assert len(taken) == 16 and {b.id for b in taken} <= {(0, 0), (1, 0), (1, 1)}
