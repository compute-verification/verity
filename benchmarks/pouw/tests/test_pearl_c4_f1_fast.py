"""`f1_fast`, the verifier's vectorised F1′: its noise law is the reference's to float rounding, and its evaluation
of C (certified modal byte, 4-lookup lower bound on p_t, safe-side charge for uncertified blocks) is the rule's
(`pearl_c4.change_probabilities`) to float rounding on every row checked, with R1's decisions unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c4"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clear"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "core" / "protocols" / "pouw" / "tests"))
import f1_fast as Q  # noqa: E402
import pearl_c4_vectors as V  # noqa: E402
import test_pouw_pearl_c4 as T  # noqa: E402
from verity.ml.protocols.pouw.schemes import pearl_c4 as F  # noqa: E402
from verity.ml.protocols.pouw.schemes import pearl_c_work as W  # noqa: E402
import clear_work as CW  # noqa: E402


def test_row_law_is_the_reference_law():
    for i in range(16):
        e = tuple(F.line(bytes([i]) * 32, F.ROLE_EA, i))
        lo, pmf = F.noise_law(e)
        lo2, pmf2 = Q.row_law(e)
        assert lo == lo2 and len(pmf) == len(pmf2) and np.abs(np.array(pmf) - pmf2).max() < 1e-15
    assert Q.row_law((0,) * 32) == (0, pytest.approx(np.ones(1)))


def _side(rows, k=1024):
    e = [F.line(b"\x03" * 32, F.ROLE_EA, i) for i in range(len(rows))]
    return F.form(F.NVFP4, rows, e, F.basis(b"\x04" * 32, F.ROLE_FA, k)), e


@pytest.mark.parametrize("family", ["spike", "narrow-cell", "gaussian"])
def test_fast_f1_is_the_rule(family):
    rows = {"spike": [T.spike_row(i) for i in range(3)], "narrow-cell": [T.narrow_cell_row(i) for i in range(3)],
            "gaussian": [F.parse_row(r, 1024) for r in V.rows("rules", "A", 3, 1024)]}[family]
    side, e = _side(rows)
    for i in range(len(rows)):
        dead = F.salt_dead(F.NVFP4, side.x[i], side.alpha[i], side.ae[i])
        rule, uncertified = F.row_f1(F.NVFP4, side, e, i, dead)
        fast, counts = Q.split_units_fast(side.x[i], side.alpha[i], e[i], side.ae[i], side.scales[i], dead)
        assert counts["uncertified"] == uncertified == 0
        assert abs(float(fast) - float(rule)) <= 1e-9 * max(1.0, float(rule))


def test_fast_path_keeps_r1_and_the_tile_on_r1s_tile():
    """PearlC4(f1=row_units) as the verifier's path: the spike row is still over R1's bound, the Gaussian rows still
    under it, and the tile still passes with row 0 excluded."""
    s, ctx, tile, a_rows, b_rows, ops = T.r1_tile()
    k = ctx.matmul.shape.k
    fast = F.PearlC4("nvfp4", f1=Q.row_units)
    rec = W.UnitRecord(ctx.matmul, 4, k)
    assert [fast.row_over_debit(rec, ops, "A", p) for p in range(4)] == [s.row_over_debit(rec, ops, "A", p) for p in range(4)]
    assert not any(fast.row_over_debit(rec, ops, "B", p) for p in range(4))
    assert CW.check_opened(fast, W.UnitRecord(ctx.matmul, 4, k, (0,)), tile, a_rows, b_rows, ops) is None
    d_fast = fast.tile_cap(rec, ops, range(4), range(4))[0]
    d_rule = s.tile_cap(rec, ops, range(4), range(4))[0]
    assert abs(d_fast - d_rule) <= Fraction(1, 10 ** 9) * max(1, d_rule)
