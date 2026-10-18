"""proved_ticks at the hand-checked point, H1 + fx1 on DS-120: ♭ is an IMMA.16832, 1/39 LDSM_X4 and fx1's 8 ALU and
8 IMAD warp instructions a unit, B_♭ with ε_skip's work is Lean's 17.097571, and a GPU-second of linears at a cycle a
unit is (1 − 0.599%) · 17.097571 s; QKᵀ counts at 1.25, and a block divides by its GPUs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_proved_ticks", Path(__file__).parents[1] / "proved_ticks.py")
P = importlib.util.module_from_spec(spec)
sys.modules["pouw_proved_ticks"] = P
spec.loader.exec_module(P)

S = P.sheet()
#: 4096 int8 products a cycle on every SMSP of one GPU for one second at f_max
SECOND = 4096 * 4 * 188 * 2430e6


def test_the_sheet_is_h1_fx1_on_ds120():
    assert (S.family, S.vector, S.design, S.sms, S.fmax_mhz) == ("sm_120", "DS-120", "dither-c-peel-fx1", 188, 2430)
    assert P.SHEET_EPS == 0.00599


def test_b_flat_is_the_floor_programs_issue_bound_with_its_skippable_work():
    data = P.device(S.family)
    assert P.EB.prices(S.design, data, S.vector).mma == pytest.approx(1 + 1 / 39 + 8 + 8, abs=5e-7)
    assert P.unit_cycles(S) == pytest.approx(17.097571, abs=5e-7)     # Accounting.TS.Headline.Sm120.flat (#2482)


def test_a_gpu_second_of_linears_is_one_minus_eps_of_b_flat_seconds():
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1.0) == pytest.approx(0.99401 * 17.097571,
                                                                                                 abs=1e-6)
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, 0.0, 1.0) == pytest.approx(17.097571, abs=1e-6)


def test_qk_counts_at_cohalfs_words_and_a_block_divides_by_its_gpus():
    one = P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1.0)
    assert P.proved_block_ticks({"mac:qk": SECOND}, S, P.SHEET_EPS, 1.0) == pytest.approx(1.25 * one)
    mixed = {"mac:linears": SECOND, "mac:lm_head": SECOND, "mac:qk": SECOND, "mac:pv": SECOND}
    assert P.proved_block_ticks(mixed, S, P.SHEET_EPS, 1.0) == pytest.approx(4.25 * one)
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1e-4, gpus=8) == pytest.approx(one * 1e4 / 8)


@pytest.mark.parametrize("census", [{"gate": 1.0}, {"mac:fp8": 1.0}, {"alu:fx1": 1.0}, {"mac:linears": -1.0}])
def test_work_outside_the_designs_products_is_refused(census):
    with pytest.raises(ValueError, match="LP over the union"):
        P.units(census, S)


def test_a_family_without_a_sheet_clock_is_refused():
    with pytest.raises(ValueError, match="FMAX_MHZ"):
        P.sheet("sm_90")
