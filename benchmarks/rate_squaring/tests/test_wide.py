"""`wide.py` against what it restates: the codec-mix study's table v5, whose κ, x₁ and d its formulas
must give back at kv.cu as built, and p512's ceiling models (16 limbs) at their pinned cycle counts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("wide", HERE / "wide.py")
wide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide)  # type: ignore[union-attr]


def test_x1_is_codec_mix_x1_of() -> None:
    assert [round(wide.x1_ref(b), 1) for b in (512, 768, 1024)] == [175.0, 338.6, 600.0]
    assert wide.r_ref(512) == pytest.approx(wide.REF)


@pytest.mark.parametrize("name", list(wide.ROWS))
def test_kappa_and_d_are_table_v5s(name: str) -> None:
    b, k, lib, kappa, scale, d = wide.ROWS[name]
    assert wide.kappa_of(lib, k, b) == pytest.approx(kappa, abs=2e-3)
    for point, want in zip(wide.STEP, d):
        assert wide.step_d(point, kappa, k, wide.x1_ref(b), scale) == pytest.approx(want, abs=2e-3)


def test_p512_ceilings_are_the_reports() -> None:
    assert round(wide.col_cycles(16)) == 713
    assert round(wide.kara_cycles(16, 1)) == 639
    assert wide.pipe_cycles(16, 0) == pytest.approx(152 * 4.02)
    assert wide.pipe_cycles(16, 1) == pytest.approx(124 * 4.02)
    c = wide.ceilings(512)
    assert [round(c[n], 2) for n in ("column form at the probe's mix", "Karatsuba x1 at the probe's mix",
                                     "column form, multiply pipe only", "Karatsuba x1, multiply pipe only")] == \
        [1.38, 1.53, 1.61, 1.97]


def test_p512_adversary_is_codec_mix_fold5() -> None:
    a = wide.adversaries(512)["measured best, host measured"]
    assert a == pytest.approx(6.53 / 5.13 * 1.076, rel=5e-3)
    assert wide.h_of(512, wide.ifma_bound(512)) == pytest.approx(0.129, abs=1e-3)


def test_best_form_decode_moves_kappa_by_its_gain() -> None:
    lib, k, b = "p1024-m", 2, 1024
    assert wide.kappa_of(lib, k, b, wide.g_of(b)) < wide.kappa_of(lib, k, b)
    assert wide.decode_a_byte(512, 2, wide.KV_SQ) == pytest.approx(1.0)
    assert wide.decode_a_byte(1024, 2, wide.KV_SQ) == pytest.approx(64 / 128 * wide.KV_SQ[512] / wide.KV_SQ[1024])
