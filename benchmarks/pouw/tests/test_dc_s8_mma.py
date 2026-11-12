"""dc/s8_mma.py's two readers on synthetic inputs: the SASS reader keeps each kernel's tensor instructions, and the
word check passes a device that returns the sm_120 step's words and names the one word that differs. The probe itself
needs an H200 or a B200 (dc/s8_mma.sh)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

DC = Path(__file__).resolve().parents[1] / "dc"
SWEEP = "random_random_acc"


@pytest.fixture(scope="module")
def s8():
    spec = importlib.util.spec_from_file_location("dc_s8_mma_under_test", DC / "s8_mma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _line(pc: int, text: str) -> str:
    return f"        /*{pc:04x}*/                   {text} ;  /* 0x0000000000000000 */"


def test_sass_kernels_keeps_each_kernels_tensor_instructions(s8):
    sass = "\n".join([
        _line(0x00, "IMMA.16832.S8.S8.SAT R0, R4.ROW, R8.COL, R0"),      # before any kernel: no kernel's
        "\t\tFunction : s8_mma_form0",
        _line(0x00, "MOV R1, c[0x0][0x28]"),
        _line(0x10, "IMMA.16832.S8.S8.SAT R4, R8.ROW, R12.COL, R4"),
        _line(0x20, "@P0 BRA 0x40"),
        _line(0x30, "EXIT"),
        "\t\tFunction : s8_wg_form1",
        _line(0x00, "IGMMA.64x8x32.S8.U8 R0, R2, R0"),
        _line(0x10, "IGMMA.64x8x32.S8.U8 R0, R2, R0"),
        "\t\tFunction : s8_none",
        _line(0x00, "IMAD R0, R1, R2, R3"),
    ]) + "\n"
    assert s8.sass_kernels(sass) == {
        "s8_mma_form0": {"tensor": ["IMMA.16832.S8.S8.SAT R4, R8.ROW, R12.COL, R4"], "count": 1,
                         "opcodes": ["IMMA.16832.S8.S8.SAT"]},
        "s8_wg_form1": {"tensor": ["IGMMA.64x8x32.S8.U8 R0, R2, R0"] * 2, "count": 2, "opcodes": ["IGMMA.64x8x32.S8.U8"]},
        "s8_none": {"tensor": [], "count": 0, "opcodes": []},
    }


def _tiles(s8, tiles: int):
    """`tiles` mma.sync tiles of random bytes, the second half against accumulators within 1000 of an s32 end, and each
    form's words computed here in Python integers: the 32 products and C, clamped once at the end or wrapped."""
    m, n = s8.SHAPES["mma"]
    rng = np.random.default_rng(7)
    A = rng.integers(0, 256, (tiles, m, 32), dtype=np.uint8)
    B = rng.integers(0, 256, (tiles, n, 32), dtype=np.uint8)
    C = rng.integers(0, 1 << 32, (tiles, m, n), dtype=np.uint64).astype(np.uint32)
    C[tiles // 2:] = s8.words(np.where(rng.integers(0, 2, (tiles - tiles // 2, m, n)) == 0, s8.S32_MAX - 1000,
                                       s8.S32_MIN + 1000))

    def signed(fmt: str, x: int) -> int:
        return x - 256 if fmt == "s8" and x >= 128 else x

    device = {}
    for form in s8.FORMS:
        fa, fb, mode = form[:2], form[2:4], form.split("_")[1]
        D = np.empty((tiles, m, n), np.uint32)
        for t in range(tiles):
            for i in range(m):
                for j in range(n):
                    c = int(C[t, i, j])
                    total = (c - (1 << 32) if c >> 31 else c) + sum(
                        signed(fa, a) * signed(fb, b) for a, b in zip(A[t, i].tolist(), B[t, j].tolist(), strict=True))
                    if mode == "sat":
                        total = min(max(total, s8.S32_MIN), s8.S32_MAX)
                    D[t, i, j] = total & 0xFFFFFFFF
        device[form] = D
    return A, B, C, [{"family": SWEEP, "start": 0, "tiles": tiles}], device


def test_check_words_passes_the_sm120_steps_words_and_names_a_word_that_differs(s8):
    m, n = s8.SHAPES["mma"]
    A, B, C, meta, device = _tiles(s8, 2)
    forms, structured, ok = s8.check_words("mma", A, B, C, meta, device)
    assert ok and structured == {} and sorted(forms) == sorted(s8.FORMS)
    for form, rec in forms.items():
        assert (rec["words"], rec["model_step_checked"]) == (2 * m * n, 2 * m * n), form
        assert (rec["mismatches"], rec["model_step_mismatches"], rec["vectorized_vs_model_step"]) == (0, 0, 0), form
        assert rec["mismatches_by_family"] == {SWEEP: 0} and rec["unwritten_words"] == 0, form
        assert rec["out_of_range_words"] > 0, form                   # the second tile's sums leave the s32 range
    assert forms["s8s8_sat"]["clamp_hypotheses_consistent"] == [] and "clamp_hypotheses_consistent" not in forms["s8s8_wrap"]
    assert not np.array_equal(device["s8s8_sat"], device["s8s8_wrap"])

    device["u8s8_sat"][1, 3, 5] ^= 1
    forms, _, ok = s8.check_words("mma", A, B, C, meta, device)
    assert not ok
    for form, rec in forms.items():
        wrong = int(form == "u8s8_sat")
        assert (rec["mismatches"], rec["model_step_mismatches"], rec["vectorized_vs_model_step"]) == (wrong, wrong, 0), form
        assert rec["mismatches_by_family"] == {SWEEP: wrong}, form
    (first,) = forms["u8s8_sat"]["first_model_step_mismatches"]
    assert (first["family"], first["tile"], first["i"], first["j"]) == (SWEEP, 1, 3, 5)
    assert first["device"] == first["model"] ^ 1 and first["C"] == int(s8.as_s32(C[1, 3, 5]))
