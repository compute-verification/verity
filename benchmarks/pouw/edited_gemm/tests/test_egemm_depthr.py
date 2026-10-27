"""The fold's same-word depth-R binding proof (egemm_depthr_check.py): the
served-GEMM scalar dial absorbs the same committed word R times through the XOR-in(13,19) fold with the same keys, and the
R-fold still binds the word with no new luck term. The fast tests reuse the proof's functions on a key sample; the slow
test runs the whole exact_kit check (every odd key exhaustively, n=10/12 key_ok_r11, 32-bit (13,19), and both mutations)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("egemm_depthr_check", HERE.parent / "egemm_depthr_check.py")
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

W8 = (0x9E, 0x37, 0x79, 0xB9)  # a committed word's four s32 limbs, at the reduced width n = 8


def test_the_fold_is_faithful():
    """One round of the proof's fold equals receipt_folds_r9.xorin_step (rotations 13, 19), the pinned r9 reference
    the dial iterates."""
    a0, a1 = np.uint32(0x9E3779B1), np.uint32(0x85EBCA77)
    y = np.arange(64, dtype=np.uint32) * np.uint32(2654435761)
    d = [np.uint32(0x1234 + i) + np.arange(64, dtype=np.uint32) for i in range(4)]
    with np.errstate(over="ignore"):
        ref = M.xorin_step_ref(y, d[0], d[1], d[2], d[3], a0, a1).astype(np.uint64)
        mine = (M.wrap.rotl(y.astype(np.uint64), 13, 32) ^ d[0].astype(np.uint64)) * np.uint64(a0) + d[1]
        mine = ((M.wrap.rotl(mine & np.uint64(0xFFFFFFFF), 19, 32) ^ d[2].astype(np.uint64)) * np.uint64(a1)
                + d[3]) & np.uint64(0xFFFFFFFF)
    assert bool((mine == ref).all())


def test_odd_depth_binds_and_key_ok_r11_binds_on_a_sample():
    """On a sample of n = 8 odd keys, R = 1..4: no all-state collapse at odd depth (the one open shape needs per-round
    alternating keys, unreachable by folding the same word), none among key_ok_r11 keys, and every collapse that does
    occur sits on a KJ19-defective key (a near +-1), which is the even-depth run-of-two g3-chains already excludes."""
    keys = np.array([(a0, a1) for a0 in range(1, 256, 2) for a1 in range(1, 256, 32)], dtype=np.uint64)
    hits = M.collapses(keys, [W8], 4, [1, 2, 3, 4], 8, 3, 5)
    assert [h for h in hits if h[4] % 2 == 1] == []
    assert [h for h in hits if M.key_ok_r11(h[0], h[1], 8)] == []
    assert all(M.kj19_bad(h[0], 8) or M.kj19_bad(h[1], 8) for h in hits)


def test_kj19_control_collapses_at_even_depth_and_is_rejected():
    """The positive control: a near-(-1) key (a0 = 63 = -1 mod 2^6 at n = 8) collapses at even depth, so the search sees
    runs of two; key_ok_r11's KJ19 clause rejects it."""
    hits = M.collapses(np.array([(63, 0x55)], dtype=np.uint64), [W8], 4, [2, 4], 8, 3, 5)
    assert len(hits) > 0
    assert not M.key_ok_r11(63, 0x55, 8)


def test_degenerate_key_collapse_is_detectable():
    """A degenerate key a0 = 0 drops limb w0, so a w0-only change is invisible: the search must (and does) see it, which
    is what makes the zero-collapse results above meaningful."""
    hits = M.collapses(np.array([(3, 5)], dtype=np.uint64), [W8], 3, [3], 8, 3, 5, force_a0=0)
    assert len(hits) > 0


@pytest.mark.slow
def test_full_depthr_check_passes():
    """The whole exact_kit check: n = 8 exhaustive over every odd key pair, n = 10/12 over key_ok_r11 keys, the 32-bit
    (13, 19) sample, and both mutations -- 10/10 checks, 2/2 mutations caught. Catches what the sampled fast tests cannot:
    an all-state collapse on any odd key or any key_ok_r11 key across the full width."""
    p = subprocess.run([sys.executable, str(HERE.parent / "egemm_depthr_check.py"), "--mutations"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "checks OK" in p.stdout and "mutations caught" in p.stdout
