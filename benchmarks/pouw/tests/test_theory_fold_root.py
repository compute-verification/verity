"""The kernel by commuting (theory/fold_root_kernel.c) agrees with the referee's brute force, reproduces the referee's
counterexamples, and KJ19's scaled last clause is the one that rejects every bad n = 8 key pair."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))
sys.path.insert(0, str(THEORY.parent / "receipts"))

import fold_root_check as fr
import fold_root_sweep as sw
import receipt_folds as rf


def test_root_kernel_matches_brute_force(tmp_path):
    """At width 6 the commuting meet in the middle finds every K_R (R <= 3) that fold_kernel.c's 2^24 brute force finds."""
    exe, brute = fr.build(tmp_path), fr.build(tmp_path, "fold_kernel.c")
    word, keys = (0x1E, 0x37, 0x39, 0x2B), [(37, 21), (5, 43), (19, 27), (61, 13)]
    summ, root = fr.root_kernels(exe, 6, 2, 4, 3, [k + word for k in keys])
    ref = fr.brute_kernels(brute, 6, 2, 4, 3, word, keys)
    assert not any(row[5] for row in summ.values())
    assert not fr.agree(root, ref, word, keys, 3, set())
    assert len(ref[(37, 21, 1)]) == 15 and summ[(37, 21) + word][:3] == [16, 16, 16]


def test_referee_counterexamples(tmp_path):
    """(97, 179) and (83, 225) at n = 8 and (321, 411) at n = 10 break K_R = K_1 = P/X/N as the referee found, and each
    has a key ≡ ±1 mod 2^r1."""
    exe = fr.build(tmp_path)
    s8, _ = fr.root_kernels(exe, 8, 3, 5, 6, [(97, 179, 255, 48, 87, 75), (83, 225, 144, 166, 44, 160)])
    assert s8[(97, 179, 255, 48, 87, 75)] == [64] * 6 + [0, 48, 0]
    assert s8[(83, 225, 144, 166, 44, 160)] == [16, 32, 16, 64, 16, 32, 1, 0, 0]
    s10, _ = fr.root_kernels(exe, 10, 4, 6, 2, [(321, 411, 827, 208, 250, 366)])
    assert s10[(321, 411, 827, 208, 250, 366)][0] == 32
    assert not any(sw.kj_last(a, r) and sw.kj_last(b, r) for a, b, r in [(97, 179, 3), (83, 225, 3), (321, 411, 4)])


def test_scaled_kj_is_kj19_at_32_bits():
    """The width-n rule read at n = 32, r2 = 19 is receipt_folds.kj19, and it implies a ≢ ±1 mod 2^13."""
    rng = random.Random(7)
    keys = [rng.randrange(1 << 32) | 1 for _ in range(4000)]
    keys += [(s + (1 << e)) & 0xFFFFFFFF | 1 for e in range(12, 31) for s in (1, -1, 3, -3)]
    assert all(sw.kj_scaled(a, 32, 19) == rf.kj19(a) for a in keys)
    assert all(sw.kj_last(a, 13) for a in keys if rf.kj19(a))


def test_not_symmetric_words_open_k2(tmp_path):
    """At words with 2 d1 ≡ a0 − 1 and 2 d3 ≡ a1 − 1, complementing d0 gives F_{w'} = NOT ∘ F_w: every even fold is fixed,
    at n = 10 under the scaled KJ (|K_even| = 32, odd R adds none) and at 32 bits at a key_ok_r11 pair; one off, it isn't."""
    jobs = sw.jobs_for("symmetric", 10, random.Random(3), 2, 0)
    summ, elems = fr.root_kernels(fr.build(tmp_path), 10, 4, 6, 4, jobs + [j[:3] + (j[3] ^ 2,) + j[4:] for j in jobs[:2]])
    assert all(summ[j] == [16, 32, 16, 32, 1, 0, 0] and (j[2] ^ 1023,) + j[3:] in elems[(j, 2)] for j in jobs)
    assert all(summ[j[:3] + (j[3] ^ 2,) + j[4:]] == [16, 16, 16, 16, 1, 0, 0] for j in jobs[:2])
    a0, a1 = 0x9E3779B1, 0x7F4A7C15
    assert rf.key_ok_r11(a0) and rf.key_ok_r11(a1)
    w = (0x12345678, (a0 - 1) // 2, 0x0BADF00D, (a1 - 1) // 2 + (1 << 31))
    w2 = (w[0] ^ 0xFFFFFFFF,) + w[1:]
    for y in (0, 1, 0xDEADBEEF, 0x80000000):
        p, q = fr.fold32(y, w, a0, a1), fr.fold32(y, w2, a0, a1)
        assert q == p ^ 0xFFFFFFFF and fr.fold32(q, w2, a0, a1) == fr.fold32(p, w, a0, a1)


def test_cut_join_settles(tmp_path):
    """A structured word at a scaled-KJ pair cuts the join at the default 3 probes and cap 2^16; 6 probes and cap 2^28
    settle it as K_R = K_1 = P/X/N, and the sweep's recut set picks out exactly the cut jobs."""
    exe, job = fr.build(tmp_path), (77, 219, 0, 0, 0, 0)
    cut, _ = fr.root_kernels(exe, 10, 4, 6, 6, [job])
    settled, _ = fr.root_kernels(exe, 10, 4, 6, 6, [job], ("-p", "6", "-c", "28"))
    assert cut[job][-1] == 1 and settled[job] == [16] * 6 + [1, 0, 0]
    prior = tmp_path / "prior.txt"
    prior.write_text(f"S {' '.join(map(str, job))} {' '.join(map(str, cut[job]))}\nS 3 5 1 2 3 4 16 16 1 0 0\n")
    assert sw.jobs_for("recut", 10, random.Random(0), 0, 0, src=prior) == [job]


@pytest.mark.slow
def test_n8_bad_pairs_fail_kj_last(tmp_path):
    """Over 1,024 random n = 8 key pairs, a random word each, every pair where K_R = K_1 = P/X/N fails for some R <= 6 has
    a key ≡ ±1 mod 8; the pairs that clause admits all hold."""
    rng = random.Random(20261017)
    odd = range(1, 256, 2)
    jobs = [(rng.choice(odd), rng.choice(odd), *(rng.randrange(256) for _ in range(4))) for _ in range(1024)]
    summ, _ = fr.root_kernels(fr.build(tmp_path), 8, 3, 5, 6, jobs)
    bad = [j for j, row in summ.items() if not fr.good(row, 6)]
    assert bad and len(summ) == len(set(jobs))
    assert not any(sw.kj_last(a, 3) and sw.kj_last(b, 3) for a, b, *_ in bad)
