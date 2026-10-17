"""The referee's FOLD enumerators (theory/fold_kernel.c, theory/chain_kernel.c) agree with exact_kit on small cases."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))

import fold_kernel_check as fk  # noqa: E402


def test_fold_kernel_matches_meet_in_the_middle(tmp_path):
    """At width 6 the same-word fold's K_1 is the 16-element P/X/N group, the enumerator's K_1 equals exact_kit's
    meet-in-the-middle answer, and odd R adds nothing at this key."""
    exe = fk.build(tmp_path)
    word = (0x1E, 0x37, 0x39, 0x2B)
    res = fk.enumerate_kernels(exe, 6, 2, 4, 3, word, [(37, 21)])[(37, 21)]
    assert len(res[1]) == 15
    assert res[3] == res[1]
    assert fk.mitm_k1(word, 37, 21, 6, 2, 4) == res[1]
    assert all(fk.g3_ok(word, w2, 6) for w2 in res[1])


def test_chain_kernel_matches_brute_force(tmp_path):
    """At width 3 the meet in the middle finds exactly the two-word chain kernels that brute force over all 2^24 word
    pairs finds, split into per-word and cross-word elements the same way, for every (c0, c1) in {1, 2}^2."""
    n, r1, r2, w0, w1, keys = 3, 1, 2, (6, 3, 1, 5), (2, 7, 4, 1), [(3, 5), (1, 7)]
    exe = tmp_path / "chain_kernel"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(THEORY / "chain_kernel.c")], check=True)
    out = subprocess.run([str(exe), str(n), str(r1), str(r2), *map(str, w0 + w1)],
                         input="".join(f"{a} {b}\n" for a, b in keys), capture_output=True, text=True, check=True).stdout
    got = {tuple(int(x) for x in p[1:5]): (int(p[5]), int(p[6])) for p in map(str.split, out.splitlines()) if p[0] == "S"}
    ys = np.arange(1 << n, dtype=np.uint64)
    words = np.arange(1 << (4 * n), dtype=np.uint64)
    limbs = [(words >> np.uint64(i * n)) & np.uint64((1 << n) - 1) for i in range(4)]
    idx = lambda w: sum(int(x) << (i * n) for i, x in enumerate(w))
    for a0, a1 in keys:
        tab = {c: fk.fold(ys[None, :], [x[:, None] for x in limbs], a0, a1, n, r1, r2, c).astype(np.int64) for c in (1, 2)}
        for c0 in (1, 2):
            for c1 in (1, 2):
                left, right = tab[c0], tab[c1]
                chain = right[:, None, :][np.arange(len(words))[:, None, None], 0, left[None, :, :]]
                same = (chain == chain[idx(w1), idx(w0)][None, None, :]).all(axis=2)
                total = int(same.sum()) - 1
                own = (left == left[idx(w0)][None, :]).all(axis=1)
                cross = int((same & ~own[None, :]).sum())
                assert got[(a0, a1, c0, c1)] == (total, cross), (a0, a1, c0, c1)
