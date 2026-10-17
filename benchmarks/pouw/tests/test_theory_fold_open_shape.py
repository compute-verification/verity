"""Two n = 8 keys that FOLD's reduced key_ok_r11 admits break K_R = K_1 = P/X/N; KJ19 scaled as at 32 bits rejects both."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import fold_kernel_check as fk
import fold_open_shape_check as fo


def test_open_shape_keys():
    """(97, 179)'s one-round kernel has 64 elements; at (83, 225) a change outside K_1 fixes the 2-fold only."""
    key, word = fo.BIG
    assert len(fk.mitm_k1(word, *key, fo.N, fo.R1, fo.R2)) + 1 == 64
    key, word = fo.EVEN
    assert fo.fixes(word, fo.EVEN_EXTRA, key, 2) and not fo.fixes(word, fo.EVEN_EXTRA, key, 1)
    assert all(fk.key_ok_r11(*k, fo.N) and (fo.kj19_scaled(k[0]) or fo.kj19_scaled(k[1]))
               for k in (fo.BIG[0], fo.EVEN[0]))
