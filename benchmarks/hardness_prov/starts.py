"""The committed starts W₀ a cheater can choose, each a cheap structure planted into the pretrained toy's MLP linears (fc
and out in every block, as HT-PUBLIC's plant test). Each takes the honest float matrix w and the pass's quantizer, and
returns the planted float matrix and the ternary t to price it against (or None).

- `honest`: w itself.
- `tern`: HT-PUBLIC's per-row scaled ternary (`plant.ternary`): exact structure at the code level, priced at one ternary
  pass (0.5 on both chips at b = 2).
- `ternpin8`: the same with each row's absmax pinned so t's entries quantize to exactly ±8 (`plant.ternary(pin=8)`).
- `s24`: 2:4 sparsity, the two largest |w| of every four along K kept (0.5).
- `small`: codes of at most 5 bits, so exact Strassen in s8 runs at depth 3 (0.766). Under `absmax` each row's codes
  are w rounded onto [-15, 15] plus one pin at ±127 that sets the scale (in the column of least norm); under `mad`,
  whose scale no pin can set, a grid: codes exactly 9·m with m on [-14, 14], m's MAD held at 3 so that 40/(1.4826·3)
  rounds every 9·m exactly.
- `lr{r}`: the rank-r SVD of w alone (a junk model, cheap at r/n + 3r/K).
- `lr{r}res{k}`: rank r plus w's residual clipped to ±k served LSBs (k = 15 keeps it in Strassen's depth 3): the
  useful-model version, as SVD-plus-low-bits quantizers build it.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hardness_public"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy
from plant import ternary

PLANTED = ("fc", "out")


def svd_r(w, r):
    U, s, Vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def small_absmax(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 15.0
    w0 = s * np.clip(np.rint(w / s), -15, 15)
    j = np.argmin(np.linalg.norm(w, axis=0))
    w0[:, j] = np.where(w[:, j] >= 0, 1.0, -1.0) * 127 * s[:, 0]
    return w0


def small_mad(w):
    """Rows Δ·m with m integer on [-14, 14], median 0 and MAD exactly 3, so mad's codes are 9·m (9·14 = 126 ≤ 127).
    A row with no such Δ on the search grid keeps the Δ whose MAD is nearest 3, and its codes are then off the grid."""
    out = np.empty_like(w, dtype=np.float64)
    for i, row in enumerate(w.astype(np.float64)):
        sd = row.std()
        best = None
        for x in np.linspace(0.05, 2.0, 2000):
            m = np.clip(np.rint(row / (x * sd)), -14, 14)
            gap = abs(np.median(np.abs(m - np.median(m))) - 3) + abs(np.median(m))
            if best is None or gap < best[0]:
                best = (gap, x, m)
            if gap == 0:
                break
        out[i] = best[1] * sd * best[2]
    return out


def residual_clip(w, r, k, quant):
    L = svd_r(w, r)
    w0 = w.astype(np.float64)
    for _ in range(4):
        s = toy.scale(w0, quant)[:, None]
        w0 = L + np.clip(w - L, -k * s, k * s)
    return w0


def plant(w, family, quant):
    w = np.asarray(w, dtype=np.float64)
    if family == "honest":
        return w, None
    if family in ("tern", "ternpin8"):
        t, wp = ternary(w, 8 if family == "ternpin8" else 0)
        return wp.astype(np.float64), t
    if family == "s24":
        n, k = w.shape
        g = np.abs(w).reshape(n, k // 4, 4)
        keep = np.argsort(-g, axis=2)[:, :, :2]
        mask = np.zeros_like(g, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=2)
        return np.where(mask.reshape(n, k), w, 0.0), None
    if family == "small":
        return (small_absmax(w) if quant == "absmax" else small_mad(w)), None
    m = re.fullmatch(r"lr(\d+)(?:res(\d+))?", family)
    if m:
        r = int(m.group(1))
        return (svd_r(w, r) if m.group(2) is None else residual_clip(w, r, int(m.group(2)), quant)), None
    raise ValueError(family)
