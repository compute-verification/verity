"""(0) The H200 reference kernel, counted: `dc90-li-l32` (wgmma m64n256k32 s8, then LI on each committed word) and
`dc90-l32` (the same with duo), with LI-CP-W's warp checkpoint after each segment.

Words: one IGMMA.64x256x32.S8.S8 a segment and warpgroup tile, its operands read from shared memory by descriptor
((64 + 256) 32 bytes a wgmma, `isa.Kernel.smem_read`); the accumulator stays in registers, so a word needs no load.
Setup (phase ``setup``, reported apart): forming At = A + R and Bt = B + S once, four int8 entries a lane instruction.

`receipt` is shared with every alternative of (a) to (d): they produce the words another way and fold them the same.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import dither as D
from . import folds as F
from .isa import Kernel

WGMMA_M, WGMMA_N = 64, 256


@dataclass
class Run:
    words: np.ndarray
    receipt: np.ndarray
    kernel: Kernel


def setup_operands(k: Kernel, inst: D.Instance) -> None:
    with k.phase("setup"):
        k.add("IADD3", 2 * inst.n * inst.k // 4)


def tensor_words(k: Kernel, inst: D.Instance) -> np.ndarray:
    """The words by the s8 tensor path: one k32 MMA a segment, accumulated in place."""
    At, Bt = inst.At, inst.Bt
    acc = np.zeros((inst.n, inst.n), dtype=np.uint32)
    out = []
    for l in range(inst.ks):
        acc = k.mma_s8(At[:, l * D.L:(l + 1) * D.L], Bt[l * D.L:(l + 1) * D.L], acc)
        k.smem_read(inst.n * inst.n * D.L * (WGMMA_M + WGMMA_N) // (WGMMA_M * WGMMA_N))
        out.append(acc)
    return np.stack(out)


def receipt(k: Kernel, W: np.ndarray, fold: str = "li", seed: int = 1, cp: str = "redux",
            form: Callable[[int], str] | None = None, absorb: Callable | None = None) -> np.ndarray:
    """Fold every word of W into its lane's state and take the warp checkpoint after each segment, counted."""
    lanes = W[0].size // D.WPL
    st = np.array(D.start(fold, lanes, seed), dtype=np.uint32)
    if absorb is None:
        absorb = F.li_absorb if fold == "li" else (lambda kk, s, p: F.duo_absorb(kk, s, p, form or (lambda q: "leahi")))
    out = []
    for Wl in W:
        sets = D.lane_sets(Wl)
        for s in range(D.WPL // D.SET):
            st = absorb(k, st, sets[:, s * D.SET:(s + 1) * D.SET])
        out.append(F.checkpoint(k, st, cp))
    return np.stack(out)


def run(inst: D.Instance, fold: str = "li", seed: int = 1, cp: str = "redux") -> Run:
    k = Kernel()
    setup_operands(k, inst)
    W = tensor_words(k, inst)
    return Run(W, receipt(k, W, fold, seed, cp), k)
