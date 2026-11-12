"""The coins a two-stage design's second stage reads, and the chance it refuses an honest run.

The second stage (`service.select_second`, Lean's `Spot.Select.selectSecond`) samples `ks[t]` of the `ns[t]` proof
blocks of each cell: template `t` of the check, in each sampled replay block.  A cell's sampler first reads
`2 k_t · width(n_t) + 64` bytes of the stage's coins (`coins.draw_first`, Lean's `Flock.Draw.Stratum.need`), and as
many again only where it rejects a word, which it does a uniform word with probability under 2^-64.  Lean bounds what
that leaves (`SpotFlock.secondDraw_refused_le_first`): at every first-stage coin on which a first stage of count K
accepts, a second stage whose coins hold every cell's first read refuses with probability at most K·Σ_t k_t·2^-64 over
them.  On fewer bytes than the first reads it refuses at every coin.

`first_read` is that theorem's condition and `refusal` its bound.  `stages` and `coins` count a first read in the units
its two readers open: SPOT[Flock]'s rounds of 2^20 bytes (`SpotFlock.L`, `Held.stage` of them) and the challenger's
64-byte coins (`challenger.COIN_BYTES`), of which one commitment holds at most `challenger.MAX_COINS`: `fits` is how
many sampled replay blocks' first reads that is.  `design_cells` gives the cells of a planner's two-stage design
(`optimize.two_stage_design`), and `at_design` all of it there, which `total_cost` carries beside every two-stage point.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import ceil
from typing import Any

import figure1 as f1
from verity.protocols.spot.device.challenger import COIN_BYTES, MAX_COINS
from verity.protocols.spot.optimize import Cell

#: the bytes SPOT[Flock]'s stages read of one round's coin, 65,536 field elements of 16 bytes (Lean's `SpotFlock.L`)
ROUND_BYTES = 65536 * 16


def width(n: int) -> int:
    """The bytes of each word a sampler of `n` reads: ⌈(bitlen(n) + 64)/8⌉ (Lean's `Flock.Draw.width`)."""
    return (n.bit_length() + 64 + 7) // 8


def _cells(sample: int, ns: Sequence[int], ks: Sequence[int]) -> None:
    if not (sample >= 1 and ns and len(ks) == len(ns) and all(1 <= k <= n for n, k in zip(ns, ks))):
        raise ValueError(f"a second stage samples 1 to n_t of each cell's n_t proof blocks, in at least one sampled "
                         f"replay block: counts {list(ks)} over cells of {list(ns)} in {sample}")


def first_read(sample: int, ns: Sequence[int], ks: Sequence[int]) -> int:
    """The bytes the second stage's cells read first over `sample` sampled replay blocks,
    sample·Σ_t (2 k_t·width(n_t) + 64): Lean's `secondRead`, which the stage's coins must hold."""
    _cells(sample, ns, ks)
    return sample * sum(2 * k * width(n) + 64 for n, k in zip(ns, ks))


def refusal(sample: int, ks: Sequence[int]) -> Fraction:
    """Lean's bound on the chance that the second stage refuses over `sample` sampled replay blocks, on coins that
    hold its first read: sample·Σ_t k_t·2^-64."""
    return Fraction(sample * sum(ks), 1 << 64)


def stages(read: int) -> int:
    """The rounds whose coins hold `read` bytes (`Held.stage`)."""
    return -(-read // ROUND_BYTES)


def coins(read: int) -> int:
    """The challenger's coins that hold `read` bytes."""
    return -(-read // COIN_BYTES)


def fits(read: int) -> int:
    """The sampled replay blocks whose first reads, `read` bytes each, one commitment of the challenger's holds: on a
    replay sample of more the second stage has too few coins and refuses every run."""
    return MAX_COINS * COIN_BYTES // read


def _split(k: int, sizes: Sequence[int]) -> list[int]:
    """`k` among cells of `sizes` in proportion to them, by largest remainders, each cell's share at least 1 and at
    most its size: the shares add up to `k` unless there are more cells than `k`, where each is 1."""
    total = sum(sizes)
    out = [max(1, min(s, k * s // total)) for s in sizes]
    order = sorted(range(len(sizes)), key=lambda i: (-(k * sizes[i] % total), i))
    left = k - sum(out)
    while left > 0 and any(out[i] < sizes[i] for i in order):
        for i in order:
            if left > 0 and out[i] < sizes[i]:
                out[i] += 1
                left -= 1
    while left < 0 and max(out) > 1:
        out[max(range(len(out)), key=lambda i: (out[i], -i))] -= 1
        left += 1
    return out


def design_cells(cells: Sequence[Cell], row: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The second stage's cells of a sampled replay block under the planner's two-stage design `row`, as (ns, ks): one
    cell a proof-block type of `cells`, of its ⌈n⌉ proof blocks.  A by-template design's strata are the cells, each
    with its own count (`row["k"]`, over `figure1.split_cells`'s strata).  In a design with a stratum of several types
    the planner samples them at one rate (`optimize.two_stage_statements`), so the stratum's k is split among its
    types' cells in proportion to their proof blocks (`_split`): each cell's count is at least 1, which the second
    stage asks of every cell, and at most its proof blocks."""
    split = f1.split_cells(cells, row["merged"])
    ns = [ceil(c.n) for c in split]
    ks = [0] * len(split)
    for t, k in row["k"].items():
        at = [i for i, c in enumerate(split) if c.stratum == t]
        for i, share in zip(at, _split(k, [ns[i] for i in at])):
            ks[i] = share
    return tuple(ns), tuple(ks)


def at_design(cells: Sequence[Cell], row: Mapping[str, Any]) -> dict[str, Any]:
    """The second stage at the planner's two-stage design `row` on `cells` (`design_cells`):

    - `cells`, `sampled` and `read`: a sampled replay block's cells, the proof blocks the second stage samples of them
      (Σ_t k_t) and the bytes they read first;
    - `sample`: the replay blocks K the first stage samples of a verification phase, a whole count
      (`registration_sample`); with no `registration` one verification phase holds every replay block, and K is the
      scope's (`replay_sample`);
    - `stages` and `coins`: the rounds and the challenger's coins that hold K replay blocks' first read;
    - `fits`: the sampled replay blocks whose first reads one commitment of the challenger's holds
      (`challenger.MAX_COINS` coins), to hold against K: on a replay sample of more, the second stage refuses;
    - `refusal`: Lean's bound on the chance a verification phase's second stage refuses an honest run, at a first-stage
      coin the first stage accepts and on coins that hold the first read, K·Σ_t k_t·2^-64;
    - `pooled`: that bound added up over the replay blocks the row's whole scope samples (`replay_sample`, expected).

    `whole` marks a design that proves every cell whole: it samples whole replay blocks, its second stage reads no
    coin, and every figure after `sampled` is 0."""
    ns, ks = design_cells(cells, row)
    sample = int(row["registration_sample"]) if row.get("registration") else max(1, round(row["replay_sample"]))
    out = {"cells": len(ns), "sampled": sum(ks), "whole": ks == ns, "sample": sample}
    if ks == ns:
        return out | {"read": 0, "stages": 0, "coins": 0, "fits": None, "refusal": 0.0, "pooled": 0.0}
    read = first_read(sample, ns, ks)
    return out | {"read": read // sample, "stages": stages(read), "coins": coins(read), "fits": fits(read // sample),
                  "refusal": float(refusal(sample, ks)),
                  "pooled": row["replay_sample"] * sum(ks) / 2.0 ** 64}


__all__ = ["ROUND_BYTES", "at_design", "coins", "design_cells", "first_read", "fits", "refusal", "stages", "width"]
