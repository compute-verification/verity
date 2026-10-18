"""served_h1's credit by the padded-window rule (PADCREDIT round 2 with RT10's fixes, coordinator 22:35Z and 22:39Z): credit
only the selected slots, from selections that always-proved units compute from the raw committed rows.

    admit     slot i of a call is selected iff its committed A row is not the canonical empty row (K zero bytes, scale 0):
              the OR of the row's K + 4 bytes, a function of the raw row alone, no noise
    words     a call's T admit bits as T / 16 words of 16 bits (bit b of word q is slot 16 q + b), one admit unit each: a
              proof unit's outputs are at most 16 bits; each word is its own hiding leaf, in the call's admit tree
    dead      a unit (32 slot rows by 32 weight rows) is dead iff its 32 slots' admit bits are all 0: its rows are the
              canonical empty row, it skips the H1 noise and receipt, and its receipt leaf is the canonical empty receipt
              (2 (KS + m) zero words), so the receipt trees keep their shape
    tally     per (step, site): the admit words' popcount over the site's calls of the step (at most 32 T, 2^11 at T = 64);
              per site: those limbs' sum over the steps (at most S 32 T, under 2^16 at S = 18); the credit Cr is
              sum_site count_site N_site K_site, a public weighting of four 16-bit limbs; the level is
              window.level(window.floor_point(Cr)) and the published credit Pub = window.point(level) <= Cr

Every admit unit and every tally unit is proved in every window (`proof_plan`); the draw is uniform over every unit position
of the window, dead ones included, never weighted by an annotation.  Stdlib and numpy only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

R = 32
WORD = 16
LIMB = 1 << 16


def admit(rows: np.ndarray) -> np.ndarray:
    """(T,) bool: slot i is selected iff its committed row (K s8 bytes ‖ f32 scale, `commit.s8_rows`) has a nonzero byte."""
    return np.asarray(rows, dtype=np.uint8).reshape(rows.shape[0], -1).any(axis=1)


def words(bits: np.ndarray) -> list[int]:
    """T admit bits as T / 16 16-bit words, bit b of word q the slot 16 q + b."""
    b = np.asarray(bits, dtype=bool)
    if b.size % WORD:
        raise ValueError(f"T = {b.size} is not a multiple of {WORD}")
    return [int(sum(int(x) << i for i, x in enumerate(b[q:q + WORD]))) for q in range(0, b.size, WORD)]


def bits_of(ws: Sequence[int]) -> np.ndarray:
    return np.array([(w >> i) & 1 for w in ws for i in range(WORD)], dtype=bool)


def word_rows(ws: Sequence[int]) -> list[bytes]:
    """The admit tree's leaves: one 16-bit word each, little-endian."""
    return [int(w).to_bytes(2, "little") for w in ws]


def live_blocks(bits: np.ndarray) -> list[int]:
    """The call's row blocks with a selected slot."""
    b = np.asarray(bits, dtype=bool)
    return [ri for ri in range(b.size // R) if b[R * ri:R * ri + R].any()]


def live_units(bits: np.ndarray, n: int) -> list[int]:
    """The call's live units in position order (u = row block · N / 32 + column block): those with a selected slot."""
    cols = n // R
    return [ri * cols + cj for ri in live_blocks(bits) for cj in range(cols)]


def tally(calls: Sequence[dict], counts: Sequence[int], steps: int, sites: Sequence[str]) -> dict:
    """The window's tally from each call's selected-slot count, in 16-bit limbs: per (step, site), then per site."""
    by_step = {(s, site): 0 for s in range(steps) for site in sites}
    for c, k in zip(calls, counts):
        by_step[(c["step"], c["site"])] += int(k)
    by_site = {site: sum(by_step[(s, site)] for s in range(steps)) for site in sites}
    for what, v in (*by_step.items(), *by_site.items()):
        if not 0 <= v < LIMB:
            raise ValueError(f"tally limb {what} = {v} is not a 16-bit value")
    weight = {}
    for c in calls:
        weight.setdefault(c["site"], set()).add(c["n"] * c["k"])
    if any(len(v) != 1 for v in weight.values()):
        raise ValueError("a site's calls differ in N K: the per-site limb's weight isn't one public number")
    w = {site: next(iter(weight[site])) for site in sites}
    return {"step_site": [[by_step[(s, site)] for site in sites] for s in range(steps)], "site": by_site,
            "weight": w, "rows": sum(by_site.values()), "macs": sum(by_site[site] * w[site] for site in sites)}


def published(macs: int) -> dict:
    """Cr's grid level and the published credit Pub = point(level) <= Cr (`pouw.window`'s grid)."""
    from verity.core.protocols.pouw import window as Wd
    if macs < 1:
        return {"level": None, "pub_macs": 0}
    ell = Wd.level(Wd.floor_point(macs))
    return {"level": ell, "pub_macs": Wd.point(ell), "grid_bits": Wd.GRID_BITS}


def proof_plan(n_calls: int, t: int, steps: int, sites: Sequence[str], population: int) -> dict:
    """What every window proves, beside the uniform draw over its units."""
    admit_units = n_calls * (t // WORD)
    tally_units = steps * len(sites) + len(sites) + 1
    return {"population": population,
            "draw": "uniform over every unit position of the window, live or dead, by the window's draw coin: public "
                    "structure, never weighted by an annotation",
            "always_proved": {"admit": admit_units, "tally": tally_units, "total": admit_units + tally_units},
            "admit_unit": "reads 16 slots' committed A rows (opened from the call's a-table), outputs their 16 admit bits "
                          "(the OR of each row's bytes), the leaf of the call's admit tree it writes",
            "tally_units": f"{steps * len(sites)} per (step, site) popcount sums (<= 2^11), {len(sites)} per-site sums over "
                           "the steps (< 2^16), and one level unit (window.level(floor_point(sum_site count N K)), <= 9 bits)",
            "dead_unit": "a drawn dead unit proves, with the live tile's proof shape (one gated template), that its 32 "
                         "admit bits are 0 and its receipt is the canonical empty receipt; no noise is computed when serving",
            "egress": "no dead slot's output reaches a served output: the receipts are beside the served y and feed no "
                      "egress, and vLLM samples each served token from the step's selected rows only (admit is the served "
                      "prefix [1] * m + [0] * (T - m) in every call)"}


__all__ = ["admit", "bits_of", "live_blocks", "live_units", "proof_plan", "published", "tally", "word_rows", "words"]
