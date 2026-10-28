"""served_h1's credit by the padded-window rule: credit
only the admitted slots, from admissions that always-proved units compute from the committed request queue and the rows'
scales.

    queue     the scheduler's request queue, committed before the window's first coin (`queue_of`, `queue_leaves`): per
              (step, slot) the index of the request in that slot (0 for none), per request the steps [start, stop) it is
              live in (`pouw.circuit.admit`)
    admit     slot i of a call at step t is selected iff the queue assigns it a request live at t (`queue_select`), never
              a function of the slot's row, and admitted iff it is selected and its row's scale is nonzero (`queue_admit`,
              the word of `PouwQueueAdmit`): a row can withhold its slot's admission, never grant it; the admit unit
              refuses a slot that is not admitted unless its row is the canonical empty row (K zero bytes, scale 0:
              `junk`) and a slot that is selected whose row is the canonical empty row (`hollow`); it reads no
              row byte, so a selected row of zero bytes under a nonzero scale (`scale_only`) is admitted
    words     a call's T admit bits as T / 16 words of 16 bits (bit b of word q is slot 16 q + b), one admit unit each: a
              proof block's outputs are at most 16 bits; each word is its own hiding leaf, in the call's admit tree
    tile      a unit (32 slot rows by 32 weight rows) is `PouwH1Tile_v4`, whose `admit` input is its 32 rows' admit bits,
              bound by commitment equality to the two admit-tree leaves the tally reads: its live bit is their OR, so
              the tiles the circuit treats as live are exactly the tiles the tally credits, and an admitted tile of
              zero-byte rows is live, noised, receipted and credited
    dead      a unit is dead iff its 32 slots' admit bits are all 0: its rows are the canonical empty row (the admit unit
              refuses any other), it skips the H1 noise and receipt, and its receipt leaf is the canonical empty receipt
              (2 (KS + m) zero words), so the receipt trees keep their shape
    tally     per (step, site): the admit words' popcount over the site's calls of the step (at most 32 T, 2^11 at T = 64);
              per site: those limbs' sum over the steps (at most S 32 T, under 2^16 at S = 18); the credit Cr is
              sum_site count_site N_site K_site, a public weighting of four 16-bit limbs; the level is
              window.level(window.floor_point(Cr)) and the published credit Pub = window.point(level) <= Cr

Every admit unit is proved in every window (`proof_plan`); the tally is no proof block until a tally Definition exists.  The
draw is uniform over every unit position of the window, dead ones included, never weighted by an annotation.  Numpy and
`pouw.circuit.admit`'s word.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

R = 32
WORD = 16
LIMB = 1 << 16


def nonempty(rows: np.ndarray) -> np.ndarray:
    """(T,) bool: slot i's committed row (K s8 bytes ‖ f32 scale, `commit.s8_rows`) is not the canonical empty row."""
    return np.asarray(rows, dtype=np.uint8).reshape(rows.shape[0], -1).any(axis=1)


def queue_of(spans: Sequence[Sequence[tuple[str, int]]], steps: int, t: int) -> dict:
    """The request queue from the scheduler's record: ``spans[s]``, step s's (request id, rows) in batch order, its rows the
    step's slots in that order.  Requests take indices 1, 2, .. by first appearance; each is live from its first step to
    one past its last.  Steps past the record are empty."""
    from verity.core.protocols.pouw.circuit import admit as A
    if len(spans) > steps:
        raise ValueError(f"the record has {len(spans)} steps, past the window's {steps}")
    index: dict[str, int] = {}
    life: list[list[int]] = []
    slot_map = np.zeros((steps, t), dtype=np.uint16)
    for s, step in enumerate(spans):
        at = 0
        for rid, n in step:
            if rid not in index:
                index[rid] = len(index) + 1
                life.append([s, s + 1])
            q = index[rid]
            life[q - 1][1] = s + 1
            if at + n > t or q > A.M16:
                raise ValueError(f"step {s}: request {rid}'s {n} rows past the call's {t} slots, or past 2^16 requests")
            slot_map[s, at:at + n] = q
            at += n
    return {"slot_map": slot_map, "requests": [(0, 0)] + [tuple(x) for x in life], "ids": list(index)}


def queue_select(queue: dict, s: int) -> np.ndarray:
    """(T,) bool: step s's selection, the slots the queue assigns a request live at s (`admit.select_bit`)."""
    from verity.core.protocols.pouw.circuit import admit as A
    req = queue["requests"]
    return np.array([bool(A.select_bit(s, int(x), *req[int(x)])) for x in queue["slot_map"][s]], dtype=bool)


def scales(rows: np.ndarray) -> list[int]:
    """Each committed row's scale, the f32 that ends it (`commit.s8_rows`), as the u32 `PouwQueueAdmit` reads."""
    r = np.ascontiguousarray(np.asarray(rows, dtype=np.uint8).reshape(rows.shape[0], -1)[:, -4:])
    return [int(v) for v in r.view("<u4").reshape(-1)]


def queue_admit(queue: dict, s: int, rows: np.ndarray) -> np.ndarray:
    """(T,) bool: step s's admit bits, `PouwQueueAdmit`'s word on each 16 slots' queue entries and committed rows' scales."""
    from verity.core.protocols.pouw.circuit import admit as A
    q = [int(x) for x in queue["slot_map"][s]]
    req, sc = queue["requests"], scales(rows)
    if len(sc) != len(q):
        raise ValueError(f"step {s} has {len(q)} slots and {len(sc)} rows")
    ws = [A.admit_word(s, q[g:g + WORD], [req[x][0] for x in q[g:g + WORD]], [req[x][1] for x in q[g:g + WORD]],
                       sc[g:g + WORD]) for g in range(0, len(q), WORD)]
    return bits_of(ws)


def queue_leaves(queue: dict) -> tuple[list[bytes], list[bytes]]:
    """The queue's two tables' leaves: the slot map, one per (step, 16 slots), 16 u16 LE in step-major order; the requests,
    (start, stop) as two u16 LE per index, the null request first, padded with null entries to S·T + 1 (every slot a request
    of its own), so the table's size shows no request count."""
    sm = queue["slot_map"]
    slots = [np.ascontiguousarray(sm[s, g:g + WORD], dtype="<u2").tobytes() for s in range(sm.shape[0])
             for g in range(0, sm.shape[1], WORD)]
    reqs = list(queue["requests"]) + [(0, 0)] * (sm.size + 1 - len(queue["requests"]))
    return slots, [int(a).to_bytes(2, "little") + int(b).to_bytes(2, "little") for a, b in reqs]


def junk(rows: np.ndarray, bits: np.ndarray) -> list[int]:
    """The slots the admit unit refuses: not admitted, yet not the canonical empty row."""
    return [int(i) for i in np.flatnonzero(nonempty(rows) & ~np.asarray(bits, dtype=bool))]


def hollow(rows: np.ndarray, selected: np.ndarray) -> list[int]:
    """The slots the admit unit refuses: selected by the queue, yet holding the canonical empty row.  The unit
    admits none of them, so they credit nothing even unrefused."""
    return [int(i) for i in np.flatnonzero(~nonempty(rows) & np.asarray(selected, dtype=bool))]


def scale_only(rows: np.ndarray) -> list[int]:
    """The slots whose rows hold zero s8 bytes under a nonzero scale.  The admit unit reads no byte, so it admits such a
    row when selected, and its tile is live under its admit bit (`PouwH1Tile_v4`): computed and credited."""
    r = np.asarray(rows, dtype=np.uint8).reshape(rows.shape[0], -1)
    return [int(i) for i in np.flatnonzero(~r[:, :-4].any(axis=1) & r[:, -4:].any(axis=1))]


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


def tile_words(ri: int) -> range:
    """Row block ri's admit words: the leaves of the call's admit tree its tile's `admit` input is bound to."""
    return range(R * ri // WORD, (R * ri + R) // WORD)


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
    return {"population": population,
            "draw": "uniform over every unit position of the window, live or dead, by the window's draw coin: public "
                    "structure, never weighted by an annotation",
            "always_proved": {"admit": admit_units},
            "admit_unit": "PouwQueueAdmit on the step, 16 slots' queue entries (the slot map's leaf and each slot's "
                          "request entry, opened from the queue committed before the first coin) and their rows' scales, "
                          "outputs their 16 admit bits, the leaf of the call's admit tree it writes; it opens the 16 slots' "
                          "A rows from the call's a-table and refuses a slot not admitted whose row is not the canonical "
                          "empty row and a slot selected whose row is; it reads no row byte, so a selected row of zero "
                          "bytes under a nonzero scale is admitted",
            "tile_unit": "PouwH1Tile_v4 on the tile's 32 A rows, 32 B rows and 32 admit bits; its admit input is bound "
                         "by commitment equality to the two leaves of the call's admit tree that the tally reads, so its "
                         "live bit (their OR) is the tally's liveness: credit counts exactly the tiles live in-circuit, "
                         "an admitted tile of zero-byte rows included (RT4-LOGIC-4, RT4-LOGIC-20)",
            "tally": f"no proof block until a tally Definition exists; the harness computes and commits {steps * len(sites)} "
                     f"per (step, site) popcount sums (<= 2^11), {len(sites)} per-site sums over the steps (< 2^16), and "
                     "the level (window.level(floor_point(sum_site count N K)), <= 9 bits)",
            "dead_unit": "a drawn dead unit proves, with the live tile's template (PouwH1Tile_v4), that its 32 bound "
                         "admit bits are 0 and so its receipt is the canonical empty receipt; no noise is computed when "
                         "serving. Its rows are the canonical empty row because the admit unit refuses a slot not "
                         "admitted whose row is not",
            "egress": "no dead slot's output reaches a served output: the receipts are beside the served y and feed no "
                      "egress, and vLLM samples each served token from the step's selected rows only (admit is the served "
                      "prefix [1] * m + [0] * (T - m) in every call)"}


__all__ = ["bits_of", "hollow", "junk", "live_blocks", "live_units", "nonempty", "proof_plan", "published", "queue_admit",
           "queue_leaves", "queue_of", "queue_select", "scale_only", "scales", "tally", "tile_words", "word_rows", "words"]
