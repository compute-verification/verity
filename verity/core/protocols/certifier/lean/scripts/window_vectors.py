"""The window charge's difftest data: random instances, and `reference_window`'s results on each, for
`WindowDifftestMain.lean`.

    python scripts/window_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/window/instances.jsonl`, one instance per line as `NetworkCertifier.WindowCharge.WindowDifftest.rawOfJson`
reads it, and `DIR/window/python.jsonl`, the same line's results from the checkout's
`verity.core.protocols.certifier.window_charge`: `reference_window` window by window from the instance's carry, each
window's charge and the carry it passes on, up to and including the first it refuses (`steps`), and the windows' charges
chained from the book's first carry (its priors, run 0 and no order entry yet), or none when one refuses (`charges`).
The Lean audit runs `WindowDifftestMain.lean` on both files (`runs` in `lean-audit.json`, which pins their SHA-256), and
`verity/core/protocols/certifier/tests/test_network_certifier_window_lean_difftest.py` checks that this script still
writes them.

Half the instances start from a random carry (KT counts small, at and just below the `2^KW` the book holds, or past it;
runs up to 3 and ranks up to `INDEX_CAP + 2`, which `reference_window` refuses; positions anywhere below `2^32 − 1`),
a window to three, and half from the book's first carry, a session of up to five windows. A window has 0 to 40 decision
slots and 0 to 16 hit and order slots each. Books are sampled with each table's bins from 1 to 2,000, totals at the
bins' sum or above it, and now and then a bin of 0, a total past `MAX_TOTAL` or below the bins' sum; steep books,
frequency-1 bins at totals near `MAX_TOTAL`, whose symbols take 17 scalings; and certain books, each table's whole total
in bin 0, under which a window with no slot that holds a value doesn't use the coder. Slots hold a value or not at a
random density; hits are at and past `HIT_CAP`, and order entries' gaps at and past `GAP_CAP` (wide enough that their
offsets take `_uniform`'s two steps), before the carry's position, and at `2^32 − 1`, with ranks at and past
`INDEX_CAP`. Every instance has its windows' slot counts and its tables' bins (`window_charge._bins`), which
`reference_window` requires (it raises otherwise).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.arith_encode import MAX_TOTAL, RUNS
from verity.core.protocols.certifier.window_charge import (GAP_CAP, HIT_CAP, INDEX_CAP, WindowBook, WindowCarry, _bins,
                                                           reference_window)

COUNT = 4_000
KW = 15


def _table(rng: random.Random, bins: int, kind: str) -> tuple[tuple[int, ...], int]:
    if kind == "steep":
        freqs = [rng.choice([1, 1, 2, rng.randint(1, 1 << 11)]) for _ in range(bins)]
        return tuple(freqs), rng.choice([MAX_TOTAL, MAX_TOTAL - rng.randint(0, 9), rng.randint(sum(freqs), MAX_TOTAL)])
    if kind == "certain":
        total = rng.randint(1, MAX_TOTAL)
        return (total,) + tuple(rng.choice([0, rng.randint(1, 50)]) for _ in range(bins - 1)), total
    freqs = [rng.choice([1, rng.randint(1, 64), rng.randint(1, 2000)]) for _ in range(bins)]
    total = min(sum(freqs) + rng.choice([0, 0, rng.randint(0, 4000)]), MAX_TOTAL)
    if kind == "bad":
        r = rng.random()
        if r < 0.4:
            freqs[rng.randrange(bins)] = 0
        elif r < 0.8:
            total = max(1, sum(freqs) - rng.randint(1, sum(freqs)))
        else:
            total = MAX_TOTAL + rng.randint(1, 10)
    return tuple(freqs), total


def _book(rng: random.Random, N: int, H: int, O: int) -> tuple[WindowBook, str]:
    """A book and its kind: every table steep, certain or random, and now and then one random table made bad (a bin of
    0, a total below the bins' sum or past `MAX_TOTAL`)."""
    kind = rng.choices(["steep", "certain", "random"], [0.12, 0.08, 0.8])[0]
    bad = rng.randrange(8) if kind == "random" and rng.random() < 0.15 else None
    bins = _bins(N, H, O)
    return WindowBook(tuple(_table(rng, b, "bad" if i == bad else kind) for i, (_name, b) in enumerate(bins))), kind


def _counts(rng: random.Random) -> tuple[tuple[int, int], ...]:
    out = []
    for _ in range(RUNS):
        r = rng.random()
        if r < 0.55:
            out.append((rng.randint(0, 20), rng.randint(0, 20)))
        elif r < 0.7:
            a = rng.randint(0, (1 << KW) - 1)
            out.append((a, max(0, (1 << KW) - a - rng.choice([1, 1, 1, 2, 3, 4, 0]))))
        elif r < 0.96:
            a = rng.randint(0, (1 << KW) - 1)
            out.append((a, rng.randint(0, (1 << KW) - 1 - a)))
        else:
            out.append((rng.randint(0, 1 << KW), rng.randint(0, 1 << KW)))
    return tuple(out)


def _carry(rng: random.Random) -> WindowCarry:
    run = 3 if rng.random() < 0.04 else rng.choice([0, 1, 2])
    pos = rng.choice([0, 0, rng.randrange(1 << 20), rng.randrange(1 << 31), rng.randrange(1 << 31),
                      (1 << 32) - 2 - rng.randrange(1 << 26), (1 << 32) - 2])
    ranks = [0, rng.randrange(64), rng.randrange(1 << 24), INDEX_CAP, INDEX_CAP + 1]
    rank = INDEX_CAP + 2 if rng.random() < 0.04 else rng.choice(ranks)
    return WindowCarry(_counts(rng), run, pos, rank)


def _slots(rng: random.Random, N: int, H: int, O: int, carry: WindowCarry,
           empty: bool = False) -> tuple[list[int], ...]:
    """A window's slots (en, bit, hen, short, oen, p, r): each field's slots hold a value at a density of its own (none
    when ``empty``), the values at their caps' edges now and then and past them, the order entries' positions after the
    last one's or before it."""
    def held(n: int) -> list[int]:
        d = 0.0 if empty else rng.choice([0.0, 0.3, 0.7, 1.0])
        return [int(rng.random() < d) for _ in range(n)]

    def past(good: list[int], bad: list[int]) -> int:
        return rng.choice(bad) if rng.random() < 0.01 else rng.choice(good)

    ones = rng.choice([0.1, 0.5, 0.9])
    en, bit = held(N), [int(rng.random() < ones) for _ in range(N)]
    hen = held(H)
    short = [past([0, 1, 2, rng.randrange(64), rng.randrange(1 << 16), 1 << 15, HIT_CAP], [HIT_CAP + 1, (1 << 17) - 1])
             for _ in range(H)]
    oen, p, r = held(O), [], []
    pos, rank = carry.pos, carry.rank
    for e in oen:
        g = past([0, 1, rng.randrange(64), rng.randrange(1 << 16), (1 << 17) + rng.randrange(1 << 10),
                  rng.randrange(1 << 24), GAP_CAP], [GAP_CAP + 1, -1, -1])
        pp = pos + g
        if not 0 <= pp < (1 << 32) - 1:
            pp = min(pos, (1 << 32) - 1)
        if rng.random() < 0.004:
            pp = (1 << 32) - 1
        rr = past([0, rank, max(rank - 1, 0), min(rank + rng.randrange(64), INDEX_CAP), rng.randrange(1 << 24),
                   INDEX_CAP], [INDEX_CAP + 1])
        p.append(pp)
        r.append(rr)
        if e:
            pos, rank = pp + 1, rr + 1
    return en, bit, hen, short, oen, p, r


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    N = rng.choice([0, 1, 2, 3, 5, 14, rng.randint(0, 40)])
    H = rng.choice([0, 0, 1, 2, 3, 6, rng.randint(0, 16)])
    O = rng.choice([0, 0, 1, 2, 3, 6, rng.randint(0, 16)])
    (book, kind), priors = _book(rng, N, H, O), _counts(rng)
    if seed % 2:
        carry, n = _carry(rng), rng.choice([1, 1, 2, 3])
    else:
        carry, n = WindowCarry(priors), rng.randint(0, 5)
    windows, c = [], carry
    for _ in range(n):
        slots = _slots(rng, N, H, O, c, kind == "certain" and rng.random() < 0.8)
        windows.append(slots)
        ok, _length, nxt = reference_window(N, H, O, book, c, *slots)
        c = nxt if ok else c
    return {"N": N, "H": H, "O": O, "book": book, "priors": priors, "carry": carry, "windows": windows}


def _pairs(ps) -> list[list[int]]:
    return [[a, b] for a, b in ps]


def lean_input(inst: dict) -> dict:
    carry = inst["carry"]
    return {"N": inst["N"], "H": inst["H"], "O": inst["O"],
            "book": {"tables": [[list(f), t] for f, t in inst["book"].tables], "priors": _pairs(inst["priors"])},
            "carry": {"counts": _pairs(carry.counts), "run": carry.run, "pos": carry.pos, "rank": carry.rank},
            "windows": [dict(zip(("en", "bit", "hen", "short", "oen", "p", "r"), slots)) for slots in inst["windows"]]}


def _chain(inst: dict, carry: WindowCarry) -> list[tuple[int, WindowCarry] | None]:
    """`reference_window` on each window from `carry`, up to and including the first it refuses (None)."""
    out = []
    for slots in inst["windows"]:
        ok, length, carry = reference_window(inst["N"], inst["H"], inst["O"], inst["book"], carry, *slots)
        if not ok:
            return out + [None]
        out.append((length, carry))
    return out


def python_results(inst: dict) -> dict:
    steps = [None if s is None else [s[0], _pairs(s[1].counts), s[1].run, s[1].pos, s[1].rank]
             for s in _chain(inst, inst["carry"])]
    charges = _chain(inst, WindowCarry(inst["priors"]))
    return {"steps": steps, "charges": None if None in charges else [s[0] for s in charges]}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/window``."""
    d = out / "window"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"window/instances.jsonl": d / "instances.jsonl", "window/python.jsonl": d / "python.jsonl"}
    with open(paths["window/instances.jsonl"], "w") as fi, open(paths["window/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(lean_input(inst)))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is audited")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
