"""The charge difftest's data: random instances, and `reference_charge`'s results on each, for `ChargeDifftestMain.lean`.

    python scripts/charge_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/charge/instances.jsonl`, one instance per line as `NetworkCertifier.AdviceV1.ChargeDifftest.rawOfJson` reads
it, and `DIR/charge/python.jsonl`, the same line's results from the checkout's
`verity.core.protocols.certifier.arith_encode`: `reference_charge` window by window from the instance's carry, each
window's charge and the carry it passes on, up to and including the first it refuses (`steps`), and the windows' charges
chained from the book's priors and run 0, or none when one refuses (`charges`). The Lean audit runs
`ChargeDifftestMain.lean` on both files (`runs` in `lean-audit.json`, which pins their SHA-256), and
`verity/core/protocols/certifier/tests/test_network_certifier_charge_lean_difftest.py` checks that this script still
writes them.

Half the instances start from a random carry (counts small, at and just below the `2^KW` the book holds, where a
decision halves them, or past it, and runs up to 3, which `reference_charge` refuses), and half from the book's priors
and run 0, a session of up to six windows. Books are drawn
with n_arrived's bins from 1 to 3,000 or 0, totals at the bins' sum, above it, below it (a symbol past its total) or past
`MAX_TOTAL`, and zero counts at their total, below it or 0; and steep books, frequency-1 symbols at totals near
`MAX_TOTAL`, whose symbols take 17 scalings (a symbol's interval is at least `2^14` wide, so none takes more than
`K_MAX`, 18); and certain books, n_arrived's whole total in bin 0 and zero counts at their
total, under which a window with no asks doesn't use the coder. Windows have 1 to 70 slots, each asking or not, and their
decisions are mostly 0, even or mostly 1. Every instance has `N` slots per window and bins `0 … bitlen N`, which
`reference_charge` requires (it raises otherwise).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.arith_encode import RUNS, Carry, ChargeBook, reference_charge

COUNT = 4_000
KW = 15


def _steep(rng: random.Random, N: int) -> ChargeBook:
    """A book of frequency-1 symbols at totals near `MAX_TOTAL`, whose symbols take the most scalings."""
    bins = N.bit_length() + 1
    arrived = [rng.choice([1, 1, 2, rng.randint(1, 1 << 15)]) for _ in range(bins)]
    total = min(1 << 16, sum(arrived) + rng.randint(0, 1 << 16))
    zeros = [(rng.choice([1, 1, 2, 3]), rng.choice([1 << 16, (1 << 16) - rng.randint(0, 9), rng.randint(1 << 14, 1 << 16)]))
             for _ in range(4)]
    return ChargeBook(tuple(arrived), total, tuple(zeros))


def _certain(rng: random.Random, N: int) -> ChargeBook:
    """A book whose bin 0 holds n_arrived's whole total and whose zero counts are certain: a window with no asks codes
    no symbol below its total, so it doesn't use the coder."""
    bins = N.bit_length() + 1
    total = rng.randint(1, 1 << 16)
    arrived = [total] + [rng.choice([0, rng.randint(1, 50)]) for _ in range(bins - 1)]
    zeros = [(t, t) for t in (rng.randint(1, 1 << 16) for _ in range(4))]
    if rng.random() < 0.3:
        zeros[rng.randrange(4)] = (1, 2)
    return ChargeBook(tuple(arrived), total, tuple(zeros))


def _book(rng: random.Random, N: int) -> tuple[ChargeBook, tuple[tuple[int, int], ...]]:
    bins = N.bit_length() + 1
    if rng.random() < 0.15:
        return _steep(rng, N), _counts(rng)
    if rng.random() < 0.08:
        return _certain(rng, N), _counts(rng)
    if rng.random() < 0.6:
        arrived = [rng.choice([0] + [rng.randint(1, 50)] * 6 + [rng.randint(1, 3000)] * 6) for _ in range(bins)]
    elif rng.random() < 0.8:
        arrived = [rng.randint(1, 3000) for _ in range(bins)]
    else:
        arrived = [rng.randint(1, 3) for _ in range(bins)]
    s, r = sum(arrived), rng.random()
    if r < 0.04:
        total = max(1, s - rng.randint(0, max(1, s)))
    elif r < 0.06:
        total = (1 << 16) + rng.randint(0, 10)
    else:
        total = s + rng.choice([0, 0, rng.randint(0, 3000)])
    zeros = []
    for _ in range(4):
        if rng.random() < 0.05:
            t = rng.choice([1, 2, rng.randint(1, 100), rng.randint(1, 1 << 16), (1 << 16) + 1])
        else:
            t = rng.choice([1, 2, rng.randint(1, 100), rng.randint(1, 1000), rng.randint(1, 1 << 16)])
        f = rng.choice([t, rng.randint(1, t), rng.randint(1, t)]) if rng.random() > 0.03 else 0
        zeros.append((f, t))
    return ChargeBook(tuple(arrived), total, tuple(zeros)), _counts(rng)


def _counts(rng: random.Random) -> tuple[tuple[int, int], ...]:
    out = []
    for _ in range(RUNS):
        r = rng.random()
        if r < 0.5:
            out.append((rng.randint(0, 20), rng.randint(0, 20)))
        elif r < 0.65:
            a = rng.randint(0, 1 << KW)
            out.append((a, (1 << KW) - a - rng.choice([0, 1, 1, 2, 3, 4])) if a < 1 << KW else (a, 0))
        elif r < 0.85:
            a = rng.randint(0, (1 << KW) - 1)
            out.append((a, rng.randint(0, (1 << KW) - 1 - a)))
        else:
            out.append((rng.randint(0, 1 << KW), rng.randint(0, 1 << KW)))
    return tuple(out)


def _window(rng: random.Random, N: int) -> tuple[list[int], list[int]]:
    asks, ones = rng.random(), rng.choice([0.1, 0.5, 0.9])
    return [int(rng.random() < asks) for _ in range(N)], [int(rng.random() < ones) for _ in range(N)]


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    N = rng.choice([1, 2, 3, 5, 14, 20, rng.randint(1, 70)])
    book, priors = _book(rng, N)
    if seed % 2:
        carry = Carry(_counts(rng), rng.choice([0, 0, 1, 2, 2, 0, 1, 2, 3]))
        windows = [_window(rng, N) for _ in range(rng.choice([1, 1, 2, 3]))]
    else:
        carry = Carry(priors, 0)
        windows = [_window(rng, N) for _ in range(rng.randint(0, 6))]
    return {"N": N, "book": book, "priors": priors, "carry": carry, "windows": windows}


def _pairs(ps) -> list[list[int]]:
    return [[a, b] for a, b in ps]


def lean_input(inst: dict) -> dict:
    book, carry = inst["book"], inst["carry"]
    return {"N": inst["N"],
            "book": {"arrived": list(book.arrived), "total": book.total, "zeros": _pairs(book.zeros),
                     "priors": _pairs(inst["priors"])},
            "carry": {"counts": _pairs(carry.counts), "run": carry.run},
            "windows": [{"en": en, "bit": bit} for en, bit in inst["windows"]]}


def python_results(inst: dict) -> dict:
    N, book = inst["N"], inst["book"]
    steps, carry = [], inst["carry"]
    for en, bit in inst["windows"]:
        ok, length, carry = reference_charge(N, book, carry, en, bit)
        if not ok:
            steps.append(None)
            break
        steps.append([length, _pairs(carry.counts), carry.run])
    charges, carry = [], Carry(inst["priors"], 0)
    for en, bit in inst["windows"]:
        ok, length, carry = reference_charge(N, book, carry, en, bit)
        if not ok:
            charges = None
            break
        charges.append(length)
    return {"steps": steps, "charges": charges}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/charge``."""
    d = out / "charge"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"charge/instances.jsonl": d / "instances.jsonl", "charge/python.jsonl": d / "python.jsonl"}
    with open(paths["charge/instances.jsonl"], "w") as fi, open(paths["charge/python.jsonl"], "w") as fp:
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
