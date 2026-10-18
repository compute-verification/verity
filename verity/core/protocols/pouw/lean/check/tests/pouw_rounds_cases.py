"""The rounds difftest's cases: `core_check.rounds`'s arguments, ``{"coins", "roots", "openings"}``.

A case is an honest live window (a live epoch's coins, `coins.live_challenger`, its round roots recorded in order and each
round's coin opened after its root, `coins.LiveRecord`), then one mutation (`ROUNDS_NAMES`) aimed at one refusal of
`rounds`. `tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py`
holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import random

from verity.core.protocols.pouw import coins as Co

#: the mutations that need at least one recorded round, and those that need one round left unrecorded
_RECORDED = {"twice", "early-root", "never", "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "label",
             "openings-form", "swapped"}
_LEFT = {"early"}


def _flip(h: str) -> str:
    b = bytearray(bytes.fromhex(h))
    b[0] ^= 1
    return b.hex()


def _record(ch: Co.LiveChallenger, roots: list[bytes]) -> list[dict]:
    rec = Co.LiveRecord(ch, Co.ROUND)
    return [rec.record(r).to_json() for r in roots]


def honest_rounds(rng: random.Random, name: str = "honest") -> tuple[dict, Co.LiveChallenger]:
    """An honest live window: its epoch's coins and the round coins it opened, after each recorded round root."""
    rounds = rng.choice([1, 2, 3, 4, 8]) if name != "swapped" else rng.choice([2, 3, 4, 8])
    ch = Co.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    lo = 2 if name == "swapped" else 1 if name in _RECORDED else 0
    k = rng.randint(lo, rounds - 1 if name in _LEFT else rounds)
    roots = [rng.randbytes(64) for _ in range(k)]
    return {"coins": ch.commitment.to_json(), "roots": [r.hex() for r in roots], "openings": _record(ch, roots)}, ch


def _mutate(d: dict, ch: Co.LiveChallenger, name: str, rng: random.Random) -> None:
    c, roots, ops = ch.commitment, d["roots"], d["openings"]
    j = rng.randrange(len(ops)) if ops else 0
    match name:
        case "verifier-coins":
            d["coins"] = rng.choice([None, {**d["coins"], "root": "00"}, {**d["coins"], "scheme": "x"},
                                     {**d["coins"], "families": "x"}, {**d["coins"], "epoch": -1},
                                     {**d["coins"], "families": [["salt", 1], ["salt", 1]]}])
        case "seeded-coins":
            d["coins"] = Co.Challenger(c.epoch, rng.randbytes).commitment.to_json()
        case "not-live":
            scheme, layout = rng.choice([
                ("verity/vllm/commit-coins/v1", Co.live_families(len(roots) + 1)),
                (Co.LIVE_SCHEME, (("draw", 1), ("salt", 1), (Co.ROUND, len(roots) + 1))),
                (Co.LIVE_SCHEME, (("salt", 1), (Co.ROUND, len(roots) + 1))),
                (Co.LIVE_SCHEME, (("salt", 1), ("draw", 1), ("rounds", len(roots) + 1)))])
            d["coins"] = Co.LiveChallenger(scheme, c.epoch, layout, rng.randbytes).commitment.to_json()
        case "roots-form":
            d["roots"] = rng.choice([None, "x", roots + ["zz"], roots + [rng.randbytes(32).hex()], [5], {"0": roots}])
        case "openings-list":
            d["openings"] = rng.choice([None, {"0": ops}, "x"])
        case "openings-form":
            k = rng.choice(["coin", "path", "label", "index", "epoch"])
            bad = {"coin": ["zz", None], "path": [None, 4, [rng.randbytes(33).hex()]], "label": [5, None],
                   "index": ["0", None, True, 1.5], "epoch": [str(c.epoch), None]}[k]
            d["openings"] = rng.choice([ops[:j] + [{**ops[j], k: rng.choice(bad)}] + ops[j + 1:], ops + [5]])
        case "twice":
            d["openings"] = ops[:j + 1] + [ops[j]] + ops[j + 1:]
        case "early":
            after = bytes.fromhex(roots[-1]) if roots else rng.randbytes(64)
            d["openings"] = ops + [ch.open(Co.ROUND, len(roots), after).to_json()]
        case "early-root":
            d["roots"] = roots[:-1]
        case "never":
            d["openings"] = ops[:-1]
        case "swapped":
            d["openings"] = [ops[1], ops[0]] + ops[2:]
        case "after":
            d["openings"] = ops[:j] + [{**ops[j], "after": _flip(ops[j]["after"])}] + ops[j + 1:]
        case "root":
            d["roots"] = roots[:j] + [_flip(roots[j])] + roots[j + 1:]
        case "foreign-epoch":
            d["openings"] = _record(Co.live_challenger(c.epoch + 1, c.family(Co.ROUND), rng.randbytes),
                                    [bytes.fromhex(r) for r in roots])
        case "foreign":
            d["openings"] = _record(Co.live_challenger(c.epoch, c.family(Co.ROUND), rng.randbytes),
                                    [bytes.fromhex(r) for r in roots])
        case "tampered":
            d["openings"] = ops[:j] + [{**ops[j], "salt": _flip(ops[j]["salt"])}] + ops[j + 1:]
        case "malformed":
            d["openings"] = ops[:j] + [{**ops[j], "coin": ops[j]["coin"][:-2]}] + ops[j + 1:]
        case "too-many":
            d["roots"] = roots + [rng.randbytes(64).hex() for _ in range(c.family(Co.ROUND) + 1 - len(roots))]
        case "label":
            d["openings"] = [ch.open("salt", 0, bytes.fromhex(roots[0])).to_json()] + ops[1:]
        case _:
            raise ValueError(name)


ROUNDS_NAMES = ["verifier-coins", "seeded-coins", "not-live", "roots-form", "openings-list", "openings-form", "twice", "early",
                "early-root", "never", "swapped", "after", "root", "foreign-epoch", "foreign", "tampered", "malformed",
                "too-many", "label"]


def rounds_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `rounds` case: an honest live window, then mutation `name`."""
    d, ch = honest_rounds(rng, name)
    if name != "honest":
        _mutate(d, ch, name, rng)
    return d


def rounds_args(d: dict) -> tuple:
    return d["coins"], d["roots"], d["openings"]
