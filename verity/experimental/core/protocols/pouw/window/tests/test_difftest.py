"""The window's Lean against `verity.experimental.protocols.pouw.window`, on the shared harness (`verity_difftest`): one
`Port` per function of `window.py`, served by `pouw-window-difftest` (`PouwWindow/Difftest.lean`).

The inputs are drawn by the generators below from a Hypothesis-controlled `random.Random` (`st.randoms`), so a run is
deterministic and shrinks. The grid's numbers run over bits 0 … 6, n up to 2^60 and the integers at or below 0; K′'s
parameters over realistic ε and δ (down to 2^-40), grid-derived and random work up to n = 2^60, cases where (1 − p)^K = δ
exactly, and every way out of its domain; the windows are well-formed advice that is then, for most, mutated to reach each
of `window_problem`'s refusals (`MUTATIONS`), whose reason is its output, so the test holds which check refused. `tile_at`
and `tile_in` run on windows it accepts. JSON floats are outside the harness's canonical form, so the entries' strictness
on them is `test_pouw_window.py`'s.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.experimental.protocols.pouw import window as Wd

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"


def _frac(q: Fraction) -> list[int]:
    return [q.numerator, q.denominator]


def _q(v: list[int]) -> Fraction:
    return Fraction(v[0], v[1])


def _work(v: dict) -> Wd.Work:
    return Wd.Work(v["level"], v["credited"], v["n"])


# ---- the grid -----------------------------------------------------------------------------------------------------------

def _number(rng: random.Random, bits: int) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.randint(0, 300)
    if r < 0.45:
        return Wd.point(rng.randint(0, 400), bits)
    if r < 0.55:
        return (1 << rng.randint(0, 60)) + rng.choice((-1, 0, 1))
    if r < 0.62:
        return -rng.randint(0, 1 << rng.randint(1, 62))
    return rng.getrandbits(rng.randint(1, 60))


def grid_n(rng: random.Random) -> dict:
    bits = rng.randint(0, 6)
    return {"bits": bits, "n": _number(rng, bits)}


def grid_ell(rng: random.Random) -> dict:
    return {"bits": rng.randint(0, 6), "ell": rng.choice((rng.randint(0, 60), rng.randint(0, 2000), -rng.randint(1, 5)))}


def levels_in(rng: random.Random) -> dict:
    bits = rng.randint(0, 6)
    lo, hi = sorted((_number(rng, bits), _number(rng, bits)))
    if rng.random() < 0.2:
        lo, hi = hi, lo
    return {"bits": bits, "lo": lo, "hi": hi}


# ---- the draw -----------------------------------------------------------------------------------------------------------

def _eps(rng: random.Random) -> Fraction:
    return rng.choice((Fraction(1, 2), Fraction(1, 2), Fraction(1, 1000), Fraction(1), Fraction(1, 3), Fraction(9, 10),
                       Fraction(rng.randint(1, 1000), 1000), Fraction(rng.randint(1, 7), 7)))


def _delta(rng: random.Random) -> Fraction:
    return rng.choice((Fraction(1, 10), Fraction(1, 2 ** 40), Fraction(1, 2 ** rng.randint(1, 40)), Fraction(1, 3),
                       Fraction(1, 10 ** rng.randint(1, 12)), Fraction(rng.randint(1, 999), 1000)))


def _draw_params(rng: random.Random) -> tuple[Fraction, Fraction, int, int, int]:
    eps, delta = _eps(rng), _delta(rng)
    r = rng.random()
    if r < 0.35:
        bits = rng.choice((3, 3, rng.randint(0, 6)))
        w = Wd.bucket(rng.choice((rng.randint(1, 5000), rng.getrandbits(rng.randint(1, 40)) + 1)), bits)
        tile = rng.choice((1, 4096, rng.randint(1, 1 << 20)))
        return eps, delta, w.credited * tile, w.n, tile
    if r < 0.55:
        n, w = rng.getrandbits(rng.randint(1, 60)) + 1, rng.randint(1, 1 << rng.randint(0, 20))
        return eps, delta, rng.randint(1, n * w), n, w
    if r < 0.65:
        n, w = rng.getrandbits(rng.randint(20, 60)) + 1, rng.randint(1, 1 << 10)
        return eps, delta, rng.randint(1, 1 << rng.randint(0, 20)), n, w
    if r < 0.8:
        # (1 − p)^K = δ exactly: p = ε·credited/(n·w), δ its complement's K-th power
        n, w = rng.randint(1, 40), 1
        credited = rng.randint(1, n)
        p = eps * credited / n
        if p == 1:
            return eps, delta, credited, n, w
        return eps, (1 - p) ** rng.randint(1, 30), credited, n, w
    n, w = rng.randint(1, 1 << 30), rng.randint(1, 100)
    bad = rng.choice(("eps0", "eps>1", "eps<0", "delta0", "delta1", "delta>1", "delta<0", "n0", "n<0", "w0", "w<0",
                      "credited0", "credited<0", "credited>nw"))
    credited = rng.randint(1, n * w)
    return (Fraction(0) if bad == "eps0" else Fraction(3, 2) if bad == "eps>1" else Fraction(-1, 2) if bad == "eps<0" else eps,
            Fraction(0) if bad == "delta0" else Fraction(1) if bad == "delta1" else Fraction(5, 4) if bad == "delta>1"
            else Fraction(-1, 10) if bad == "delta<0" else delta,
            0 if bad == "credited0" else -credited if bad == "credited<0" else n * w + rng.randint(1, 9)
            if bad == "credited>nw" else credited,
            0 if bad == "n0" else -n if bad == "n<0" else n, 0 if bad == "w0" else -w if bad == "w<0" else w)


def k_prime_in(rng: random.Random) -> dict:
    eps, delta, credited, n, w = _draw_params(rng)
    return {"eps": _frac(eps), "delta": _frac(delta), "credited": credited, "n": n, "w_max": w}


def _some_work(rng: random.Random, bits: int) -> Wd.Work:
    return Wd.Work.at(rng.randint(0, 400), bits) if rng.random() < 0.6 else Wd.bucket(rng.randint(1, 1 << 40), bits)


def sampling_in(rng: random.Random) -> dict:
    eps, delta, bits = _eps(rng), _delta(rng), rng.randint(0, 6)
    if rng.random() < 0.15:
        eps = rng.choice((Fraction(0), Fraction(2)))
    return {"eps": _frac(eps), "delta": _frac(delta), "work": asdict(_some_work(rng, bits)),
            "tile_work": rng.choice((1, 4096, rng.randint(1, 1 << 20), 0, -3)), "bits": bits}


def work_to_json_in(rng: random.Random) -> dict:
    return {"work": asdict(_some_work(rng, rng.randint(0, 6))), "tile_work": rng.choice((1, 4096, rng.randint(-5, 1 << 20)))}


# ---- the advice ---------------------------------------------------------------------------------------------------------

def _side(rng: random.Random, big: bool) -> dict:
    rows = rng.randint(1, 64 if big else 8)
    return {"rows": rows, "keep": sorted(rng.sample(range(rows), rng.randint(1, rows)))}


def window_advice(rng: random.Random) -> tuple[list, Wd.Work, int, int]:
    """Well-formed advice: its entries, its work level, the bound and the grid."""
    bits = rng.choice((3, 3, 3, rng.randint(0, 6)))
    big = rng.random() < 0.3
    entries, at = [], 0
    for _ in range(rng.randint(1, 4)):
        a, b = _side(rng, big), _side(rng, big)
        size = len(a["keep"]) * len(b["keep"])
        entries.append({"kind": "call", "lo": at, "hi": at + size, "a": a, "b": b})
        at += size
    work = Wd.bucket(at, bits)
    rest = work.n - at
    ra = rng.randint(1, 4)
    rb = -(-rest // ra) + rng.randint(0, 2)
    entries.append({"kind": "filler", "lo": at, "hi": work.n, "a": {"rows": ra, "keep": list(range(ra))},
                    "b": {"rows": rb, "keep": list(range(rb))}})
    bound = len(entries) + rng.randint(0, 3)
    for i in range(len(entries), bound):
        entries.append(rng.choice(({"kind": "none"}, {"kind": "none", "index": i}, {"kind": "none", "lo": "x", "a": 7})))
    return entries, work, bound, bits


def _real(entries: list) -> int:
    return len(entries) - 1 - [e.get("kind") if isinstance(e, dict) else None for e in entries][::-1].index("filler")


def _bad_number(rng: random.Random, x: int) -> Any:
    return rng.choice((x + 1, x - 1, -1 - x, str(x), None, [x], x == 1 or x == 0, {"n": x}))


def _mutate_side(rng: random.Random, side: dict) -> Any:
    rows, keep = side["rows"], side["keep"]
    m = rng.randint(0, 13)
    if m == 0:
        return {**side, "rows": rng.choice((0, -1, str(rows), True, None, [rows]))}
    if m == 1:
        return {"keep": keep}
    if m == 2:
        return {**side, "keep": rng.choice(([], "x", {}, None, 3))}
    if m == 3:
        return {**side, "keep": keep + [rows + rng.randint(0, 3)]}
    if m == 4:
        return {**side, "keep": [-1] + keep}
    if m == 5:
        return {**side, "keep": keep[::-1] if len(keep) > 1 else keep + keep}
    if m == 6:
        return {**side, "keep": keep + keep[-1:]}
    if m == 7:
        return {**side, "keep": [True] + keep[1:]} if keep[0] == 1 else {**side, "keep": [None] + keep[1:]}
    if m == 8:
        return {**side, "keep": [str(keep[0])] + keep[1:]}
    if m == 9:
        return rng.choice((None, [], 5, "side", [rows, keep]))
    if m == 10:
        return {**side, "rows": max(keep)}
    if m == 11:
        return {**side, "rows": rows + rng.randint(1, 5)}
    if m == 12:
        return {**side, "extra": 1}
    return {"rows": rows}


def _m_bound(rng, es, work, bound):
    return es, work, rng.choice((bound + 1, bound - 1, -bound, 0))


def _m_kind(rng, es, work, bound):
    i = rng.randrange(len(es))
    es[i] = rng.choice(({**es[i], "kind": rng.choice(("Call", "", "filler ", 1, None, ["call"], {"k": 1}, True))},
                        {k: v for k, v in es[i].items() if k != "kind"}, rng.choice((5, "call", [], None, False))))
    return es, work, bound


def _m_order(rng, es, work, bound):
    f = _real(es)
    m = rng.randint(0, 5)
    if m == 0:
        es[0] = {**es[0], "kind": "filler"}
    elif m == 1:
        es[f] = {**es[f], "kind": "call"}
    elif m == 2 and f + 1 < len(es):
        es[f + 1] = {**es[f + 1], "kind": rng.choice(("call", "filler"))}
    elif m == 3 and f >= 1:
        es[f - 1] = {**es[f - 1], "kind": "none"}
    elif m == 4:
        es.insert(0, es.pop(f))
    else:
        es[f], es[-1] = es[-1], es[f]
    return es, work, bound


def _m_keep(rng, es, work, bound):
    i, s = rng.randint(0, _real(es)), rng.choice("ab")
    es[i] = {**es[i], s: _mutate_side(rng, es[i][s])}
    if rng.random() < 0.3:
        es[i] = {k: v for k, v in es[i].items() if k != s}
    return es, work, bound


def _m_range(rng, es, work, bound):
    i = rng.randint(0, _real(es))
    k = rng.choice(("lo", "hi"))
    e = dict(es[i])
    if rng.random() < 0.2:
        e.pop(k)
    elif k == "hi" and rng.random() < 0.3:
        e["hi"] = e["lo"] - rng.randint(0, 1)
    else:
        e[k] = _bad_number(rng, e[k])
    es[i] = e
    return es, work, bound


def _m_tiles(rng, es, work, bound):
    """A call's or the filler's range off its tiles, the later ranges moved with it."""
    i, d = rng.randint(0, _real(es)), rng.choice((-1, 1, 2, -2, 7))
    if es[i]["hi"] + d <= es[i]["lo"]:
        d = 1
    es[i] = {**es[i], "hi": es[i]["hi"] + d}
    for j in range(i + 1, _real(es) + 1):
        es[j] = {**es[j], "lo": es[j]["lo"] + d, "hi": es[j]["hi"] + d}
    if rng.random() < 0.5:
        work = Wd.Work(work.level, work.credited, work.n + d)
    return es, work, bound


def _m_filler(rng, es, work, bound):
    f = _real(es)
    size = es[f]["hi"] - es[f]["lo"]
    ra = rng.randint(1, max(1, size))
    rb = max(1, (size - 1) // ra) if size > 1 else 1
    es[f] = {**es[f], "a": {"rows": ra, "keep": list(range(ra))}, "b": {"rows": rb, "keep": list(range(rb))}}
    return es, work, bound


def _m_work(rng, es, work, bound):
    m = rng.randint(0, 4)
    if m == 0:
        return es, Wd.Work(work.level, work.credited, work.n + rng.choice((-1, 1, 1 << 40))), bound
    if m == 1:
        return es, Wd.Work(work.level + rng.choice((-1, 1)) if work.level else 1, work.credited, work.n), bound
    if m == 2:
        return es, Wd.Work(work.level, work.credited + 1, work.n), bound
    if m == 3:
        return es, Wd.Work.at(work.level + 1), bound
    return es, Wd.Work(work.level, work.credited - 1 if work.credited else 1, work.n), bound


MUTATIONS = (_m_bound, _m_kind, _m_order, _m_keep, _m_range, _m_tiles, _m_filler, _m_work)


def window_in(rng: random.Random) -> dict:
    es, work, bound, bits = window_advice(rng)
    for _ in range(rng.choice((0, 0, 1, 1, 1, 1, 2))):
        try:
            es, work, bound = rng.choice(MUTATIONS)(rng, json.loads(json.dumps(es)), work, bound)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            pass  # a second mutation that the first left nothing to act on
    return {"entries": es, "work": asdict(work), "bound": bound, "bits": bits if rng.random() < 0.95 else rng.randint(0, 6)}


def tile_at_in(rng: random.Random) -> dict:
    es, work, _, _ = window_advice(rng)
    return {"entries": es, "t": rng.randrange(work.n) if rng.random() < 0.85 else work.n + rng.randint(0, 3)}


def tile_in_in(rng: random.Random) -> dict:
    es, work, bound, _ = window_advice(rng)
    return {"entry": es[rng.randrange(bound)], "t": rng.randrange(work.n + 2)}


# ---- the ports ----------------------------------------------------------------------------------------------------------

#: The reasons `window_problem` gives, each of which the run must reach, and "accepted" for None.
REASONS = ("accepted", "not the bound", "kind is none of", "the calls, then one filler", "kept rows are not",
           "does not start where", "not its", "the filler's range", "not the padded count", "not the registered level")
SEEN: dict[str, set[str]] = {"window_problem": set(), "k_prime": set()}
#: What each of those runs must reach, so that none is vacuous for one of its checks or regimes.
REACHES = {"window_problem": set(REASONS), "k_prime": {"refused", "capped", "k > 10,000", "n > 2^50", "exact"}}


def window_problem(v: dict) -> str | None:
    reason = Wd.window_problem(v["entries"], _work(v["work"]), v["bound"], v["bits"])
    SEEN["window_problem"].update(r for r in REASONS if (r == "accepted") == (reason is None) and (reason is None or r in reason))
    return reason


def k_prime(v: dict) -> int:
    eps, delta, credited, n, w = _q(v["eps"]), _q(v["delta"]), v["credited"], v["n"], v["w_max"]
    try:
        k = Wd.k_prime(eps, delta, credited, n, w)
    except ValueError:
        SEEN["k_prime"].add("refused")
        raise
    q = 1 - eps * credited / (n * w)
    SEEN["k_prime"].update(x for x, hit in (("capped", k == n > 1), ("k > 10,000", k > 10_000), ("n > 2^50", n > 1 << 50),
                                            ("exact", 0 < q and k < delta.denominator.bit_length() and q ** k == delta))
                            if hit)
    return k


def _port(fn: str, reference: Callable[[dict], Any], gen: Callable[[random.Random], dict], **kw) -> D.Port:
    return D.Port(EXE, fn, reference=reference, inputs=st.randoms(use_true_random=False).map(gen), package=LEAN,
                  rejects=(ValueError,), **kw)


PORTS = {
    "on_grid": (_port("on_grid", lambda v: Wd.on_grid(v["n"], v["bits"]), grid_n, need_rejects=False), 500),
    "floor_point": (_port("floor_point", lambda v: Wd.floor_point(v["n"], v["bits"]), grid_n), 500),
    "pad": (_port("pad", lambda v: Wd.pad(v["n"], v["bits"]), grid_n), 500),
    "level": (_port("level", lambda v: Wd.level(v["n"], v["bits"]), grid_n), 500),
    "bucket": (_port("bucket", lambda v: asdict(Wd.bucket(v["n"], v["bits"])), grid_n), 500),
    "point": (_port("point", lambda v: Wd.point(v["ell"], v["bits"]), grid_ell), 500),
    "work_at": (_port("work_at", lambda v: asdict(Wd.Work.at(v["ell"], v["bits"])), grid_ell), 500),
    "levels": (_port("levels", lambda v: Wd.levels(v["lo"], v["hi"], v["bits"]), levels_in), 500),
    "k_prime": (_port("k_prime", k_prime, k_prime_in), 3000),
    "sampling": (_port("sampling", lambda v: Wd.sampling(_q(v["eps"]), _q(v["delta"]), _work(v["work"]), v["tile_work"],
                                                         v["bits"]), sampling_in), 1000),
    "work_to_json": (_port("work_to_json", lambda v: _work(v["work"]).to_json(v["tile_work"]), work_to_json_in,
                           need_rejects=False), 300),
    "window_problem": (_port("window_problem", window_problem, window_in, need_rejects=False), 3000),
    "tile_at": (_port("tile_at", lambda v: list(Wd.tile_at(v["entries"], v["t"])), tile_at_in), 1000),
    "tile_in": (_port("tile_in", lambda v: list(Wd.tile_in(v["entry"], v["t"])), tile_in_in), 1000),
    "constants": (D.Port(EXE, "constants", reference=lambda _: {
        "PUBLIC": [list(p) for p in Wd.PUBLIC], "REGISTERED": list(Wd.REGISTERED), "KINDS": list(Wd.KINDS),
        "GRID_BITS": Wd.GRID_BITS}, inputs=st.just({}), package=LEAN, need_rejects=False), 1),
}

server = D.server_fixture(PORTS["constants"][0])


SLOW = {"window_problem"}


@pytest.mark.parametrize("fn", [pytest.param(f, marks=pytest.mark.slow) if f in SLOW else f for f in sorted(PORTS)])
def test_the_lean_is_the_python(fn: str, server) -> None:
    port, examples = PORTS[fn]
    SEEN.get(fn, set()).clear()
    D.run(port, examples=examples, server=server)
    if fn in REACHES:
        assert SEEN[fn] == REACHES[fn]
