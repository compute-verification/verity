"""The window's Lean against `verity.core.protocols.pouw.window`, on the shared harness (`verity_difftest`): one
`Port` per ported function of `window.py`, served by `pouw-window-difftest` (`PouwWindow/Difftest.lean`).

The inputs are drawn by the generators below from a Hypothesis-controlled `random.Random` (`st.randoms`), so a run is
deterministic and shrinks. The grid's numbers run over bits 0 … 6, n up to 2^60 and the integers at or below 0; K′'s
parameters over realistic ε and δ (down to 2^-40), grid-derived and random work up to n = 2^60, cases where (1 − p)^K = δ
exactly, and every way out of its domain. The window statement on the advice is not ported yet: `window.py` now
reads #1426's fixed-width entry rows (`Entry.from_row`, `layout_problem`, `record_problem`, `tile_read`, the side trees
and `public_inputs`), and their port is the follow-up.
"""
from __future__ import annotations

import random
import shutil
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core.protocols.pouw import window as Wd

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


# ---- the ports ----------------------------------------------------------------------------------------------------------

SEEN: dict[str, set[str]] = {"k_prime": set()}
#: What each port's run must reach, so that none is vacuous for one of its checks or regimes.
REACHES = {"k_prime": {"refused", "capped", "k > 10,000", "n > 2^50", "exact"}}


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
    "constants": (D.Port(EXE, "constants", reference=lambda _: {
        "PUBLIC": [list(p) for p in Wd.PUBLIC], "REGISTERED": list(Wd.REGISTERED), "KINDS": list(Wd.KINDS),
        "GRID_BITS": Wd.GRID_BITS}, inputs=st.just({}), package=LEAN, need_rejects=False), 1),
}

server = D.server_fixture(PORTS["constants"][0])


@pytest.mark.parametrize("fn", sorted(PORTS))
def test_the_lean_is_the_python(fn: str, server) -> None:
    port, examples = PORTS[fn]
    SEEN.get(fn, set()).clear()
    D.run(port, examples=examples, server=server)
    if fn in REACHES:
        assert SEEN[fn] == REACHES[fn]
