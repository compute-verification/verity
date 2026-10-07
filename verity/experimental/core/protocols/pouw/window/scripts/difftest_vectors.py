"""The difftest's Python half: random instances of `verity.experimental.protocols.pouw.window` and its answers on each, as
`PouwWindow.Difftest.evaluate` reads and gives them (`PouwWindow/Difftest.lean` lists the ops).

Instance `seed` is drawn from `random.Random(seed)`. The grid's numbers run over bits 0 … 6, n up to 2^60 and the integers
at or below 0; K′'s parameters over realistic ε and δ (down to 2^-40), grid-derived and random work up to n = 2^60, cases
where (1 − p)^K = δ exactly, and every way out of its domain; the windows are well-formed advice that is then, for most,
mutated to reach each of `window_problem`'s refusals (`MUTATIONS`), with `tile_at` and `tile_in` on the ones it accepts.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from verity.experimental.protocols.pouw import window as Wd

COUNT = 12000


def answer(f: Callable[[], Any], shape: Callable[[Any], Any] = lambda v: v) -> dict:
    try:
        return {"ok": shape(f())}
    except ValueError as e:
        return {"refused": str(e)}


def _work(w: Wd.Work) -> dict:
    return asdict(w)


def _frac(q: Fraction) -> list[int]:
    return [q.numerator, q.denominator]


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


def grid(rng: random.Random) -> tuple[dict, dict]:
    bits, n = rng.randint(0, 6), 0
    n = _number(rng, bits)
    ell = rng.choice((rng.randint(0, 60), rng.randint(0, 2000), -rng.randint(1, 5)))
    inst = {"op": "grid", "bits": bits, "n": n, "ell": ell}
    return inst, {"on_grid": Wd.on_grid(n, bits), "floor_point": answer(lambda: Wd.floor_point(n, bits)),
                  "pad": answer(lambda: Wd.pad(n, bits)), "level": answer(lambda: Wd.level(n, bits)),
                  "point": answer(lambda: Wd.point(ell, bits)), "work_at": answer(lambda: Wd.Work.at(ell, bits), _work),
                  "bucket": answer(lambda: Wd.bucket(n, bits), _work)}


def levels(rng: random.Random) -> tuple[dict, dict]:
    bits = rng.randint(0, 6)
    lo, hi = sorted((_number(rng, bits), _number(rng, bits)))
    if rng.random() < 0.2:
        lo, hi = hi, lo
    return {"op": "levels", "bits": bits, "lo": lo, "hi": hi}, {"levels": answer(lambda: Wd.levels(lo, hi, bits))}


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
        k = rng.randint(1, 30)
        return eps, (1 - p) ** k, credited, n, w
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


def k_prime(rng: random.Random) -> tuple[dict, dict]:
    eps, delta, credited, n, w = _draw_params(rng)
    inst = {"op": "k_prime", "eps": _frac(eps), "delta": _frac(delta), "credited": credited, "n": n, "w_max": w}
    return inst, {"k_prime": answer(lambda: Wd.k_prime(eps, delta, credited, n, w))}


def sampling(rng: random.Random) -> tuple[dict, dict]:
    eps, delta, bits = _eps(rng), _delta(rng), rng.randint(0, 6)
    if rng.random() < 0.15:
        eps = rng.choice((Fraction(0), Fraction(2)))
    work = Wd.Work.at(rng.randint(0, 400), bits) if rng.random() < 0.6 else Wd.bucket(rng.randint(1, 1 << 40), bits)
    tile = rng.choice((1, 4096, rng.randint(1, 1 << 20), 0, -3))
    inst = {"op": "sampling", "eps": _frac(eps), "delta": _frac(delta), "work": _work(work), "tile_work": tile, "bits": bits}
    return inst, {"sampling": answer(lambda: Wd.sampling(eps, delta, work, tile, bits)), "to_json": work.to_json(tile)}


# ---- the advice ---------------------------------------------------------------------------------------------------------

def _side(rng: random.Random, big: bool) -> dict:
    rows = rng.randint(1, 64 if big else 8)
    keep = sorted(rng.sample(range(rows), rng.randint(1, rows)))
    return {"rows": rows, "keep": keep}


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
    return rng.choice((x + 1, x - 1, -1 - x, float(x), str(x), None, [x], x == 1 or x == 0, 0.5))


def _mutate_side(rng: random.Random, side: dict) -> Any:
    rows, keep = side["rows"], side["keep"]
    m = rng.randint(0, 13)
    if m == 0:
        return {**side, "rows": rng.choice((0, -1, float(rows), str(rows), True, None, [rows]))}
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
        return {**side, "keep": [True] + keep[1:]} if keep[0] == 1 else {**side, "keep": [float(keep[0])] + keep[1:]}
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
                        {k: v for k, v in es[i].items() if k != "kind"}, rng.choice((5, "call", [], None, 1.5))))
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


def window(rng: random.Random) -> tuple[dict, dict]:
    es, work, bound, bits = window_advice(rng)
    for _ in range(rng.choice((0, 0, 1, 1, 1, 1, 2))):
        try:
            es, work, bound = rng.choice(MUTATIONS)(rng, json.loads(json.dumps(es)), work, bound)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            pass  # a second mutation that the first left nothing to act on
    check_bits = bits if rng.random() < 0.95 else rng.randint(0, 6)
    problem = Wd.window_problem(es, work, bound, check_bits)
    draws, within = [], []
    if problem is None:
        draws = [rng.randrange(work.n) for _ in range(5)] + [work.n + rng.randint(0, 3)]
        within = [[rng.randrange(bound), rng.randrange(work.n + 2)] for _ in range(4)]
    inst = {"op": "window", "entries": es, "work": _work(work), "bound": bound, "bits": check_bits, "draws": draws,
            "within": within}
    return inst, {"problem": problem,
                  "tile_at": [answer(lambda t=t: Wd.tile_at(es, t), list) for t in draws],
                  "tile_in": [answer(lambda e=e, t=t: Wd.tile_in(es[e], t), list) for e, t in within]}


def constants() -> tuple[dict, dict]:
    return {"op": "constants"}, {"PUBLIC": [list(p) for p in Wd.PUBLIC], "REGISTERED": list(Wd.REGISTERED),
                                 "KINDS": list(Wd.KINDS), "GRID_BITS": Wd.GRID_BITS}


OPS = ((grid, 0.25), (levels, 0.08), (k_prime, 0.27), (sampling, 0.1), (window, 0.30))


def instance(seed: int) -> tuple[dict, dict]:
    """Instance `seed` and the Python's answer on it."""
    if seed == 0:
        return constants()
    rng = random.Random(seed)
    r, acc = rng.random(), 0.0
    for op, weight in OPS:
        acc += weight
        if r < acc:
            return op(rng)
    return window(rng)


def line(obj: dict) -> str:
    """One JSON line. A float is written with its fraction (`3.0`), never in exponent form, which Lean's reader would take
    for an integer."""
    if any(isinstance(x, float) and not (x == 0 or 1e-4 <= abs(x) < 1e16) for x in _leaves(obj)):
        raise ValueError("a float json writes in exponent form")
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def _leaves(x: Any):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst, ans = instance(seed)
            fi.write(line(inst))
            fp.write(line(ans))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.experimental.protocols.pouw.window is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(Wd.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.experimental.protocols.pouw.window comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
