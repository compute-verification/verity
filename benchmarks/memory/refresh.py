"""PoUB's refresh rule replayed over a recorded memory map: how much written data it credits, the re-key work it costs,
and the credit beside today's rule's on the same map.

The map is memory-map/v1 and the replay is its one replay, `memory_map.Replay`: `replay` applies each step as today and,
with a `Refresh` setting, also notes each block's resident interval. With the setting off (`refresh=None`, the default)
it returns `Replay.apply`'s changes and nothing more, byte for byte today's.

~~~text
time      a map has steps, not seconds: step s ends at t(s) = (s − s₀ + 1)·step_s, s₀ the map's first step; a block
          written at s is resident from t(s), one freed or overwritten at s' until t(s'); a registered block from 0;
          a block still resident at the map's end is censored there, T = t(last step)
window    w(t) = ⌊t / window_s⌋; the refresh period is P = L·window_s, L the session's `lifetime` in windows
today     every resident block is credited (registered blocks, written blocks from their write)
refresh   rule 1: every credited block is re-keyed once a period (uniform rota: each resident block once in each period,
          a full encoding of 4,096 bytes); rule 2: a written block of birth window b is credited from (b + L)·window_s,
          the window `ROTA` gives, a registered block while resident
(a)       written share = Σ max(0, end − max(born, (b + L)·window_s)) / Σ (end − born), by byte-seconds; by bytes, the
          share of written blocks with end > (b + L)·window_s
(b)       work a period = mean resident bytes, all blocks (W_all), or those that reach credit (W_credit); the share of the
          card's encode rate R it takes is s = W / (P·R), and the period a share s affords is L = ⌈W_all / (s·R·window_s)⌉
(c)       credited bytes a pass, mean over the run, each pass reading every credited block once
~~~

With every written block living D seconds and the store S re-keyed whole, (a) is f = 1 − (P − u)/D for P ≤ D, u the
block's birth within its window (0 ≤ u < window_s), and 0 past it: tonight's f = max(0, 1 − P/L) with P = S/(s·R), to
within one window, P/L being P over the data's life D.  Rule 2 counts from the window's start, so the replay's figure is
at most window_s/D above that formula's.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from verity.protocols.resource_accounting.memory.poub.verifier import codec, memory_map

BLOCK = codec.BLOCK_BYTES
P512_K2 = codec.Codec(codec.P512, 2, False)
#: Squarings a second of one RTX PRO 6000 at each width, measured: `kv.cu`'s FOLD 6 at 1,024 bits and codec-mix's
#: reference at 512 (`poub/verifier/timing.py`, `GPU_SQUARINGS_PER_S`'s docstring), and FOLD 5 at 512
#: (`calibration/rate_squaring/gamma.py`'s `GPU`, the best measured form).
RATES = {"deployed@2.044e10": (codec.DEFAULT, 2.044e10), "p512@5.13e10": (P512_K2, 5.13e10),
         "p512@6.531e10": (P512_K2, 6.531e10)}
SHARES = (0.05, 0.10, 0.20)


def uniform_rota(born_window: int, lifetime: int) -> int:
    """The first window a written chunk of birth window `born_window` is credited in, under the uniform rota: its first
    rota re-key lands within the L windows after its birth, and probation ends at born + L, so credit starts there."""
    return born_window + lifetime


#: The one place the replay reads the rota from: the first credited window of a written chunk, as a function of its
#: birth window and L.  The slot schedule (`poub/verifier/rota.py`) starts credit at the same window.
ROTA = uniform_rota


@dataclass(frozen=True, slots=True)
class Refresh:
    """The refresh rule's setting: the period, `lifetime` windows of `window_s` seconds, and `step_s`, the seconds a
    serving step of the map takes (a map records steps, not times)."""

    lifetime: int
    window_s: float
    step_s: float

    def __post_init__(self) -> None:
        if type(self.lifetime) is not int or self.lifetime < 1:
            raise ValueError(f"the refresh period is a positive number of windows, not {self.lifetime!r}")
        if not self.window_s > 0 or not self.step_s > 0:
            raise ValueError("a window and a step each take a positive number of seconds")


@dataclass(frozen=True, slots=True)
class Lives:
    """Each block's resident interval in seconds: the registered blocks' (from 0, `registered_ends` for those released,
    the rest to `end`) and each written block's `Segment`."""

    blocks: int
    end: float
    registered_ends: tuple[float, ...]
    written: tuple[Segment, ...]


@dataclass(frozen=True, slots=True)
class Segment:
    """A written block resident from `start` to `end`, its payload first written at `born`: its own start, or for a
    block whose digest a block released before it had (today's re-key, `memory_map`'s Lifetime), that block's birth.
    `censored` if still resident at the map's end; `last` unless a later block re-keys its payload."""

    born: float
    start: float
    end: float
    censored: bool
    last: bool


def replay(head: memory_map.Header, steps: Iterable[memory_map.Step],
           refresh: Refresh | None = None) -> tuple[list[memory_map.Change], Lives | None]:
    """The map's changes, each step applied by `memory_map.Replay` as today, and with `refresh`, each block's life."""
    r = memory_map.Replay(head)
    changes: list[memory_map.Change] = []
    live: dict[int, tuple[float, float, bytes]] = {}
    released: dict[bytes, tuple[float, int]] = {}
    reg_ends: list[float] = []
    written: list[list] = []
    first = last = None
    for s in steps:
        c = r.apply(s)
        changes.append(c)
        if refresh is None:
            continue
        first = s.step if first is None else first
        last = s.step
        t = (s.step - first + 1) * refresh.step_s
        gone = {j for f, n in c.withdrawn for j in range(f, f + n)}
        for f, n in c.released:
            for j in range(f, f + n):
                if j in gone:
                    continue
                if j < head.blocks:
                    reg_ends.append(t)
                else:
                    b, start, d = live.pop(j)
                    released[d] = (b, len(written))
                    written.append([b, start, t, False, True])
        for j, d in zip(c.indices, c.digests):
            b = t
            if d in released:
                b, k = released.pop(d)
                written[k][4] = False
            live[j] = (b, t, d)
    if refresh is None:
        return changes, None
    end = 0.0 if first is None else (last - first + 1) * refresh.step_s
    written += [[b, start, end, True, True] for b, start, _ in live.values()]
    return changes, Lives(head.blocks, end, tuple(reg_ends), tuple(Segment(*w) for w in written))


def _quantile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, math.ceil(q * len(ys)) - 1)]


def credit(lives: Lives, refresh: Refresh) -> dict:
    """(a) and (c) at the setting's period, and (b)'s work a period, from a map's lives."""
    period = refresh.lifetime * refresh.window_s
    reg_bs = BLOCK * (sum(lives.registered_ends) + (lives.blocks - len(lives.registered_ends)) * lives.end)
    total = credited = credit_bs = 0.0
    n = n_credited = 0
    for g in lives.written:
        start = max(g.start, ROTA(math.floor(g.born / refresh.window_s), refresh.lifetime) * refresh.window_s)
        total += g.end - g.start
        if g.end > start:
            credited += g.end - start
            credit_bs += g.end - g.start
        if g.last:
            n += 1
            n_credited += g.end > start
    t = lives.end or 1.0
    return {
        "lifetime_windows": refresh.lifetime, "period_s": period,
        "written_share_bytes": n_credited / n if n else None,
        "written_share_byte_seconds": credited / total if total else None,
        "work_all_bytes": (reg_bs + BLOCK * total) / t, "work_credit_bytes": (reg_bs + BLOCK * credit_bs) / t,
        "credited_bytes_today": (reg_bs + BLOCK * total) / t, "credited_bytes_refresh": (reg_bs + BLOCK * credited) / t,
    }


def appraise(lives: Lives, refresh: Refresh, rates: dict[str, tuple[codec.Codec, float]],
             shares: Sequence[float] = SHARES, store_bytes: float | None = None) -> dict:
    """The replay's figures for one map at one step time: lives, (a) to (c) at the setting's period, and for each rate
    the share (b) that period needs and, for each share, the period it affords and (a) and (c) there.  The work a
    period is the map's mean resident bytes, or `store_bytes` (a whole card's store, re-keyed once a period)."""
    lives_s = [g.end - g.born for g in lives.written if g.last]
    out = {"blocks": lives.blocks, "seconds": lives.end, "step_s": refresh.step_s, "window_s": refresh.window_s,
           "written": len(lives_s), "censored": sum(g.censored for g in lives.written if g.last),
           "rekeyed": sum(not g.last for g in lives.written),
           "life_median_s": _quantile(lives_s, 0.5), "life_p90_s": _quantile(lives_s, 0.9),
           "at_period": credit(lives, refresh), "store_bytes": store_bytes, "rates": {}}
    work = store_bytes or out["at_period"]["work_all_bytes"]
    for name, (cod, squarings) in rates.items():
        rate = squarings * BLOCK / cod.encode_squarings
        row = {"codec": cod.name, "squarings_per_s": squarings, "encode_bytes_per_s": rate,
               "share_needed": work / (out["at_period"]["period_s"] * rate), "shares": {}}
        for s in shares:
            life = max(1, math.ceil(work / (s * rate * refresh.window_s)))
            row["shares"][f"{s:g}"] = credit(lives, Refresh(life, refresh.window_s, refresh.step_s))
        out["rates"][name] = row
    return out


FORMULA = ("refresh: a written block born in window b is credited from (b + L)·window_s; every resident block is "
           "re-keyed once a period P = L·window_s (uniform rota); share needed s = W/(P·R), W the mean resident bytes; "
           "a share s affords L = ceil(W/(s·R·window_s)); with every block living D and W = S this is "
           "f = max(0, 1 - P/D), P = S/(s·R), to within one window")


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def lines(name: str, a: dict) -> list[str]:
    p = a["at_period"]
    store = "the map's mean resident bytes" if a["store_bytes"] is None else f"{a['store_bytes']:.0f} B"
    out = [f"{name}: {a['blocks']} registered blocks, {a['written']} payloads written ({a['censored']} still resident at "
           f"the end, {a['rekeyed']} re-keyed by today's rule), {a['seconds']:.3f} s at {a['step_s']} s a step; payload "
           f"life median {a['life_median_s']} s, p90 {a['life_p90_s']} s; re-keyed a period: {store}",
           f"  at L = {p['lifetime_windows']} windows of {a['window_s']} s: written credited {_pct(p['written_share_bytes'])} "
           f"by bytes, {_pct(p['written_share_byte_seconds'])} by byte-seconds; credited bytes a pass today "
           f"{p['credited_bytes_today']:.0f}, under refresh {p['credited_bytes_refresh']:.0f}; work a period "
           f"{p['work_all_bytes']:.0f} B (all) / {p['work_credit_bytes']:.0f} B (credited)"]
    for rname, r in a["rates"].items():
        cells = "; ".join(f"{float(s):.0%}: L {c['lifetime_windows']}, written {_pct(c['written_share_byte_seconds'])}, "
                          f"credited {c['credited_bytes_refresh']:.0f} B" for s, c in r["shares"].items())
        out.append(f"  {rname} ({r['codec']}, R {r['encode_bytes_per_s'] / 1e9:.3f} GB/s): share needed at L "
                   f"{p['lifetime_windows']} {r['share_needed']:.2e}; {cells}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+", type=Path, help="memory-map/v1 files")
    ap.add_argument("--lifetime", type=int, default=2, help="the refresh period in windows (default 2)")
    ap.add_argument("--window-s", type=float, default=30.0, help="a window's seconds (default 30)")
    ap.add_argument("--step-s", type=float, action="append", required=True, help="a step's seconds; repeatable")
    ap.add_argument("--store-bytes", type=float, action="append", default=[],
                    help="also take the work a period as this store, re-keyed whole (a card's); repeatable")
    ap.add_argument("--rate", action="append", default=[], metavar="NAME=CODEC:SQUARINGS",
                    help="price at this rate instead of RATES: CODEC deployed or p512, squarings a second; repeatable")
    ap.add_argument("--json", type=Path, help="write every figure here as JSON")
    args = ap.parse_args(argv)
    codecs = {"deployed": codec.DEFAULT, "p512": P512_K2}
    rates = dict(RATES)
    if args.rate:
        rates = {}
        for spec in args.rate:
            name, _, rest = spec.partition("=")
            cname, _, squarings = rest.partition(":")
            if not name or cname not in codecs or not squarings:
                ap.error(f"--rate is NAME=CODEC:SQUARINGS with CODEC one of {sorted(codecs)}, not {spec!r}")
            rates[name] = (codecs[cname], float(squarings))
    module = "benchmarks.memory.refresh" if ROTA.__module__ == "__main__" else ROTA.__module__
    rota = f"{module}.{ROTA.__qualname__}"
    print(f"{FORMULA}; rota: {rota}")
    doc = {"formula": FORMULA, "rota": rota, "maps": {}}
    for path in args.maps:
        with path.open() as f:
            head, steps = memory_map.parse(f)
            steps = list(steps)
        for step_s in args.step_s:
            setting = Refresh(args.lifetime, args.window_s, step_s)
            _, lives = replay(head, steps, setting)
            for store in [None, *args.store_bytes]:
                a = appraise(lives, setting, rates, store_bytes=store)
                doc["maps"].setdefault(str(path), {})[f"{step_s:g}/{store or 'map'}"] = a
                print("\n".join(lines(str(path), a)))
    if args.json:
        args.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
