"""`CacheStep_v1` (`verity.core.protocols.certifier.cache_step`) against the derived rules' prefix rules
(`derived.Prefixes`) over a trace's rows, and its difftest vectors.

`drive` runs the cache as the replay's composition calls it. It takes the rows in the advice's order, which must be the
scheduling order (`row_order` "scheduling"; a corpus trace loads so with `row_order="scheduling"`). An admitting row makes two calls: the lookup, then the units
the row computes. Any other row makes one call, from its request's walk entry. Beside it, `drive` runs `Prefixes` on
the same rows and holds the two to each other:
- an admitting row's hit is accepted exactly at `Prefixes.cached` while the walk derives it, and one unit either side
  is refused; past the bound, any cached prefix is accepted and the next longer one refused;
- at the step's end, each hit's sources (its trail's owners) are `Prefixes.sources`, and the advice's own;
- the lookup and the computed units in one call give the same table as the two calls.

    python cache_vectors.py --trace tests/fixtures/corpus-smallkv --trace tests/fixtures/sync-crush.json.gz \\
        --out ../../../verity/core/protocols/certifier/tests/fixtures/cache_step_vectors.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from verity.core.protocols.certifier import cache_step as CS

HERE = Path(__file__).parent


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("enc_derived", HERE / "derived.py")


@dataclass(frozen=True)
class Call:
    """One call as the composition makes it: the step, the table before, the row and what the cache gives."""
    step: int
    table: tuple[CS.Entry, ...]
    row: CS.Row
    out: CS.Out


def window(tokens: Sequence[int], u0: int, S) -> tuple[tuple[int, ...], ...]:
    """KU units of `tokens` from unit u0, zero past their end."""
    KU = CS.widths(S).KU
    ts = list(tokens[u0 * S.U:(u0 + KU) * S.U])
    ts += [0] * (KU * S.U - len(ts))
    return tuple(tuple(ts[k * S.U:(k + 1) * S.U]) for k in range(KU))


def first_sent(adv, cfg, held, order) -> int:
    """The first step whose hits the walk sends rather than derives (past the bound, or none derived at all)."""
    res = D.residual(adv, cfg, held, order=order)
    if cfg.scheduler != "sync" or cfg.row_order == "explicit":
        return 0
    sent = D.reconstruct(res, cfg, held).sent_hits
    return min(sent) if sent else len(adv.steps)


def statics_for(adv, tokens, cfg, N: int) -> dict:
    return CS.statics(N, cfg.unit, max(max(max(t) for t in tokens).bit_length(), 1),
                      max((len(adv.requests) - 1).bit_length(), 1), cfg.max_model_len)


def drive(adv, tokens: Sequence[Sequence[int]], cfg, sent_from: int, st: dict, *, steps: int | None = None,
          record: bool = False) -> dict[str, Any]:
    """The cache over the advice's first `steps` steps, held to `Prefixes` (the module's docstring); the calls if
    `record`. Raises AssertionError where the two disagree."""
    S = SimpleNamespace(**st)
    U = cfg.unit
    if S.U != U:
        raise ValueError(f"the statics' unit {S.U} isn't the config's {U}")
    P = D.Prefixes(tokens, U)
    table: tuple[CS.Entry, ...] = ()
    walk: dict[int, int] = {}
    seen: set[int] = set()
    samples: dict[int, int] = {}
    calls: list[Call] = []
    counts = {"calls": 0, "admitting": 0, "derived": 0, "sent": 0, "hit_units": 0, "refused": 0, "steps": 0}

    def call(s: int, row: CS.Row) -> CS.Out:
        nonlocal table
        out = CS.reference(S, table, row)
        if record:
            calls.append(Call(s, table, row, out))
        counts["calls"] += 1
        return out

    def refused(s: int, row: CS.Row) -> None:
        out = CS.reference(S, table, row)
        assert not out.ok, row
        if record:
            calls.append(Call(s, table, row, out))
        counts["refused"] += 1

    for s, st_ in enumerate(adv.steps[:steps]):
        sent = s >= sent_from
        hits = []
        for r in st_.rows:
            i = r.request
            K = adv.requests[i].prompt_len + samples.get(i, 0)
            end = r.start + r.n
            if i not in seen or r.preempted:
                counts["admitting"] += 1
                counts["sent" if sent else "derived"] += 1
                longest = P.cached(i, K) if cfg.enable_prefix_caching else 0
                if r.cache_hit % U or r.start != r.cache_hit:
                    raise ValueError(f"step {s}: request {i}'s hit {r.cache_hit} isn't its start in whole units")
                wh = r.cache_hit // U
                assert (wh == longest // U) if not sent else (wh <= longest // U), (s, i, wh, longest)
                look = CS.Row(r=i, units=window(tokens[i], 0, S), admit=True, sent=sent, K=K, wh=wh)
                for other in (wh - 1, wh + 1) if not sent else (longest // U + 1,):
                    if 0 <= other <= CS.widths(S).KU:
                        refused(s, replace(look, wh=other))
                got = call(s, look)
                assert got.ok, (s, i, wh)
                both = CS.reference(S, table, replace(look, nins=end // U))
                table = got.table
                rest = call(s, CS.Row(r=i, units=window(tokens[i], wh, S), pfx=got.pfx, nins=end // U - wh))
                assert rest.ok and both.ok and (both.table, both.pfx) == (rest.table, rest.pfx), (s, i)
                hits.append((i, wh, got.trail, r.sources))
                counts["hit_units"] += wh
            else:
                u0 = r.start // U
                rest = call(s, CS.Row(r=i, units=window(tokens[i], u0, S), pfx=walk[i], nins=end // U - u0))
                assert rest.ok, (s, i)
            table, walk[i] = rest.table, rest.pfx
            P.add(i, r.start, end)
            seen.add(i)
            samples[i] = samples.get(i, 0) + (end >= K)
        for i, wh, trail, theirs in hits:
            mine = CS.sources(S, table, trail, wh)
            assert mine == tuple((x.request, x.n) for x in P.sources(i, wh * U)), (s, i)
            assert mine == tuple((x.request, x.n) for x in theirs), (s, i)
        counts["steps"] += 1
    counts["entries"] = len(table)
    return {"counts": counts, "calls": calls}


def load(corpus: Path) -> dict:
    """A trace-corpus trace (Llama 3's stop ids) with its rows in the scheduler's order."""
    return D.load_corpus_trace(corpus, eos_token_id=128009, stop_token_ids=(128001, 128008), row_order="scheduling")


def vectors(adv, tokens, cfg, sent_from: int, st: dict, steps: int) -> list[Call]:
    """The calls of the first `steps` steps, up to the first whose table outgrows the statics' N."""
    big, N = 1 << 20, st["N"]
    calls = drive(adv, tokens, cfg, sent_from, {**st, "N": big}, steps=steps, record=True)["calls"]

    def root(e: int) -> int:
        return N if e == big else e

    def table(t: Sequence[CS.Entry]) -> tuple[CS.Entry, ...]:
        return tuple(replace(x, parent=root(x.parent)) for x in t)

    out = []
    for c in calls:
        if len(c.table) > N or len(c.out.table) > N:
            break
        row = replace(c.row, pfx=root(c.row.pfx))
        mine = CS.reference(SimpleNamespace(**st), table(c.table), row)
        assert mine == (replace(c.out, table=table(c.out.table), pfx=root(c.out.pfx), trail=tuple(map(root, c.out.trail)))
                        if c.out.ok else c.out)
        out.append(Call(c.step, table(c.table), row, mine))
    return out


def slice_of(tr: dict, N: int, steps: int) -> dict:
    """A trace's first `steps` steps as difftest vectors at N entries, in the smallest L (whole units) that holds the
    requests they reach: each request's tokens once, and each call's row and what the cache gives (its trail up to the
    last unit it touches)."""
    cfg = D.config_of(tr)
    adv, order = D.advice_of(tr)
    held = D.holdings_of(tr)
    tokens = D.tokens_of(held, order, [q.prompt_len for q in adv.requests])
    sent_from = first_sent(adv, cfg, held, order)
    st = statics_for(adv, tokens, cfg, N)
    reached = {c.row.r for c in vectors(adv, tokens, cfg, sent_from, st, steps)}
    st["L"] = -(-max(len(tokens[r]) for r in reached) // cfg.unit) * cfg.unit
    calls = vectors(adv, tokens, cfg, sent_from, st, steps)
    rows = []
    for c in calls:
        r, o = c.row, c.out
        u0 = 0 if r.admit else next(u for u in range(len(tokens[r.r]) // cfg.unit + 1)
                                     if window(tokens[r.r], u, SimpleNamespace(**st)) == r.units)
        touched = max(r.wh, r.nins, 1)
        rows.append([r.r, u0, int(r.admit), int(r.sent), r.K, r.wh, r.nins, r.pfx, int(o.ok), o.pfx,
                     list(o.trail[:touched]), len(o.table)])
    return {"statics": st, "steps": steps, "tokens": {str(r): list(tokens[r]) for r in sorted({c.row.r for c in calls})},
            "calls": rows}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, action="append", required=True,
                    help="a trace-corpus directory or a sched_trace JSON, each a slice")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    slices = {p.name: slice_of(load(p) if p.is_dir() else D.load_trace(p), a.N, a.steps) for p in a.trace}
    a.out.write_text(json.dumps({"kind": "cache-step-vectors/v1", "slices": slices}, separators=(",", ":")) + "\n")
    print(json.dumps({k: {"statics": v["statics"], "calls": len(v["calls"]), "refused": sum(not c[8] for c in v["calls"])}
                      for k, v in slices.items()}))


if __name__ == "__main__":
    main()
