"""SPOT's caller's check (`VStar.Recursive.Caller.check`): what the caller hands SPOT's verifier beside the docket
and SPOT's own log, which SPOT's check (`docket.check`) reads none of.

The record names its day's one declared program (Declared day: `R2-undeclared`, `R2-declared-twice`, `R2-declared`);
its window names the stream and the prover the verifier is bound to (Binding: `R6-stream`, `R6-prover`), the seed salt
the verifier read from the challenger's log (Seed salt: `R6-seed-salt`), the pool's reset, week and spent mark, at most
one reset a week (Pool chain: `R6-pool-reset`, `R6-pool-week`, `R6-pool-spent`, `R6-pool-reset-week`), and the
registration log's compiler and advice within its length (Registered compiler: `R6-compiler`, `R6-advice-length`); and,
when the verifier expects values registered, the record's entry for the pool's `pin` is the pool its window opens
(`R7-pool-chain`), and no output value reads a port of a record the verifier's verdict log names refused
(Verdicts: `R7-output-refused`). The sections named are `docket`'s.

`check` takes `docket.check`'s arguments and reads them as it does (ValueError where it raises), and returns the
caller's codes in `Caller.check`'s order, whatever SPOT's check made of the record. `gate` is `Caller.gate`: SPOT's
verdict where it refuses, else the caller's. The verifier (`verifier.check`) runs it with the Lean of record
(`spot-flock-caller`).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from verity.circuits.program import Program

from . import docket as R, registered as G
from .verdict import Verdict, _Collect


def _declared(c: _Collect, program: Any, day: int, declarations: Sequence[Mapping[str, Any]]) -> None:
    """The declared day (`Caller.declaredCode`): the window's day has exactly one declaration, and the record's program is
    its program."""
    mine = [d for d in declarations if d["day"] == day]
    if not mine:
        c.fail("R2-undeclared", {"day": day})
    elif len(mine) > 1:
        c.fail("R2-declared-twice", {"day": day, "programs": [d["program"] for d in mine]})
    elif program != mine[0]["program"]:
        c.fail("R2-declared", {"day": day, "registered": program, "declared": mine[0]["program"]})


def _compiler(c: _Collect, w: Mapping[str, Any], reads: tuple[str, int] | None) -> None:
    """The registered compiler (`Caller.compilerMarks`): the window names the log's compiler by its digest
    (`R6-compiler`) and advice of at most its length ℓ in bits (`R6-advice-length`), and names neither where the log
    registers none (`R6-compiler`)."""
    named, advice = w.get("compiler"), w.get("advice")
    if reads is None:
        if named is not None or advice is not None:
            c.fail("R6-compiler", {"registered": {"compiler": named, "advice": advice}, "verifier": None})
        return
    digest, cap_bits = reads
    if not (isinstance(named, Mapping) and named.get("digest") == digest):
        c.fail("R6-compiler", {"registered": named, "verifier": digest})
    bits = advice.get("bits") if isinstance(advice, Mapping) else None
    if not (type(bits) is int and 0 <= bits <= cap_bits):
        c.fail("R6-advice-length", {"registered": bits, "verifier": cap_bits})


def _pool_chain(c: _Collect, rec: Mapping[str, Any], pool: Mapping[str, Any], want: Any) -> None:
    """The pool chain (`Caller.poolChained`): the record's entry for `pin` (its last naming it) has the domain and root of
    the pool its window opens (`want()`, `docket.pool_open`, or why there is none), else `R7-pool-chain`."""
    got = rec.get("registered")
    hits = [e for e in got if isinstance(e, Mapping) and e.get("value") == pool["pin"]] if isinstance(got, list) else []
    mine = (hits[-1].get("domain"), hits[-1].get("root")) if hits else None
    try:
        opens = want() if hits else None
    except (KeyError, TypeError, ValueError) as e:
        opens = f"the record opens no pool: {e}"
    if mine is None or mine != opens:
        c.fail("R7-pool-chain", {"value": pool["pin"], "registered": mine, "verifier": opens})


PORTS_REFUSED_WHY = ("an output value reads a port of an earlier record whose run the verifier refused, so nothing the "
                     "verifier accepted computed it")


def _ports_refused(c: _Collect, outs: Mapping[str, tuple[str, str, str]], verdicts: Mapping[str, bool]) -> None:
    """The verdicts (`Docket.portsAccepted`): no output value reads a port of an earlier record the verdict log
    names refused (`docket.refused`), else
    one `R7-output-refused` naming those values and records. A reused registered value (`value`) is not held to it."""
    bad = sorted(v for v, (d, kind, _) in outs.items() if kind == "port" and R.refused(verdicts, d))
    if bad:
        c.fail("R7-output-refused", {"values": bad, "records": sorted({outs[v][0] for v in bad}),
                                     "why": PORTS_REFUSED_WHY})


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None,
          supplied: Mapping[str, Any] | None = None, pinned: Mapping[str, str] | None = None,
          binding: Mapping[str, str] | None = None,
          outputs: Mapping[str, Mapping[str, str]] | None = None,
          schemas: Mapping[str, str] | None = None, window_root: str | None = None,
          held: Sequence[str] | None = None, members: Sequence[Any] = (), day: int | None = None,
          declarations: Sequence[Mapping[str, Any]] | None = None, seed_salt: str | None = None,
          pool: Mapping[str, Any] | None = None, tied: Sequence[str] | None = None,
          history: Mapping[str, Any] | None = None, compiler: Any = R.UNREAD,
          certified: Mapping[str, Mapping[str, Any]] | None = None,
          verdicts: Mapping[str, bool] | None = None) -> Verdict:
    """The caller's codes for `rec` against `docket.check`'s arguments (`Caller.check`). The arguments it doesn't read
    are `docket.check`'s, so a caller hands both the same."""
    c = _Collect()
    days = R._declarations(day, declarations)
    log = list(log)
    pool = R._pool(pool, registered or {}, supplied, pinned, outputs, schemas)
    supplied, outputs, schemas = R.pool_values(pool, program, supplied, outputs, schemas, log)
    supplied = R.with_history(supplied, history, leaf_layer)
    sup = G.sources(supplied if supplied is not None else {})
    v2 = bool(registered) or bool(R._certified(certified, registered or {}, sup))
    pins = R._pins(pinned, registered or {}, sup)
    outs = R._outputs(outputs, registered or {}, sup, pins, log)
    formats = R._schemas(schemas, registered or {})
    bound = R._binding(binding)
    window, fixed = R._window(window_root, held, registered or {}, sup, pins, outs)
    salt = R._seed_salt(seed_salt, window)
    reads = R._registered_compiler(compiler)
    verdicts = R._verdicts(verdicts) if verdicts is not None else {}
    rec_ = rec if isinstance(rec, Mapping) else {}
    w = rec_.get("window") if isinstance(rec_.get("window"), Mapping) else {}
    if days is not None:
        _declared(c, rec_.get("program"), *days)
    if bound is not None:
        for k in ("stream", "prover"):
            if w.get(k) != bound[k]:
                c.fail(f"R6-{k}", {"registered": w.get(k), "verifier": bound[k]})
    if w.get(R.SEED_SALT) != salt:
        c.fail("R6-seed-salt", {"registered": w.get(R.SEED_SALT), "verifier": salt})
    reset = pool["reset"] if pool is not None else None
    if w.get(R.POOL_RESET) is not reset:
        c.fail("R6-pool-reset", {"registered": w.get(R.POOL_RESET), "verifier": reset})
    week = pool["week"] if pool is not None else None
    if not R._names(w.get(R.POOL_WEEK), week):
        c.fail("R6-pool-week", {"registered": w.get(R.POOL_WEEK), "verifier": week})
    before = pool is not None and R.pool_spent_before(pool, log)
    spent = (pool["reset"] or before) if pool is not None else None
    if w.get(R.POOL_SPENT) is not spent:
        c.fail("R6-pool-spent", {"registered": w.get(R.POOL_SPENT), "verifier": spent})
    if pool is not None and pool["reset"] and before:
        c.fail("R6-pool-reset-week", {"registered": True, "verifier": "the week's reset is spent before this window"})
    if reads is not R.UNREAD:
        _compiler(c, w, reads)
    if v2 and pool is not None:
        _pool_chain(c, rec_, pool, lambda: R.pool_open(pool, rec_, program, registered, formats, window, fixed, sup, log,
                                                       bound))
    if v2:
        _ports_refused(c, outs, verdicts)
    return c.verdict()


def both(*args: Any, **kw: Any) -> Verdict:
    """SPOT's check (`docket.check`) and the caller's on the same arguments, every code of both, SPOT's first: the
    verdict that refuses what either refuses, where `gate` stops at SPOT's refusal."""
    spot, caller = R.check(*args, **kw), check(*args, **kw)
    return Verdict(spot.ok and caller.ok, spot.codes + tuple(c for c in caller.codes if c not in spot.detail),
                   {**caller.detail, **spot.detail}, fanout=spot.fanout,
                   rests_on_period_check=spot.rests_on_period_check)


def gate(spot: Verdict, caller: Verdict) -> Verdict:
    """`Caller.gate`: SPOT's verdict `spot` where it refuses or the caller's verdict `caller` finds nothing, else the
    caller's, with SPOT's fan-out and tied values."""
    if not spot.ok or caller.ok:
        return spot
    return Verdict(caller.ok, caller.codes, caller.detail, caller.verifiers, spot.fanout, spot.rests_on_period_check)
