"""PoUB over a monitoring period: a week, a month and a year of every machine of a
deployment, at δ = 1% and 10⁻⁶, from one accepted session's answer as `RemoteAudit.Protocol.certify` publishes it and
that session's error in its verifier's ledger (`ledger.Gamma`'s `error`): `q_bytes` certified as read, except with the
session's whole error p.

The whole error is the ledger's for the session's openings read pooled (its `horizon`): the openings' pooled bound's δ₀
(2^-40) with AuditRO's slack a pass, the sample's terms (`draw`: the rejection ⌈W/g⌉·s/2^64, the session's openings
coins at its `samples` s each, and the receipts' binding term) and the sketch's term δs at the node's outside-compute
factor (`sketch`: `timing.sketch_failure` at `ledger.Node.sketch_outside`; 0 for a receipt that is no sketch). The
answer's own `eps` is δ₀ and the rejection alone, the error of "U − W receipts are correct" (`LineHeldAtComponent`'s);
a correct receipt is a read only on the sketch bound's event, so a table that prints `q_bytes` as reads carries the
whole error. An error below the answer's `eps` is not that session's, and `table` refuses it.

A deployment of `machines` machines, each running one session every `window_s` seconds audited as this one was, has
T = machines·⌊period/window_s⌋ sessions in a period (the core's `block_count`), each on its own live coins, against one
adaptive prover. A period's row covers those T sessions, so its error is their whole errors summed, T·p
(`reporting.summed`), and the row certifies T·`q_bytes` except with it. Where the sum is 1 or more the row prints
`reporting.NO_BOUND` with the sum and no amount, neither that total nor a figure below (`reporting.no_bound`). Under 1
it also prints, at each δ:

- **Pooled** (`pooled_step`): the count bound `FlockSoundness.Audit.Periods.prCoin_count_le_choose`. When each
  session, given the earlier ones, is accepted and short of its `q_bytes` with probability at most p, at least M
  of the T are with probability at most C(T, M)·p^M. M is the least count within δ (`short_sessions`), so except with δ
  at least T − M + 1 sessions certify their `q_bytes`: the period certifies at least (T − M + 1)·`q_bytes`, and at most
  (M − 1)/T of its sessions are uncovered.
- **Per window** (`per_window`): the session's openings bound at δ/T instead of δ₀ (`pooled.bound` of its windows),
  so that every one of the T sessions certifies that much at once, except with δ + T·(p − δ₀), each session's terms
  beyond δ₀ once; no amount where that is 1 or more.

All leave out (`EXCEPT`) each session's Σ_t adv_t, its windows' PRF advantages (SHA-512 stream as PRF: not priced,
not assumed), so every δ and error of this module reads δ + Σ_t adv_t over the period's sessions; a session's sample
has no hiding term, since its prover is sent the coin commitment only at the session's end. What the session proves is
its verdict and outputs, by the Lean verifier; what is scaled is the
number of sessions and that each is audited as this one was."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

# The count bound is the wipe's too, held in pous: remote_bandwidth may import pous, and pous not it.
from verity.protocols.pous.verifier.wipe_monitoring import block_count, short_sessions
from verity.protocols.reporting import DELTAS, PERIODS, no_bound, summed

from . import codec, pooled

FORMAT = "verity/remote-bandwidth-monitoring-period/v1"
STATEMENT = {"name": "FlockSoundness.Audit.Periods.prCoin_count_le_choose",
             "file": "security/Proofs/Flock/Soundness/Audit/Periods.lean"}
EXCEPT = ("Σ_t adv_t, the PRF advantage of each window of each of the period's sessions (SHA-512 stream as PRF: not "
          "priced, not assumed): every δ and error here reads δ + Σ_t adv_t")
"""What every δ and error of a table leaves out, which the table carries beside them (`except_also`)."""


def _share(x: int, of: int) -> str:
    return f"{x / of:.3g}"


def whole_error(error: Mapping[str, Any] | None, eps: Fraction) -> tuple[Fraction, dict[str, float]]:
    """A session's whole error from its ledger's `error` (`ledger.Gamma.to_json`'s): the pooled reading's total
    (`horizon`) at the float's exact value, and the ledger's terms of it beyond its own δ₀ and slack: the sketch's and
    the sample's two. Refused when it is no such object, or is below `eps`, the answer's (module docstring)."""
    try:
        p = Fraction(error["horizon"])
        terms = {"sketch": float(error["sketch"]), "binding": float(error["draw"]["binding"]),
                 "rejection": float(error["draw"]["rejection"])}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("a period's table reads the session's whole error from its verifier's ledger, the error of "
                         "ledger.declared(record, node)'s γ (its horizon, sketch and draw terms; --ledger), not "
                         f"{error!r}") from e
    if p < eps:
        raise ValueError(f"the ledger's error, {float(p):.4g}, is below the answer's eps, {float(eps):.4g}, which it "
                         "holds with the sketch's term: it is not this session's")
    return p, terms


def _observed(answer: Mapping[str, Any], error: Mapping[str, Any] | None) -> dict[str, Any]:
    public, profile = answer["public"], answer["profile"]
    windows = [(int(u), int(s)) for u, s in profile["windows"]]
    eps = Fraction(public["eps"])
    p, terms = whole_error(error, eps)
    return {"q_bytes": int(public["q_bytes"]), "eps": str(eps), "eps_float": float(eps), "error": float(p),
            "error_terms": terms, "windows": windows,
            "receipts": sum(u for u, _ in windows), "draws": sum(s for _, s in windows),
            "gaps": list(profile.get("gaps") or [])}


def pooled_step(t: int, q_bytes: int, p: Fraction, delta: Fraction) -> dict[str, Any]:
    """The period's pooled total over `t` sessions of `q_bytes` each, each short with probability at most `p`, its
    whole error (module docstring)."""
    m, bound = short_sessions(t, p, delta)
    certified = max(0, t - m + 1)
    return {"short_sessions_at_most": min(m - 1, t), "certified_sessions_at_least": certified,
            "q_bytes_at_least": certified * q_bytes, "uncovered_at_most": _share(t - certified, t),
            "bound": f"{float(bound):.3g}", "statement": STATEMENT["name"]}


def per_window(t: int, windows: Sequence[tuple[int, int]], delta: Fraction, beyond: Fraction) -> dict[str, Any]:
    """Each of `t` sessions at δ/t: its openings' pooled bound at that δ, the reads it certifies and the period's total,
    except with δ and each session's `beyond` (its whole error less its δ₀); where that error is 1 or more, no amount
    and why (`reporting.no_bound`)."""
    error = delta + summed(beyond, t)
    said = {"at": f"δ/T = {delta}/{t} a session", "except_with": f"δ + T·(error − δ₀) = {float(error):.3g}"}
    if (why := no_bound(error)) is not None:
        return said | {"no_bound": why}
    b = pooled.bound(windows, delta / t)
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    each = (sum(u for u, _ in windows) - wrong) * codec.BLOCK_BYTES
    return {"q_bytes_each": each, "q_bytes": t * each} | said


def table(answer: Mapping[str, Any], error: Mapping[str, Any] | None, machines: int, window_s: int, *,
          deployment: str | None = None) -> dict[str, Any]:
    """The monitoring-period result of one session's core answer (`RemoteAudit.Protocol`'s) and its error in its
    verifier's ledger (`ledger.Gamma.to_json`'s `error`; read only for an accepted session) when every one of
    `machines` runs such a session every `window_s` seconds."""
    head = {"format": FORMAT, "deployment": {"id": deployment, "machines": machines, "window_s": window_s},
            "deltas": [str(d) for d in DELTAS], "statement": STATEMENT, "except_also": EXCEPT,
            "verdict": answer.get("verdict")}
    if answer.get("verdict") != "accept":
        return head | {"observed": None, "periods": None,
                       "why": f"the session wasn't accepted ({answer.get('reason')}), so it certifies nothing to pool"}
    o = _observed(answer, error)
    p = Fraction(o["error"])
    beyond = p - Fraction(answer["profile"]["delta"])
    periods = {}
    for name, seconds in PERIODS:
        per_machine, t = block_count(seconds, window_s), block_count(seconds, window_s, machines)
        row: dict[str, Any] = {"seconds": seconds, "sessions": t, "sessions_per_machine": per_machine,
                               "basis": f"one session scaled up to {t} sessions, not a measured {name}"}
        if t == 0:
            periods[name] = row | {"why": f"a {window_s} s window doesn't fit a {name}"}
            continue
        row["error"] = float(summed(p, t))
        if (why := no_bound(summed(p, t))) is not None:
            periods[name] = row | {"no_bound": why}
            continue
        row["q_bytes"] = t * o["q_bytes"]
        for d in DELTAS:
            row[str(d)] = {"pooled": pooled_step(t, o["q_bytes"], p, d),
                           "per_window": per_window(t, o["windows"], d, beyond) if not o["gaps"] else None}
        periods[name] = row
    return head | {"observed": o, "periods": periods,
                   "scaled": [f"the sessions: {machines} machines, one session every {window_s} s each",
                              (f"every session audited as this one: {o['q_bytes']} bytes certified, except with "
                               f"{o['error']:.4g}, its whole error (its eps, {o['eps_float']:.4g}, and the sketch's "
                               f"term, {o['error_terms']['sketch']:.4g}, among it)")]}


def ledger_error(doc: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The session's error in a ledger document: `ledger.Declared.to_json`'s, or an object that carries one under
    `declared` (a `verifier.json` session's `ledger`), at the specification's rate; the rate is not in the error, so
    the measured reading's is the same. None where the document holds none."""
    at = doc.get("declared", doc) if isinstance(doc, Mapping) else None
    for key in ("gamma", "spec", "error"):
        at = at.get(key) if isinstance(at, Mapping) else None
    return at


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity.protocols.poub.verifier.monitoring",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("answer", type=Path, help="the session's core answer, or a leg's bandwidth.json carrying it")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="the session's declared outputs in its verifier's ledger (ledger.declared's JSON, or a "
                         "verifier.json session's ledger), whose error the table carries; an accepted session's table "
                         "is refused without it")
    ap.add_argument("--machines", type=int, required=True)
    ap.add_argument("--window-s", type=int, required=True)
    ap.add_argument("--deployment", default=None, help="the deployment's name, recorded beside its numbers")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = json.loads(a.answer.read_text())
    error = None if a.ledger is None else ledger_error(json.loads(a.ledger.read_text()))
    try:
        result = table(doc.get("answer", doc), error, a.machines, a.window_s, deployment=a.deployment)
    except ValueError as e:
        ap.error(str(e))
    text = json.dumps(result, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


__all__ = ["FORMAT", "STATEMENT", "ledger_error", "main", "per_window", "pooled_step", "short_sessions", "table",
           "whole_error"]

if __name__ == "__main__":
    sys.exit(main())
