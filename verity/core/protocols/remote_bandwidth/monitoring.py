"""The remote bandwidth audit over a monitoring period: a week, a month and a year of every machine of a
deployment, at δ = 1% and 10⁻⁶, the pooled total first and the per-window figure beside it, from one accepted session's
answer as `RemoteAudit.Protocol.certify` publishes it: `q_bytes` certified, except with `eps` = δ₀ + S/2^64 (δ₀ the
openings' pooled bound's δ, 2^-40, and S the session's draws).

A deployment of `machines` machines, each running one session every `window_s` seconds audited as this one was, has
T = machines·⌊period/window_s⌋ sessions in a period, each on its own live coins, against one adaptive prover.

- **Pooled** (`pooled_step`): the count bound `FlockSoundness.Audit.Periods.prCoin_count_le_choose`. When each
  session, given the earlier ones, is accepted and short of its `q_bytes` with probability at most p = `eps`, at least M
  of the T are with probability at most C(T, M)·p^M. M is the least count within δ (`short_sessions`), so except with δ
  at least T − M + 1 sessions certify their `q_bytes`: the period certifies at least (T − M + 1)·`q_bytes`, and at most
  (M − 1)/T of its sessions are uncovered.
- **Per window** (`per_window`): the session's openings bound at δ/T instead of δ₀ (`pooled.bound` of its windows),
  so that every one of the T sessions certifies that much at once, except with δ + T·S/2^64.

Both leave out each session's coin PRF advantage, a named assumption, not a number; a session's sample has no hiding
term, since its prover is sent the coin commitment only at the session's end. What the session proves is its verdict
and outputs, by the Lean verifier; what is scaled is the
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
from verity.core.protocols.pous.wipe_monitoring import DELTAS, PERIODS, short_sessions

from . import codec, pooled

FORMAT = "verity/remote-bandwidth-monitoring-period/v1"
STATEMENT = {"name": "FlockSoundness.Audit.Periods.prCoin_count_le_choose", "pr": 4413,
             "file": "Security/Proofs/Flock/Soundness/Audit/Periods.lean"}
EXCEPT = "each session's coin PRF advantage (a named assumption) over the period's sessions"


def _share(x: int, of: int) -> str:
    return f"{x / of:.3g}"


def _observed(answer: Mapping[str, Any]) -> dict[str, Any]:
    public, profile = answer["public"], answer["profile"]
    windows = [(int(u), int(s)) for u, s in profile["windows"]]
    eps = Fraction(public["eps"])
    return {"q_bytes": int(public["q_bytes"]), "eps": str(eps), "eps_float": float(eps), "windows": windows,
            "receipts": sum(u for u, _ in windows), "draws": sum(s for _, s in windows),
            "gaps": list(profile.get("gaps") or [])}


def pooled_step(t: int, q_bytes: int, eps: Fraction, delta: Fraction) -> dict[str, Any]:
    """The period's pooled total over `t` sessions of `q_bytes` each at `eps` (module docstring)."""
    m, bound = short_sessions(t, eps, delta)
    certified = max(0, t - m + 1)
    return {"short_sessions_at_most": min(m - 1, t), "certified_sessions_at_least": certified,
            "q_bytes_at_least": certified * q_bytes, "uncovered_at_most": _share(t - certified, t),
            "bound": f"{float(bound):.3g}", "statement": STATEMENT["name"]}


def per_window(t: int, windows: Sequence[tuple[int, int]], delta: Fraction) -> dict[str, Any]:
    """Each of `t` sessions at δ/t: its openings' pooled bound at that δ, the reads it certifies and the period's total."""
    b = pooled.bound(windows, delta / t)
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    each = (sum(u for u, _ in windows) - wrong) * codec.BLOCK_BYTES
    draws = sum(s for _, s in windows)
    return {"q_bytes_each": each, "q_bytes": t * each, "at": f"δ/T = {delta}/{t} a session",
            "except_with": f"δ + T·S/2^64 = {float(delta + Fraction(t * draws, 2 ** 64)):.3g}"}


def table(answer: Mapping[str, Any], machines: int, window_s: int, *, deployment: str | None = None) -> dict[str, Any]:
    """The monitoring-period result of one session's core answer (`RemoteAudit.Protocol`'s) when every one of `machines`
    runs such a session every `window_s` seconds."""
    head = {"format": FORMAT, "deployment": {"id": deployment, "machines": machines, "window_s": window_s},
            "deltas": [str(d) for d in DELTAS], "statement": STATEMENT, "except_also": EXCEPT,
            "verdict": answer.get("verdict")}
    if answer.get("verdict") != "accept":
        return head | {"observed": None, "periods": None,
                       "why": f"the session wasn't accepted ({answer.get('reason')}), so it certifies nothing to pool"}
    o = _observed(answer)
    eps = Fraction(o["eps"])
    periods = {}
    for name, seconds in PERIODS:
        per_machine = seconds // window_s
        t = machines * per_machine
        row: dict[str, Any] = {"seconds": seconds, "sessions": t, "sessions_per_machine": per_machine}
        if t == 0:
            periods[name] = row | {"why": f"a {window_s} s window doesn't fit a {name}"}
            continue
        for d in DELTAS:
            row[str(d)] = {"pooled": pooled_step(t, o["q_bytes"], eps, d),
                           "per_window": per_window(t, o["windows"], d) if not o["gaps"] else None}
        periods[name] = row
    return head | {"observed": o, "periods": periods,
                   "scaled": [f"the sessions: {machines} machines, one session every {window_s} s each",
                              (f"every session audited as this one: {o['q_bytes']} bytes certified at eps "
                               f"{o['eps_float']:.4g}")]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity.core.protocols.remote_bandwidth.monitoring",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("answer", type=Path, help="the session's core answer, or a leg's bandwidth.json carrying it")
    ap.add_argument("--machines", type=int, required=True)
    ap.add_argument("--window-s", type=int, required=True)
    ap.add_argument("--deployment", default=None, help="the deployment's name, recorded beside its numbers")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = json.loads(a.answer.read_text())
    result = table(doc.get("answer", doc), a.machines, a.window_s, deployment=a.deployment)
    text = json.dumps(result, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


__all__ = ["DELTAS", "FORMAT", "PERIODS", "STATEMENT", "main", "per_window", "pooled_step", "short_sessions", "table"]

if __name__ == "__main__":
    sys.exit(main())
