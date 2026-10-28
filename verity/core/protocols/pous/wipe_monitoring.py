"""The wipe over a monitoring period: a week, a month and a year of every machine of a deployment, at δ = 1% and 10⁻⁶,
from one accepted wipe's answer as `wipe_period.certify` publishes it: at most `foreign_bits` of the declared tiers'
`memory_bits` hold anything but the fill, except with `eps` (2^-128 when the bound covers the session; with `memory_bits`
for `foreign_bits` and `eps` 0, the vacuous statement, when it doesn't).

A deployment of `machines` machines, each wiped every `cadence_s` seconds (`CADENCE_S`, a day, by default) as this one
was, has T = machines·⌊period/cadence_s⌋ wipes in a period, each on the memory challenger's own live coins, against one
adaptive prover.

- **Pooled** (`pooled_step`): the count bound `FlockSoundness.Audit.Periods.prCoin_count_le_choose`. When each wipe,
  given the earlier ones, is accepted and leaves more than its `foreign_bits` with probability at most p = `eps`, at
  least M of the T are with probability at most C(T, M)·p^M. M is the least count within δ (`short_sessions`), so
  except with δ at most M − 1 of the T wipes fail their bound.
- **Per wipe** (`per_wipe`): every one of the T wipes meets its `foreign_bits` at once, except with T·`eps`.

`PERIODS` is the core's `MONITORING_PERIODS` and T its `block_count` (`verity.core.periods`). `short_sessions`, `PERIODS`
and `DELTAS` are the remote bandwidth audit's too (`remote_bandwidth.monitoring`).

A wipe signed `UNCOUNTED` is vacuous until a network certifier counts its egress: `counted` recomputes its bound from a
covered `wipe-egress/v1` record's `egress_mib` (`wipe_period.certify`, on the session's parameters in the answer's
profile). Nothing here checks that record's certificates or that its window is the wipe's session.

Both statements leave out (`EXCEPT`) each wipe's PRF advantage, δ_net and the egress count's premises, which are named
assumptions, not numbers, and what the wipe's own bound doesn't cover (`wipe_period.uncertified` of the answer's
placement). What the wipe proves is its verdict and outputs, by the Lean verifier of record; what is scaled is the number
of wipes and that each is wiped as this one was."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.periods import MONITORING_PERIODS, block_count

from . import streamed_wipe as sw
from . import wipe_period

PERIODS = MONITORING_PERIODS
DELTAS: tuple[Fraction, ...] = (Fraction(1, 100), Fraction(1, 10 ** 6))
FORMAT = "verity/wipe-monitoring-period/v1"
STATEMENT = {"name": "FlockSoundness.Audit.Periods.prCoin_count_le_choose",
             "file": "Security/Proofs/Flock/Soundness/Audit/Periods.lean"}
CADENCE_S = 86400
EGRESS = "wipe-egress/v1"
HBM = "gpu-hbm"
EXCEPT = ("each wipe's PRF advantage (its labels' and coins' PRF, a named assumption), δ_net (the network certifier's "
          "error in the egress count) and the egress count's premises (every link leaving the node counted, its window "
          "the wipe's session), with the guarantee's own (every tier declared, no side channel), over the period's wipes")


def short_sessions(t: int, p: Fraction, delta: Fraction) -> tuple[int, Fraction]:
    """The least M ≥ 1 with C(t, M)·p^M ≤ δ, and that bound; (t + 1, 1) when even M = t isn't within δ."""
    if not 0 < delta < 1 or not 0 <= p <= 1 or t < 1:
        raise ValueError(f"δ is in (0, 1), p in [0, 1] and the sessions at least 1, not {delta}, {p}, {t}")
    term = t * p
    for m in range(1, t + 1):
        if term <= delta:
            return m, term
        term = term * (t - m) * p / (m + 1)
    return t + 1, Fraction(1)


def pooled_step(t: int, eps: Fraction, delta: Fraction) -> dict[str, Any]:
    """The period's pooled statement over `t` wipes at `eps` (module docstring)."""
    m, bound = short_sessions(t, eps, delta)
    failing = min(m - 1, t)
    return {"failing_at_most": failing, "meeting_at_least": t - failing, "bound": f"{float(bound):.3g}",
            "statement": STATEMENT["name"]}


def per_wipe(t: int, eps: Fraction) -> dict[str, Any]:
    """Every one of `t` wipes meets its `foreign_bits` at once, except with t·`eps`."""
    return {"except_with": f"T·eps = {float(t * eps):.3g}"}


def params_of(profile: Mapping[str, Any]) -> sw.WipeParams:
    """The session's parameters, as the answer's profile (`wipe_period.profile`) gives them."""
    return sw.WipeParams(int(profile["labels"]), int(profile["k"]), int(profile["batch"]), int(profile["limit_ns"]),
                         tuple((str(t), int(b)) for t, b in profile["tiers"]))


def counted(answer: Mapping[str, Any], egress: Mapping[str, Any]) -> dict[str, Any]:
    """`answer`, a wipe signed `UNCOUNTED`, with `wipe_period.certify(p, egress_mib)` for its outputs, `egress` a covered
    `wipe-egress/v1` record."""
    if egress.get("format") != EGRESS or not egress.get("covered"):
        raise ValueError(f"the egress record is not a covered {EGRESS} record: {egress.get('format')}, "
                         f"covered {egress.get('covered')}, why {egress.get('why')}")
    public, p = answer["public"], params_of(answer["profile"])
    if public["egress_mib"] != sw.EGRESS_MAX_UNITS:
        raise ValueError(f"the wipe's egress was counted when it was signed ({public['egress_mib']} MiB): an egress "
                         "record replaces only an UNCOUNTED one")
    if wipe_period.certify(p, public["egress_mib"]) != public:
        raise ValueError("the answer's profile doesn't give its outputs: certify(p, egress_mib) differs from its public")
    return dict(answer) | {"public": wipe_period.certify(p, int(egress["egress_mib"]))}


def memory(profile: Mapping[str, Any]) -> dict[str, Any]:
    """The declared tiers, whether the wipe is of GPU memory alone, and the devices its prover's store names."""
    tiers = [{"tier": t, "bytes": int(b)} for t, b in profile["tiers"]]
    store = (profile.get("placement") or {}).get("prover_store") or {}
    devices = [{k: s[k] for k in ("tier", "device", "uuid") if k in s}
               for s in (store.get("parts") or [store]) if s.get("device")]
    return {"tiers": tiers, "tier": "+".join(t["tier"] for t in tiers), "hbm_only": [t["tier"] for t in tiers] == [HBM],
            "devices": devices}


def each(public: Mapping[str, Any]) -> dict[str, Any]:
    """What each wipe certifies: at most `foreign_bits` of `memory_bits` foreign, the rest wiped, except with `eps`."""
    eps, f, m = Fraction(public["eps"]), int(public["foreign_bits"]), int(public["memory_bits"])
    return {"foreign_bits": f, "memory_bits": m, "wiped_bits": m - f, "wiped_share": f"{(m - f) / m:.4g}",
            "eps": str(eps), "eps_float": float(eps), "k": public["k"], "r": public["r"],
            "egress_mib": public["egress_mib"], "certified": public["certified"]}


def table(answer: Mapping[str, Any], machines: int, cadence_s: int = CADENCE_S, *,
          egress: Mapping[str, Any] | None = None, deployment: str | None = None) -> dict[str, Any]:
    """The monitoring-period result of one wipe's answer (`wipe_period.answer`'s) when every one of `machines` is wiped
    as this one was every `cadence_s` seconds; `egress`, a `wipe-egress/v1` record, counts an `UNCOUNTED` wipe's egress
    (`counted`)."""
    head = {"format": FORMAT, "deployment": {"id": deployment, "machines": machines, "cadence_s": cadence_s},
            "deltas": [str(d) for d in DELTAS], "statement": STATEMENT, "verdict": answer.get("verdict"),
            "egress": None if egress is None else {k: egress.get(k) for k in ("link", "egress_bits", "egress_mib",
                                                                              "delta", "covered")}}
    if answer.get("verdict") != "accept":
        return head | {"each": None, "memory": None, "periods": None,
                       "why": f"the wipe wasn't accepted ({answer.get('reason')}), so it certifies nothing to pool"}
    if egress is not None:
        answer = counted(answer, egress)
    where = answer["profile"].get("placement")
    leaves = [EXCEPT, *([] if where is None else wipe_period.uncertified(where, None))]
    head |= {"each": each(answer["public"]), "memory": memory(answer["profile"]), "leaves_out": leaves}
    if not answer["public"]["certified"]:
        gaps = "; ".join(answer["profile"].get("gaps") or [])
        return head | {"periods": None, "why": f"the wipe's bound is vacuous (foreign_bits = memory_bits): {gaps}"}
    eps = Fraction(answer["public"]["eps"])
    periods = {}
    for name, seconds in PERIODS:
        per_machine, t = block_count(seconds, cadence_s), block_count(seconds, cadence_s, machines)
        row: dict[str, Any] = {"seconds": seconds, "wipes": t, "wipes_per_machine": per_machine}
        if t == 0:
            periods[name] = row | {"why": f"a wipe every {cadence_s} s doesn't fit a {name}"}
            continue
        periods[name] = row | {str(d): pooled_step(t, eps, d) for d in DELTAS} | {"per_wipe": per_wipe(t, eps)}
    return head | {"periods": periods,
                   "scaled": [f"the wipes: {machines} machines, one wipe every {cadence_s} s each",
                              f"every wipe as this one: {head['each']['foreign_bits']} of {head['each']['memory_bits']} "
                              f"bits foreign at most ({head['memory']['tier']}), at eps {head['each']['eps_float']:.4g}"]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity.core.protocols.pous.wipe_monitoring",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("answer", type=Path, help="the wipe's answer, or a leg's wipe.json carrying it")
    ap.add_argument("--machines", type=int, required=True)
    ap.add_argument("--cadence-s", type=int, default=CADENCE_S, help="seconds between one machine's wipes")
    ap.add_argument("--egress", type=Path, default=None, help=f"a {EGRESS} record counting an UNCOUNTED wipe's egress")
    ap.add_argument("--deployment", default=None, help="the deployment's name, recorded beside its numbers")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = json.loads(a.answer.read_text())
    egress = None if a.egress is None else json.loads(a.egress.read_text())
    result = table(doc.get("answer", doc), a.machines, a.cadence_s, egress=egress, deployment=a.deployment)
    text = json.dumps(result, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


__all__ = ["CADENCE_S", "DELTAS", "EGRESS", "EXCEPT", "FORMAT", "HBM", "PERIODS", "STATEMENT", "counted", "each", "main",
           "memory", "params_of", "per_wipe", "pooled_step", "short_sessions", "table"]

if __name__ == "__main__":
    sys.exit(main())
