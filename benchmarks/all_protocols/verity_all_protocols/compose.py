"""The composed verdict through onsite Verity (`core_onsite`, ``verity/onsite/v0``), and the bound over a horizon of periods.

Onsite's check (verity/core/protocols/composition/README.md) runs each component's own ``answer`` on that component's
case and decides, reject first: any component's rejection rejects onsite whatever the others answer; otherwise onsite
accepts only when every component accepts and its commitment binds their digests. So the runner decides:

* **reject** as soon as one leg's protocol rejects: by reject dominance onsite rejects whatever the others would answer;
* **pending-ports** while onsite can't decide the rest: a leg is missing or failed, a leg has no case for its port, or one
  of onsite's components is still a stub (`onsite_components` reads `Core.Onsite.components`), since a stub given a real
  case answers for the stub, not for the protocol;
* otherwise onsite's own executable decides on the legs' cases (`onsite_check`, ``onsite check PARAMS COMMITMENT
  EVIDENCE``), and its answer is the composed verdict.

Bandwidth is not one of onsite's components yet: its verdict is reported beside the composition, and a rejection there
rejects the composed verdict too (no accounting accepts over a refused audit), but its acceptance can't complete it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.ml.circuits.codec import canonical_json

ROOT = Path(__file__).resolve().parents[3]
ONSITE = ROOT / "verity" / "core" / "protocols" / "onsite" / "CoreOnsite.lean"
PROTOCOL = "verity/onsite/v0"
#: a month, the horizon `horizon` projects to
MONTH_S = 30 * 86400
#: the network certifier's period check, whose public carries K and eps
CERTIFIER = "verity/certifier/v0"
#: the spans `communication` pools the network certifier's K over
SPANS_S = {"week": 7 * 86400, "month": MONTH_S, "year": 365 * 86400}
#: the δ `communication` pools at
POOL_DELTAS = (Fraction(1, 100), Fraction(1, 10 ** 6))
#: a proof unit's outputs are at most 16 bits (`NetworkCheck.Period`'s faulty bits, `16 · b`)
OUT_BITS = 16
#: what `horizon_leak`'s K rests on until its two per-window inputs are discharged
HORIZON_CONDITION = ("NetTiming.Guarantees.HorizonLeakAccepted (#4367) is conditional: its per-window inputs hesc "
                     "(proofs' SampledKSLive) and hleak (the live SampledLeakAccepted) are not discharged")
#: the machines `communication`'s fleet row joins, a stated parameter (the census holds no fleet size)
FLEET = 1000


def onsite_components(source: str | None = None) -> list[str]:
    """`Core.Onsite.components` as written in ``CoreOnsite.lean``: each entry's Lean name (``Stub.pouw``, or a port's
    ``Pouw.Core.protocol``)."""
    text = ONSITE.read_text() if source is None else source
    m = re.search(r"def components : List Protocol :=\s*\[([^\]]*)\]", text)
    if m is None:
        raise ValueError("CoreOnsite.lean has no `def components : List Protocol := [...]`")
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def stub_component(name: str) -> str | None:
    """The leg a stub stands for (``Stub.sampledProofs`` -> ``sampled-proofs``), or None for a port."""
    if not name.startswith("Stub."):
        return None
    return re.sub(r"(?<!^)([A-Z])", r"-\1", name.removeprefix("Stub.")).lower()


def bind(protocol: str, components: Sequence[tuple[str, str]]) -> bytes:
    """`Core.Composition.bind`: SHA-512 of the product's id, a zero byte and
    ``{"components":[{"commitment":HEX,"id":ID},...]}``, over (id, commitment hex) in the order of the ids."""
    body = canonical_json({"components": [{"commitment": c, "id": i} for i, c in components]})
    return hashlib.sha512(protocol.encode() + b"\0" + body).digest()


def onsite_inputs(cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str]) -> tuple[bytes, bytes, bytes]:
    """Onsite's params, commitment and evidence from each component's case and its commitment's digest (its answer's
    ``commitment``), both keyed by the component's protocol id: what ``onsite check`` reads (the composition README's
    table)."""
    ids = sorted(cases)
    params = canonical_json({"components": [{"id": i, "params": cases[i]["params"]} for i in ids]})
    commitment = bind(PROTOCOL, [(i, digests[i]) for i in ids])
    evidence = canonical_json({"components": [{"commitment": cases[i]["commitment"], "evidence": cases[i]["evidence"],
                                               "id": i, "params": cases[i]["params"]} for i in ids]})
    return params, commitment, evidence


def onsite_check(exe: str, cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str]) -> dict[str, Any]:
    """``onsite check`` on the components' cases (`onsite_inputs`): its answer line, parsed."""
    params, commitment, evidence = onsite_inputs(cases, digests)
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params.json", params), ("commitment.bin", commitment), ("evidence.json", evidence)):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([exe, "check", *files], capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1, 2):
        raise RuntimeError(f"onsite check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def composed(rows: Mapping[str, Mapping[str, Any]], components: Mapping[str, str | None],
             cases: Mapping[str, Mapping[str, Any] | None], onsite: Sequence[str] | None = None,
             exe: str | None = None) -> dict[str, Any]:
    """The composed verdict (module docstring) over the period's rows (by leg name, each with its ``verdict``),
    ``components`` (leg name -> onsite component, or None), the legs' ``cases`` and onsite's component list."""
    onsite = onsite_components() if onsite is None else list(onsite)
    stubs = sorted(c for c in (stub_component(x) for x in onsite) if c is not None)
    out: dict[str, Any] = {"protocol": PROTOCOL, "onsite_components": onsite, "stubs": stubs,
                           "outside_onsite": sorted(n for n, c in components.items() if c is None)}
    rejected = sorted(n for n, r in rows.items() if r.get("verdict") == "reject")
    if rejected:
        return {**out, "verdict": "reject", "reason": "rejected by " + ", ".join(rejected) + " (reject dominates)"}
    waiting = []
    if stubs:
        waiting.append("onsite's stubs: " + ", ".join(stubs))
    for n, r in sorted(rows.items()):
        if r.get("verdict") in ("missing", "error"):
            waiting.append(f"{n} {r['verdict']}")
        elif components.get(n) is not None and cases.get(n) is None:
            waiting.append(f"{n} has no case for its port")
    if waiting or exe is None:
        return {**out, "verdict": "pending-ports",
                "reason": "; ".join(waiting) if waiting else "no onsite executable given"}
    legs = [n for n, c in components.items() if c is not None]
    got = onsite_check(exe, {rows[n]["protocol"]: cases[n] for n in legs},
                       {rows[n]["protocol"]: rows[n]["commitment"] for n in legs})
    undecided = sorted(n for n, r in rows.items() if components.get(n) is None and r.get("verdict") != "accept")
    verdict = got["verdict"] if not (got["verdict"] == "accept" and undecided) else "undecided"
    return {**out, "verdict": verdict, "reason": got.get("reason"), "answer": got}


def horizon(reports: Sequence[Mapping[str, Any]], period_s: float, month_s: float = MONTH_S) -> dict[str, Any]:
    """The integrity bound over the periods so far and projected to a month of ``period_s`` periods: each period is its
    own audit on its own live coins, so by a union bound, except with probability at most the sum of their δ, every
    accepted period's wrong units are within its bound, and the horizon's wrong units within their sum. ``vacuous`` when
    that probability reaches 1. A period that didn't accept certifies nothing, and counts in ``refused``."""
    rows = [r["protocols"]["integrity"] for r in reports]
    accepted = [r for r in rows if r.get("verdict") == "accept"]
    delta = sum((Fraction(r["delta"]) for r in accepted), Fraction(0))
    so_far = {"periods": len(rows), "accepted": len(accepted), "refused": len(rows) - len(accepted),
              "wrong_units_bound": sum(r["quantity"]["bound"] for r in accepted),
              "units": sum(r["answer"]["profile"]["levels"][-1]["size"] for r in accepted),
              "delta": str(min(delta, Fraction(1))), "vacuous": delta >= 1}
    n = max(1, math.floor(month_s / period_s)) if period_s > 0 else None
    month: dict[str, Any] = {"periods": n, "period_s": period_s}
    if n is not None and accepted:
        per = delta / len(accepted)
        month.update(delta=str(min(n * per, Fraction(1))), vacuous=n * per >= 1,
                     wrong_units_bound=n * so_far["wrong_units_bound"] // len(accepted),
                     units=n * so_far["units"] // len(accepted))
    worst = "reject" if any(r["composed"]["verdict"] == "reject" for r in reports) else \
        "accept" if reports and all(r["composed"]["verdict"] == "accept" for r in reports) else "pending-ports"
    return {"rule": "union bound over periods, each audited on its own live coins", "integrity": so_far, "month": month,
            "communication": communication(reports, period_s), "composed": worst}


def horizon_leak(periods: Sequence[Mapping[str, Any]], delta: Fraction, count: int | None = None) -> dict[str, Any]:
    """The network certifier's K over a monitoring period of ``count`` periods like ``periods`` (accepted
    ``verity/certifier/v0`` publics; by default those periods alone), its windows one sequence, each on its own live
    coins (`NetTiming.Guarantees.HorizonLeakAccepted`, #4367, on `HorizonWrongUnitsLe`, #4355): the faulty units are
    paid for once over the period, ``Fmax = ⌈ln(1/δ)/q⌉`` with ``q`` the least ``k_i/n_i`` of its audits, so
    ``K = log2 #Ss + Σ_c log2(cap_c + 1) + Fmax·(16 + log2(N + 1)) + Σ_i (free_i + k_i·log2 Dj)``, ``N = Σ_i n_i``,
    except with probability ``δ + Σ_i (e_i + e'_i)``: δ once, and each window's soundness and leak errors, which have no
    number yet (``e_for_eps_2delta`` is the per-window error at which that sum stays within 2δ). The schedule, cell,
    free-input and jitter terms are the periods' own, scaled to ``count``. ``share`` is K over the period's registered
    output bits, 16 a unit. #4367 is conditional on its two per-window inputs (`HORIZON_CONDITION`), so this K is cited
    as conditional."""
    count = len(periods) if count is None else count
    audits = [a for p in periods for a in p["audits"]]
    q = min(Fraction(a["drawn"], a["units"]) for a in audits)
    fmax = math.ceil(math.log(1 / float(delta)) / float(q) * (1 + 1e-12))
    scale = Fraction(count, len(periods))
    N = math.ceil(scale * sum(a["units"] for a in audits))
    terms = {"schedules": math.ceil(scale * sum(p["schedules"] for p in periods)),
             "cells": math.ceil(scale * sum(p["cells"] for p in periods)),
             "faulty_bits": OUT_BITS * fmax, "faulty_sets": ((N + 1) ** fmax - 1).bit_length(),
             "free": math.ceil(scale * sum(a["free"] for a in audits)),
             "jitter": math.ceil(scale * sum(a["jitter"] for a in audits))}
    K = sum(terms.values())
    windows = math.ceil(scale * len(audits))
    return {"periods": count, "windows": windows, "units": N, "q": str(q), "Fmax": fmax, "K": K, "terms": terms,
            "share": K / (OUT_BITS * N), "eps": f"{delta} + Σ_i (e_i + e'_i) over {windows} windows",
            "e_for_eps_2delta": str(delta / windows), "conditional": HORIZON_CONDITION}


def fleet(periods: Sequence[Mapping[str, Any]], delta: Fraction, count: int, machines: int = FLEET) -> dict[str, Any]:
    """``machines`` machines, each its own network certifier over ``count`` periods like ``periods``, joined by a union
    bound: each machine's `horizon_leak` at ``delta / machines``, so the K add and the δ add back to ``delta``. The
    union bound across machines is not in Lean."""
    one = horizon_leak(periods, delta / machines, count)
    return {"machines": machines, "delta_each": str(delta / machines), "K": machines * one["K"], "share": one["share"],
            "units": machines * one["units"], "Fmax_each": one["Fmax"], "eps": f"{delta} + Σ over machines and windows "
            "of (e_i + e'_i)", "lean": "per machine; the union bound over machines is not in Lean",
            "conditional": HORIZON_CONDITION}


def communication(reports: Sequence[Mapping[str, Any]], period_s: float, spans: Mapping[str, float] = SPANS_S,
                  deltas: Sequence[Fraction] = POOL_DELTAS, machines: int = FLEET) -> dict[str, Any]:
    """The network certifier over the periods so far and over each of ``spans`` of ``period_s`` periods, at each of
    ``deltas``: the pooled K first (`horizon_leak`, one machine; `fleet`, ``machines`` of them), and beside it the
    per-window figure, each accepted period's own ``K`` at its ``eps`` (`NetTiming.Guarantees.PeriodLeakAccepted`) with
    its share of the period's output bits. A period that didn't accept certifies nothing, and counts in ``refused``, as
    does another protocol's answer."""
    rows = [r["protocols"].get("communication") or {} for r in reports]
    got = [r["quantity"] for r in rows if r.get("verdict") == "accept" and r.get("protocol") == CERTIFIER]
    out: dict[str, Any] = {
        "periods": len(rows), "accepted": len(got), "refused": len(rows) - len(got), "period_s": period_s,
        "per_window": [{"K": q["K"], "eps": q["eps"], "windows": q["windows"],
                        "share": q["K"] / (OUT_BITS * sum(a["units"] for a in q["audits"]))} for q in got],
        "pooled": {str(d): horizon_leak(got, d) for d in deltas} if got else None}
    for name, span in spans.items():
        n = max(1, math.floor(span / period_s)) if period_s > 0 else None
        out[name] = {"periods": n, **({"pooled": {str(d): horizon_leak(got, d, n) for d in deltas},
                                       "fleet": {str(d): fleet(got, d, n, machines) for d in deltas}}
                                      if n is not None and got else {})}
    return out


__all__ = ["CERTIFIER", "FLEET", "HORIZON_CONDITION", "MONTH_S", "OUT_BITS", "POOL_DELTAS", "PROTOCOL", "SPANS_S",
           "bind", "communication", "composed", "fleet", "horizon", "horizon_leak", "onsite_check", "onsite_components",
           "onsite_inputs", "stub_component"]
