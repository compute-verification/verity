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


def communication(reports: Sequence[Mapping[str, Any]], period_s: float,
                  spans: Mapping[str, float] = SPANS_S) -> dict[str, Any]:
    """The network certifier's leak bound over the periods so far and pooled over each of ``spans`` of ``period_s``
    periods: a covert sender in the server moves at most an accepted period's ``K`` bits over its link, except with
    probability its ``eps`` (`NetTiming.Guarantees.PeriodLeakAccepted`), so over periods the bits add and, by a union
    bound, the eps too, ``vacuous`` once they reach 1. ``K_per_window`` is K over the periods' link-windows. A period
    that didn't accept certifies nothing, and counts in ``refused``, as does another protocol's answer."""
    rows = [r["protocols"].get("communication") or {} for r in reports]
    got = [r["quantity"] for r in rows if r.get("verdict") == "accept" and r.get("protocol") == CERTIFIER]
    K, eps, windows = sum(q["K"] for q in got), sum((Fraction(q["eps"]) for q in got), Fraction(0)), \
        sum(q["windows"] for q in got)
    out: dict[str, Any] = {
        "periods": len(rows), "accepted": len(got), "refused": len(rows) - len(got), "K": K,
        "eps": str(min(eps, Fraction(1))), "vacuous": eps >= 1, "windows": windows,
        "K_per_window": str(Fraction(K, windows)) if windows else None, "period_s": period_s}
    for name, span in spans.items():
        n = max(1, math.floor(span / period_s)) if period_s > 0 else None
        row: dict[str, Any] = {"periods": n}
        if n is not None and got:
            pooled = n * eps / len(got)
            row.update(K=math.ceil(Fraction(n * K, len(got))), eps=str(min(pooled, Fraction(1))), vacuous=pooled >= 1)
        out[name] = row
    return out


__all__ = ["CERTIFIER", "MONTH_S", "PROTOCOL", "SPANS_S", "bind", "communication", "composed", "horizon", "onsite_check",
           "onsite_components", "onsite_inputs", "stub_component"]
