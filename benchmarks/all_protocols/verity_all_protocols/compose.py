"""The composed verdict through onsite Verity (`core_onsite`, ``verity/onsite/v0``), and the bound over a horizon of periods.

Onsite's check (verity/core/protocols/composition/README.md) runs each component's own ``answer`` on that component's
case and decides, reject first: any component's rejection rejects onsite whatever the others answer; otherwise onsite
accepts only when every component accepts and its commitment binds their digests. So the runner decides:

* **reject** as soon as one leg's protocol rejects: by reject dominance onsite rejects whatever the others would answer;
* **pending-ports** while onsite can't decide the rest: a leg is missing or failed, a leg has no case for its port, or one
  of onsite's components is still a stub (`onsite_components` reads `Core.Onsite.components`), since a stub answers for
  the stub, not for the protocol;
* otherwise onsite's own executable decides on the legs' cases (`onsite_check`, ``onsite check PARAMS COMMITMENT
  EVIDENCE``), and its answer is the composed verdict.

Given onsite's executable, the runner runs ``onsite check`` every period, whatever it decides: each port on its leg's
case, each stub on a case with no statement yet (`STUB_CASE`, undecided), and a port whose leg wrote no case on one its
decoders refuse (`NO_CASE`). Its answer is ``answer``, where each component's case came from is ``inputs``, and ``agree``
says onsite rejected on every leg whose own case rejected.

A protocol outside onsite's components (`table.Entry.onsite` false) is reported beside the composition, and a rejection
there rejects the composed verdict too (no accounting accepts over a refused audit), but its acceptance can't complete it.
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


#: The ports `Core.Onsite.components` may name, by their Lean names: the leg each answers for and its protocol id.
PORTS = {
    "SampledProofs.OneStage.protocol": ("sampled-proofs", "verity/sampled-proofs/one-stage/v1"),
    "PouwCheck.protocol": ("pouw", "verity/pouw/v1"),
    "NetworkCheck.Check.protocol": ("certifier", "verity/network-certificate/v4"),
    "NetworkCheck.Period.protocol": ("certifier", "verity/certifier/v0"),
    "PousCheck.RunProtocol.protocol": ("pous", "verity/pous/p2/v4/run"),
    "RemoteAudit.Protocol.protocol": ("bandwidth", "verity/remote-bandwidth/v0"),
}
#: A stub's case with no statement yet (`Core.Example.protocol`: none of its one unit drawn), so it is undecided: a stub
#: never accepts or rejects for the protocol it stands for. Its 32-byte root is its own digest.
STUB_CASE = {"commitment": "00" * 32, "evidence": canonical_json({"drawn": 0, "passed": 0}).hex(), "params": {"k": 1}}
#: The case of a port whose leg wrote none: its own decoders refuse it, so that component rejects.
NO_CASE = {"commitment": "", "evidence": "", "params": {}}


def component_of(name: str) -> tuple[str, str, bool]:
    """The leg an entry of `Core.Onsite.components` answers for, its protocol id, and whether it is a stub."""
    stub = stub_component(name)
    if stub is not None:
        return stub, f"verity/stub/{stub}/v0", True
    if name not in PORTS:
        raise ValueError(f"onsite's component {name} is neither a stub nor a port `compose.PORTS` names: add it there")
    return (*PORTS[name], False)


def onsite_cases(rows: Mapping[str, Mapping[str, Any]], components: Mapping[str, str | None],
                 cases: Mapping[str, Mapping[str, Any] | None], onsite: Sequence[str]) -> tuple[dict, dict, dict, dict]:
    """Every onsite component's case and digest, by protocol id, where each came from: its leg's case and its answer's
    ``commitment`` (``leg``), `STUB_CASE` for a stub (``stub``), or `NO_CASE` for a port whose leg wrote none
    (``no case``, with the leg's verdict); and the leg whose case each ``leg`` one is."""
    leg = {c: n for n, c in components.items() if c is not None}
    got, digests, source, legs = {}, {}, {}, {}
    for x in onsite:
        component, pid, stub = component_of(x)
        name = leg.get(component)
        row = rows.get(name) or {}
        if stub:
            got[pid], digests[pid], source[pid] = STUB_CASE, STUB_CASE["commitment"], "stub"
        elif name is not None and cases.get(name) is not None:
            got[pid], digests[pid], source[pid], legs[pid] = cases[name], row.get("commitment") or "", "leg", name
        else:
            got[pid], digests[pid] = NO_CASE, ""
            source[pid] = f"no case ({name or component} {row.get('verdict', 'missing')})"
    return got, digests, source, legs


def composed(rows: Mapping[str, Mapping[str, Any]], components: Mapping[str, str | None],
             cases: Mapping[str, Mapping[str, Any] | None], onsite: Sequence[str] | None = None,
             exe: str | None = None) -> dict[str, Any]:
    """The composed verdict (module docstring) over the period's rows (by leg name, each with its ``verdict``),
    ``components`` (leg name -> onsite component, or None), the legs' ``cases`` and onsite's component list. With
    ``exe``, ``onsite check`` runs on every period (`onsite_cases`), and its answer is ``answer``; ``agree`` says whether
    it rejected on each leg whose own case rejected."""
    onsite = onsite_components() if onsite is None else list(onsite)
    stubs = sorted(c for c in (stub_component(x) for x in onsite) if c is not None)
    out: dict[str, Any] = {"protocol": PROTOCOL, "onsite_components": onsite, "stubs": stubs,
                           "outside_onsite": sorted(n for n, c in components.items() if c is None)}
    got = None
    if exe is not None:
        cs, digests, source, legs = onsite_cases(rows, components, cases, onsite)
        got = onsite_check(exe, cs, digests)
        mine = [pid for pid, n in legs.items() if rows[n].get("verdict") == "reject"]
        out.update(inputs=source, answer=got, agree=all(
            got["verdict"] == "reject" and f"{pid} reject: " in (got.get("reason") or "") for pid in mine))
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
    if waiting or got is None:
        return {**out, "verdict": "pending-ports",
                "reason": "; ".join(waiting) if waiting else "no onsite executable given"}
    undecided = sorted(n for n, r in rows.items() if components.get(n) is None and r.get("verdict") != "accept")
    verdict = got["verdict"] if not (got["verdict"] == "accept" and undecided) else "undecided"
    return {**out, "verdict": verdict, "reason": got.get("reason")}


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
            "composed": worst}


__all__ = ["MONTH_S", "NO_CASE", "PORTS", "PROTOCOL", "STUB_CASE", "bind", "component_of", "composed", "horizon",
           "onsite_cases", "onsite_check", "onsite_components", "onsite_inputs", "stub_component"]
