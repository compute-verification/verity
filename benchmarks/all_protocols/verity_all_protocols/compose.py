"""The composed verdict through onsite Verity (`core_onsite`, ``verity/onsite/v0``), and the network certifier's K
pooled over a monitoring period (`communication`, which `monitoring.monitoring` reports).

Onsite's check (verity/core/protocols/composition/README.md) first runs its registration check, NCI's
(`REGISTRATION`), on the registration's case in its commit phase: a registration it refuses refuses onsite. Then it runs
each component's own ``answer`` on that component's case and decides, reject first: any component's rejection rejects
onsite whatever the others answer; otherwise onsite accepts only when every component accepts and its commitment binds
their digests and the registration's. So the runner decides:

* **reject** as soon as the registration check or one leg's protocol rejects, or the registration check accepted a
  registration other than the one sampled proofs' leg committed to (`COMMITTER_LEAN`; one registration digest in both,
  `mismatch`): onsite rejects whatever the others would answer;
* **pending-ports** while onsite can't decide the rest: the registration check or a leg is missing or failed, it or a
  leg has no case for its port, or one of onsite's components is still a stub (`onsite_components` reads
  `Core.Onsite.components`), since a stub answers for the stub, not for the protocol;
* otherwise onsite's own executable decides on the cases (`onsite_check`, ``onsite check PARAMS COMMITMENT
  EVIDENCE [BLOB ...]``, the blobs the files sampled proofs' audit on the proofs reads), and its answer is the
  composed verdict, labelled with the policy the registration check's verdict is at (``policy``).

Given onsite's executable, the runner runs ``onsite check`` every period, whatever it decides: the registration check on
its leg's case, each port on its leg's case, each stub on a case with no statement yet (`STUB_CASE`, undecided), and a
port or registration check whose leg wrote no case on one its decoders refuse (`NO_CASE`). Its answer is ``answer``,
where each case came from is ``inputs`` (the registration's under its id), and ``agree`` says onsite rejected on the
registration's case if its check rejected it, on the registration's digest if it isn't sampled proofs' case's, and
otherwise on every leg whose own case rejected.

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
#: onsite's registration check (`Core.Onsite.protocol`'s, `onsite_registration`): NCI's check of record, its Lean name
REGISTRATION, REGISTRATION_LEAN = "verity/nci/v1", "NciCheck.Protocol.protocol"
#: the component whose commitment is the registration the registration check checks (`Core.Onsite.protocol`'s,
#: `onsite_registration`): sampled proofs' component, whose commitment is the run's registration record
COMMITTER_LEAN = "VStar.Recursive.Sampled.protocol"
#: the network certifier's period check, whose public carries K and eps
CERTIFIER = "verity/certifier/v0"
#: a proof unit's outputs are at most 16 bits (`NetworkCheck.Period`'s faulty bits, `16 · b`)
OUT_BITS = 16
#: what `horizon_leak`'s K rests on until its two per-window inputs are discharged
HORIZON_CONDITION = ("NetTiming.Guarantees.HorizonLeakAccepted (#4367) is conditional: its per-window inputs hesc "
                     "(proofs' SampledKSLive) and hleak (the live SampledLeakAccepted) are not discharged")


def onsite_components(source: str | None = None) -> list[str]:
    """`Core.Onsite.components` as written in ``CoreOnsite.lean``: each entry's Lean name (``Stub.pouw``, or a port's
    ``Pouw.Core.protocol``)."""
    text = ONSITE.read_text() if source is None else source
    m = re.search(r"def components : List (?:Blob)?Protocol :=\s*\[([^\]]*)\]", text)
    if m is None:
        raise ValueError("CoreOnsite.lean has no `def components : List BlobProtocol := [...]`")
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def onsite_registration(source: str | None = None) -> tuple[str, str]:
    """The Lean names of `Core.Onsite.protocol`'s registration check and of the component whose commitment is the
    registration, as written in ``CoreOnsite.lean`` (``def committer : String := S.id`` and ``registered
    "verity/onsite/v0" R committer``)."""
    text = ONSITE.read_text() if source is None else source
    c = re.search(r"def committer : String :=\s*(\S+)\.id\b", text)
    m = re.search(r"registered \"" + re.escape(PROTOCOL) + r"\" (\S+) committer\b", text)
    if c is None or m is None:
        raise ValueError(f"CoreOnsite.lean has no `def committer : String := S.id` and `registered \"{PROTOCOL}\" R "
                         "committer`")
    return m.group(1), c.group(1)


def mismatch(registration: str, committed: str | None) -> str:
    """Onsite's reason when the registration check checked the registration digest ``registration`` and sampled proofs'
    case committed to ``committed`` (`Core.Composition.mismatch`; hex, None when its case has no digest)."""
    return (f"registration: {REGISTRATION} checked {registration}, {PORTS[COMMITTER_LEAN][1]} committed "
            f"{committed or 'no digest'}")


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


def onsite_inputs(cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str],
                  registration: Mapping[str, Any], registration_digest: str) -> tuple[bytes, bytes, bytes]:
    """Onsite's params, commitment and evidence from each component's case and its commitment's digest (its answer's
    ``commitment``), both keyed by the component's protocol id, and the registration's case and digest: what ``onsite
    check`` reads (the composition README's table). The commitment is the binding of the components' digests, then the
    registration's digest."""
    ids = sorted(cases)

    def entry(i: str, c: Mapping[str, Any]) -> dict[str, Any]:
        return {"commitment": c["commitment"], "evidence": c["evidence"], "id": i, "params": c["params"]}

    params = canonical_json({"components": [{"id": i, "params": cases[i]["params"]} for i in ids],
                             "registration": {"id": REGISTRATION, "params": registration["params"]}})
    commitment = bind(PROTOCOL, [(i, digests[i]) for i in ids]) + bytes.fromhex(registration_digest)
    evidence = canonical_json({"components": [entry(i, cases[i]) for i in ids],
                               "registration": entry(REGISTRATION, registration)})
    return params, commitment, evidence


def onsite_check(exe: str, cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str],
                 registration: Mapping[str, Any], registration_digest: str) -> dict[str, Any]:
    """``onsite check`` on the components' cases and the registration's (`onsite_inputs`), with every blob file a case
    names (``blobs``, the files sampled proofs' audit on the proofs reads) after them: its answer line, parsed."""
    params, commitment, evidence = onsite_inputs(cases, digests, registration, registration_digest)
    blobs = [str(b) for i in sorted(cases) for b in cases[i].get("blobs") or ()]
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params.json", params), ("commitment.bin", commitment), ("evidence.json", evidence)):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([exe, "check", *files, *blobs], capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1, 2):
        raise RuntimeError(f"onsite check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


#: The ports `Core.Onsite.components` may name, by their Lean names: the leg each answers for and its protocol id.
PORTS = {
    "VStar.Recursive.Sampled.protocol": ("sampled-proofs", "verity/sampled-proofs/v1"),
    "SampledProofs.OneStageZk.protocol": ("sampled-proofs", "verity/sampled-proofs/one-stage/v2"),
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
             exe: str | None = None, registration: Mapping[str, Any] | None = None,
             registration_case: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The composed verdict (module docstring) over the period's rows (by leg name, each with its ``verdict``),
    ``components`` (leg name -> onsite component, or None), the legs' ``cases``, onsite's component list, and the
    registration check's row (missing when None) and its leg's case. With ``exe``, ``onsite check`` runs on every period
    (`onsite_cases`), and its answer is ``answer``; ``agree`` says whether it rejected on the registration's case if the
    registration check rejected it, and otherwise on each leg whose own case rejected."""
    onsite = onsite_components() if onsite is None else list(onsite)
    reg = registration or {"verdict": "missing"}
    stubs = sorted(c for c in (stub_component(x) for x in onsite) if c is not None)
    out: dict[str, Any] = {"protocol": PROTOCOL, "registration": REGISTRATION, "onsite_components": onsite,
                           "stubs": stubs, "outside_onsite": sorted(n for n, c in components.items() if c is None),
                           **({"policy": reg["policy"]["label"]} if isinstance(reg.get("policy"), dict) else {})}
    component, committer = PORTS[COMMITTER_LEAN]
    got = None
    if exe is not None:
        cs, digests, source, legs = onsite_cases(rows, components, cases, onsite)
        if registration_case is not None:
            rcase, rdigest, source[REGISTRATION] = registration_case, reg.get("commitment") or "", "leg"
        else:
            rcase, rdigest = NO_CASE, ""
            source[REGISTRATION] = f"no case (registration {reg.get('verdict', 'missing')})"
        got = onsite_check(exe, cs, digests, rcase, rdigest)
        reason = got.get("reason") or ""
        if reg.get("verdict") == "reject":
            agree = got["verdict"] == "reject" and reason.startswith(f"registration: {REGISTRATION} reject: ")
        elif (reg.get("verdict") == "accept" and registration_case is not None and committer in digests
              and digests[committer] != rdigest):
            agree = got["verdict"] == "reject" and reason == mismatch(rdigest, digests[committer] or None)
        else:
            agree = all(got["verdict"] == "reject" and f"{pid} reject: " in reason
                        for pid, n in legs.items() if rows[n].get("verdict") == "reject")
        out.update(inputs=source, answer=got, agree=agree)
    rejected = sorted(n for n, r in rows.items() if r.get("verdict") == "reject")
    leg = next((n for n, c in components.items() if c == component), None)
    committed = (rows.get(leg) or {}).get("commitment") if leg is not None else None
    if reg.get("verdict") == "accept" and reg.get("commitment") and committed and committed != reg["commitment"]:
        rejected = [f"the registration, which {leg}'s leg did not commit to", *rejected]
    if reg.get("verdict") == "reject":
        rejected = ["the registration check", *rejected]
    if rejected:
        return {**out, "verdict": "reject", "reason": "rejected by " + ", ".join(rejected) + " (reject dominates)"}
    waiting = []
    if reg.get("verdict") in ("missing", "error"):
        waiting.append(f"the registration check {reg['verdict']}")
    elif registration_case is None:
        waiting.append("the registration check has no case")
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


def fleet(periods: Sequence[Mapping[str, Any]], delta: Fraction, count: int, machines: int) -> dict[str, Any]:
    """``machines`` machines, each its own network certifier over ``count`` periods like ``periods``, joined by a union
    bound: each machine's `horizon_leak` at ``delta / machines``, so the K add and the δ add back to ``delta``. The
    union bound across machines is not in Lean."""
    one = horizon_leak(periods, delta / machines, count)
    return {"machines": machines, "delta_each": str(delta / machines), "K": machines * one["K"], "share": one["share"],
            "units": machines * one["units"], "Fmax_each": one["Fmax"], "eps": f"{delta} + Σ over machines and windows "
            "of (e_i + e'_i)", "lean": "per machine; the union bound over machines is not in Lean",
            "conditional": HORIZON_CONDITION}


def communication(reports: Sequence[Mapping[str, Any]], period_s: float, spans: Mapping[str, float],
                  deltas: Sequence[Fraction], machines: int) -> dict[str, Any]:
    """The network certifier over the run's windows so far and over each of ``spans`` of one machine's ``period_s``
    windows, at each of ``deltas``: the pooled K first (`horizon_leak`, one machine; `fleet`, ``machines`` of them), and
    beside it the per-window figure, each accepted window's own ``K`` at its ``eps``
    (`NetTiming.Guarantees.PeriodLeakAccepted`) with its share of the window's output bits. A window that didn't accept
    certifies nothing, and counts in ``refused``, as does another protocol's answer. `monitoring.monitoring` passes its
    monitoring periods and δs and the census deployment's window length and machines."""
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


__all__ = ["CERTIFIER", "COMMITTER_LEAN", "HORIZON_CONDITION", "NO_CASE", "OUT_BITS", "PORTS", "PROTOCOL",
           "REGISTRATION", "REGISTRATION_LEAN", "STUB_CASE", "bind", "communication", "component_of", "composed",
           "fleet", "horizon_leak", "mismatch", "onsite_cases", "onsite_check", "onsite_components", "onsite_inputs",
           "onsite_registration", "stub_component"]
