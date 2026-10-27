"""The composed verdict through onsite Verity (`core_onsite`, ``verity/onsite/v0``), and the network certifier's K
pooled over a monitoring period (`communication`, which `monitoring.monitoring` reports).

Onsite's check (verity/core/protocols/composition/README.md) runs each component's own ``answer`` on that component's
case and decides, reject first: any component's rejection rejects onsite whatever the others answer; otherwise onsite
accepts only when every component accepts and its commitment binds their digests. So the runner decides:

* **reject** as soon as one leg's protocol rejects: by reject dominance onsite rejects whatever the others would answer;
* **pending-ports** while onsite can't decide the rest: a leg is missing or failed, a leg has no case for its port, or one
  of onsite's components is still a stub (`onsite_components` reads `Core.Onsite.components`), since a stub answers for
  the stub, not for the protocol;
* otherwise onsite's own executable decides on the legs' cases (`onsite_check`, ``onsite check PARAMS COMMITMENT
  EVIDENCE [BLOB ...]``, the blobs the files integrity's audit on the proofs reads), and its answer is the composed
  verdict.

Onsite is a registered composition (`Core.Composition.registeredB`): before its components it runs NCI's check
(``verity/nci/v1``, `NciCheck.Protocol`) on the run's registration, the record integrity's case commits to, so its inputs
carry NCI's case beside the components' (`registration_case`: the deployment's NCI declaration, ``period.json``'s
``nci``, on that record and its log) and its commitment ends with that record's digest. A period whose ``period.json``
states no declaration is **pending-ports** too, and onsite runs on `NO_CASE` as the registration, which NCI refuses.

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

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits.codec import canonical_json

ROOT = Path(__file__).resolve().parents[3]
ONSITE = ROOT / "verity" / "core" / "protocols" / "onsite" / "CoreOnsite.lean"
PROTOCOL = "verity/onsite/v0"
#: the network certifier's period check, whose public carries K and eps
CERTIFIER = "verity/certifier/v0"
#: a proof block's outputs are at most 16 bits (`NetworkCheck.Period`'s faulty bits, `16 · b`)
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


def registration_case(declaration: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    """NCI's case for onsite's registration (`NciCheck.Protocol`'s module docstring): the deployment's NCI declaration
    (`NCI_PARAMS` and `NCI_EVIDENCE`) on the run's own registration record, the commitment of integrity's case, and the
    log before it, that case's params' ``log``: one registration in both."""
    p = integrity.get("params")
    log = p.get("log", []) if isinstance(p, Mapping) else []
    return {"commitment": integrity["commitment"],
            "evidence": canonical_json({k: declaration[k] for k in NCI_EVIDENCE}).hex(),
            "params": {**{k: declaration[k] for k in NCI_PARAMS}, "log": log}}


def registration_digest(case: Mapping[str, Any]) -> str:
    """The digest of the registration record a case commits to (`NciCheck.Baseline.digest`, `registration.digest`), or
    ``""`` when its commitment is no record, which onsite then refuses."""
    try:
        record = json.loads(bytes.fromhex(case["commitment"]))
    except (KeyError, TypeError, ValueError):
        return ""
    return R.digest(record) if isinstance(record, dict) else ""


def onsite_inputs(cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str], protocol: str = PROTOCOL,
                  registration: Mapping[str, Any] | None = None) -> tuple[bytes, bytes, bytes]:
    """The product ``protocol``'s params, commitment and evidence from each component's case and its commitment's digest
    (its answer's ``commitment``), both keyed by the component's protocol id: what ``product IDS check`` reads (the
    composition README's table). With NCI's ``registration`` case, the registered composition's, what ``onsite check``
    reads: the registration's entry and case beside the components', and its digest after their binding."""
    ids = sorted(cases)
    params: dict[str, Any] = {"components": [{"id": i, "params": cases[i]["params"]} for i in ids]}
    commitment = bind(protocol, [(i, digests[i]) for i in ids])
    evidence: dict[str, Any] = {"components": [{"commitment": cases[i]["commitment"], "evidence": cases[i]["evidence"],
                                                "id": i, "params": cases[i]["params"]} for i in ids]}
    if registration is not None:
        params["registration"] = {"id": NCI, "params": registration["params"]}
        evidence["registration"] = {"commitment": registration["commitment"], "evidence": registration["evidence"],
                                    "id": NCI, "params": registration["params"]}
        commitment += bytes.fromhex(registration_digest(registration))
    return canonical_json(params), commitment, canonical_json(evidence)


def onsite_check(exe: str | Sequence[str], cases: Mapping[str, Mapping[str, Any]], digests: Mapping[str, str],
                 protocol: str = PROTOCOL, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``onsite check`` (with NCI's ``registration`` case) or ``product IDS check`` (``exe`` as an argv, with
    ``protocol`` its product's id) on the components' cases (`onsite_inputs`), with every blob file a case names
    (``blobs``, the files integrity's audit on the proofs reads) after them: its answer line, parsed."""
    params, commitment, evidence = onsite_inputs(cases, digests, protocol, registration)
    blobs = [str(b) for i in sorted(cases) for b in cases[i].get("blobs") or ()]
    argv = [exe] if isinstance(exe, str) else list(exe)
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params.json", params), ("commitment.bin", commitment), ("evidence.json", evidence)):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([*argv, "check", *files, *blobs], capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1, 2):
        raise RuntimeError(f"{' '.join(argv)} check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


#: The ports `Core.Onsite.components` may name, by their Lean names: the leg each answers for and its protocol id.
PORTS = {
    "SampledProofs.OneStageZk.protocol": ("sampled-proofs", "verity/sampled-proofs/one-stage/v2"),
    "PouwCheck.protocol": ("pouw", "verity/pouw/v1"),
    "PouwCheck.Period.protocol": ("pouw", "verity/pouw/period/v1"),
    "NetworkCheck.Check.protocol": ("certifier", "verity/network-certificate/v4"),
    "NetworkCheck.Period.protocol": ("certifier", "verity/certifier/v0"),
    "PousCheck.RunProtocol.protocol": ("pous", "verity/pous/p2/v4/run"),
    "RemoteAudit.Protocol.protocol": ("bandwidth", "verity/remote-bandwidth/v0"),
    "MemoryChallenger.WipeProtocol.protocol": ("wipe", "verity/wipe/v0"),
}
#: A stub's case with no statement yet (`Core.Example.protocol`: none of its one unit drawn), so it is undecided: a stub
#: never accepts or rejects for the protocol it stands for. Its 32-byte root is its own digest.
STUB_CASE = {"commitment": "00" * 32, "evidence": canonical_json({"drawn": 0, "passed": 0}).hex(), "params": {"k": 1}}
#: The case of a port whose leg wrote none: its own decoders refuse it, so that component rejects.
NO_CASE = {"commitment": "", "evidence": "", "params": {}}
#: Onsite's registration check, and the component whose commitment is the run's registration record.
NCI = "verity/nci/v1"
COMMITTER = PORTS["SampledProofs.OneStageZk.protocol"][1]
#: A deployment's NCI declaration (``period.json``'s ``nci``): NCI's params but the log, and its evidence.
NCI_PARAMS = ("alpha", "gamma", "parameters", "program")
NCI_EVIDENCE = ("owner", "provenance", "theta")


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
             exe: str | None = None, declaration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The composed verdict (module docstring) over the period's rows (by leg name, each with its ``verdict``),
    ``components`` (leg name -> onsite component, or None), the legs' ``cases``, onsite's component list and the
    deployment's NCI ``declaration``. With ``exe``, ``onsite check`` runs on every period (`onsite_cases`, and NCI's case
    on integrity's, `registration_case`, else `NO_CASE`), and its answer is ``answer``; ``agree`` says whether it
    rejected on each leg whose own case rejected."""
    onsite = onsite_components() if onsite is None else list(onsite)
    stubs = sorted(c for c in (stub_component(x) for x in onsite) if c is not None)
    out: dict[str, Any] = {"protocol": PROTOCOL, "onsite_components": onsite, "stubs": stubs,
                           "outside_onsite": sorted(n for n, c in components.items() if c is None)}
    got = None
    if exe is not None:
        cs, digests, source, legs = onsite_cases(rows, components, cases, onsite)
        committed = source.get(COMMITTER) == "leg"
        registration = registration_case(declaration, cs[COMMITTER]) if declaration and committed else NO_CASE
        source[NCI] = "declaration" if registration is not NO_CASE else \
            "no case (" + ("no NCI declaration" if not declaration else f"{COMMITTER} has no case") + ")"
        got = onsite_check(exe, cs, digests, registration=registration)
        mine = [pid for pid, n in legs.items() if rows[n].get("verdict") == "reject"]
        out.update(inputs=source, answer=got, agree=all(
            got["verdict"] == "reject" and f"{pid} reject: " in (got.get("reason") or "") for pid in mine))
    rejected = sorted(n for n, r in rows.items() if r.get("verdict") == "reject")
    if rejected:
        return {**out, "verdict": "reject", "reason": "rejected by " + ", ".join(rejected) + " (reject dominates)"}
    waiting = []
    if stubs:
        waiting.append("onsite's stubs: " + ", ".join(stubs))
    if not declaration:
        waiting.append("onsite's registration: period.json states no NCI declaration (nci)")
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
    coins (`NetTiming.Guarantees.HorizonLeakAccepted`, #4367, on `HorizonWrongUnitsLe`, #4355): the faulty proof blocks are
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


__all__ = ["CERTIFIER", "COMMITTER", "HORIZON_CONDITION", "NCI", "NCI_EVIDENCE", "NCI_PARAMS", "NO_CASE", "OUT_BITS",
           "PORTS", "PROTOCOL", "STUB_CASE", "bind", "communication", "component_of", "composed", "fleet",
           "horizon_leak", "onsite_cases", "onsite_check", "onsite_components", "onsite_inputs", "registration_case",
           "registration_digest", "stub_component"]
