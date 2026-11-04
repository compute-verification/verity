"""kind-scan: each field the network certifier's piece releases to the verifier, by kind. A field is one of the four
public kinds (`verity.core.wire.PUBLIC`: protocol structure, the protocol's outputs, the verifier's coins and draws,
hiding commitments) or what the network certifier signed (`CERTIFICATE`: its certificates reach the verifier in the
clear from the auditor's device, a provisional exception to the four kinds, not a fifth). A field of no declared kind is
a finding.

    python benchmarks/network_traces/kind_scan.py DIR
    python benchmarks/network_traces/kind_scan.py --received OUT

DIR is a piece's DIR, or its hold before the leak guard releases it (`certifier_piece.settle` runs the scan there on
every served run and refuses on a finding, releasing nothing): `DOCS`' files, the case as DIR/case.json (its commitment
and evidence hex, `certifier_piece.write_case`, as the composite reads it) or as the three files ``network-period`` reads
(DIR/period). Each file's JSON is walked to its leaves, the commitment's and evidence's bytes decoded as JSON, and each
leaf's key path (a list's items ``[]``) must match one its file declares: ``<name>`` is any one key its source gives
(`SOURCES`: a template the answer's profile draws by, a δ the chain's report bounds at), a key from anywhere else a
finding naming the source in its place, never the key; ``*`` is any one key, in the tables no source names yet; and
``**`` everything below, a Program's descriptor (`DESCRIPTOR`, the circuit, which is structure) or a development check's
field. A list's length and an object's keys are the shape of the leaves they hold. So a field no declaration names, one
a change adds included, is a finding until its kind is declared here: a window's own number, a count of its requests,
the advice's length.

A descriptor is structure only as the Program the verifier fixes, so DIR's Programs are held to it (`pinned`): the
parameters' ``program``, the scan's descriptor (``scan``) and each scan registration's Program (the commitment's windows'
and the scan's answers') are the scan Program pinned at the parameters' schedule's digest (`scan_service.SCAN_PINS`;
where DIR holds no parameters, some class's), the descriptor with no ``annotations``, which no digest covers; and each
audit's descriptor and registration are the class circuit, the parameters' ``circuit``. V3c's own copy of its audit's
Program (``v3c-audit/period.json``) is not held here.

The record path's two development checks are findings wherever they reach the verifier (`Development`, each naming its
check): an advice binding in a registration (``window.compiler``, ``window.advice``: the compiler and the advice's root and
length an audit opens, `records_audit.py audit --opening`), since no audit of record reads an advice (V3c's class
window's circuit is the class's) and no registration of the record path binds one to be opened; and the chain's
endpoint rebuild (the report's ``endpoint`` step, chain_record.sh's CR_ENDPOINT on the endpoint's records and advice in
the clear), which the active-warden path refuses. The report's steps are the active-warden path's (`STEPS`). A text field (a verdict's reason, a
step's account of what ran in the clear) is declared by its key, its text not read, and so is a text file (`TEXT`, a
refusal). Any other file is a finding, a log included: the steps' logs stay on the node (`served_verdicts.sh`'s
CERTIFIER_PRIVATE).

``--received OUT`` scans what a verifier on another machine reads of a served run's OUT (`RECEIVED`, the files
`served_verdicts.RECEIVED` names, which `served_verdicts.publish` leaves as they are only once this scan is clean): the
certifier's piece (``pieces/certifier``, `DOCS`) and V3c's audit (``v3c-audit``, `V3C_DOCS`, what `served_integrity.answer`
reads of it and the period's compute audit), each whole, so a file in either that no declaration covers is a finding;
and the window circuit's inputs and count, each the verifier's: the registered compiler (`COMPILER`), the advice
registration (`ADVICE_REGISTRATION`, its length held to the cap: `padded`), the weights registration (`WEIGHTS`) and each
class's count (`COUNT`). None of them reads the advice: an advice binding, a compile, an unpadded length or a φ that
carries a workload is a finding.

Prints each finding (its file, key path and the value's type, never the value) and a count of leaves by kind; exit 0
with no finding, 1 with one.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from verity.core import wire as W
from verity.core.protocols.certifier.scan_service import SCAN_PINS
from verity.core.protocols.certifier.warden import WardenParams
from verity.ml.circuits.codec import program_digest

CERTIFICATE = "certificate"
KINDS = (*(k for k in W.KINDS if k in W.PUBLIC), CERTIFICATE)
S, O, C, M, X = (*KINDS,)


@dataclasses.dataclass(frozen=True)
class Development:
    """A field only a development check writes: a finding wherever it is, ``why`` naming the check."""

    why: str


@dataclasses.dataclass(frozen=True)
class Sourced:
    """A table's ``<name>`` in its trie: no key of a document equals it, so a key matches it only by its source
    (`SOURCES`)."""

    name: str


@dataclasses.dataclass(frozen=True)
class Unsourced:
    """A key only a ``<name>`` declares, which its source doesn't give: a finding at ``<name>``, never the key, ``why``
    naming the source."""

    why: str


OPENING = Development("an advice binding, which an audit opens (records_audit.py audit --opening, a development check): "
                      "no audit of record reads an advice or its length (V3c's class window's circuit is the class's), "
                      "and nothing opens one")
ENDPOINT = Development("the chain's endpoint rebuild (CR_ENDPOINT: the endpoint's records and advice, in the clear), a "
                       "development check the active-warden path refuses")
#: the chain's steps on the active-warden path (`chain_record.report`)
STEPS = ("custody", "certificates", "n2", "scan", "lean")
#: the δs the chain's report bounds at, as `chain_record.eps_at_delta` keys them (`chain_record.DELTAS`)
DELTA_KEYS = frozenset({"0.5", "9.54e-07"})
#: each ``<name>`` key's source (module docstring): a fixed set, or the strings at a key path of the nearest object
#: enclosing the key's own that holds the path's first field; and why a key from elsewhere is a finding
SOURCES: dict[str, tuple[frozenset[str] | str, str]] = {
    "template": ("profile.levels[].draw.strata[].name",
                 "not a template the answer's profile draws by (profile.levels[].draw.strata[].name)"),
    "delta": (DELTA_KEYS, "not a δ the chain's report bounds at (chain_record.DELTAS)"),
}


def under(prefix: str, table: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in table.items()}


#: a Program's descriptor (``verity-ir/descriptor/v1``): the circuit, every field of it structure, as the Program the
#: verifier pins (`pinned`)
DESCRIPTOR = {"**": S}
#: a window's circuit record (``verity/window-circuit/v1``) and its wires' order (``verity/wire-order/v1``): the circuit's
#: description, every field of it structure
RECORD = {"**": S}
#: a registered value's entry (`registered.Registered.entry`)
ENTRY = {"value": S, "root": M, "leaves": S, "words": S, "word_bits": S, "domain": S}
#: a one-stage registration (`one_stage.registration.record`): the digests and population of the Program and query, the
#: law the verifier's sampling design over them, the roots and registered values hiding commitments under domains of
#: their structure; its window the provenance: a scan round's account and members, its state's reads (`stateRead`'s
#: runs, ``sin`` and ``o1``) and row 0's opening (`stateEmpty`: the empty state, which the protocol fixes, its width and
#: salt structure and its path's siblings hiding commitments; no unit reads leaf 0), each under the state's port
#: ``scan/state`` (`stateName`, the one key the verifier reads them at), or a committed window's run root, circuit
#: record, class, per-window values, registered roots and index leaf. The window's own number is none of them, and the
#: advice's binding (`interface.advice_binding`) is `OPENING`'s.
REGISTRATION = {
    "format": S, "protocol": S, "program": S, "query": S, "partition": S, "population": S, "scheme": S, "leaf_layer": S,
    "anchors[]": S, "previous": M,
    "law.law": S, "law.k": S, "law.p": S, "law.strata[].template": S, "law.strata[].k": S,
    "law.strata[].units[][]": S, "law.strata[].work": S, "law.strata[].floor": S,
    "roots[].port": S, "roots[].leaves": S, "roots[].domain": S, "roots[].root": M,
    **under("registered[]", ENTRY),
    "window.kind": S, "window.link": S, "window.of": S, "window.members.*.template": S, "window.members.*.units[]": S,
    **{f"window.reads.scan/state.{r}.{k}": S for r in ("sin", "o1") for k in ("leaf", "units[]")},
    "window.empty.scan/state.words": S, "window.empty.scan/state.salt": S, "window.empty.scan/state.path": M,
    "window.run": M, "window.circuit": S, "window.order": S, "window.index_leaf": M, "window.registered_roots.*": M,
    "window.window_circuit.class[]": S, "window.activations.program": S,
    **under("window.activations.values.*", ENTRY),
    "window.compiler.**": OPENING, "window.advice.**": OPENING,
}
#: a one-stage answer (`served_integrity.answer`, `Answer.toJson`): its verdict and bound the protocol's outputs, its
#: profile the integrity profile's structure (its one level ``unit``, `one_stage.audit.LEVEL`), and the units drawn
#: and proved the verifier's draws
ONE_STAGE = {
    "protocol": S, "verdict": O, "reason": O, "commitment": M,
    "profile": O, "profile.programs[]": S, "profile.prescribed[]": S, "profile.covered[]": S, "profile.linked": S,
    "profile.delta": S, "profile.drawn.unit": S,
    "profile.levels[].name": S, "profile.levels[].query": S, "profile.levels[].size": S, "profile.levels[].draw": S,
    **{f"profile.levels[].draw.{k}": S for k in ("law", "k", "subset", "bernoulli")},
    **{f"profile.levels[].draw.strata[].{k}": S for k in ("name", "size", "k", "work", "floor")},
    "public": O, "public.bound": O, "public.per_template": O, "public.per_template.<template>": O, "public.drawn[]": C,
    "public.proved[]": C,
}
#: a row registration of the network certifier's (`warden_rows`): the registered layout and schedule's digest, its root a
#: hiding commitment, and the certificate the network certifier signed over it
ROWS = {
    **{k: S for k in ("cell_bytes", "direction", "domain", "format", "link", "parameters", "row_schema", "rows", "value",
                      "window", "wiring", "word_bits", "words")},
    **{f"layout.{k}": S for k in ("buckets", "connection_slots", "first_bucket", "slots")},
    "root": M, "certificate.digest": X, "certificate.key": X, "certificate.signature": X,
}
#: a record's certificate (`scan_period.CERTIFICATE_KEYS`), the network certifier's in the clear
CERTIFICATE_DOC = dict.fromkeys(("digest", "direction", "key_id", "link", "signature", "window"), X)
#: `NetworkCheck.Period`'s rows form: the commitment, the evidence and the parameters (`scan_period.rows_case`)
COMMITMENT = {**under("audits[]", REGISTRATION), **under("windows[]", REGISTRATION)}
EVIDENCE = {**under("audits[]", ONE_STAGE), **under("windows[].answer", ONE_STAGE),
            **under("windows[].certificates[]", CERTIFICATE_DOC), **under("windows[].registrations[]", ROWS)}
#: the parameters, each structure: a connection's cap (``cells``) is the registered schedule's (`WardenParams.caps`)
PARAMS = {
    "audits[].free": S, "audits[].steps": S, **under("audits[].program", DESCRIPTOR), **under("scan", DESCRIPTOR),
    **under("audits[].circuit", RECORD), **under("audits[].order", RECORD), "audits[].pins.*": M,
    "cells[]": S, "circuit": S, "first": S, "jitter": S, "key": S, "link": S, "program": S, "windows": S,
    **{f"lms.{k}": S for k in ("H", "N", "W", "toy")},
    **{f"schedule.{f.name}": S for f in dataclasses.fields(WardenParams)},
    "session": C,
}
#: ``verity/certifier/v0``'s answer (`Answer.toJson`) with the piece's placement: K, ε and their terms the protocol's
#: outputs, the profile where each party ran
ANSWER = {
    "protocol": S, "verdict": O, "reason": O, "commitment": M, "profile": O, "public": O,
    "profile.link": S, "profile.windows": S,
    **{f"profile.placement.{k}": S for k in ("kind", "trusted-host", "uncounted")},
    **{f"profile.placement.parties.{p}": S for p in ("network_certifier", "prover", "verifier")},
    **{f"profile.placement.links[].{k}": S for k in ("name", "from", "to", "counted", "via")},
    **{f"public.{k}": O for k in ("K", "eps", "cells", "schedules", "windows")}, "public.held[]": O,
    "public.scans[].tied[]": O,
    **{f"public.{f}[].{k}": O for f in ("audits", "scans")
       for k in ("delta", "drawn", "faulty_bits", "faulty_entries", "faulty_sets", "free", "jitter", "units",
                 "wrong_units")},
}
#: a step of the chain's report: its verdict, the draw's law, population and size with its bound at each δ, and what
#: it ran in the clear
STEP = {"verdict": O, "in_the_clear": S, "eps_at_delta": O,
        **{f"eps_at_delta.{k}": S for k in ("law", "units", "drawn", "per_proof")},
        "eps_at_delta.incorrect_at_delta.<delta>": O}
#: the chain's report less what `certifier_piece.PUBLIC` drops: its verdict, each `STEPS` step's and what the path stubs
REPORT = {
    "schema": S, "verdict": O, "path": S, "scope": S, "stubbed[]": S, "negatives_refused.*": O,
    **{k: v for s in STEPS for k, v in under(f"steps.{s}", STEP).items()},
    "steps.endpoint.**": ENDPOINT,
}
#: the scan's answers (`scan_zk.answers`) less what `certifier_piece.PUBLIC` drops: each execution phase's answer and the
#: registrations it is checked against
SCAN_ANSWERS = {"format": S, "delta": S, **under("registration", REGISTRATION), "windows[].window": S,
                **under("windows[].answer", ONE_STAGE), **under("windows[].registrations[]", ROWS)}
#: DIR's files, each by its declaration: the case's two hex fields (`HEX`) are walked as the JSON their bytes are
DOCS: dict[str, dict[str, Any]] = {
    "answer.json": ANSWER,
    "chain.json": {"chain": O},
    "chain/report/report.json": REPORT,
    "chain/scan/answers.json": SCAN_ANSWERS,
    "case.json": {**under("commitment", COMMITMENT), **under("evidence", EVIDENCE), **under("params", PARAMS)},
    "period/commitment.bin": COMMITMENT,
    "period/evidence.json": EVIDENCE,
    "period/params.json": PARAMS,
    "piece.json": {"server": S, "rc": O, "limit": S},
}
HEX = {"case.json": ("commitment", "evidence")}
#: text files, each declared whole, its text not read: a piece's refusal (`certifier_piece.REFUSED`), the protocol's
#: output
TEXT = {"refused.txt": O}

#: what `served_integrity.answer` reads of a records audit's ``summary.json`` (`served_integrity.SUMMARY`): its refusal,
#: its checks before the draw and after it, and the audit's verdict, its coins' outcome and Lean's refusals, each the
#: protocol's output. An audit that compiled the advice (``compile``) opened it: `OPENING`'s
SUMMARY_DOC = {
    "refused.**": O, "checks_before_draw": O, "checks_before_draw.*": O,
    "fixed_before_draw": O, "fixed_before_draw.ok": O, "fixed_before_draw.codes[]": O,
    "audit": O, "audit.accepted": O, "audit.outcome.**": O, "audit.why.**": O,
    "compile.**": OPENING,
}
#: what it reads of the audit record ``window/audit.json`` (`served_integrity.RECORD`): its verdict and each check's
#: outcome, the units drawn and those the proved statement covers, the draw and the units proved (the verifier's draws),
#: the registration it audited, and the bound on incorrect proof blocks over the population at δ
AUDIT_RECORD = {
    "accepted": O, "checks.*": O, "checks.*.ok": O, "checks.*.codes[]": O,
    "checks.proved.detail.units.drawn[]": C, "checks.proved.detail.units.statement[]": C,
    "draw.**": C, "proved_units[]": C, **under("registration", REGISTRATION),
    "wrong_units": O, "wrong_units.delta": S, "wrong_units.of": S, "wrong_units.rule": S, "wrong_units.bound": O,
    "wrong_units.per_template.*": O,
}
#: V3c's audit as one compute audit of the network certifier's period (`records_audit.period`): its answer, its free
#: input bits and steps, the Program and the registration
PERIOD = {**under("answer", ONE_STAGE), "free": S, "steps": S, **under("program", DESCRIPTOR),
          **under("registration", REGISTRATION)}
#: V3c's audit as the run publishes it (``v3c-audit``: `served_integrity.public_copy`, the registration it audited and
#: its period audit)
V3C_DOCS: dict[str, dict[str, Any]] = {"summary.json": SUMMARY_DOC, "window/audit.json": AUDIT_RECORD,
                                       "registration.json": REGISTRATION, "period.json": PERIOD}
#: the run's registered compiler (`prove_path.compiler_doc`): its name, digest and cap, the compiler it wraps and the
#: server's code, its model's transfer gates, the default seed and φ (the model, target, TP, sampling, engine and
#: execution), all structure. φ's workload is the requests', which a registered φ never carries: nothing under it is
#: declared
COMPILER = {
    "name": S, "digest": S, "cap_bits": S, "inner.name": S, "inner.digest": S, "served.name": S, "served.digest": S,
    "transfer_gates[]": S, "default_seed": S, "phi.schema": S, "phi.execution": S, "phi.profile": S,
    **{f"phi.{k}.**": S for k in ("model", "target", "tp", "sampling", "engine")},
}
#: a window's advice registration as the run publishes it (`served_verdicts.public_registration`): its binding's compiler
#: and the registered compiler, structure; the advice's root and each weight's root (`registered_roots`), hiding
#: commitments; the advice's length and rows, which `padded` holds to the cap; and the window commit it names
ADVICE_REGISTRATION = {
    "schema": S, "scheme": S, "binding.compiler.name": S, "binding.compiler.digest": S, "binding.advice.root": M,
    "binding.advice.bits": S, **{f"compiler.{k}": S for k in ("name", "digest", "cap_bits", "inner.name", "inner.digest")},
    "rows": S, "row_bits": S, "T": S, "commit": S, "commit.run_id": S, "commit.run_root": M, "registered_roots.*": M,
    "padded[]": S,
}
#: the committer's weights registration (`window_weights.weights_registration`): the Program and model it registers
#: under, each weight's entry (its root a hiding commitment) and the parameters it leaves unrooted, with why
WEIGHTS = {"schema": S, "program": S, "model.repo": S, "model.revision": S, "grain.**": S, **under("values.*", ENTRY),
           "unrooted.*": S}
#: the window circuit's input count at a class (`prove_path.window_circuit_of_class`), the class circuit's structure
COUNT = {"count.*": S, "seconds.*": S}
#: what a verifier on another machine reads of a served run's OUT, by path (`served_verdicts.RECEIVED`): each of
#: `RECEIVED_DIRS` whole by its declarations, and each other path (a glob) by its one
RECEIVED_DIRS: dict[str, dict[str, dict[str, Any]]] = {"pieces/certifier": DOCS, "v3c-audit": V3C_DOCS}
RECEIVED: dict[str, dict[str, Any]] = {"compiler.json": COMPILER, "registration.json": ADVICE_REGISTRATION,
                                       "weights_registration.json": WEIGHTS, "window-circuit-*.json": COUNT}

_KIND = None


@dataclasses.dataclass(frozen=True)
class Finding:
    file: str
    path: str
    why: str

    def __str__(self) -> str:
        return f"{self.file} {self.path or '(the file)'}: {self.why}"


def _segments(pattern: str) -> list[str]:
    out = []
    for part in pattern.split("."):
        name = part.rstrip("[]")
        out += [name] * bool(name) + ["[]"] * ((len(part) - len(name)) // 2)
    return out


def _trie(table: Mapping[str, Any]) -> dict:
    root: dict = {}
    for pattern, kind in table.items():
        node = root
        for seg in _segments(pattern):
            node = node.setdefault(Sourced(seg[1:-1]) if seg[:1] == "<" else seg, {})
        node[_KIND] = kind
    return root


def _at(value: Any, segs: list[str]) -> Iterator[Any]:
    """The values at the key path ``segs`` (`_segments`) under ``value``."""
    if not segs:
        yield value
    elif segs[0] == "[]":
        for v in value if isinstance(value, list) else ():
            yield from _at(v, segs[1:])
    elif isinstance(value, dict) and segs[0] in value:
        yield from _at(value[segs[0]], segs[1:])


def _source(name: str, up: tuple[dict, ...]) -> frozenset[str]:
    """The keys ``<name>`` takes in an object the objects ``up`` enclose, outermost first (`SOURCES`)."""
    at = SOURCES[name][0]
    if not isinstance(at, str):
        return at
    segs = _segments(at)
    return frozenset(v for v in _at(next((o for o in reversed(up) if segs[0] in o), None), segs) if isinstance(v, str))


def _name(path: tuple[str, ...]) -> str:
    return "".join(s if s == "[]" or not i else f".{s}" for i, s in enumerate(path))


def _leaves(value: Any, nodes: list[dict], path: tuple[str, ...] = (), up: tuple[dict, ...] = ()
            ) -> Iterator[tuple[tuple[str, ...], list, Any]]:
    """Each leaf of ``value`` (a scalar, or a subtree a ``**`` declares) with the kinds the trie nodes ``nodes`` declare at
    its key path, ``up`` the objects enclosing ``value``, outermost first. A key only a ``<name>`` declares, which its
    source doesn't give, is one leaf at ``<name>``, declared `Unsourced`, and nothing under it is read."""
    whole = [n["**"][_KIND] for n in nodes if "**" in n]
    if whole:
        yield path, whole, None
    elif isinstance(value, dict):
        named = [(w.name, c) for n in nodes for w, c in n.items() if isinstance(w, Sourced)]
        sources = {w: _source(w, up) for w, _ in named}
        for k, v in value.items():
            kids = [c for n in nodes for c in (n.get(k), n.get("*")) if c is not None]
            kept = [c for w, c in named if kids or k in sources[w]]
            if named and not kids and not kept:
                yield (*path, f"<{named[0][0]}>"), [Unsourced(SOURCES[named[0][0]][1])], None
            else:
                yield from _leaves(v, kids + kept, (*path, k), (*up, value))
    elif isinstance(value, list):
        inner = [n["[]"] for n in nodes if "[]" in n]
        for v in value:
            yield from _leaves(v, inner, (*path, "[]"), up)
    else:
        yield path, [n[_KIND] for n in nodes if _KIND in n], value


def scan_doc(rel: str, doc: Any, kinds: Counter, docs: Mapping[str, Mapping[str, Any]] = DOCS) -> list[Finding]:
    """``doc``'s findings, as the file ``rel`` of a dir ``docs`` declares (`DOCS`, a piece's DIR), each leaf counted in
    ``kinds`` by its kind."""
    for k in HEX.get(rel, ()) if docs is DOCS else ():
        if isinstance(doc, dict) and isinstance(doc.get(k), str):
            try:
                doc = {**doc, k: json.loads(bytes.fromhex(doc[k]))}
            except ValueError as e:
                return [Finding(rel, k, f"isn't hex of JSON ({type(e).__name__})")]
    out = []
    for path, declared, value in _leaves(doc, [_trie(docs[rel])]):
        if not declared:
            out.append(Finding(rel, _name(path), f"a {type(value).__name__} of no declared kind"))
        elif len(set(declared)) > 1:
            out.append(Finding(rel, _name(path), f"declared as {sorted(map(str, declared))}, not one kind"))
        elif isinstance(declared[0], (Development, Unsourced)):
            out.append(Finding(rel, _name(path), declared[0].why))
        else:
            kinds[declared[0]] += 1
    return out


#: why `pinned` finds a Program the verifier doesn't fix
NO_PIN = "a schedule no scan class pins (scan_service.SCAN_PINS), so no scan Program is the verifier's at it"
NOT_PINNED = "not the scan Program pinned at the schedule's digest (scan_service.SCAN_PINS)"
NOT_CLASS = "not the class circuit (the parameters' circuit), which each audit runs"
ANNOTATED = "a descriptor's annotations, which no digest covers"


def _hex(doc: Any, k: str) -> Any:
    """``doc[k]``, hex of JSON (`HEX`), as its JSON; None where it isn't, which `scan_doc` finds."""
    try:
        return json.loads(bytes.fromhex(doc[k]))
    except (KeyError, TypeError, ValueError):
        return None


def _programs(doc: Any, key: str) -> list[Any]:
    """The ``program`` of each item of ``doc``'s list ``key`` that names one."""
    items = doc.get(key) if isinstance(doc, dict) else None
    return [x["program"] for x in items if isinstance(x, dict) and "program" in x] if isinstance(items, list) else []


def _digest(desc: Any) -> str | None:
    return program_digest(desc) if isinstance(desc, dict) else None


def pinned(seen: Mapping[str, Any]) -> list[Finding]:
    """DIR's Programs held to the verifier's (module docstring), ``seen`` its files' JSON by path: each finding names
    its file and key path, never the value."""
    case, out, pins = seen.get("case.json"), [], []
    periods = [("case.json", "params.", case.get("params"), "case.json", "commitment.", _hex(case, "commitment"))
               ] if isinstance(case, dict) else []
    if {"period/params.json", "period/commitment.bin"} & seen.keys():
        periods.append(("period/params.json", "", seen.get("period/params.json"), "period/commitment.bin", "",
                        seen.get("period/commitment.bin")))
    for pf, pp, params, cf, cp, commitment in periods:
        params = params if isinstance(params, dict) else {}
        try:
            pin = SCAN_PINS.get(WardenParams(**params["schedule"]).digest.hex())
        except (KeyError, TypeError, ValueError):
            pin = None
        scans = [(cf, f"{cp}windows[].program", p) for p in _programs(commitment, "windows")]
        scans += [(pf, f"{pp}program", params["program"])] if "program" in params else []
        if "scan" in params:
            scans.append((pf, f"{pp}scan", _digest(params["scan"])))
            if isinstance(params["scan"], dict) and params["scan"].get("annotations", {}) != {}:
                out.append(Finding(pf, f"{pp}scan.annotations", ANNOTATED))
        if pin is None:
            out.append(Finding(pf, f"{pp}schedule", NO_PIN))
        else:
            pins.append(pin)
            out += [Finding(f, p, NOT_PINNED) for f, p, v in scans if v != pin]
        audits = [(pf, f"{pp}audits[].program", _digest(d)) for d in _programs(params, "audits")]
        audits += [(cf, f"{cp}audits[].program", p) for p in _programs(commitment, "audits")]
        out += [Finding(f, p, NOT_CLASS) for f, p, v in audits if v is None or v != params.get("circuit")]
    answers = seen.get("chain/scan/answers.json")
    reg = answers.get("registration") if isinstance(answers, dict) else None
    if isinstance(reg, dict) and "program" in reg and reg["program"] not in (pins if periods else [*SCAN_PINS.values()]):
        out.append(Finding("chain/scan/answers.json", "registration.program", NOT_PINNED))
    return out


def _json(rel: str, p: Path, out: list[Finding]) -> Any:
    if p.is_symlink():
        out.append(Finding(rel, "", "a symlink, which the verifier would receive as the file it names"))
        return None
    try:
        return json.loads(p.read_bytes())
    except ValueError as e:
        out.append(Finding(rel, "", f"isn't JSON ({type(e).__name__})"))
        return None


def scan(d: Path, docs: Mapping[str, Mapping[str, Any]] = DOCS) -> tuple[list[Finding], Counter]:
    """Every file under ``d`` scanned by its declaration in ``docs`` (module docstring; `DOCS`, a piece's DIR, its
    Programs held to the verifier's: `pinned`): the findings, and the leaves counted by kind."""
    kinds, out, seen = Counter(), [], {}
    for p in sorted(x for x in d.rglob("*") if x.is_file() or x.is_symlink()):
        rel = p.relative_to(d).as_posix()
        if rel in TEXT and docs is DOCS and not p.is_symlink():
            kinds[TEXT[rel]] += 1
            continue
        if rel not in docs:
            out.append(Finding(rel, "", "a file the verifier would receive that no declaration covers"))
            continue
        if (doc := _json(rel, p, out)) is not None:
            out += scan_doc(rel, doc, kinds, docs)
            seen[rel] = doc
    return out + (pinned(seen) if docs is DOCS else []), kinds


def padded(rel: str, reg: Any) -> list[Finding]:
    """The advice registration's length at its cap (`served_verdicts.public_registration`): ``binding.advice.bits`` its
    compiler's ``cap_bits`` and ``rows`` the rows those fill, else a finding, the advice's own length being the
    window's."""
    try:
        cap, row_bits = int(reg["compiler"]["cap_bits"]), int(reg["row_bits"])
        got = {"binding.advice.bits": reg["binding"]["advice"]["bits"], "rows": reg["rows"]}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return [Finding(rel, "binding.advice.bits", "no length the cap can be read against")]
    want = {"binding.advice.bits": cap, "rows": -(-cap // row_bits) if row_bits > 0 else None}
    return [Finding(rel, k, "the advice's own length, not the cap's") for k in got if got[k] != want[k]]


def received(out: Path) -> tuple[list[Finding], Counter]:
    """What a verifier on another machine reads of a served run's ``out`` (module docstring): each dir of
    `RECEIVED_DIRS` scanned whole, each path of `RECEIVED` by its declaration, the advice registration's length held to
    its cap (`padded`). A dir or file missing is no finding: what reads it reads its absence."""
    kinds, found = Counter(), []
    for sub, docs in RECEIVED_DIRS.items():
        if (out / sub).is_dir():
            f, k = scan(out / sub, docs)
            found += [dataclasses.replace(x, file=f"{sub}/{x.file}") for x in f]
            kinds.update(k)
    for pattern, table in RECEIVED.items():
        for p in sorted(out.glob(pattern)):
            rel = p.relative_to(out).as_posix()
            if (doc := _json(rel, p, found)) is not None:
                found += scan_doc(rel, doc, kinds, {rel: table})
                if table is ADVICE_REGISTRATION:
                    found += padded(rel, doc)
    return found, kinds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kind_scan.py", description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path, nargs="?")
    ap.add_argument("--received", type=Path, metavar="OUT", help="a served run's OUT: what a verifier reads of it")
    a = ap.parse_args(argv)
    if (a.dir is None) == (a.received is None):
        ap.error("DIR or --received OUT, one of them")
    found, kinds = scan(a.dir) if a.received is None else received(a.received)
    for f in found:
        print(f"KIND-SCAN finding: {f}")
    print(f"KIND-SCAN {'clean' if not found else f'{len(found)} findings'}: "
          f"{json.dumps({k: kinds[k] for k in KINDS}, sort_keys=False)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
