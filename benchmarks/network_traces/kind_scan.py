"""kind-scan: each field the network certifier's piece releases to the verifier, by kind. A field is one of the four
public kinds (`verity.core.wire.PUBLIC`: protocol structure, the protocol's outputs, the verifier's coins and draws,
hiding commitments) or what the network certifier signed (`CERTIFICATE`: its certificates reach the verifier in the
clear from the auditor's device, a provisional exception to the four kinds, not a fifth). A field of no declared kind is
a finding.

    python benchmarks/network_traces/kind_scan.py DIR

DIR is a piece's DIR, or its hold before the leak guard releases it (`certifier_piece.settle` runs the scan there on
every served run and refuses on a finding, releasing nothing): `DOCS`' files, the case as DIR/case.json (its commitment
and evidence hex, `certifier_piece.write_case`, as the composite reads it) or as the three files ``network-period`` reads
(DIR/period). Each file's JSON is walked to its leaves, the commitment's and evidence's bytes decoded as JSON, and each
leaf's key path (a list's items ``[]``) must match one its file declares: ``*`` is any one key (a Definition's, a port's,
a stratum's, a δ's), and ``**`` everything below, a Program's descriptor (`DESCRIPTOR`, the circuit, which is structure)
or a development check's field. A list's length and an object's keys are the shape of the leaves they hold. So a field
no declaration names, one a change adds included, is a finding until its kind is declared here: a window's own number, a
count of its requests, the advice's length.

The record path's two development checks are findings wherever they reach the verifier (`Development`, each naming its
check): an advice binding in a registration (``window.compiler``, ``window.advice``: the compiler and the advice's root and
length an audit opens, `records_audit.py audit --opening`), since the window commit's proofs read the advice under its
hiding root and no registration of the record path binds it to be opened; and the chain's endpoint rebuild (the report's
``endpoint`` step, chain_record.sh's CR_ENDPOINT on the endpoint's records and advice in the clear), which the
active-warden path refuses. The report's steps are the active-warden path's (`STEPS`). A text field (a verdict's reason, a
step's account of what ran in the clear) is declared by its key, its text not read. The logs (`LOGS`) are the run's
account, published in its public form (`served_verdicts.publish`), not the verifier's input, and aren't read; any other
file is a finding.

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
from verity.core.protocols.certifier.warden import WardenParams

CERTIFICATE = "certificate"
KINDS = (*(k for k in W.KINDS if k in W.PUBLIC), CERTIFICATE)
S, O, C, M, X = (*KINDS,)


@dataclasses.dataclass(frozen=True)
class Development:
    """A field only a development check writes: a finding wherever it is, ``why`` naming the check."""

    why: str


OPENING = Development("an advice binding, which an audit opens (records_audit.py audit --opening, a development check): "
                      "on the record path the window commit's proofs read the advice under its hiding root, its length "
                      "included, and nothing opens it")
ENDPOINT = Development("the chain's endpoint rebuild (CR_ENDPOINT: the endpoint's records and advice, in the clear), a "
                       "development check the active-warden path refuses")
#: the chain's steps on the active-warden path (`chain_record.report`)
STEPS = ("custody", "certificates", "n2", "scan", "lean")


def under(prefix: str, table: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in table.items()}


#: a Program's descriptor (``verity-ir/descriptor/v1``): the circuit, every field of it structure
DESCRIPTOR = {"**": S}
#: a window's circuit record (``verity/window-circuit/v1``) and its wires' order (``verity/wire-order/v1``): the circuit's
#: description, every field of it structure
RECORD = {"**": S}
#: a registered value's entry (`registered.Registered.entry`)
ENTRY = {"value": S, "root": M, "leaves": S, "words": S, "word_bits": S, "domain": S}
#: a one-stage registration (`one_stage.registration.record`): the digests and population of the Program and query, the
#: law the verifier's sampling design over them, the roots and registered values hiding commitments under domains of
#: their structure; its window the provenance: a scan round's account and members, or a committed window's run root,
#: circuit record, class, per-window values, registered roots and index leaf. The window's own number is none of them,
#: and the advice's binding (`interface.advice_binding`) is `OPENING`'s.
REGISTRATION = {
    "format": S, "protocol": S, "program": S, "query": S, "partition": S, "population": S, "scheme": S, "leaf_layer": S,
    "anchors[]": S, "previous": M,
    "law.law": S, "law.k": S, "law.p": S, "law.strata[].template": S, "law.strata[].k": S,
    "law.strata[].units[][]": S, "law.strata[].work": S, "law.strata[].floor": S,
    "roots[].port": S, "roots[].leaves": S, "roots[].domain": S, "roots[].root": M,
    **under("registered[]", ENTRY),
    "window.kind": S, "window.link": S, "window.of": S, "window.members.*.template": S, "window.members.*.units[]": S,
    "window.run": M, "window.circuit": S, "window.order": S, "window.index_leaf": M, "window.registered_roots.*": M,
    "window.window_circuit.class[]": S, "window.activations.program": S,
    **under("window.activations.values.*", ENTRY),
    "window.compiler.**": OPENING, "window.advice.**": OPENING,
}
#: a one-stage answer (`served_integrity.answer`, `Answer.toJson`): its verdict and bound the protocol's outputs, its
#: profile the integrity profile's structure, and the units drawn and proved the verifier's draws
ONE_STAGE = {
    "protocol": S, "verdict": O, "reason": O, "commitment": M,
    "profile": O, "profile.programs[]": S, "profile.prescribed[]": S, "profile.covered[]": S, "profile.linked": S,
    "profile.delta": S, "profile.drawn.*": S,
    "profile.levels[].name": S, "profile.levels[].query": S, "profile.levels[].size": S, "profile.levels[].draw": S,
    **{f"profile.levels[].draw.{k}": S for k in ("law", "k", "subset", "bernoulli")},
    **{f"profile.levels[].draw.strata[].{k}": S for k in ("name", "size", "k", "work", "floor")},
    "public": O, "public.bound": O, "public.per_template": O, "public.per_template.*": O, "public.drawn[]": C,
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
    "profile.placement.parties.*": S,
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
        "eps_at_delta.incorrect_at_delta.*": O}
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
#: the logs `served_verdicts.sh` and `served_verdicts.py` write in DIR (`certifier_piece.LOGS`)
LOGS = ("piece.log", "chain.log", "period.log")

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
            node = node.setdefault(seg, {})
        node[_KIND] = kind
    return root


def _name(path: tuple[str, ...]) -> str:
    return "".join(s if s == "[]" or not i else f".{s}" for i, s in enumerate(path))


def _leaves(value: Any, nodes: list[dict], path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], list, Any]]:
    """Each leaf of ``value`` (a scalar, or a subtree a ``**`` declares) with the kinds the trie nodes ``nodes`` declare at
    its key path."""
    whole = [n["**"][_KIND] for n in nodes if "**" in n]
    if whole:
        yield path, whole, None
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, [c for n in nodes for c in (n.get(k), n.get("*")) if c is not None], (*path, k))
    elif isinstance(value, list):
        inner = [n["[]"] for n in nodes if "[]" in n]
        for v in value:
            yield from _leaves(v, inner, (*path, "[]"))
    else:
        yield path, [n[_KIND] for n in nodes if _KIND in n], value


def scan_doc(rel: str, doc: Any, kinds: Counter) -> list[Finding]:
    """``doc``'s findings, as DIR's file ``rel`` (`DOCS`), each leaf counted in ``kinds`` by its kind."""
    for k in HEX.get(rel, ()):
        if isinstance(doc, dict) and isinstance(doc.get(k), str):
            try:
                doc = {**doc, k: json.loads(bytes.fromhex(doc[k]))}
            except ValueError as e:
                return [Finding(rel, k, f"isn't hex of JSON ({type(e).__name__})")]
    out = []
    for path, declared, value in _leaves(doc, [_trie(DOCS[rel])]):
        if not declared:
            out.append(Finding(rel, _name(path), f"a {type(value).__name__} of no declared kind"))
        elif len(set(declared)) > 1:
            out.append(Finding(rel, _name(path), f"declared as {sorted(map(str, declared))}, not one kind"))
        elif isinstance(declared[0], Development):
            out.append(Finding(rel, _name(path), declared[0].why))
        else:
            kinds[declared[0]] += 1
    return out


def scan(d: Path) -> tuple[list[Finding], Counter]:
    """Every file under ``d`` scanned (module docstring): the findings, and the leaves counted by kind."""
    kinds, out = Counter(), []
    for p in sorted(x for x in d.rglob("*") if x.is_file() or x.is_symlink()):
        rel = p.relative_to(d).as_posix()
        if rel in LOGS:
            continue
        if rel not in DOCS or p.is_symlink():
            out.append(Finding(rel, "", "a file the verifier would receive that no declaration covers"))
            continue
        try:
            doc = json.loads(p.read_bytes())
        except ValueError as e:
            out.append(Finding(rel, "", f"isn't JSON ({type(e).__name__})"))
            continue
        out += scan_doc(rel, doc, kinds)
    return out, kinds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kind_scan.py", description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    found, kinds = scan(a.dir)
    for f in found:
        print(f"KIND-SCAN finding: {f}")
    print(f"KIND-SCAN {'clean' if not found else f'{len(found)} findings'}: "
          f"{json.dumps({k: kinds[k] for k in KINDS}, sort_keys=False)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
