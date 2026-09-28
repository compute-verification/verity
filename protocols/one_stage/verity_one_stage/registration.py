"""The registration record (`docs/audit-protocols.md` §0.1): what serving is committed to, fixed before the verifier draws.

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <units the query derives>, "law": {"law": "subset", "k": K} | {"law": "stratified", "k": K, "strata": [...]},
     "scheme": "<leaf and tree scheme>",
     "roots": [{"port": ..., "root": ..., "leaves": ..., "domain": ...}, ...],
     "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
     "anchors": [], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}

* **Digest.** `SHA-512("verity/registration/v1\\0" || canon(record))`.
* **Rule.** Everything a drawn unit's statement can depend on is fixed by the record, or by objects the verifier already
  holds, before the draw. The verifier checks R1-R6 (`check`) against its own copies, writes a `receipt` with its own
  clock, and only then draws.
* **`window`** is provenance only. `kind` says whether the roots are what serving committed (`served`) or a fresh
  commitment to values captured from serving (`stand-in`, whose link to the served root is provenance, not proof).
* **Served domains** (`served_domain`, agreed with vllm-serving-commit): each port's frame domain binds the program, the
  partition, the port, its schema, its leaf count and the run, so the roots can't be read as anything else.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.ir.codec import canonical_json
from verity.ir.program import Program

from . import draw as D, partition as P
from .verdict import Verdict, _Collect

FORMAT = "verity/registration/v1"
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
RECEIPT = "verity/registration-receipt/v1"
SERVED_DOMAIN = "verity/one-stage/served-domain/v0"
KINDS = ("served", "stand-in")
_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
         "anchors", "window", "previous"}
_ROOT_KEYS = {"port", "root", "leaves", "domain"}
_HEX = re.compile(r"^[0-9a-f]+$")


def query_digest(query: Mapping[str, Any]) -> str:
    return hashlib.sha512(canonical_json(dict(query))).hexdigest()


def served_domain(program: str, partition: str, port: str, schema: str, leaves: int, run: str) -> str:
    binding = identity_digest(SERVED_DOMAIN, {"program": program, "partition": partition, "port": port, "schema": schema,
                                              "leaves": int(leaves), "run": run})
    return CommitmentDomain(bytes.fromhex(binding), -1, RangeIndexedDomain(0, int(leaves)), hash="sha512").domain_id.hex()


def record(program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str, roots: Sequence[Mapping[str, Any]],
           window: Mapping[str, Any], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
           previous: str | None = None, partition: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`partition`: an explicit partition object over `program` (default: the canonical `P.partition(program, query)`).
    `law` is written expanded over the program and query (`draw.derive`: a `stratified` rule becomes its strata)."""
    part = dict(partition) if partition is not None else P.partition(program, query)
    return {"format": FORMAT, "protocol": PROTOCOL, "program": part["program"], "query": query_digest(part["query"]),
            "partition": P.digest(part), "population": P.units(program, query), "law": D.derive(program, query, law),
            "scheme": scheme,
            "roots": [{"port": r["port"], "root": r["root"], "leaves": int(r["leaves"]), "domain": r["domain"]} for r in roots],
            "leaf_layer": leaf_layer, "anchors": list(anchors), "window": dict(window), "previous": previous}


def digest(rec: Mapping[str, Any]) -> str:
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(dict(rec))).hexdigest()


def receipt(rec: Mapping[str, Any], verifier: str, received: str | None = None) -> dict[str, Any]:
    """The verifier's own log entry for a record it accepted, with its own clock; the draw names it."""
    return {"format": RECEIPT, "registration": digest(rec), "verifier": verifier,
            "received": received or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None) -> Verdict:
    """R1-R6 against the verifier's own copies: its program and query, its law and scheme, the ports its layout derives
    (`{port: {"leaves", "domain"}}`), the digest of its copy of the leaf layer, its anchors, and its log of this prover's
    earlier records.  `partition`: the verifier's own partition object when it isn't the canonical one (a record made
    before the canonical envelope); its program must still be `program`'s, and the population is `query`'s.  R1 holds
    the registered law to the verifier's own law expanded over its program and query (`draw.derive`): for `stratified`,
    the strata by template descriptor id and each stratum's k by the rule."""
    c = _Collect()
    if not isinstance(rec, Mapping) or set(rec) != _KEYS:
        c.fail("keys", sorted(set(rec) ^ _KEYS) if isinstance(rec, Mapping) else rec)
        return c.verdict()
    # R1
    if (rec["format"], rec["protocol"]) != (FORMAT, PROTOCOL):
        c.fail("R1-protocol", [rec["format"], rec["protocol"]])
    try:
        mine = D.derive(program, query, law)
    except (P.QueryRefused, ValueError) as e:
        c.fail("R1-law", {"registered": rec["law"], "verifier": f"{dict(law)} does not apply: {e}"})
    else:
        if not D.same_law(rec["law"], mine):
            c.fail("R1-law", {"registered": rec["law"], "verifier": mine})
    if not isinstance(rec["window"], Mapping) or rec["window"].get("kind") not in KINDS:
        c.fail("window", rec["window"])
    # R2
    part = dict(partition) if partition is not None else P.partition(program, query)
    if part["program"] != P.program_sha512(program):
        c.fail("R2-program", {"partition": part["program"], "verifier": P.program_sha512(program)})
    for field, mine in (("program", part["program"]), ("query", query_digest(part["query"])), ("partition", P.digest(part))):
        if rec[field] != mine:
            c.fail(f"R2-{field}", {"registered": rec[field], "verifier": mine})
    try:
        population = P.units(program, query)
    except P.QueryRefused as e:
        c.fail("R2-query", str(e))
    else:
        if rec["population"] != population:
            c.fail("R2-population", {"registered": rec["population"], "verifier": population})
    # R3
    if rec["scheme"] != scheme:
        c.fail("R3-scheme", {"registered": rec["scheme"], "verifier": scheme})
    roots = rec["roots"] if isinstance(rec["roots"], list) else []
    if not roots or not all(isinstance(r, Mapping) and set(r) == _ROOT_KEYS for r in roots):
        c.fail("R3-roots", rec["roots"])
    else:
        got = {r["port"]: r for r in roots}
        if len(got) != len(roots) or set(got) != set(ports):
            c.fail("R3-ports", {"registered": [r["port"] for r in roots], "verifier": sorted(ports)})
        for p, want in ports.items():
            r = got.get(p)
            if r is None:
                continue
            if not (isinstance(r["root"], str) and _HEX.match(r["root"])):
                c.fail("R3-root", p)
            if r["leaves"] != want["leaves"]:
                c.fail("R3-leaves", {"port": p, "registered": r["leaves"], "verifier": want["leaves"]})
            if r["domain"] != want["domain"]:
                c.fail("R3-domain", {"port": p, "registered": r["domain"], "verifier": want["domain"]})
    # R4: when the verifier holds the leaf layer, the record must name it (a null leaf layer is refused)
    if rec["leaf_layer"] != leaf_layer:
        c.fail("R4-leaf-layer", {"registered": rec["leaf_layer"], "verifier": leaf_layer})
    # R5
    if rec["anchors"] != list(anchors):
        c.fail("R5-anchors", rec["anchors"])
    # R6
    earlier = list(log)
    if rec["previous"] != (digest(earlier[-1]) if earlier else None):
        c.fail("R6-previous", rec["previous"])
    if any(e.get("window") == rec["window"] for e in earlier):
        c.fail("R6-window", rec["window"])
    return c.verdict()


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int) -> Verdict:
    """The verifier's own public file (its scheme, roots, their domains and its instance count) against the record."""
    c = _Collect()
    reg = {r["port"]: r for r in rec["roots"]}
    if scheme != rec["scheme"]:
        c.fail("scheme", {"registered": rec["scheme"], "public": scheme})
    if {p: r["root"] for p, r in reg.items()} != dict(roots):
        c.fail("roots", {"registered": {p: r["root"] for p, r in reg.items()}, "public": dict(roots)})
    if {p: domains.get(p) for p in reg} != {p: r["domain"] for p, r in reg.items()}:
        c.fail("domains", {"registered": {p: r["domain"] for p, r in reg.items()}, "public": dict(domains)})
    if population != rec["population"]:
        c.fail("population", {"registered": rec["population"], "public": population})
    return c.verdict()


__all__ = ["FORMAT", "KINDS", "PROTOCOL", "RECEIPT", "SERVED_DOMAIN", "check", "digest", "matches_public", "query_digest",
           "receipt", "record", "served_domain"]
