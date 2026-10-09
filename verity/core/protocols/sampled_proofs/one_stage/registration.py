"""The registration record (`docs/audit-protocols.md` §0.1): what serving is committed to, fixed before the verifier draws.

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <units the query derives>, "law": {"law": "subset", "k": K} | {"law": "stratified" | "work", "k": K, "strata": [...]},
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
* **Registered values** (`verity/registration/v2`): the record adds `"registered": [{"value", "root", "leaves", "words",
  "domain"}, ...]` (with `"word_bits"` for a value whose leaves are not 16 bits), the program's weights, constants and
  tables committed once under hiding leaves (`registered`), and R7 holds them to the verifier's own list and to the roots
  the log's earlier records registered. A record registers values exactly when it is v2; v1's bytes are unchanged.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.program import Program

from . import draw as D, partition as P, registered as G
from .verdict import Verdict, _Collect

FORMAT = "verity/registration/v1"
FORMAT_V2 = "verity/registration/v2"
PROTOCOL = "verity/sampled-proofs/one-stage/v1"
RECEIPT = "verity/registration-receipt/v1"
SERVED_DOMAIN = "verity/one-stage/served-domain/v1"
KINDS = ("served", "stand-in")
_KEYS = {"format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots", "leaf_layer",
         "anchors", "window", "previous"}
_ROOT_KEYS = {"port", "root", "leaves", "domain"}
_HEX = re.compile(r"[0-9a-f]+")


def query_digest(query: Mapping[str, Any]) -> str:
    return hashlib.sha512(canonical_json(dict(query))).hexdigest()


def served_domain(program: str, partition: str, port: str, schema: str, leaves: int, run: str) -> str:
    binding = identity_digest_sha512(SERVED_DOMAIN, {"program": program, "partition": partition, "port": port, "schema": schema,
                                                     "leaves": int(leaves), "run": run})
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, int(leaves)), hash="sha512").domain_id.hex()


def record(program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str, roots: Sequence[Mapping[str, Any]],
           window: Mapping[str, Any], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
           previous: str | None = None, partition: Mapping[str, Any] | None = None,
           work_table: Mapping[str, Any] | None = None, registered: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """`partition`: an explicit partition object over `program` (default: the canonical `P.partition(program, query)`).
    `law` is written expanded over the program and query (`draw.derive`: a `stratified` rule becomes its strata, and a
    `work` rule its strata sized by `work_table`). `registered`: the registered values' entries (`Registered.entry`),
    which make the record v2."""
    part = dict(partition) if partition is not None else P.partition(program, query)
    rec = {"format": FORMAT_V2 if registered else FORMAT, "protocol": PROTOCOL, "program": part["program"],
           "query": query_digest(part["query"]), "partition": P.digest(part), "population": P.units(program, query),
           "law": D.derive(program, query, law, work_table), "scheme": scheme,
           "roots": [{"port": r["port"], "root": r["root"], "leaves": int(r["leaves"]), "domain": r["domain"]} for r in roots],
           "leaf_layer": leaf_layer, "anchors": list(anchors), "window": dict(window), "previous": previous}
    if registered:
        rec["registered"] = [{**{k: (int(e[k]) if k in ("leaves", "words") else e[k]) for k in G.ENTRY_KEYS},
                              **G.width_entry(int(e.get(G.WIDTH_KEY, G.WORD_BITS)))} for e in registered]
    return rec


def digest(rec: Mapping[str, Any]) -> str:
    """`SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag."""
    tag = FORMAT_V2 if rec.get("format") == FORMAT_V2 else FORMAT
    return hashlib.sha512(tag.encode() + b"\0" + canonical_json(dict(rec))).hexdigest()


def held_roots(log: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """`{value: root}` for every value an earlier record of the log registered, the root of its first registration."""
    held: dict[str, str] = {}
    for rec in log:
        for e in rec.get("registered") or ():
            held.setdefault(e["value"], e["root"])
    return held


def receipt(rec: Mapping[str, Any], verifier: str, received: str | None = None) -> dict[str, Any]:
    """The verifier's own log entry for a record it accepted, with its own clock; the draw names it."""
    return {"format": RECEIPT, "registration": digest(rec), "verifier": verifier,
            "received": received or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any], law: Mapping[str, Any], scheme: str,
          ports: Mapping[str, Mapping[str, Any]], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
          log: Iterable[Mapping[str, Any]] = (), partition: Mapping[str, Any] | None = None,
          work_table: Mapping[str, Any] | None = None, registered: Mapping[str, int] | None = None) -> Verdict:
    """R1-R7 against the verifier's own copies: its program and query, its law and scheme, the ports its layout derives
    (`{port: {"leaves", "domain"}}`), the digest of its copy of the leaf layer, its anchors, its log of this prover's
    earlier records, and the parameters it expects registered (`registered`: `{parameter: words}`).  `partition`: the
    verifier's own partition object when it isn't the canonical one (a record made before the canonical envelope); its
    program must still be `program`'s, and the population is `query`'s.  R1 holds the registered law to the verifier's own
    law expanded over its program and query (`draw.derive`): for `stratified`, the strata by template descriptor id and
    each stratum's k by the rule; for `work`, also its own `work_table`.  R7 holds the registered values to the verifier's
    own (`registered_entries`) and a value an earlier record of the log registered to that record's root (`held_roots`):
    a value is committed once, and the log is the only place the verifier holds its roots.  R3 reads `ports` by name, so
    its codes come in the ports' sorted order whatever order the mapping has, as the Lean port reads them."""
    c = _Collect()
    v2 = bool(registered)
    keys = _KEYS | {"registered"} if v2 else _KEYS
    if not isinstance(rec, Mapping) or set(rec) != keys:
        c.fail("keys", sorted(set(rec) ^ keys) if isinstance(rec, Mapping) else rec)
        return c.verdict()
    # R1
    if (rec["format"], rec["protocol"]) != (FORMAT_V2 if v2 else FORMAT, PROTOCOL):
        c.fail("R1-protocol", [rec["format"], rec["protocol"]])
    try:
        mine = D.derive(program, query, law, work_table)
    except (P.QueryRefused, ValueError) as e:
        c.fail("R1-law", {"registered": rec["law"], "verifier": f"{dict(law)} does not apply: {e}"})
    else:
        if not D.same_law(rec["law"], mine):
            c.fail("R1-law", {"registered": rec["law"], "verifier": mine})
    if not isinstance(rec["window"], Mapping) or rec["window"].get("kind") not in KINDS:
        c.fail("window", rec["window"])
    # R2
    part = dict(partition) if partition is not None else P.partition(program, query)
    if part["program"] != P.program_digest(program):
        c.fail("R2-program", {"partition": part["program"], "verifier": P.program_digest(program)})
    try:
        part_digest = P.digest(part)
    except ValueError as e:
        c.fail("R2-partition", {"registered": rec["partition"], "verifier": str(e)})
        part_digest = None
    for field, mine in (("program", part["program"]), ("query", query_digest(part["query"])), ("partition", part_digest)):
        if mine is not None and rec[field] != mine:
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
        for p, want in sorted(ports.items()):
            r = got.get(p)
            if r is None:
                continue
            if not (isinstance(r["root"], str) and _HEX.fullmatch(r["root"])):
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
    # R7
    if v2:
        _check_registered(c, rec["registered"], program, registered, held_roots(earlier))
    return c.verdict()


def _check_registered(c: _Collect, got: Any, program: Program, values: Mapping[str, int], held: Mapping[str, str]) -> None:
    try:
        mine = G.registered_entries(program, values)
    except ValueError as e:
        c.fail("R7-registered", {"verifier": f"{dict(values)} does not apply: {e}"})
        return
    keys = set(G.ENTRY_KEYS)
    if not isinstance(got, list) or not all(isinstance(e, Mapping) and keys <= set(e) <= keys | {G.WIDTH_KEY} for e in got):
        c.fail("R7-registered", got)
        return
    by_value = {e["value"]: e for e in got}
    if len(by_value) != len(got) or set(by_value) != set(mine):
        c.fail("R7-values", {"registered": [e["value"] for e in got], "verifier": sorted(mine)})
    for v, want in mine.items():
        e = by_value.get(v)
        if e is None:
            continue
        if not (isinstance(e["root"], str) and len(e["root"]) == 128 and _HEX.fullmatch(e["root"])):
            c.fail("R7-root", v)
        # a count is an int, never a float or bool equal to it, and 16 is named by leaving word_bits out: one record
        # digest per registration
        for k in ("leaves", "words", "domain"):
            if e[k] != want[k] or (k != "domain" and type(e[k]) is not int):
                c.fail(f"R7-{k}", {"value": v, "registered": e[k], "verifier": want[k]})
        width = e.get(G.WIDTH_KEY, G.WORD_BITS)
        if type(width) is not int or width != want["word_bits"] or e.get(G.WIDTH_KEY) == G.WORD_BITS:
            c.fail("R7-word_bits", {"value": v, "registered": width, "verifier": want["word_bits"]})
        if v in held and e["root"] != held[v]:
            c.fail("R7-held", {"value": v, "registered": e["root"], "verifier": held[v]})


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


registered_entries = G.registered_entries

__all__ = ["FORMAT", "FORMAT_V2", "KINDS", "PROTOCOL", "RECEIPT", "SERVED_DOMAIN", "check", "digest", "held_roots",
           "matches_public", "query_digest", "receipt", "record", "registered_entries", "served_domain"]
