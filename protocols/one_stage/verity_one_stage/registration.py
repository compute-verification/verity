"""The registration record: what the serving side is committed to, written before the verifier draws.

    {"format": "verity/one-stage/registration/v0",
     "commitment": {"scheme": "<leaf and tree scheme>", "source": {"kind": "served" | "stand-in", ...}},
     "roots": {"<port>": "<root hex>"}, "leaves": {"<port>": <leaf count>}, "domains": {"<port>": "<domain id hex>"},
     "partition": {"program": ..., "query": ...}, "partition_digest": "<SHA-512 hex>",
     "population": <units the query derives>,
     "registered_utc": "YYYY-MM-DDTHH:MM:SSZ"}

* **Digest.** `SHA-512("verity/one-stage/registration/v0\\0" || canon(record))`.
* **Checked by content** (`check`): the verifier recomputes the partition and its digest from its own copy of the program
  and the query, and the population by applying the query. Nothing in the record is taken on the prover's word.
* **Its public file** (`matches_public`): the verifier's own public file must carry the registered scheme, the same
  roots over the same domains, and the same population before it serves a session over it. A root means nothing without
  its domain (what the leaves are and where they sit), so the registration fixes both.
* **`source`.** `served` when the roots are what the serving path committed. `stand-in` when they are a fresh commitment
  to values captured from serving; it names the root those values were read under, and that link is provenance, not proof.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Mapping

from verity.ir.codec import canonical_json
from verity.ir.program import Program

from . import partition as P
from .verdict import Verdict, _Collect

FORMAT = "verity/one-stage/registration/v0"
_KEYS = {"format", "commitment", "roots", "leaves", "domains", "partition", "partition_digest", "population", "registered_utc"}
SOURCES = ("served", "stand-in")
_HEX = re.compile(r"^[0-9a-f]+$")


def record(scheme: str, roots: Mapping[str, str], leaves: Mapping[str, int], domains: Mapping[str, str], program: Program,
           query: Mapping[str, Any], source: Mapping[str, Any], registered_utc: str | None = None) -> dict[str, Any]:
    part = P.partition(program, query)
    return {"format": FORMAT, "commitment": {"scheme": scheme, "source": dict(source)},
            "roots": dict(roots), "leaves": {k: int(v) for k, v in leaves.items()}, "domains": dict(domains),
            "partition": part, "partition_digest": P.digest(part), "population": P.units(program, query),
            "registered_utc": registered_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def digest(rec: Mapping[str, Any]) -> str:
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(dict(rec))).hexdigest()


def check(rec: Mapping[str, Any], program: Program, query: Mapping[str, Any]) -> Verdict:
    """The record's form, and its partition, digest and population against the verifier's own program and query."""
    c = _Collect()
    if set(rec) != _KEYS:
        c.fail("keys", sorted(set(rec) ^ _KEYS))
        return c.verdict()
    if rec["format"] != FORMAT:
        c.fail("format", rec["format"])
    com = rec["commitment"]
    if (not isinstance(com, Mapping) or set(com) != {"scheme", "source"} or not isinstance(com.get("scheme"), str)
            or not isinstance(com.get("source"), Mapping) or com["source"].get("kind") not in SOURCES):
        c.fail("commitment", com)
    roots, leaves, domains = rec["roots"], rec["leaves"], rec["domains"]
    if not roots or not all(isinstance(r, str) and _HEX.match(r) for r in roots.values()):
        c.fail("roots", roots)
    if set(leaves) != set(roots) or not all(isinstance(n, int) and n > 0 for n in leaves.values()):
        c.fail("leaves", leaves)
    if set(domains) != set(roots) or not all(isinstance(d, str) and _HEX.match(d) for d in domains.values()):
        c.fail("domains", domains)
    part = P.partition(program, query)
    if rec["partition"] != part:
        c.fail("partition", {"registered": rec["partition"], "verifier": part})
    if rec["partition_digest"] != P.digest(rec["partition"]):
        c.fail("partition-digest", rec["partition_digest"])
    try:
        population = P.units(program, query)
    except P.QueryRefused as e:
        c.fail("query", str(e))
    else:
        if rec["population"] != population:
            c.fail("population", {"registered": rec["population"], "verifier": population})
    return c.verdict()


def matches_public(rec: Mapping[str, Any], scheme: str, roots: Mapping[str, str], domains: Mapping[str, str],
                   population: int) -> Verdict:
    """The verifier's own public file (its scheme, roots, their domains and its instance count) against the record."""
    c = _Collect()
    if scheme != rec["commitment"]["scheme"]:
        c.fail("scheme", {"registered": rec["commitment"]["scheme"], "public": scheme})
    if dict(roots) != rec["roots"]:
        c.fail("roots", {"registered": rec["roots"], "public": dict(roots)})
    if {p: domains.get(p) for p in rec["domains"]} != rec["domains"]:
        c.fail("domains", {"registered": rec["domains"], "public": dict(domains)})
    if population != rec["population"]:
        c.fail("population", {"registered": rec["population"], "public": population})
    return c.verdict()


__all__ = ["FORMAT", "check", "digest", "matches_public", "record"]
