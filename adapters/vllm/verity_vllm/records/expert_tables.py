"""The committer's registration of a TP rank's per-expert tables (`verity-vllm/expert-tables/v1`): each table's entry (core's
`registered.Registered.entry()` of a ``hm96-sha512/row-seg/v1`` value, ``weights/<label>@<H>/b<block>/table``,
`pipeline.expert_rows.table_param`) at the rank's ``program``, its rows committed under the committer's secret key at the salt
nonce ``nonces`` names, before the first window's coins.  The verifier pins its ``program`` and ``values``
(``records_audit.py audit --registered``), and a window's committer commits the tables it serves at the same program and nonces
(`pipeline.expert_rows.committer`).  It names no key and no salt: a nonce only numbers the salts the key derives."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/expert-tables/v1"
#: an entry's fields: core's `registered.Registered.entry()` (``word_bits`` where its words are not 16-bit)
ENTRY = ("value", "root", "leaves", "words", "word_bits", "domain")
ROOT = re.compile(r"[0-9a-f]{128}")


@dataclass(frozen=True)
class ExpertTables:
    program: str                                      # the rank's program digest, which each entry's domain binds
    repo: str                                         # the model registered, at its revision
    revision: str
    world: int                                        # the TP world, and the rank whose shards the tables hold
    rank: int
    prefix: str                                       # the Programs' wrapper prefix of the tables' labels
    blocks: Mapping[str, int]                         # each projection's block rows
    values: Mapping[str, Mapping[str, Any]]           # each table's entry, `ENTRY`'s fields
    nonces: Mapping[str, int]                         # each table's salt nonce under the committer's key

    @classmethod
    def of(cls, doc: Any, where: str) -> ExpertTables:
        """The registration as its writer wrote it, refused by field (`fields.RecordError`) where it isn't one: another schema,
        a field no writer writes, no table, an entry that doesn't name its table or whose root isn't 64 bytes in hex, or nonces
        that aren't one for each table, each its own."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the registration is a {type(doc).__name__}, not a JSON object")
        if doc.get("schema") != SCHEMA:
            raise RecordError(where, "schema", f"is {doc.get('schema')!r}, not {SCHEMA}: a committer's expert tables registration")
        only(doc, ("schema", "program", "repo", "revision", "world", "rank", "prefix", "blocks", "values", "nonces"), where)
        values, nonces = required(doc, "values", where, Mapping), required(doc, "nonces", where, Mapping)
        if not values:
            raise RecordError(where, "values", "is empty: a registration registers at least one table")
        for p in values:
            e = required(values, p, where, Mapping, f"values.{p}")
            only(e, ENTRY, where, f"values.{p}.")
            if e.get("value") != p or not (isinstance(e.get("root"), str) and ROOT.fullmatch(e["root"])):
                raise RecordError(where, f"values.{p}", "doesn't name its table with a root of 64 bytes in hex")
        if set(nonces) != set(values) or not all(type(n) is int and n >= 0 for n in nonces.values()) \
                or len(set(nonces.values())) != len(nonces):
            raise RecordError(where, "nonces", "aren't one salt nonce for each table registered, each its own")
        return cls(program=required(doc, "program", where, str), repo=required(doc, "repo", where, str),
                   revision=required(doc, "revision", where, str), world=required(doc, "world", where, int),
                   rank=required(doc, "rank", where, int), prefix=required(doc, "prefix", where, str),
                   blocks=required(doc, "blocks", where, Mapping), values=values, nonces=nonces)

    def dump(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "program": self.program, "repo": self.repo, "revision": self.revision, "world": self.world,
                "rank": self.rank, "prefix": self.prefix, "blocks": dict(sorted(self.blocks.items())),
                "values": {p: dict(e) for p, e in sorted(self.values.items())}, "nonces": dict(sorted(self.nonces.items()))}
