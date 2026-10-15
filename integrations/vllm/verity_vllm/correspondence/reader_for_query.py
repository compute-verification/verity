"""The per-Call runtime correspondence of a derived Program for the query: ONE reader, an explicit fallback.

`Correspondence.module_of(cid)` -- the owning module path (the body the Call was lowered from, in the SOURCE module's own vocabulary:
`named_modules()` of the export wrapper; "" = the root body) -- is what `Q_module_body_v1` partitions by.  `Correspondence.name(cid)`
is the frontend's naming of the Call (module, fx node, return-slot hint, row) the v1 bridge's member naming reads.  Where they come from:

  correspondence   `source = "correspondence"`: the `runtime-correspondence/v1` record the frontend emitted into the descriptor's
                   `annotations["correspondence"]` (`verity_vllm.correspondence.runtime`, read through its one loader
                   `read_program_dir(program_dir)`): `owning_module(cid)` and the record's `node` / `member` / `row` (its raw `hint`
                   when the record did not parse the naming suffix, parsed here exactly as the fallback parses a name).
  v1_annotations   `source = "v1_annotations"`: a v1-frontend Build (no such namespace): the ONE annotation those Builds record per Call,
                   its naming hint (`annotations["node_hints"][k]` in the descriptor, `name` in the instance sequence), parsed::

                       <module path>/<fx node>[/<member hint>][<row>]        model.model.layers.0.self_attn.rotary_emb/rotary_embedding/query[7]
                       <fx node>[<row>]                                       argmax_2[0]            (a Call outside every module: the root body)

                   `<row>` is the token row (`[7]`) or the attention position (`[t=3]`); a collective's row name ends in `/all_reduce`.

The fallback is taken ONLY when the record is absent, and which reader served is recorded (`source`, `digest`) in the manifests the
bridge writes -- never silent.

The second correspondence fact is the map from a Program module path to the IMPLEMENTATION's path (the collector's `named_modules()`
name): the export wrapper holds the served model under an attribute, so `model.model.layers.0...` in the Program is `model.layers.0...` at
the collector.  It is determined from the Build's observed implementation paths (`result.json: attention_impls_observed`, keys =
implementation module paths) by suffix match against the Program's module paths -- the one prefix every observed path is a suffix under
-- never assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Mapping

from verity_vllm.correspondence.runtime import SCHEMA, CallCorrespondence, RuntimeCorrespondence, correspondence_digest, read_program_document

from verity_vllm.query.program_view import ProgramView

__all__ = ["CallName", "parse_call_name", "Correspondence", "implementation_prefix", "SOURCES"]

SOURCES = ("correspondence", "v1_annotations")
_ROW_RE = re.compile(r"\[([^\[\]]*)\]$")


@dataclass(frozen=True, slots=True)
class CallName:
    module: str                      # owning module path in the Program's (export wrapper's) vocabulary; "" = no module (root body)
    node: str                        # the fx node
    hint: str                        # the recorded return-slot hint ("" when the rule recorded none)
    row: str | None                  # the row / position label inside `[...]`, None when absent
    suffix: str                      # a trailing `/<tag>` after the row (`all_reduce`), "" otherwise


def parse_call_name(name: str) -> CallName:
    """`model.a.b/node/hint[7]` -> CallName("model.a.b", "node", "hint", "7", "")."""
    suffix = ""
    row = None
    m = _ROW_RE.search(name)
    if m is None and "]/" in name:                         # `<...>[i]/all_reduce`
        head, _, suffix = name.rpartition("/")
        m = _ROW_RE.search(head)
        if m is not None:
            name = head
    if m is not None:
        row = m.group(1)
        name = name[: m.start()]
    parts = name.split("/")
    if len(parts) == 1:
        return CallName("", parts[0], "", row, suffix)
    module, node = parts[0], parts[1]
    hint = parts[2] if len(parts) > 2 else ""
    return CallName(module, node, hint, row, suffix)


def call_name_of(rec: CallCorrespondence) -> CallName:
    """The record's naming of a Call as a CallName.  When the record kept the frontend's naming suffix RAW (`hint`: a form it did not parse
    into member / row -- `.../<fx>/<member>[row]`, `[t=3]`, a Call lowered from no source node) it is parsed here exactly as the fallback
    parses a name, the owning module always the record's.  A collective's `member` is its kind -- the `/<kind>` tag after the row
    (`<module>/all_reduce2_7[0]/all_reduce`), the fallback's `suffix`, never a return-slot hint."""
    if rec.hint:
        n = parse_call_name(rec.hint)
        return CallName(rec.module, n.node, n.hint, n.row, n.suffix)
    row = str(rec.row) if rec.row is not None else None
    if rec.collective is not None and rec.member == rec.collective.kind:
        return CallName(rec.module, rec.node or "", "", row, rec.member)
    return CallName(rec.module, rec.node or "", rec.member or "", row, "")


def implementation_prefix(program_modules: Iterable[str], implementation_paths: Iterable[str]) -> str:
    """The wrapper prefix `p` such that an observed implementation path `a` is the Program module path `p + a`: the shortest prefix
    under which EVERY observed path is found as a suffix of some Program module path.  "" when nothing is observed or the
    Program's paths are already the implementation's."""
    mods = [m for m in set(program_modules) if m]
    obs = [a for a in set(implementation_paths) if a]
    if not obs or not mods:
        return ""
    candidates: list[set[str]] = []
    for a in obs:
        c = set()
        for m in mods:
            if m == a:
                c.add("")
            elif m.endswith("." + a):
                c.add(m[: len(m) - len(a)])
        candidates.append(c)
    common = set.intersection(*candidates) if candidates else set()
    if not common:
        return ""
    return min(common, key=len)


class Correspondence:
    """The correspondence table of one ProgramView: Call id -> CallName (module first), plus the implementation-path map.

    `table`: the Program's `runtime-correspondence/v1` record (`source = correspondence`), or its `document`, whose Call records are
    read one at a time and not held (what `of` gives: a request Program's table is GBs); neither = the v1-annotation fallback (the
    Calls' names).  `Correspondence.of(P, program_dir)` reads the Build dir and picks."""

    def __init__(self, P: ProgramView, *, implementation_paths: Iterable[str] = (), table: RuntimeCorrespondence | None = None,
                 digest: str | None = None, problem: str | None = None, document: dict | None = None) -> None:
        self.P = P
        self._impl_paths = tuple(implementation_paths)
        self.table = table
        self.problem = problem
        #: the Calls the record lowers to no runtime tensor: every return slot is neither an fx output nor a written argument (a Value
        #: internal to its occurrence's lowering, `runtime.ReturnSlot`); empty under the fallback, which records no return slots
        self.tensorless: frozenset[int] = frozenset()
        if table is not None or document is not None:
            records = iter(table) if table is not None else RuntimeCorrespondence.records(document)
            recorded, tensorless = {}, []
            for rec in records:
                recorded[rec.call] = call_name_of(rec)
                if rec.returns and all(r.output is None and r.written_arg is None for r in rec.returns):
                    tensorless.append(rec.call)
            if len(recorded) != len(P.calls):
                raise ValueError(f"correspondence record has {len(recorded)} Calls, the Program {len(P.calls)}: the record is of another Program")
            self.names: dict[int, CallName] = {c.id: recorded[c.id] for c in P.calls}
            self.tensorless = frozenset(tensorless)
            self.source = "correspondence"
            self.digest: str | None = digest or correspondence_digest(table.doc if table is not None else document)
        else:
            self.names = {c.id: parse_call_name(c.name) for c in P.calls}
            self.source = "v1_annotations"
            self.digest = None

    @classmethod
    def of(cls, P: ProgramView, program_dir: str | None, *, implementation_paths: Iterable[str] = ()) -> "Correspondence":
        """The Program's correspondence from its Build dir: the record when the Build carries it, else the v1-annotation fallback.  A
        record that fails to load / mismatches its recorded digest is a `problem` carried on the table (and into the manifest header)."""
        document, digest, problem = read_program_document(program_dir) if program_dir else (None, None, None)
        return cls(P, implementation_paths=implementation_paths, document=document, digest=digest, problem=problem)

    def module_of(self, cid: int) -> str:
        return self.names[cid].module

    def name(self, cid: int) -> CallName:
        return self.names[cid]

    def site_path(self, cid: int) -> str:
        """The path of a MID-module collective SITE (a collective whose rank-local operand another module produced, called from its
        module's own forward: an attention module gathering its q / k): `<module>.<collective fx node>`, the v1 frontend's spelling.  The
        fallback's names already spell it (the frontend named the Call under it); the record's owning module is the module itself, so the
        path is composed from the record's (module, node)."""
        n = self.names[cid]
        return n.module if self.source != "correspondence" else f"{n.module}.{n.node}"

    def provenance(self) -> dict[str, Any]:
        """What served the owning modules: recorded in the manifest header (`correspondence`)."""
        out: dict[str, Any] = {"source": self.source, "schema": SCHEMA if self.source == "correspondence" else None, "digest": self.digest}
        if self.source != "correspondence":
            out["fallback"] = "annotations.node_hints / instance names parsed as <module>/<fx>[/<member>][<row>]"
        if self.problem:
            out["problem"] = self.problem
        return out

    @cached_property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted({n.module for n in self.names.values()}))

    @cached_property
    def prefix(self) -> str:
        return implementation_prefix(self.modules, self._impl_paths)

    def implementation_path(self, module: str) -> str:
        """Program module path -> the implementation's (collector's) path."""
        p = self.prefix
        if p and module[: len(p)] == p:
            return module[len(p):]
        return module

    @staticmethod
    def implementation_paths_of(result_json: Mapping[str, Any] | None) -> tuple[str, ...]:
        """The observed implementation module paths a Build's `result.json` records (`attention_impls_observed` keys)."""
        if not result_json:
            return ()
        obs = result_json.get("attention_impls_observed") or {}
        return tuple(str(k) for k in obs)
