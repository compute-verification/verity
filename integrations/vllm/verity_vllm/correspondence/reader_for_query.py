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

The record's return leaves place a Call in the runtime tensors it lands in (`call_place`): `Correspondence.rows` holds the row of
them every return of the Call lies in (an attention row's `[t=<position>]` names its position, not that row), and
`Correspondence.columns`, for a Call whose one return covers part of its row, the columns it covers (a narrowed linear's rows:
`program.prune.kept_records` cuts their leaves to the columns read).  The fallback records neither.

The second correspondence fact is the map from a Program module path to the IMPLEMENTATION's path (the collector's `named_modules()`
name): the export wrapper holds the served model under an attribute, so `model.model.layers.0...` in the Program is `model.layers.0...` at
the collector.  It is determined from the Build's implementation paths (`result.json: attention_impls_observed`, or predict's
`attention_impls_declared`, keys = implementation module paths) by suffix match against the Program's module paths -- the one prefix every observed path is a suffix under
-- never assumed.

A HELD module path (`held_module_paths`) is resolved to the registered module it names: a module that another module holds under an
attribute and calls by that name, outside every registered path (GLM-5.3's MoE runner calls the layer's shared expert as
`_shared_experts._layer`, so a Build before the derive fix named all 75 layers' shared-expert Calls `_shared_experts._layer.<leaf>`).
Held means the path's first component is no registered parameter's first component.  A held Call that reads parameters resolves to the
module they are registered under when that module ends with the same leaf (`_shared_experts._layer.down_proj` reading
`model.layers.4.mlp.shared_experts.down_proj.weight` is `model.layers.4.mlp.shared_experts.down_proj`: only that module reads its weight),
which maps the holder `_shared_experts._layer` to `model.layers.4.mlp.shared_experts` for that Call; a held Call that reads no parameter
resolves through its producers' holders (the shared expert's SiLU reads that layer's `gate_up_proj` rows, so it is that layer's
`act_fn`).  Several or no candidates leave the path as recorded and count it (`provenance()["held_modules"]`): coverage then fails it.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Mapping, Sequence

from verity_vllm.correspondence.runtime import SCHEMA, CallCorrespondence, RuntimeCorrespondence, correspondence_digest, read_program_document

from verity_vllm.query.program_view import ProgramView, param_reads

__all__ = ["CallName", "parse_call_name", "row_place", "call_place", "runtime_rows", "recorded_names", "Correspondence", "implementation_prefix",
           "held_module_paths", "SOURCES"]

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


def row_place(leaves: Any, shape: Any) -> tuple[int, int, int, int] | None:
    """(row, c0, c1, width): the row of a row-major tensor of `shape` that leaves [lo, hi) lie in, the columns [c0, c1) of it they
    cover and the row's width; None when they are not located or span rows."""
    if leaves is None or not shape:
        return None
    w = 1
    for s in shape[1:]:
        w *= int(s)
    lo, hi = int(leaves[0]), int(leaves[1])
    if w <= 0 or hi <= lo or lo // w != (hi - 1) // w:
        return None
    return lo // w, lo % w, lo % w + hi - lo, w


def call_place(slots: Iterable[tuple[Any, Any]]) -> tuple[int | None, tuple[int, int] | None]:
    """A Call's row and columns from its located return slots, (leaves, shape of the tensor they land in) each: the row of that tensor
    every slot lies in, when one; and for a Call of one slot that covers part of the row, the columns it covers.  (None, None) when no
    slot is located or one spans rows."""
    places = [row_place(leaves, shape) for leaves, shape in slots]
    if not places or None in places or len({p[0] for p in places}) != 1:
        return None, None
    r, c0, c1, w = places[0]
    return r, ((c0, c1) if len(places) == 1 and c1 - c0 < w else None)


def runtime_rows(at: Sequence[int | None]) -> list[int] | None:
    """`[r0, r0 + n)`: the rows of their runtime tensor n Calls' Values are, when the Calls' rows (`call_place`) are r0, r0 + 1, ... in
    Program order with r0 > 0 (the last layer's prompt step, whose rows before the last `dead_free` removed); None when they start at
    row 0, are not consecutive or one is not placed."""
    if not at or at[0] is None or at[0] <= 0 or any(r != at[0] + i for i, r in enumerate(at)):
        return None
    return [at[0], at[0] + len(at)]


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


def recorded_names(doc: Mapping[str, Any]) -> tuple[dict[int, CallName], list[int], dict[int, int], dict[int, tuple[int, int]]]:
    """`call_name_of` of each Call record of a `runtime-correspondence/v1` document, the tensorless Calls (every return slot neither
    an fx output nor a written argument), and each placed Call's row and columns (`call_place`), read off the fields
    `RuntimeCorrespondence.records` reads them from without building the records: their argument slots (a request Program's are
    millions) are read by neither."""
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document (schema={doc.get('schema')!r})")
    occs = doc["occurrences"]
    names: dict[int, CallName] = {}
    tensorless: list[int] = []
    rows: dict[int, int] = {}
    columns: dict[int, tuple[int, int]] = {}
    for k, c in enumerate(doc["calls"]):
        o = (occs[c["occ"]] if c.get("occ") is not None else None) or {}
        module, member, hint = o.get("module", ""), c.get("member"), c.get("hint")
        coll = o.get("collective")
        kind = coll["kind"] if coll else None
        if hint:
            n = parse_call_name(hint)
            names[k] = CallName(module, n.node, n.hint, n.row, n.suffix)
        else:
            row = c.get("row")
            row = str(row) if row is not None else None
            if coll and member == kind:
                names[k] = CallName(module, o.get("node") or "", "", row, member)
            else:
                names[k] = CallName(module, o.get("node") or "", member or "", row, "")
        rets = c.get("returns")
        if rets and all(r.get("output") is None and r.get("written_arg") is None for r in rets):
            tensorless.append(k)
        elif rets:
            outs, ins = o.get("outputs") or [], o.get("inputs") or []

            def shape(r: Mapping[str, Any]) -> Any:
                if r.get("output") is not None:
                    return (outs[r["output"]] or {}).get("shape") if r["output"] < len(outs) else None
                wa = r.get("written_arg")
                return ((ins[wa] or {}).get("view") or {}).get("shape") if wa is not None and wa < len(ins) else None

            r, cols = call_place([(x["leaves"], shape(x)) for x in rets if x.get("leaves") is not None])
            if r is not None:
                rows[k] = r
            if cols is not None:
                columns[k] = cols
    return names, tensorless, rows, columns


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
        # the path itself, else its nearest parent down to the layer that is a Program module: GLM-5.3's MLA attention is a custom op
        # inside `self_attn`, so `model.layers.0.self_attn.attn` is observed and `inner.model.model.layers.0.self_attn` is the module
        parts = a.split(".")
        for t in [a, *(".".join(parts[:k]) for k in range(len(parts) - 1, 2, -1))]:
            for m in mods:
                if m == t:
                    c.add("")
                elif m.endswith("." + t):
                    c.add(m[: len(m) - len(t)])
            if c:
                break
        candidates.append(c)
    common = set.intersection(*candidates) if candidates else set()
    if not common:
        return ""
    return min(common, key=len)


def _holder(module: str, registered: str) -> tuple[str, str] | None:
    """(held holder, registered holder) when `module` and `registered` end with the same components below a non-empty held holder."""
    a, b = module.split("."), registered.split(".")
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    if k == 0 or k == len(a) or k == len(b):
        return None
    return ".".join(a[:-k]), ".".join(b[:-k])


def held_module_paths(P: ProgramView, names: Mapping[int, CallName]) -> tuple[dict[int, CallName], dict[str, Any]]:
    """`names` with every held module path resolved (module docstring), and what was resolved: {"resolved": n, "unresolved": {path: n}}
    (empty when the Program has no held path whose holder another Call resolved)."""
    fields = {(p.id, port.name): port.name for p in P.params for port in p.port_table if "." in port.name}
    roots = {f.split(".", 1)[0] for f in fields.values()}
    if not roots:
        return dict(names), {}
    out = dict(names)
    holder_of: dict[int, tuple[str, str]] = {}           # resolved held Call -> (held holder, registered holder)
    holders: set[str] = set()                            # the held holders some Call resolved
    unresolved: Counter = Counter()
    held: dict[str, bool] = {}
    for cid, n in names.items():
        m = n.module
        if not m or not held.setdefault(m, m.split(".", 1)[0] not in roots):
            continue
        c = P.call(cid)
        pairs = {_holder(m, f.rsplit(".", 1)[0]) for v in param_reads(c) if (f := fields.get((v.call, v.port))) is not None}
        if not pairs:
            pairs = {holder_of[v.call] for v in c.reads if v.call >= 0 and v.call in holder_of}
            pairs = {(h, x) for h, x in pairs if m == h or m.startswith(h + ".")}
        if len(pairs) != 1 or None in pairs:
            if any(m == h or m.startswith(h + ".") for h in holders):
                unresolved[m] += 1
            continue                                     # else no holder of this path resolved: not a held path we know
        h, x = next(iter(pairs))
        holder_of[cid] = (h, x)
        holders.add(h)
        out[cid] = dataclasses.replace(n, module=x + m[len(h):])
    if not holder_of:
        return out, {}
    return out, {"resolved": len(holder_of), "unresolved": dict(unresolved)}


class Correspondence:
    """The correspondence table of one ProgramView: Call id -> CallName (module first), plus the implementation-path map.

    `table`: the Program's `runtime-correspondence/v1` record (`source = correspondence`), or its `document`, whose Calls' names are
    read off its fields and no record is built (`recorded_names`; what `of` gives: a request Program's table is GBs); neither = the
    v1-annotation fallback (the Calls' names).  `Correspondence.of(P, program_dir)` reads the Build dir and picks."""

    def __init__(self, P: ProgramView, *, implementation_paths: Iterable[str] = (), table: RuntimeCorrespondence | None = None,
                 digest: str | None = None, problem: str | None = None, document: dict | None = None) -> None:
        self.P = P
        self._impl_paths = tuple(implementation_paths)
        self.table = table
        self.problem = problem
        #: the Calls the record lowers to no runtime tensor: every return slot is neither an fx output nor a written argument (a Value
        #: internal to its occurrence's lowering, `runtime.ReturnSlot`); empty under the fallback, which records no return slots
        self.tensorless: frozenset[int] = frozenset()
        #: Call -> the row of the runtime tensors its returns land in, and the columns [c0, c1) of it its one return covers when part
        #: of the row (`call_place`); empty under the fallback
        self.rows: dict[int, int] = {}
        self.columns: dict[int, tuple[int, int]] = {}
        if table is not None or document is not None:
            if table is None:
                recorded, tensorless, self.rows, self.columns = recorded_names(document)
            else:
                recorded, tensorless = {}, []
                for rec in table:
                    recorded[rec.call] = call_name_of(rec)
                    if rec.returns and all(r.output is None and r.written_arg is None for r in rec.returns):
                        tensorless.append(rec.call)
                        continue
                    r, cols = call_place([(x.leaves, x.view.shape if x.view is not None else None) for x in rec.returns if x.leaves is not None])
                    if r is not None:
                        self.rows[rec.call] = r
                    if cols is not None:
                        self.columns[rec.call] = cols
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
        self.names, self.held = held_module_paths(P, self.names)

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
        if self.held:
            out["held_modules"] = self.held
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
        """The implementation module paths a Build's `result.json` records: `attention_impls_observed` keys (a trace's), else
        `attention_impls_declared` keys (predict's, `program.predict.naming`)."""
        if not result_json:
            return ()
        obs = result_json.get("attention_impls_observed") or result_json.get("attention_impls_declared") or {}
        return tuple(str(k) for k in obs)
