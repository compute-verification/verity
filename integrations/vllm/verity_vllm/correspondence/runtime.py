"""Runtime correspondence (`runtime-correspondence/v1`): per Call, the facts that resolve a logical Value of the Program to a
runtime / source occurrence -- emitted by the frontend as data, carried in the descriptor's `annotations["correspondence"]`, and
digested SEPARATELY from the Program (`correspondence_digest`; `program_digest` excludes annotations).

The record answers one question and nothing else: *which runtime occurrence does this Call's Value correspond to?*  It carries
the owning module path (the body the Call was lowered from), the fx node (the source occurrence), the return slot / argument
slots for each Value the Call produces / consumes (leaf ranges within the occurrence's tensors), the tensor views the frontend
saw (shape, dtype, strides, contiguity), the mutation / lifetime facts visible at lowering (in-place op, alias of an input,
view-of), and -- for a rank Program of a tensor-parallel world -- the rank coordinate and the collective occurrence mapping
(explicit integers, never a string prefix).  It carries NO verification policy, NO requiredness, NO commitment layout.

Layout (two levels; a Call's record is the join): `occurrences[o]` is one lowered fx node (module, op, views, lifetime,
collective); `calls[k]` is Call k (= node k of the program function's body, the same index `instances.json.gz` rows carry)
with its occurrence, return slots and argument slots.  `load_correspondence(descriptor)` joins them into `CallCorrespondence`
records and answers `owning_module(call_id)`, `return_slot(call_id, port)`, `arg_slot(call_id, input_index)`, and the
module-body partition `calls_of_module(path)` the query lane consumes.  Pure `verity.ml.circuits`-free stdlib: imports without torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

SCHEMA = "runtime-correspondence/v1"
NAMESPACE = "correspondence"   # the one annotation key the frontend writes the record under

# ---------------------------------------------------------------------------------------------------------------------------
# JSON schema (draft 2020-12 vocabulary; the document `annotations["correspondence"]` validates against it)
# ---------------------------------------------------------------------------------------------------------------------------

_VIEW = {"type": ["object", "null"],
         "properties": {"shape": {"type": "array", "items": {"type": "integer"}}, "dtype": {"type": "string"},
                        "strides": {"type": ["array", "null"], "items": {"type": "integer"}}, "contiguous": {"type": ["boolean", "null"]},
                        "storage_offset": {"type": ["integer", "null"]}},
         "required": ["shape", "dtype"]}

_RANGE = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}

JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA,
    "title": "runtime correspondence of one Program (per Call: owning module, source occurrence, return / argument slots)",
    "type": "object",
    "required": ["schema", "body", "inputs", "occurrences", "calls"],
    "properties": {
        "schema": {"const": SCHEMA},
        "body": {"const": "program_fn", "description": "Call k = node k of the program function's body (instances.json.gz row k)"},
        "inputs": {"type": "array", "items": {"type": "string"}, "description": "user input parameter names, in parameter order"},
        "weights": {"type": ["array", "null"], "items": {"type": "string"}, "description": "fields of the trailing `weights` record, in field order"},
        "rank": {"type": ["object", "null"], "required": ["world", "rank"],
                 "properties": {"world": {"type": "integer", "minimum": 1}, "rank": {"type": "integer", "minimum": 0}},
                 "description": "rank coordinate of a tensor-parallel rank Program; null when the Program is not one rank of a world"},
        "occurrences": {"type": "array", "items": {
            "type": "object", "required": ["node", "module", "op"],
            "properties": {
                "node": {"type": "string", "description": "fx node name of the runtime occurrence"},
                "module": {"type": "string", "description": "owning module path in the source module's own vocabulary ('' = root)"},
                "module_class": {"type": ["string", "null"]},
                "op": {"type": "string", "description": "the source op (OpOverload / callable) the node calls"},
                "rule": {"type": ["string", "null"], "description": "lowering rule that lowered the node (rule id + version)"},
                "outputs": {"type": "array", "items": _VIEW, "description": "tensor view of each output (one per fx output port)"},
                "written": {"type": "array", "items": {"type": "object", "required": ["arg", "node"], "properties": {
                    "arg": {"type": "integer", "description": "index into `inputs`"}, "node": {"type": "string"}, "view": _VIEW}},
                            "description": "fx arguments whose storage the lowering WROTE (an in-place op): a return slot may land in one of these"},
                "inputs": {"type": "array", "items": {
                    "type": "object", "required": ["node", "source"],
                    "properties": {"node": {"type": "string"}, "arg": {"type": ["integer", "null"], "description": "flat position among the node's positional fx-node arguments (null for a keyword argument)"},
                                   "kwarg": {"type": ["string", "null"], "description": "keyword the argument was passed by, when it was"},
                                   "source": {"enum": ["input", "weight", "call", "constant", "other"]},
                                   "name": {"type": ["string", "null"], "description": "input name / weight field for source input / weight"},
                                   "view": _VIEW}}},
                "lifetime": {"type": "object", "properties": {
                    "in_place": {"type": "boolean"}, "view_of": {"type": ["string", "null"]}, "aliases_input": {"type": ["string", "null"]},
                    "aliases_weight": {"type": ["string", "null"]}, "rebound": {"type": "boolean"}, "stale": {"type": "boolean"}}},
                "collective": {"type": ["object", "null"], "required": ["kind", "occurrence", "operand_ranks"], "properties": {
                    "kind": {"type": "string"}, "occurrence": {"type": "integer", "description": "k-th collective of this rank's forward, in graph order"},
                    "operand_ranks": {"type": "array", "items": {"type": "integer"}, "description": "rank that produced operand i"},
                    "own_operand": {"type": ["integer", "null"], "description": "operand index computed by this rank (structural: not a Program input)"},
                    "peer_operand": {"type": ["integer", "null"]}, "peer_input": {"type": ["string", "null"], "description": "user input carrying the peer's partial"},
                    "peer_operands": {"type": "array", "items": {"type": "integer"}, "description": "world > 2: every peer's operand index, in rank order"},
                    "peer_inputs": {"type": "array", "items": {"type": ["string", "null"]}, "description": "world > 2: the user input carrying each peer's partial"}}}}}},
        "calls": {"type": "array", "items": {
            "type": "object", "required": ["occ"],
            "properties": {
                "occ": {"type": ["integer", "null"], "description": "index into `occurrences`; null = the Call was not lowered from a source node"},
                "member": {"type": ["string", "null"], "description": "frontend member hint of the produced Value (a component name the rule gave it)"},
                "row": {"type": ["integer", "null"], "description": "row of the occurrence's output this Call produces, when the rule split the node by rows"},
                "hint": {"type": ["string", "null"], "description": "the frontend's raw naming suffix when it does not parse into member / row"},
                "returns": {"type": "array", "items": {"type": "object", "required": ["port"], "properties": {
                    "port": {"type": "integer", "description": "flat tensor chunk of the Call's return type"},
                    "output": {"type": ["integer", "null"], "description": "fx output port of the occurrence the Value lands in"},
                    "written_arg": {"type": "integer", "description": "instead of an output: the `inputs` index whose storage the Value lands in (in-place lowering)"},
                    "leaves": {"type": ["array", "null"], "items": {"type": "integer"},
                               "description": "[lo, hi) leaves of that output / written storage (row-major) this port occupies; null = not located "
                                              "(a Value internal to the lowering, consumed only by sibling Calls of the same occurrence)"}}}},
                "args": {"type": "array", "items": {"type": "object", "required": ["input"], "properties": {
                    "input": {"type": ["integer", "null"], "description": "index into the occurrence's `inputs`; null = not one source tensor"},
                    "leaves": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "[lo, hi) leaves of that input this argument reads (null = several ranges)"},
                    "spans": {"type": "array", "items": {"type": "object", "required": ["input", "leaves"], "properties": {"input": {"type": "integer"}, "leaves": _RANGE}},
                              "description": "when the argument spans SEVERAL fx inputs (a collective's `own ++ peer` operand): in argument order, the input and leaf range of each piece"}}},
                         "description": "one per argument of the Call, in argument order.  WHICH Calls / inputs / weight fields the argument's leaves come "
                                        "from is the Program's own data (the body's reference sequences) and is not repeated here"}}}},
    },
}


# ---------------------------------------------------------------------------------------------------------------------------
# records (the joined per-Call view a consumer reads)
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorView:
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...] | None = None
    contiguous: bool | None = None
    storage_offset: int | None = None

    @classmethod
    def from_json(cls, d: dict | None) -> "TensorView | None":
        if d is None:
            return None
        return cls(tuple(d["shape"]), d["dtype"], tuple(d["strides"]) if d.get("strides") is not None else None,
                   d.get("contiguous"), d.get("storage_offset"))


@dataclass(frozen=True)
class ReturnSlot:
    """Where port `port` of a Call's return Value lands at runtime: output `output` of fx node `node` -- or, for an in-place lowering,
    the storage of the occurrence's fx argument `written_arg` (then `written_node` names it) -- leaves [lo, hi) of it.  Both None with
    `leaves` None = a Value internal to the lowering (consumed only by sibling Calls of the same occurrence): no runtime tensor."""
    call: int
    port: int
    node: str
    output: int | None
    leaves: tuple[int, int] | None
    member: str | None
    row: int | None
    view: TensorView | None
    written_arg: int | None = None
    written_node: str | None = None

    @property
    def located(self) -> bool:
        return self.leaves is not None


@dataclass(frozen=True)
class ArgSlot:
    """Where argument `index` of a Call is read from at runtime: fx node `node` (position `arg` among the occurrence's fx
    arguments), leaves [lo, hi) of it; `source` says whether that tensor is a user input, a weight field, another Call's output, ..."""
    call: int
    index: int
    node: str | None
    arg: int | None
    source: str | None
    name: str | None
    leaves: tuple[int, int] | None
    view: TensorView | None
    spans: tuple["ArgSpan", ...] = ()     # non-empty exactly when the argument is read from several fx inputs (then node/arg/leaves are None)


@dataclass(frozen=True)
class ArgSpan:
    """One piece of an argument that spans several fx inputs of its occurrence: fx node `node` (position `arg`), leaves [lo, hi)."""
    node: str | None
    arg: int | None
    source: str | None
    name: str | None
    leaves: tuple[int, int]
    view: TensorView | None


@dataclass(frozen=True)
class RankOccurrence:
    world: int
    rank: int


@dataclass(frozen=True)
class CollectiveOccurrence:
    kind: str
    occurrence: int
    operand_ranks: tuple[int, ...]
    own_operand: int | None
    peer_operand: int | None
    peer_input: str | None
    peer_operands: tuple[int, ...] = ()
    peer_inputs: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class CallCorrespondence:
    call: int
    module: str
    module_class: str | None
    node: str | None
    op: str | None
    rule: str | None
    member: str | None
    row: int | None
    hint: str | None
    returns: tuple[ReturnSlot, ...]
    args: tuple[ArgSlot, ...]
    lifetime: dict
    rank: RankOccurrence | None
    collective: CollectiveOccurrence | None


@dataclass
class RuntimeCorrespondence:
    """The loaded table: Call -> record, module path -> Calls."""
    doc: dict
    _calls: dict[int, CallCorrespondence] = field(default_factory=dict, repr=False)
    _by_module: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # -- construction -----------------------------------------------------------------------------------------------------
    @classmethod
    def from_json(cls, doc: dict) -> "RuntimeCorrespondence":
        t = cls(doc)
        for k, rec in enumerate(cls.records(doc)):
            t._calls[k] = rec
            t._by_module.setdefault(rec.module, []).append(k)
        return t

    @staticmethod
    def records(doc: dict) -> Iterator[CallCorrespondence]:
        """The document's Call records in Call order, each built when it is reached (`from_json` holds them all)."""
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} document (schema={doc.get('schema')!r})")
        rank = RankOccurrence(int(doc["rank"]["world"]), int(doc["rank"]["rank"])) if doc.get("rank") else None
        occs = doc["occurrences"]
        for k, c in enumerate(doc["calls"]):
            o = occs[c["occ"]] if c.get("occ") is not None else None
            outs = (o or {}).get("outputs") or []
            ins = (o or {}).get("inputs") or []
            rets = []
            for r in c.get("returns") or ():
                wa = r.get("written_arg")
                win = ins[wa] if wa is not None and wa < len(ins) else None
                view = None
                if r.get("output") is not None and r["output"] < len(outs):
                    view = TensorView.from_json(outs[r["output"]])
                elif win is not None:
                    view = TensorView.from_json(win.get("view"))
                rets.append(ReturnSlot(k, int(r["port"]), (o or {}).get("node", ""), r.get("output"),
                                       tuple(r["leaves"]) if r.get("leaves") is not None else None, c.get("member"), c.get("row"), view,
                                       wa, (win or {}).get("node")))
            rets = tuple(rets)
            args = []
            for i, a in enumerate(c.get("args") or ()):
                src = ins[a["input"]] if a.get("input") is not None and a["input"] < len(ins) else None
                spans = []
                for s in a.get("spans") or ():
                    si = ins[s["input"]] if s.get("input") is not None and s["input"] < len(ins) else {}
                    spans.append(ArgSpan(si.get("node"), si.get("arg"), si.get("source"), si.get("name"), tuple(s["leaves"]), TensorView.from_json(si.get("view"))))
                args.append(ArgSlot(k, i, (src or {}).get("node"), (src or {}).get("arg"), (src or {}).get("source"), (src or {}).get("name"),
                                    tuple(a["leaves"]) if a.get("leaves") is not None else None,
                                    TensorView.from_json((src or {}).get("view")), tuple(spans)))
            coll = None
            if o and o.get("collective"):
                cj = o["collective"]
                coll = CollectiveOccurrence(cj["kind"], int(cj["occurrence"]), tuple(int(x) for x in cj["operand_ranks"]),
                                            cj.get("own_operand"), cj.get("peer_operand"), cj.get("peer_input"),
                                            tuple(int(x) for x in cj.get("peer_operands") or ()), tuple(cj.get("peer_inputs") or ()))
            yield CallCorrespondence(k, (o or {}).get("module", ""), (o or {}).get("module_class"), (o or {}).get("node"), (o or {}).get("op"),
                                     (o or {}).get("rule"), c.get("member"), c.get("row"), c.get("hint"), rets, tuple(args),
                                     dict((o or {}).get("lifetime") or {}), rank, coll)

    # -- lookups ----------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._calls)

    def __iter__(self) -> Iterator[CallCorrespondence]:
        return iter(self._calls[k] for k in sorted(self._calls))

    def record(self, call_id: int) -> CallCorrespondence:
        return self._calls[int(call_id)]

    def owning_module(self, call_id: int) -> str:
        """The module path (body) Call `call_id` was lowered from ('' = the root body)."""
        return self._calls[int(call_id)].module

    def return_slot(self, call_id: int, port: int = 0) -> ReturnSlot:
        rec = self._calls[int(call_id)]
        for r in rec.returns:
            if r.port == port:
                return r
        raise KeyError(f"Call {call_id} has no return port {port} (ports: {[r.port for r in rec.returns]})")

    def arg_slot(self, call_id: int, input_index: int) -> ArgSlot:
        rec = self._calls[int(call_id)]
        if not 0 <= input_index < len(rec.args):
            raise KeyError(f"Call {call_id} has {len(rec.args)} arguments, no argument {input_index}")
        return rec.args[input_index]

    def calls_of_module(self, module: str) -> list[int]:
        """Every Call whose owning module path is exactly `module` (the module-body partition)."""
        return list(self._by_module.get(module, ()))

    def calls_under(self, prefix: str) -> list[int]:
        """Every Call whose owning module path is `prefix` or lies below it."""
        out: list[int] = []
        for m, ks in self._by_module.items():
            if m == prefix or (prefix and m.startswith(prefix + ".")) or prefix == "":
                out.extend(ks)
        return sorted(out)

    def modules(self) -> list[str]:
        return sorted(self._by_module)

    @property
    def rank(self) -> RankOccurrence | None:
        d = self.doc.get("rank")
        return RankOccurrence(int(d["world"]), int(d["rank"])) if d else None

    def collective_calls(self) -> list[int]:
        return sorted(k for k, r in self._calls.items() if r.collective is not None)


# ---------------------------------------------------------------------------------------------------------------------------
# digest and loading
# ---------------------------------------------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def correspondence_digest(doc_or_desc_or_program: Any) -> str | None:
    """sha256 of the canonical JSON of the `runtime-correspondence/v1` document.  Accepts the document itself, a descriptor dict
    (reads `annotations["correspondence"]`) or a `Program` / `DecodedProgram` (reads `.annotations`).  None when absent."""
    doc = _document_of(doc_or_desc_or_program)
    if doc is None:
        return None
    return hashlib.sha256(canonical_json(doc)).hexdigest()


def _document_of(x: Any) -> dict | None:
    if x is None:
        return None
    if isinstance(x, dict):
        if x.get("schema") == SCHEMA:
            return x
        ann = x.get("annotations")
        if isinstance(ann, dict):
            return ann.get(NAMESPACE)
        return None
    ann = getattr(x, "annotations", None)
    if isinstance(ann, dict):
        return ann.get(NAMESPACE)
    return None


def load_correspondence(descriptor_or_program: Any) -> RuntimeCorrespondence:
    """The table of a descriptor dict (`annotations["correspondence"]`), a `Program` / `DecodedProgram`, or the document itself."""
    doc = _document_of(descriptor_or_program)
    if doc is None:
        raise KeyError(f"no {NAMESPACE!r} annotation ({SCHEMA}) on the descriptor / program")
    return RuntimeCorrespondence.from_json(doc)


_CHUNK = 1 << 22


class _Stream:
    """A JSON text read in chunks with a cursor: `raw_decode` at the cursor, refilling until the value is complete."""

    def __init__(self, fh) -> None:
        self.f, self.buf, self.pos, self.eof = fh, "", 0, False
        self.dec = json.JSONDecoder()

    def _more(self) -> bool:
        if self.eof:
            return False
        chunk = self.f.read(_CHUNK)
        if not chunk:
            self.eof = True
            return False
        if self.pos > _CHUNK:
            self.buf, self.pos = self.buf[self.pos:] + chunk, 0
        else:
            self.buf += chunk
        return True

    def peek(self) -> str:
        """The next non-whitespace character (not consumed); "" at the end of the text."""
        while True:
            b, p = self.buf, self.pos
            while p < len(b) and b[p] in " \n\r\t":
                p += 1
            self.pos = p
            if p < len(b):
                return b[p]
            if not self._more():
                return ""

    def take(self, ch: str) -> None:
        c = self.peek()
        if c != ch:
            raise ValueError(f"descriptor: expected {ch!r} at offset {self.pos}, found {c!r}")
        self.pos += 1

    def value(self) -> Any:
        """One complete JSON value at the cursor, decoded by the C scanner (the text is refilled until it parses; a value ending
        exactly at the buffer's end is re-read with more text, so a number is never cut)."""
        self.peek()
        while True:
            try:
                obj, end = self.dec.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                if not self._more():
                    raise
                continue
            if end >= len(self.buf) and self._more():
                continue
            # a number cut inside itself (`1.5e|-07`, `123|45`) decodes as a valid prefix: what follows a complete number is never a digit / e / sign / dot
            if type(obj) in (int, float) and end < len(self.buf) and self.buf[end] in "0123456789.eE+-":
                if self._more():
                    continue
                raise ValueError(f"descriptor: malformed number at offset {self.pos}")
            self.pos = end
            return obj


def _walk(s: _Stream, build: bool) -> Any:
    """The value at the cursor: an object is walked key by key and an array element by element (each element one `raw_decode`), so
    a container's text is never held whole; `build` assembles the value, else it is discarded as it goes."""
    c = s.peek()
    if c == "{":
        s.pos += 1
        out: dict | None = {} if build else None
        while True:
            c = s.peek()
            if c == "}":
                s.pos += 1
                return out
            if c == ",":
                s.pos += 1
                continue
            key = s.value()
            s.take(":")
            v = _walk(s, build)
            if build:
                out[key] = v
    if c == "[":
        s.pos += 1
        lst: list | None = [] if build else None
        while True:
            c = s.peek()
            if c == "]":
                s.pos += 1
                return lst
            if c == ",":
                s.pos += 1
                continue
            v = s.value()
            if build:
                lst.append(v)
    return s.value()


def read_descriptor_annotation(desc_path: str, namespace: str = NAMESPACE) -> dict | None:
    """`annotations[namespace]` of a descriptor file (`descriptor.json[.gz]`) WITHOUT loading the descriptor: the top-level object is
    walked key by key, `definitions` (the Program body -- a request Program's is GBs of reference runs) is streamed through and
    dropped, only `annotations` is built.  None when the descriptor carries no `annotations` or no such namespace.  The value is
    exactly `json.load(f)["annotations"][namespace]` (`correspondence_digest` of it is the recorded digest)."""
    import gzip
    opener = gzip.open(desc_path, "rt", encoding="utf-8") if str(desc_path).endswith(".gz") else open(desc_path, encoding="utf-8")
    with opener as fh:
        s = _Stream(fh)
        s.take("{")
        found: dict | None = None
        while True:
            c = s.peek()
            if c == "}":
                break
            if c == ",":
                s.pos += 1
                continue
            key = s.value()
            s.take(":")
            if key == "annotations":
                found = _walk(s, True)
            else:
                _walk(s, False)
    if not isinstance(found, dict):
        return None
    return found.get(namespace)


def read_program_dir(program_dir: Any) -> tuple["RuntimeCorrespondence | None", str | None, str | None]:
    """The table of one Build output dir (`derive_step`: `result.json` + `descriptor.json[.gz]`), as (table, digest, problem).
    `result.json:program.correspondence_digest` is the cheap presence test (the frontend that emits the record records its digest there):
    a Build without it is a v1-frontend Build -> (None, None, None) without decoding the descriptor.  With it, the descriptor's record is
    loaded and its digest checked against the recorded one (a mismatch is the `problem`; the table is still returned).  The ONE loader
    every consumer (acquire, query, verdict) reads a Build's correspondence through; `read_program_document` is its document, for a
    reader that holds no table (`RuntimeCorrespondence.records`)."""
    doc, digest, problem = read_program_document(program_dir)
    return (RuntimeCorrespondence.from_json(doc) if doc is not None else None), digest, problem


def read_program_document(program_dir: Any) -> tuple[dict | None, str | None, str | None]:
    """`read_program_dir`'s (document, digest, problem): the record's document, not its table."""
    import os
    d = str(program_dir)
    res_p = os.path.join(d, "result.json")
    recorded = None
    if os.path.isfile(res_p):
        try:
            with open(res_p) as f:
                recorded = (json.load(f).get("program") or {}).get("correspondence_digest")
        except (OSError, ValueError) as e:
            return None, None, f"{res_p}: unreadable ({e})"
        if recorded is None:
            return None, None, None
    desc_p = os.path.join(d, "descriptor.json.gz")
    if not os.path.isfile(desc_p):
        desc_p = os.path.join(d, "descriptor.json")
    if not os.path.isfile(desc_p):
        return None, None, None if recorded is None else f"{d}: result.json records a correspondence_digest but no descriptor is beside it"
    doc = read_descriptor_annotation(desc_p, NAMESPACE)    # the record alone: the descriptor's body is streamed through, never held
    if doc is None:
        return None, None, None if recorded is None else f"{d}: result.json records a correspondence_digest but the descriptor carries no {NAMESPACE!r} annotation"
    digest = correspondence_digest(doc)
    problem = None if recorded in (None, digest) else f"{d}: correspondence_digest {digest[:16]} != result.json's {str(recorded)[:16]}"
    return doc, digest, problem


def validate_document(doc: dict) -> list[str]:
    """Structural check of a document against the parts of JSON_SCHEMA a stdlib validator can state (required keys, the enum of
    `source`, integer ranks, dense `calls`); returns the problems (empty = valid).  A full JSON-Schema validator is not a dependency."""
    problems: list[str] = []
    for k in JSON_SCHEMA["required"]:
        if k not in doc:
            problems.append(f"missing top-level key {k!r}")
    if problems:
        return problems
    if doc["schema"] != SCHEMA:
        problems.append(f"schema {doc['schema']!r} != {SCHEMA!r}")
    if doc["body"] != "program_fn":
        problems.append(f"body {doc['body']!r} != 'program_fn'")
    r = doc.get("rank")
    if r is not None and not (isinstance(r.get("world"), int) and isinstance(r.get("rank"), int) and 0 <= r["rank"] < r["world"]):
        problems.append(f"rank coordinate {r!r} is not integers 0 <= rank < world")
    n_occ = len(doc["occurrences"])
    for o, occ in enumerate(doc["occurrences"]):
        for k in ("node", "module", "op"):
            if k not in occ:
                problems.append(f"occurrences[{o}] lacks {k!r}")
        for j, inp in enumerate(occ.get("inputs") or ()):
            if inp.get("source") not in ("input", "weight", "call", "constant", "other"):
                problems.append(f"occurrences[{o}].inputs[{j}].source {inp.get('source')!r} not in the enum")
        c = occ.get("collective")
        if c is not None:
            if not all(isinstance(x, int) for x in c.get("operand_ranks", [])) or not isinstance(c.get("occurrence"), int):
                problems.append(f"occurrences[{o}].collective has a non-integer rank coordinate: {c!r}")
    for k, c in enumerate(doc["calls"]):
        if "occ" not in c:
            problems.append(f"calls[{k}] lacks 'occ'")
        elif c["occ"] is not None and not (0 <= c["occ"] < n_occ):
            problems.append(f"calls[{k}].occ {c['occ']} out of range")
    return problems


__all__ = ["SCHEMA", "NAMESPACE", "JSON_SCHEMA", "TensorView", "ReturnSlot", "ArgSlot", "ArgSpan", "RankOccurrence", "CollectiveOccurrence",
           "CallCorrespondence", "RuntimeCorrespondence", "correspondence_digest", "load_correspondence", "read_program_dir", "read_program_document",
           "read_descriptor_annotation", "validate_document", "canonical_json"]
