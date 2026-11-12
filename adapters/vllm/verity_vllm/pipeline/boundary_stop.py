"""A serve's registered stop rule on its boundary commitment (`boundary_rows`): the rule (`Stop`), the stop block
that computes where a response ends (`egress_stop`), which class each span is filed in (`filed`), a stop class's
stop record (`stop_registration`, `stop_record`), and a statement's read of the certified words (`read_refusal`).

**The rule** is the registered model's stop ids, its length, and T, the most ids a response has. A span the rule ends as
it was served (`Stop.word`) is in the stop class of its prompt's size, (P, T) whatever n is, which commits `b.stop`: the
output of the span's stop block (`egress_stop`: `StopRule16_v1{T, NSTOP}` over the span's `b.egress` row and its
request's ``max_tokens`` and prompt length, at the registered stop ids and length), so the ids served n is a function of
the ids, the registered model and the request, and no number the serve states. Every other span is filed by its size
class with no `b.stop` and no stop block (`filed`, ``p<P>-o<O>-nostop``): an abort, an error, a request still running,
one the rule runs past T (finish 0), an end the rule doesn't read (a body's own stop strings), a ``min_tokens`` other
than 0, a request that sets ``ignore_eos``, and, where the serve knows its requests' releases, a request the network
certifier didn't release. A failure the engine doesn't see (the certifier's, after the engine's last step for the
request) is not known here: such a span is filed as the engine ended it.

**What the block reads.** Two rows, and no word of the request that the serve wrote. ``ids`` is the span's `b.egress`
row. ``words`` is its request's row of the network certifier's words table: `COLUMNS`, the request's ``max_tokens`` and
the count of its prompt's ids as the certifier parsed them from the request bytes it admitted, each a u32, one row a
release, in the class's release order. The certifier signs the table's root in its registration head. The verifier
reads the head under the certifier's key and takes the table as a certified value (`stop_registration`'s ``words``, the
verifier's own ``--registered`` port of it: block r reads row r). A stop class registers a stop record only with that
table: a class whose requests have no certified words registers none (`UNCERTIFIED`). Such a class is one behind no
network certifier, or under a schedule whose head lists no words table, as the text form's, where a request has no
certified count of ids. The id form a certified request takes has no ``ignore_eos``, so the block reads the registered
stop ids for every request.

**The stop record.** The stop blocks are a population of their own, one a span, slot r its r-th span's
(`Layout.stop_program`), registered one-stage in a record of its own that names the class's boundary record. It
registers two values. ``ids`` is that record's port `b.egress`, taken as an output value under its root and domain. The
certified table is under the root and domain the certifier signed: it is none of the program's parameters' registered
values, and R7 holds it to the verifier's certified entry (`StopRegistration.certified`). `b.stop` is the record's one
port, under the window domain of the stop population's own program, which the serve commits it under as the window
closes. The serve's `b.request` row stays a value of the boundary record, which the replay block reads; the stop record
doesn't name it.

**Refusals.** Each is a `StopRefused` under its own key, with the fix in its message: `UNCERTIFIED`, `PORT` and `COUNT`
of the table the verifier holds (`certified`), and `READ_NONE`, `READ_ROOT` and `READ_ROW` of a statement's read of it
(`read_refusal`, the reading the Lean verifier's registered reads make of the port). A stop record that registers the
table under another root than the verifier's is R7's to refuse (``R7-certified``).

**Whose rule.** The verifier's own copy, as the check template is: `stop_registration` and `stop_record` take it, as
`boundary_rows.layout_of` does, and a ``commit.json`` that names another rule is refused, so the block a stop record
proves is the one the verifier holds and never one the serve chose.

**What nothing here holds.** That span r's `b.egress` row is the response to the class's r-th release: the count of
rows and the row each block reads are held here, and which response a row is belongs to the response's own tie.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.windows import rows as CS
from verity.protocols.spot.verifier.one_stage import produced
from verity.protocols.spot.verifier.one_stage import registered as G
from verity.protocols.spot.verifier.one_stage import registration as R
from verity.protocols.spot.verifier.one_stage import served as RS

from verity_vllm.pipeline.boundary_rows import (
    EGRESS,
    SCHEMA,
    Span,
    class_size,
    layout_of,
)
from verity_vllm.records.boundary_commit import FILE as COMMIT
from verity_vllm.records.boundary_commit import BoundaryCommit

#: the stop block's two parameters: its request's certified words, and the ids it was served
WORDS, IDS = "words", "ids"
#: a certified row's columns, each a u32: the request's ``max_tokens`` and the count of its prompt's ids
COLUMNS = ("max_tokens", "prompt_ids")
#: the stop block's parameter that is a boundary value's rows, and that value; its output is `STOP`
STOP_READS = {IDS: EGRESS}
#: a `StopRefused`'s keys: of the certified table the verifier holds (`certified`)
UNCERTIFIED, PORT, COUNT = "stop-uncertified", "stop-certified-port", "stop-certified-count"
#: and of a statement's read of it (`read_refusal`)
READ_NONE, READ_ROOT, READ_ROW = "stop-read-none", "stop-read-root", "stop-read-row"


class StopRefused(ValueError):
    """A stop class's certified words refused: ``key`` is which refusal (the module docstring's), and the message,
    after the key, says what to change."""

    def __init__(self, key: str, why: str) -> None:
        super().__init__(f"{key}: {why}")
        self.key = key


def _stop_refusal(S: Mapping[str, Any]) -> str | None:
    t, stop, mml = S.get("T", 1), S.get("STOP", ()), S.get("MAX_MODEL_LEN", 1)
    if not (type(t) is int and 1 <= t < 1 << 13 and isinstance(stop, tuple) and len(stop) < 1 << 16
            and all(type(s) is int and 0 <= s < 1 << 32 for s in stop) and list(stop) == sorted(set(stop))
            and type(mml) is int and 1 <= mml < 1 << 31):
        return (f"T must be in [1, 2^13), which the stop word's n holds, STOP increasing u32 ids and MAX_MODEL_LEN in "
                f"[1, 2^31) (got T={t!r}, STOP={stop!r}, MAX_MODEL_LEN={mml!r})")
    return None


def egress_stop(outputs: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`EgressStop_v2{T, STOP, MAX_MODEL_LEN}` (words: Array<2, i32>, ids: Array<T, i32>) -> u16 at T = ``outputs``:
    `StopRule16_v1{T, NSTOP}`'s word over ``ids``, ``words``' ``max_tokens`` and count of prompt ids (`COLUMNS`), the
    stop ids STOP and MAX_MODEL_LEN: where a span's `b.egress` row stops and why, n + finish * 8192. ``words`` is the
    request's row of the network certifier's words table, a certified value and none the serve commits
    (`stop_registration`). Where STOP is empty the rule reads `stop_check.NOT_AN_ID`, no token's id, in place of a stop
    id (`StopCheck_v1`'s way). Its gates are the rule's and the constants': the registered stop ids and model length are
    statics, public in its id, where the rule takes them as inputs, since a population reads each parameter as one
    value's whole rows and the registered model's are no value."""
    from verity.definitions.registry import stop_check as SC
    from verity.definitions.registry import stop_rule as X

    I32 = X.WORDS.I32

    def signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
        return ((WORDS, Array(len(COLUMNS), I32)), (IDS, Array(S.T, I32))), X.WORDS.U16

    def body(B: Any, S: Any, words: Any, ids: Any) -> Any:
        o = X.word_ops(B)
        stops = [o.word(x) for x in S.STOP or (SC.NOT_AN_ID,)]
        return B.call(bind(X.StopRule16, T=S.T, NSTOP=len(stops)), ids, words[COLUMNS.index("max_tokens")],
                      Coll(Array(len(stops), I32), concat([w.refs for w in stops])), words[COLUMNS.index("prompt_ids")],
                      o.word(S.MAX_MODEL_LEN))

    defn = CompositeDefinition("EgressStop", 2, ("T", "STOP", "MAX_MODEL_LEN"), signature, body, register=False,
                               refuse=_stop_refusal, doc="StopRule16_v1 over a span's b.egress ids and its request's "
                               "certified max_tokens and count of prompt ids, at the registered stop ids and model length",
                               conformance="model-untested: StopRule16_v1's, which holds no capture of vLLM's finish yet")
    return bind(defn, T=outputs, STOP=tuple(stops), MAX_MODEL_LEN=max_model_len)


@dataclass(frozen=True)
class Stop:
    """A serve's registered stop rule: the registered model's stop ids ``ids``, its length ``max_model_len``, and
    ``outputs``, T, the most ids a response has, a power of two: every stop class's output size. Public: each class's
    ``commit.json`` names it, and the stop block's id holds its ids, length and T."""

    ids: tuple[int, ...]
    max_model_len: int
    outputs: int

    def __post_init__(self) -> None:
        why = _stop_refusal({"T": self.outputs, "STOP": self.ids, "MAX_MODEL_LEN": self.max_model_len})
        if why is None and not class_size(self.outputs):
            why = f"its T is a class's output size, a power of two (`boundary_rows.class_size`), not {self.outputs}"
        if why is not None:
            raise ValueError(f"a stop rule's T, ids and model length: {why}")

    @classmethod
    def of(cls, doc: Any) -> Stop | None:
        """From a serve's configuration or a ``commit.json`` (`to_json`'s object); None for none."""
        if doc is None:
            return None
        if not (isinstance(doc, Mapping) and set(doc) == {"ids", "max_model_len", "outputs"}
                and isinstance(doc["ids"], (list, tuple))):
            raise ValueError(f"a stop rule is {{\"ids\": [...], \"max_model_len\": n, \"outputs\": T}}, not {doc!r}")
        return cls(tuple(doc["ids"]), doc["max_model_len"], doc["outputs"])

    def to_json(self) -> dict[str, Any]:
        return {"ids": list(self.ids), "max_model_len": self.max_model_len, "outputs": self.outputs}

    @cached_property
    def template(self) -> SpecializedDefinition:
        """The stop block's template: `egress_stop` at the rule."""
        return egress_stop(self.outputs, self.ids, self.max_model_len)

    def word(self, span: Span) -> int | None:
        """The stop block's output on ``span``'s ids at T and the words the engine served it under, n + finish * 8192
        (`StopRule16_v1`'s reference), where that is how the span was served: n the ids it was served and its finish a
        stop on 1 or a length on 2 or 3, with a ``min_tokens`` of 0 and no ``ignore_eos``, neither of which the rule
        reads. None otherwise: a span no stop class holds. The block reads the certified words, which are the engine's
        for a request served as the certifier admitted it; where they differ the block's output is not this word, and
        the span's stop block has no proof."""
        from verity.definitions.registry import stop_check as SC
        from verity.definitions.registry import stop_rule as X

        s = span.sampling
        mt = s.get("max_tokens") or 0
        if (span.finish not in ("stop", "length") or len(span.served) > self.outputs or type(mt) is not int
                or not 0 <= mt < 1 << 32 or (s.get("min_tokens") or 0) != 0 or s.get("ignore_eos")):
            return None
        word = X.reference16([*span.served, *[0] * (self.outputs - len(span.served))], mt, self.ids or (SC.NOT_AN_ID,),
                             len(span.prompt), self.max_model_len)
        n, finish = word & ((1 << X.N_BITS) - 1), word >> X.N_BITS
        return word if n == len(span.served) and finish in ((1,) if span.finish == "stop" else (2, 3)) else None


def filed(spans: Sequence[Span], stop: Stop | None = None,
          released: bool = False) -> dict[tuple[int, int, bool], list[Span]]:
    """The spans by class, and whether it is a stop class: a span rule ``stop`` ends as it was served (`Stop.word`) in
    the stop class of its prompt's size, (P, T); every other by its size class, without a stop block; all without one where
    ``stop`` is None. ``released``: the serve knows its requests' releases, and a span without one (`Span.release`) is in
    no stop class. Each class's spans in admission order, the classes in order of their first span."""
    out: dict[tuple[int, int, bool], list[Span]] = {}
    for s in spans:
        held = stop is not None and not (released and s.release is None) and stop.word(s) is not None
        out.setdefault((s.shape[0], stop.outputs, True) if held else (*s.shape, False), []).append(s)
    return out


def certified(words: Mapping[str, Any] | None, spans: int) -> dict[str, dict[str, Any]]:
    """R7's certified entry of a stop class's words table (`registration.check(..., certified=)`), from ``words``: the
    verifier's own ``--registered`` port of the table, as it read the table in the registration head under the network
    certifier's key (`connection_rows.port`'s form: the table's value, root, rows, a row's leaves and domain, and proof
    block r at row r), or None where the head lists no words table for the class. `StopRefused`: `UNCERTIFIED` for no
    table; `PORT` for a port that is no words table's (rows of `COLUMNS`, each a u32, as `SCHEMA` rows, block r at row
    r); `COUNT` where its rows are not the class's ``spans``, one a stop block."""
    if words is None:
        raise StopRefused(UNCERTIFIED, "the class's requests have no certified words: the network certifier's head "
                          "lists no words table for it, as under no certifier or a schedule in the text form, where a "
                          "request has no certified count of ids. File the class's spans by size class, without a stop "
                          "block (`filed`), or serve it under a schedule whose head lists the class's words table")
    try:
        n, per, width = words["leaves"], words["words"], words.get(G.WIDTH_KEY, G.WORD_BITS)
        shaped = (isinstance(words["value"], str) and all(type(x) is int for x in (n, per, width))
                  and words.get(G.SCHEMA_KEY, G.SCHEMA) == SCHEMA and per * width == 32 * len(COLUMNS)
                  and list(words["reads"]) == list(range(n)) and not words.get(G.READS_FROM))
        entry = {"root": words["root"], "leaves": n, "words": per, **G.width_entry(width), "domain": words["domain"]}
    except (KeyError, TypeError, ValueError):
        shaped = False
    if not shaped:
        raise StopRefused(PORT, f"the certified port is no words table's: its rows are {len(COLUMNS)} u32 "
                          f"({', '.join(COLUMNS)}) as {SCHEMA} rows, proof block r at row r. Hand the verifier's own "
                          f"port of the class's words table, from the head it read under the network certifier's key "
                          f"(`connection_rows.port`), not {words!r}")
    if n != spans:
        raise StopRefused(COUNT, f"the class holds {spans} stop blocks and its certified table {n} rows: block r reads "
                          f"row r, the class's r-th release, so the two counts are one. File in the stop class exactly "
                          f"the requests the network certifier released in the class, in its release order")
    return {words["value"]: entry}


def read_refusal(words: Mapping[str, Any], dom: CommitmentDomain, entry: Mapping[str, Any] | None,
                 bcs: Sequence[bytes], refs: Sequence[int] | None = None,
                 units: Sequence[int] | None = None) -> StopRefused | None:
    """A statement's read of a stop class's certified words, as the Lean verifier's registered reads make it of the
    port and in their order (`Flock.Registered.checkPort`), against the verifier's port ``words`` (`certified`'s) and
    the table's domain ``dom`` as the verifier derived it. ``entry``: the statement's header's registered entry of the
    port (its value, and each table row's position and path), None where it has none; ``bcs``: its table rows'
    ``b || c``; ``refs``: the table row each instance reads, and ``units``: the stop block each instance is (default:
    instance r is block r and reads table row r). None where the reads accept, else the refusal: `READ_NONE` where the
    statement ties the port's rows to no row of the table (no entry, an entry of another value, rows without a position
    and a path each, or an instance with no table row); `READ_ROOT` where a table row does not open the certified root
    at its position; `READ_ROW` where a block reads another row than its own."""
    value = words["value"]
    if (words["domain"], words["leaves"]) != (dom.domain_id.hex(), dom.count):
        raise ValueError(f"the port names another domain or count of rows than the domain handed for {value}")
    stage = (f"Stage the stop blocks' {WORDS} port as a registered read of {value}: each table row with its position and "
             f"its path to the certified root")
    if entry is None or entry.get("value") != value:
        named = "no registered value" if entry is None else f"another value, {entry.get('value')!r}"
        return StopRefused(READ_NONE, f"the statement's {WORDS} rows are rows of {named}, and no row of the certified "
                           f"table: its stop blocks would be proved on words the prover chose. {stage}")
    positions, paths = list(entry.get("positions", ())), list(entry.get("paths", ()))
    if len(positions) != len(bcs) or len(paths) != len(bcs):
        return StopRefused(READ_NONE, f"the statement holds {len(bcs)} {WORDS} rows, {len(positions)} positions and "
                           f"{len(paths)} paths: a row without both is a row of no place in the table. {stage}")
    root = bytes.fromhex(words["root"])
    for r, (p, bc, path) in enumerate(zip(positions, bcs, paths)):
        try:
            opens = type(p) is int and G.verify_read(dom, root, p, bytes(bc), bytes.fromhex(path), SCHEMA)
        except (TypeError, ValueError):
            opens = False
        if not opens:
            return StopRefused(READ_ROOT, f"table row {r} does not open {value}'s certified root at position {p}: the "
                               f"words its stop block is proved at are not the ones the network certifier signed for "
                               f"that release. Prove the block on the certifier's own row, under the salt and the path "
                               f"the certifier hands the prover")
    refs = list(range(len(bcs)) if refs is None else refs)
    units = list(range(len(refs)) if units is None else units)
    reads = list(words["reads"])
    for i, (ref, u) in enumerate(zip(refs, units, strict=True)):
        if not 0 <= ref < len(bcs):
            return StopRefused(READ_NONE, f"instance {i} reads no row of the statement's {WORDS} table. {stage}")
        own = reads[u] if 0 <= u < len(reads) else None
        if positions[ref] != own:
            return StopRefused(READ_ROW, f"stop block {u} reads row {positions[ref]} of {value}, and its own is "
                               f"{'none' if own is None else f'row {own}'}: block r reads row r, its release's words. "
                               f"Stage each block on the row of its own release")
    return None


@dataclass(frozen=True)
class StopRegistration:
    """What `stop_registration` returns: a stop class's stop population's registered values as its verifier holds them.
    ``served``: the boundary record's port ``ids`` reads, as an output value (`served.register_served`'s). ``certified``:
    R7's certified entry of the words table (`certified`). ``registered``: the stop record's ``registered`` entries,
    ``ids``'s then the table's. ``file``: the verifier's ``--registered`` document, the two parameters' ports by circuit
    port. `copies`: what R7 holds the record to."""

    served: RS.ServedRegistration
    certified: dict[str, dict[str, Any]]
    registered: list[dict[str, Any]]
    file: dict[str, Any]

    @property
    def table(self) -> str:
        """The certified table's value name."""
        return next(iter(self.certified))

    def copies(self) -> dict[str, Any]:
        """R7's own copies, as `registration.check`, `verifier.check` and `Stream.register` take them by keyword."""
        return {"registered": self.served.values, "outputs": self.served.outputs, "schemas": self.served.schemas,
                "certified": self.certified}


def stop_registration(pub: Mapping[str, Any], check: SpecializedDefinition, stop: Stop, boundary: Mapping[str, Any],
                      words: Mapping[str, Any] | None) -> StopRegistration:
    """A stop class's stop population's registered values, from its ``commit.json``, the verifier's rule ``stop``, its
    boundary record (``boundary``, `boundary_rows.record`'s, the log's) and the class's certified words table
    (``words``, `certified`'s): the block's ``ids`` is the boundary record's port `b.egress`, one row a span, as an
    output value under that port's root and domain (`served.register_served`), and its ``words`` the certified table,
    one row a release, under the root and domain the network certifier signed. What a prover's record registers
    (``registered``) and what a verifier holds it to (`registration.check(..., **copies())`, the log holding
    ``boundary``). ValueError where the commitment names another rule than ``stop``, the class is no stop class or
    ``boundary`` isn't its record; `StopRefused` as `certified` refuses the table."""
    L, c = layout_of(pub, check, stop), BoundaryCommit.of(pub, COMMIT)
    program = L.stop_program
    mine = {r["port"]: r for r in boundary["roots"]}
    want = {p: {"port": p, "root": c.values[p]["root"], **e} for p, e in L.ports().items()}
    if mine != want:
        raise ValueError("the boundary record's roots are not this commitment's values, under its layout's domains")
    held = certified(words, L.spans)
    digest = R.digest(boundary)
    trees = [RS.ServedTree(name, mine[p]["root"], L.spans, L.words(p) // 2, mine[p]["domain"], 32, output=(digest, p),
                           row_schema=SCHEMA) for name, p in STOP_READS.items()]
    served = RS.register_served(program, L.stop_query, trees, log=[boundary])
    (value, e), = held.items()
    port = produced.port(value, e, e["root"], list(range(L.spans)), SCHEMA)
    file = {**served.file, "ports": dict(sorted({**served.file["ports"], WORDS: port}.items()))}
    return StopRegistration(served, held, [*served.registered, {"value": value, **e}], file)


def stop_record(pub: Mapping[str, Any], check: SpecializedDefinition, stop: Stop, design: Mapping[str, Any],
                boundary: Mapping[str, Any], words: Mapping[str, Any] | None,
                window: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A stop class's stop population's registration (one-stage's record) under the verifier's rule ``stop``, the next
    of its log after its boundary record ``boundary``: `Layout.stop_program` under `Layout.stop_query` at the sampling
    design ``design`` over the class's stop blocks, one a span; its one port `b.stop`, the blocks' outputs; its
    registered values the boundary record's port ``ids`` reads and the class's certified words table ``words``
    (`stop_registration`); and that record's window with ``window``'s items, named the stop population's."""
    L, c = layout_of(pub, check, stop), BoundaryCommit.of(pub, COMMIT)
    reg = stop_registration(pub, check, stop, boundary, words)
    roots = [{"port": p, "root": c.values[p]["root"], **e} for p, e in L.stop_ports().items()]
    w = {**boundary["window"], **(window or {}), "population": "stop"}
    return R.record(L.stop_program, L.stop_query, design, CS.SCHEME, roots, w, previous=R.digest(boundary),
                    registered=reg.registered)


__all__ = ["COLUMNS", "COUNT", "IDS", "PORT", "READ_NONE", "READ_ROOT", "READ_ROW", "STOP_READS", "UNCERTIFIED", "WORDS",
           "Stop", "StopRefused", "StopRegistration", "certified", "egress_stop", "filed", "read_refusal", "stop_record",
           "stop_registration"]
