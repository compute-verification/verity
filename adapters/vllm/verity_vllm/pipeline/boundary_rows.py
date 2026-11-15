"""A served window's boundary commitment under the two-stage sampling design (core's `spot.service.TwoStage`):
what each request's replay block reads and writes at its edges, committed inside the engine core as the window closes, so
the first stage samples requests whose boundary was fixed while they were served.

**Spans and classes.** A window's spans are the requests that finished in it, in admission order (`boundary_windows.BoundaryWindows`); a
request still running at a close is carried to the next window, so a span never crosses one. A span's class is (P, O): the
least powers of two, at least `MIN_CAP`, covering its prompt's ids and the ids it was served. Each class is its own
registration over N replay blocks, slot r its r-th span.

**Values** (`PORTS`): one `hm96-sha512/row/v2` row a span each, under its window domain (`verity.commitments.window`:
program, partition, port, schema, N leaves, run); a u32 is two u16 words, low first, a float its f32 bits.

| value | row |
|---|---|
| `b.ingress` | `INGRESS` (the prompt's length, the sampling inputs, the seed as a u64), then P ids, zero past the prompt |
| `b.egress` | the ids served n, the finish (`FINISH`), then O ids, zero past n |

The step shapes are no value: they choose the Programs, and the window's advice commits them. Each leaf's salt is fresh
from the OS (`commit`).

**The stop rule.** A serve may register one (`boundary_stop.Stop`: the registered model's stop ids, its length, and T,
the most ids a response has). Its classes then commit each row above as its parts, one row a span each and no word
twice, and the ids served n is no number the serve states:

| value | row |
|---|---|
| `b.request` | `b.ingress`'s fields before its ids (`FIELDS`), the prompt's length and ``max_tokens`` the first two |
| `b.prompt` | P prompt ids, zero past lp |
| `b.egress` | O ids, zero past n: no n and no finish |
| `b.stop` | one u16, n + finish * 8192 (1 a stop id, 2 ``max_tokens``, 3 the model's length): a stop class's alone |

The first three are the class's boundary record's values (`PARTS`). A span the rule ends as it was served is in the stop
class of its prompt's size, (P, T) whatever n is, which commits `b.stop` too: the output of the span's stop block, under
the window domain of the stop population's own program (`Layout.stop_program`), and the one port of a stop record of
its own (`boundary_stop`). Every other span is filed by its size class with no `b.stop` (``p<P>-o<O>-nostop``). The rule
a class is laid out under is the verifier's own copy, as its check template is (`layout_of`): a ``commit.json`` that
names another rule is refused.

**History tables** (`Layout.layers`, off by default): each span's latent rows at each of those layers, one
`boundary_tables` table a (slot, layer) of `boundary_tables.WORDS` words and ``capacity`` leaves under its own
`history.domain` (the class's program and the table's name, from the run, the class and the slot), listed beside the class's
values in slot then layer order. Rank 0 builds them once the window assigns each span its slot (`boundary_tables.Builder`),
and the class's commitment waits for their roots, so they are fixed before any coin, as the values are. A serve under a
stop rule has none (`Layout`): a stop class and its sibling without a stop block at the same (P, O) would share
their tables' names.

**The return**, one more value of a class that says what its replay block returns: its check's verdicts
(`Layout.verdicts`) or the ids it served (`Layout.ids`), at most one. It is the block's output port, named as round 2
names the value a return link reads (`return_port`: ``o0``, or ``<replay block's id>/o0`` in a record that names a port
with a ``/``, as a history table's name does). Its rows are the ones round 2's return links read (`Layout.return_rows`),
a span's block's in return order then the next span's, under the same row schema and each its own fresh salt. A round 2
then holds each sampled proof block's return to its row of this port, opened at round 1's root, where a record without
it would leave the return to whatever the interior registered. The class says which because a return's type can't: bits
that are an id are no verdicts, and 32 bits a unit need not be the id served. It is a value of the class's boundary
record, after `PORTS` or, under a stop rule, after `PARTS`.

* **Verdicts** (`Layout.verdict_rows`): each bit a one, what the block returns where its check passes, so the serve
  writes them without a replay.
* **Ids** (`id_rows`): O rows a span, row j the id of output j as the check returns it, two u16 words, the low one
  first. The ids are the ones `b.egress` holds; nothing ties the two values to each other while the replay block is a
  stand-in that reads neither (below).

**The replay block** (`Layout.replay`) is a stand-in until the class's request Definition lands: `per_prompt * P +
per_output * O` instances of one check template (`population`), so a window committed under it is audited with that one
template; or, ``inline``, those instances with each one's body in place (`inlined`), audited with the templates that body
calls, for a check template too large for one proof block whose pieces are. The boundary's values, layout and serving
path stay as they are when the replay block becomes the class's own; its program, and so its domains, name whichever
replay block it is configured with.

**Files** (`write`), under ``w<window>/p<P>-o<O>[-nostop]/``: ``commit.json`` (public: the layout, each value's root,
domain and leaves, the index's hiding leaf), and prover-only ``index.json`` (each span's engine id, prompt, sampling
inputs, ids served and finish, and its release where the serve knows one), ``index-salt.bin`` and ``salts.bin`` (every
leaf's salt, value by value); `reopen` recommits from them and checks the roots, and `Committed.port` opens a value's
rows, strings and paths for a later statement to read.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.population import population
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Concat, coll_concat
from verity.circuits.types import Array
from verity.commitments import history as HT
from verity.commitments.identity import identity_digest_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.windows import links as LK, rows as CS
from verity_vllm.pipeline import boundary_tables as BT
from verity_vllm.records.boundary_commit import FILE as COMMIT, SCHEMA as FORMAT, BoundaryCommit
from verity_vllm.records.boundary_tables import TableRoots

SCHEMA = CS.SCHEMA_HM96_SHA512_ROW_V2
INGRESS, EGRESS = "b.ingress", "b.egress"
PORTS = (INGRESS, EGRESS)
REQUEST, PROMPT, STOP = "b.request", "b.prompt", "b.stop"
#: a class's values under a stop rule, one row a span each: `b.ingress`'s row in two and `b.egress`'s ids
PARTS = (REQUEST, PROMPT, EGRESS)
#: every row value's name but the return's (`Layout.returned`): any other name of a class's is a history table's
ROWS = (INGRESS, EGRESS, REQUEST, PROMPT, STOP)
#: the replay block's output port: its return's name where it is a value (`return_port`)
RETURN = "o0"
#: why no class holds both a stop rule and history tables
TABLES_UNDER_A_RULE = ("a serve under a stop rule commits no history tables: a table's name holds its class's (P, O) and "
                       "slot, which a stop class and its sibling without a stop block share at the rule's T. The fix: the "
                       "table's key names which of the two its class is (`boundary_tables.KEY`)")
#: why no class says both `verdicts` and `ids`: its replay block has one return, and the class commits it once
ONE_RETURN = "a class commits its replay block's return as verdicts or as ids, not as both"
MIN_CAP = 2
#: `b.ingress`'s fields before its ids, each a u32 (``u64``: two, low first) of the request's own sampling inputs
FIELDS = (("lp", "u32"), ("max_tokens", "u32"), ("min_tokens", "u32"), ("ignore_eos", "u32"), ("temperature", "f32"),
          ("top_p", "f32"), ("top_k", "i32"), ("min_p", "f32"), ("presence_penalty", "f32"), ("frequency_penalty", "f32"),
          ("repetition_penalty", "f32"), ("seeded", "u32"), ("seed", "u64"))
#: the sampling inputs `FIELDS` reads off a request's SamplingParams
SAMPLING = tuple(k for k, _ in FIELDS if k not in ("lp", "seeded"))
HEAD_U32 = sum(2 if t == "u64" else 1 for _, t in FIELDS)
#: each field's first u32 among `HEAD_U32`
FIELD_AT = {k: sum(2 if t == "u64" else 1 for _, t in FIELDS[:i]) for i, (k, _) in enumerate(FIELDS)}
#: `b.egress`'s finish codes; 0 is a request still running when the serve stopped
FINISH = ("stop", "length", "abort", "error")
INDEX, SALTS = "index.json", "salts.bin"


def check_template(t: str) -> SpecializedDefinition:
    """The check template a serving template's descriptor id ``t`` names (`serving_templates.template`)."""
    from verity_vllm.pipeline import serving_templates as ST

    return ST.template(t).fn


def cap(n: int) -> int:
    """The least power of two at least `MIN_CAP` and ``n``."""
    return max(MIN_CAP, 1 << max(0, int(n) - 1).bit_length())


def class_size(v: Any) -> bool:
    """Whether ``v`` is a class's size, of prompt ids or of outputs (`Span.shape`)."""
    return type(v) is int and v >= MIN_CAP and v & (v - 1) == 0


@dataclass(frozen=True)
class Span:
    """One request as the engine core served it: its engine id, prompt ids, sampling inputs (`SAMPLING`), the ids it was
    served, how it finished (a `FINISH` name, None while running) and, where the serve knows it, its release: the
    network certifier's run and the request's index N in that run's release order."""

    engine_id: str
    prompt: tuple[int, ...]
    sampling: Mapping[str, Any]
    served: tuple[int, ...] = ()
    finish: str | None = None
    release: tuple[str, int] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Its class (P, O)."""
        return cap(len(self.prompt)), cap(len(self.served))

    def to_json(self) -> dict[str, Any]:
        return {"engine_id": self.engine_id, "prompt": list(self.prompt), "sampling": dict(self.sampling),
                "served": list(self.served), "finish": self.finish,
                **({} if self.release is None else {"release": list(self.release)})}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Span:
        release = d.get("release")
        return cls(str(d["engine_id"]), tuple(d["prompt"]), dict(d["sampling"]), tuple(d["served"]), d["finish"],
                   None if release is None else (str(release[0]), int(release[1])))


def sampling_of(params: Any) -> dict[str, Any]:
    """A request's `SAMPLING` inputs off its SamplingParams (None where it has none)."""
    out = {}
    for k in SAMPLING:
        v = getattr(params, k, None)
        out[k] = v if v is None or isinstance(v, (bool, int, float)) else float(v)
    return out


def _u32(name: str, kind: str, v: Any) -> list[int]:
    if kind == "f32":
        return [struct.unpack("<I", struct.pack("<f", float(v or 0.0)))[0]]
    x, bits = int(v or 0), 64 if kind == "u64" else 32
    lo = -(1 << (bits - 1)) if kind in ("i32", "u64") else 0
    if not lo <= x < 1 << bits:
        raise ValueError(f"{name} = {x} doesn't fit b.ingress's {kind}")
    x &= (1 << bits) - 1
    return [x & 0xFFFFFFFF, x >> 32] if kind == "u64" else [x]


def _words(u32s: Sequence[int]) -> np.ndarray:
    a = np.asarray(u32s, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() >= 1 << 32):
        raise ValueError("an id is a u32")
    return np.stack([a & 0xFFFF, a >> 16], axis=-1).astype(np.uint16).reshape(-1)


def ingress_row(span: Span, prompt_cap: int) -> np.ndarray:
    """Span's `b.ingress` row at prompt cap P, as u16 words."""
    if len(span.prompt) > prompt_cap:
        raise ValueError(f"{span.engine_id}: {len(span.prompt)} prompt ids past the class's {prompt_cap}")
    s = {**{k: None for k in SAMPLING}, **span.sampling}
    vals = {**s, "lp": len(span.prompt), "seeded": s["seed"] is not None}
    head = [w for k, t in FIELDS for w in _u32(k, t, vals[k])]
    return _words(head + list(span.prompt) + [0] * (prompt_cap - len(span.prompt)))


def egress_row(span: Span, output_cap: int, head: bool = True) -> np.ndarray:
    """Span's `b.egress` row at output cap O, as u16 words: the ids served n, the finish and the O ids; without ``head``
    (a serve that registers a stop rule), the O ids alone."""
    if len(span.served) > output_cap:
        raise ValueError(f"{span.engine_id}: {len(span.served)} ids served past the class's {output_cap}")
    ids = [*span.served, *[0] * (output_cap - len(span.served))]
    if not head:
        return _words(ids)
    if span.finish is not None and span.finish not in FINISH:
        raise ValueError(f"{span.engine_id}: finish {span.finish!r} is none of {FINISH}")
    finish = 0 if span.finish is None else 1 + FINISH.index(span.finish)
    return _words([len(span.served), finish, *ids])


def id_rows(span: Span, outputs: int) -> np.ndarray:
    """Span's rows of its replay block's return in a class of O outputs, (O, 2) u16: row j the id of output j, and past
    the ids served the last one again, as the block's instances past them are copies of its last
    (`interior_rows.Span.padded`). A span served no id has no instance to copy, and its rows are zero: no replay
    reproduces it."""
    if len(span.served) > outputs:
        raise ValueError(f"{span.engine_id}: {len(span.served)} ids served past the class's {outputs}")
    last = span.served[-1] if span.served else 0
    return _words([*span.served] + [last] * (outputs - len(span.served))).reshape(outputs, 2)


def return_port(block: str, ports: Iterable[str]) -> str:
    """The name of replay block ``block``'s return in a record whose other ports are ``ports``, as round 2 names the
    value a return link reads (`second_stage.replay_links`): `RETURN`, or ``<block>/o0`` where the record names a port
    with a ``/``."""
    return f"{block}/{RETURN}" if any("/" in p for p in ports) else RETURN


def _parts(r: Any) -> list[Any]:
    """A body reference's parts, its concatenations spliced in."""
    return [x for q in r.parts for x in _parts(q)] if isinstance(r, Concat) else [r]


def run_root(serve: str, window: int) -> str:
    """The window's run, which every domain binds: the SHA-512 over `FORMAT`, the serve's nonce and the window's index."""
    return hashlib.sha512(f"{FORMAT}\0{serve}\0{int(window)}".encode()).hexdigest()


def inlined(template: SpecializedDefinition, n: int) -> SpecializedDefinition:
    """`TemplateInlined_v1{template, N=n}`: `n` instances of the composite ``template``, each one's body emitted in place, every
    parameter an array over instances and the result the instances' results in order. It computes `population`'s function,
    but its roots are the calls inside ``template``'s body, so a check of the templates that body calls counts their
    instances in it, while its outputs stay ``template``'s alone."""
    if not isinstance(getattr(template, "defn", None), CompositeDefinition) or n < 1:
        raise ValueError(f"inlines one instance or more of a composite template, not {n} of {template.id}")
    params, ret = template.signature()

    def signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B: Any, S: Any, *args: Any) -> Any:
        return coll_concat(Array(S.N, ret), [template.defn.body(B, template.S(), *(a[i] for a in args)) for i in range(S.N)])

    defn = CompositeDefinition("TemplateInlined", 1, ("template", "N"), signature, body, register=False,
                               doc=f"{n} instances of {template.id}, each body in place")
    return bind(defn, template=descriptor_id(template), N=n)


@dataclass(frozen=True)
class Layout:
    """One class's registration, from public items alone: the check template ``check``, its instances a span
    (``per_prompt`` a prompt row, ``per_output`` an output), the class (``prompt``, ``outputs``), the spans N and the run.
    ``inline``: the replay block holds them `inlined`, and the check's proof blocks are the templates the check template's
    body calls (`templates`); else they are its instances (`population`). ``layers``: each span's history tables' layers,
    of ``capacity`` leaves (the module docstring). ``stop``: the serve's registered stop rule (a `boundary_stop.Stop`),
    under which the class's values are `PARTS` and it has no history tables; ``held``, under a rule: the class is a stop
    class, its outputs the rule's T, with a stop block a span in a population of its own (`stop_program`) whose output
    it commits as `b.stop`. ``verdicts`` or ``ids`` (a class says at most one): the replay block returns the check's
    verdicts, every bit a one where it passes, or the ids the span was served, one check instance an output, and the
    class commits that return as one more value of its boundary record (`return_rows`).
    What the prover commits under and the challenger's own copies (`ports`, `stop_ports`)."""

    check: SpecializedDefinition
    per_prompt: int
    per_output: int
    prompt: int
    outputs: int
    spans: int
    run: str
    inline: bool = False
    layers: tuple[int, ...] = ()
    capacity: int = HT.CAPACITY
    stop: Any = None
    held: bool = True
    verdicts: bool = False
    ids: bool = False

    def __post_init__(self) -> None:
        for k in ("prompt", "outputs"):
            v = getattr(self, k)
            if not class_size(v):
                raise ValueError(f"a class's {k} cap is a power of two at least {MIN_CAP}, not {v!r}")
        if not (type(self.spans) is int and self.spans >= 1):
            raise ValueError(f"a class registers one span or more, not {self.spans!r}")
        ls = self.layers
        if not (isinstance(ls, tuple) and all(type(x) is int and x >= 0 for x in ls) and list(ls) == sorted(set(ls))):
            raise ValueError(f"a class's table layers are distinct natural numbers in order, not {ls!r}")
        if not HT.cap_ok(self.capacity):
            raise ValueError(f"a table's capacity is a power of two of at least 2, not {self.capacity!r}")
        if self.verdicts and self.ids:
            raise ValueError(ONE_RETURN)
        if self.n_rho < 2:
            raise ValueError(f"a replay block holds two proof blocks or more (TwoStage), not {self.n_rho}: "
                             f"{self.per_prompt} a prompt row and {self.per_output} an output at ({self.prompt}, {self.outputs})")
        if self.stop is not None and ls:
            raise ValueError(TABLES_UNDER_A_RULE)
        if self.stopped and self.outputs != self.stop.outputs:
            raise ValueError(f"a stop class is filed at its rule's T, {self.stop.outputs} outputs, not {self.outputs}")
        if self.inline:
            composite = isinstance(getattr(self.check, "defn", None), CompositeDefinition)
            forms = {n.form for n in self.check.body.nodes} if composite else {"primitive"}
            if not forms <= {"call", "batch"}:
                raise ValueError(f"an inlined check template's body is calls of templates, each a proof block, not "
                                 f"{P.template_id(self.check)}'s {sorted(forms - {'call', 'batch'})}")

    @property
    def n_rho(self) -> int:
        return self.per_prompt * self.prompt + self.per_output * self.outputs

    @property
    def stopped(self) -> bool:
        """The class is a stop class: its serve registers a rule, and the rule ends each of its spans as it was served."""
        return self.stop is not None and self.held

    @cached_property
    def templates(self) -> tuple[SpecializedDefinition, ...]:
        """The check's proof blocks' templates: the check template, or, inlined, each template its body calls, in the order
        of its first call."""
        if not self.inline:
            return (self.check,)
        return tuple({P.template_id(n.fn): n.fn for n in self.check.body.nodes}.values())

    @property
    def query(self) -> dict[str, Any]:
        """The check's proof blocks in a replay block (`service.Check.query`): one template's instances, or each one's."""
        ts = self.templates
        return P.template_instance_query(ts[0]) if len(ts) == 1 else P.template_instances_query(ts)

    @cached_property
    def replay(self) -> SpecializedDefinition:
        return inlined(self.check, self.n_rho) if self.inline else population(self.check, self.n_rho)

    @cached_property
    def program(self) -> Program:
        return P.population_program(self.replay, self.spans)

    @cached_property
    def grain(self) -> dict[str, Any]:
        return P.template_instance_query(self.replay)

    @cached_property
    def partition(self) -> dict[str, Any]:
        return P.partition(self.program, self.grain)

    @cached_property
    def stop_program(self) -> Program:
        """A stop class's stop blocks, one a span, slot r its r-th span's: a population of their own."""
        if not self.stopped:
            raise ValueError("only a stop class has stop blocks: a class under a stop rule, whose spans the rule ends as "
                             "they were served")
        return P.population_program(self.stop.template, self.spans)

    @cached_property
    def stop_query(self) -> dict[str, Any]:
        """The stop population's proof blocks: the stop block's instances, one a span."""
        return P.template_instance_query(self.stop.template)

    @cached_property
    def stop_partition(self) -> dict[str, Any]:
        return P.partition(self.stop_program, self.stop_query)

    @cached_property
    def verdict_rows(self) -> tuple[int, int] | None:
        """The replay block's return as rows, where the class commits it (`verdicts`): ``(rows a block, bits a row)``, a
        row what one of round 2's return links reads (`second_stage._return_links`): a member's share of one part of the
        block's body's return, so an instance's whole output or, inlined, one field of a call's (a read's ``ok``). None
        for a class that doesn't; ValueError for one whose block returns anything but bits in such rows of one width,
        which round 2 links nothing to."""
        if not self.verdicts:
            return None
        ret, body, widths = self.replay.signature()[1], self.replay.body, set()
        for r in _parts(body.ret):
            node = body.nodes[r.idx] if isinstance(r, Affine) and r.space == "n" else None
            m = {"call": 1, "batch": getattr(node, "n", 0)}.get(getattr(node, "form", None), 0)
            widths.add(r.count // m if m and r.count % m == 0 else 0)
        w = widths.pop() if len(widths) == 1 else 0
        if ret.width_set() != {1} or not w or ret.leaves % w:
            raise ValueError(f"a class commits its replay block's return as verdicts where it is bits, in rows of one width "
                             f"its body's nodes return, not {P.template_id(self.check)}'s {ret!r}")
        return ret.leaves // w, w

    @cached_property
    def return_rows(self) -> tuple[int, int] | None:
        """The replay block's return as rows, where the class commits it: ``(rows a block, bits a row)``. A class of
        `verdicts`: `verdict_rows`. A class of `ids`: one row an output, the 32-bit id the check's instance there returns,
        as round 2's return link reads them (`second_stage.link_runs`: unit j of a block its row j); ValueError for one
        whose block returns anything else, which round 2 would hold to no row of such a port: other than one template's
        32-bit output, one an output of the class, in output order. None for a class that commits no return."""
        if not self.ids:
            return self.verdict_rows
        params = {P.template_id(t): {name for name, _ty in t.signature()[0]} for t in self.templates}
        runs = SS.link_runs(definitions(self.program), descriptor_id(self.replay)) or ()
        got = [(x, per, rs) for x, per, rs in runs if x.port not in params.get(x.template, ())]
        by = {P.template_id(t): t for t in self.templates}
        ok = (len(got) == 1 and got[0][0].value == RETURN and got[0][0].template in by
              and sum(by[got[0][0].template].signature()[1].leaf_widths()) == 32 and got[0][1] == self.outputs == self.n_rho
              and [SS.row_at(got[0][2], j) for j in range(self.outputs)] == list(range(self.outputs)))
        if not ok:
            raise ValueError(f"a class commits its replay block's return as ids where the block returns one 32-bit id an "
                             f"output, in output order, by one return link round 2 derives: not "
                             f"{P.template_id(self.replay)}'s, of {self.n_rho} instances at {self.outputs} outputs with "
                             f"return links {[(x.template, x.value, per) for x, per, _rs in got]}")
        return self.outputs, 32

    @cached_property
    def returned(self) -> str | None:
        """The return's name where the class commits it (`return_port`, among its row values and the tables)."""
        if self.return_rows is None:
            return None
        return return_port(descriptor_id(self.replay), (*(PORTS if self.stop is None else PARTS), *self.tables))

    @cached_property
    def values(self) -> tuple[str, ...]:
        """The boundary record's row values, in its order: `PORTS`, or under a stop rule `PARTS`; then the return where
        the class commits it, under its name (`returned`)."""
        rows = PORTS if self.stop is None else PARTS
        return rows if self.returned is None else (*rows, self.returned)

    @property
    def committed(self) -> tuple[str, ...]:
        """Every value the serve commits rows of, in the salts' order: `values`, then in a stop class `b.stop`, the stop
        record's port."""
        return (*self.values, STOP) if self.stopped else self.values

    @cached_property
    def tables(self) -> tuple[str, ...]:
        """Each span's history tables' names, slot by slot, then layer by layer."""
        return tuple(BT.name(self.run, self.prompt, self.outputs, r, x) for r in range(self.spans) for x in self.layers)

    def leaves(self, port: str) -> int:
        """A value's leaves: a row value's rows, one a span, or the return's rows a block a span; a table's capacity."""
        if port in ROWS:
            return self.spans
        return self.spans * self.return_rows[0] if port == self.returned else self.capacity

    def words(self, port: str) -> int:
        """The u16 words in a row of ``port``, a row value or a history table; `b.egress` holds its two head fields only
        where no rule is registered."""
        rows = {INGRESS: 2 * (HEAD_U32 + self.prompt), EGRESS: 2 * (self.outputs + (2 if self.stop is None else 0)),
                REQUEST: 2 * HEAD_U32, PROMPT: 2 * self.prompt, STOP: 1}
        return -(-self.return_rows[1] // 16) if port == self.returned else rows.get(port, BT.WORDS)

    def domain(self, port: str) -> CommitmentDomain:
        """``port``'s domain: a history table's own, or a row value's window domain, over the program of the record it is
        a port of: the class's, or for `b.stop` the stop population's."""
        if port not in ROWS and port != self.returned:
            return HT.domain(self.partition["program"], port, BT.WORDS, self.capacity)
        part, n = self.stop_partition if port == STOP else self.partition, self.leaves(port)
        binding = identity_digest_sha512(CS.DOMAIN_TAG, {"program": part["program"], "partition": P.digest(part),
                                                         "port": port, "schema": SCHEMA, "leaves": n, "run": self.run})
        return CommitmentDomain(binding, CS.OWNER, RangeIndexedDomain(0, n), hash="sha512")

    def _ports(self, names: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {p: {"leaves": self.leaves(p), "domain": self.domain(p).domain_id.hex()} for p in names}

    def ports(self) -> dict[str, dict[str, Any]]:
        """Each of the boundary record's values' leaves and domain id, as `Stream.register` takes them: its row values',
        then each table's."""
        return {**self._ports(self.values),
                **{t: {"leaves": self.capacity, "domain": self.domain(t).domain_id.hex()} for t in self.tables}}

    def stop_ports(self) -> dict[str, dict[str, Any]]:
        """The stop record's port's leaves and domain id, as `registration.check` takes them."""
        return self._ports(self.committed[len(self.values):])

    def schemas(self) -> dict[str, str]:
        """Each of the boundary record's values' row schema, in `ports`' order."""
        return {**dict.fromkeys(self.values, SCHEMA), **dict.fromkeys(self.tables, HT.SCHEMA)}

    def provenance(self) -> dict[str, Any]:
        tables = {"tables": {"layers": list(self.layers), "capacity": self.capacity}} if self.layers else {}
        return {"prompt": self.prompt, "outputs": self.outputs, "check": P.template_id(self.check),
                "per_prompt": self.per_prompt, "per_output": self.per_output, "inline": self.inline, **tables,
                **({} if self.stop is None else {"stop": self.stop.to_json(), "stopped": self.held}),
                **({"verdicts": True} if self.verdicts else {}), **({"ids": True} if self.ids else {})}


def port_rows(spans: Sequence[Span], layout: Layout) -> dict[str, np.ndarray]:
    """Each committed value's rows (`Layout.committed`), (leaves, words) u16, span by span. Under a stop rule: `b.ingress`'s
    row as its fields and its prompt ids, and the ids served alone; in a stop class, whose spans are at the rule's T
    whatever they were served, also the stop block's output on those rows, the rule's word. Where the class commits its
    replay block's return, its rows too: of verdicts (`Layout.verdicts`), every bit a one, low bit first in its words,
    as the block returns it where its check passes; of ids (`Layout.ids`), each span's (`id_rows`). ValueError for a
    span of a stop class the rule doesn't end as it was served (`boundary_stop.Stop.word`): the block's output on its
    rows is not its end."""
    L = layout
    if len(spans) != L.spans or any((s.shape[0], L.outputs if L.stopped else s.shape[1]) != (L.prompt, L.outputs)
                                    for s in spans):
        raise ValueError(f"the spans are not {L.spans} of class ({L.prompt}, {L.outputs})")
    ingress = np.stack([ingress_row(s, L.prompt) for s in spans])
    egress = np.stack([egress_row(s, L.outputs, head=L.stop is None) for s in spans])
    back = {}
    if L.ids:
        back[L.returned] = np.concatenate([id_rows(s, L.outputs) for s in spans])
    elif L.returned is not None:
        ones = (1 << L.return_rows[1]) - 1
        row = np.array([ones >> 16 * k & 0xFFFF for k in range(L.words(L.returned))], dtype=np.uint16)
        back[L.returned] = np.tile(row, (L.leaves(L.returned), 1))
    if L.stop is None:
        return {INGRESS: ingress, EGRESS: egress, **back}
    rows = {REQUEST: ingress[:, :2 * HEAD_U32], PROMPT: ingress[:, 2 * HEAD_U32:], EGRESS: egress, **back}
    if L.stopped:
        words = [L.stop.word(s) for s in spans]
        bad = next((s for s, w in zip(spans, words) if w is None), None)
        if bad is not None:
            raise ValueError(f"{bad.engine_id}: the stop rule doesn't end this span as it was served ({len(bad.served)} "
                             f"ids, finish {bad.finish!r}), so no stop class holds it: `boundary_stop.filed` files it without a stop block")
        rows[STOP] = np.array(words, dtype="<u2").reshape(L.spans, 1)
    return rows


@dataclass
class Committed:
    """One class's commitment in a window: its layout and spans, each value's root (and each history table's, rank 0's),
    the leaves' salts and the index's."""

    window: int
    layout: Layout
    spans: tuple[Span, ...]
    roots: dict[str, bytes]
    salts: dict[str, bytes]
    index_salt: bytes
    seconds: float = 0.0

    @cached_property
    def index(self) -> bytes:
        return CS.index_bytes({"format": FORMAT, "spans": [s.to_json() for s in self.spans]})

    @property
    def index_leaf(self) -> str:
        return CS.index_leaf(self.index, self.index_salt).hex()

    def public(self) -> dict[str, Any]:
        L, ports = self.layout, {**self.layout.ports(), **self.layout.stop_ports()}
        return BoundaryCommit(
            window=self.window, run=L.run, **L.provenance(), spans=L.spans, replay=P.template_id(L.replay),
            program=L.partition["program"], partition=P.digest(L.partition), schema=SCHEMA,
            values={p: {"root": self.roots[p].hex(), "words": L.words(p), **ports[p]} for p in ports},
            index=self.index_leaf, index_rule=CS.INDEX_RULE, seconds=round(self.seconds, 3)).dump()

    def record(self, law: Mapping[str, Any], previous: str | None = None,
               window: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return record(self.public(), self.layout.check, law, previous, window, stop=self.layout.stop)

    def port(self, value: str) -> LK.Port:
        """Value ``value`` as a later statement reads its rows (`links.Port`): its rows, every row's ``b || c`` and
        salt, recommitted here and held to its root. What the replay side opens a return's rows from (`Layout.returned`),
        prover only: a row's salt opens the row."""
        L = self.layout
        if value not in L.committed:
            raise ValueError(f"{value!r} is no value the class commits rows of: {list(L.committed)}")
        rows, domain = port_rows(self.spans, L)[value], L.domain(value)
        roots, bc = CS.commit_ports({value: domain}, {value: rows}, {}, {value: self.salts[value]}, schemas={value: SCHEMA})
        if roots[value] != self.roots[value]:
            raise ValueError(f"{value}: its rows and salts recommit to another root than the class's")
        return LK.Port(value, SCHEMA, L.words(value), domain, bytes(bc[value]), rows, bytes(self.salts[value]))


def layout_of(pub: Mapping[str, Any], check: SpecializedDefinition, stop: Any = None) -> Layout:
    """The class's `Layout` from its ``commit.json`` (``pub``) and the caller's own check template ``check`` and stop
    rule ``stop`` (a `boundary_stop.Stop`; None: no rule): the challenger's own copy, from public items alone. ValueError
    (`BoundaryCommit.of`'s by field) when ``pub`` isn't one, the check it names isn't ``check``, the rule it names isn't
    ``stop``, it says whether it is a stop class without a rule or doesn't under one, or its values aren't its
    layout's."""
    c = BoundaryCommit.of(pub, COMMIT)
    if c.check != P.template_id(check):
        raise ValueError(f"a {FORMAT} commitment of check {c.check}, not {P.template_id(check)}")
    mine = None if stop is None else stop.to_json()
    if json.dumps(c.stop and dict(c.stop), sort_keys=True) != json.dumps(mine, sort_keys=True):
        raise ValueError(f"a {FORMAT} commitment under the stop rule {c.stop!r}, not the verifier's own, {mine!r}: a class "
                         f"is laid out under the rule the verifier holds, never one its commitment names")
    if (c.stop is None) != (c.stopped is None):
        raise ValueError(f"a {FORMAT} commitment says whether it is a stop class (stopped) exactly under a stop rule (stop)")
    t = c.tables or {"layers": [], "capacity": HT.CAPACITY}
    if set(t) != {"layers", "capacity"} or not isinstance(t["layers"], list):
        raise ValueError(f"a {FORMAT} commitment's tables are its layers and capacity, not {sorted(t)}")
    L = Layout(check, c.per_prompt, c.per_output, c.prompt, c.outputs, c.spans, c.run, c.inline, tuple(t["layers"]),
               t["capacity"], stop=stop, held=c.stopped is not False, verdicts=bool(c.verdicts), ids=bool(c.ids))
    listed = (*L.ports(), *L.stop_ports())
    if tuple(c.values) != listed:
        raise ValueError(f"the commitment's values {list(c.values)[:4]} are not its layout's {list(listed)[:4]}")
    return L


def record(pub: Mapping[str, Any], check: SpecializedDefinition, law: Mapping[str, Any], previous: str | None = None,
           window: Mapping[str, Any] | None = None, stop: Any = None) -> dict[str, Any]:
    """The class's boundary registration (one-stage's record) from its ``commit.json``, laid out under ``check`` and the
    stop rule ``stop`` (`layout_of`): `Layout.program` under `Layout.grain` at ``law`` (the first stage's,
    `service.selection_law`), each of its values' root (`Layout.values`), and its window: served, the run, the window's
    index as its row, the class, the check and the index's leaf, with ``window``'s items (the advice's binding)."""
    L, c = layout_of(pub, check, stop), BoundaryCommit.of(pub, COMMIT)
    roots = [{"port": p, "root": c.values[p]["root"], **e} for p, e in L.ports().items()]
    w = {"kind": "served", "run": L.run, "row": c.window, **L.provenance(), "index": c.index, "boundary": FORMAT,
         **(window or {})}
    return R.record(L.program, L.grain, law, CS.SCHEME, roots, w, previous=previous)


def commit(spans: Sequence[Span], layout: Layout, window: int, *, salts: Mapping[str, bytes] | None = None,
           index_salt: bytes | None = None, tables: Mapping[str, bytes] | None = None,
           clock: Callable[[], float] = time.perf_counter) -> Committed:
    """Commit one class's spans in ``window``: each value's rows under its domain, every leaf's salt fresh from the OS
    unless ``salts`` gives them (`reopen`), beside each history table's root (``tables``, by name: rank 0's)."""
    t0 = clock()
    if set(tables or {}) != set(layout.tables):
        raise ValueError(f"a class with {len(layout.tables)} history tables is committed with their roots, not "
                         f"{len(tables or {})}")
    rows = port_rows(spans, layout)
    domains = {p: layout.domain(p) for p in rows}
    ys = {p: os.urandom(CS.SALT_BYTES * layout.leaves(p)) for p in rows} if salts is None else dict(salts)
    roots, _bc = CS.commit_ports(domains, rows, {}, ys, schemas=dict.fromkeys(rows, SCHEMA))
    roots |= {t: bytes(tables[t]) for t in layout.tables}
    return Committed(window, layout, tuple(spans), roots, ys, os.urandom(CS.SALT_BYTES) if index_salt is None else index_salt,
                     clock() - t0)


def table_roots(answer: Mapping[str, Any], layout: Layout) -> dict[str, bytes]:
    """Rank 0's answer to the class's `boundary_tables.assignment`: each table's root by name, refused unless it answers
    every table of the layout in order, at the layout's own domain, leaves and words, or says why it built none."""
    r = TableRoots.of(answer, "rank 0's answer")
    if r.error is not None:
        raise ValueError(f"rank 0 built no history tables: {r.error}")
    got, ports = list(r.tables or ()), layout.ports()
    if [e.get("value") for e in got] != list(layout.tables):
        raise ValueError(f"rank 0 answered tables {[e.get('value') for e in got][:2]}, not the class's {list(layout.tables)[:2]}")
    for e in got:
        want = {"value": e["value"], "leaves": layout.capacity, "words": BT.WORDS, "domain": ports[e["value"]]["domain"]}
        if {k: e.get(k) for k in want} != want or len(bytes.fromhex(e["root"])) != HT.DIGEST_BYTES:
            raise ValueError(f"rank 0's table {e['value']} isn't at the class's leaves, words and domain")
    return {e["value"]: bytes.fromhex(e["root"]) for e in got}


def write(c: Committed, directory: Path) -> Path:
    """`c`'s files under ``directory`` (module docstring), ``commit.json`` last."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in ((INDEX, c.index), (CS.INDEX_SALT, c.index_salt),
                       (SALTS, b"".join(c.salts[p] for p in c.layout.committed))):
        (d / name).write_bytes(data)
        os.chmod(d / name, 0o600)
    tmp = d / f".{COMMIT}"
    tmp.write_text(json.dumps(c.public(), indent=1) + "\n")
    tmp.replace(d / COMMIT)
    return d


def reopen(directory: Path, check: SpecializedDefinition, kept: BT.Kept | None = None, stop: Any = None) -> Committed:
    """The class's commitment from its files, recommitted and held to ``commit.json``'s roots and index leaf: what the
    replay opens its spans' boundary rows from; with ``kept``, each history table rebuilt from rank 0's private files
    (`boundary_tables.Kept.table`) too. ValueError when they differ, or it names another check than ``check`` or
    another stop rule than ``stop`` (`layout_of`)."""
    d = Path(directory)
    pub = BoundaryCommit.of(json.loads((d / COMMIT).read_text()), str(d / COMMIT))
    layout = layout_of(pub.dump(), check, stop)
    spans = [Span.from_json(s) for s in json.loads((d / INDEX).read_bytes())["spans"]]
    raw, salts, end = (d / SALTS).read_bytes(), {}, 0
    for p in layout.committed:
        salts[p] = raw[end:end + CS.SALT_BYTES * layout.leaves(p)]
        end += CS.SALT_BYTES * layout.leaves(p)
    if len(raw) != end:
        raise ValueError(f"{d}: {len(raw)} bytes of salts, not {end}: {CS.SALT_BYTES} a leaf of its "
                         f"{len(layout.committed)} values")
    tables = {t: bytes.fromhex(pub.values[t]["root"]) for t in layout.tables if t in pub.values}
    if kept is not None:
        at = (pub.window, layout.run, layout.prompt, layout.outputs)
        tables = {layout.tables[r * len(layout.layers) + i]: kept.table(*at, r, x).root
                  for r in range(layout.spans) for i, x in enumerate(layout.layers)}
    c = commit(spans, layout, pub.window, salts=salts, index_salt=(d / CS.INDEX_SALT).read_bytes(), tables=tables)
    got = BoundaryCommit.of(c.public(), str(d))
    for k in ("values", "index", "program", "partition"):
        if getattr(got, k) != getattr(pub, k):
            raise ValueError(f"{d}: its files recommit to another {k} than {COMMIT}'s")
    return c


__all__ = ["COMMIT", "EGRESS", "FIELD_AT", "FIELDS", "FINISH", "FORMAT", "INDEX", "INGRESS", "MIN_CAP", "PARTS", "PORTS",
           "PROMPT", "REQUEST", "RETURN", "SALTS", "SCHEMA", "STOP", "Committed", "Layout", "Span", "cap", "check_template",
           "class_size", "commit", "egress_row", "id_rows", "ingress_row", "inlined", "layout_of", "port_rows", "record",
           "reopen", "return_port", "run_root", "sampling_of", "write"]
