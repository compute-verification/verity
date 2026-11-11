"""A served window's boundary commitment under the two-stage sampling design (core's `spot.service.TwoStage`):
what each request's replay block reads and writes at its edges, committed inside the engine core as the window closes, so
the first stage samples requests whose boundary was fixed while they were served.

**Spans and classes.** A window's spans are the requests that finished in it, in admission order (`BoundaryWindows`); a
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

**The stop block**, in a class of a serve that registers a stop rule (`Stop`: the registered model's stop ids, none
under a schedule that ignores them, its length, and the cap T on a response's ids). Each replay block of such a class
holds one instance of `served_stop`, a proof block of the class's population beside its check's: `StopRule_v1{T, NSTOP}`
over the span's own ids, so the ids served n and the finish are a block's output, a function of committed values, and
no number the prover states. Each of its ports is a value of its span, a parameter or an output of the replay block
under the value's name (`stopped`), read whole, so round 2's links hold it to the boundary rows. The class is (P, T)
whatever n is, and commits each of the two rows above as its two parts (`STOP_PORTS`), one row a span each, nothing
twice:

| value | row |
|---|---|
| `b.request` | `b.ingress`'s fields before its ids (`FIELDS`), the prompt's length and ``max_tokens`` the first two |
| `b.prompt` | P prompt ids, zero past lp |
| `b.served` | T ids, zero past n |
| `o1` | the ids served n, then the rule's finish (1 a stop id, 2 ``max_tokens``, 3 the model's length): the block's output, the replay block's second, which the serve writes without a replay |

The block reads `b.request` and `b.served`. What holds `b.request` to the request the network certifier took in is
the ingress tie's: until it does, its ``max_tokens`` is the engine's word, and a serve that writes any length there
has a correct stop block. The rule ends every response within T ids where ``max_tokens`` is at most T; past that it
can run on (finish 0), and such a span is no stop class's. A span the rule doesn't end as it was served (an abort, an
error, a request still running, a ``max_tokens`` past T, or an end the rule doesn't read: a body's own stop strings,
an ``ignore_eos`` the rule doesn't share) is filed by its size class without a stop block, its values `PORTS` (`filed`,
``p<P>-o<O>-nostop``).

**The replay block** (`Layout.replay`) is a stand-in until the class's request Definition lands: `per_prompt * P +
per_output * O` instances of one check template (`population`), so a window committed under it is audited with that one
template; or, ``inline``, those instances with each one's body in place (`inlined`), audited with the templates that body
calls, for a check template too large for one proof block whose pieces are. The boundary's values, layout and serving
path stay as they are when the replay block becomes the class's own; its program, and so its domains, name whichever
replay block it is configured with.

**Files** (`write`), under ``w<window>/p<P>-o<O>[-nostop]/``: ``commit.json`` (public: the layout, each value's root,
domain and leaves, the index's hiding leaf), and prover-only ``index.json`` (each span's engine id, prompt, sampling
inputs, ids served and finish), ``index-salt.bin`` and ``salts.bin`` (every leaf's salt, value by value); `reopen`
recommits from them and checks the roots.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import secrets
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import util as mp_util
from pathlib import Path
from typing import Any

import numpy as np

from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.population import population
from verity.circuits.program import Program
from verity.circuits.refs import Coll, coll_concat, concat
from verity.circuits.types import Array, Tuple
from verity.commitments.identity import identity_digest_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.windows import rows as CS
from verity_vllm.records.boundary_commit import FILE as COMMIT, SCHEMA as FORMAT, BoundaryCommit

SCHEMA = CS.SCHEMA_HM96_SHA512_ROW_V2
INGRESS, EGRESS = "b.ingress", "b.egress"
PORTS = (INGRESS, EGRESS)
REQUEST, PROMPT, SERVED = "b.request", "b.prompt", "b.served"
#: the stop block's output port, as round 2 names the value a return link reads: the replay block's second output
STOP_OUT = "o1"
#: the values of a class with a stop block, one row a span each: `b.ingress`'s row in two, then `b.egress`'s
STOP_PORTS = (REQUEST, PROMPT, SERVED, STOP_OUT)
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


@dataclass(frozen=True)
class Span:
    """One request as the engine core served it: its engine id, prompt ids, sampling inputs (`SAMPLING`), the ids it was
    served and how it finished (a `FINISH` name, None while running)."""

    engine_id: str
    prompt: tuple[int, ...]
    sampling: Mapping[str, Any]
    served: tuple[int, ...] = ()
    finish: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Its class (P, O)."""
        return cap(len(self.prompt)), cap(len(self.served))

    def to_json(self) -> dict[str, Any]:
        return {"engine_id": self.engine_id, "prompt": list(self.prompt), "sampling": dict(self.sampling),
                "served": list(self.served), "finish": self.finish}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Span:
        return cls(str(d["engine_id"]), tuple(d["prompt"]), dict(d["sampling"]), tuple(d["served"]), d["finish"])


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


def egress_row(span: Span, output_cap: int) -> np.ndarray:
    """Span's `b.egress` row at output cap O, as u16 words."""
    if len(span.served) > output_cap:
        raise ValueError(f"{span.engine_id}: {len(span.served)} ids served past the class's {output_cap}")
    if span.finish is not None and span.finish not in FINISH:
        raise ValueError(f"{span.engine_id}: finish {span.finish!r} is none of {FINISH}")
    finish = 0 if span.finish is None else 1 + FINISH.index(span.finish)
    return _words([len(span.served), finish, *span.served] + [0] * (output_cap - len(span.served)))


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


def _stop_refusal(S: Mapping[str, Any]) -> str | None:
    t, stop, mml = S.get("T", MIN_CAP), S.get("STOP", ()), S.get("MAX_MODEL_LEN", 1)
    if not (type(t) is int and 1 <= t < (1 << 31) - 1 and isinstance(stop, tuple) and len(stop) < 1 << 16
            and all(type(s) is int and 0 <= s < 1 << 32 for s in stop) and list(stop) == sorted(set(stop))
            and type(mml) is int and 1 <= mml < 1 << 31):
        return (f"T must be in [1, 2^31 - 1), STOP increasing u32 ids and MAX_MODEL_LEN in [1, 2^31) (got T={t!r}, "
                f"STOP={stop!r}, MAX_MODEL_LEN={mml!r})")
    return None


def served_stop(outputs: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`ServedStop_v1{T, STOP, MAX_MODEL_LEN}` (request: Array<`HEAD_U32`, i32>, ids: Array<T, i32>) -> Array<2, i32> at
    T = ``outputs``: `StopRule_v1{T, NSTOP}`'s (n, finish) over ``ids``, ``request``'s ``max_tokens`` and prompt length
    (`FIELD_AT`), the stop ids STOP and MAX_MODEL_LEN, as one value: where a span's `b.served` row stops and why. Its
    gates are the rule's and the constants': the registered stop ids and model length are statics, public in its id,
    where the rule takes them as arguments, and the rule's two words are one output, since round 2 links a template of
    one output whose every argument is a replay block's parameter (`second_stage.replay_links`). With no stop ids the
    rule reads `stop_check.NOT_AN_ID` alone, as `StopCheck_v1` does: no token's id."""
    from verity.definitions.registry import stop_check as SC, stop_rule as X
    from verity.definitions.registry.prims import I32

    def signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
        return (("request", Array(HEAD_U32, I32)), ("ids", Array(S.T, I32))), Array(2, I32)

    def body(B: Any, S: Any, request: Any, ids: Any) -> Any:
        o, stop_ids = X.word_ops(B), S.STOP or (SC.NOT_AN_ID,)
        words = [o.word(x) for x in stop_ids]
        r = B.call(bind(X.StopRule, T=S.T, NSTOP=len(stop_ids)), ids, request[FIELD_AT["max_tokens"]],
                   Coll(Array(len(stop_ids), I32), concat([w.refs for w in words])), request[FIELD_AT["lp"]],
                   o.word(S.MAX_MODEL_LEN))
        return Coll(Array(2, I32), r.refs)

    defn = CompositeDefinition("ServedStop", 1, ("T", "STOP", "MAX_MODEL_LEN"), signature, body, register=False,
                               refuse=_stop_refusal, doc="StopRule_v1 over a span's served ids at the registered stop ids "
                               "and model length, its two words one output",
                               conformance="model-untested: StopRule_v1's, which holds no capture of vLLM's finish yet")
    return bind(defn, T=outputs, STOP=tuple(stops), MAX_MODEL_LEN=max_model_len)


def stopped(replay: SpecializedDefinition, stop: SpecializedDefinition) -> SpecializedDefinition:
    """`ReplayStopped_v1{replay, stop}`: the replay block ``replay``, its body in place, then one instance of ``stop`` (a
    `served_stop`) over two parameters of its own, each named for the boundary value it is (`REQUEST`, `SERVED`) and
    read whole. Its result is ``replay``'s, then the stop's output, a second output port (`STOP_OUT`), so round 2 links
    each of the block's ports to a boundary value's rows (`second_stage.replay_links`)."""
    params, ret = replay.signature()
    mine, out = stop.signature()
    names = (REQUEST, SERVED)
    if (getattr(replay.defn, "ports", None) is not None or len(mine) != len(names)
            or {name for name, _ in params} & set(names)):
        raise ValueError(f"a stop block of {len(names)} parameters beside a replay block that declares no ports and takes "
                         f"none of {names}, not {stop.id} beside {replay.id}")

    def signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
        return (*params, *((name, t) for name, (_, t) in zip(names, mine))), Tuple(ret, out)

    def body(B: Any, S: Any, *args: Any) -> Any:
        inner = replay.defn.body(B, replay.S(), *args[:len(params)])
        return coll_concat(Tuple(ret, out), [inner, B.call(stop, *args[len(params):])])

    defn = CompositeDefinition("ReplayStopped", 1, ("replay", "stop"), signature, body, register=False,
                               doc=f"{replay.id}, then one {stop.id} over the span's boundary rows")
    return bind(defn, replay=descriptor_id(replay), stop=descriptor_id(stop))


@dataclass(frozen=True)
class Stop:
    """A serve's registered stop rule: the registered model's stop ids ``ids`` (none under a schedule that ignores them),
    its length ``max_model_len``, and ``outputs``, the cap T on a response's ids, a power of two: the output cap of every
    class that holds a stop block. Public: such a class's ``commit.json`` names it, and its stop block's id holds it."""

    ids: tuple[int, ...]
    max_model_len: int
    outputs: int

    def __post_init__(self) -> None:
        why = _stop_refusal({"T": self.outputs, "STOP": self.ids, "MAX_MODEL_LEN": self.max_model_len})
        if why is None and self.outputs != cap(self.outputs):
            why = f"its cap is a class's outputs, a power of two at least {MIN_CAP}, not {self.outputs}"
        if why is not None:
            raise ValueError(f"a stop rule's cap, ids and model length: {why}")

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
        """The stop block's template: `served_stop` at the rule."""
        return served_stop(self.outputs, self.ids, self.max_model_len)

    def end(self, span: Span) -> tuple[int, int] | None:
        """`StopRule_v1`'s (n, finish) over ``span``'s rows at the cap (its reference), where that is how the span was
        served: n the ids it was served, and its finish a stop on 1 or a length on 2 or 3. None otherwise: a span no
        stop class holds."""
        from verity.definitions.registry import stop_check as SC, stop_rule as X

        mt = span.sampling.get("max_tokens") or 0
        if (span.finish not in ("stop", "length") or len(span.served) > self.outputs or type(mt) is not int
                or not 0 <= mt < 1 << 32):
            return None
        got = X.reference([*span.served, *[0] * (self.outputs - len(span.served))], mt, self.ids or (SC.NOT_AN_ID,),
                          len(span.prompt), self.max_model_len)
        return got if got[0] == len(span.served) and got[1] in ((1,) if span.finish == "stop" else (2, 3)) else None


@dataclass(frozen=True)
class Layout:
    """One class's registration, from public items alone: the check template ``check``, its instances a span
    (``per_prompt`` a prompt row, ``per_output`` an output), the class (``prompt``, ``outputs``), the spans N and the run.
    ``inline``: the replay block holds them `inlined`, and the check's proof blocks are the templates the check template's
    body calls (`templates`); else they are its instances (`population`). ``stop``: the class's output cap is the rule's
    T, each replay block also holds one stop block (`Stop.template`, `stopped`), and the class's values are `STOP_PORTS`.
    What the prover commits under and the challenger's own copies (`ports`)."""

    check: SpecializedDefinition
    per_prompt: int
    per_output: int
    prompt: int
    outputs: int
    spans: int
    run: str
    inline: bool = False
    stop: Stop | None = None

    def __post_init__(self) -> None:
        for k in ("prompt", "outputs"):
            v = getattr(self, k)
            if not (type(v) is int and v >= MIN_CAP and v & (v - 1) == 0):
                raise ValueError(f"a class's {k} cap is a power of two at least {MIN_CAP}, not {v!r}")
        if not (type(self.spans) is int and self.spans >= 1):
            raise ValueError(f"a class registers one span or more, not {self.spans!r}")
        if self.n_rho < 2:
            raise ValueError(f"a replay block holds two proof blocks or more (TwoStage), not {self.n_rho}: "
                             f"{self.per_prompt} a prompt row and {self.per_output} an output at ({self.prompt}, {self.outputs})")
        if self.stop is not None and self.outputs != self.stop.outputs:
            raise ValueError(f"a class with a stop block is filed at its rule's cap, {self.stop.outputs} outputs, not "
                             f"{self.outputs}")
        if self.inline:
            composite = isinstance(getattr(self.check, "defn", None), CompositeDefinition)
            forms = {n.form for n in self.check.body.nodes} if composite else {"primitive"}
            if not forms <= {"call", "batch"}:
                raise ValueError(f"an inlined check template's body is calls of templates, each a proof block, not "
                                 f"{P.template_id(self.check)}'s {sorted(forms - {'call', 'batch'})}")

    @property
    def n_rho(self) -> int:
        return self.per_prompt * self.prompt + self.per_output * self.outputs

    @cached_property
    def templates(self) -> tuple[SpecializedDefinition, ...]:
        """A replay block's proof blocks' templates: the check template, or, inlined, each template its body calls, in the
        order of its first call; then the stop block's."""
        ts = (self.check,) if not self.inline else tuple({P.template_id(n.fn): n.fn for n in self.check.body.nodes}.values())
        return ts if self.stop is None else (*ts, self.stop.template)

    @property
    def query(self) -> dict[str, Any]:
        """The proof blocks in a replay block (`service.Check.query`): one template's instances, or each one's."""
        ts = self.templates
        return P.template_instance_query(ts[0]) if len(ts) == 1 else P.template_instances_query(ts)

    @cached_property
    def replay(self) -> SpecializedDefinition:
        inner = inlined(self.check, self.n_rho) if self.inline else population(self.check, self.n_rho)
        return inner if self.stop is None else stopped(inner, self.stop.template)

    @cached_property
    def program(self) -> Program:
        return P.population_program(self.replay, self.spans)

    @cached_property
    def grain(self) -> dict[str, Any]:
        return P.template_instance_query(self.replay)

    @cached_property
    def partition(self) -> dict[str, Any]:
        return P.partition(self.program, self.grain)

    @property
    def values(self) -> tuple[str, ...]:
        """The values the serve commits rows of, in the record's order: `PORTS`, or `STOP_PORTS` where the class holds a
        stop block."""
        return PORTS if self.stop is None else STOP_PORTS

    def words(self, port: str) -> int:
        return {INGRESS: 2 * (HEAD_U32 + self.prompt), EGRESS: 2 * (2 + self.outputs), REQUEST: 2 * HEAD_U32,
                PROMPT: 2 * self.prompt, SERVED: 2 * self.outputs, STOP_OUT: 4}[port]

    def domain(self, port: str) -> CommitmentDomain:
        binding = identity_digest_sha512(CS.DOMAIN_TAG, {"program": self.partition["program"],
                                                         "partition": P.digest(self.partition), "port": port, "schema": SCHEMA,
                                                         "leaves": self.spans, "run": self.run})
        return CommitmentDomain(binding, CS.OWNER, RangeIndexedDomain(0, self.spans), hash="sha512")

    def ports(self) -> dict[str, dict[str, Any]]:
        """Each value's leaves and domain id, as `Stream.register` takes them."""
        return {p: {"leaves": self.spans, "domain": self.domain(p).domain_id.hex()} for p in self.values}

    def provenance(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "outputs": self.outputs, "check": P.template_id(self.check),
                "per_prompt": self.per_prompt, "per_output": self.per_output, "inline": self.inline,
                **({} if self.stop is None else {"stop": self.stop.to_json()})}


def _of_class(spans: Sequence[Span], layout: Layout) -> None:
    if len(spans) != layout.spans or any((cap(len(s.prompt)), layout.outputs if layout.stop else s.shape[1])
                                         != (layout.prompt, layout.outputs) for s in spans):
        raise ValueError(f"the spans are not {layout.spans} of class ({layout.prompt}, {layout.outputs})")


def stop_rows(spans: Sequence[Span], layout: Layout) -> dict[str, np.ndarray]:
    """A stop class's rows (`STOP_PORTS`), span by span: `b.ingress`'s row as its fields and its prompt ids, the ids
    served at the cap, and the stop block's output, the rule's (n, finish). ValueError for a span the rule doesn't end
    as it was served (`Stop.end`): the block's output on its rows is not its end."""
    _of_class(spans, layout)
    ends = [layout.stop.end(s) for s in spans]
    bad = next((s for s, e in zip(spans, ends) if e is None), None)
    if bad is not None:
        raise ValueError(f"{bad.engine_id}: the stop rule doesn't end this span as it was served ({len(bad.served)} ids, "
                         f"finish {bad.finish!r}), so no class with a stop block holds it: `filed` files it without one")
    ingress = np.stack([ingress_row(s, layout.prompt) for s in spans])
    return {REQUEST: ingress[:, :2 * HEAD_U32], PROMPT: ingress[:, 2 * HEAD_U32:],
            SERVED: np.stack([_words([*s.served, *[0] * (layout.outputs - len(s.served))]) for s in spans]),
            STOP_OUT: np.stack([_words(e) for e in ends])}


def port_rows(spans: Sequence[Span], layout: Layout) -> dict[str, np.ndarray]:
    """Each value's rows, (N, words) u16, span by span."""
    if layout.stop is not None:
        return stop_rows(spans, layout)
    _of_class(spans, layout)
    return {INGRESS: np.stack([ingress_row(s, layout.prompt) for s in spans]),
            EGRESS: np.stack([egress_row(s, layout.outputs) for s in spans])}


@dataclass
class Committed:
    """One class's commitment in a window: its layout and spans, each value's root, the leaves' salts and the index's."""

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
        L, ports = self.layout, self.layout.ports()
        return BoundaryCommit(
            window=self.window, run=L.run, **L.provenance(), spans=L.spans, replay=P.template_id(L.replay),
            program=L.partition["program"], partition=P.digest(L.partition), schema=SCHEMA,
            values={p: {"root": self.roots[p].hex(), "words": L.words(p), **ports[p]} for p in ports},
            index=self.index_leaf, index_rule=CS.INDEX_RULE, seconds=round(self.seconds, 3)).dump()

    def record(self, law: Mapping[str, Any], previous: str | None = None,
               window: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return record(self.public(), self.layout.check, law, previous, window)


def layout_of(pub: Mapping[str, Any], check: SpecializedDefinition) -> Layout:
    """The class's `Layout` from its ``commit.json`` (``pub``) and the check template it names: the challenger's own copy,
    from public items alone. ValueError (`BoundaryCommit.of`'s by field) when ``pub`` isn't one, or ``check`` isn't the one
    it names."""
    c = BoundaryCommit.of(pub, COMMIT)
    if c.check != P.template_id(check):
        raise ValueError(f"a {FORMAT} commitment of check {c.check}, not {P.template_id(check)}")
    return Layout(check, c.per_prompt, c.per_output, c.prompt, c.outputs, c.spans, c.run, c.inline, Stop.of(c.stop))


def record(pub: Mapping[str, Any], check: SpecializedDefinition, law: Mapping[str, Any], previous: str | None = None,
           window: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The class's boundary registration (one-stage's record) from its ``commit.json``: `Layout.program` under
    `Layout.grain` at ``law`` (the first stage's, `service.selection_law`), each value's root, and its window: served, the
    run, the window's index as its row, the class, the check and the index's leaf, with ``window``'s items (the advice's
    binding)."""
    L, c = layout_of(pub, check), BoundaryCommit.of(pub, COMMIT)
    ports = L.ports()
    if set(c.values) != set(ports):
        raise ValueError(f"the commitment's values {sorted(c.values)} are not its layout's {sorted(ports)}")
    roots = [{"port": p, "root": c.values[p]["root"], **ports[p]} for p in ports]
    w = {"kind": "served", "run": L.run, "row": c.window, **L.provenance(), "index": c.index, "boundary": FORMAT,
         **(window or {})}
    return R.record(L.program, L.grain, law, CS.SCHEME, roots, w, previous=previous)


def commit(spans: Sequence[Span], layout: Layout, window: int, *, salts: Mapping[str, bytes] | None = None,
           index_salt: bytes | None = None, clock: Callable[[], float] = time.perf_counter) -> Committed:
    """Commit one class's spans in ``window``: each value's rows under its domain, every leaf's salt fresh from the OS
    unless ``salts`` gives them (`reopen`)."""
    t0 = clock()
    rows = port_rows(spans, layout)
    domains = {p: layout.domain(p) for p in rows}
    ys = {p: os.urandom(CS.SALT_BYTES * layout.spans) for p in rows} if salts is None else dict(salts)
    roots, _bc = CS.commit_ports(domains, rows, {}, ys, schemas=dict.fromkeys(rows, SCHEMA))
    return Committed(window, layout, tuple(spans), roots, ys, os.urandom(CS.SALT_BYTES) if index_salt is None else index_salt,
                     clock() - t0)


def write(c: Committed, directory: Path) -> Path:
    """`c`'s files under ``directory`` (module docstring), ``commit.json`` last."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in ((INDEX, c.index), (CS.INDEX_SALT, c.index_salt),
                       (SALTS, b"".join(c.salts[p] for p in c.layout.values))):
        (d / name).write_bytes(data)
        os.chmod(d / name, 0o600)
    tmp = d / f".{COMMIT}"
    tmp.write_text(json.dumps(c.public(), indent=1) + "\n")
    tmp.replace(d / COMMIT)
    return d


def reopen(directory: Path, check: SpecializedDefinition) -> Committed:
    """The class's commitment from its files, recommitted and held to ``commit.json``'s roots and index leaf: what the
    replay opens its spans' boundary rows from. ValueError when they differ, or ``check`` isn't the one it names."""
    d = Path(directory)
    pub = BoundaryCommit.of(json.loads((d / COMMIT).read_text()), str(d / COMMIT))
    layout = layout_of(pub.dump(), check)
    spans = [Span.from_json(s) for s in json.loads((d / INDEX).read_bytes())["spans"]]
    raw, n = (d / SALTS).read_bytes(), CS.SALT_BYTES * layout.spans
    if len(raw) != n * len(layout.values):
        raise ValueError(f"{d}: {len(raw)} bytes of salts, not {len(layout.values)} values' {n}")
    c = commit(spans, layout, pub.window, salts={p: raw[k * n:(k + 1) * n] for k, p in enumerate(layout.values)},
               index_salt=(d / CS.INDEX_SALT).read_bytes())
    got = BoundaryCommit.of(c.public(), str(d))
    for k in ("values", "index", "program", "partition"):
        if getattr(got, k) != getattr(pub, k):
            raise ValueError(f"{d}: its files recommit to another {k} than {COMMIT}'s")
    return c


def filed(spans: Sequence[Span], stop: Stop | None = None) -> dict[tuple[int, int, bool], list[Span]]:
    """The spans by class, and whether the class holds a stop block: a span rule ``stop`` ends as it was served
    (`Stop.end`) in the stop class of its prompt cap, (P, T); every other by its size class, without one; all without one
    where ``stop`` is None. Each class's spans in admission order, the classes in order of their first span."""
    out: dict[tuple[int, int, bool], list[Span]] = {}
    for s in spans:
        held = stop is not None and stop.end(s) is not None
        out.setdefault((s.shape[0], stop.outputs, True) if held else (*s.shape, False), []).append(s)
    return out


#: vLLM's RequestStatus names (past ``finished_``) for a `FINISH` name; its FinishReason names are `FINISH`'s own
_STATUS = {"stopped": "stop", "length_capped": "length", "aborted": "abort"}


def _finish(reason: Any) -> str:
    name = str(getattr(reason, "name", reason)).lower().removeprefix("finished_")
    name = _STATUS.get(name, name)
    return name if name in FINISH else "error"


class BoundaryWindows:
    """The serving path's boundary commitment, on one scheduler (`install`): every request it adds (`admit`), the ids
    each step serves it (`served`) and the requests it finishes from outside a step (`finished`: an abort). A window
    closes at the first schedule ``window_s`` after it opened (`tick`), at a step that leaves no request running once it
    holds a finished one (``drained``), and at `stop`; its finished spans are committed class by class (`commit`, `write`)
    on a thread of their own, under ``out``/w<window>/, before anything can sample them, and each class's
    ``commit.json`` goes to ``sink`` too. ``check`` is the check template, or a callable giving it on that thread, and
    ``inline`` its `Layout`'s. ``stop``: the serve's registered stop rule; the spans it ends as they were served are
    committed in the stop classes, (P, T), and the rest by size class without a stop block, under ``p<P>-o<O>-nostop``
    (`filed`). A failed commit is reported to ``sink`` (``error``) and registers nothing."""

    def __init__(self, out: Path, check: SpecializedDefinition | Callable[[], SpecializedDefinition], per_prompt: int,
                 per_output: int, window_s: float, *, drained: bool = True, inline: bool = False,
                 sink: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 serve: str | None = None, stop: Stop | None = None) -> None:
        self.out, self.per_prompt, self.per_output = Path(out), int(per_prompt), int(per_output)
        self.window_s, self.drained, self.inline, self.stop_rule = float(window_s), drained, inline, stop
        self.sink, self.clock = sink, clock
        self.serve = serve or secrets.token_hex(32)
        self._check, self._check_fn = (check, None) if isinstance(check, SpecializedDefinition) else (None, check)
        self.open: dict[str, dict[str, Any]] = {}
        self.window, self.opened, self.stopped = 0, clock(), False
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="boundary-commit", daemon=True)
        self.thread.start()

    @classmethod
    def of(cls, cfg: Mapping[str, Any], out: Path, sink: Callable[[dict], None] | None = None) -> BoundaryWindows:
        """From a serve's configuration: ``check``, a serving template's descriptor id (`check_template`), and
        ``per_prompt``, ``per_output``, ``window_s``, ``drained``, ``inline`` and ``stop`` (`Stop.of`: the registered
        model's stop ids and length, and the cap on a response's ids)."""
        t = str(cfg["check"])
        return cls(out, lambda: check_template(t), int(cfg.get("per_prompt", 0)), int(cfg.get("per_output", 1)),
                   float(cfg.get("window_s", 600)), drained=bool(cfg.get("drained", True)),
                   inline=bool(cfg.get("inline", False)), sink=sink, stop=Stop.of(cfg.get("stop")))

    @property
    def check(self) -> SpecializedDefinition:
        if self._check is None:
            self._check = self._check_fn()
        return self._check

    def admit(self, request: Any) -> None:
        self.open[request.request_id] = {"prompt": tuple(request.prompt_token_ids or ()),
                                         "sampling": sampling_of(request.sampling_params), "served": [], "finish": None}

    def served(self, outputs: Any) -> None:
        """A step's outputs (the scheduler's ``update_from_output``'s: engine core outputs by client)."""
        for eco in (outputs or {}).values():
            for o in getattr(eco, "outputs", ()) or ():
                r = self.open.get(o.request_id)
                if r is None:
                    continue
                r["served"].extend(o.new_token_ids or ())
                if o.finish_reason is not None:
                    r["finish"] = _finish(o.finish_reason)
        self._drained()

    def finished(self, request_ids: Any, status: Any) -> None:
        """The scheduler's ``finish_requests``: requests finished from outside a step (a client's abort)."""
        for k in [request_ids] if isinstance(request_ids, str) else list(request_ids or ()):
            r = self.open.get(k)
            if r is not None and r["finish"] is None:
                r["finish"] = _finish(status)
        self._drained()

    def _drained(self) -> None:
        if self.drained and self.open and all(r["finish"] is not None for r in self.open.values()):
            self.close()

    def tick(self) -> None:
        if self.clock() - self.opened >= self.window_s:
            self.close()

    def close(self, every: bool = False) -> None:
        done = [k for k, r in self.open.items() if every or r["finish"] is not None]
        now = self.clock()
        if done:
            spans = [Span(k, self.open[k]["prompt"], self.open[k]["sampling"], tuple(self.open[k]["served"]),
                          self.open[k]["finish"]) for k in done]
            for k in done:
                del self.open[k]
            self.jobs.put((self.window, self.opened, now, len(self.open), spans))
            self.window += 1
        self.opened = now

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.close(every=True)
        self.jobs.put(None)
        self.thread.join()

    def _run(self) -> None:
        while (job := self.jobs.get()) is not None:
            window, opened, closed, carried, spans = job
            for (p, o, held), mine in filed(spans, self.stop_rule).items():
                unheld = self.stop_rule is not None and not held
                try:
                    layout = Layout(self.check, self.per_prompt, self.per_output, p, o, len(mine),
                                    run_root(self.serve, window), self.inline, self.stop_rule if held else None)
                    c = commit(mine, layout, window)
                    write(c, self.out / f"w{window}" / f"p{p}-o{o}{'-nostop' if unheld else ''}")
                    line = {**c.public(), "serve": self.serve, "open_s": opened, "close_s": closed, "carried": carried}
                except (ValueError, TypeError, OverflowError, struct.error, OSError) as e:  # the sink says why
                    line = {"format": FORMAT, "window": window, "prompt": p, "outputs": o, "spans": len(mine),
                            **({"stop": self.stop_rule.to_json()} if held else {}), "error": f"{type(e).__name__}: {e}"}
                if self.sink is not None:
                    self.sink(line)

    def install(self, sched: Any, hooks: Any) -> BoundaryWindows:
        """Wrap ``sched``'s ``add_request``, ``update_from_output``, ``finish_requests`` and ``schedule`` through ``hooks``
        (an `engine.hooks.Hooks`), and stop at the process's exit: at exit and at a `multiprocessing` child's, which runs
        no `atexit` hook (vLLM's engine core is one). Refused while the scheduler caches prefixes: a hit would read
        another request's rows, across the replay blocks' boundary."""
        if getattr(getattr(sched, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError("the boundary commitment needs prefix caching off (--no-enable-prefix-caching): a cache hit "
                             "reads another request's rows")

        def add_request(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(request: Any, *a: Any, **kw: Any) -> Any:
                self.admit(request)
                return orig(request, *a, **kw)
            return wrapped

        def update_from_output(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*a: Any, **kw: Any) -> Any:
                outs = orig(*a, **kw)
                self.served(outs)
                return outs
            return wrapped

        def finish_requests(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
                ids = request_ids if isinstance(request_ids, str) else list(request_ids or ())
                out = orig(ids, finished_status, *a, **kw)
                self.finished(ids, finished_status)
                return out
            return wrapped

        def schedule(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*a: Any, **kw: Any) -> Any:
                self.tick()
                return orig(*a, **kw)
            return wrapped

        hooks.wrap(sched, "add_request", add_request)
        hooks.wrap(sched, "update_from_output", update_from_output)
        hooks.wrap(sched, "finish_requests", finish_requests)
        hooks.wrap(sched, "schedule", schedule)
        atexit.register(self.stop)
        mp_util.Finalize(None, self.stop, exitpriority=100)
        return self


__all__ = ["COMMIT", "EGRESS", "FIELD_AT", "FIELDS", "FINISH", "FORMAT", "INDEX", "INGRESS", "MIN_CAP", "PORTS", "PROMPT",
           "REQUEST", "SALTS", "SCHEMA", "SERVED", "STOP_OUT", "STOP_PORTS", "BoundaryWindows", "Committed", "Layout", "Span",
           "Stop", "cap", "check_template", "commit", "egress_row", "filed", "ingress_row", "inlined", "layout_of",
           "port_rows", "record", "reopen", "run_root", "sampling_of", "served_stop", "stop_rows", "stopped", "write"]
