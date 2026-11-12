"""A serve's registered stop rule on its boundary commitment (`boundary_rows`): the rule (`Stop`), the stop block
that computes where a response ends (`egress_stop`), which class each span is filed in (`filed`), and a stop class's
stop record (`stop_registration`, `stop_record`).

**The rule** is the registered model's stop ids, its length, and T, the most ids a response has. A span the rule ends as
it was served (`Stop.word`) is in the stop class of its prompt's size, (P, T) whatever n is, which commits `b.stop`: the
output of the span's stop block (`egress_stop`: `StopRule16_v1{T, NSTOP}` over the span's `b.request` and `b.egress`
rows at the registered stop ids and length, with no stop id for a span whose row sets ``ignore_eos``), so the ids served
n is a function of committed values and no number the serve states. Every other span is filed by its size class with no
`b.stop` and no stop block (`filed`, ``p<P>-o<O>-nostop``): an abort, an error, a request still running, one the rule
runs past T (finish 0), an end the rule doesn't read (a body's own stop strings), a ``min_tokens`` other than 0, and,
where the serve knows its requests' releases, a request the network certifier didn't release. A failure the engine
doesn't see (the certifier's, after the engine's last step for the request) is not known here: such a span is filed as
the engine ended it.

**The stop record.** The stop blocks are a population of their own, one a span, slot r its r-th span's
(`Layout.stop_program`), registered one-stage in a record of its own that names the class's boundary record: its two
inputs are that record's ports `b.request` and `b.egress`, taken as output values under their roots and domains, and
`b.stop` is its one port, under the window domain of the stop population's own program, which the serve commits it
under as the window closes.

**Whose rule.** The verifier's own copy, as the check template is: `stop_registration` and `stop_record` take it, as
`boundary_rows.layout_of` does, and a ``commit.json`` that names another rule is refused, so the block a stop record
proves is the one the verifier holds and never one the serve chose.

**What the block reads.** What holds `b.request` to the request the network certifier took in is the ingress tie's:
until it does, its ``max_tokens`` is the engine's word, and a serve that writes any length there has a correct stop
block.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.commitments.windows import rows as CS
from verity.protocols.spot.verifier.one_stage import registration as R
from verity.protocols.spot.verifier.one_stage import served as RS

from verity_vllm.pipeline.boundary_rows import (
    EGRESS,
    FIELD_AT,
    HEAD_U32,
    REQUEST,
    SCHEMA,
    Span,
    class_size,
    layout_of,
)
from verity_vllm.records.boundary_commit import FILE as COMMIT
from verity_vllm.records.boundary_commit import BoundaryCommit

#: the stop block's parameters and the boundary value each one's rows are; its output is `STOP`
STOP_READS = {"request": REQUEST, "ids": EGRESS}


def _stop_refusal(S: Mapping[str, Any]) -> str | None:
    t, stop, mml = S.get("T", 1), S.get("STOP", ()), S.get("MAX_MODEL_LEN", 1)
    if not (type(t) is int and 1 <= t < 1 << 13 and isinstance(stop, tuple) and len(stop) < 1 << 16
            and all(type(s) is int and 0 <= s < 1 << 32 for s in stop) and list(stop) == sorted(set(stop))
            and type(mml) is int and 1 <= mml < 1 << 31):
        return (f"T must be in [1, 2^13), which the stop word's n holds, STOP increasing u32 ids and MAX_MODEL_LEN in "
                f"[1, 2^31) (got T={t!r}, STOP={stop!r}, MAX_MODEL_LEN={mml!r})")
    return None


def egress_stop(outputs: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`EgressStop_v1{T, STOP, MAX_MODEL_LEN}` (request: Array<`HEAD_U32`, i32>, ids: Array<T, i32>) -> u16 at T =
    ``outputs``: `StopRule16_v1{T, NSTOP}`'s word over ``ids``, ``request``'s ``max_tokens`` and prompt length
    (`FIELD_AT`), the stop ids STOP and MAX_MODEL_LEN: where a span's `b.egress` row stops and why, n + finish * 8192.
    A request whose ``ignore_eos`` word in ``request`` is not 0 has no stop id: the rule reads `stop_check.NOT_AN_ID`,
    no token's id, in place of each, as it does where STOP is empty (`StopCheck_v1`'s way). Its gates are the rule's,
    the constants' and that choice's: the registered stop ids and model length are statics, public in its id, where the
    rule takes them as inputs, since a population reads each parameter as one value's whole rows (`stop_registration`)
    and the registered model's are no value."""
    from verity.definitions.registry import stop_check as SC
    from verity.definitions.registry import stop_rule as X

    I32 = X.WORDS.I32

    def signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
        return (("request", Array(HEAD_U32, I32)), ("ids", Array(S.T, I32))), X.WORDS.U16

    def body(B: Any, S: Any, request: Any, ids: Any) -> Any:
        o = X.word_ops(B)
        words = [none := o.word(SC.NOT_AN_ID)]
        if S.STOP:
            heeds = o.ieq(request[FIELD_AT["ignore_eos"]], o.word(0))
            words = [o.isel(heeds, o.word(x), none) for x in S.STOP]
        return B.call(bind(X.StopRule16, T=S.T, NSTOP=len(words)), ids, request[FIELD_AT["max_tokens"]],
                      Coll(Array(len(words), I32), concat([w.refs for w in words])), request[FIELD_AT["lp"]],
                      o.word(S.MAX_MODEL_LEN))

    defn = CompositeDefinition("EgressStop", 1, ("T", "STOP", "MAX_MODEL_LEN"), signature, body, register=False,
                               refuse=_stop_refusal, doc="StopRule16_v1 over a span's b.egress ids and b.request row at "
                               "the registered stop ids and model length, with no stop id where the row sets ignore_eos",
                               conformance="model-untested: StopRule16_v1's, which holds no capture of vLLM's finish yet")
    return bind(defn, T=outputs, STOP=tuple(stops), MAX_MODEL_LEN=max_model_len)


@dataclass(frozen=True)
class Stop:
    """A serve's registered stop rule: the registered model's stop ids ``ids`` (a request that sets ``ignore_eos`` has
    none, which the stop block reads off its row), its length ``max_model_len``, and ``outputs``, T, the most ids a
    response has, a power of two: every stop class's output size. Public: each class's ``commit.json`` names it, and the
    stop block's id holds its ids, length and T."""

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
        """The stop block's output on ``span``'s rows at T, n + finish * 8192 (`StopRule16_v1`'s reference, over no
        stop id where the span sets ``ignore_eos``), where that is how the span was served: n the ids it was served and
        its finish a stop on 1 or a length on 2 or 3, with a ``min_tokens`` of 0, which the rule doesn't read. None
        otherwise: a span no stop class holds."""
        from verity.definitions.registry import stop_check as SC
        from verity.definitions.registry import stop_rule as X

        s = span.sampling
        mt = s.get("max_tokens") or 0
        if (span.finish not in ("stop", "length") or len(span.served) > self.outputs or type(mt) is not int
                or not 0 <= mt < 1 << 32 or (s.get("min_tokens") or 0) != 0):
            return None
        stops = () if s.get("ignore_eos") else self.ids
        word = X.reference16([*span.served, *[0] * (self.outputs - len(span.served))], mt, stops or (SC.NOT_AN_ID,),
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


def stop_registration(pub: Mapping[str, Any], check: SpecializedDefinition, stop: Stop,
                      boundary: Mapping[str, Any]) -> RS.ServedRegistration:
    """A stop class's stop population's registered values, from its ``commit.json``, the verifier's rule ``stop`` and
    its boundary record (``boundary``, `boundary_rows.record`'s, the log's): each of the stop block's parameters
    (`STOP_READS`) is the boundary record's port of that value, one row a span, as an output value under that port's
    root and domain (`served.register_served`). What a prover's record registers (``registered``) and what a verifier
    holds it to (`registration.check(..., registered=values, outputs=outputs, schemas=schemas)`, the log holding
    ``boundary``). ValueError where the commitment names another rule than ``stop``, the class is no stop class or
    ``boundary`` isn't its record."""
    L, c = layout_of(pub, check, stop), BoundaryCommit.of(pub, COMMIT)
    mine = {r["port"]: r for r in boundary["roots"]}
    want = {p: {"port": p, "root": c.values[p]["root"], **e} for p, e in L.ports().items()}
    if mine != want:
        raise ValueError("the boundary record's roots are not this commitment's values, under its layout's domains")
    digest = R.digest(boundary)
    trees = [RS.ServedTree(name, mine[p]["root"], L.spans, L.words(p) // 2, mine[p]["domain"], 32, output=(digest, p),
                           row_schema=SCHEMA) for name, p in STOP_READS.items()]
    return RS.register_served(L.stop_program, L.stop_query, trees, log=[boundary])


def stop_record(pub: Mapping[str, Any], check: SpecializedDefinition, stop: Stop, design: Mapping[str, Any],
                boundary: Mapping[str, Any], window: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A stop class's stop population's registration (one-stage's record) under the verifier's rule ``stop``, the next
    of its log after its boundary record ``boundary``: `Layout.stop_program` under `Layout.stop_query` at the sampling
    design ``design`` over the class's stop blocks, one a span; its one port `b.stop`, the blocks' outputs; its
    registered values the boundary record's ports (`stop_registration`); and that record's window with ``window``'s
    items, named the stop population's."""
    L, c = layout_of(pub, check, stop), BoundaryCommit.of(pub, COMMIT)
    reg = stop_registration(pub, check, stop, boundary)
    roots = [{"port": p, "root": c.values[p]["root"], **e} for p, e in L.stop_ports().items()]
    w = {**boundary["window"], **(window or {}), "population": "stop"}
    return R.record(L.stop_program, L.stop_query, design, CS.SCHEME, roots, w, previous=R.digest(boundary),
                    registered=reg.registered)


__all__ = ["STOP_READS", "Stop", "egress_stop", "filed", "stop_record", "stop_registration"]
