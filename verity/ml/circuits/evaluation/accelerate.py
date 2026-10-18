"""Word accelerators of Boolean Definitions: the Boolean Definition is the definition, an admitted word Definition its speed.

A Boolean Definition names the word Definition it computes (`boolean.trace.word_of`): the same function on the words of its word
view (`boolean.word_type`). `admit(defn)` admits that word Definition as the accelerator of every complete specialization of the
Boolean composite ``defn``, and circuit-check holds it equal to the gates (`circuit_check.checks.word_view`: on the word
Definition's vectors, the word Definition must equal the Boolean one evaluated gate by gate). Admission is explicit: a Boolean
Definition that isn't admitted is evaluated on its gates.

    admit(defn)                            admit ``defn``'s word Definition as its accelerator; returns ``defn``
    accelerator(target)                    the admitted word Definition of ``target``, else None
    evaluate_accelerated(target, word, *bits)   ``N`` instances on bits: bits to words (`word_type`), the word Definition
                                           (`evaluate_batch`, so its kernel when one is registered), words back to bits (`bit_type`)
    evaluate_circuit(circuit, transcript)  a Program's circuit on its Input values: each Call of an admitted Definition through
                                           its accelerator, writing only its returns; every other gate gate by gate

`evaluate`, `evaluate_batch` and `evaluate_circuit` dispatch here by themselves; ``gates=True`` (`evaluate_batch`'s
``kernel="gates"``) evaluates the gates instead, as `bits.evaluate_bits` and `bits.evaluate_words` always do. Values are the
same either way, so nothing a run commits depends on the path; only a transcript's gates inside an accelerated Call are absent.
"""
from __future__ import annotations

import itertools
import weakref
from typing import Any, Sequence

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, FunctionDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.evaluate import ExecutionTranscript, MissingValue, evaluate_region
from verity.ml.circuits.layout import Circuit, FunctionCall, Scope
from verity.ml.circuits.types import Value

__all__ = ["accelerator", "admit", "admitted", "evaluate_accelerated", "evaluate_circuit", "to_bits", "to_words"]

_ADMITTED: dict[str, CompositeDefinition] = {}
_WORD: "weakref.WeakKeyDictionary[SpecializedDefinition, FunctionDefinition]" = weakref.WeakKeyDictionary()
_REACHES: "weakref.WeakKeyDictionary[SpecializedDefinition, bool]" = weakref.WeakKeyDictionary()


def admit(defn: CompositeDefinition) -> CompositeDefinition:
    """Admit the word Definition ``defn`` names (its ``word``) as the accelerator of ``defn``'s complete specializations."""
    if not isinstance(defn, CompositeDefinition) or getattr(defn, "word", None) is None:
        raise TypeError(f"{getattr(defn, 'id', defn)!r}: only a Boolean composite that names its word Definition is admitted")
    prev = _ADMITTED.setdefault(defn.id, defn)
    if prev is not defn:
        raise ValueError(f"{defn.id} is already admitted as a different Definition")
    _REACHES.clear()
    return defn


def admitted() -> tuple[CompositeDefinition, ...]:
    return tuple(_ADMITTED.values())


def _spec(target: Any) -> SpecializedDefinition | None:
    if isinstance(target, CompositeDefinition):
        if target.statics:
            return None
        target = bind(target)
    return target if isinstance(target, SpecializedDefinition) and target.complete else None


def accelerator(target: Any) -> FunctionDefinition | None:
    """The word Definition that evaluates ``target``: when ``target`` is a complete specialization of an admitted Definition,
    its ``word_of``, whose signature must be ``target``'s word view; else None."""
    t = _spec(target)
    if t is None or _ADMITTED.get(t.defn.id) is not t.defn:
        return None
    hit = _WORD.get(t)
    if hit is None:
        from verity.ml.circuits.boolean.trace import word_of

        from .reference import signature_of

        if not BL.is_boolean(t):
            raise TypeError(f"{t.id} is admitted, but it is not a Boolean Definition")
        word = word_of(t)
        word = bind(word) if isinstance(word, CompositeDefinition) else word
        params, ret = t.signature()
        view = (tuple(BL.word_type(p) for _, p in params), BL.word_type(ret))
        if signature_of(word) != view:
            raise TypeError(f"{t.id}'s accelerator {word.id} has signature {signature_of(word)}, not its word view {view}")
        hit = _WORD[t] = word
    return hit


# ---- bits and words --------------------------------------------------------------------------------------------------------

def to_words(bits: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    """(N, sum(widths)) of 0/1 -> (N, len(widths)) words, element e from its widths[e] bits, least significant first; uint64
    when every element fits, else Python ints."""
    n = bits.shape[0]
    wide = max(widths, default=1) > 64
    out = np.zeros((n, len(widths)), dtype=object if wide else np.uint64)
    k = e = 0
    for w, run in itertools.groupby(widths):
        r = len(list(run))
        block = np.asarray(bits[:, k:k + r * w]).reshape(n, r, w)
        if w <= 64:
            out[:, e:e + r] = (block.astype(np.uint64) << np.arange(w, dtype=np.uint64)).sum(axis=2, dtype=np.uint64)
        else:
            for j in range(w):
                out[:, e:e + r] += block[:, :, j].astype(object) << j
        k, e = k + r * w, e + r
    return out


def to_bits(words: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    """(N, len(widths)) words -> (N, sum(widths)) uint8 of 0/1, the inverse of `to_words`."""
    n = words.shape[0]
    out = np.empty((n, sum(widths)), dtype=np.uint8)
    k = e = 0
    for w, run in itertools.groupby(widths):
        r = len(list(run))
        block = words[:, e:e + r]
        if w <= 64 and block.dtype.kind != "O":
            bits = (block.astype(np.uint64)[:, :, None] >> np.arange(w, dtype=np.uint64)) & np.uint64(1)
        else:
            bits = np.array([[[(int(v) >> j) & 1 for j in range(w)] for v in row] for row in block], dtype=np.uint8)
        out[:, k:k + r * w] = bits.reshape(n, r * w)
        k, e = k + r * w, e + r
    return out


def evaluate_accelerated(target: SpecializedDefinition, word: FunctionDefinition, *inputs: np.ndarray) -> np.ndarray:
    """``N`` instances of ``target`` through ``word`` (its `accelerator`): inputs and output in ``target``'s own signature, shaped
    as `evaluate_batch` shapes them, every leaf 0 or 1."""
    from .batch import evaluate_batch, type_shape

    params, ret = target.signature()
    words = []
    for a, (_, p) in zip(inputs, params):
        a = np.asarray(a)
        wt = BL.word_type(p)
        words.append(to_words(a.reshape(a.shape[0], p.leaves), BL.elements(p)).reshape((a.shape[0], *type_shape(wt))))
    n = inputs[0].shape[0] if inputs else 0
    out = np.asarray(evaluate_batch(word, *words))
    return to_bits(out.reshape(n, -1), BL.elements(ret)).reshape((n, *type_shape(ret)))


def evaluate_one(target: SpecializedDefinition, word: FunctionDefinition, args: Sequence[Any]) -> tuple[int, ...]:
    """`evaluate`'s one instance through ``word``: one bit list per parameter, the output's bits."""
    from .reference import evaluate

    params, ret = target.signature()
    wargs: list[Any] = []
    for a, (_, p) in zip(args, params):
        bits = [a] if isinstance(a, int) else [int(v) for v in a]
        if len(bits) != p.leaves:
            raise ValueError(f"{target.id}: expected {p.leaves} leaves, got {len(bits)}")
        ws = to_words(np.asarray([bits], dtype=np.uint8), BL.elements(p))[0].tolist()
        wargs.append(int(ws[0]) if isinstance(BL.word_type(p), Value) else [int(v) for v in ws])
    out = evaluate(word, *wargs)
    return tuple(int(b) for b in to_bits(np.asarray([out], dtype=object), BL.elements(ret))[0])


# ---- a Program's circuit ---------------------------------------------------------------------------------------------------

def _reaches(fn: SpecializedDefinition) -> bool:
    """Whether ``fn`` is admitted or calls, at any depth, a Definition that is."""
    hit = _REACHES.get(fn)
    if hit is None:
        hit = accelerator(fn) is not None or any(
            not isinstance(nd.fn, PrimitiveDefinition) and _reaches(nd.fn) for nd in fn.body.nodes)
        _REACHES[fn] = hit
    return hit


def _accelerate(fn: SpecializedDefinition, word: FunctionDefinition, calls: list[FunctionCall], t: ExecutionTranscript) -> None:
    from .batch import type_shape

    params, _ = fn.signature()
    rows = []
    for c in calls:
        try:
            rows.append([t[g] for g in c.args])
        except KeyError as e:
            raise MissingValue(f"{c!r}: argument gate {e.args[0]} is not in the transcript") from None
    a = np.asarray(rows, dtype=np.uint8).reshape(len(calls), -1)
    ins, k = [], 0
    for _, p in params:
        ins.append(a[:, k:k + p.leaves].reshape((len(calls), *type_shape(p))))
        k += p.leaves
    out = evaluate_accelerated(fn, word, *ins).reshape(len(calls), -1).tolist()
    for c, vals in zip(calls, out):
        for g, v in zip(c.returns, vals):
            if c.lo <= g < c.hi:
                t[g] = v
            elif t.get(g, v) != v:
                raise ValueError(f"{c!r}: its accelerator {word.id} returns {v} for parameter gate {g}, which holds {t[g]}")


def _scope(circuit: Circuit, sc: Scope, t: ExecutionTranscript) -> None:
    body = sc.spec.body
    for k, nd in enumerate(body.nodes):
        lo = sc.offset + body.offsets[k]
        if isinstance(nd.fn, PrimitiveDefinition) or not _reaches(nd.fn):
            evaluate_region(circuit, circuit.region(lo, lo + nd.gates), t)
            continue
        calls = [FunctionCall(circuit, sc, k, m) for m in range(1 if nd.form == "call" else nd.n)]
        word = accelerator(nd.fn)
        if word is None:
            for c in calls:
                _scope(circuit, c.scope(), t)
        elif nd.form == "scan":
            for c in calls:
                _accelerate(nd.fn, word, [c], t)
        else:
            _accelerate(nd.fn, word, calls, t)


def evaluate_circuit(circuit: Circuit, transcript: ExecutionTranscript, *, gates: bool = False) -> ExecutionTranscript:
    """Evaluate ``circuit`` on the Input values in ``transcript`` (mutated and returned). Each Call of an admitted Definition is
    evaluated through its accelerator, a batch's members together, and writes only its returns; every other gate is evaluated
    gate by gate (`evaluate_region`). ``gates=True`` is `evaluate_region` over the whole circuit: every gate's value."""
    if gates:
        return evaluate_region(circuit, circuit.region(0, circuit.size), transcript)
    _scope(circuit, circuit.root_scope, transcript)
    return transcript
