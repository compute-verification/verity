"""Explicit-expansion evaluator (the oracle, IR §6).

An execution trace is an indexed assignment gate index -> encoding (Python int bit pattern). Evaluation
walks a gate selection gate by gate in canonical order, reading operands from the trace and applying the
primitive's total evaluator. `Input` gates are never evaluated: their values must already be in the
trace (they are the prescribed inputs; the correspondence tool authenticates them separately).
"""

from __future__ import annotations

from typing import Iterable

from .layout import FunctionCall, Circuit, GateSelection

ExecutionTranscript = dict[int, int]


class MissingValue(KeyError):
    pass


def evaluate_region(circuit: Circuit, sub: GateSelection, transcript: ExecutionTranscript, *, strict: bool = True) -> ExecutionTranscript:
    """Evaluate every gate of `sub` in canonical order, writing into `transcript` (mutated and returned).
    `Input` gates must be present already. With `strict=False`, gates whose operands are missing are skipped."""
    for gv in sub:
        prim = gv.prim
        if getattr(prim, "family", None) == "Input":
            if gv.index not in transcript:
                raise MissingValue(f"Input gate {gv.index} has no prescribed value in the transcript")
            continue
        lazy = getattr(prim, "evaluate_lazy", None)
        if lazy is not None:
            ops_seq = gv.operands
            try:
                transcript[gv.index] = lazy(lambda j, _s=ops_seq: transcript[_s[j]])
            except KeyError as e:
                if strict:
                    raise MissingValue(f"gate {gv.index} ({prim.id}) operand {e.args[0]} is not in the transcript") from None
            continue
        try:
            ops = [transcript[o] for o in gv.operands]
        except KeyError as e:
            if strict:
                raise MissingValue(f"gate {gv.index} ({prim.id}) operand {e.args[0]} is not in the transcript") from None
            continue
        transcript[gv.index] = prim.evaluate(*ops)
    return transcript


def evaluate_call(circuit: Circuit, call: FunctionCall, args: Iterable[int], transcript: ExecutionTranscript | None = None) -> tuple[int, ...]:
    """Bind `args` (encodings, in `call.args` order) then evaluate the call's region; return the encodings
    of `call.returns`. Values of `call.args` that lie inside the call (impossible for a well-formed call)
    or repeated references are bound consistently or rejected."""
    t: ExecutionTranscript = {} if transcript is None else transcript
    arg_gates = call.args
    args = list(args)
    if len(args) != len(arg_gates):
        raise ValueError(f"{call!r}: expected {len(arg_gates)} argument encodings, got {len(args)}")
    for g, v in zip(arg_gates, args):
        if g in t and t[g] != v:
            raise ValueError(f"{call!r}: gate {g} bound to two different values")
        t[g] = v
    evaluate_region(circuit, call, t)
    return tuple(t[g] for g in call.returns)
