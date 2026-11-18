"""A class's selects and its length check (`class_window`'s module docstring has the design): each a registered Definition
bound from the class's statics alone, `first_select` the prefill's (`greedy_stages.GreedyCombineFirst_v1`), `live_select`
each decode step's (`greedy_stages.GreedyCombineLive_v1`) and `class_check` the length check
(`greedy_stop.GreedyStopCheckClass_v1`).
`class_window`'s stubs of the same names delegate here. What holds, each by its Definition's body:

- **The first bit and the first count are constants** inside the first select (bit 1, count 0, so it writes the count 1),
  never parameters, so neither is a row at the window's boundary that the developer fills: `first_select` has no ``live``
  and no ``count`` parameter.
- **Once off, off:** the next bit is ``live`` AND the rest, so a step whose bit is 0 writes 0, whatever else it reads.
- **A step off writes the count it read:** ``o2`` is count + 1 only where ``live`` is 1 (and the count is below O), else
  ``count``, so the steps past the stop are alike and the last count is the number of ids served.
- **The count never passes O:** a live step reading a count of O writes the bit 0 and holds O (no window of O selects
  reaches it, step s reading at most s), and the next bit is 0 where count + 1 reaches O, so the count, log2(O) + 1 bits,
  never wraps.
- **Two ids, two ANDs:** ``TOKEN`` (``o0``), the served id, is the token ANDed with the step's own bit, so the stopping
  step's id is written and every id past it is 0; ``KEY`` (``o3``), the next step's embedding key, is the token ANDed with
  the NEXT bit, so the step after the stop reads token 0. The ANDs are in each select's Definition, its text and its hash,
  so a select without them is another Definition and cannot be dropped in.
- **The limits are one row:** ``request``, `b.request`'s first four u32 fields (``lp``, ``max_tokens``, ``min_tokens``,
  ``ignore_eos``, `greedy_stages.LIMIT_FIELDS`), is a parameter of every select and of the check, read by one boundary read
  (`class_window.LIMITS`), so the chain and the check cannot be given two different limits; the stop ids and the model's
  length are the class's statics.
- **The check's n is the last count:** `class_check` is 1 only where the rule's n equals the last select's ``o2``, its
  ``o1`` is 0, and every id past n is 0.
- **One word a wire:** a select writes the bit (``o1``) as one bit and the count (``o2``) as log2(O) + 1, each one u16 word
  of its row, zero-padded; its reader takes each as one u16 word (`greedy_stages.WORD_BITS`, since a unit's parameter is
  whole words) and reads the bit's bit 0 and the count's low log2(O) + 1 bits, no other.

A select's rule is `stop_rule.StopRule_v1`'s, step by step: at L = count + 1 ids it ends where ``max_tokens`` is at least 1
and the token is a stop id (none where ``ignore_eos`` is not 0) or L reaches the length limit (``max_tokens`` or the model's
length less ``lp``, at least 1: `stop_rule.length_limit`). The check runs the rule over the served ids, so a chain that
stopped where the rule does not is refused, never accepted. Refused by the check's bit until proofs rule on them: a request
the rule does not end within O, a ``min_tokens`` other than 0, and a prompt of other than P ids.
"""
from __future__ import annotations

from collections.abc import Sequence

from verity.circuits.defs import SpecializedDefinition, bind

from verity_vllm.definitions.registry import greedy_stages as G
from verity_vllm.definitions.registry import greedy_stop as GS


def _stops(stops: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in stops)


def first_select(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyCombineFirst_v1{V, N, O, STOP, MAX_MODEL_LEN}`, the prefill's select, at a row of V logits in chunks of N,
    the class's O served ids, its stop ids ``stops`` (increasing) and the model's length."""
    return bind(G.GreedyCombineFirst, V=V, N=N, O=O, STOP=_stops(stops), MAX_MODEL_LEN=max_model_len)


def live_select(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyCombineLive_v1{V, N, O, STOP, MAX_MODEL_LEN}`, each decode step's select, at `first_select`'s statics."""
    return bind(G.GreedyCombineLive, V=V, N=N, O=O, STOP=_stops(stops), MAX_MODEL_LEN=max_model_len)


def class_check(V: int, N: int, P: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyStopCheckClass_v1{V, N, P, T, STOP, MAX_MODEL_LEN}` at T = ``O``, the class's length check, at the selects'
    statics and the class's P prompt ids."""
    return bind(GS.GreedyStopCheckClass, V=V, N=N, P=P, T=O, STOP=_stops(stops), MAX_MODEL_LEN=max_model_len)


__all__ = ["class_check", "first_select", "live_select"]
