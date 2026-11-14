"""A request's Program at its size class (P prompt ids, O output ids) and its node window (`node_window`), with every
decode step's units switched off past the request's stop by one bit the circuit carries from step to step, so one
window serves every request of the class and a request that stops early commits zeros past its stop (CIR-290, agreed on
PRF-618, Sat 14 Nov, 10:58 to 11:10 AM PST). This module states the interface; every body raises `NotImplementedError`
naming the issue until it is built.

**The class.** A span's class is (P, O) (`pipeline.boundary_rows.Span.shape`): P and O are the least powers of two, at
least 2, that hold its prompt ids and its served ids, and a stop class is (P, T) for the serve's stop rule's T whatever
n is (`pipeline.boundary_stop`). Today a request's Program is built at its own (LP, T), with its length rule's statics
its own (`steps.request_program`, `frontend.stop_rule.definition`: `StopCheck_v1{T = n, ..., MAX_TOKENS, PROMPT_LEN,
FINISH}`), so two requests of one class have two Programs. `class_request` builds the Program at (P, O) once for the
class: O selects, the registered stop ids and the model's length as statics (`stop_rule.CLASS_KEYS`), and the request's
length limits (``max_tokens``, the prompt's length) as values, as `stop_rule.StopClass_v1` takes them.

**The bit.** Each step s has one bit, live at s. Step s's select, a new version of `greedy_stages.GreedyCombine_v2`
(`live_select`), takes its step's bit and writes its id and the next step's bit, the AND of its own bit with the rest:
live after s exactly when live at s, the id is no stop id and the length limit is not reached. So once off, off: the
live steps are a prefix, and the length check's n is their count and nothing else. The first step's bit is 1, a
constant inside the first step's select (`first_select`), never a row at the window's boundary, which would be one the
developer fills. Every later bit is a committed output wired to the next step by an ordinary step-to-step wire, the
writer one step before its reader, as the decode's state is wired today. One unit writing every step's bit is no form:
the unit that decides the length reads every id, and each id comes from a select the bit switches, so it would be read
by units it reads, a cycle the wire check's order refuses (circuits, PRF-618, 11:09 AM PST).

**The limits** reach every step's select and the length check from one committed source, so the bit and the check
cannot be given two different limits: the class's `b.request` row, its first four fields (`boundary_rows.FIELDS`: the
prompt's length ``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), 8 sixteen-bit leaves from leaf 0, which each
of those units reads by one boundary read (`interface.Read`, ``step [0]`` across them: `LIMITS`).

**Masked units.** Every unit of a decode step is its Definition followed by a select on that bit (`masked`): live, the
Definition's outputs; not live, zeros. A step past n reads the step before by wire: if that step is off too it reads
zeros and writes zeros, and the first step off reads live rows, ignores them and writes zeros, so no row disagrees with
its writer. A masked Definition is a new version with its own id, so its hash and its Lean verdict are its own; a class
that always serves O ids keeps the unmasked ones. The prefill is not masked.

**The length check** (`class_check`) keeps one bit and reads the last step's bit with the ids. Past n the selects are
off and the ids they write are zeros, so the served-ids boundary read of the combines' ``o0``
(`node_window_reads.served_reads`) equals the `b.egress` row (`boundary_rows.egress_row` under a stop rule: the n ids,
then O - n zeros, low word first) at every n, and the check needs no second output.

**The wires** (`class_window`): each reader template of step s reads its bit from step s - 1's select, one wire pattern
a reader template of ``radix [steps, units]`` whose writer steps by one unit a step. Each rank holds its own select, so
the bit a rank's units read is its own rank's, and `node_window_join.join` keeps it among that rank's own wires.

**What is NOT decided here**, each named where it is met below:

- the form of the count a select compares with the limits, since one template serves every decode step and so holds no
  step's index as a static (`live_select` states one: the ids written so far, carried beside the bit);
- a prompt shorter than P: which row the prefill samples, the decode steps' positions and the keys past the prompt
  (`class_request`);
- class G's launch context, which its kernels read and which is a request's own, not its class's (`class_request`);
- whether a masked Definition is its family's next version or one registered combinator over any Definition
  (`masked`);
- what the length check's bit asserts for a request the rule does not end within O (finish 0), and for a span the rule
  does not read (an abort, an error, one still running): PRF-618's questions of 10:58 AM PST (`class_check`);
- what the serve commits past the stop (vllm's) and the sampling design's weight on the steps past it (proofs'),
  which CIR-290 leaves out.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from verity.commitments.windows import interface as IF

from .node_window import _GRAIN, NAME, Lowering, NodeWindow
from .window_builder import Grain

if TYPE_CHECKING:
    from verity.circuits.defs import SpecializedDefinition

    from .build import BuildResult
    from .steps import Registered

#: what every body here raises until it is built
TODO = "circuits/request-built-as-proof-blocks: CIR-290"
#: a masked Definition's added parameter, the last: its step's bit
LIVE = "live"
#: the parameter of the selects and the length check that reads the limits, and its leaves of `b.request`: the first
#: four fields (``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), two sixteen-bit leaves each, from leaf 0
REQUEST, LIMITS = "request", (0, 8)
#: the later selects' parameter holding the ids written before their step (`live_select`)
COUNT = "count"
#: the selects' output fields: the token's two words (``o0``, as `GreedyCombine_v2`'s, which the served-ids read
#: takes), the next step's bit, and the next step's ``count``
TOKEN, NEXT, AHEAD = "o0", "o1", "o2"
#: the class window's name, beside `node_window.NAME`
CLASS_NAME = f"{NAME}/class"


def class_request(registered: Registered, P: int, O: int, policy: Mapping[str, Any] | None = None,
                  named: bool = False) -> BuildResult:
    """The Program every request of class (``P``, ``O``) runs at the registration ``registered``, the counterpart of
    `steps.request_program` (whose `build.build_program(phi, "request", LP, T, ...)` it calls at LP = P and T = O - 1
    decode steps, in the registered compiler's form, `dead_free`, at φ's rank) with:

    - the prefill of P prompt rows, read from the class's `b.prompt` (P ids, zero past the prompt), then O - 1 decode
      steps, each fed the step before's id, returning O ids;
    - the prefill's select `first_select`, each decode step's Calls their `masked` versions reading that step's bit,
      and each decode step's select `live_select`'s, lowered by `class_window`;
    - the length check last (`class_check`), at T = O;
    - the limits a value, `b.request`'s first four fields (`LIMITS`), read by every select and by the check; the
      statics the class's alone: the registered model's stop ids (`frontend.stop_rule.stop_ids`) and its length
      (`stop_rule.CLASS_KEYS`).

    ``policy``: the sampling policy, greedy (`window_class.GREEDY`) and None alike; a stochastic one is refused, since
    the selects are the greedy select's. ``named``: its trace's names, as `steps.request_program` gives them.

    NOT decided: a prompt shorter than P. Its sample is at row lp - 1, its decode step s at position lp + s, and its
    decode keys past lp are the prompt's zero rows: each is a value's choice, which the Program at P must make in the
    circuit (a select on lp), or else the class holds only prompts of exactly P ids. Nor class G's launch context
    (`steps.RequestUnit.launch`), which its kernels read and which is the request's batch, not its class's: a batched
    target's class Program needs the launch fixed by the class or read as a value."""
    raise NotImplementedError(TODO)


def masked(fn: SpecializedDefinition) -> SpecializedDefinition:
    """``fn``'s masked version: ``fn``'s parameters, then one bit `LIVE`; its outputs ``fn``'s where ``live`` is 1 and
    zeros where it is 0, field for field and word for word. Its body is ``fn``'s followed by one AND of each output bit
    with ``live``, so it adds one gate an output bit and one input row (the bit) a unit. It is a new version with its
    own id, at ``fn``'s statics, so its hash and its Lean verdict are its own, and ``fn`` stays for a class that always
    serves O ids. A step's units read its bit by wire (`class_window`).

    NOT decided: whether each family's masked Definition is that family's next version in the catalog
    (`verity/definitions/registry`, with its circuit-check sidecar), or one combinator registered once over any
    Definition, whose property check covers every family; nor how a template whose rows are near
    `interface.MAX_ROWS` takes the bit's row (CIR-290's plan counts them)."""
    raise NotImplementedError(TODO)


def first_select(V: int, N: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """The prefill's select, the first step's: `live_select` at a step whose bit is 1, a constant its Definition holds,
    and whose ids written before are 0, so it has neither a ``live`` nor a ``count`` parameter:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>)
            -> (o0: Array<2, Value<16>>, o1: bit, o2: i32)

    ``o0`` the token, ``o1`` the next step's bit (the token is none of the stop ids and the limit is not reached at one
    id), ``o2`` = 1. A template of its own, one unit a request, beside `live_select`'s template of the O - 1 later
    steps. Its chunks are `greedy_stages.GreedyChunk_v2` unmasked, the prefill's."""
    raise NotImplementedError(TODO)


def live_select(V: int, N: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyCombine_v3{V, N, STOP, MAX_MODEL_LEN}`, a decode step's select: `greedy_stages.GreedyCombine_v2{V, N}`'s
    token over its logits and its chunks' results, taking its step's bit and writing the next step's:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>, count: i32, live: bit)
            -> (o0: Array<2, Value<16>>, o1: bit, o2: i32)

    ``request`` is `b.request`'s first four fields (`LIMITS`), the ones the length check reads; ``count`` the ids
    written before this step (the step before's ``o2``); ``live`` this step's bit (the step before's ``o1``).

    - ``o0``: the token's low and high words where ``live`` is 1, zeros where it is 0, so a step past the stop writes
      the id 0, which is `b.egress`'s word there.
    - ``o1``: ``live`` AND the token is none of STOP (none where ``ignore_eos`` is not 0, as
      `boundary_stop.egress_stop` reads it) AND the length limit is not reached at count + 1 ids as
      `stop_rule.StopRule_v1` reads it (below ``max_tokens`` and below MAX_MODEL_LEN - ``lp``, at the rule's edges as
      it reads them). Once off, off: a step off writes 0 whatever it reads.
    - ``o2``: count + 1.

    Its chunks are `masked` `GreedyChunk_v2`s on the same bit. Over logits that are all zero (a step whose LM head is
    off) the token is 0 (the lowest index of the largest logit), so the check's recomputed token agrees on a step past
    the stop too.

    NOT decided: ``count``, which is this interface's and not CIR-290's. One template serves every decode step, so no
    static holds the step's index, and the select can compare the limits only with a value the step before hands it;
    the other form is one template a step, the index a static, which multiplies the select's templates by O."""
    raise NotImplementedError(TODO)


def class_check(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyStopCheck_v2{V, N, T, STOP, MAX_MODEL_LEN}` at T = ``O``, the next version of
    `greedy_stop.GreedyStopCheck_v1`, one bit:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, ids: Array<T, i32>, request: Array<4, i32>, live: bit)
            -> bit

    ``request`` is `b.request`'s first four fields (`LIMITS`), the ones every select reads; ``live`` the last step's
    next bit (`live_select`'s ``o1``). It is 1 exactly when `frontend.stop_rule.StopClass_v1{T, STOP, MAX_MODEL_LEN}`
    over ``ids`` and the limits (no stop id where ``ignore_eos`` is not 0) gives (n, finish) with finish not 0, the ids
    from n on are 0, ``live`` is 0, and ``ids[T - 1]`` is `GreedyCombine_v2{V, N}`'s token over the logits and results,
    as v1 holds its last id. The ids come a row each from the steps' ``o0``; the logits row is what makes the unit a
    proof block (`interface.check_proof_block`), as in v1. It calls `StopClass_v1` (written, unregistered: its
    registration comes with this check), `GreedyCombine_v2` and the scalar ops v1 calls.

    Its ids a row each, it has T + K + 2 rows and more, so near T = 240 it passes `interface.MAX_ROWS` (256): a larger
    class reads its ids gathered first (`node_window.GATHER_ROWS`).

    NOT decided (PRF-618, 10:58 AM PST, proofs' to say): what the bit asserts for a request the rule does not end
    within O (finish 0, ``live`` 1 after the last step), for a ``min_tokens`` other than 0, which the rule doesn't read
    (`boundary_stop` files such a span with no stop class), and for a span the rule does not read (an abort, an error,
    one still running), which has no rule n."""
    raise NotImplementedError(TODO)


def class_window(fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN,
                 name: str = CLASS_NAME, stand_ins: bool = False) -> tuple[IF.Circuit, NodeWindow]:
    """The node window of a class's Program ``fn`` (`class_request`), at one rank: `node_window.node_window` with

    - O selects: the prefill's `first_select` over unmasked chunks, then each decode step's `live_select` over its
      `masked` chunks; and the check `class_check` (`node_window_select.token_select` and `stop_check`, in their class
      forms);
    - every decode step's units `masked` by its step's bit, the prefill's as they are;
    - each reader template's ``live`` port wired from the step before's select's ``o1``, by one wire pattern a reader
      template: ``radix [steps, units]``, ``step [1, 0]``, its base the select one step before the template's first,
      so the writer steps by one unit a step and every unit of a step reads one row; each ``live_select``'s ``count``
      from the step before's ``o2`` the same way; the check's ``live`` from the last select's ``o1``;
    - each select's and the check's ``request`` a boundary read of `b.request`'s `LIMITS` leaves, one row read by all.

    Each rank's window reads its own rank's bit, so `node_window_join.join` joins the class windows as it joins request
    windows, the bit's wires among each rank's own. The served-ids read (`node_window_reads.served_reads`) takes the
    selects' ``o0`` at all O steps, 2·O sixteen-bit leaves of `b.egress`, each written once (`written_once`).

    ``families``, ``grain``, ``name`` and ``stand_ins`` are `node_window.node_window`'s. NOT decided: whether the
    masking is the Program's (its Calls `masked`, as `class_request` states) and lowered as any Call, or added here at
    lowering with the Program's Calls unmasked."""
    raise NotImplementedError(TODO)


__all__ = ["AHEAD", "CLASS_NAME", "COUNT", "LIMITS", "LIVE", "NEXT", "REQUEST", "TODO", "TOKEN", "class_check",
           "class_request", "class_window", "first_select", "live_select", "masked"]
