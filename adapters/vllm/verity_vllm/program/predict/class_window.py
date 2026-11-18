"""A request's Program at its size class (P prompt ids, O output ids) and its node window (`node_window`), with the decode
steps past the request's stop switched off by one bit the circuit carries from step to step, so one window serves every
request of the class. Two templates are masked by the bit, the sparse attention's key blocks and the select; every other
unit keeps its Definition. Past the stop the served ids are zeros and every other row is its Definition's on its wired
inputs.

**The class.** A request's class is (P, O): P its prompt's exact length, any integer of at least 1, and O a power of
two, at least 2, that holds its served ids. A prompt shorter than P is not padded: it has no class at that P, only at
its own length. A request's own Program is built at its own (LP, T), with its length rule's statics
its own (`steps.request_program`, `frontend.stop_rule.definition`: `StopCheck_v1{T = n, ..., MAX_TOKENS, PROMPT_LEN,
FINISH}`), so two requests of one class have two Programs. `class_request` builds the Program at (P, O) once for the
class: O selects, the registered stop ids and the model's length as statics (`stop_rule.CLASS_KEYS`), and the request's
length limits as values, the `b.request` row's.

**The bit.** Each step s has one bit, live at s. Step s's select (`live_select`) takes its step's bit and writes its id
and the next step's bit, the AND of its own bit with the rest: live after s exactly when live at s, the id is no stop id
and the length limit is not reached. So once off, off: the live steps are a prefix, and the length check's n is their
count and nothing else. The first step's bit is 1, a constant inside the first step's select (`first_select`), never a
row at the window's boundary, which would be one the developer fills. Every later bit is a committed output wired to the
next step by an ordinary step-to-step wire, the writer one step before its reader, as the decode's state is wired. One
unit writing every step's bit is no form: the unit that decides the length reads every id, and each id comes from a
select the bit switches, so it would be read by units it reads, a cycle the wire check's order refuses.

**The count** runs beside the bit by the same rule: the ids written so far, committed, wired step to step by an ordinary
wire, the first a constant inside `first_select` (1), never a boundary row. One template serves every decode step, so
no static holds a step's index, and a select compares the limits only with the count the step before hands it. It holds
0 to O, with O a static of each select and of the check, and the select refuses to count past O: it never wraps. Past
the stop it holds still: a step off writes the count it read, so the steps past the stop are alike. The length check's
n is tied to the last step's count: `class_check` reads that count and holds the rule's n equal to it, so the count the
selects carried and the n the rule gives cannot be two numbers.

**The ports of the bit and the count** are one u16 word each, `WORD`: 16 bits, least significant first, the bit in bit
0 and the count in its low log2(O) + 1 bits, the rest 0. A unit's parameter is whole u16 words (`interface.layout`), so
a narrower one is no row; the outputs that feed them have the same type, so the class Program's Calls agree in type.

**The limits** reach every step's select and the length check from one committed source, so the bit and the check
cannot be given two different limits: the class's `b.request` row, its first four fields (`boundary_span.FIELDS`: the
prompt's length ``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), 8 sixteen-bit leaves from leaf 0, which each
of those units reads by one boundary read (`interface.Read`, ``step [0]`` across them: `LIMITS`, `request_reads`).

**The select is where the mask lives.** Each select writes two ids (`live_select`): the served id `TOKEN` (``o0``), its
token ANDed with its own bit, which the served-ids read takes; and the next step's key `KEY` (``o3``), its token ANDed
with the NEXT bit, which the next step's embedding read takes as its ``tok``. At the step that stops they differ: its
id is the n-th served id and is written, while the step after it, the first one off, must read token 0. So the
embedding read stays `EmbeddingRead_v1`, unmasked, and opens its table's row 0 at every step past the stop, and no unit
downstream of it reads the request's last token. The AND is the select Definition's own, in its text and its hash: a
select without it cannot be dropped in, since a step past the stop would then read the request's last id.

**Masked units: the key blocks.** The one other decode template that reads the request is the sparse attention's key
block (`SparseMlaHeadBlock_v1`, every NVIS and FIRST): its ``kv`` reads the cache, the latent rows of the prefill and of
the steps before, which hold the prompt and the request's ids. Its masked version (`masked`) reads the step's bit and
writes zeros past the stop. It writes no cache row: its outputs ``o0``, ``o1``, ``o2`` (the running max, the lanes and
O) are its state, read by the head's next block or its end; the cache rows are the latent row's (`DsaLatentRow_v1`'s
``o0``, `serving_templates.HISTORY`). Past the stop a step's latent row is its Definition's on the step's inputs, which
past the stop are the class's, so a dead step's cache rows do not differ by request; and nothing that is not masked
reads them, since only later steps read a step's cache rows and every later step is off. Every other decode unit keeps
its Definition: past the stop its rows are what it gives on its wired inputs. The prefill is not masked.

The key blocks are masked in the window (`class_window`), not in the Program: the Program's `SparseMlaSplit_v1` Call
is one node the window cuts into its chain's blocks (`node_window_sparse.sparse_mla_split`), and no Definition states
the split in its chain form, which a masked Call's body would need. The Program's outputs are the same either way: past
the stop each select writes 0 for both ids and holds the bit and the count whatever its logits, and the check reads the
last step's logits only where that step is live.

**Why the form is sound whatever the list says:** a unit past the stop that is not masked is proved like any other, its
committed rows its Definition's outputs on its committed inputs, read by wire from their writers. Soundness needs only
the bit and the count right and the served ids past n zeros, which the selects and the length check hold. The list of
what reaches a decode step (the key blocks' ``kv``, the embedding read's ``tok``, the check's ``ids``) is the serve's:
it lets the serve write a step past the stop as the class's constant rows without running the model. A hole in it costs
the serve work, a unit's rows computed for the request; it cannot make the auditor accept anything false.

**The length check** (`class_check`) keeps one bit and reads the last step's bit and count with the ids. The ids are the
selects' ``o0``, zeros past n, so the served-ids boundary read of the selects' ``o0``
(`node_window_reads.served_reads`) equals the `b.egress` row (`boundary_span.egress_row` under a stop rule: the n ids,
then O - n zeros, low word first) at every n, and the check needs no second output.

**In the Program** each select is one Call of `ClassSelect_v1` and the check one of `ClassStopCheck_v1`, rows over the
whole gathered logits whose bodies are their units' Definitions: the `greedy_stages.GreedyChunk_v2` chunks, then the
select or the check (`first_select`, `live_select`, `class_check`). A select row's ids are i32, as `TokenSelect_v1`'s:
its body takes the token from `TokenSelect_v1`, the function of the chunks and the combine (`node_window_select`), and
ANDs it with the bits its select gives, word for word the select's ``o0`` and ``o3``. Its ``o1`` and ``o2`` are the
select's.

**Refused until proofs rules on them:** a request the rule does not end within O, a ``min_tokens`` other than 0 and a
span with no stop. The check refuses the first two by its bit, 0 where the rule's finish is 0 and where ``min_tokens``
is not 0, and a prompt of other than P ids the same way, 0 where ``lp`` is not P (`class_check`); a span with no stop
has no stop class (`pipeline.boundary_stop`), so no class Program is built for it.

**What is NOT decided here**, each named where it is met below:

- the three cases above, proofs';
- an indexer past 2,048 keys (GLM-5.3's ``index_topk``): the window has no lowering for one, so a class with more
  keys than that has no window (`class_window`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.prims import const
from verity.circuits.ops.scalar import SelectI32
from verity.circuits.refs import Coll, Refs, concat, runs
from verity.circuits.types import Array, Tuple
from verity.commitments.periods import interface as IF

from verity_vllm.definitions.registry import b1
from verity_vllm.definitions.registry import greedy_stages as G
from verity_vllm.definitions.registry.prims import BF16, BIT, I32

from . import node_window_join as NJ
from . import node_window_reads as NR
from . import node_window_select as NSEL
from . import node_window_sparse as NS
from .build import Unexpressible
from .node_window import (
    _GRAIN,
    NAME,
    PROMPT,
    STAND_IN,
    Lowering,
    NodeWindow,
    Piece,
    _family,
    _Free,
)
from .window_builder import WEIGHTS, Grain, Row

if TYPE_CHECKING:
    from verity.circuits.defs import SpecializedDefinition

    from .build import BuildResult
    from .steps import Registered

#: the masked key block's added parameter, the last: its step's bit
LIVE = "live"
#: the parameter of the selects and the length check that reads the limits, the class Program's Input of them, and its
#: leaves of `b.request`: the first four fields (``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), two sixteen-bit
#: leaves each, from leaf 0
REQUEST, LIMITS = "request", (0, 8)
#: the boundary row the limits are read from
LIMITS_ROW = "b.request"
#: the later selects' parameter holding the ids written before their step (`live_select`)
COUNT = "count"
#: the selects' output fields: the served id's two words (``o0``, as `GreedyCombine_v2`'s, which the served-ids read
#: takes), the next step's bit, the next step's ``count``, and the id the next step's embedding read takes
TOKEN, NEXT, AHEAD, KEY = "o0", "o1", "o2", "o3"
#: the bit's and the count's port: one u16 word of bits, least significant first
WORD = Array(16, BIT)
#: the key block's family, the one decode template besides the select that `masked` takes, and its masked version's
KEY_BLOCK, MASKED_BLOCK = "SparseMlaHeadBlock_v1", "SparseMlaHeadBlockLive_v1"
#: the class window's name, beside `node_window.NAME`
CLASS_NAME = f"{NAME}/class"
#: the class Program's select and check rows' families (`ClassSelect`, `ClassStopCheck`)
SELECT, CHECK = "ClassSelect", "ClassStopCheck"
#: GLM-5.3's ``index_topk``: past it a key block reads the indexer's selection, which the window doesn't lower
INDEX_TOPK = 2048


# ---- the Definitions, each its module's

def masked(fn: SpecializedDefinition) -> SpecializedDefinition:
    """The masked version of a decode unit ``fn`` that reads the cache, `KEY_BLOCK` at any of its statics (every NVIS and
    FIRST) or, in the column cut, `SparseMlaHeadKeysWire_v1` (`class_masked.MASKED`). For a key block: `MASKED_BLOCK` at
    ``fn``'s statics, a family of its own, with ``fn``'s parameters, then `LIVE` (`WORD`), and its outputs ``fn``'s
    (``o0`` the running max, ``o1`` the lanes, ``o2`` O, 1,034 words at DV = 512) where the bit is 1 and zeros where it
    is 0, word for word: ``fn``'s body followed by one AND of each output bit with the bit. It has its own id, so its hash
    and its Lean verdict are its own, and `KEY_BLOCK` stays for the prefill and for a class that always serves O ids. A
    step's blocks read its bit by wire (`class_window`). It writes no cache row (module docstring, "Masked units").
    `class_masked`'s."""
    from .class_masked import masked as built
    return built(fn)


def first_select(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """The prefill's select, the first step's: `live_select` at a step whose bit is 1, a constant its Definition holds,
    and whose ids written before are 0, so it has neither a ``live`` nor a ``count`` parameter:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>)
            -> (o0: Array<2, Value<16>>, o1: bit, o2: Array<k, bit>, o3: Array<2, Value<16>>)

    with k = log2(O) + 1 the count's bits, least significant first, each output one u16 word of its row. ``o0`` the token, the first served id, always written; ``o1`` the next step's bit (the token is none of the stop ids
    and the limit is not reached at one id); ``o2`` = 1, the count's first value; ``o3`` the token ANDed with ``o1``, the
    first decode step's key. A template of its own, one unit a request. `class_select`'s."""
    from .class_select import first_select as built
    return built(V, N, O, stops, max_model_len)


def live_select(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyCombineLive_v1{V, N, O, STOP, MAX_MODEL_LEN}`, a decode step's select: `greedy_stages.GreedyCombine_v2{V, N}`'s
    token over its logits and its chunks' results, taking its step's bit and count and writing the next step's, and the
    two ids (module docstring, "The select is where the mask lives"):

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>, count: WORD, live: WORD)
            -> (o0: Array<2, Value<16>>, o1: bit, o2: Array<k, bit>, o3: Array<2, Value<16>>)

    reading the count's low k bits and the bit's bit 0, and no other bit of either word.

    - ``o1``: ``live`` AND the token is none of STOP (none where ``ignore_eos`` is not 0) AND the length limit is not
      reached at count + 1 ids as `stop_rule.StopRule_v1` reads it, AND ``count`` + 1 is below O.
    - ``o0``, the served id: the token ANDed with ``live``.
    - ``o3``, the next step's key: the token ANDed with ``o1``.
    - ``o2``: ``count`` + 1 where ``live`` is 1, ``count`` where it is 0; never past O. `class_select`'s."""
    from .class_select import live_select as built
    return built(V, N, O, stops, max_model_len)


def class_check(V: int, N: int, P: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyStopCheckClass_v1{V, N, P, T, STOP, MAX_MODEL_LEN}` at T = ``O``, one bit:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, ids: Array<T, i32>, request: Array<4, i32>, count: WORD,
         live: WORD) -> bit

    1 exactly when the length rule over ``ids`` and the limits gives (n, finish) with finish not 0 and n equal to
    ``count``, ``min_tokens`` is 0, ``lp`` is P, the ids from n on are 0, ``live`` is 0, and ``ids[T - 1]`` is
    `GreedyCombine_v2{V, N}`'s token over the logits and results where n is T. Its ids a row each, it has T + k + K + 3
    rows (k its logits port's, 64 at GLM-5.3's select at TP8), so at GLM-5.3 it passes `interface.MAX_ROWS` past
    T = 169. `class_select`'s."""
    from .class_select import class_check as built
    return built(V, N, P, O, stops, max_model_len)


def _bits(B: Any, value: int, n: int) -> Coll:
    return Coll(Array(n, BIT), concat([B.call(const(1, (value >> i) & 1)).refs for i in range(n)]))


def _widened(B: Any, f: Coll) -> Coll:
    """An output of fewer than 16 bits as the `WORD` its row holds: its bits, then zeros."""
    n = f.type.leaves
    return f if n == 16 else Coll(WORD, concat([f.refs, _bits(B, 0, 16 - n).refs]))


def _statics(stops: Sequence[int], max_model_len: int) -> dict:
    return {"STOP": tuple(int(s) for s in stops), "MAX_MODEL_LEN": int(max_model_len)}


def _select_def(first: bool, V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> Any:
    return (first_select if first else live_select)(V, N, O, tuple(stops), max_model_len)


def _check_def(V: int, N: int, P: int, O: int, stops: Sequence[int], max_model_len: int) -> Any:
    return class_check(V, N, P, O, tuple(stops), max_model_len)


# ---- the class Program's rows (module docstring, "In the Program")

def _chunks(B: Any, V: int, logits: Coll) -> tuple[int, Coll, list[Coll]]:
    """The row's chunks but the last as `greedy_stages.plan(V)` calls them, and the last chunk's lanes: (N, lanes, results)."""
    plan = G.plan(V)
    results = [B.call(u.defn, logits[u.lanes[0]:u.lanes[1]]) for u in plan[:-1]]
    lo, hi = plan[-1].lanes
    return hi - lo, logits[lo:hi], results


def _select_sig(S: Any) -> tuple:
    return ((("logits", Array(S.V, BF16)), (REQUEST, Array(4, I32)), *(() if S.FIRST else ((COUNT, WORD), (LIVE, WORD)))),
            Tuple(I32, WORD, WORD, I32))


def _select_row(B: Any, S: Any, logits: Coll, request: Coll, *state: Coll) -> Coll:
    N, lanes, results = _chunks(B, S.V, logits)
    st = B.call(_select_def(S.FIRST, S.V, N, S.O, S.STOP, S.MAX_MODEL_LEN), lanes, *results, request, *state)
    token = B.call(bind(b1.TokenSelect, V=S.V), logits)
    zero = B.call(const(32, 0))
    served = token if S.FIRST else B.call(SelectI32, state[1][0], token, zero)
    nxt, count = _widened(B, st[1]), _widened(B, st[2])
    key = B.call(SelectI32, nxt[0], token, zero)
    return Coll(Tuple(I32, WORD, WORD, I32), concat([served.refs, nxt.refs, count.refs, key.refs]))


#: `ClassSelect_v1{V, O, STOP, MAX_MODEL_LEN, FIRST}` (logits: bf16[V], request: Array<4, i32>[, count: WORD,
#: live: WORD]) -> (i32, WORD, WORD, i32): a step's select over its whole row, the served id, the next bit and count, and the
#: next step's key (module docstring, "In the Program")
ClassSelect = CompositeDefinition(
    SELECT, 1, ("V", "O", "STOP", "MAX_MODEL_LEN", "FIRST"), _select_sig, _select_row, register=False,
    doc="a class Program's select: the chunks, then first_select or live_select, its ids TokenSelect_v1's token ANDed with "
        "its bits")


def _check_row(B: Any, S: Any, logits: Coll, ids: Coll, request: Coll, count: Coll, live: Coll) -> Coll:
    N, lanes, results = _chunks(B, S.V, logits)
    return B.call(_check_def(S.V, N, S.P, S.O, S.STOP, S.MAX_MODEL_LEN), lanes, *results, ids, request, count, live)


#: `ClassStopCheck_v1{V, P, O, STOP, MAX_MODEL_LEN}` (logits: bf16[V], ids: Array<O, i32>, request: Array<4, i32>,
#: count: WORD, live: WORD) -> bit: the class's length check over the last step's whole row (`class_check`)
ClassStopCheck = CompositeDefinition(
    CHECK, 1, ("V", "P", "O", "STOP", "MAX_MODEL_LEN"),
    lambda S: ((("logits", Array(S.V, BF16)), ("ids", Array(S.O, I32)), (REQUEST, Array(4, I32)), (COUNT, WORD), (LIVE, WORD)),
               BIT),
    _check_row, register=False, doc="a class Program's length check: the chunks, then class_check")


@dataclass
class Selects:
    """A class's selects and check as `build.build_program` takes them (its ``klass``): the ``request`` Input, each
    select's Call (`select`, the first `first_select`'s) with the step's bit and count from the select before, the key
    each decode is fed (``key``), and the check over the outputs last (`check`)."""

    P: int
    O: int
    stops: tuple[int, ...]
    max_model_len: int
    request: Coll | None = None
    state: tuple[Coll, Coll] | None = None
    key: Coll | None = None
    last: tuple[int, Refs] | None = None
    n: int = 0

    def params(self) -> list[tuple[str, Any]]:
        return [(REQUEST, Array(4, I32))]

    def config(self) -> dict:
        return {"P": self.P, "O": self.O, "stops": list(self.stops), "max_model_len": self.max_model_len}

    def bind(self, inputs: Sequence[Coll]) -> Selects:
        (self.request,) = inputs
        return self

    def _statics(self, V: int) -> dict:
        return {"V": V, "O": self.O, **_statics(self.stops, self.max_model_len)}

    def select(self, call: Any, k: Any, logits: Refs, V: int) -> Coll:
        if self.request is None or self.n >= self.O:
            raise Unexpressible(f"the class (P {self.P}, O {self.O}): select {self.n + 1}, with "
                                f"{'no request Input' if self.request is None else f'{self.O} outputs'}")
        first = self.n == 0
        fn = bind(ClassSelect, FIRST=first, **self._statics(V))
        out = call(fn, logits, self.request.refs, *(() if first else (s.refs for s in self.state or ())))
        self.n, self.last = self.n + 1, (V, logits)
        self.state, self.key = (out[2], out[1]), out[3]
        return out[0]

    def check(self, B: Any, ids: Coll) -> tuple[Coll, str]:
        if self.last is None or self.state is None or self.n != self.O:
            raise Unexpressible(f"the class (P {self.P}, O {self.O}): its check after {self.n} selects")
        V, logits = self.last
        fn = bind(ClassStopCheck, P=self.P, **self._statics(V))
        count, live = self.state
        what = (f"the class's length check ({CHECK}_v1 at P {self.P}, O {self.O}): its bit and count, the limits of "
                f"{LIMITS_ROW} and the ids zero past the stop")
        return B.call(fn, Coll(Array(V, BF16), logits), ids, self.request, count, live), what


def _power(x: Any) -> bool:
    return type(x) is int and x >= 2 and x & (x - 1) == 0


def class_request(registered: Registered, P: int, O: int, policy: Mapping[str, Any] | None = None,
                  named: bool = False) -> BuildResult:
    """The Program every request of class (``P``, ``O``) runs at the registration ``registered``, the counterpart of
    `steps.request_program` (whose `build.build_program(phi, "request", LP, T, ...)` it calls at LP = P and T = O - 1
    decode steps, in the registered compiler's form, `dead_free`, at φ's rank) with:

    - the prefill of P prompt rows, then O - 1 decode steps, each fed the step before's `KEY`, returning the O served
      ids `TOKEN`;
    - each step's select a `ClassSelect_v1` Call (`first_select`, then `live_select`), lowered by `class_window`;
    - the length check last, a `ClassStopCheck_v1` Call (`class_check`, at T = O);
    - the limits a value, the ``request`` Input (`b.request`'s first four fields, `LIMITS`), read by every select and by
      the check; the statics the class's alone: the registered model's stop ids (`frontend.stop_rule.stop_ids`) and its
      length (`stop_rule.CLASS_KEYS`), the least of its positions and the registered ``max_model_len``;
    - in class G, each step its own launch of the request alone (P rows, then one row a decode step), the launch a
      request served alone runs; a class whose requests ran in other launches has no Program here (NOT decided).

    ``policy``: the sampling policy, greedy (`window_class.GREEDY`) and None alike; a stochastic one is refused, since
    the selects are the greedy select's. ``named`` is refused: no trace names a class Program.

    ``P`` is the class's exact prompt length, so every row the Program reads is a prompt id: no position, sample row or key
    depends on a value. Refused: a span with no stop class, for which no class Program is built; and, by the check's
    bit, a prompt of other than P ids.

    NOT decided: a launch context other than the request alone: a batched target's class Program needs the launch fixed by the
    class or read as a value."""
    from verity.circuits.codec import program_digest

    from verity_vllm.program.frontend import stop_rule as SR

    from .build import build_program
    from .phi import Phi
    if not (type(P) is int and P >= 1 and _power(O)):
        raise ValueError(f"a class is (P, O), P a prompt length of at least 1 and O a power of two of at least 2, not "
                         f"({P!r}, {O!r})")
    if dict(policy or {}).get("kind", "greedy") != "greedy":
        raise Unexpressible(f"a class Program's selects are the greedy select's, and the policy is {policy!r}")
    if named:
        raise Unexpressible("a class Program has no trace, so nothing names it")
    phi = Phi.from_json({**registered.phi.to_json(), "sampling": {"kind": "greedy"}})
    hf, eng = phi.hf_config, phi.engine or {}
    mml = min(int(hf["max_position_embeddings"]), int(eng.get("max_model_len") or hf["max_position_embeddings"]))
    if mml < P + O:
        raise Unexpressible(f"the class (P {P}, O {O}): positions to {P + O - 1}, past the model's length {mml}")
    klass = Selects(P, O, SR.stop_ids(hf), mml)
    batched = not phi.target_profile().batch_invariant
    batch = [[[P, 0]], *([[1, P + s]] for s in range(O - 1))] if batched else None
    r = build_program(phi, "request", LP=P, T=O - 1, dead_free=True, batch=batch, with_desc=False, batched=batched,
                      table_reads=registered.dsa_table_reads, klass=klass)
    return replace(r, digest=program_digest(r.program))


# ---- the window

class ClassWindow(NodeWindow):
    """`NodeWindow` over a class Program (`class_request`): its ``request`` Input beside the prompt, its selects' rows ending
    its steps, and each decode step's key blocks masked by the bit of the select before (`masked`)."""

    INPUTS = (PROMPT, REQUEST, WEIGHTS)
    ENDS = frozenset({SELECT})

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        #: the bit the next decode step reads: the last lowered select's ``o1`` row
        self.live: Row | None = None
        self._masking: Row | None = None
        #: each masked allocation: (template, first unit, units)
        self.masked: list[tuple[str, int, int]] = []

    def unit(self, family: str, fn: Any, count: int) -> tuple[str, int]:
        from .class_masked import MASKED

        if self._masking is None or descriptor_id(fn).split("{")[0] not in MASKED:
            return super().unit(family, fn, count)
        tid, at = super().unit(family, masked(fn), count)
        self.b.feed(tid, LIVE, at, count, (self._masking,), ((count, 0),))
        self.masked.append((tid, at, count))
        return tid, at


def _cut_logits(w: NodeWindow, node: Any, plan: Sequence[tuple[G.Unit, Any]]) -> tuple[int, int, list, dict]:
    """The node's logits (argument 0) as `node_window_select.token_select` cuts them for ``plan``'s units: (N, R, each unit's
    grids by key, ...)."""
    M = w.members(node)
    N = plan[0][0].lanes[1] - plan[0][0].lanes[0]
    logits = [NSEL.through(w, w.arg(node, 0, t)) for t in range(M)]
    R = NSEL.width(w, logits[0], N)
    if N // R + 1 + len(plan) + 3 > IF.MAX_ROWS:
        raise Unexpressible(f"{descriptor_id(node.fn)}: chunks of {N} lanes in rows of {R} are {N // R} rows a port, past a proof "
                            f"block's {IF.MAX_ROWS}")
    grids = {}
    for u, _fn in plan:
        lo, rows = u.lanes[0], []
        for i in range(N // R):
            got = [NSEL._row(w, ls.slice(lo + i * R, lo + (i + 1) * R)) for ls in logits]
            free = next((g for g in got if isinstance(g, _Free)), None)
            rows.append(free if free is not None else [g for g in got if not isinstance(g, _Free)])
        grids[u.key] = rows
    return N, R, logits, grids


def lower_select(w: NodeWindow, j: int, node: Any) -> None:
    """A `ClassSelect_v1` row as its units: the K - 1 `GreedyChunk_v2` chunks as `node_window_select.token_select` lowers
    them, then the select (`first_select` or `live_select`) in the combine's place, its ``request`` the class Program's
    Input (a boundary read, `request_reads`), its ``count`` and ``live`` the select before's ``o2`` and ``o1`` by wire;
    the node's four fields the select's four outputs."""
    S, fam, M = node.fn.bindings, _family(node), w.members(node)
    if M != 1:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): {M} members; a class Program's select is one request's")
    V, O, first = int(S["V"]), int(S["O"]), bool(S["FIRST"])
    plan = list(NSEL.stages(V))
    last = plan[-1][0]
    N = last.lanes[1] - last.lanes[0]
    plan[-1] = (last, _select_def(first, V, N, O, S["STOP"], S["MAX_MODEL_LEN"]))
    N, _R, _logits, grids = _cut_logits(w, node, plan)
    at: dict[tuple, tuple[str, int]] = {}
    for u, fn in plan:
        t, a = w.unit(fam, fn, M)
        at[u.key] = (t, a)
        w.feed_rows(t, "logits", a, grids[u.key], (M,))
        for param, key, _field in u.wires:
            wt, wa = at[key]
            w.feed(t, param, a, [(Row(wt, wa, "o0", 3),)], (M,))
    t, a = at[last.key]
    w.feed(t, REQUEST, a, w.take(node, 1), (M,))
    if not first:
        for k, port in ((2, COUNT), (3, LIVE)):
            took = w.take(node, k)
            if isinstance(took, _Free) or any(r.tid.startswith(STAND_IN) for rows in took for r in rows):
                raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its {port} is no select's output")
            w.feed(t, port, a, took, (M,))
    w.combines[j] = (V, N, M, grids[last.key], [(param, *at[key]) for param, key, _field in last.wires])
    if f"{t}/{TOKEN}" not in w.external:
        w.external.append(f"{t}/{TOKEN}")
    w.out(j, node, [Piece(0, 2, Row(t, a, TOKEN, 2)), Piece(2, 3, Row(t, a, NEXT, 1)), Piece(3, 4, Row(t, a, AHEAD, 1)),
                    Piece(4, 6, Row(t, a, KEY, 2))])
    if isinstance(w, ClassWindow):
        w.live = Row(t, a, NEXT, 1)


def lower_check(w: NodeWindow, j: int, node: Any) -> None:
    """A `ClassStopCheck_v1` row as one unit of `class_check`: the last select's logits rows and chunks' results, as
    `node_window_select.stop_check` reads them, its ids a row each from the selects' ``o0``, ``request`` a boundary read,
    ``count`` and ``live`` the last select's ``o2`` and ``o1``."""
    S, fam, M = node.fn.bindings, _family(node), w.members(node)
    space, sel, *_ = list(runs(w.arg(node, 3, 0)))[0]
    if M != 1 or space != "n" or sel not in w.combines:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its count is no lowered select's")
    V, N, _M, grids, wires = w.combines[sel]
    if [x for x in runs(w.arg(node, 0, 0))] != [x for x in runs(w.fn.body.nodes[sel].args[0].refs)]:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its logits are not its last select's")
    fn = _check_def(V, N, int(S["P"]), int(S["O"]), S["STOP"], S["MAX_MODEL_LEN"])
    t, a = w.unit(fam, fn, M)
    w.feed_rows(t, "logits", a, grids, (M,))
    for param, wt, wa in wires:
        w.feed(t, param, a, [(Row(wt, wa, "o0", 3),)], (M,))
    O, took = int(S["O"]), w.take(node, 1)
    if isinstance(took, _Free) or any(len(rows) != O or any(r.words != 2 for r in rows) for rows in took):
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): its ids are not its {O} selects' served ids, a row each")
    w.feed_rows(t, "ids", a, [[(rows[i],) for rows in took] for i in range(O)], (M,))
    w.feed(t, REQUEST, a, w.take(node, 2), (M,))
    for k, port in ((3, COUNT), (4, LIVE)):
        w.feed(t, port, a, w.take(node, k), (M,))
    w.out(j, node, [Piece(0, 1, Row(t, a, "o0", 1))])


def masked_split(w: NodeWindow, j: int, node: Any) -> None:
    """`node_window_sparse.sparse_mla_split`, with a decode step's key blocks masked (`masked`): each block template's
    ``live`` port wired from the select before's ``o1``, every unit of the step reading that one row. The prefill's
    blocks are not masked."""
    if not isinstance(w, ClassWindow) or w._step_of[j] == 0:
        return NS.sparse_mla_split(w, j, node)
    keys = int(node.fn.bindings.get("T", 0) or 0)
    if keys > INDEX_TOPK:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): {keys} keys, past {INDEX_TOPK} (index_topk), where a key "
                            "block reads the indexer's selection, which the window doesn't lower")
    if w.live is None:
        raise Unexpressible(f"{w.name}: node {j} ({node.fn.id}): a decode step's attention before any select")
    w._masking = w.live
    try:
        NS.sparse_mla_split(w, j, node)
    finally:
        w._masking = None


#: the class's lowerings, beside the request window's (`node_window.FAMILIES` and the others)
FAMILIES: dict[str, Lowering] = {SELECT: lower_select, CHECK: lower_check, "SparseMlaSplit": masked_split}


def class_window(fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN,
                 name: str = CLASS_NAME, stand_ins: bool = False) -> tuple[IF.Circuit, NodeWindow]:
    """The node window of a class's Program ``fn`` (`class_request`), at one rank: `node_window.node_window` (a
    `ClassWindow`) with ``families`` and the class's lowerings (`FAMILIES`) over them:

    - O selects: the prefill's `first_select`, then each decode step's `live_select`, over `GreedyChunk_v2` chunks as
      they are (`lower_select`); and the check `class_check` (`lower_check`);
    - each decode step's key blocks `masked` (`masked_split`), and every other unit, the prefill's and the decode steps',
      at its Definition as `node_window.node_window` lowers it;
    - each masked key-block template's ``live`` port wired from the step before's select's ``o1``, each unit of a step
      reading one row; each ``live_select``'s ``live`` and ``count`` from the step before's ``o1`` and ``o2``; the
      check's from the last select's; each decode step's embedding read's ``tok`` from the step before's ``o3``;
    - each select's and the check's ``request`` the Program's Input, free here, which `rank_window` reads from
      `b.request` (`request_reads`).

    ``families``, ``grain``, ``name`` and ``stand_ins`` are `node_window.node_window`'s. Refused: a class whose keys pass
    2,048, GLM-5.3's ``index_topk``, since the window lowers no indexer (NOT decided, module docstring)."""
    fams = {**(families or {}), **FAMILIES}
    w = ClassWindow(fn, fams, grain, name, stand_ins=stand_ins).lower()
    return w.circuit(), w


def request_reads(w: NodeWindow, param: str = LIMITS_ROW) -> list[IF.Read]:
    """Each port ``w`` leaves free under the Program's ``request`` Input, read from ``param``'s `LIMITS` leaves: one row
    every select and the check read, ``step [0]`` across a template's units."""
    if not w.stand_ins:
        raise Unexpressible(f"{w.name}: a window built without stand_ins keeps no rows for what it leaves free")
    stood = f"{STAND_IN}p:{REQUEST}"
    reads = []
    for tid, port, at, count, rows, grid in w.stood:
        if not any(r.tid == stood for rs in grid for r in rs):
            continue
        if rows != (0, 1, 1) or any(len(rs) != 1 or rs[0].tid != stood or rs[0].out != f"{LIMITS[0]}:{LIMITS[1]}" for rs in grid):
            raise Unexpressible(f"{w.name}: {tid} {port} from unit {at}: a port reading the limits reads them whole")
        reads.append(IF.Read(tid, port, param, at, (count,), LIMITS[0], (0,)))
    return IF.coalesce(reads)


def boundary_reads(w: NodeWindow, rank: int) -> list[IF.Read]:
    """Rank ``rank``'s class window's boundary reads: its prompt's, its limits' and, at rank 0, the served ids."""
    return [*NR.prompt_reads(w), *request_reads(w), *NR.served_reads(w, rank)]


def rank_window(rank: int, fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN) -> NJ.Rank:
    """Rank ``rank``'s class Program ``fn`` lowered into its class window (`class_window`, with stand-ins), the frame
    dropped: its circuit declares the rank's `boundary_reads`, and the free inputs they fill are no longer free
    (`node_window_join.unfilled`), as `node_window_reads.rank_window` builds a request's."""
    c, w = class_window(fn, families, grain, stand_ins=True)
    reads = tuple(boundary_reads(w, rank))
    c = replace(c, boundary_reads=reads)
    return NJ.Rank(rank, c, NJ.unfilled(w.free, reads), {t: tuple(s) for t, s in w.b.spans.items()}, w.steps, w.census(c))


__all__ = ["AHEAD", "CHECK", "CLASS_NAME", "COUNT", "FAMILIES", "KEY", "KEY_BLOCK", "LIMITS", "LIMITS_ROW", "LIVE", "MASKED_BLOCK",
           "NEXT", "REQUEST", "SELECT", "TOKEN", "WORD", "ClassSelect", "ClassStopCheck", "ClassWindow", "Selects",
           "boundary_reads", "class_check", "class_request", "class_window", "first_select", "live_select", "lower_check",
           "lower_select", "masked", "masked_split", "rank_window", "request_reads"]
