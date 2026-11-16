"""A request's Program at its size class (P prompt ids, O output ids) and its node window (`node_window`), with the decode
steps past the request's stop switched off by one bit the circuit carries from step to step, so one window serves every
request of the class (CIR-290, agreed on PRF-618, Sat 14 Nov, 10:58 to 11:10 AM PST; its form settled at 11:47 AM PST).
Two templates are masked by the bit, the sparse attention's key blocks and the select; every other unit keeps its
Definition. Past the stop the served ids are zeros and every other row is its Definition's on its wired inputs. This
module states the interface; every body raises `NotImplementedError` naming the issue until it is built.

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

**The count** runs beside the bit by the same rule (proofs, PRF-618, 11:26 AM PST): the ids written so far, committed,
wired step to step by an ordinary wire, the first a constant inside `first_select` (1), never a boundary row. One
template serves every decode step, so no static holds a step's index, and a select compares the limits only with the
count the step before hands it. Its width is fixed by the class's O: an unsigned value of log2(O) + 1 bits, holding 0
to O, with O a static of each select and of the check. The select refuses to count past O: it never wraps (below).
Past the stop it holds still: a step off writes the count it read, not count + 1, so the steps past the stop are alike.
The length check's n is tied to the last step's count: `class_check` reads that count and holds the rule's n equal to
it, so the count the selects carried and the n the rule gives cannot be two numbers.

**The limits** reach every step's select and the length check from one committed source, so the bit and the check
cannot be given two different limits: the class's `b.request` row, its first four fields (`boundary_span.FIELDS`: the
prompt's length ``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), 8 sixteen-bit leaves from leaf 0, which each
of those units reads by one boundary read (`interface.Read`, ``step [0]`` across them: `LIMITS`).

**The select is where the mask lives.** Each select writes two ids (`live_select`): the served id `TOKEN` (``o0``), its
token ANDed with its own bit, which the served-ids read takes; and the next step's key `KEY` (``o3``), its token ANDed
with the NEXT bit, which the next step's embedding read takes as its ``tok``. At the step that stops they differ: its
id is the n-th served id and is written, while the step after it, the first one off, must read token 0. So the
embedding read stays `EmbeddingRead_v1`, unmasked, and opens its table's row 0 at every step past the stop, and no unit
downstream of it reads the request's last token. The AND is the select Definition's own, in its text and its hash: a
select without it cannot be dropped in, since a step past the stop would then read the request's last id (proofs,
PRF-618, 11:47 AM PST).

**Masked units: the key blocks.** The one other decode template that reads the request is the sparse attention's key
block (`SparseMlaHeadBlock_v1`, every NVIS and FIRST): its ``kv`` reads the cache, the latent rows of the prefill and of
the steps before, which hold the prompt and the request's ids. Its masked version (`masked`) reads the step's bit and
writes zeros past the stop. It writes no cache row: its outputs ``o0``, ``o1``, ``o2`` (the running max, the lanes and
O) are its state, read by the head's next block or its end; the cache rows are the latent row's (`DsaLatentRow_v1`'s
``o0``, `serving_templates.HISTORY`). Past the stop a step's latent row is its Definition's on the step's inputs, which
past the stop are the class's, so a dead step's cache rows do not differ by request (proofs' condition, PRF-618,
11:47 AM PST); and nothing that is not masked reads them, since only later steps read a step's cache rows and every
later step is off. Every other decode unit keeps its Definition: past the stop its rows are what it gives on its wired
inputs. The prefill is not masked.

**Why the form is sound whatever the list says** (proofs, PRF-618, 11:47 AM PST): a unit past the stop that is not
masked is proved like any other, its committed rows its Definition's outputs on its committed inputs, read by wire from
their writers. Soundness needs only the bit and the count right and the served ids past n zeros, which the selects and
the length check hold. The list of what reaches a decode step (CIR-290, run r20261114-193515-bf9c: the key blocks'
``kv``, the embedding read's ``tok``, the check's ``ids``) is the serve's: it lets the serve write a step past the stop
as the class's constant rows without running the model. A hole in it costs the serve work, a unit's rows computed for
the request; it cannot make the auditor accept anything false.

**The length check** (`class_check`) keeps one bit and reads the last step's bit and count with the ids. The ids are the
selects' ``o0``, zeros past n, so the served-ids boundary read of the combines' ``o0``
(`node_window_reads.served_reads`) equals the `b.egress` row (`boundary_span.egress_row` under a stop rule: the n ids,
then O - n zeros, low word first) at every n, and the check needs no second output.

**The wires** (`class_window`): each reader of the bit (the masked key blocks and the next select) reads it from step
s - 1's select, one wire pattern a reader template of ``radix [steps, units]`` whose writer steps by one unit a step;
the next select's ``count`` the same way; each decode step's embedding read's ``tok`` from the step before's ``o3``.
Each rank holds its own select, so the bit a rank's units read is its own rank's, and `node_window_join.join` keeps it
among that rank's own wires.

**Refused until proofs rules on them** (proofs', PRF-618, 11:26 AM PST, after Sun 15 Nov): a request the rule does not
end within O, a ``min_tokens`` other than 0, a span with no stop, and a prompt shorter than P. The interface refuses
each: the first, second and fourth by the check's bit, 0 where the rule's finish is 0, where ``min_tokens`` is not 0
and where ``lp`` is not P (`class_check`); a span with no stop has no stop class (`pipeline.boundary_stop`), so no
class Program is built for it (`class_request`).

**What is NOT decided here**, each named where it is met below:

- the four cases above, proofs' after Sun 15 Nov;
- how a prompt shorter than P would be served, once proofs rules on it: which row the prefill samples, the decode
  steps' positions and the keys past the prompt (`class_request`);
- class G's launch context, which its kernels read and which is a request's own, not its class's (`class_request`);
- an indexer past 2,048 keys (GLM-5.3's ``index_topk``): the window has no lowering for one, so a class with more
  keys than that has no window (`class_window`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from verity.commitments.periods import interface as IF

from .node_window import _GRAIN, NAME, Lowering, NodeWindow
from .window_builder import Grain

if TYPE_CHECKING:
    from verity.circuits.defs import SpecializedDefinition

    from .build import BuildResult
    from .steps import Registered

#: what every body here raises until it is built
TODO = "circuits/request-built-as-proof-blocks: CIR-290"
#: the masked key block's added parameter, the last: its step's bit
LIVE = "live"
#: the parameter of the selects and the length check that reads the limits, and its leaves of `b.request`: the first
#: four fields (``lp``, ``max_tokens``, ``min_tokens``, ``ignore_eos``), two sixteen-bit leaves each, from leaf 0
REQUEST, LIMITS = "request", (0, 8)
#: the later selects' parameter holding the ids written before their step (`live_select`)
COUNT = "count"
#: the selects' output fields: the served id's two words (``o0``, as `GreedyCombine_v2`'s, which the served-ids read
#: takes), the next step's bit, the next step's ``count``, and the id the next step's embedding read takes
TOKEN, NEXT, AHEAD, KEY = "o0", "o1", "o2", "o3"
#: the key block's family, the one decode template besides the select that `masked` takes, and its masked version's
KEY_BLOCK, MASKED_BLOCK = "SparseMlaHeadBlock_v1", "SparseMlaHeadBlockLive_v1"
#: the class window's name, beside `node_window.NAME`
CLASS_NAME = f"{NAME}/class"


def class_request(registered: Registered, P: int, O: int, policy: Mapping[str, Any] | None = None,
                  named: bool = False) -> BuildResult:
    """The Program every request of class (``P``, ``O``) runs at the registration ``registered``, the counterpart of
    `steps.request_program` (whose `build.build_program(phi, "request", LP, T, ...)` it calls at LP = P and T = O - 1
    decode steps, in the registered compiler's form, `dead_free`, at φ's rank) with:

    - the prefill of P prompt rows, read from the class's `b.prompt` (P ids, zero past the prompt), then O - 1 decode
      steps, each fed the step before's `KEY`, returning the O served ids `TOKEN`;
    - the prefill's select `first_select`, each decode step's select `live_select`, each decode step's key blocks
      `masked` reading that step's bit, and every other Call as it is, lowered by `class_window`;
    - the length check last (`class_check`), at T = O;
    - the limits a value, `b.request`'s first four fields (`LIMITS`), read by every select and by the check; the
      statics the class's alone: the registered model's stop ids (`frontend.stop_rule.stop_ids`) and its length
      (`stop_rule.CLASS_KEYS`).

    ``policy``: the sampling policy, greedy (`window_class.GREEDY`) and None alike; a stochastic one is refused, since
    the selects are the greedy select's. ``named``: its trace's names, as `steps.request_program` gives them.

    Refused (until proofs rules): a span with no stop class (`pipeline.boundary_stop`: an abort, an error, one still
    running, a ``min_tokens`` other than 0), for which no class Program is built; and, by the check's bit, a prompt
    shorter than P.

    NOT decided: how a prompt shorter than P would be served. Its sample is at row lp - 1, its decode step s at
    position lp + s, and its decode keys past lp are the prompt's zero rows: each is a value's choice, which the
    Program at P must make in the circuit (a select on lp); until then the class holds only prompts of exactly P ids.
    Nor class G's launch context (`steps.RequestUnit.launch`), which its kernels read and which is the request's batch,
    not its class's: a batched target's class Program needs the launch fixed by the class or read as a value."""
    raise NotImplementedError(TODO)


def masked(fn: SpecializedDefinition) -> SpecializedDefinition:
    """The masked version of a key block ``fn``, `KEY_BLOCK` at any of its statics (every NVIS and FIRST: the GLM-5.3
    window lowers ``SparseMlaHeadBlock_v1{DR=64, DV=512, NVIS, FIRST, BN=64, DOT, SCALE}``, at 43 prompt ids NVIS = 44
    and FIRST = true, run r20261114-193515-bf9c): `MASKED_BLOCK` at ``fn``'s statics, a family of its own (version 2
    of `KEY_BLOCK`'s is its Boolean body, `boolean_sparse_mla`, the same function), with ``fn``'s parameters, then one
    bit `LIVE`, and its outputs ``fn``'s (``o0`` the running max, ``o1`` the lanes,
    ``o2`` O, 1,034 words at DV = 512) where ``live`` is 1 and zeros where it is 0, word for word. Its body is ``fn``'s
    followed by one AND of each output bit with ``live``: one gate an output bit, 16,544 at DV = 512, and one input row
    a unit. It has its own id, so its hash and its Lean verdict are its own, and `KEY_BLOCK` stays for the prefill and
    for a class that always serves O ids. A step's blocks read its bit by wire
    (`class_window`).

    It writes no cache row (module docstring, "Masked units"): the zeros it writes past the stop are its state, which
    only the head's next block and its end read. Refused: any template but `KEY_BLOCK`; every other decode unit keeps
    its Definition."""
    raise NotImplementedError(TODO)


def first_select(V: int, N: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """The prefill's select, the first step's: `live_select` at a step whose bit is 1, a constant its Definition holds,
    and whose ids written before are 0, so it has neither a ``live`` nor a ``count`` parameter. Its statics are
    `live_select`'s, the class's O among them (this function takes O when it is built):

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>)
            -> (o0: Array<2, Value<16>>, o1: bit, o2: Value<log2(O) + 1>, o3: Array<2, Value<16>>)

    ``o0`` the token, the first served id, always written, since the first step is always live; ``o1`` the next step's
    bit (the token is none of the stop ids and the limit is not reached at one id); ``o2`` = 1, the count's first
    value, a constant of the Definition; ``o3`` the token ANDed with ``o1``, the first decode step's key. A template of
    its own, one unit a request, beside `live_select`'s template of the O - 1 later steps. Its chunks are
    `greedy_stages.GreedyChunk_v2` unmasked, the prefill's."""
    raise NotImplementedError(TODO)


def live_select(V: int, N: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyCombine_v3{V, N, O, STOP, MAX_MODEL_LEN}`, a decode step's select: `greedy_stages.GreedyCombine_v2{V, N}`'s
    token over its logits and its chunks' results, taking its step's bit and count and writing the next step's, and
    the two ids (module docstring, "The select is where the mask lives"). O, the class's, fixes the count's width (this
    function takes O when it is built):

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, request: Array<4, i32>, count: Value<log2(O) + 1>,
         live: bit) -> (o0: Array<2, Value<16>>, o1: bit, o2: Value<log2(O) + 1>, o3: Array<2, Value<16>>)

    ``request`` is `b.request`'s first four fields (`LIMITS`), the ones the length check reads; ``count`` the ids
    written before this step (the step before's ``o2``); ``live`` this step's bit (the step before's ``o1``).

    - ``o1``: ``live`` AND the token is none of STOP (none where ``ignore_eos`` is not 0, as
      `boundary_stop.egress_stop` reads it) AND the length limit is not reached at count + 1 ids as
      `stop_rule.StopRule_v1` reads it (below ``max_tokens`` and below MAX_MODEL_LEN - ``lp``, at the rule's edges as
      it reads them), AND ``count`` + 1 is below O. Once off, off: a step off writes 0 whatever it reads.
    - ``o0``, the served id: the token's low and high words ANDed with ``live``. At a live step, the token, the stopping
      step's included (it is the n-th served id); past the stop, 0, `b.egress`'s word there.
    - ``o3``, the next step's key: the token's words ANDed with ``o1``. At a live step that does not stop, the token,
      equal to ``o0``; at the stopping step 0, while ``o0`` is its id; past the stop 0. So the step after the stop and
      every later one read token 0 through the unmasked embedding read.
    - ``o2``: ``count`` + 1 where ``live`` is 1, ``count`` where it is 0: past the stop the count holds still. It never
      counts past O: a live step reading a count of O (which no window of O selects reaches, step s reading at most s)
      writes the bit 0 and holds O, so the count never wraps.

    One field cannot serve as both ids: ANDed with ``live`` it hands the step after the stop the request's last id,
    which reaches every unit downstream of the embedding read; ANDed with ``o1`` it drops the n-th served id from
    `b.egress`. The ANDs are this Definition's, so a select without them cannot be dropped in (proofs, PRF-618,
    11:47 AM PST). The count is proofs' agreed form (PRF-618, 11:26 AM PST), the same rule as the bit: committed, wired
    step to step, the first a constant (`first_select`'s ``o2``). Its chunks are `GreedyChunk_v2`s as they are: past
    the stop their logits are the class's."""
    raise NotImplementedError(TODO)


def class_check(V: int, N: int, O: int, stops: Sequence[int], max_model_len: int) -> SpecializedDefinition:
    """`GreedyStopCheck_v2{V, N, T, STOP, MAX_MODEL_LEN}` at T = ``O``, the next version of
    `greedy_stop.GreedyStopCheck_v1`, one bit:

        (logits: bf16[N], r0: Array<3, Value<16>>, .., r{K-2}, ids: Array<T, i32>, request: Array<4, i32>,
         count: Value<log2(T) + 1>, live: bit) -> bit

    ``request`` is `b.request`'s first four fields (`LIMITS`), the ones every select reads; ``count`` and ``live`` the
    last step's next count and bit (`live_select`'s ``o2`` and ``o1``); ``ids`` the selects' served ids (``o0``). It is
    1 exactly when `frontend.stop_rule.StopClass_v1{T, STOP, MAX_MODEL_LEN}` over ``ids`` and the limits (no stop id
    where ``ignore_eos`` is not 0) gives (n, finish) with finish not 0 and n equal to ``count``, ``min_tokens`` is 0,
    ``lp`` is P, the ids from n on are 0, ``live`` is 0, and ``ids[T - 1]`` is `GreedyCombine_v2{V, N}`'s token over
    the logits and results where n is T (where n is below T it is 0, as the ids past n are). The last step's logits
    past the stop are the class's, not zeros, so its token there is not 0 in general: the check compares
    ``ids[T - 1]`` with the token only where the last step is live. Holding n equal to ``count`` ties the rule's n to
    the count the selects carried, so they cannot be two numbers (proofs, PRF-618, 11:26 AM PST). P is a static of the
    check (the class's), beside T. The ids come a row each from the steps' ``o0``; the logits row is what makes the
    unit a proof block (`interface.check_proof_block`), as in v1. It calls `StopClass_v1` (written, unregistered: its
    registration comes with this check), `GreedyCombine_v2` and the scalar ops v1 calls.

    Its ids a row each, it has T + k + K + 3 rows: k its logits port's (cut as the combine's, 64 at GLM-5.3's select
    at TP8) and K - 1 results. At GLM-5.3 it passes `interface.MAX_ROWS` (256) past T = 169, so its O = 128 class fits
    and an O of 256 or more reads its ids gathered first (`node_window.GATHER_ROWS`).

    Refused until proofs rules on them (PRF-618, 11:26 AM PST): the bit is 0 for a request the rule does not end within
    O (finish 0, ``live`` 1 after the last step), for a ``min_tokens`` other than 0, which the rule doesn't read, and for
    a prompt shorter than P; a span the rule does not read (an abort, an error, one still running) has no stop class
    and no class Program. What the bit should assert for each is proofs'."""
    raise NotImplementedError(TODO)


def class_window(fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN,
                 name: str = CLASS_NAME, stand_ins: bool = False) -> tuple[IF.Circuit, NodeWindow]:
    """The node window of a class's Program ``fn`` (`class_request`), at one rank: `node_window.node_window` with

    - O selects: the prefill's `first_select`, then each decode step's `live_select`, over `GreedyChunk_v2` chunks as
      they are; and the check `class_check` (`node_window_select.token_select` and `stop_check`, in their class forms);
    - each decode step's key blocks `masked` (`MASKED_BLOCK`), and every other unit, the prefill's and the
      decode steps', at its Definition as `node_window.node_window` lowers it;
    - each masked key-block template's ``live`` port wired from the step before's select's ``o1``, by one wire pattern a
      template: ``radix [steps, units]``, ``step [1, 0]``, its base the select one step before the template's first,
      so the writer steps by one unit a step and every unit of a step reads one row; each ``live_select``'s ``live``
      and ``count`` from the step before's ``o1`` and ``o2`` the same way; the check's ``live`` and ``count`` from the
      last select's; each decode step's embedding read's ``tok`` from the step before's ``o3``;
    - each select's and the check's ``request`` a boundary read of `b.request`'s `LIMITS` leaves, one row read by all.

    Each rank's window reads its own rank's bit, so `node_window_join.join` joins the class windows as it joins request
    windows, the bit's wires among each rank's own. The served-ids read (`node_window_reads.served_reads`) takes the
    selects' ``o0`` at all O steps, 2·O sixteen-bit leaves of `b.egress`, each written once (`written_once`). The
    masking is the Program's (`class_request`'s Calls), lowered as any Call.

    ``families``, ``grain``, ``name`` and ``stand_ins`` are `node_window.node_window`'s. Refused: a class whose keys
    pass 2,048, GLM-5.3's ``index_topk``, since the window lowers no indexer (NOT decided, module docstring)."""
    raise NotImplementedError(TODO)


__all__ = ["AHEAD", "CLASS_NAME", "COUNT", "KEY", "KEY_BLOCK", "LIMITS", "LIVE", "MASKED_BLOCK", "NEXT", "REQUEST", "TODO",
           "TOKEN", "class_check", "class_request", "class_window", "first_select", "live_select", "masked"]
