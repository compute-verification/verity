"""Deterministic fold (pure): raw observation log + profile -> unexpanded Verity Program `Captured_v1{PROFILE, WORKLOAD}`.

Contract:

    result = fold(events, profile)          # events: Iterable[Event] (e.g. `log.read_log(path)`)
    result.program                          # verity.ml.circuits.program.Program over an unregistered CompositeDefinition
    result.spec                             # the SpecializedDefinition (Captured_v1 bound to PROFILE, WORKLOAD)
    result.report                           # resolver.ResolutionReport + memory stats + step summaries
    result.digest                           # program digest (str)
    result.outputs                          # per request: list of TokenSelect output refs, one per served step
    result.instances                        # flat list of (step, resolution index, node index range) for navigation

Root parameters (in order):
    prompt  : one `Array(LP_r, I32)` per request (in `header.workload["requests"]` order) from `Root(role="prompt_token")`;
    weights : `verity.ml.definitions.registry.b1.weights_type(C)` Record, fields populated from `Root(role="weight"|"buffer")`
              events through `profile.weight_field(name)`; a Root whose field is None is a Note: nothing is written
              under its view, so a later read of it is Unresolved (raised under an accepted profile) — the M1 log's
              120 per-layer fp8 `_k/_v/_q/_prob_scale` fp32 scalars are such roots and nothing reads them;
    return  : `Array(STEPS_r + 1, I32)` for one request (`Serve_v4`'s signature `(prompt, weights) -> Array(16, I32)`,
              which `match_program` checks), `Tuple(Array(STEPS_r + 1, I32), ...)` for several — the sampled token per
              served step per request.  (The weights record is the LAST parameter, as `match_program` requires.)

Per step (see `tree.step_events`, `resolver.resolve_step`):
    1. StepStart: register the step's `input_ids`/`positions` views; write prompt Root refs (prefill rows) and
       the previous step's TokenSelect output refs (rows named by StepLink) into `input_ids_view` rows —
       checking StepLink.value == the token sampled at src_step;
    2. resolve the forest; for each resolution in order:
         Compute      read operands per row -> Coll(param type, refs) ; `B.call(spec, *colls, name=hint/row)`
                      then write each output row's refs into memory
         Transparent  effect move: read src rows now, write dst rows;  alias/noop: nothing;  alloc: nothing (the
                      allocation was registered by its StorageAlloc; its contents are undefined until written)
         Group        parts in order: each part is its own read-emit-write step; `resolver.parts_of` for both spellings
         Unsupported  development: record + continue (its outputs stay unwritten, later reads become Unresolved);
                      accepted: raise
    3. SampleEnd: the TokenSelect instances' outputs are the step's sampled tokens (check ids against the log).  A sample
       is SERVED only if every token of the request is in the KV cache after the step — vLLM's own rule
       (`gpu_model_runner`: `discard_request_mask = seq_len < num_tokens`, then `valid_sampled_token_ids[i].clear()`,
       with `num_tokens = len(prompt) + len(output_token_ids)`), decided from the log as
       `StepStart.seq_lens[r] >= prompt_len_r + (served samples of r so far)`; a sample taken at a partial-prefill step
       (chunked prefill) or at a partial chunk of a post-preemption recompute is computed — its TokenSelect instance
       stays in the Program — but the engine drops it: it is not returned, not in `result.sampled`, never the target of
       a StepLink.  Such samples are listed in `result.discarded`.  Every SERVED sample is also
       recorded by the position it produces (`logits_rows[j].position + 1`), which is how a re-fed generated token
       (preemption → recompute) finds its producing TokenSelect output when the observer emitted no StepLink for it.

Identity: `CompositeDefinition("Captured", 1, ("PROFILE", "WORKLOAD"), signature, replayer(body), register=False)`
bound to PROFILE=header.profile_id and WORKLOAD=sha256 of the canonical workload JSON, exactly the `derive.py`
pattern (`program.replay.replayer`), so `Program(spec)` works unchanged and `Node.name` hints do not enter the digest.

KV: attention operands are `GatherMany(kv_cache_view, slots per request position)` read through the memory
model (`MemoryModel.read_rows`) — O(T) parts per instance, no separate KV identity.

Implementation notes
--------------------
* Token refs.  At StepStart the fold writes, for every token row `i` of the step, the row's token reference into
  `row_view(input_ids_view, i)` through the memory model: prefill rows (no StepLink) get the prompt parameter leaf
  `("p", r, position)`; decode rows get the TokenSelect output refs recorded at `SampleEnd` of `StepLink.src_step`
  (checked: `StepLink.value == input_token_ids[row] == the token sampled at src_step`).  A past-prompt row WITHOUT a
  StepLink (a generated token re-fed by the recompute after a preemption; the adapter links only the request's most
  recent sample) gets the output refs of the SERVED sample that produced its position (`sample_pos[(req, position)]`,
  checked against the input token); such rows are listed in `result.refed`.  When no served sample produced the
  position the fold FAILS LOUDLY (`FoldError`, class `TokenWiring` in development mode) — and the row's `input_ids` slot
  is left holding whatever the memory model last wrote there, which the error record names (`stale_refs`).  The Embedding pattern's
  `PerRow(input_ids)` operand is then an ordinary memory read.  Width: token buffers are integer views whose element
  may be wider than the Verity token value (vLLM int64 buffers carry 32-bit ids); the fold's width check therefore
  admits a 32-bit `I32` leaf against an `int64` *integer* view (`_TOKEN_WIDTHS`) for token operands/outputs; every
  other width mismatch is an error.  The memory model itself only sees consistent (view dtype) writes and reads.
* Per-row emission.  A `Compute(rows=M)` reads every row's operands first (Shared operands once), emits `M`
  instances (`B.call(spec, *colls, name=...)`), then writes every row's outputs (`out[k]` of a Tuple return goes to
  `outputs[k]`).  Reads-before-writes is what makes in-place kernels (fused add+norm, RoPE) correct.
* Group.  Parts execute in order, each with the ordinary discipline; a later part may read an earlier part's write.
* Instance names: `t=<pos>/layers.<L>/<tail>` where `pos` comes from `Compute.attribution`, else the step's token
  rows (`rows == num_tokens`) or logits rows (`rows == len(logits_rows)`), `L` from the hint (`layers.<L>`), `tail` =
  the hint's last module component or the specialization's base id.  Names are outside the digest.
* Memory events (`StorageAlloc`/`StorageFree`) recorded during a step are applied in log order relative to the
  top-level dispatcher calls (all events up to a root's completion are applied before that root's resolution).
* `fold(..., with_program=False)` stops after the flat body (no `Program`, no digest) — for fast tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, array_of, concat, runs, tuple_of
from verity_vllm.program import sampling_event as SE
from verity.ml.definitions.registry import b1
from verity.ml.definitions.replay import replayer
from verity.ml.circuits.types import Array, Record, Type, Value
from verity.ml.circuits.types import Tuple as TupleT

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    Event,
    Note,
    Root,
    RunEnd,
    RunHeader,
    SampleEnd,
    StepEnd,
    StepLink,
    StepStart,
    StorageAlloc,
    StorageFree,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.log import split_steps
from verity_vllm.observe.fold.memory import MemoryModel, Unresolved, compact_refs
from verity_vllm.observe.fold.patterns.collectives import FUSED_KIND as FUSED_COLLECTIVE_KIND, KINDS as COLLECTIVE_KINDS
from verity_vllm.observe.fold.patterns.target import GEMM_TARGET_SCRATCH_KEY, X01_SCRATCH_KEY, WeightOnlyCalls
from verity_vllm.observe.fold.resolver import (
    Collective,
    Compute,
    Const,
    Gather,
    GatherMany,
    Group,
    Operand,
    Param,
    PerRow,
    Pieces,
    Profile,
    ResolutionReport,
    Shared,
    StepContext,
    Supplied,
    Transparent,
    Unsupported,
    WeightField,
    WitnessParam,
    external_buffers,
    operand_rows,
    operand_view,
    parts_of,
    resolve_step,
    spec_label,
)
from verity_vllm.observe.fold.summary import REFERENCE_INTEGRITY_RULE, summarize  # noqa: F401  (re-exported: fold_compare, tests)
from verity_vllm.observe.fold.supplied import SuppliedInputs
from verity_vllm.observe.fold.tree import StepEvents, TreeNode, step_events
from verity_vllm.observe.fold.token_select import token_select_output
from verity_vllm.observe.fold.views import row_view
from verity_vllm.observe.fold.witnesses import WitnessInputs

I32 = Value(32)

#: integer view dtypes that may hold a 32-bit token value (vLLM's int64 token / sampler buffers)
_TOKEN_WIDTHS = {("int64", 32), ("uint64", 32), ("int32", 32), ("uint32", 32)}


class FoldError(Exception):
    """A fold-level contract violation (pattern/operand shape, width, StepLink wiring, unknown allocation)."""


@dataclass
class FoldResult:
    program: Any
    spec: Any
    report: ResolutionReport
    digest: str
    outputs: dict[str, list]
    instances: list[dict] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    body: Any = None                      # the flat verity.ml.circuits.defs.Body (nodes = instances in log order)
    steps: list[dict] = field(default_factory=list)      # per-step summaries
    errors: list[dict] = field(default_factory=list)     # Unresolved / fold errors recorded in development mode
    timings: dict = field(default_factory=dict)
    sampled: dict = field(default_factory=dict)          # request -> [(step, token id)] from SampleEnd, SERVED samples only
    #: request -> [(step, token id, logits position)] sampled at a partial-prefill / partial-recompute step and dropped by the
    #: engine (`StepStart.seq_lens[r] < prompt_len_r + served so far`): computed (TokenSelect instance in the Program), never
    #: served, never fed back
    discarded: dict = field(default_factory=dict)
    #: token rows past the prompt WITHOUT a StepLink that were wired to the served sample producing their
    #: position (post-preemption recompute): `{step, row, request, position, src_step, token}` — the audit evidence that the
    #: recompute wiring path was exercised (empty on a run without preemption)
    refed: list[dict] = field(default_factory=list)
    weights_type: Any = None
    params: tuple = ()
    requests: list = field(default_factory=list)         # the RunHeader's requests (request_id, prompt_token_ids, ...)
    #: the memory-access journal (exhaustive provenance).  Every read and write the fold made, in order, as plain
    #: dicts: {"seq", "event", "step", "kind": "alloc"|"free"|"root"|"read"|"write", "view": ViewDesc|None, "alloc": int,
    #: "rows": [row indices]|None (GatherMany), "refs": [(space, idx, lo, hi), ...] (reads: what the memory model
    #: answered; writes: what was stored), "what": str}.
    accesses: list[dict] = field(default_factory=list)
    #: tensor-parallel rank log: {"rank", "world"} (None on a TP1 log); one record per collective the fold executed
    #: ({step, k, kind, site, rows, N, rank, world, event, own_view, fields:[{request, param, field, ordinal, rows, offset}]}); and the
    #: `peers_<i>` parameter layout (`_Fold._peers_layout`) -- what program.json states about the collective_received roots
    tp: dict | None = None
    collectives: list[dict] = field(default_factory=list)
    peers_layout: list[dict] = field(default_factory=list)
    #: the witness parameters (`WitnessInputs.layout`): parameter index, request, witness name and the leaf range each read took
    witness_layout: list[dict] = field(default_factory=list)
    #: a `registered_values` profile's supplied inputs (`_Fold.supplied`): the registered `values` and each request's `run_<i>`,
    #: every field's name, width, bits and leaf offset; empty otherwise
    supplied: dict = field(default_factory=dict)
    #: the Program's descriptor (`verity.ml.circuits.codec.encode_program`) as built for the digest: `persist_program`
    #: writes THIS instead of encoding the same Program a second time (36 s + a second digest at B8 1024/128; the descriptor is the
    #: digest's input, so program.json's `program` and `digest` come from one encoding by construction)
    descriptor: Any = None

    @property
    def unresolved(self) -> int:
        return sum(1 for e in self.errors if e.get("class") == "Unresolved")

    @property
    def reference_eligible(self) -> bool:
        """A fold with ANY recorded error / Unsupported resolution is NOT eligible as an
        accepted reference Program (`gates.fold_integrity` applies the same rule to the persisted `fold_summary.json`)."""
        return not self.errors and int(getattr(self.report, "unsupported", 0) or 0) == 0


def fold(events: Iterable[Event], profile: Profile, *, strict: bool | None = None, with_program: bool = True) -> FoldResult:
    """`strict` defaults to `profile.accepted`: raise on the first Unsupported / Unresolved instead of recording."""
    strict = bool(profile.accepted) if strict is None else bool(strict)
    f = _Fold(profile, strict)
    return f.run(events, with_program)


def workload_digest(workload: dict) -> str:
    """sha256 of the canonical (sorted keys, compact) JSON of `RunHeader.workload`."""
    return hashlib.sha256(json.dumps(workload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def config_for_weights(config: dict) -> dict:
    """The static configuration `b1.weights_type` needs, with the derived sizes filled in when a profile config
    omits them (QKV = (NH + 2 KVH) D, GU = 2 I)."""
    C = dict(config)
    if "QKV" not in C:
        C["QKV"] = (C["NH"] + 2 * C["KVH"]) * C["D"]
    if "GU" not in C:
        C["GU"] = 2 * C["I"]
    C.setdefault("BIAS", False)
    if "MAX_POS" not in C:
        raise FoldError("profile config lacks MAX_POS (the cos_sin table length)")
    return C


def weights_type_of(profile: Any, config: dict) -> Record:
    """The fold's `weights` Record: the profile's own when it states one (`Profile.weights_type`), else `analytic.weights_type`
    over the configuration (`b1.weights_type(C)`, + `lm_head` when untied; b1.py is frozen)."""
    own = getattr(profile, "weights_type", None)
    return own if own is not None else analytic.weights_type(config_for_weights(config))


_LAYER_RE = re.compile(r"layers\.(\d+)")


def _layer_of(hint: str) -> int | None:
    m = _LAYER_RE.search(hint or "")
    return int(m.group(1)) if m else None


def _tail_of(hint: str) -> str | None:
    """The op label of a hint: `model.layers.0.self_attn.qkv_proj/model.layers.0.self_attn.qkv_proj.weight` → `qkv_proj`,
    `model.norm/model.norm.weight` → `norm`, `.../rope.q` → `q`, `.../attn` → `attn`."""
    if not hint:
        return None
    h = hint.split()[0]
    h = h.split("/")[-1]
    parts = h.split(".")
    while len(parts) > 1 and parts[-1] in ("weight", "bias"):   # a weight root name: label by its module
        parts.pop()
    return parts[-1] or None


def _width_of(t: Type) -> int:
    w = t.homogeneous_width()
    if w is None:
        raise FoldError(f"operand type {t!r} is not width-homogeneous")
    return w


# ---------------------------------------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------------------------------------


class _Fold:
    def __init__(self, profile: Profile, strict: bool) -> None:
        self.profile, self.strict = profile, strict
        self.mem = MemoryModel()
        self.report = ResolutionReport()
        self.notes: list[str] = []
        self.errors: list[dict] = []
        self.roots_by_alloc: dict[int, list[Root]] = {}
        self.buffers: dict[str, ViewDesc] = {}
        self.instances: list[dict] = []
        self.accesses: list[dict] = []
        self.steps: list[dict] = []
        self.timings: dict = {}
        self.header: RunHeader | None = None
        self.config: dict = {}
        self.requests: list[dict] = []
        self.req_index: dict[str, int] = {}
        self.params: tuple = ()
        self.B: Builder | None = None
        self.weights_type = None
        self.weights_coll: Coll | None = None
        self.prompt_colls: list[Coll] = []
        self.seed_param: dict[int, int] = {}      # request index -> seed parameter index
        self.seed_colls: dict[int, Coll] = {}
        # a TP rank log: `tp` = {"rank", "world"} from RunHeader.notes.tp2 (None on a TP1 log: nothing below is allocated);
        # `peers_param[i]` = the `peers_<i>` parameter index of request i; `peers_fields[i]` = its fields in issue order (finalised into a
        # Record at `_finish`); `peers_leaves[i]` = leaves allocated so far; `collectives` = one record per collective the fold executed
        self.tp: dict[str, int] | None = None
        self.peers_param: dict[int, int] = {}
        self.peers_fields: dict[int, list[dict]] = {}
        self.peers_leaves: dict[int, int] = {}
        self.witness_inputs = WitnessInputs()   # `Profile.witnesses`
        self.supplied_inputs = SuppliedInputs()   # a `registered_values` profile's run and registered values
        self.collectives: list[dict] = []
        self.cur_coll_k: int = 0
        self._open_call: tuple[dict, int] | None = None          # (record, next row) of a call whose per-request parts are still due
        self._coll_prep: dict[int, tuple[Compute, dict]] = {}   # id(c) -> (c, prep); c held so an id() reuse cannot alias a stale prep
        self.sampled: dict[tuple[str, int], int] = {}
        self.outputs: dict[tuple[str, int], Refs] = {}
        #: (request, position) -> step whose SERVED sample produced the token at `position` (= logits row position + 1)
        self.sample_pos: dict[tuple[str, int], int] = {}
        self.refed: list[dict] = []
        self.per_request: dict[str, list[tuple[int, Refs]]] = {}
        self.sampled_by_request: dict[str, list[tuple[int, int]]] = {}
        self.discarded_by_request: dict[str, list[tuple[int, int, int]]] = {}
        self.token_refs: dict[tuple[int, int], Refs] = {}
        self.weight_roots: dict[tuple, list[tuple[str, int]]] = {}
        self.n_alloc = self.n_free = 0
        self.cur_step: int = -1
        self.cur_start: StepStart | None = None
        self.cur_resolution: int = -1
        self.cur_event: int = -1
        self.shared_consts: tuple[int, dict[tuple[int, int], Coll]] = (-1, {})
        self.ended = False

    # -- driver --------------------------------------------------------------------------

    def run(self, events: Iterable[Event], with_program: bool) -> FoldResult:
        t0 = time.time()
        for pre, step_slice in split_steps(events):
            for ev in pre:
                self._pre_event(ev)
            if step_slice:
                self._step(step_slice)
        if self.header is None:
            raise FoldError("log has no RunHeader")
        self.timings["fold_s"] = round(time.time() - t0, 3)
        return self._finish(with_program)

    def _pre_event(self, ev: Event) -> None:
        if isinstance(ev, RunHeader):
            self._header(ev)
        elif isinstance(ev, StorageAlloc):
            self._alloc(ev)
        elif isinstance(ev, StorageFree):
            self._free(ev)
        elif isinstance(ev, Root):
            self._root(ev)
        elif isinstance(ev, Note):
            self.notes.append(f"log note: {ev.text}")
            self.buffers.update(external_buffers(ev, self.mem.device))
        elif isinstance(ev, RunEnd):
            self.ended = True
        elif isinstance(ev, (DispatchEnter, DispatchExit, TritonLaunch)):
            self._problem("OutsideStep", f"{type(ev).__name__} id {getattr(ev, 'id', '?')} outside any step",
                          {"op": getattr(ev, "op", getattr(ev, "fn", "?"))})
        elif isinstance(ev, (StepLink, SampleEnd, StepEnd)):
            self._problem("OutsideStep", f"{type(ev).__name__} outside any step", {})
        else:
            self.notes.append(f"ignored preamble event {type(ev).__name__}")

    # -- preamble --------------------------------------------------------------------------

    def _header(self, h: RunHeader) -> None:
        if self.header is not None:
            raise FoldError("second RunHeader")
        self.header = h
        cfg = dict(self.profile.config or {})
        if not cfg:
            cfg = dict((h.model or {}).get("config") or {})
        self.config = cfg
        self.weights_type = weights_type_of(self.profile, cfg)
        reqs = list((h.workload or {}).get("requests") or [])
        if not reqs:
            raise FoldError("RunHeader.workload has no requests")
        self.requests = reqs
        self.req_index = {r["request_id"]: i for i, r in enumerate(reqs)}
        # n > 1: the runner's rows carry vLLM's child ids (`{index}_{request_id}`, header `workload.children`, a
        # fact of the fork recorded by the adapter).  A child's prompt IS its parent's prompt parameter (one root, not n copies);
        # the served sequences (Program outputs) are per child.  A child naming an unknown parent is a header error.
        children = list((h.workload or {}).get("children") or [])
        for c in children:
            if c.get("parent") not in self.req_index:
                raise FoldError(f"RunHeader.workload.children: {c.get('request_id')!r} names unknown parent {c.get('parent')!r}")
            self.req_index[c["request_id"]] = self.req_index[c["parent"]]
        forked = {c["parent"] for c in children}
        self.output_ids = [cid for r in reqs for cid in ([c["request_id"] for c in children if c["parent"] == r["request_id"]]
                                                          if r["request_id"] in forked else [r["request_id"]])]
        params: list[tuple[str, Type]] = []
        for i, r in enumerate(reqs):
            lp = len(r["prompt_token_ids"])
            params.append(("prompt" if len(reqs) == 1 else f"prompt_{i}", Array(lp, I32)))
        # a stochastic run (header sampling not greedy): every seeded request's seed is a declared runtime INPUT of the
        # Program -- one `seed` / `seed_<i>` parameter (Array<1, Value<64>>, i = the request's index, like the prompts) between the
        # prompts and the weights; the sampler instance reads it through a `Param("seed", ...)` operand (resolver).  Layout:
        # prompts..., seeds..., weights (weights stay the LAST input node: cb_a.compare / batch_decomp rely on that).
        samp = dict((h.workload or {}).get("sampling") or {})
        self.seed_param: dict[int, int] = {}      # request index -> parameter index
        if samp.get("greedy") is not True:
            for i, r in enumerate(reqs):
                if r.get("seed") is not None:
                    self.seed_param[i] = len(params)
                    params.append(("seed" if len(reqs) == 1 else f"seed_{i}", Array(1, Value(64))))
        # a TENSOR-PARALLEL rank log (RunHeader.notes.tp2 = {rank, world_size} written by the rank observer,
        # tp2_worker.tp2_install): every request gets ONE `peers` / `peers_<i>` root parameter (root class `collective_received`) between
        # the seeds and the weights -- a Record whose fields, in issue order, are the OTHER rank's partials of the collectives whose rows the
        # request owns (`s<step>_k<k>`: Array<M_r, Array<N, Bf16>>), filled in as the fold executes the collectives and finalised at
        # `_finish`.  Layout: prompts..., seeds..., peers..., weights (weights stay the LAST input node; the codec lowers a homogeneous bf16
        # Record to ONE Input16 node per request, so the request-owned Input nodes stay the leading ones -- batch_decomp).  A TP1 log
        # (no tp2 note) allocates nothing here: its layout and digests are unchanged.
        tp = dict((h.notes or {}).get("tp2") or {})
        self.tp = None
        if tp and int(tp.get("world_size") or 1) > 1:
            if tp.get("rank") is None:
                raise FoldError("RunHeader.notes.tp2 has world_size > 1 but no rank")
            self.tp = {"rank": int(tp["rank"]), "world": int(tp["world_size"])}
            # any world >= 2: world 2 binds AllReduce2_v1 / AllGather2_v1 (the recorded identities), world > 2 the
            # world-parametric AllReduce_v2{WORLD,N} / AllGather_v1{WORLD,N} (b1_tp2.all_reduce / all_gather); a rank outside the world is refused
            if self.tp["world"] < 2 or not 0 <= self.tp["rank"] < self.tp["world"]:
                raise FoldError(f"tensor-parallel world {self.tp['world']} rank {self.tp['rank']}: rank must be in [0, world) with world >= 2; refused")
            self.tp_peer_ranks = [r for r in range(self.tp["world"]) if r != self.tp["rank"]]
            for i, _r in enumerate(reqs):
                self.peers_param[i] = len(params)
                self.peers_fields[i] = []
                self.peers_leaves[i] = 0
                params.append(("peers" if len(reqs) == 1 else f"peers_{i}", Record()))   # placeholder type: fields finalised at _finish
        self.witness_inputs.declare(params, tuple(getattr(self.profile, "witnesses", ())), len(reqs))
        if getattr(self.profile, "registered_values", False):
            self.supplied_inputs.declare(params, len(reqs))
        params.append(("weights", self.weights_type))
        self.params = tuple(params)
        self.B = Builder(self.params, I32)  # ret type is set once the outputs are known
        self.prompt_colls = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(self.params[:len(reqs)])]
        self.seed_colls = {i: Coll(self.params[k][1], Affine("p", k, 0, self.params[k][1].leaves)) for i, k in self.seed_param.items()}
        self.weight_only = WeightOnlyCalls(h, len(self.params) - 1)
        self.weights_coll = Coll(self.weights_type, Affine("p", self.weight_only.w_idx, 0, self.weights_type.leaves))

    def _alloc(self, ev: StorageAlloc) -> None:
        self.mem.alloc(ev)
        self.n_alloc += 1
        self._journal("alloc", None, None, alloc=ev.alloc, what=f"nbytes={getattr(ev, 'nbytes', None)} gen={getattr(ev, 'generation', None)}")

    def _free(self, ev: StorageFree) -> None:
        try:
            self.mem.free(ev.alloc)
        except KeyError:
            self.notes.append(f"StorageFree of unknown allocation {ev.alloc}")
        self.n_free += 1
        self._journal("free", None, None, alloc=ev.alloc, what="")

    def _journal(self, kind: str, view: ViewDesc | None, refs: Refs | None, *, alloc: int | None = None, rows: list | None = None,
                 what: str = "", event: int | None = None) -> None:
        """Append one memory access to `accesses`, its refs the `Refs` read (expanded by a reader, `memory.access_runs`)."""
        self.accesses.append({"seq": len(self.accesses), "event": event if event is not None else self.cur_event, "step": self.cur_step,
                              "kind": kind, "view": view, "alloc": alloc if alloc is not None else (view.alloc if view is not None else None),
                              "rows": rows, "refs": refs, "what": what})

    def _root(self, r: Root) -> None:
        self.roots_by_alloc.setdefault(r.view.alloc, []).append(r)
        if r.role in ("weight", "buffer", "const"):
            self._weight_root(r)
        elif r.role == "prompt_token":
            self._prompt_root(r)
        # positions are public step facts: nothing to bind

    def _weight_root(self, r: Root) -> None:
        path = self.profile.weight_field(r.name)
        if path is None:
            # Not an error by itself (the M1 log has 120 per-layer fp8 `_k/_v/_q/_prob_scale` fp32 scalars no bf16
            # kernel reads): nothing is written under the view, so a later *read* of it is what surfaces — as
            # Unresolved (raised in strict mode).  Recorded as a note either way.
            self.notes.append(f"unbound root: Root {r.name!r} ({r.role}, {r.view!r}) has no weights field in profile {self.profile.id}")
            return
        coll = self.weights_coll
        try:
            for k in path:
                coll = coll[k]
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise FoldError(f"weight field path {path!r} for {r.name!r} is not in {self.weights_type!r}: {e}") from None
        if coll.type.leaves != r.view.numel:
            raise FoldError(f"Root {r.name!r}: view has {r.view.numel} elements, field {path!r} has {coll.type.leaves} leaves")
        w = _width_of(coll.type)
        if w != r.view.bits:
            raise FoldError(f"Root {r.name!r}: field {path!r} is {w}-bit, the view is {r.view.dtype}")
        self._mem("write", self.mem.write_root, r.view, coll.refs)
        self._journal("root", r.view, coll.refs, what=f"{r.role}:{r.name}")
        self.weight_roots.setdefault(tuple(path), []).append((r.name, r.view.alloc))

    def _prompt_root(self, r: Root) -> None:
        # informational: the fold binds prefill rows by position (StepStart.rows); the Root's values are checked
        if r.values is None:
            return
        m = re.match(r"prompt\[(.*)\]", r.name or "")
        req = m.group(1) if m else None
        if req is not None and req in self.req_index:
            prompt = self.requests[self.req_index[req]]["prompt_token_ids"]
            if list(r.values) and not all(v in prompt for v in r.values):
                self._problem("PromptRoot", f"prompt Root {r.name} values {list(r.values)[:4]} not in the workload prompt", {})

    # -- steps -----------------------------------------------------------------------------

    def _step(self, step_slice: list[Event]) -> None:
        if self.header is None:
            raise FoldError("StepStart before RunHeader")
        t0 = time.time()
        se = step_events(step_slice)
        start = se.start
        self.cur_step, self.cur_start = start.step, start
        self.cur_coll_k = 0            # collective issue ordinal restarts every step
        node_lo = len(self.B.nodes)
        computes_before, unsupported_before, errors_before = self.report.computes, self.report.unsupported, len(self.errors)
        by_spec_before = dict(self.report.by_spec)
        # allocations / roots observed inside the step but before the first top-level call are applied first
        for ev in se.other:
            if isinstance(ev, Root):
                self._root(ev)
            elif isinstance(ev, Note):
                self.notes.append(f"step {start.step} note: {ev.text}")
        ctx = StepContext(self.header, start, se.sample, se.links, self.roots_by_alloc, self.config, self.profile, buffers=self.buffers)
        self._write_tokens(start, se.links)
        resolutions = resolve_step(se, ctx)
        x01 = ctx.scratch.get(X01_SCRATCH_KEY)
        if x01:                                    # [X-01] record the target fact the resolver admitted (host SM count, not the profile pin)
            self.notes.append(f"step {start.step} target: {x01['launches']} persistent-GEMM launches pinned to NUM_SMS={x01['pinned']} "
                              f"from {x01['source']} (profile default {x01['profile_default']}; {x01['basis']})")
        gt = ctx.scratch.get(GEMM_TARGET_SCRATCH_KEY)
        if gt:                                     # the registered GEMM target whose launch geometry the GEMM launches were pinned to
            self.notes.append(f"step {start.step} gemm target: {gt['launches']} persistent-GEMM launches pinned to the launch geometry of registered "
                              f"target {gt['target']!r} (cc {'.'.join(map(str, gt['compute_capability']))}, dot {gt['dot']}, tuned table "
                              f"{gt['tuned_table_sha256']}; header family {gt['header_tuned_gemm_arch_family']!r}; refusal {gt['refusal']!r})")
        # execute in log order, interleaving memory events with the top-level calls
        pairs = list(zip(se.roots, resolutions))
        if len(pairs) != len(se.roots):
            raise FoldError("resolve_step returned fewer resolutions than roots")
        nxt = 0
        pending: list[Event] = []
        top_open: int | None = None
        for ev in step_slice[1:]:
            if isinstance(ev, (StorageAlloc, StorageFree)):
                pending.append(ev)
            elif isinstance(ev, DispatchEnter):
                if ev.parent is None:
                    top_open = ev.id
            elif isinstance(ev, DispatchExit):
                if top_open is not None and ev.id == top_open:
                    top_open = None
                    nxt = self._flush_and_execute(pending, pairs, nxt)
            elif isinstance(ev, TritonLaunch):
                if ev.parent is None:
                    nxt = self._flush_and_execute(pending, pairs, nxt)
        for ev in pending:
            self._mem_event(ev)
        pending.clear()
        while nxt < len(pairs):  # roots whose completion was not seen (tolerated by tree.py): execute in order
            root, r = pairs[nxt]
            self.cur_resolution, self.cur_event = nxt, root.id
            self._execute(r, root)
            nxt += 1
        self._sample_end(se)
        by_spec = {k: v - by_spec_before.get(k, 0) for k, v in self.report.by_spec.items() if v - by_spec_before.get(k, 0)}
        self.steps.append({"step": start.step, "rows": len(start.rows), "roots": len(se.roots),
                           "computes": self.report.computes - computes_before, "nodes": len(self.B.nodes) - node_lo,
                           "node_range": [node_lo, len(self.B.nodes)], "by_spec": dict(sorted(by_spec.items())),
                           "unsupported": self.report.unsupported - unsupported_before,
                           "errors": len(self.errors) - errors_before, "seconds": round(time.time() - t0, 3)})
        self.cur_start = None

    def _flush_and_execute(self, pending: list[Event], pairs: list, nxt: int) -> int:
        for ev in pending:
            self._mem_event(ev)
        pending.clear()
        if nxt < len(pairs):
            root, r = pairs[nxt]
            self.cur_resolution, self.cur_event = nxt, root.id
            self._execute(r, root)
            nxt += 1
        return nxt

    def _mem_event(self, ev: Event) -> None:
        if isinstance(ev, StorageAlloc):
            self._alloc(ev)
        elif isinstance(ev, StorageFree):
            self._free(ev)

    # -- token wiring --------------------------------------------------------------------------

    def _write_tokens(self, start: StepStart, links: list[StepLink]) -> None:
        link_by_row: dict[int, StepLink] = {}
        for l in links:
            if l.step != start.step:
                self._problem("StepLink", f"StepLink for step {l.step} inside step {start.step}", {})
                continue
            link_by_row[l.row] = l
        if len(start.input_token_ids) < len(start.rows):
            raise FoldError(f"step {start.step}: {len(start.input_token_ids)} input token ids for {len(start.rows)} rows")
        for i, (req, pos) in enumerate(start.rows):
            try:
                refs = self._token_refs(start, i, req, pos, link_by_row.get(i))
            except FoldError as e:
                if self.strict:
                    raise
                # fail loudly, never silently: the row's `input_ids` slot is NOT rewritten, so record what
                # it currently holds (the previous step's ref — exactly the stale content the qwen8 divergence embedded)
                detail: dict[str, Any] = {"row": i, "request": req, "position": pos}
                if start.input_ids_view is not None:
                    try:
                        stale = self.mem.read(row_view(start.input_ids_view, i))
                        detail["stale_refs"] = [list(r) for r in runs(stale)[:4]]
                    except Exception as se:  # noqa: BLE001 - diagnostics only
                        detail["stale_refs"] = f"unreadable: {type(se).__name__}"
                self._problem("TokenWiring", str(e), detail)
                continue
            self.token_refs[(start.step, i)] = refs
            if start.input_ids_view is not None:
                view = row_view(start.input_ids_view, i)
                self._check_width(I32, view, f"step {start.step} token row {i}", token=True)
                self._mem("write", self.mem.write, view, refs)
                self._journal("token", view, refs, what=f"step {start.step} row {i}")
        if start.input_ids_view is None and start.rows:
            self.notes.append(f"step {start.step}: no input_ids_view; token refs were not written into memory")

    def _token_refs(self, start: StepStart, i: int, req: str, pos: int, l: StepLink | None) -> Refs:
        """The reference of token row `i`: a prompt parameter leaf, or the sampler output a StepLink names."""
        tok = int(start.input_token_ids[i])
        if req not in self.req_index:
            raise FoldError(f"step {start.step} row {i}: unknown request {req!r}")
        if l is not None:
            if l.request_id != req:
                raise FoldError(f"step {start.step} row {i}: StepLink names request {l.request_id!r}, the row is {req!r}")
            if l.relation != "eq":
                raise FoldError(f"step {start.step} row {i}: StepLink relation {l.relation!r} is not 'eq'")
            samp = self.sampled.get((req, l.src_step))
            if (l.value is not None and int(l.value) != tok) or (samp is not None and int(samp) != tok):
                raise FoldError(f"step {start.step} row {i}: token {tok} != StepLink.value {l.value} / sampled at step {l.src_step} {samp}")
            refs = self.outputs.get((req, l.src_step))
            if refs is None:
                raise FoldError(f"step {start.step} row {i}: StepLink to step {l.src_step} whose sampler output for {req!r} is unknown")
            return refs
        r = self.req_index[req]
        prompt = self.requests[r]["prompt_token_ids"]
        if pos >= len(prompt):
            # a generated token re-fed by the recompute after a preemption: the adapter emits a StepLink
            # only for the request's most recent sample, so the row is wired by the position it occupies — to the SERVED
            # sample that produced that position (registered at its SampleEnd) — and checked against the input token.
            src = self.sample_pos.get((req, pos))
            if src is None:
                raise FoldError(f"step {start.step} row {i}: position {pos} of {req!r} is past the prompt ({len(prompt)}), has no StepLink "
                                f"and no served sample of {req!r} produced position {pos}")
            samp = self.sampled.get((req, src))
            if samp is None or int(samp) != tok:
                raise FoldError(f"step {start.step} row {i}: re-fed token {tok} at position {pos} of {req!r} != token {samp} sampled at step {src}")
            refs = self.outputs.get((req, src))
            if refs is None:
                raise FoldError(f"step {start.step} row {i}: position {pos} of {req!r} was sampled at step {src} but that sampler output is unknown")
            self.refed.append({"step": start.step, "row": i, "request": req, "position": pos, "src_step": src, "token": tok})
            return refs
        if int(prompt[pos]) != tok:
            raise FoldError(f"step {start.step} row {i}: input token {tok} != prompt[{pos}] = {prompt[pos]}")
        return self.prompt_colls[r][pos].refs

    # -- resolutions -----------------------------------------------------------------------------

    def _execute(self, r, root: TreeNode) -> None:
        self.report.add(r)
        if isinstance(r, Unsupported):
            if self.strict:
                raise FoldError(f"Unsupported event {r.event_id} {r.op}: {r.reason} {r.detail}")
            return
        try:
            if not isinstance(r, (Group, Compute, Transparent)):
                raise FoldError(f"unknown resolution {type(r).__name__}")
            # a grouped resolution (`resolver.Group` or a primary with `before`/`also` members) is executed part by
            # part IN ORDER, each part its own memory step (reads, instances, writes), so a scope's transparent glue
            # can feed the compute that follows it.  `parts_of` is the one accessor for both spellings.
            for m in parts_of(r):
                if isinstance(m, Unsupported):  # already counted by report.add(r)
                    if self.strict:
                        raise FoldError(f"Unsupported part {m.event_id} {m.op}: {m.reason} {m.detail}")
                    continue
                self._parts([m], root)
        except (Unresolved, FoldError) as e:
            if self.strict:
                raise
            self._problem(type(e).__name__, str(e), {"event": root.id, "op": root.op, "pattern": getattr(r, "pattern", "")})

    def _parts(self, parts: list, root: TreeNode) -> None:
        """One memory step: the parts' reads, then their instances, then their writes (called with one part at a time)."""
        reads: list[tuple[str, Any, Any]] = []
        for p in parts:
            if isinstance(p, Compute):
                try:
                    reads.append(("c", p, self._compute_reads(p)))
                except BaseException:
                    self._coll_prep.pop(id(p), None)      # a failed read leaves no prep behind for an id() reuse
                    raise
            elif isinstance(p, Transparent):
                if p.effect == "move":
                    reads.append(("m", p, self._move_reads(p)))
                elif p.effect in ("alias", "noop", "alloc"):
                    reads.append(("-", p, None))
                else:
                    raise FoldError(f"unknown Transparent effect {p.effect!r} (event {root.id} {root.op})")
            else:
                raise FoldError(f"Group part of type {type(p).__name__} (event {root.id})")
        emitted = []
        for tag, p, data in reads:
            emitted.append(self.weight_only.emit(p, data, self._compute_emit, root) if tag == "c" else None)
        for (tag, p, data), outs in zip(reads, emitted):
            if tag == "c":
                self._compute_writes(p, outs)
            elif tag == "m":
                self._move_writes(p, data)

    # Compute ------------------------------------------------------------------------------------

    def _compute_reads(self, c: Compute) -> list[list[Coll]]:
        params, _ = c.spec.signature()
        if len(c.operands) != len(params):
            raise FoldError(f"{spec_label(c.spec)}: {len(c.operands)} operands for {len(params)} parameters ({c.hint})")
        for k, op in enumerate(c.operands):
            n = operand_rows(op)
            if n is not None and n < c.rows:
                raise FoldError(f"{spec_label(c.spec)}: operand {k} {type(op).__name__} covers {n} rows, the compute has {c.rows}")
        shared: dict[int, Coll] = {}
        # a collective operand is numbered (step, k) and its peers fields allocated ONCE per Compute, before any row is read
        coll_prep = {k: self._collective_prepare(c, op, params[k][1], params[k][0]) for k, op in enumerate(c.operands) if isinstance(op, Collective)}
        if coll_prep:
            # keyed by id(c) AND holding c: a prep whose Compute's reads raise (Unresolved own row -> a non-strict `problem`) is
            # never popped, and a LATER Compute allocated at the same address (a 2-row decode Compute with no collective operand) would
            # read its stale 1-row row_map.  The emit side checks the identity and the failing part drops its prep (see _parts).
            self._coll_prep[id(c)] = (c, next(iter(coll_prep.values())))
        rows: list[list[Coll]] = []
        pieces: dict[ViewDesc, Refs] = {}   # a view several rows' pieces read is read once: every read of a Compute precedes its writes
        for i in range(c.rows):
            colls: list[Coll] = []
            for k, (op, (pname, ptype)) in enumerate(zip(c.operands, params)):
                if isinstance(op, Shared):
                    if k not in shared:
                        shared[k] = self._wrap(ptype, self._read_operand(op, i), op.view, f"{spec_label(c.spec)}.{pname}")
                    colls.append(shared[k])
                elif isinstance(op, Collective):
                    colls.append(self._collective_parts(coll_prep[k], op, i, ptype, c, pname))
                elif isinstance(op, Const):
                    # a Program constant: ONE `Const<W>[bits]` instance per Compute (emitted here, i.e. before the instance
                    # that reads it, in operand order), shared by its rows -- the derived Program's `ctx.kind("const")` of the same value
                    if k not in shared:
                        shared[k] = self._const(op, ptype, c, pname)
                    colls.append(shared[k])
                elif isinstance(op, Pieces):
                    colls.append(self._pieces(op, i, ptype, c, pname, pieces))
                elif isinstance(op, Param):
                    colls.append(self._param(op, i, ptype, c, pname))
                elif isinstance(op, Supplied):
                    colls.append(Coll(ptype, self.supplied_inputs.leaf(op, ptype, f"{spec_label(c.spec)}.{pname}", self.req_index, FoldError)))
                elif isinstance(op, WeightField):
                    if k not in shared:
                        shared[k] = self._weight_field(op, ptype, f"{spec_label(c.spec)}.{pname}")
                    colls.append(shared[k])
                elif isinstance(op, WitnessParam):
                    colls.append(self._witness(op, i, ptype, c, pname))
                else:
                    colls.append(self._wrap(ptype, self._read_operand(op, i), op.view, f"{spec_label(c.spec)}.{pname}[row {i}]"))
            rows.append(colls)
        return rows

    def _pieces(self, op: Pieces, i: int, ptype: Type, c: Compute, pname: str, read: dict[ViewDesc, Refs]) -> Coll:
        """Row i of a `Pieces` operand: its views' refs (`read`: the views the Compute has read) and its constants' copies, concatenated in
        order."""
        what, w, parts = f"{spec_label(c.spec)}.{pname}[row {i}]", _width_of(ptype), []
        for piece in op.rows[i]:
            if isinstance(piece, Const):
                if int(piece.width) != w or not piece.shared:
                    raise FoldError(f"{what}: a piece's Const{piece.width} is not a shared constant of the parameter's {w}-bit leaves")
                parts += [self._const(piece, Value(w), c, pname).refs] * int(piece.count)
            else:
                self._check_width(ptype, piece, what)
                if piece not in read:
                    read[piece] = self._mem("read", self.mem.read, piece)
                    self._journal("read", piece, read[piece], what=f"piece of row {i}")
                parts.append(read[piece])
        refs = concat(parts)
        if refs.count != ptype.leaves:
            raise FoldError(f"{what}: {refs.count} refs, the parameter type {ptype!r} has {ptype.leaves} leaves")
        return Coll(ptype, refs)

    def _const(self, op: Const, ptype: Type, c: Compute, pname: str) -> Coll:
        """Emit the constant's primitive instance and return its output (typed `Value<W>`); a `shared` constant once per
        value in the resolution."""
        if isinstance(op.bits, tuple):
            if ptype != Array(len(op.bits), Value(int(op.width))):
                raise FoldError(f"{spec_label(c.spec)}.{pname}: {len(op.bits)} Const{op.width} for a parameter of type {ptype!r}")
            tail = op.name or pname
            return array_of([self._const(Const(op.width, int(b), f"{tail}{i}"), Value(int(op.width)), c, pname) for i, b in enumerate(op.bits)])
        if ptype != Value(int(op.width)):
            raise FoldError(f"{spec_label(c.spec)}.{pname}: Const{op.width} for a parameter of type {ptype!r}")
        if op.shared:
            if self.shared_consts[0] != self.cur_resolution:
                self.shared_consts = (self.cur_resolution, {})
            cache, key = self.shared_consts[1], (int(op.width), int(op.bits))
            if key not in cache:
                cache[key] = self._const(dataclasses.replace(op, shared=False), ptype, c, pname)
            return cache[key]
        prim = b1.const(int(op.width), int(op.bits))
        tail = op.name or pname
        rp = self._attribution(c, 0)
        name = "/".join(([f"t={rp[1]}"] if rp else []) + [f"{_tail_of(c.hint) or c.spec.base_id}/{tail}"])
        out = self.B.call(prim, name=name)
        k = len(self.B.nodes) - 1
        event = min(c.consumed) if c.consumed else self.cur_event
        self.instances.append({"step": self.cur_step, "event": event, "resolution": self.cur_resolution, "node": k, "spec": prim.id,
                               "row": 0, "rows": 1, "name": name, "hint": f"{c.hint}: {tail}", "request": rp[0] if rp else None,
                               "position": rp[1] if rp else None, "pattern": c.pattern, "operand_views": [], "output_views": [],
                               "const": {"width": int(op.width), "bits": f"0x{int(op.bits):0{int(op.width) // 4}x}"}})
        return out

    def _param(self, op: Param, i: int, ptype: Type, c: Compute, pname: str) -> Coll:
        """Row i's declared runtime parameter (the request's `seed`): the parameter's leaves, no memory read."""
        if op.kind != "seed":
            raise FoldError(f"{spec_label(c.spec)}.{pname}: unknown Param kind {op.kind!r}")
        rid = op.keys[i]
        if rid not in self.req_index:
            raise FoldError(f"{spec_label(c.spec)}.{pname}[row {i}]: request {rid!r} is not in the run header")
        coll = self.seed_colls.get(self.req_index[rid])
        if coll is None:
            raise FoldError(f"{spec_label(c.spec)}.{pname}[row {i}]: request {rid!r} has no seed parameter (unseeded request record)")
        leaf = coll[0] if isinstance(coll.type, Array) else coll
        if leaf.type != ptype:
            raise FoldError(f"{spec_label(c.spec)}.{pname}[row {i}]: seed parameter is {leaf.type!r}, the parameter wants {ptype!r}")
        return leaf

    def _weight_field(self, op: WeightField, ptype: Type, what: str) -> Coll:
        """The weights Record's field at `op.path`, whole: the parameter's leaves, no memory read."""
        coll = self.weights_coll
        try:
            for k in op.path:
                coll = coll[k]
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise FoldError(f"{what}: weight field path {op.path!r} is not in the weights Record: {e}") from None
        if coll.type.leaves != ptype.leaves or _width_of(coll.type) != _width_of(ptype):
            raise FoldError(f"{what}: field {op.path!r} is {coll.type!r}, the parameter wants {ptype!r}")
        return Coll(ptype, coll.refs)

    def _witness(self, op: WitnessParam, i: int, ptype: Type, c: Compute, pname: str) -> Coll:
        """Row i's next leaves of its request's `op.name` witness parameter (`WitnessInputs.leaves`)."""
        read = {"step": self.cur_step, "event": min(c.consumed) if c.consumed else self.cur_event, "spec": spec_label(c.spec), "hint": c.hint}
        return Coll(ptype, self.witness_inputs.leaves(op, i, ptype, f"{spec_label(c.spec)}.{pname}[row {i}]", self.req_index, read, FoldError))

    # collectives ------------------------------------------------------------------------------

    def _collective_prepare(self, c: Compute, op: Collective, ptype: Type, pname: str) -> dict:
        """Number the collective (step, k), check its geometry against the parameter type `Array<world, Array<N, Bf16>>`, attribute its rows
        to requests and allocate one `peers_<request>` field per request that owns rows (the OTHER rank's partial rows, in row order).
        Everything here is decided from the rank's own log (StepStart rows / logits rows, RunHeader.notes.tp2); nothing is read from the
        peer -- the peer's value enters only as the root parameter the Program declares."""
        what = f"{spec_label(c.spec)}.{pname}"
        if self.tp is None:
            raise FoldError(f"{what}: a collective ({op.kind} at {op.site!r}) in a log without RunHeader.notes.tp2 (rank/world_size): the fold "
                            f"cannot name the peer; refused")
        rank, world = self.tp["rank"], self.tp["world"]
        if not (isinstance(ptype, Array) and ptype.n == world and isinstance(ptype.elem, Array) and ptype.elem.elem == Value(16)):
            raise FoldError(f"{what}: a Collective operand needs a parameter of type Array<{world}, Array<N, Value<16>>>, got {ptype!r}")
        own = op.own
        M = max(int(own.shape[0]), 1) if op.flat and own.rank == 2 else c.rows        # token rows: one Compute row when flat
        N = int(ptype.elem.n) // M if op.flat else int(ptype.elem.n)
        if own.rank != 2 or tuple(own.shape) != (M, N) or own.dtype != "bfloat16" or tuple(own.strides) != (N, 1) or (op.flat and (c.rows != 1 or M * N != ptype.elem.n)):
            raise FoldError(f"{what}: own partial must be a row-major bf16 [{M}, {N}] view{' (one flat row)' if op.flat else ''}, got {own!r}")
        if op.kind not in (*COLLECTIVE_KINDS.values(), FUSED_COLLECTIVE_KIND):
            raise FoldError(f"{what}: unknown collective kind {op.kind!r}")
        if not op.site:
            raise FoldError(f"{what}: collective without an issuing module (site); refused (fail closed)")
        rec = self._collective_call(op, M, N, min(c.consumed) if c.consumed else self.cur_event, what)
        step, k = rec["step"], rec["k"]
        by_req: dict[int, list[int]] = {}
        for i in range(M):
            rp = self._attribution(c, i)
            if rp is None:
                raise FoldError(f"{what}: collective row {i} of {c.rows} has no request attribution (the row count matches neither the step's "
                                f"{len(self.cur_start.rows) if self.cur_start else '?'} token rows nor its "
                                f"{len(self.cur_start.logits_rows) if self.cur_start else '?'} logits rows)")
            ri = self.req_index.get(rp[0])
            if ri is None:
                raise FoldError(f"{what}: collective row {i} attributed to request {rp[0]!r}, not in the run header")
            by_req.setdefault(ri, []).append(i)
        # one field per (request, PEER rank) in rank order: `s<step>_k<k>` at world 2 (the recorded field names), else
        # `s<step>_k<k>_r<peer>`; the request's field ordinal is k*(world-1)+j = the rank Program's `peers_<ordinal>` input (cb_a.compare)
        peer_ranks = self.tp_peer_ranks
        row_map: dict[int, tuple[int, int, dict[int, int], int, str]] = {}   # row -> (peers param index, {peer rank: field offset}, row within the request, field name)
        fields = []
        for ri, rws in by_req.items():
            pidx = self.peers_param[ri]
            base_of: dict[int, int] = {}
            for pr in peer_ranks:
                base = self.peers_leaves[ri]
                name = f"s{step}_k{k}" if world == 2 else f"s{step}_k{k}_r{pr}"
                fld = {"name": name, "step": step, "k": k, "kind": op.kind, "site": op.site, "rows": len(rws), "N": N, "offset": base, "peer_rank": pr,
                       "type": Array(len(rws), Array(N, Value(16))), "request": self.requests[ri]["request_id"], "ordinal": len(self.peers_fields[ri])}
                self.peers_fields[ri].append(fld)
                self.peers_leaves[ri] += len(rws) * N
                base_of[pr] = base
                fields.append({"request": fld["request"], "param": pidx, "field": name, "ordinal": fld["ordinal"], "rows": len(rws), "offset": base, "peer_rank": pr})
            for j, row in enumerate(rws):
                row_map[row] = (pidx, base_of, j, f"s{step}_k{k}")
        rec["fields"].extend(fields)
        return {"step": step, "k": k, "N": N, "M": M, "rank": rank, "world": world, "kind": op.kind, "site": op.site, "row_map": row_map}

    def _collective_call(self, op: Collective, M: int, N: int, event: int, what: str) -> dict:
        """The record of the call `op` reduces rows of: a fresh (step, k) for a whole partial or a call's first rows, else the open
        call it continues (`Collective.call`: the next rows of the same call, at the same step and site)."""
        own, call = op.own, op.call or op.own
        lo, rem = divmod(own.offset - call.offset, N)
        if call.alloc != own.alloc or call.rank != 2 or call.shape[1] != N or tuple(call.strides) != (N, 1) or rem or not 0 <= lo <= call.shape[0] - M:
            raise FoldError(f"{what}: own partial {own!r} is not a row slice of the call's partial {call!r}")
        held, self._open_call = self._open_call, None
        if lo == 0:
            rec = {"step": self.cur_step, "k": self.cur_coll_k, "kind": op.kind, "site": op.site, "rows": int(call.shape[0]), "N": N,
                   "rank": self.tp["rank"], "world": self.tp["world"], "event": event, "own_view": call, "fields": [],
                   **({"flat": True} if op.flat else {})}
            self.cur_coll_k += 1
            self.collectives.append(rec)
        elif held is not None and held[1] == lo and (held[0]["step"], held[0]["own_view"], held[0]["site"]) == (self.cur_step, call, op.site):
            rec = held[0]
        else:
            raise FoldError(f"{what}: rows {lo}.. of the call {call!r} at {op.site!r} follow no open part of that call")
        if lo + M < call.shape[0]:
            self._open_call = (rec, lo + M)
        return rec

    def _collective_parts(self, prep: dict, op: Collective, i: int, ptype: Type, c: Compute, pname: str) -> Coll:
        """Row i's `parts`: [own row, peer row] in RANK order.  The own row is read from memory (its producer must have written it: a
        collective input nobody computed is Unresolved, exactly as any other operand); the peer row is the request's `peers` parameter
        leaves at the field allocated for this collective -- a declared input of the Program, never a value the fold invents."""
        v = op.own if op.flat else row_view(op.own, i)
        own = self._mem("read", self.mem.read, v)
        self._journal("read", v, own, what=f"collective {op.kind} (step {prep['step']}, k {prep['k']}) own row {i}")
        N, maps = prep["N"], [prep["row_map"][t] for t in (range(prep["M"]) if op.flat else (i,))]
        peer = lambda r: compact_refs([Affine("p", pidx, base_of[r] + j * N, N) for pidx, base_of, j, _ in maps])  # noqa: E731
        parts = [own if r == prep["rank"] else peer(r) for r in range(prep["world"])]   # RANK order over the world
        refs = concat(parts)
        if refs.count != ptype.leaves:
            raise FoldError(f"{spec_label(c.spec)}.{pname}[row {i}]: {refs.count} refs for {ptype!r} ({ptype.leaves} leaves)")
        self._check_width(ptype.elem, v, f"{spec_label(c.spec)}.{pname}[row {i}]")
        return Coll(ptype, refs)

    def _compute_emit(self, c: Compute, rows: list[list[Coll]], root: TreeNode) -> list[tuple[Coll, dict]]:
        # an instance is attributed to the first event its Compute claims, else the root.
        # A primary claims the whole subtree (min = the root) and `also`/`before` members claim nothing, so every
        # pre-existing pattern still stamps `root.id`; a Group part that claims its own nested op (a scope such
        # as `vllm.moe_forward` whose intermediates are only snapshotted under the nested kernels' event ids) is
        # stamped with that op, so `SnapshotStore.find(view, at=event)` (`fold_compare.Values`) reaches its snapshot.
        event = min(c.consumed) if c.consumed else root.id
        outs: list[tuple[Coll, dict]] = []
        for i, colls in enumerate(rows):
            name = self._name(c, i)
            req_pos = self._attribution(c, i)
            out = self.B.call(c.spec, *colls, name=name)
            k = len(self.B.nodes) - 1
            meta = {"step": self.cur_step, "event": event, "resolution": self.cur_resolution, "node": k,
                    "spec": spec_label(c.spec), "row": i, "rows": c.rows, "name": name, "hint": c.hint,
                    # the row's request (and position) when the Compute identifies its rows -- the same rule as
                    # `_name`'s `t=<pos>`; None when the row count identifies nothing (the comparator then attributes by data flow)
                    "request": req_pos[0] if req_pos else None, "position": req_pos[1] if req_pos else None,
                    "pattern": c.pattern, "operand_views": [operand_view(op, i) for op in c.operands],
                    "output_views": []}
            if getattr(c, "roots", ()):
                meta["operand_roots"] = list(c.roots)     # per-operand weight names (multi-weight Computes)
            held = self._coll_prep.get(id(c))
            prep = held[1] if held is not None and held[0] is c else None
            if prep is not None:                          # the row's collective identity + the peers field it reads
                pidx, base, j, fname = prep["row_map"][i]
                meta["collective"] = {"kind": prep["kind"], "step": prep["step"], "k": prep["k"], "site": prep["site"], "rank": prep["rank"],
                                      "world": prep["world"], "N": prep["N"], "peer_param": pidx, "peer_field": fname, "row_in_request": j}
            self.instances.append(meta)
            outs.append((out, meta))
        self._coll_prep.pop(id(c), None)
        return outs

    def _compute_writes(self, c: Compute, outs: list[tuple[Coll, dict]]) -> None:
        _, ret = c.spec.signature()
        comps = ret.fields if isinstance(ret, TupleT) else (ret,)
        if len(c.outputs) != len(comps):
            raise FoldError(f"{spec_label(c.spec)}: {len(c.outputs)} outputs for a return with {len(comps)} components")
        for i, (out, meta) in enumerate(outs):
            for k, op in ((k, op) for k, op in enumerate(c.outputs) if op is not None):   # None: kept in registers, never written
                part = out[k] if isinstance(ret, TupleT) else out
                self._write_operand(op, i, part.refs, part.type, f"{spec_label(c.spec)} output {k} row {i}",
                                    token=(part.type == I32))
                meta["output_views"].append(operand_view(op, i))

    # Transparent move ----------------------------------------------------------------------------

    def _move_reads(self, t: Transparent) -> list[Refs]:
        if t.src is None or t.dst is None:
            raise FoldError(f"move without src/dst ({t.hint})")
        for op in (t.src, t.dst):
            n = operand_rows(op)
            if n is not None and n < t.rows:
                raise FoldError(f"move: {type(op).__name__} covers {n} rows, the move has {t.rows}")
        if t.src.view.bits != t.dst.view.bits:
            raise FoldError(f"move between widths {t.src.view.dtype} -> {t.dst.view.dtype} ({t.hint})")
        return [self._read_operand(t.src, i) for i in range(t.rows)]

    def _move_writes(self, t: Transparent, refs: list[Refs]) -> None:
        for i, r in enumerate(refs):
            self._write_operand(t.dst, i, r, None, f"move row {i} ({t.hint})")

    # memory access ----------------------------------------------------------------------------------

    def _read_operand(self, op: Operand, i: int) -> Refs:
        if isinstance(op, Shared):
            r = self._mem("read", self.mem.read, op.view)
            self._journal("read", op.view, r, what="shared")
            return r
        if isinstance(op, PerRow):
            v = row_view(op.view, i)
            r = self._mem("read", self.mem.read, v)
            self._journal("read", v, r, what=f"row {i}")
            return r
        if isinstance(op, Gather):
            v = row_view(op.view, op.indices[i])
            r = self._mem("read", self.mem.read, v)
            self._journal("read", v, r, what=f"gather row {op.indices[i]}")
            return r
        if isinstance(op, GatherMany):
            idx = list(op.indices[i])
            r = self._mem("read_rows", self.mem.read_rows, op.view, idx)
            self._journal("read", op.view, r, rows=idx, what="gather-many")
            return r
        raise FoldError(f"unknown operand {type(op).__name__}")

    def _write_operand(self, op: Operand, i: int, refs: Refs, t: Type | None, what: str, token: bool = False) -> None:
        if isinstance(op, GatherMany):
            idxs = op.indices[i]
            rows = [row_view(op.view, j) for j in idxs]
            n = rows[0].numel if rows else 0
            if refs.count != n * len(rows):
                raise FoldError(f"{what}: {refs.count} refs for {len(rows)} rows of {n}")
            for j, v in enumerate(rows):
                if t is not None:
                    self._check_width(t, v, what, token)
                part = refs.slice(j * n, (j + 1) * n)
                self._mem("write", self.mem.write, v, part)
                self._journal("write", v, part, what=what)
            return
        view = operand_view(op, i)
        if view is None:
            raise FoldError(f"{what}: a {type(op).__name__} operand is no output")
        if refs.count != view.numel:
            raise FoldError(f"{what}: {refs.count} refs for a view of {view.numel} elements ({view!r})")
        if t is not None:
            self._check_width(t, view, what, token)
        self._journal("write", view, refs, what=what)
        self._mem("write", self.mem.write, view, refs)

    def _wrap(self, ptype: Type, refs: Refs, view: ViewDesc, what: str) -> Coll:
        if refs.count != ptype.leaves:
            raise FoldError(f"{what}: read {refs.count} refs, the parameter type {ptype!r} has {ptype.leaves} leaves")
        self._check_width(ptype, view, what, token=(ptype == I32))
        return Coll(ptype, refs)

    def _check_width(self, t: Type, view: ViewDesc, what: str, token: bool = False) -> None:
        w = _width_of(t)
        if w == view.bits:
            return
        if token and (view.dtype, w) in _TOKEN_WIDTHS:
            return
        raise FoldError(f"{what}: {w}-bit leaves vs a {view.dtype} view {view!r}")

    def _mem(self, what: str, fn, *args):
        try:
            return fn(*args)
        except Unresolved:
            raise
        except (KeyError, ValueError, IndexError) as e:
            raise FoldError(f"memory {what} failed: {type(e).__name__}: {e}") from e

    # naming ----------------------------------------------------------------------------------------

    def _attribution(self, c: Compute, i: int) -> tuple[str, int] | None:
        """(request_id, position) of row `i`: the pattern's attribution, else the step's token rows / logits rows by row count."""
        start = self.cur_start
        if c.attribution and i < len(c.attribution):
            return (str(c.attribution[i][0]), int(c.attribution[i][1]))
        if start is not None and c.rows == len(start.rows):
            return (str(start.rows[i][0]), int(start.rows[i][1]))
        if start is not None and c.rows == len(start.logits_rows):
            return (str(start.logits_rows[i][0]), int(start.logits_rows[i][1]))
        return None

    def _name(self, c: Compute, i: int) -> str:
        rp = self._attribution(c, i)
        pos = rp[1] if rp else None
        layer = _layer_of(c.hint)
        tail = _tail_of(c.hint) or c.spec.base_id
        parts = []
        if pos is not None:
            parts.append(f"t={pos}")
        if layer is not None:
            parts.append(f"layers.{layer}")
        parts.append(tail)
        return "/".join(parts)

    # SampleEnd -----------------------------------------------------------------------------------------

    def _sample_end(self, se: StepEvents) -> None:
        sample, start = se.sample, se.start
        if sample is None:
            if start.logits_rows:
                self._problem("SampleEnd", f"step {start.step} has logits rows but no SampleEnd", {})
            return
        if len(sample.sampled_token_ids) != len(start.logits_rows):
            raise FoldError(f"step {start.step}: {len(sample.sampled_token_ids)} sampled ids for {len(start.logits_rows)} logits rows")
        for j, (req, pos) in enumerate(start.logits_rows):
            try:
                tok = SE.circuit_token_id(sample.sampled_token_ids[j])
            except ValueError as e:
                raise FoldError(f"step {start.step}: sampled row {j} of {req!r}: {e}") from None
            refs: Refs | None = None
            failure: Exception | None = None
            if sample.sampled_view is not None:
                try:
                    refs = self._mem("read", self.mem.read, row_view(sample.sampled_view, j))
                except (Unresolved, FoldError) as e:
                    failure = e
            if refs is None:
                # The sampler's output buffer may already be freed when SampleEnd is logged (the observer orders
                # StorageFree after the step buffer; the last step has no successor): fall back to the step's
                # TokenSelect instance for this logits row, which is what the view held.  A note, not an error.
                refs = self._token_select_output(start.step, j, row_view(sample.sampled_view, j) if sample.sampled_view is not None else None)
                if refs is not None:
                    self.notes.append(f"step {start.step}: sampled row {j} taken from its TokenSelect instance"
                                      + (f" (sampled view unreadable: {failure})" if failure else " (no sampled view)"))
                elif failure is not None:
                    if self.strict:
                        raise failure
                    self._problem(type(failure).__name__, f"step {start.step}: sampled row {j} unresolved: {failure}", {})
            if refs is not None and refs.count != 1:
                raise FoldError(f"step {start.step}: sampled row {j} resolves to {refs.count} refs (expected 1)")
            self.sampled[(req, start.step)] = tok           # (req, step) maps stay complete: StepLink checks look them up
            if refs is not None:
                self.outputs[(req, start.step)] = refs
            # Served or discarded?  vLLM samples a row for every request in the batch, including a request whose prompt is
            # still being prefilled in chunks (or whose prompt + generated tokens are being recomputed in chunks after a
            # preemption); the engine then drops that sample (gpu_model_runner: `discard_request_mask = seq_len < num_tokens`
            # -> `valid_sampled_token_ids[i].clear()`, `num_tokens = len(prompt_token_ids) + len(output_token_ids)`).
            # `StepStart.seq_lens[r]` is that seq_len (tokens in the KV cache after this step) and the fold knows the served
            # samples so far, so the rule is decidable from the log: served <=> seq_lens[r] >= prompt_len_r + served_so_far_r.
            # The discarded sample's TokenSelect instance is real computation and stays in the Program; it is simply not a
            # served token (not returned, not in `sampled`, never fed back).
            served = self._prompt_complete(req, start)
            if served:
                self.sampled_by_request.setdefault(req, []).append((start.step, tok))
                if refs is not None:
                    self.per_request.setdefault(req, []).append((start.step, refs))
                # the served sample IS the token at position pos + 1: register it so a re-fed row at that
                # position (recompute without StepLink) finds its producer.  Two served samples for one position cannot happen
                # under vLLM's rule; if the log says otherwise the fold must not guess.
                nxt = int(pos) + 1
                prev = self.sample_pos.get((req, nxt))
                if prev is not None and prev != start.step:
                    self._problem("SamplePosition", f"step {start.step}: position {nxt} of {req!r} was already produced by the served sample of "
                                                    f"step {prev}; a position has one producer", {"request": req, "position": nxt, "steps": [prev, start.step]})
                else:
                    self.sample_pos[(req, nxt)] = start.step
            else:
                self.discarded_by_request.setdefault(req, []).append((start.step, tok, int(pos)))
                self.notes.append(f"step {start.step}: sampled row {j} of {req!r} (position {pos}, token {tok}) discarded by the engine: "
                                  f"seq_len {start.seq_lens.get(req)} < prompt_len {self._prompt_len(req)} + {self._served_so_far(req)} served "
                                  f"(partial prefill / partial recompute)")

    def _prompt_len(self, req: str) -> int:
        r = self.req_index.get(req)
        return len(self.requests[r]["prompt_token_ids"]) if r is not None else 0

    def _served_so_far(self, req: str) -> int:
        return len(self.sampled_by_request.get(req, ()))

    def _prompt_complete(self, req: str, start: StepStart) -> bool:
        """vLLM's served/discarded rule for the sample of `req` at this step: every token of the request (prompt + the tokens
        generated so far) is in the KV cache after the step — `seq_len >= num_tokens` with `num_tokens = len(prompt) +
        len(output_token_ids)`.  Identical to the former `seq_len >= prompt_len` on every run without preemption (a decode
        step always has seq_len == prompt_len + served); the two differ only for a partial chunk of a post-preemption
        recompute that ends inside the generated tokens, which vLLM discards.  A request the step does not list in `seq_lens`
        (or an unknown request) is treated as served (no evidence to discard)."""
        seq_len = (start.seq_lens or {}).get(req)
        return seq_len is None or int(seq_len) >= self._prompt_len(req) + self._served_so_far(req)

    def _token_select_output(self, step: int, row: int, view: ViewDesc | None = None) -> Refs | None:
        """Fallback when the sampler view is absent/unresolved (`token_select.token_select_output`)."""
        return token_select_output(self.instances, self.cur_start, step, row, view)

    # problems ------------------------------------------------------------------------------------------

    def _problem(self, cls: str, msg: str, detail: dict) -> None:
        if self.strict:
            raise FoldError(f"{cls}: {msg}")
        if len(self.errors) < 4096:
            self.errors.append({"class": cls, "step": self.cur_step, "message": msg[:400], **detail})

    # finish ----------------------------------------------------------------------------------------------

    def _finish(self, with_program: bool) -> FoldResult:
        h = self.header
        assert h is not None and self.B is not None
        outs: list[Coll] = []
        outputs: dict[str, list] = {}
        for rid in self.output_ids:
            seq = self.per_request.get(rid, [])
            outputs[rid] = [refs for _, refs in seq]
            if not seq:
                if self.strict:
                    raise FoldError(f"request {rid!r} produced no sampled tokens")
                self.notes.append(f"request {rid!r} produced no sampled tokens; its return is empty")
            n = len(seq)
            outs.append(Coll(Array(n, I32), concat([refs for _, refs in seq])))
            expect = int(self.requests[self.req_index[rid]].get("max_tokens") or 0)
            if expect and n != expect:
                self.notes.append(f"request {rid!r}: {n} served steps, max_tokens {expect}")
        out = outs[0] if len(outs) == 1 else tuple_of(*outs)
        self.B.ret_type = out.type
        # finalise every `peers_<i>` parameter: the Record of the request's collective fields in issue order (the refs the
        # collective instances hold, ("p", pidx, offset, N), were allocated against exactly these offsets).  A request whose rows met no
        # collective would lower to NO Input node (0 leaves) and shift the weights node: refused by name rather than encoded.
        if self.tp is not None:
            params = list(self.params)
            for ri, pidx in self.peers_param.items():
                flds = self.peers_fields[ri]
                if not flds:
                    msg = (f"tensor-parallel rank log: request {self.requests[ri]['request_id']!r} owns no collective rows; its "
                           f"{params[pidx][0]!r} parameter would be empty (input layout shift); refused")
                    if self.strict:
                        raise FoldError(msg)
                    self._problem("PeersEmpty", msg, {"request": self.requests[ri]["request_id"], "param": params[pidx][0]})
                params[pidx] = (params[pidx][0], Record(*[(f["name"], f["type"]) for f in flds]))
            self.params = tuple(params)
            self.B.params = self.params
        self.params, empty = self.witness_inputs.finalise(self.params, self.requests)
        self.B.params = self.params
        for msg, detail in empty:
            if self.strict:
                raise FoldError(msg)
            self._problem("WitnessEmpty", msg, detail)
        if self.supplied_inputs:
            if not self.supplied_inputs.fields["values"]:
                msg = "registered_values profile: no Definition read a program value; the `values` parameter would be empty"
                if self.strict:
                    raise FoldError(msg)
                self._problem("ValuesEmpty", msg, {})
            self.params = self.B.params = self.supplied_inputs.finalise(self.params)
        body = self.B.finish(out)
        params, ret_type = self.params, out.type
        defn = CompositeDefinition("Captured", 1, ("PROFILE", "WORKLOAD"), lambda S, _p=params, _r=ret_type: (_p, _r), replayer(body),
                                   doc=f"captured from a raw observation log under profile {h.profile_id}",
                                   conformance="derived: capture fold", register=False)
        spec = bind(defn, PROFILE=h.profile_id, WORKLOAD=workload_digest(h.workload))
        spec.set_body(body)
        program = None
        digest = ""
        descriptor = None
        if with_program:
            t0 = time.time()
            program = Program(spec, registered=("values",) if self.supplied_inputs else ())
            self.timings["program_s"] = round(time.time() - t0, 3)
            t0 = time.time()
            descriptor = encode_program(program)          # encoded once: the digest's input AND program.json's `program`
            digest = program_digest(descriptor)
            self.timings["digest_s"] = round(time.time() - t0, 3)
        self.report.steps = len(self.steps)
        try:
            st = self.mem.stats()
            mem = {k: getattr(st, k) for k in ("allocations", "live_allocations", "writes", "reads", "moves", "intervals")}
            mem["extra"] = dict(getattr(st, "extra", {}) or {})
        except Exception as e:  # noqa: BLE001 - diagnostics only
            mem = {"error": f"{type(e).__name__}: {e}"}
        mem.update({"storage_allocs": self.n_alloc, "storage_frees": self.n_free})
        sampled = {rid: list(v) for rid, v in self.sampled_by_request.items()}
        discarded = {rid: list(v) for rid, v in self.discarded_by_request.items()}
        return FoldResult(program=program, spec=spec, report=self.report, digest=digest, outputs=outputs, instances=self.instances,
                          memory=mem, notes=self.notes, body=body, steps=self.steps, errors=self.errors, timings=self.timings,
                          sampled=sampled, discarded=discarded, refed=list(self.refed), weights_type=self.weights_type, params=self.params,
                          requests=list(self.requests), accesses=self.accesses, tp=dict(self.tp) if self.tp else None,
                          collectives=list(self.collectives), peers_layout=self._peers_layout(),
                          witness_layout=self.witness_inputs.layout(self.params, self.requests),
                          descriptor=descriptor, supplied=self.supplied_inputs.layout(self.params, self.requests, program))

    def _peers_layout(self) -> list[dict]:
        """The `peers_<i>` parameters as the record states them: parameter index, request, and every field (name, step, k, kind,
        site, rows, N, leaf offset) in issue order -- what `program.json` carries so a checker (cb_a.compare, the cross-rank check, root
        policy) can locate every peer leaf range without the profile's patterns."""
        out = []
        for ri, pidx in self.peers_param.items():
            out.append({"param": pidx, "name": self.params[pidx][0] if pidx < len(self.params) else None, "request": self.requests[ri]["request_id"],
                        "request_index": ri, "leaves": self.peers_leaves[ri],
                        "fields": [{k: v for k, v in f.items() if k != "type"} for f in self.peers_fields[ri]]})
        return out


def instance_nodes(result: FoldResult, *, spec_prefix: str | None = None, step: int | None = None, layer: int | None = None,
                   row: int | None = None) -> list[dict]:
    """Navigation: the instance records matching the filters (spec id prefix, step, layer from the hint, row)."""
    out = []
    for m in result.instances:
        if spec_prefix is not None and not m["spec"].startswith(spec_prefix):
            continue
        if step is not None and m["step"] != step:
            continue
        if layer is not None and _layer_of(m["hint"]) != layer:
            continue
        if row is not None and m["row"] != row:
            continue
        out.append(m)
    return out

