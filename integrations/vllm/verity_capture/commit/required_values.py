"""RequiredValues(P, Q) -- the deterministic, lazy traversal of a Program `P` under a declared query `Q` that names every required Value a
Commit must bind (BRIEF R16 §7A; design handoff `out/gen/r15/REQUIRED-VALUES-HANDOFF.md`).  The required-value manifest of `required_manifest.py` is a
CACHED SERIALIZATION of this traversal (DECISIONS-2/-3: RequiredValues(P,Q) is the function; the required-value manifest is its cache).

Identity key (unchanged, TABLE-RULES r14-v1):

    (program_digest, request_id, step, invocation, op_path, output_member, element_range, rank)

What is new here, relative to `required_manifest.build_manifest` (the R14/R15 hand-maintained per-family inventory):

* **lazy** -- `Traversal` is a generator over the Program's Calls (the `instances.json.gz` rows) in Program order: Calls are consumed
  once, identities are yielded
  per engine step, and the per-step state is O(modules) (claim groups keyed by (module, family, hint, spec), attention / MoE geometry
  accumulators), never the identity list of the whole Program.  Program-wide state is three int arrays (module id, family id, boundary
  class per Call) that the dataflow audit needs; `iter_rows()` streams `instances.json.gz` without loading the Call list.
* **boundary derived from the Program's own dataflow** -- every Call's `args` (operand runs `["n", producer, lo, hi]`) say which producers
  are read OUTSIDE their own module.  A Call whose output crosses a module boundary is a boundary value of Q_module_body_v1; one consumed
  only inside its module is interior.  The traversal reports, per (family, vocabulary class), how many Calls cross vs how many the vocabulary NAMES
  (`boundary_audit`): a family that crosses and is named by nothing (not a declared aggregate such as the MoE block stream) is an
  `unnamed_crossing` -- the inventory-omission class the old code could not see; a named family that never crosses is `named_interior`.
  Programs without args (synthetic fixtures) report `audit.available: false`.  Tonight the audit is a REPORT beside the population, not a
  change of the population (a semantic change of the required set is a DECISIONS-PENDING item, never a silent edit).
* **nested calls** -- `Workload` (GP-01 `Workload_v1{requests, shared}`) is traversed by recursing into each request component with its
  own (request_id, arrival) context; the shared prescribed inputs (weights) are yielded ONCE, the attention hidden launch once per layer
  per engine step over the co-scheduled requests (buffered per engine step, flushed as components arrive).
* **fail closed** -- a family without a naming rule raises `UnmodelledFamily` from the vocabulary; the traversal records it and the
  population is `complete: False` (`required_manifest.coverage_check` refuses).  A zero-width Value is never yielded (MAN-02).  The
  audit adds the second gate: a boundary-crossing family named by no rule is recorded in `unnamed_crossing`.

The NAMING vocabulary (which output Values a family's Call produces, in the collector's `op_path/member` occurrence keys) is `required_manifest.
members_for` and the geometry helpers -- shared on purpose: this module owns the traversal (what runs, per step, per request, per rank,
crossing which boundary); the vocabulary is the bridge to the collector's names and is consumed, not re-invented.

    PYTHONPATH=. python -m verity_capture.commit.required_values audit --program <derive dir> [--workload wl.json]
    PYTHONPATH=. python -m verity_capture.commit.required_values diff  --program <derive dir> [--workload wl.json]   # old vs new

Dependency-free (json/gzip/hashlib/array) like `required_manifest`.

PROTOCOL SUMMARY (Commit stage: the population function behind the required-value manifest of record)
  Purpose: RequiredValues(P, Q) -- walk the Build's Program `P` (its Calls: the instances.json.gz rows, in Program order) under the
    verification query `Q` (Q_module_body_v1) and yield every required Value a Commit must bind, per engine step / request / rank;
    the DEFAULT engine of required_manifest.
  Inputs: a derive dir (descriptor.json.gz + instances.json.gz; a Workload_v1 root recurses into its request components), the
    workload manifest (request ids, arrivals, caps), the naming vocabulary `required_manifest.members_for`.
  Outputs: identity rows in the TABLE-RULES key (program_digest, request_id, step, invocation, op_path, output_member,
    element_range, rank) consumed by required_manifest.build_manifest; `boundary_audit` (unnamed_crossing / named_interior classes
    per family); the CLI `audit` / `diff` documents (inventory engine vs traversal).
  Invariants: Calls are consumed once (per-step state is O(modules), never the whole identity list); a Call's boundary class is derived
    from its own `args` (a producer read outside its module crosses), never declared by hand; a family with no naming rule raises
    UnmodelledFamily -> the population is `complete: False` (fail closed); shared prescribed inputs (weights) are yielded ONCE;
    the audit is a REPORT beside the population -- it never changes the required set.
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""
from __future__ import annotations

import gzip
import json
import os
import re
from array import array
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, NamedTuple

from . import required_manifest as RM
# members_for stays while this traversal builds v1 manifests: TP rank Programs (until TP=N) and Lifted[X] Programs (the v2 query has no unlift)
from .required_manifest import (EPILOGUE_FAMILIES, IDENTITY_KEY, INTERIOR_CROSSING, MOE_LEAF, PROMOTED_MEMBER, UnmodelledFamily, _AGGREGATE_PREFIXES,
                                _STRUCTURE_PREFIXES, _fa2_geometry, _fa3_geometry, _layer_of, _REQ_RE, collector_path, is_aggregate_family,
                                is_stochastic_select, is_structure_family, is_token_select, mat_total_words, members_for, moe_identities,
                                promote_interior_crossings, promoted_annotations, promotion_header, query_requirements, request_label, spec_family,
                                spec_statics, split_name, wrapper_prefix_of)

SCHEMA = "verity-sweep/required-values/v1"

#: boundary classes of a Program row, as the traversal decides them at the step flush; [R19 int-20] `CLASS_INTERIOR_CROSSING` is decided
#: AFTER the walk (a consumer may be in a later step): an interior row whose output another module reads, PROMOTED to a required member
(CLASS_UNSEEN, CLASS_BOUNDARY, CLASS_INTERIOR, CLASS_AGGREGATE, CLASS_DECLARED, CLASS_SAMPLER, CLASS_STRUCTURE, CLASS_UNMODELLED, CLASS_GUARD,
 CLASS_INTERIOR_CROSSING) = range(10)
#: the Program params the vocabulary names (RequiredValues carries a value for each), BY RULE of who reads them and which family carries
#: the value -- never by the param's name alone [R17 revb F-r17b-08: the step wrapper calls the token input `input_ids`, the request wrapper
#: `prompt`; a name whitelist flagged the former as unnamed while the population carried it]: the token ids -> `input_token_ids` (every
#: reader is an Embedding row, or a lifted guard row whose value the lifted Embedding row names); the weights param `_weights()` enumerated
#: fields from -> `weights`; `seed` -> `sampler_seed` (every reader a stochastic selector, R15 stoch); `splits` -> `sampler_splits` WHEN
#: the family is emitted (REQ-03).  Any other param a row reads -- or one of these read by a row outside its rule -- is an `unnamed_input`
#: (audit + incomplete population).  `NAMED_INPUTS` is kept as the documented vocabulary of param names, not as the decision.
NAMED_INPUTS = frozenset({"prompt", "input_ids", "weights", "seed"})
#: [R17 finding 7] the collective families of a rank Program (derive_step --tp): their peer operand is a `peers_<k>` Program input
_COLLECTIVE_PREFIXES = ("AllReduce2", "AllGather2")

CLASS_NAMES = {CLASS_UNSEEN: "unseen", CLASS_BOUNDARY: "boundary", CLASS_INTERIOR: "interior", CLASS_AGGREGATE: "aggregate",
               CLASS_DECLARED: "declared", CLASS_SAMPLER: "sampler", CLASS_STRUCTURE: "structure", CLASS_UNMODELLED: "unmodelled", CLASS_GUARD: "guard",
               CLASS_INTERIOR_CROSSING: INTERIOR_CROSSING}
#: (`_AGGREGATE_PREFIXES` / `_STRUCTURE_PREFIXES`: the vocabulary's, imported -- the promotion pass and the classifier read one table)
#: families the query DECLARES required although the Program's dataflow keeps them inside their module (TP-04 rank partials feed the
#: module's own collective; DECISION M-0248): named by the vocabulary, not a boundary crossing
_DECLARED_MEMBER_PREFIXES = ("part_rank", "shard_rank")
#: [R16 pad M-0069 / reqval M-0050] the bot-lifted Program (padded baseline, experimental): every Definition is re-registered as
#: `Lifted[<id>]_v1{same statics}` (verity_vllm/registry/lifted.py) and the step guard is three hand-authored relations -- `LStepLive<E>`
#: (`s=<s>/live`: Active iff the previous committed token is an Active non-EOS), `LSelect<w>` (`s=<s>/tok_in`: the step's guarded INPUT
#: token, bot when inactive) and the `LBot<w>` constant.  RULE: RequiredValues(Lift(P), Q) has the SAME identity keys as RequiredValues(P, Q):
#: the lifted family is the wrapped family (same members, same element counts -- a lifted word is one value of w+1 bits, the count is
#: unchanged), the guard rows carry no identity (the guarded input token is the value the lifted Embedding row names as
#: `input_token_ids`; `live` is derived from committed tokens and never committed itself; the constant is structure).  Guard rows in an
#: UN-lifted Program are unmodelled (fail closed): the guard is only meaningful as part of a lift.
_LIFT_RE = re.compile(r"^Lifted\[(?P<inner>[^\]]+)\]_v\d+(?P<rest>\{.*)?$", re.S)
#: R17-32 (pad, Continuation_v1 -- CONTINUATION-SPEC v0.1): the controller family `LIsEOS<E>` (`slot=<t>/is_eos`), `LEmitNext[H]`
#: (`slot=<t>/emit`) and the designated output `LSelect<w>` (`slot=<t>/out`) join the guard rows: computed controller bits of the lifted
#: Program, carrying no identity under THIS rule (so RequiredValues(Lift(P), Q) == RequiredValues(P, Q) row for row still holds).  revb
#: S-r17b-16 R4 (the bits as NAMED families / replay strata) is DECIDED by reqval (M-0616, Q-EXPANSION 4b): "controller uncovered BY NAME" --
#: the manifest header's `query.guard_families` counts rows per family from THIS tuple, `population.controller` names them with their
#: mechanism (the lifted record's R8 controller replay + the executed prefix from r_t); identities for the bits would be Q_module_v2.
GUARD_PREFIXES = ("LStepLive", "LSelect", "LBot", "LIsEOS", "LEmitNext")


#: [pad register_hand_lift b79e9623d; moe MOE-PAD-SPEC route A (2)] lifts registered BY HAND carry their own family name, not `Lifted[..]`:
#: the lifted row of `MoeFfnFixed_v1` is `LMoeFfn_v1{same statics}` (and `MoeFfnFixedL_v1` -> `LMoeFfnL_v1`).  Same population rule as
#: `Lifted[..]` (Lift(Q): ids modulo the lift, M-0954): the row is counted as a lifted row and modelled as its unlifted twin.
HAND_LIFTS: dict[str, str] = {"LMoeFfnL_v1": "MoeFfnFixedL_v1", "LMoeFfn_v1": "MoeFfnFixed_v1"}
_HAND_LIFT_RE = re.compile(r"^(?P<fam>" + "|".join(re.escape(k) for k in HAND_LIFTS) + r")(?P<rest>\{.*)?$", re.S)


def unlift(spec: str) -> tuple[str, bool]:
    """`Lifted[Gemm_v1]_v1{K=576,N=960}` -> ("Gemm_v1{K=576,N=960}", True); a registered hand lift `LMoeFfnL_v1{..}` -> ("MoeFfnFixedL_v1{..}",
    True); anything else unchanged, False."""
    m = _LIFT_RE.match(spec)
    if m:
        return m.group("inner") + (m.group("rest") or ""), True
    h = _HAND_LIFT_RE.match(spec)
    if h:
        return HAND_LIFTS[h.group("fam")] + (h.group("rest") or ""), True
    return spec, False


def is_guard(family: str) -> bool:
    return family.startswith(GUARD_PREFIXES)


# ---------------------------------------------------------------------------------------------------------------------------------
# Program view / query / identity
# ---------------------------------------------------------------------------------------------------------------------------------

def _decode_rows_streaming(fh) -> Iterator[dict]:
    """Yield the objects of the top-level `"rows": [...]` array of an instance-sequence JSON stream without loading the array."""
    dec = json.JSONDecoder()
    buf = fh.read(1 << 16)
    k = buf.find('"rows"')
    while k < 0:
        more = fh.read(1 << 16)
        if not more:
            raise ValueError("instance sequence without a rows array")
        buf += more
        k = buf.find('"rows"')
    buf = buf[k + len('"rows"'):]
    k = buf.find("[")
    while k < 0:
        more = fh.read(1 << 16)
        if not more:
            raise ValueError("instance sequence rows array not opened")
        buf += more
        k = buf.find("[")
    buf = buf[k + 1:]
    while True:
        buf = buf.lstrip(" \n\r\t,")
        if buf.startswith("]"):
            return
        try:
            obj, end = dec.raw_decode(buf)
        except json.JSONDecodeError:
            more = fh.read(1 << 16)
            if not more:
                raise
            buf += more
            continue
        yield obj
        buf = buf[end:]


def iter_rows(program_dir: str) -> Iterator[dict]:
    """Stream the instance rows of a derive_step output dir (never materializes the `rows` list)."""
    with gzip.open(os.path.join(program_dir, "instances.json.gz"), "rt") as f:
        yield from _decode_rows_streaming(f)


def _read_header(program_dir: str) -> dict[str, Any]:
    """`instances.json.gz` minus its rows (schema, digest, params): the text before `"rows"`, closed."""
    with gzip.open(os.path.join(program_dir, "instances.json.gz"), "rt") as f:
        buf = f.read(1 << 16)
        k = buf.find('"rows"')
        while k < 0:
            more = f.read(1 << 16)
            if not more:
                break
            buf += more
            k = buf.find('"rows"')
    head = buf[:k].rstrip().rstrip(",") + "}" if k >= 0 else buf
    try:
        return json.loads(head)
    except json.JSONDecodeError:
        return {}


@dataclass
class ProgramView:
    """What the traversal reads of one (request or step) Program: the Build's `result.json` metadata, the parameter list, the rows."""
    digest: str
    params: list[tuple[str, str]]
    result: dict[str, Any]
    rows: Iterable[dict]
    dir: str | None = None
    instances_digest: str | None = None
    artifact: dict[str, Any] | None = None
    n_rows: int | None = None

    @staticmethod
    def of(program: dict[str, Any]) -> "ProgramView":
        """From `required_manifest.load_program`'s dict (rows in memory)."""
        inst, res = program["instances"], program["result"]
        pdig = (program.get("artifact") or {}).get("program_digest") or res["program"]["digest"]
        return ProgramView(digest=str(pdig), params=[tuple(p) for p in (inst.get("params") or [])], result=res, rows=inst["rows"],
                           dir=program.get("dir"), instances_digest=inst.get("digest"), artifact=program.get("artifact"), n_rows=len(inst["rows"]))

    @staticmethod
    def streaming(program_dir: str) -> "ProgramView":
        """From a derive_step output dir, rows streamed (`iter_rows`)."""
        res = json.load(open(os.path.join(program_dir, "result.json")))
        art = None
        if os.path.exists(os.path.join(program_dir, "artifact.json")):
            art = json.load(open(os.path.join(program_dir, "artifact.json")))
        hdr = _read_header(program_dir)
        pdig = (art or {}).get("program_digest") or res["program"]["digest"]
        return ProgramView(digest=str(pdig), params=[tuple(p) for p in (hdr.get("params") or [])], result=res, rows=iter_rows(program_dir),
                           dir=program_dir, instances_digest=hdr.get("digest"), artifact=art)


@dataclass(frozen=True)
class Query:
    """The declared query `Q`: which request / arrival / rank the Program serves and which VU boundary is selected."""
    request_id: str = "r0"
    arrive_step: int = 0
    rank: int = 0
    ranks: int = 1
    profile: str | None = None          # ampere | hopper | None = from the Program's target / FA version
    shared_rope: bool = True
    boundary: str = "module"            # the VU boundary the query selects (module outputs in the implementation's named_modules vocabulary)
    audit: bool = True                  # derive the dataflow boundary audit from the rows' args


class Identity(NamedTuple):
    family: str
    program_digest: str
    request_id: str
    step: int | str
    invocation: int
    op_path: str
    output_member: str
    lo: int
    hi: int | None
    rank: int
    ann: dict

    def key(self) -> tuple:
        return (self.program_digest, self.request_id, self.step, self.invocation, self.op_path, self.output_member, (self.lo, self.hi), self.rank)

    def row(self) -> dict[str, Any]:
        """The manifest row (the cached serialization's unit): key fields + `family` + annotations."""
        d = {"family": self.family, "program_digest": self.program_digest, "request_id": self.request_id, "step": self.step, "invocation": self.invocation,
             "op_path": self.op_path, "output_member": self.output_member, "element_range": [self.lo, self.hi], "rank": self.rank}
        d.update(self.ann)
        return d


# ---------------------------------------------------------------------------------------------------------------------------------
# the streaming traversal
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class _Claim:
    """One (module, family, hint, spec) claim group of a step: how many rows, which fx nodes (ordered, unique), row indices."""
    fam: str
    mod: str
    hint: str
    spec: str
    statics: dict
    rows: int = 0
    nodes: list = field(default_factory=list)
    node_set: set = field(default_factory=set)
    idx: array = field(default_factory=lambda: array("I"))
    ordinals: Counter = field(default_factory=Counter)          # (program_ordinal, program_operand, why) -> rows  ([M-0323] families only)


class _Step:
    """Per-step accumulators (O(modules))."""

    def __init__(self, step: int):
        self.step = step
        self.claims: dict[tuple, _Claim] = {}
        self.epi_mods: dict[str, int] = {}
        self.coll_of: dict[str, str] = {}
        self.attn: dict[str, dict[str, Any]] = {}
        self.moe: dict[str, dict[str, Any]] = {}
        self.softcap: dict[str, str] = {}
        self.fam_pos: dict[tuple, int] = {}
        self.rows_under: Counter = Counter()
        self.selects: list[tuple[int, str]] = []                    # (row index, family) of the sampling event(s)


def _note_bn_disagreement(into: dict[str, dict[str, Any]], stream: str, program_bn: set[int], table_bn: int, module: str | None) -> None:
    d = into.setdefault(stream, {"program_BN": [], "table_BN": [], "steps": 0, "attention_module": module})
    d["steps"] += 1
    d["program_BN"] = sorted(set(d["program_BN"]) | set(program_bn))
    d["table_BN"] = sorted(set(d["table_BN"]) | {int(table_bn)})


class Traversal:
    """`Traversal(program, query)` iterates the required identities lazily; after exhaustion the summary attributes are set
    (`unmodelled`, `aliases`, `steps`, `query_info`, `boundary_audit`, `moe_layers`, `stochastic_selectors`, `populations`)."""

    def __init__(self, program: ProgramView | dict[str, Any], query: Query | None = None):
        self.p = program if isinstance(program, ProgramView) else ProgramView.of(program)
        self.q = query or Query()
        res = self.p.result
        self.pdig = self.p.digest
        self.attn_obs = res.get("attention_impls_observed") or {}
        self.tgt = (res.get("target") or {}).get("declared") or {}
        self.fa = None
        for v in self.attn_obs.values():
            self.fa = v.get("vllm_flash_attn_version")
            break
        profile = self.q.profile
        if profile is None:
            cc = self.tgt.get("compute_capability") or [8, 9]
            profile = "hopper" if (self.fa == 3 or (self.fa is None and int(cc[0]) >= 9)) else "ampere"
        self.profile = profile
        self.query_info = query_requirements(profile)
        self.hidden_family = self.query_info["attention_hidden_family"]
        if self.q.boundary == "module-pad":                                      # [revb S-r17b-51 (b)] Q_module_pad_v1: the declared padded-boundary
            RM.query_requirements_pad(self.query_info)                             # query -- Q_module_v1 + the slot units of padded MoE rows
        elif self.q.boundary != "module":
            raise ValueError(f"Query.boundary {self.q.boundary!r}: 'module' (Q_module_v1) or 'module-pad' (Q_module_pad_v1) in this cut")
        tpinfo = res.get("tp") if isinstance(res.get("tp"), dict) else {}
        self.world = int(tpinfo.get("world") or 1)
        self.prog_rank = int(tpinfo.get("rank") or 0)
        if self.world > 1 or self.q.ranks > 1:
            if int(self.q.ranks) != self.world or int(self.q.rank) != self.prog_rank:
                raise ValueError(f"rank Program mismatch: Program is rank {self.prog_rank} of world {self.world}, manifest requested rank {self.q.rank} of {self.q.ranks}")
            self.query_info["required_families"].append("tp_rank_partials")
            self.query_info["why"]["tp_rank_partials"] = (
                f"TP world {self.world}: every collective of rank {self.prog_rank} consumes this rank's OWN partial (row-parallel Gemm on the "
                "K/W weight shard, masked embedding shard, lm_head logits shard) -> the partial is required per rank IN ADDITION to the "
                "collective output the module binds (AllReduce2/AllGather2 of both ranks' partials; TP-04 conservative rule)")
            self.query_info["tp"] = {"world": self.world, "rank": self.prog_rank, "rule": "TP-04: partials AND collective outputs both required, per rank"}
        self.rank = int(self.q.rank)
        self.request_id = request_label(self.q.request_id)
        self.arrive = int(self.q.arrive_step)
        self.n_seq = 1
        # summary (filled while iterating)
        self.unmodelled: dict[str, int] = {}
        self.aliases: dict[str, list] = {}
        self.fam_counts: Counter = Counter()
        self.steps = 0
        self.moe_layers: set[int] = set()
        self.stochastic_selectors: set[str] = set()
        self.splits_param = RM.has_splits_param(self.p.params)                  # [R16 REQ-03] the derived-S request-owned `splits` Input
        self.n_identities = 0
        self.prefix: str | None = None
        self.hidden_layers: set[str] = set()
        self.hidden_geo: dict[str, dict[int, dict[str, Any]]] = {}
        self.hidden_bn_disagreement: dict[str, dict[str, Any]] = {}             # stream -> {program_BN, table_BN, steps} (audit; see _flush)
        self.hidden_program_bn: dict[str, dict[int, list[int]]] = {}           # stream -> step -> the BN the Program's attention rows carry
        self.in_fold = False                                                     # set by WorkloadTraversal: the launch is the fold's, not this component's
        self.boundary_audit: dict[str, Any] = {"available": False}
        self.weight_names: list[str] = []
        # program-wide compact arrays (dataflow audit): module id / family id / class per row
        self._mod_ids: dict[str, int] = {}
        self._fam_ids: dict[str, int] = {}
        self._mod_names: list[str] = []
        self._fam_names: list[str] = []
        self._mod_of = array("i")
        self._fam_of = array("i")
        self._class_of = bytearray()
        self._crosses = bytearray()
        # [R19 int-20] the CONSUMER modules of a crossing row (module id of the first reader per row; the rare further distinct readers
        # beside) -- the promoted member names them, the collector observes the value at one of them
        self._xfirst = array("i")
        self._xmore: dict[int, list[int]] = {}
        # the claims the vocabulary called interior (step, module, family, spec, statics, row indices) and the vocabulary's claims / alias
        # sites per step: the promotion pass after the walk (`_promote`) decides which interior rows crossed
        self._interior: list[tuple] = []
        self._named_by_step: dict[int, set[tuple[str, str]]] = {}
        self._aliased_by_step: dict[int, set[str]] = {}
        self.promotion: dict[str, Any] | None = None
        self._args_seen = False
        self._unwired = 0                    # rows without operand runs (a synthetic / partially wired Program): a missing edge can hide a crossing, never invent one
        self._param_readers: dict[int, dict[str, int]] = {}   # Program param index -> {family: rows reading it} (the inputs the Program consumes)
        self._weights_param: int | None = None                # the param `_weights()` enumerated the weight fields from (named by `weights`)
        self.lifted_rows = 0                 # `Lifted[..]` rows seen (the bot-lifted Program); the guard rows are only accepted alongside them
        self.padded_moe_rows = 0             # MoeFfnFixedL_v1 / MoeFfnFixed_v1 rows seen (the padded fixed-slot MoE construction, MOE-PAD-SPEC)
        self.guard_rows = 0
        self.guard_families: Counter = Counter()   # controller / guard rows per family prefix (LStepLive, LSelect, LBot, LIsEOS, LEmitNext) -- R4
        self._done = False

    # ---- ids ----------------------------------------------------------------------------------------------------------------------
    def _mid(self, m: str) -> int:
        k = self._mod_ids.get(m)
        if k is None:
            k = self._mod_ids[m] = len(self._mod_names)
            self._mod_names.append(m)
        return k

    def _fid(self, f: str) -> int:
        k = self._fam_ids.get(f)
        if k is None:
            k = self._fam_ids[f] = len(self._fam_names)
            self._fam_names.append(f)
        return k

    def _um(self, key: str, n: int = 1) -> None:
        self.unmodelled[key] = self.unmodelled.get(key, 0) + n

    def _mk(self, family: str, step: int | str, invocation: int, op_path: str, member: str, lo: int, hi: int | None, **extra) -> Identity:
        ann: dict[str, Any] = {}
        if step != "*":
            ann["engine_step"] = self.arrive + int(step)
        ann.update(extra)
        self.fam_counts[family] += 1
        self.n_identities += 1
        return Identity(family, self.pdig, self.request_id, step, invocation, op_path, member, lo, hi, self.rank, ann)

    # ---- the walk -----------------------------------------------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Identity]:
        if self._done:
            raise RuntimeError("a Traversal iterates once; build a new one")
        rows_it = iter(self.p.rows)
        # the export wrapper prefix is determined from the attention rows (never assumed): peek the first attention row, replaying the
        # buffered prefix afterwards -- O(rows until the first attention row) once, the rest streams
        head: list[dict] = []
        prefix = None
        for r in rows_it:
            head.append(r)
            mod, _n, _h, _row = split_name(r.get("name") or "")
            if mod and spec_family(unlift(r["spec"])[0]).startswith("Attention"):
                prefix = wrapper_prefix_of([r], self.attn_obs.keys())
                break
        if prefix is None:
            prefix = wrapper_prefix_of(head, self.attn_obs.keys())
        self.prefix = prefix
        yield from self._weights()
        st = _Step(0)
        i = 0
        for r in _chain(head, rows_it):
            if r.get("i") is not None and int(r["i"]) != i:
                raise ValueError(f"instance rows are not in Program order (row index {r.get('i')} at position {i})")
            spec, lifted = unlift(r["spec"])
            if lifted:
                r = dict(r, spec=spec)                                             # a copy: the caller's rows are never rewritten
                self.lifted_rows += 1
            fam = spec_family(spec)
            if fam.startswith(("MoeFfnFixedL", "MoeFfnFixed")):
                self.padded_moe_rows += 1
            mod, node, hint, rownum = split_name(r.get("name") or "")
            self._mod_of.append(self._mid(mod))
            self._fam_of.append(self._fid(fam))
            self._class_of.append(CLASS_UNSEEN)
            self._crosses.append(0)
            self._xfirst.append(-1)
            if self.q.audit:
                self._audit_args(r, i)
            if is_guard(fam):
                self.guard_rows += 1
                self.guard_families[next(px for px in GUARD_PREFIXES if fam.startswith(px))] += 1   # [revb S-r17b-16 R4] named per controller family
                self._class_of[i] = CLASS_GUARD
                i += 1
                continue
            if is_token_select(fam):
                st.selects.append((i, fam))
                yield from self._flush(st)
                st = _Step(st.step + 1)
                i += 1
                continue
            self._absorb(st, r, i, fam, mod, node, hint, rownum)
            i += 1
        if st.claims or st.attn or st.moe:
            yield from self._flush(st)                                            # trailing instances without a sampling event (padded wrapper)
        yield from self._promote()                                                # [int-20] deferred: every consumer of every row has been walked
        self._finish()
        self._done = True

    def _audit_args(self, r: dict, i: int) -> None:
        args = r.get("args")
        if not args:
            if not spec_family(r["spec"]).startswith(_STRUCTURE_PREFIXES + ("LBot",)):
                self._unwired += 1
            return
        self._args_seen = True
        me = self._mod_of[i]
        fam_i: str | None = None
        for operand in args:
            for ref in operand:
                try:
                    kind = ref[0]
                    j = int(ref[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if kind == "p":                                                    # a Program INPUT read by this row (prompt / seed / weights / ...)
                    if fam_i is None:
                        fam_i = spec_family(r["spec"])
                    d = self._param_readers.setdefault(j, {})
                    d[fam_i] = d.get(fam_i, 0) + 1
                    continue
                if kind != "n":
                    continue
                if 0 <= j < i and self._mod_of[j] != me:
                    self._crosses[j] = 1
                    first = self._xfirst[j]                                        # [int-20] who reads it, Program order (the promoted `consumers`)
                    if first < 0:
                        self._xfirst[j] = me
                    elif first != me:
                        more = self._xmore.setdefault(j, [])
                        if me not in more:
                            more.append(me)

    def _consumers_of(self, i: int) -> list[str]:
        """The module paths reading row i's output from outside its module, Program order (empty when it does not cross)."""
        first = self._xfirst[i]
        if first < 0:
            return []
        return [self._mod_names[first]] + [self._mod_names[m] for m in self._xmore.get(i, ())]

    def _weights(self) -> Iterator[Identity]:
        wparam = next((t for n, t in self.p.params if n == "weights"), None)
        self._weights_param = next((j for j, (n, _t) in enumerate(self.p.params or []) if str(n) == "weights"), None)
        wnames: list[str] = []
        wwidth: dict[str, int] = {}
        if wparam and str(wparam).startswith("{"):
            wnames = re.findall(r"([A-Za-z0-9_.]+):\s*(?:Array|Value)", wparam)
            for m in re.finditer(r"([A-Za-z0-9_.]+):\s*((?:Array<\d+,)*Value<(\d+)>)", wparam):
                wwidth[m.group(1)] = int(m.group(3))
        if not wnames:
            wnames = list(((self.p.result.get("weights") or {}).get("roles") or {}).keys())
        self.weight_names = sorted(wnames)
        for w in self.weight_names:
            yield self._mk("weights", "*", 0, collector_path(w, self.prefix or ""), "weight", 0, None,
                           dtype={8: "w8", 32: "f32", 16: "bf16"}.get(wwidth.get(w, 16), f"w{wwidth.get(w)}"))

    def _absorb(self, st: _Step, r: dict, i: int, fam: str, mod: str, node: str, hint: str, rownum: int | None) -> None:
        stx = spec_statics(r["spec"])
        st.rows_under[mod] += 1
        if fam.startswith(EPILOGUE_FAMILIES):
            st.epi_mods[mod] = int(stx.get("N", 0))
        if self.world > 1:
            if fam.startswith("AllReduce2"):
                st.coll_of[mod] = "all_reduce"
            elif fam.startswith("AllGather2"):
                st.coll_of[mod] = "all_gather"
        if fam.startswith(("Bf16DivScalar", "Bf16Tanh", "Bf16MulScalar_")) and not hint:
            st.softcap.setdefault(mod, fam)
        if fam.startswith(("Attention", "AttentionHead")):
            a = st.attn.setdefault(mod, {"rows": 0, "T_max": 0, "NH": int(stx.get("NH", 0)), "KVH": int(stx.get("KVH", stx.get("NH", 0))), "D": int(stx.get("D", 0)),
                                         "program_BN": set()})
            a["rows"] += 1
            a["T_max"] = max(a["T_max"], int(stx.get("T", 0)))
            if "BN" in stx:                                                      # the Program's own block size (Attention_v2/_v3{..,BN,..})
                a["program_BN"].add(int(stx["BN"]))
        if fam.startswith(("MoeRouterTopK", "MoeExpertGemm", "MoeSum")):
            g = st.moe.setdefault(mod, {"M": 0, "TOPK": 0, "E": 0, "H": 0, "I": 0, "N_up": 0, "expert_gemm_rows": 0})
            if fam.startswith("MoeRouterTopK"):
                g["M"] += 1
                g["TOPK"] = int(stx.get("TOPK", g["TOPK"]))
                g["E"] = int(stx.get("E", g["E"]))
            elif fam.startswith("MoeExpertGemmW"):
                g["H"] = int(stx.get("N", g["H"]))
                g["I"] = int(stx.get("K", g["I"]))
            elif fam.startswith("MoeExpertGemm"):
                g["N_up"] = int(stx.get("N", g["N_up"]))
                g["expert_gemm_rows"] += 1
            else:
                g["H"] = int(stx.get("H", g["H"]))
                g["TOPK"] = int(stx.get("TOPK", g["TOPK"]))
        key = (mod, fam, hint, r["spec"])
        c = st.claims.get(key)
        if c is None:
            c = st.claims[key] = _Claim(fam, mod, hint, r["spec"], stx)
        c.rows += 1
        c.idx.append(i)
        if node not in c.node_set:
            c.node_set.add(node)
            c.nodes.append(node)
        if fam.startswith(RM._ORDINAL_FAMILIES):
            po = st.fam_pos.get((mod, fam, rownum), 0)
            st.fam_pos[(mod, fam, rownum)] = po + 1
            operand, why = self._operand_family(r)
            c.ordinals[(po, operand, why)] += 1

    def _operand_family(self, r: dict) -> tuple[str | None, str]:
        """[rev-pm M-0402] the family of the producer of the row's FIRST operand, from the rows already walked (Program order)."""
        args = r.get("args")
        if not args:
            return None, "the Program row carries no args"
        try:
            ref = args[0][0]
        except (IndexError, TypeError):
            return None, "the Program row has no first operand ref"
        try:
            if ref[0] == "p":
                return None, "the Program feeds this op from a parameter, not an instance output"
            if ref[0] != "n":
                return None, f"the Program row's first operand ref kind is {ref[0]!r}, not an instance output"
            return self._fam_names[self._fam_of[int(ref[1])]], ""
        except (KeyError, IndexError, TypeError, ValueError):
            return None, "the Program row's first operand ref does not resolve to an instance"

    def _classify(self, c: _Claim, cls: int) -> None:
        for i in c.idx:
            self._class_of[i] = cls

    def _flush(self, st: _Step) -> Iterator[Identity]:
        step = st.step
        prefix = self.prefix or ""
        self.steps = step + 1
        tp_ctx = {"rank": self.prog_rank, "world": self.world}
        # sampling event(s) of the step
        for i, fam in st.selects:
            self._class_of[i] = CLASS_SAMPLER
            yield self._mk("sampled_token_ids", step, 0, "runner.sampler", "sampled_token_ids", 0, self.n_seq, dtype="i64", spec=fam)
            if is_stochastic_select(fam):
                yield self._mk("sampler_seed", step, 0, "runner.sampler", "seed", 0, self.n_seq, dtype="i64", spec=fam)
                self.stochastic_selectors.add(fam)
                if self.splits_param:                                            # [R16 REQ-03] the split count S the top-p mask used at this event
                    yield self._mk("sampler_splits", step, 0, "runner.sampler", "splits", 0, self.n_seq, dtype="i32", spec=fam)
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for c in st.claims.values():
            fam, mod, hint, stx = c.fam, c.mod, c.hint, c.statics
            op = collector_path(mod, prefix)
            coll = st.coll_of.get(mod)
            if RM.is_epilogue_interior(fam, mod, st.epi_mods) and not coll:
                if int(stx.get("N", 0)) != st.epi_mods[mod]:                     # rev-cb D2: the epilogue must return the Gemm's width
                    self._um(f"{fam} under {op}: Gemm width {int(stx.get('N', 0))} != its BiasAdd epilogue width {st.epi_mods[mod]} (D2)", c.rows)
                    self._classify(c, CLASS_UNMODELLED)
                else:
                    self._classify(c, CLASS_INTERIOR)
                    self._interior.append((step, mod, fam, c.spec, stx, list(c.idx)))   # [int-20] the pre-bias Gemm: interior unless it crosses
                continue
            try:
                members = members_for(fam, stx, mod, hint, tp=(dict(tp_ctx, collective=coll) if self.world > 1 else None),
                                      padded_slots=self.q.boundary == "module-pad")
            except UnmodelledFamily as e:
                self._um(f"{e.family} under {op}", c.rows)
                self._classify(c, CLASS_UNMODELLED)
                continue
            if not members:
                if is_aggregate_family(fam, mod):
                    self._classify(c, CLASS_AGGREGATE)
                elif is_structure_family(fam):
                    self._classify(c, CLASS_STRUCTURE)
                else:
                    self._classify(c, CLASS_INTERIOR)
                    self._interior.append((step, mod, fam, c.spec, stx, list(c.idx)))   # [int-20] promotable: decided after the walk
                continue
            if "." not in mod:
                # [MAN-08] an identity-bearing row under an UNROOTED module path: incomplete, named with the path
                self._um(f"{fam} under unrooted module path {mod!r} (Program names {st.rows_under[mod]} rows of this step under a bare leaf)", c.rows)
                self._classify(c, CLASS_UNMODELLED)
                continue
            claimed = False
            for mem in members:
                member, width, dtype = mem[0], mem[1], mem[2]
                extras = dict(mem[3]) if len(mem) > 3 and mem[3] else {}
                if width <= 0:                                                   # [MAN-02] never a zero-width identity
                    self._um(f"{fam} under {op} ({member}: width 0)", c.rows)
                    continue
                n_rows = c.rows
                if c.ordinals:
                    # [M-0323 / M-0402] the declared producer_ordinal / producer_operand must be the Program's own, per row
                    bad = 0
                    for (po, operand, why), n in sorted(c.ordinals.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
                        if "producer_ordinal" in extras and extras["producer_ordinal"] != po:
                            self._um(f"{fam} under {op} ({member}: declared producer_ordinal {extras['producer_ordinal']} but the Program's row order puts this op at {po})", n)
                            bad += n
                            continue
                        if "producer_operand" in extras and extras["producer_operand"] != operand:
                            w = (f"the Program's dataflow feeds this op from {operand}" if operand is not None else f"{why}: the declared operand cannot be confirmed")
                            self._um(f"{fam} under {op} ({member}: declared producer_operand {extras['producer_operand']} but {w})", n)
                            bad += n
                    n_rows -= bad
                    if n_rows <= 0:
                        continue
                explicit = hint == member
                g = groups.get((mod, member))
                if g is None:
                    g = groups[(mod, member)] = {"nodes": list(c.nodes), "rows": n_rows, "width": width, "dtype": dtype, "family": fam, "hinted": explicit, "extras": extras}
                    claimed = True
                    continue
                if g["hinted"] and not explicit:
                    if (g["width"], g["dtype"]) != (width, dtype):              # [rev-cb D2] order-independent check
                        self._um(f"{g['family']} under {op} ({member}: hinted producer {g['width']}x{g['dtype']} vs a default claim {fam} {width}x{dtype})", n_rows)
                    continue                                                     # default claim under a module with a hinted producer
                if g["hinted"] and explicit and g["family"] != fam:
                    self._um(f"{fam} under {op} ({member}: two hinted producers {g['family']} / {fam})", n_rows)
                    continue
                if not g["hinted"] and explicit:                                 # default rows seen first, now the hinted producer: replace
                    if (g["width"], g["dtype"]) != (width, dtype):
                        self._um(f"{fam} under {op} ({member}: hinted producer {width}x{dtype} replaces a default claim {g['family']} {g['width']}x{g['dtype']})", n_rows)
                        continue
                    g.update(nodes=list(c.nodes), rows=n_rows, width=width, dtype=dtype, family=fam, hinted=True, extras=extras)
                    claimed = True
                    continue
                if g["hinted"] and explicit and (g["width"], g["dtype"]) != (width, dtype):
                    self._um(f"{fam} under {op} ({member}: hinted rows of unequal width {g['width']} / {width})", n_rows)
                    continue
                if extras and g.get("extras", {}) != extras:
                    raise ValueError(f"{mod}/{member} step {step}: the rows of one member disagree on their extras ({g.get('extras')} vs {extras})")
                if g["family"] != fam:
                    # two DEFAULT families claim one member (Gemm_v1 and Gemm_v2 rows of one module, an FP8 twin beside its bf16 rows):
                    # the old inventory folded them silently under the first family; kept identical here, named in the audit
                    pass
                for nd in c.nodes:
                    if nd not in g["nodes"]:
                        g["nodes"].append(nd)
                g["rows"] += n_rows
                claimed = True
            if claimed:
                self._classify(c, CLASS_DECLARED if any(str(m[0]).startswith(_DECLARED_MEMBER_PREFIXES) for m in members) else CLASS_BOUNDARY)
            else:
                self._classify(c, CLASS_UNMODELLED if not any(int(m[1]) > 0 for m in members) else CLASS_INTERIOR)
        # [rev-cb D1] a soft-cap chain without a hinted producer: fail closed
        for mod, fam in st.softcap.items():
            for (m, member), g in groups.items():
                if m == mod and not g["hinted"]:
                    self._um(f"{fam} under {collector_path(mod, prefix)} ({member}: soft-cap chain without a hinted producer; default claim {g['family']})")
        # shared RotaryEmbedding: one module instance called once per layer -> alias (first path, invocation = layer order)
        rope_alias = RM.shared_module_aliases(groups, shared=self.q.shared_rope)     # [R17-39 (c)] one rule, signature-grouped, fail-closed
        rope_alias.update(RM.collective_site_aliases(groups))                        # [R19 moe F-r19-moe-1] mid-module collective sites -> (module, k)
        for m, (canon, k) in rope_alias.items():
            if m != canon:
                self.aliases[collector_path(m, prefix)] = [collector_path(canon, prefix), k]
        self._named_by_step[step] = set(groups)                                      # [int-20] a promoted member never collides silently
        self._aliased_by_step[step] = set(rope_alias)
        for (mod, member), g in sorted(groups.items(), key=lambda kv: (_layer_of(kv[0][0]), kv[0])):
            op = collector_path(mod, prefix)
            if mod in rope_alias:
                canon, k = rope_alias[mod]
                yield self._mk(RM.alias_family(member), step, k, collector_path(canon, prefix), member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"],
                               dtype=g["dtype"], spec=g["family"], program_op_path=op, **g.get("extras", {}))
                continue
            fam_label = "tp_rank_partials" if member.startswith(_DECLARED_MEMBER_PREFIXES) else "instance_outputs"
            yield self._mk(fam_label, step, 0, op, member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"], dtype=g["dtype"],
                           spec=g["family"], fx_nodes=len(g["nodes"]), **g.get("extras", {}))
            if g["family"].startswith("Embedding"):
                yield self._mk("input_token_ids", step, 0, op, "in", 0, g["rows"], rows=g["rows"], width=1, dtype="i32")
        # MoE block stream planes
        for mod, g in sorted(st.moe.items(), key=lambda kv: _layer_of(kv[0])):
            layer = _layer_of(mod)
            if g["M"] <= 0 or g["TOPK"] <= 0:
                continue
            if g["expert_gemm_rows"] != g["M"] * g["TOPK"]:
                raise ValueError(f"MoE layer {layer} step {step}: {g['expert_gemm_rows']} expert-GEMM instances != M*TOPK = {g['M']}*{g['TOPK']}")
            self.moe_layers.add(layer)
            for plane, n, dtype in moe_identities(g):
                yield self._mk("moe_block_stream", step, 0, f"moe/L{layer}", plane, 0, n, dtype=dtype, rows=g["M"], topk=g["TOPK"], moe_module=collector_path(mod, prefix))
        # attention hidden stream: one launch per attention layer per step
        for mod, a in sorted(st.attn.items()):
            if self.hidden_family.startswith("fa3"):
                geo = _fa3_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], self.n_seq)
            else:
                win = (self.attn_obs.get(collector_path(mod, prefix)) or self.attn_obs.get(mod) or {}).get("sliding_window")
                w0 = win[0] if isinstance(win, (list, tuple)) and win else (win if isinstance(win, int) else -1)
                geo = _fa2_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], self.n_seq, windowed=(w0 is not None and int(w0) >= 0))
            HB, M, NB, BN, gname = geo
            layer = int(mod.split(".layers.")[1].split(".")[0]) if ".layers." in mod else -1
            tap = "fa3.m1" if self.hidden_family.startswith("fa3") else "fa2.m1"
            self.hidden_layers.add(f"{tap}/L{layer}")
            if a["program_BN"]:
                self.hidden_program_bn.setdefault(f"{tap}/L{layer}", {})[step] = sorted(a["program_BN"])
                if not self.in_fold and BN not in a["program_BN"]:
                    # [R16 reqval, Hopper preflight] the hidden-stream geometry is the launch table's (`_fa2_geometry` / `_fa3_geometry`,
                    # mirror of hidden_source.fa2_kblock_n / fa3_tile) while the Program's attention rows carry the block size the Build
                    # bound (verity_vllm.registry.targets.fa3_kblock_n on the declared launch context): a disagreement means the required
                    # stream name `g..x<BN>x<D>` is one no launch exports (coverage MISSING: fail closed) -- e.g. a wrong profile, or a
                    # table that drifted from the pinned kernel's heuristics.  Named in the audit; the identities are the inventory's (a
                    # change of the geometry source is a DECISIONS-PENDING item, not a traversal rule).  Under a B>=2 fold the launch is
                    # the ENGINE step's (max_seqlen_q over the batch), so the comparison is the fold's (WorkloadTraversal._hidden_union).
                    _note_bn_disagreement(self.hidden_bn_disagreement, f"{tap}/L{layer}", a["program_BN"], BN, collector_path(mod, prefix))
            geometry = {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": a["D"], "seqlen_k": a["T_max"], "rows": a["rows"]}
            if step in (0, 1):
                self.hidden_geo.setdefault(f"{tap}/L{layer}", {})[step] = geometry     # the global fold recovers NH / KVH / D from these
            yield self._mk(self.hidden_family, step, 0, f"{tap}/L{layer}", gname, 0, mat_total_words(HB, M, NB, BN, a["D"]), dtype="u32",
                           geometry=geometry, attention_module=collector_path(mod, prefix))

    def _promote(self) -> Iterator[Identity]:
        """[R19 int-20] the deferred promotion pass (`RM.PROMOTION_RULE`), after the walk: every interior claim whose output the Program's
        dataflow reads from another module is a required member `(module, PROMOTED_MEMBER)` of its step -- class INTERIOR_CROSSING in the
        audit, `consumers` on the row.  Not armed without operand runs; a claim the pass cannot name is a named fault (`_finish`)."""
        if not self._args_seen or not self._interior:
            return
        res = promote_interior_crossings(self._interior, self._crosses.__getitem__, self._consumers_of,
                                         named=self._named_by_step.get, aliased=self._aliased_by_step.get, prefix=self.prefix or "")
        self.promotion = res
        for i in res["promoted_rows"]:
            self._class_of[i] = CLASS_INTERIOR_CROSSING
        for d in res["identities"]:
            yield self._mk("instance_outputs", d["step"], 0, d["op_path"], d["member"], 0, d["rows"] * d["width"], **promoted_annotations(d))

    def _finish(self) -> None:
        q = self.query_info
        promo = self.promotion or {}
        if promo.get("identities") or promo.get("faults"):
            q["promotion"] = promotion_header(promo)
        for text, n in (promo.get("faults") or {}).items():
            self._um(text, n)
        if self.guard_rows and not self.lifted_rows:
            self._um("guard rows (LStepLive/LSelect/LBot/LIsEOS/LEmitNext) in a Program with no Lifted[..] row: the step guard / controller is only meaningful as part of a bot-lift", self.guard_rows)
        if self.lifted_rows:                                                     # (an un-lifted Program's query header stays the inventory's)
            q["lifted"] = True
            q["lifted_rows"] = self.lifted_rows
            q["guard_rows"] = self.guard_rows
            q["guard_families"] = {k: int(v) for k, v in sorted(self.guard_families.items())}   # [revb S-r17b-16 R4] the controller, BY NAME
            q["why"]["lifted"] = ("bot-lifted Program (padded baseline, experimental): every identity is the wrapped family's, on the same key and "
                                  "element count (one lifted value = w+1 bits); the guard / controller rows (LStepLive/LSelect/LBot; R17-32 LIsEOS/LEmitNext) "
                                  "carry no identity under this rule -- the population names them as uncovered BY NAME (sampled_replay.query_population)")
        if self.stochastic_selectors:
            q["required_families"].append("sampler_seed")
            q["why"]["sampler_seed"] = (f"the stochastic selector(s) {sorted(self.stochastic_selectors)} consume the per-request seed as a registered runtime input "
                                        "(Philox(seed, pos) Gumbel noise): the committed sampled id is reproducible from the committed logits only with it")
            q["stochastic_selectors"] = sorted(self.stochastic_selectors)
            if self.splits_param:
                q["required_families"].append("sampler_splits")
                q["why"]["sampler_splits"] = ("the Program's request-owned `splits` input (one i32 per sampling event: the split count S the top-p mask kernel "
                                              "launched with, SplitsFor_v1(n_live, num_SMs)) is read by the stochastic selector as its 6th operand: the committed "
                                              "sampled id is reproducible from the committed logits only with the seed AND that S (REQ-03; F-stoch-01 is a wrong S)")
        if self.padded_moe_rows:
            # [reqval RequiredValues authority, MOE-PAD-SPEC s7 open item] the padded fixed-slot block is one module-boundary row per token:
            # y is required, the slot bodies are interior under Q_module_v1 -- stated on the header so no record can read the planes as required
            q["moe_construction"] = {"declared_by": "the Program's rows (MoeFfnFixedL_v1 / LMoeFfnL_v1 per token row)", "construction": "padded",
                                     "rows": int(self.padded_moe_rows),
                                     "required": "the block output y (member `out`, H words) per (token row, MoE layer) + the row's inputs by linkage (x, logits)",
                                     "slot_units": ("REQUIRED (Q_module_pad_v1: padded_slot_members per row)" if self.q.boundary == "module-pad" else
                                                    "NOT required (Q_module_v1: interior to the row)"),
                                     "not_required_under_Q_module_v1": ("the E slot bodies (up 2I / act I / down H / g per slot): interior to the row on this Program; "
                                                                        "required only under a query that partitions the block into slot units (Lift(Q_fine_v1) or a "
                                                                        "declared padded-boundary Q) -- moe's 5,121 words/slot planes are NOT R(P_pad, Q_module_v1)"),
                                     "moe_block_stream": "absent on this Program by construction (no MoeRouterTopK/MoeExpertGemm/MoeSum rows)"}
        if self.moe_layers:
            q["required_families"].append("moe_block_stream")
            q["why"]["moe_block_stream"] = ("the query's MoE families (MoeRouterTopK weights/ids, MoeExpertGemm up, SiluMul act, MoeExpertGemmW down, "
                                            "MoeSum) are produced inside one module forward -> the six-plane block stream of every MoE layer x step is required")
            q["moe_layers"] = sorted(self.moe_layers)
        self.boundary_audit = self._audit()
        # [BRIEF s7A: a missing required family cannot disappear from BOTH the expectation and the collector unnoticed] a family whose
        # rows' outputs the Program reads across a module boundary while the vocabulary names no member for it is a candidate omission
        # of the vocabulary -- the population is NOT declared complete on it (the manifest is complete:false and coverage refuses), the
        # family is named.  Reported from the dataflow, so a vocabulary rule that says "interior" cannot silence it.
        named_faults = (self.promotion or {}).get("named_families") or set()   # [int-20] already named, per module, by the promotion pass
        for fam, n in sorted(self.boundary_audit["unnamed_crossing"].items()):
            if fam in named_faults:
                continue
            self._um(f"{fam}: {n} row(s) whose output the Program reads across a module boundary while the vocabulary names no member for it (unnamed crossing)", n)
        for name, info in sorted(self.boundary_audit["unnamed_input"].items()):
            readers = ", ".join(f"{f} x{c}" for f, c in sorted(info["readers"].items()))
            self._um(f"input `{name}` ({info['type'] or 'param ' + str(info['param'])}): read by {readers} while no family names it (unnamed input: a required "
                     f"value the population does not carry; cf. `sampler_seed` for `seed`)", sum(info["readers"].values()))

    def _input_named_by(self, j: int, name: str, readers: dict[str, int]) -> str | None:
        """The family that NAMES Program input `j` -- its value is in the population -- by the vocabulary's RULE (who reads it, which family
        carries it), never by the param's name alone [R17 revb F-r17b-08].  None = an unnamed input (a required value the population does not
        carry: the population is not complete on it)."""
        fams = [unlift(f)[0] for f in readers]
        if j == self._weights_param:                                             # the weights param `_weights()` enumerated the fields from
            return "weights"                                                     # (an empty field list names no identity, and nothing is unnamed)
        # the token ids: every reader is an Embedding row (the lifted Embedding under a bot-lift; the guard rows LSelect/LStepLive read the
        # same input and their value is named at the lifted Embedding row) and the population carries `input_token_ids`
        if fams and all(f.startswith("Embedding") or is_guard(f) for f in fams) and self.fam_counts.get("input_token_ids", 0) > 0:
            return "input_token_ids"
        if name == "seed" and self.stochastic_selectors and all(is_stochastic_select(f) for f in fams):
            return "sampler_seed"                                                # [R15 stoch M-0107] read by the stochastic selector(s) only
        if name == RM.SPLITS_PARAM and self.splits_param and self.stochastic_selectors and all(is_stochastic_select(f) for f in fams):
            return "sampler_splits"                                              # [R16 REQ-03] named by `sampler_splits` (read by the selector only)
        # [R17 reqval finding 7; tp D90 M-0072] a rank Program's RECEIVED peer partial (`peers_<k>`, derive_step --tp 2): every reader is a
        # collective row (AllReduce2 / AllGather2) -> the value is the PEER rank's own partial at the reading row's (request, step, module),
        # an identity the peer rank's population already carries (`tp_rank_partials`: part_rank<1-r> / shard_rank<1-r>).  Named by that
        # identity, never a new one; the TP merge binds every collective output to it identity for identity (`tp_peer_binding`).  A world
        # other than 2 has no single peer -> not named (fail closed); a peers param some non-collective row reads stays unnamed by name.
        if self.world == 2 and fams and all(f.startswith(_COLLECTIVE_PREFIXES) for f in fams):
            return "tp_rank_partials"
        return None

    def _audit(self) -> dict[str, Any]:
        """Per (family, class): rows / rows whose output is read outside their module.  Disagreements are named, never silently dropped."""
        n = len(self._class_of)
        per: dict[tuple[str, int], list[int]] = {}
        for i in range(n):
            k = (self._fam_names[self._fam_of[i]], self._class_of[i])
            v = per.get(k)
            if v is None:
                v = per[k] = [0, 0]
            v[0] += 1
            v[1] += self._crosses[i]
        families: dict[str, dict[str, Any]] = {}
        unnamed_crossing: dict[str, int] = {}
        promoted: dict[str, int] = {}
        named_interior: dict[str, int] | None = {} if (self._args_seen and self._unwired == 0) else None
        for (fam, cls), (rows, crossing) in sorted(per.items()):
            families.setdefault(fam, {})[CLASS_NAMES[cls]] = {"rows": rows, "crossing": crossing}
            if not self._args_seen:
                continue
            if cls == CLASS_INTERIOR and crossing:
                unnamed_crossing[fam] = unnamed_crossing.get(fam, 0) + crossing        # an observed crossing is a fact whatever else is unwired
            if cls == CLASS_INTERIOR_CROSSING:
                promoted[fam] = promoted.get(fam, 0) + rows                            # [int-20] named by the promotion pass (every row of it crosses)
            # CLASS_GUARD rows cross by design (tok_in feeds the step): the value is named at the lifted Embedding row (`input_token_ids`)
            if named_interior is not None and cls == CLASS_BOUNDARY and crossing == 0 and rows > 0:
                named_interior[fam] = named_interior.get(fam, 0) + rows                # needs EVERY edge: a missing edge would fake an interior
        # [R16 reqval, stoch M-0254 derived S] the Program's INPUTS: every param some row reads must be named by a family -- `prompt` by
        # `input_token_ids`, `weights` by `weights`, `seed` by `sampler_seed` (R15 stoch M-0107).  A param the rows read and no family names
        # (e.g. the derived-S `splits` input of GumbelTopPTokenSelect_v1{V}: the top-p mask's split count, one i32 per sampling event) is
        # a required value the population does not carry: named here, and the population is not complete on it (`_finish`).
        unnamed_input: dict[str, dict[str, Any]] = {}
        named_input: dict[str, str] = {}
        peer_rows: dict[str, int] = {"all_reduce": 0, "all_gather": 0}
        peer_params = 0
        if self._args_seen:
            pnames = [str(nm) for nm, _t in (self.p.params or [])]
            for j, readers in sorted(self._param_readers.items()):
                name = pnames[j] if 0 <= j < len(pnames) else f"param[{j}]"
                fam = self._input_named_by(j, name, readers)
                if fam:
                    named_input[name] = fam
                    if fam == "tp_rank_partials":
                        peer_params += 1
                        for f, c in readers.items():
                            peer_rows["all_reduce" if unlift(f)[0].startswith("AllReduce2") else "all_gather"] += c
                    continue
                unnamed_input[name] = {"param": j, "type": str(next((t for nm, t in self.p.params if str(nm) == name), "")), "readers": readers}
        # [R17 finding 7] the received peer partials of a rank Program, as a summary (the identities they alias are the peer rank's)
        peer_inputs = ({"world": self.world, "rank": self.prog_rank, "peer_rank": (1 - self.prog_rank) if self.world == 2 else None,
                        "params": peer_params, "reader_rows": sum(peer_rows.values()), "reader_rows_by_kind": peer_rows,
                        "fields": "params = the `peers_<k>` Program inputs named tp_rank_partials; reader_rows = the collective rows READING them (per rank)",
                        "rule": ("a Program input read only by collective rows is the RECEIVED peer partial = the peer rank's own partial "
                                 "(`tp_rank_partials`: part_rank<peer> / shard_rank<peer>) at the reading row's (request, step, module): named by that "
                                 "rank's identity, never a new one; `required_manifest.merge_manifests` binds every collective output to its peer "
                                 "partial identity for identity (`tp_peer_binding`), a missing one is named and the merge is not complete")}
                       if (self.world > 1 and self._args_seen) else None)
        # [int-20] `promoted` is on the audit only when the pass named something: a Program without a crossing keeps its manifest byte for byte
        return {"available": self._args_seen, "fully_wired": self._args_seen and self._unwired == 0, "unwired_rows": self._unwired, "rows": n,
                "boundary": self.q.boundary, "families": families, "unnamed_crossing": unnamed_crossing, **({"promoted": promoted} if promoted else {}),
                "named_interior": named_interior,
                "inputs_read": {(str(self.p.params[j][0]) if 0 <= j < len(self.p.params or []) else f"param[{j}]"): sum(v.values()) for j, v in sorted(self._param_readers.items())},
                "named_input": named_input, "unnamed_input": unnamed_input, "peer_inputs": peer_inputs,
                "hidden_bn_disagreement": {k: dict(v) for k, v in sorted(self.hidden_bn_disagreement.items())},
                "note": ("a family under `unnamed_crossing` has rows whose output the Program reads from another module while the vocabulary names no "
                         "member for it: a candidate inventory omission (ALERT, never a silent skip); `named_interior` (only on a fully wired Program) is a "
                         "named family whose rows the Program never reads across a module boundary (over-claim or Program attribution defect); "
                         "`unnamed_input` is a Program input some row reads that no family names (a required value missing from the population); "
                         "`hidden_bn_disagreement` is an attention hidden stream whose launch-table block size differs from the BN the Program's own "
                         "attention rows carry (the required stream geometry is not the launch the Build bound: coverage will be MISSING on it)")}

    # ---- summary ------------------------------------------------------------------------------------------------------------------
    @property
    def complete(self) -> bool:
        return not self.unmodelled

    def populations(self) -> dict[str, Any]:
        return {"identities": self.n_identities, "by_family": {k: v for k, v in self.fam_counts.items() if v}, "steps": self.steps,
                "attention_layers": len(self.hidden_layers)}


def _chain(head: list[dict], rest: Iterator[dict]) -> Iterator[dict]:
    yield from head
    yield from rest


def required_values(program: ProgramView | dict[str, Any], query: Query | None = None) -> Iterator[Identity]:
    """RequiredValues(P, Q) as a lazy generator of identities (`Traversal` for the summary)."""
    return iter(Traversal(program, query))


# ---------------------------------------------------------------------------------------------------------------------------------
# nested calls: the workload Program (GP-01 Workload_v1) as components with arrivals
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Component:
    request_id: str                     # r<index>
    index: int
    arrive_step: int
    prompt_len: int
    max_tokens: int
    program: ProgramView | dict[str, Any]
    workload_request_id: str | None = None


@dataclass
class Workload:
    """`Workload_v1{requests, shared}`: one call node per request component; the weights are the shared prescribed input."""
    components: list[Component]
    digest: str | None = None           # gprog's Workload_v1 Program digest; None -> components-fold (until the Build carries it)
    fixture: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def of(fixture: dict[str, Any], programs: dict[str, Any], *, workload_digest: str | None = None) -> "Workload":
        declared = [dict(r, index=int(r.get("index", i))) for i, r in enumerate(fixture["requests"])]
        if len({r["index"] for r in declared}) != len(declared):
            raise ValueError("workload declaration: duplicate request index")
        comps = []
        for r in sorted(declared, key=lambda r: (int(r.get("arrive_step", 0)), r["index"])):
            rid = f"r{r['index']}"
            prog = programs.get(rid) if rid in programs else programs.get(str(r.get("request_id")))
            if prog is None:
                raise KeyError(f"no request Program for {rid} (workload id {r.get('request_id')})")
            comps.append(Component(rid, int(r["index"]), int(r.get("arrive_step", 0)), int(r["prompt_len"]), int(r["max_tokens"]), prog, r.get("request_id")))
        return Workload(comps, workload_digest, fixture)


class WorkloadTraversal:
    """RequiredValues over a nested workload Program: recurse into every component with its (request_id, arrival) query; shared inputs
    once; attention hidden launches unioned per engine step (request `*`).  Yields lazily: a component's request-local identities as
    they are produced; the engine-step hidden unions once every component that can contribute to that step has been walked."""

    def __init__(self, wl: Workload, *, rank: int = 0, ranks: int = 1, profile: str | None = None):
        self.wl = wl
        self.rank, self.ranks, self.profile = rank, ranks, profile
        self.parts: dict[str, Traversal] = {}
        self.engine_steps: dict[int, list[tuple[str, int]]] = {}
        self.rows_of: dict[tuple[int, str], int] = {}
        self.component_digests: list[str] = []
        self.pdig: str | None = None
        self.hidden_family: str | None = None
        self.n_identities = 0
        self.fam_counts: Counter = Counter()
        self.unmodelled: dict[str, int] = {}
        self.arrive_of: dict[str, int] = {}
        self.hidden_bn_disagreement: dict[str, dict[str, Any]] = {}             # the fold's launches vs the components' attention BN (audit)

    def __iter__(self) -> Iterator[Identity]:
        comps = self.wl.components
        for c in comps:
            self.arrive_of[c.request_id] = c.arrive_step
            for s in range(c.max_tokens):
                self.engine_steps.setdefault(c.arrive_step + s, []).append((c.request_id, s))
        views = {c.request_id: (c.program if isinstance(c.program, ProgramView) else ProgramView.of(c.program)) for c in comps}
        self.component_digests = sorted({v.digest for v in views.values()})
        self.pdig = str(self.wl.digest) if self.wl.digest else RM.components_fold_digest(self.component_digests)
        seen_w: set[tuple] = set()
        pending_hidden: dict[int, dict[str, dict[str, Any]]] = {}               # engine step -> op_path -> union accumulator
        for c in comps:
            t = Traversal(views[c.request_id], Query(request_id=c.request_id, arrive_step=c.arrive_step, rank=self.rank, ranks=self.ranks, profile=self.profile))
            t.in_fold = True
            self.parts[c.request_id] = t
            comp_dig = views[c.request_id].digest
            # one request step at a time: the step's identities are held until its ids row (the request's row count in the step) is seen
            cur: int | None = None
            buf: list[Identity] = []
            for ident in t:
                if ident.family == "weights":
                    k = (ident.rank, ident.op_path)
                    if k in seen_w:
                        continue
                    seen_w.add(k)
                    yield self._re(ident, comp_dig)
                    continue
                e = int(ident.ann["engine_step"])
                if cur is not None and e != cur:
                    yield from self._step_out(buf, c, comp_dig, cur)
                    buf = []
                cur = e
                if ident.family == "input_token_ids":
                    self.rows_of[(e, c.request_id)] = int(ident.ann["rows"])
                if t.hidden_family == ident.family:
                    self.hidden_family = ident.family
                    a = pending_hidden.setdefault(e, {}).setdefault(ident.op_path, {"requests": [], "n_seq": 0, "max_rows": 0, "T_max": 0,
                                                                                     "module": ident.ann.get("attention_module"), "comp": c.request_id})
                    a["requests"].append(c.request_id)
                    a["n_seq"] += 1
                    a["max_rows"] = max(a["max_rows"], int(ident.ann["geometry"]["rows"]))
                    a["T_max"] = max(a["T_max"], int(ident.ann["geometry"]["seqlen_k"]))
                    continue
                buf.append(ident)
            if buf:
                yield from self._step_out(buf, c, comp_dig, cur)
            if t.steps != c.max_tokens:
                raise ValueError(f"request {c.request_id}: Program has {t.steps} steps, the declaration caps {c.max_tokens} tokens (LP={c.prompt_len}): wrong component")
            self.unmodelled.update(t.unmodelled)
            # engine steps no later component can join are final: every later component arrives at >= this one's arrival
            for e in sorted(k for k in pending_hidden if k < c.arrive_step):
                yield from self._hidden_union(e, pending_hidden.pop(e))
        for e in sorted(pending_hidden):
            yield from self._hidden_union(e, pending_hidden.pop(e))

    def _step_out(self, buf: list[Identity], c: Component, comp_dig: str, e: int) -> Iterator[Identity]:
        n_rows = self.rows_of.get((e, c.request_id))
        for ident in buf:
            if ident.family in ("instance_outputs", "input_token_ids", "tp_rank_partials"):
                if ident.hi is not None and ident.hi <= ident.lo:
                    raise ValueError(f"request {c.request_id} step {ident.step} {ident.op_path}/{ident.output_member}: zero-width identity {[ident.lo, ident.hi]}")
                if n_rows is None:
                    raise ValueError(f"request {c.request_id} engine step {e}: no ids row (input_token_ids) names the request's row count in the step")
                # [R17 reqval C2 audit finding 1; mirrors build_global_manifest] `rows` stays the component's own block height: per-token
                # members == n_rows, per-SEQUENCE members (logits_processor/out, lm_head shard_rank<r>) == 1; the ids row must agree
                if ident.family == "input_token_ids" and int(ident.ann.get("rows") or 0) != n_rows:
                    raise ValueError(f"request {c.request_id} engine step {e} {ident.op_path}/in: ids rows {ident.ann.get('rows')} != the engine step's {n_rows}")
                yield self._re(ident, comp_dig)
            else:
                yield self._re(ident, comp_dig)

    def _re(self, ident: Identity, comp_dig: str, **over) -> Identity:
        ann = dict(ident.ann, component_digest=comp_dig)
        ann.update(over)
        self.n_identities += 1
        self.fam_counts[ident.family] += 1
        return ident._replace(program_digest=self.pdig, ann=ann)

    def _hidden_union(self, e: int, ops: dict[str, dict[str, Any]]) -> Iterator[Identity]:
        hidden_family = self.hidden_family
        for op, a in sorted(ops.items()):
            NH, KVH, D = self._heads_of(a["comp"], op)
            fn = _fa3_geometry if hidden_family.startswith("fa3") else _fa2_geometry
            HB, M, NB, BN, gname = fn(a["T_max"], a["max_rows"] * a["n_seq"], NH, KVH, D, a["n_seq"])
            prog_bn: set[int] = set()                                            # the BN the components' attention rows carry at this engine step
            for rid in a["requests"]:
                prog_bn.update(self.parts[rid].hidden_program_bn.get(op, {}).get(e - self.arrive_of[rid], []))
            if prog_bn and BN not in prog_bn:
                _note_bn_disagreement(self.hidden_bn_disagreement, op, prog_bn, BN, a["module"])
            self.n_identities += 1
            self.fam_counts[hidden_family] += 1
            yield Identity(hidden_family, self.pdig, "*", "*", 0, op, gname, 0, mat_total_words(HB, M, NB, BN, D), self.rank,
                           {"requests": a["requests"], "engine_step": e, "dtype": "u32",
                            "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, "seqlen_k": a["T_max"], "rows": a["max_rows"] * a["n_seq"], "n_seq": a["n_seq"]},
                            "attention_module": a["module"]})

    def _heads_of(self, rid: str, op: str) -> tuple[int, int, int]:
        """NH / KVH / D of an attention layer from the component's prefill (HB = NH x 1 seq) and decode (HB = KVH, M = NH/KVH) geometries."""
        geos = self.parts[rid].hidden_geo.get(op)
        if geos is None or 0 not in geos:
            raise KeyError(f"no prefill hidden geometry recorded for {rid} {op}")
        pre, dec = geos.get(0), geos.get(1)
        D = int(pre["D"])
        NH = int(pre["HB"])
        KVH = int(dec["HB"]) if dec is not None and int(dec["M"]) > 1 else NH
        return NH, KVH, D

    def composition(self) -> list[dict[str, Any]]:
        out = []
        for e in sorted(self.engine_steps):
            members = self.engine_steps[e]
            out.append({"engine_step": e, "members": [{"request_id": rid, "step": s, "rows": self.rows_of[(e, rid)]} for rid, s in members],
                        "rows": sum(self.rows_of[(e, rid)] for rid, _s in members), "n_seq": len(members)})
        return out

    @property
    def complete(self) -> bool:
        return all(t.complete for t in self.parts.values())


# ---------------------------------------------------------------------------------------------------------------------------------
# the expanded reference: materialize every candidate value of every row, then filter -- small circuits only
# ---------------------------------------------------------------------------------------------------------------------------------

def reference_required_values(program: ProgramView | dict[str, Any], query: Query | None = None) -> tuple[list[Identity], dict[str, Any]]:
    """The straightforward reference: load every row, cut the steps, expand EVERY row into its candidate (module, member) values with
    the full step in view (epilogues, collectives, geometry known up front), then coalesce and filter.  Returns (identities sorted by
    the manifest row key, summary {unmodelled, aliases, steps}).  Independent of `Traversal`'s streaming mechanics; shares only the
    vocabulary (`members_for`) and the geometry helpers."""
    p = program if isinstance(program, ProgramView) else ProgramView.of(program)
    q = query or Query()
    rows = list(p.rows)
    res = p.result
    attn_obs = res.get("attention_impls_observed") or {}
    prefix = wrapper_prefix_of(rows, attn_obs.keys())
    tgt = (res.get("target") or {}).get("declared") or {}
    fa = next((v.get("vllm_flash_attn_version") for v in attn_obs.values()), None)
    profile = q.profile or ("hopper" if (fa == 3 or (fa is None and int((tgt.get("compute_capability") or [8, 9])[0]) >= 9)) else "ampere")
    hidden_family = query_requirements(profile)["attention_hidden_family"]
    tpinfo = res.get("tp") if isinstance(res.get("tp"), dict) else {}
    world, prog_rank = int(tpinfo.get("world") or 1), int(tpinfo.get("rank") or 0)
    if (world > 1 or q.ranks > 1) and (int(q.ranks) != world or int(q.rank) != prog_rank):
        raise ValueError(f"rank Program mismatch: Program is rank {prog_rank} of world {world}, manifest requested rank {q.rank} of {q.ranks}")
    rid = request_label(q.request_id)
    arrive = int(q.arrive_step)
    unmodelled: Counter = Counter()
    out: list[Identity] = []

    def mk(family, step, inv, op, member, lo, hi, **ann):
        a = {} if step == "*" else {"engine_step": arrive + int(step)}
        a.update(ann)
        return Identity(family, p.digest, rid, step, inv, op, member, lo, hi, int(q.rank), a)

    # (1) shared prescribed inputs
    wparam = next((t for n, t in p.params if n == "weights"), None)
    wnames = re.findall(r"([A-Za-z0-9_.]+):\s*(?:Array|Value)", wparam) if wparam and str(wparam).startswith("{") else []
    wwidth = {m.group(1): int(m.group(3)) for m in re.finditer(r"([A-Za-z0-9_.]+):\s*((?:Array<\d+,)*Value<(\d+)>)", wparam or "")}
    if not wnames:
        wnames = list(((res.get("weights") or {}).get("roles") or {}).keys())
    for w in sorted(set(wnames)):
        out.append(mk("weights", "*", 0, collector_path(w, prefix), "weight", 0, None, dtype={8: "w8", 32: "f32", 16: "bf16"}.get(wwidth.get(w, 16), f"w{wwidth.get(w)}")))
    # (2) every step: expand every row -- on the UNLIFTED specs (a bot-lifted Program: the same keys as the wrapped Program's)
    lifted_n = 0
    unl = []
    for r in rows:
        sp, lf = unlift(r["spec"])
        lifted_n += lf
        unl.append(dict(r, spec=sp) if lf else r)
    rows = unl
    guard_n = sum(1 for r in rows if is_guard(spec_family(r["spec"])))
    if guard_n and not lifted_n:
        unmodelled["guard rows (LStepLive/LSelect/LBot) in a Program with no Lifted[..] row: the step guard is only meaningful as part of a bot-lift"] += guard_n
    segs = RM.segment_steps(rows)
    aliases: dict[str, list] = {}
    interior_claims: list[tuple] = []                                            # [int-20] (step, module, family, spec, statics, [row]) per interior row
    named_by_step: dict[int, set[tuple[str, str]]] = {}
    aliased_by_step: dict[int, set[str]] = {}
    for step, (lo_i, hi_i) in enumerate(segs):
        seg = rows[lo_i: hi_i + 1]
        parsed = [(r, spec_family(r["spec"]), *split_name(r.get("name") or ""), spec_statics(r["spec"])) for r in seg]
        epi = RM.epilogue_modules(seg)
        coll_of: dict[str, str] = {}
        if world > 1:
            for r, fam, mod, *_rest in parsed:
                if fam.startswith("AllReduce2"):
                    coll_of[mod] = "all_reduce"
                elif fam.startswith("AllGather2"):
                    coll_of[mod] = "all_gather"
        rows_under = Counter(mod for _r, _f, mod, *_ in parsed)
        # expanded candidates: one per (row, member)
        cands: list[dict[str, Any]] = []
        fam_pos: Counter = Counter()
        softcap: dict[str, str] = {}
        for k_row, (r, fam, mod, node, hint, rownum, stx) in enumerate(parsed):
            op = collector_path(mod, prefix)
            if is_guard(fam):
                continue
            if is_token_select(fam):
                out.append(mk("sampled_token_ids", step, 0, "runner.sampler", "sampled_token_ids", 0, 1, dtype="i64", spec=fam))
                if is_stochastic_select(fam):
                    out.append(mk("sampler_seed", step, 0, "runner.sampler", "seed", 0, 1, dtype="i64", spec=fam))
                    if RM.has_splits_param(p.params):                            # [R16 REQ-03]
                        out.append(mk("sampler_splits", step, 0, "runner.sampler", "splits", 0, 1, dtype="i32", spec=fam))
                continue
            if fam.startswith(("Bf16DivScalar", "Bf16Tanh", "Bf16MulScalar_")) and not hint:
                softcap.setdefault(mod, fam)
            if RM.is_epilogue_interior(fam, mod, epi) and not coll_of.get(mod):
                if int(stx.get("N", 0)) != epi[mod]:
                    unmodelled[f"{fam} under {op}: Gemm width {int(stx.get('N', 0))} != its BiasAdd epilogue width {epi[mod]} (D2)"] += 1
                else:
                    interior_claims.append((step, mod, fam, r["spec"], stx, [lo_i + k_row]))   # [int-20] the pre-bias Gemm: interior unless it crosses
                continue
            ordinal = operand = None
            why = ""
            if fam.startswith(RM._ORDINAL_FAMILIES):
                ordinal = fam_pos[(mod, fam, rownum)]
                fam_pos[(mod, fam, rownum)] += 1
                operand, why = RM._operand_family(r, rows)
            try:
                members = members_for(fam, stx, mod, hint, tp=(dict(rank=prog_rank, world=world, collective=coll_of.get(mod)) if world > 1 else None),
                                      padded_slots=q.boundary == "module-pad")
            except UnmodelledFamily as e:
                unmodelled[f"{e.family} under {op}"] += 1
                continue
            if not members:
                if not is_aggregate_family(fam, mod) and not is_structure_family(fam):
                    interior_claims.append((step, mod, fam, r["spec"], stx, [lo_i + k_row]))   # [int-20] interior by the vocabulary: promotable
                continue
            if "." not in mod:
                unmodelled[f"{fam} under unrooted module path {mod!r} (Program names {rows_under[mod]} rows of this step under a bare leaf)"] += 1
                continue
            for mem in members:
                member, width, dtype = mem[0], mem[1], mem[2]
                extras = dict(mem[3]) if len(mem) > 3 and mem[3] else {}
                if width <= 0:
                    unmodelled[f"{fam} under {op} ({member}: width 0)"] += 1
                    continue
                if ordinal is not None and "producer_ordinal" in extras and extras["producer_ordinal"] != ordinal:
                    unmodelled[f"{fam} under {op} ({member}: declared producer_ordinal {extras['producer_ordinal']} but the Program's row order puts this op at {ordinal})"] += 1
                    continue
                if ordinal is not None and "producer_operand" in extras and extras["producer_operand"] != operand:
                    w = f"the Program's dataflow feeds this op from {operand}" if operand is not None else f"{why}: the declared operand cannot be confirmed"
                    unmodelled[f"{fam} under {op} ({member}: declared producer_operand {extras['producer_operand']} but {w})"] += 1
                    continue
                cands.append({"mod": mod, "member": member, "width": width, "dtype": dtype, "fam": fam, "hinted": hint == member, "extras": extras, "node": node})
        # coalesce per (module, member): the hinted producer wins; defaults under a hinted module are dropped (D2 width agreement)
        by_key: dict[tuple[str, str], list[dict]] = {}
        for c in cands:
            by_key.setdefault((c["mod"], c["member"]), []).append(c)
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for (mod, member), cs in by_key.items():
            op = collector_path(mod, prefix)
            hinted = [c for c in cs if c["hinted"]]
            default = [c for c in cs if not c["hinted"]]
            if hinted:
                fams = sorted({c["fam"] for c in hinted})
                if len(fams) > 1:
                    keep = hinted[0]["fam"]
                    for c in hinted:
                        if c["fam"] != keep:
                            unmodelled[f"{c['fam']} under {op} ({member}: two hinted producers {keep} / {c['fam']})"] += 1
                    hinted = [c for c in hinted if c["fam"] == keep]
                w0, d0 = hinted[0]["width"], hinted[0]["dtype"]
                for c in default:
                    if (c["width"], c["dtype"]) != (w0, d0):
                        unmodelled[f"{hinted[0]['fam']} under {op} ({member}: hinted producer {w0}x{d0} vs a default claim {c['fam']} {c['width']}x{c['dtype']})"] += 1
                bad = [c for c in hinted if (c["width"], c["dtype"]) != (w0, d0)]
                for c in bad:
                    unmodelled[f"{c['fam']} under {op} ({member}: hinted rows of unequal width {w0} / {c['width']})"] += 1
                use = [c for c in hinted if (c["width"], c["dtype"]) == (w0, d0)]
            else:
                use = default
            if not use:
                continue
            ex = use[0]["extras"]
            for c in use:
                if c["extras"] and c["extras"] != ex:
                    raise ValueError(f"{mod}/{member} step {step}: the rows of one member disagree on their extras ({ex} vs {c['extras']})")
            nodes = list(dict.fromkeys(c["node"] for c in use))
            groups[(mod, member)] = {"rows": len(use), "width": use[0]["width"], "dtype": use[0]["dtype"], "family": use[0]["fam"], "hinted": bool(hinted),
                                     "extras": ex, "nodes": nodes}
        for mod, fam in softcap.items():
            for (m, member), g in groups.items():
                if m == mod and not g["hinted"]:
                    unmodelled[f"{fam} under {collector_path(mod, prefix)} ({member}: soft-cap chain without a hinted producer; default claim {g['family']})"] += 1
        rope_alias = RM.shared_module_aliases(groups, shared=q.shared_rope)          # [R17-39 (c)] one rule, signature-grouped, fail-closed
        rope_alias.update(RM.collective_site_aliases(groups))                        # [R19 moe F-r19-moe-1] mid-module collective sites -> (module, k)
        for m, (canon, k) in rope_alias.items():
            if m != canon:
                aliases[collector_path(m, prefix)] = [collector_path(canon, prefix), k]
        named_by_step[step] = set(groups)
        aliased_by_step[step] = set(rope_alias)
        for (mod, member), g in groups.items():
            op = collector_path(mod, prefix)
            if mod in rope_alias:
                canon, k = rope_alias[mod]
                out.append(mk(RM.alias_family(member), step, k, collector_path(canon, prefix), member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"],
                              dtype=g["dtype"], spec=g["family"], program_op_path=op, **g["extras"]))
                continue
            label = "tp_rank_partials" if member.startswith(_DECLARED_MEMBER_PREFIXES) else "instance_outputs"
            out.append(mk(label, step, 0, op, member, 0, g["rows"] * g["width"], rows=g["rows"], width=g["width"], dtype=g["dtype"], spec=g["family"],
                          fx_nodes=len(g["nodes"]), **g["extras"]))
            if g["family"].startswith("Embedding"):
                out.append(mk("input_token_ids", step, 0, op, "in", 0, g["rows"], rows=g["rows"], width=1, dtype="i32"))
        for mod, g in RM.moe_geometry(seg).items():
            layer = _layer_of(mod)
            if g["M"] <= 0 or g["TOPK"] <= 0:
                continue
            if g["expert_gemm_rows"] != g["M"] * g["TOPK"]:
                raise ValueError(f"MoE layer {layer} step {step}: {g['expert_gemm_rows']} expert-GEMM instances != M*TOPK = {g['M']}*{g['TOPK']}")
            for plane, n, dtype in moe_identities(g):
                out.append(mk("moe_block_stream", step, 0, f"moe/L{layer}", plane, 0, n, dtype=dtype, rows=g["M"], topk=g["TOPK"], moe_module=collector_path(mod, prefix)))
        attn: dict[str, dict[str, Any]] = {}
        for r, fam, mod, node, hint, rownum, stx in parsed:
            if fam.startswith(("Attention", "AttentionHead")):
                a = attn.setdefault(mod, {"rows": 0, "T_max": 0, "NH": int(stx.get("NH", 0)), "KVH": int(stx.get("KVH", stx.get("NH", 0))), "D": int(stx.get("D", 0))})
                a["rows"] += 1
                a["T_max"] = max(a["T_max"], int(stx.get("T", 0)))
        for mod, a in attn.items():
            if hidden_family.startswith("fa3"):
                geo = _fa3_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], 1)
            else:
                win = (attn_obs.get(collector_path(mod, prefix)) or attn_obs.get(mod) or {}).get("sliding_window")
                w0 = win[0] if isinstance(win, (list, tuple)) and win else (win if isinstance(win, int) else -1)
                geo = _fa2_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], 1, windowed=(w0 is not None and int(w0) >= 0))
            HB, M, NB, BN, gname = geo
            layer = _layer_of(mod)
            tap = "fa3.m1" if hidden_family.startswith("fa3") else "fa2.m1"
            out.append(mk(hidden_family, step, 0, f"{tap}/L{layer}", gname, 0, mat_total_words(HB, M, NB, BN, a["D"]), dtype="u32",
                          geometry={"HB": HB, "M": M, "NB": NB, "BN": BN, "D": a["D"], "seqlen_k": a["T_max"], "rows": a["rows"]}, attention_module=collector_path(mod, prefix)))
    # [R19 int-20] the promotion pass over the whole Program's dataflow (a consumer may sit in a later step than its producer)
    crosses, consumers = RM.program_crossings(rows)
    promo = promote_interior_crossings(interior_claims, crosses.__getitem__, lambda i: consumers.get(i, ()), named=named_by_step.get, aliased=aliased_by_step.get, prefix=prefix)
    for text, n in promo["faults"].items():
        unmodelled[text] += n
    for d in promo["identities"]:
        out.append(mk("instance_outputs", d["step"], 0, d["op_path"], d["member"], 0, d["rows"] * d["width"], **promoted_annotations(d)))
    out.sort(key=lambda x: RM._row_key(x.row()))
    return out, {"unmodelled": dict(unmodelled), "aliases": aliases, "steps": len(segs), "complete": not unmodelled,
                 "promotion": promotion_header(promo) if (promo["identities"] or promo["faults"]) else None}


# ---------------------------------------------------------------------------------------------------------------------------------
# the manifest as the cached serialization of the traversal (adapter for the existing binder / coverage_check / commit_delta)
# ---------------------------------------------------------------------------------------------------------------------------------

def manifest_from_traversal(program: dict[str, Any], workload: dict[str, Any] | None = None, *, request_id: str | int | None = None,
                            rank: int = 0, ranks: int = 1, profile: str | None = None, shared_rope: bool = True, boundary: str = "module") -> dict[str, Any]:
    """Same signature and same result dict as `required_manifest.build_manifest`, produced by `Traversal` (the manifest is the cache of
    the traversal).  `boundary_audit` is added beside (additive)."""
    inst, res = program["instances"], program["result"]
    sweep = (workload or {}).get("sweep") or {}
    wr = (workload or {}).get("requests") or []
    workload_request_id = wr[0].get("request_id") if wr and isinstance(wr[0], dict) else None
    rid = request_label(request_id, workload)
    k_req = int(_REQ_RE.match(rid).group(1))
    if wr and k_req < len(wr) and isinstance(wr[k_req], dict) and str(wr[k_req].get("request_id")) != str(workload_request_id):
        workload_request_id = wr[k_req].get("request_id")
    arr = sweep.get("arrive_steps") or []
    arrive = int(arr[k_req] if k_req < len(arr) else (arr[0] if arr else 0)) if sweep else 0
    t = Traversal(program, Query(request_id=rid, arrive_step=arrive, rank=rank, ranks=ranks, profile=profile, shared_rope=shared_rope, boundary=boundary))
    identities = [x.row() for x in t]
    identities.sort(key=RM._row_key)
    digest = RM.manifest_digest(identities)
    pops = t.populations()
    return {"schema": RM.SCHEMA, "identity_key": list(IDENTITY_KEY), "program_digest": t.pdig,
            "program": {"dir": program.get("dir"), "function": res["program"].get("function"), "wrapper": res.get("wrapper"), "LP": res.get("LP"), "T": res.get("T"),
                        "model_pin": res.get("model_pin"), "artifact_identity": (program.get("artifact") or {}).get("identity"),
                        "instances": len(inst["rows"]), "instances_digest": inst.get("digest")},
            "workload": {"row_id": sweep.get("row_id"), "batch": sweep.get("batch"), "request_id": rid, "workload_request_id": workload_request_id,
                         "arrive_step": arrive} if workload else None,
            "query": t.query_info, "wrapper_prefix": t.prefix, "op_path_aliases": t.aliases, "target": t.tgt, "flash_attn_version": t.fa, "ranks": ranks, "rank": rank,
            "steps": list(range(t.steps)), "engine_steps": [arrive + s for s in range(t.steps)], "requests": [rid],
            "populations": {"identities": pops["identities"], "by_family": pops["by_family"], "steps": t.steps, "attention_layers": pops["attention_layers"]},
            "complete": t.complete, "unmodelled": t.unmodelled,
            "manifest_digest": digest, "identities": identities,
            "engine": {"schema": SCHEMA, "traversal": "verity_capture.commit.required_values.Traversal", "identity_keys": identity_key_uniqueness(identities)},
            "boundary_audit": t.boundary_audit, "guard": guard_summary(t.boundary_audit, engine="traversal")}


def identity_key_uniqueness(identities: list[dict[str, Any]]) -> dict[str, Any]:
    """[R16 reqval, DECISIONS-PENDING REQ-01] does the canonical 8-tuple identify each row?  At B=1 yes; in a GLOBAL manifest the hidden-stream
    union rows carry request_id "*" / step "*" and put the engine step + the live requests in ANNOTATIONS (`engine_step`, `requests`) --
    the binder expands those into per-request per-engine-step identities (binding.load_required_manifest), so coverage is per engine step,
    but two engine steps with the same live geometry share one canonical key.  Reported, never repaired here (an identity-key change)."""
    keys = Counter(json.dumps(RM.canonical_identity(r), sort_keys=True) for r in identities)
    colliding = {k: c for k, c in keys.items() if c > 1}
    fams = Counter(r["family"] for r in identities if keys[json.dumps(RM.canonical_identity(r), sort_keys=True)] > 1)
    return {"rows": len(identities), "distinct_canonical": len(keys), "colliding_keys": len(colliding), "rows_sharing_a_key": sum(colliding.values()),
            "families": dict(fams), "unique": not colliding,
            "note": ("rows_sharing_a_key > 0 only for global hidden-stream union rows (engine step and live requests are annotations outside the "
                     "canonical key; the binder reads them) -- see DECISIONS-PENDING REQ-01")}


def manifest_equal(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Old manifest vs new: the identities (as rows), the digest, completeness, the unmodelled families, the aliases and the query.
    Returns {equal, why, only_old, only_new, changed} -- the identity lists of the difference, for the ALERT."""
    def keyed(m):
        return {json.dumps([r["family"]] + RM.canonical_identity(r), sort_keys=True): r for r in m["identities"]}
    ka, kb = keyed(a), keyed(b)
    only_old = sorted(set(ka) - set(kb))
    only_new = sorted(set(kb) - set(ka))
    changed = []
    for k in sorted(set(ka) & set(kb)):
        if ka[k] != kb[k]:
            changed.append({"identity": json.loads(k), "old": ka[k], "new": kb[k]})
    why = []
    if a.get("manifest_digest") != b.get("manifest_digest"):
        why.append(f"manifest_digest {str(a.get('manifest_digest'))[:16]} != {str(b.get('manifest_digest'))[:16]}")
    if a.get("complete") != b.get("complete"):
        why.append(f"complete {a.get('complete')} != {b.get('complete')}")
    if (a.get("unmodelled") or {}) != (b.get("unmodelled") or {}):
        why.append(f"unmodelled {a.get('unmodelled')} != {b.get('unmodelled')}")
    if (a.get("op_path_aliases") or {}) != (b.get("op_path_aliases") or {}):
        why.append("op_path_aliases differ")
    if a.get("query") != b.get("query"):
        why.append("query differs")
    if a.get("populations") != b.get("populations"):
        why.append(f"populations {a.get('populations')} != {b.get('populations')}")
    if only_old or only_new or changed:
        why.append(f"{len(only_old)} identities only in old, {len(only_new)} only in new, {len(changed)} with different annotations")
    return {"equal": not why, "why": "; ".join(why), "only_old": [json.loads(k) for k in only_old[:64]], "only_new": [json.loads(k) for k in only_new[:64]],
            "changed": changed[:16], "n_old": len(ka), "n_new": len(kb)}


def global_manifest_from_traversal(fixture: dict[str, Any], programs: dict[str, Any], *, workload_digest: str | None = None,
                                   rank: int = 0, ranks: int = 1, profile: str | None = None) -> dict[str, Any]:
    """Same result dict as `required_manifest.build_global_manifest`, produced by `WorkloadTraversal` over the nested components.
    `programs`: {request_id: loaded Program dict | ProgramView (`ProgramView.streaming(dir)` keeps the rows out of memory) | dir path}."""
    programs = {rid: (ProgramView.streaming(p) if isinstance(p, str) else p) for rid, p in programs.items()}
    wl = Workload.of(fixture, programs, workload_digest=workload_digest)
    wt = WorkloadTraversal(wl, rank=rank, ranks=ranks, profile=profile)
    ids = [x.row() for x in wt]
    ids.sort(key=RM._row_key)
    ref = {}
    for r in ids:
        if r["family"] in ("input_token_ids", "sampled_token_ids") and r["family"] not in ref:
            ref[r["family"]] = {"op_path": r["op_path"], "output_member": r["output_member"], "per_row": 1}
    reqs = wl.components
    first = wt.parts[reqs[0].request_id]
    ba = dict({c.request_id: wt.parts[c.request_id].boundary_audit for c in reqs},
              **{"*": {"hidden_bn_disagreement": {k: dict(v) for k, v in sorted(wt.hidden_bn_disagreement.items())},
                       "note": "the fold's attention launches (engine step, max_seqlen_q over the batch) vs the BN the components' "
                               "attention rows carry; the per-request audits do not compare (their own rows are not the launch)"}})
    return {"schema": RM.SCHEMA, "identity_key": list(IDENTITY_KEY), "program_digest": wt.pdig, "program_digest_kind": ("workload" if workload_digest else "components-fold"),
            "component_digests": wt.component_digests, "global": True, "addressing": "request-local", "composition_reference": ref,
            "fixture": {k: v for k, v in fixture.items() if k in ("schema", "fixture", "model", "revision", "sampling", "expected_engine_steps", "case")}
                       | ({"row_id": fixture["sweep"].get("row_id"), "seed": fixture["sweep"].get("seed")} if isinstance(fixture.get("sweep"), dict) else {}),
            "requests_declared": [{"request_id": c.request_id, "workload_request_id": c.workload_request_id, "index": c.index, "prompt_len": c.prompt_len,
                                   "max_tokens": c.max_tokens, "arrive_step": c.arrive_step} for c in reqs],
            "components": {c.request_id: {"program_digest": wt.parts[c.request_id].pdig, "steps": wt.parts[c.request_id].steps,
                                          "LP": wt.parts[c.request_id].p.result.get("LP"), "T": wt.parts[c.request_id].p.result.get("T"),
                                          "workload_request_id": c.workload_request_id} for c in reqs},
            "complete": wt.complete, "unmodelled": dict(wt.unmodelled),
            "query": first.query_info, "op_path_aliases": first.aliases, "ranks": ranks, "rank": rank,
            "engine_steps": wt.composition(), "steps": sorted(wt.engine_steps), "requests": [c.request_id for c in reqs],
            "populations": {"identities": len(ids), "by_family": dict(wt.fam_counts), "engine_steps": len(wt.engine_steps), "requests": len(reqs)},
            "manifest_digest": RM.manifest_digest(ids), "identities": ids,
            "engine": {"schema": SCHEMA, "traversal": "verity_capture.commit.required_values.WorkloadTraversal", "identity_keys": identity_key_uniqueness(ids)},
            "boundary_audit": ba, "guard": guard_summary(ba, engine="traversal")}


def eq_n(rec: dict[str, Any]) -> int:
    return int(rec.get("n_new") or 0)


def audit_flags(boundary_audit: dict[str, Any]) -> dict[str, Any]:
    """The audit's disagreement lists only, flattened over a B=1 audit or a B>=2 {request_id | "*": audit} (the console line of `diff`):
    everything that is not {} here is a finding to read before a Commit; {} everywhere is 'the population is the Program's and the launch
    geometry is the Build's'."""
    single = "available" in boundary_audit or "unnamed_crossing" in boundary_audit       # a Traversal's audit vs {request_id | "*": audit}
    parts = {"": boundary_audit} if single else boundary_audit
    out: dict[str, Any] = {"unnamed_crossing": {}, "unnamed_input": {}, "hidden_bn_disagreement": {}}
    for rid, ba in parts.items():
        if not isinstance(ba, dict):
            continue
        for key in out:
            v = ba.get(key) or {}
            if v:
                out[key][rid or "r"] = v
    out["fully_wired"] = all(bool(ba.get("fully_wired")) for rid, ba in parts.items() if isinstance(ba, dict) and rid != "*")
    return out


#: [R17 revb F-r17b-09] the guard's state, on the manifest header (and through it on the Commit record): the "missing family cannot vanish
#: from both sides" guard runs only over rows that carry operand runs (`args`); a Program without them leaves `available` false and the
#: population's `complete` untouched -- so the record must SAY whether the guard was armed, rather than let a disarmed guard read as clean.
GUARD_RULE = ("armed = every request Program's rows carry operand runs (boundary_audit.available) -- then an unnamed crossing / unnamed input / "
              "hidden-BN disagreement makes the population incomplete (coverage refuses) and is named under flags; not armed = the dataflow was "
              "not audited (no args, or the inventory engine): the population is the vocabulary's word alone")


def guard_summary(boundary_audit: dict[str, Any] | None, *, engine: str) -> dict[str, Any]:
    """The manifest header's `guard`: {engine, armed, available, fully_wired, flags: {kind: n}, rule} from a Traversal's audit or a B>=2
    {request_id | "*": audit}; the inventory engine (no dataflow audit) is `armed: False` by name."""
    if engine != "traversal" or not boundary_audit:
        return {"engine": engine, "armed": False, "available": False, "fully_wired": False, "flags": None,
                "why": "the inventory engine has no dataflow audit (the traversal engine's guard: --engine traversal / VERITY_REQUIRED_VALUES_ENGINE)"
                       if engine != "traversal" else "no boundary audit on the manifest", "rule": GUARD_RULE}
    single = "available" in boundary_audit or "unnamed_crossing" in boundary_audit
    parts = {"": boundary_audit} if single else boundary_audit
    per_req = [ba for rid, ba in parts.items() if isinstance(ba, dict) and rid != "*"]
    available = bool(per_req) and all(bool(ba.get("available")) for ba in per_req)
    flags = audit_flags(boundary_audit)
    out = {"engine": engine, "armed": available, "available": available, "fully_wired": bool(flags["fully_wired"]),
           "requests_audited": len(per_req),
           "flags": {k: sum(len(v) for v in flags[k].values()) for k in ("unnamed_crossing", "unnamed_input", "hidden_bn_disagreement")},
           "named_input": named_input_summary({n: fam for ba in per_req for n, fam in (ba.get("named_input") or {}).items()}),
           "rule": GUARD_RULE}
    promoted: dict[str, int] = {}                                                # [int-20] rows the promotion pass named (per family, all requests)
    for ba in per_req:
        for fam, n in (ba.get("promoted") or {}).items():
            promoted[fam] = promoted.get(fam, 0) + int(n)
    if promoted:                                                                 # (absent = nothing promoted: a manifest without a crossing is unchanged)
        out["promoted"] = promoted
    return out


_NUMBERED = re.compile(r"^(.*?)_(\d+)$")


def named_input_summary(named: dict[str, str]) -> list[str]:
    """`name -> family` lines for the guard; a numbered run of params named by ONE family (`peers_0..peers_1087` of a rank Program) is one
    line `peers_<0..1087> (1088) -> tp_rank_partials` -- the count is the fact, the 1088 names are the Program's."""
    runs: dict[tuple[str, str], list[int]] = {}
    lines: list[str] = []
    for n, fam in named.items():
        m = _NUMBERED.match(n)
        if m:
            runs.setdefault((m.group(1), fam), []).append(int(m.group(2)))
        else:
            lines.append(f"{n} -> {fam}")
    for (stem, fam), ks in runs.items():
        if len(ks) > 2:
            lines.append(f"{stem}_<{min(ks)}..{max(ks)}> ({len(ks)}) -> {fam}")
        else:
            lines.extend(f"{stem}_{k} -> {fam}" for k in ks)
    return sorted(lines)


def _main() -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("audit", help="stream a Program, print the population summary + boundary audit")
    a1.add_argument("--program", required=True)
    a1.add_argument("--workload", default=None)
    a1.add_argument("--request-id", default=None)
    a1.add_argument("--rank", type=int, default=0)
    a1.add_argument("--ranks", type=int, default=1)
    a1.add_argument("--profile", default=None)
    a1.add_argument("--out", default=None)
    d = sub.add_parser("diff", help="the inventory (required_manifest.build_manifest) vs the traversal: B=1 --program [--workload]; B>=2 --fixture "
                                    "--programs-root [--workload-digest] (the global manifest); exit 0 iff equal, else the identity lists")
    d.add_argument("--program", default=None)
    d.add_argument("--workload", default=None)
    d.add_argument("--fixture", default=None)
    d.add_argument("--programs-root", default=None)
    d.add_argument("--workload-digest", default=None)
    d.add_argument("--request-id", default=None)
    d.add_argument("--rank", type=int, default=0)
    d.add_argument("--ranks", type=int, default=1)
    d.add_argument("--profile", default=None)
    d.add_argument("--out", default=None)
    a = ap.parse_args()
    wl = json.load(open(a.workload)) if a.workload else None
    if a.cmd == "audit":
        sweep = (wl or {}).get("sweep") or {}
        rid = request_label(a.request_id, wl)
        t0 = time.time()
        t = Traversal(ProgramView.streaming(a.program), Query(request_id=rid, arrive_step=int((sweep.get("arrive_steps") or [0])[0]), rank=a.rank, ranks=a.ranks, profile=a.profile))
        n = sum(1 for _ in t)
        rec = {"schema": SCHEMA, "program": a.program, "program_digest": t.pdig, "identities": n, "populations": t.populations(), "complete": t.complete,
               "unmodelled": t.unmodelled, "boundary_audit": t.boundary_audit, "seconds": round(time.time() - t0, 3), "rows": len(t._class_of)}
        if a.out:
            json.dump(rec, open(a.out, "w"), indent=1)
        print(json.dumps(rec, indent=1)[:6000])
        return 0 if t.complete and not t.boundary_audit["unnamed_crossing"] else 1
    if a.fixture and a.programs_root:                                          # B>=2: the global manifest, both engines (the inventory named explicitly:
        fx = json.load(open(a.fixture))                                        # never the env-selected engine against itself)
        dirs = RM._program_dirs_for(fx, a.programs_root)
        t0 = time.time()
        old = RM.build_global_manifest(fx, RM._programs_for(fx, a.programs_root), workload_digest=a.workload_digest, rank=a.rank, ranks=a.ranks, profile=a.profile, engine="inventory")
        t1 = time.time()
        new = global_manifest_from_traversal(fx, dirs, workload_digest=a.workload_digest, rank=a.rank, ranks=a.ranks, profile=a.profile)   # streamed
        t2 = time.time()
        subject = {"fixture": a.fixture, "programs_root": a.programs_root, "components": len(dirs)}
    elif a.program:
        prog = RM.load_program(a.program)
        t0 = time.time()
        old = RM.build_manifest(prog, wl, request_id=a.request_id, rank=a.rank, ranks=a.ranks, profile=a.profile, engine="inventory")
        t1 = time.time()
        new = manifest_from_traversal(prog, wl, request_id=a.request_id, rank=a.rank, ranks=a.ranks, profile=a.profile)
        t2 = time.time()
        subject = {"program": a.program}
    else:
        raise SystemExit("diff: give --program [--workload] (B=1) or --fixture --programs-root [--workload-digest] (B>=2)")
    rec = manifest_equal(old, new)
    rec.update(subject)
    rec.update({"old_seconds": round(t1 - t0, 3), "new_seconds": round(t2 - t1, 3), "boundary_audit": new["boundary_audit"],
                "old_digest": old["manifest_digest"], "new_digest": new["manifest_digest"], "complete": new["complete"], "identities": eq_n(rec),
                "audit_flags": audit_flags(new["boundary_audit"])})
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("only_old", "only_new", "changed", "boundary_audit")}, indent=1)[:6000])
    if not rec["equal"]:
        print(json.dumps({k: rec[k] for k in ("only_old", "only_new", "changed")}, indent=1)[:6000])
    print(f"[required_values] {'EQUAL' if rec['equal'] else 'DIFF: ' + rec['why']}")
    return 0 if rec["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
