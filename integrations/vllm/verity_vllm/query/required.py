"""The required values of a request Program under `Q_module_body_v1`, spelt as v1 `IDENTITY_KEY` rows.

What this module DECIDES (structure + correspondence, no by-name table):

* WHICH Values are required: `verity_vllm.query.units.required_values(P, Q_module_body_v1, policy)` -- every Value that crosses a
  module-body boundary in the Program's own dataflow, plus the policy below.  v1's "promotion" of interior rows whose output another
  module reads is not a pass here: such a Value crosses the boundary and is required by definition.  A literal (a Call
  that reads no Value, `module_body.literal_calls`) is the descriptor's: never a boundary Value, never protocol-required.
* WHERE a Value lives: the producing Call's recorded owning module (correspondence), mapped to the implementation's path; a
  collective SITE's at its enclosing module, invocation k (`collective_sites`); at `<module>/<fx node>` when another fx node of the
  module produces a member of the same name at the step (`Population.contested`): two producers never share an identity.
* WHAT the member is called, structurally (`member_name`): the port name when the Definition returns a tuple (`<hint>.<port>` when
  the Build recorded a hint, which names the Call); else the recorded return-slot hint; else `out`.  Rows of one fx node's output at a
  step that the Program computes at several widths (a linear narrowed to the columns read, `program.prune.narrowed`) are one member
  per width: the widest rows keep the name, the narrower are `<member>@<width>` (`Population.narrowed`).  How a runtime tensor carries
  it (vLLM's return slots, a tap's label) is the binding's (`verity_vllm.query.manifest.bind`), never a spelling here.
* WHEN: the step of a Call = the number of sampling events before it (the sampling event ENDS a step).
* Shared module instances (`op_path_aliases`): two module bodies that read the SAME registered parameter and have the same body
  signature are one module instance called twice; canonical = the body the parameter is registered under; invocation = body order.

The POLICY (`Policy`, explicit, listed in the header as `query.policy`):

* prescribed inputs -- every field of the Program's record-typed parameters (the registered weights of record);
* protocol-required -- (a) the output of every sampling event (the committed sampled id), (b) RETURN-SLOT: every Value the Build
  recorded as a module's return slot (`<module>/<node>/<slot>[row]`), (c) TENSOR-RETURN: the acquisition binds the TENSORS a module
  returns, and a tensor is one fx node's output -- every row the Program unrolled it into, every port of its tuple; once any Value of
  a node's output at a step leaves its body, every Value of that node's output is required (the final norm's rows the logits never
  read, the residual slot of the final fused norm nobody consumes).  Neither (b) nor (c) is a Call the correspondence lowers to no
  runtime tensor (its fx node does not return it; `Correspondence.tensorless`) whose Values no Call reads and the Program does not
  return: the DSA indexer's MQA logits and top-k while every key is selected.  Switch (b)+(c) off (`return_slots=False`) to see the
  pure-dataflow population;
* the rank policy, a RANK Program only (result.json `tp.world` >= 2): every collective's output and its rank-local partial, with every other port
  of the partial's producer (`Population`); the row's rank manifests merge by `compose.merge_ranks` (rows keep their rank's digest,
  `tp_peer_binding` binds every output to each peer's partial).

UNRETURNED: a Call whose Values no Call reads, the Program does not return and the policy does not require (a sparse-attention
indexer's projections on its short path) computes nothing anything depends on.  It is not required, the word check skips it like a literal, and
the manifest lists each of its outputs under `unreturned`, spelt as a required one would be, so the omission is auditable.

The identity ENCODINGS, each one constant the manifest and the committing tap share: a collective's rank-local operand as
`part_rank<r>` / `shard_rank<r>` beside its output; a narrowed linear's rows as `<member>@<width>` beside the whole rows' `<member>`; a
collective SITE at (enclosing module, k) and a norm's collective under the norm (`collective_names`); the sampler rows under the fixed
path `runner.sampler`; the token ids under the consuming module as `in`; the attention hidden stream under `fa2.m1/L<k>`; the weights
as `<field>/weight`.  Kernel geometry (the FA hidden stream, the MoE block planes) and the identity constants (`SCHEMA`,
`IDENTITY_KEY`, the row order, the digest) are the manifest format's (`verity_vllm.query.manifest.format`).  A row carries its Call's
`spec`, `port` and `position` (Program order) for the binding, a narrowed one the member it narrows and its columns of those rows
(`narrowed_of`, `columns`), and one whose Values start at row r0 > 0 of their runtime tensor those rows (`runtime_rows`), outside the digest.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable, Mapping, Sequence

from verity.ml.circuits import partition_object as PO
from verity_vllm.query.units import Policy, RequiredValues, ValueRef, partition_by, required_values
from verity_vllm.query.manifest import format as fmt
from verity_vllm.program.sampling_event import GREEDY_SELECTS, SAMPLING_EVENT_FAMILIES
from verity_vllm.correspondence.reader_for_query import CallName, Correspondence, runtime_rows
from verity.ml.definitions import word as word_query
from verity_vllm.query.collective_names import NORM_SITE_MEMBER, collective_sites, norm_sites
from verity_vllm.query.module_body import (Q_MODULE_BODY_ID, Q_MODULE_BODY_TEXT, Q_module_body_v1, boundary_without_literals, literal_calls,
                                           module_body_boundary)
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView, param_reads, reads_any_of

__all__ = ["SAMPLING_EVENTS", "STOCHASTIC_SAMPLING_EVENTS", "FLASH_ATTENTION_LAUNCHES", "collective_kind",
           "member_name", "shared_instances", "collective_sites", "request_manifest", "BridgeResult", "under",
           "CallMember", "Population", "population", "Q_WORD_ID", "Q_WORD_TEXT", "QUERY_OF_RECORD", "QUERIES", "partition_of"]

#: the query of record: every Call cut into word units (`verity.ml.circuits.cut`); the Program's partition is the `verity/partition/v1` object
#: naming it (`partition_of`).  `Q_module_body_v1` stays selectable, for A/B only.
Q_WORD_ID = word_query.query_id(16, 32)
Q_WORD_TEXT = (f"{Q_WORD_ID}: units = the word units of every Call's cut (verity.ml.circuits.cut; {PO.FORMAT} names the partition as Q_word v1 "
               "{X: 16, W: 32}); committed = every Value one Call produces and another Call reads or the Program returns, and the prescribed "
               "inputs a Call reads (U_C (in(C) u out(C)) over the Calls), minus the Program's literals (the Values of Calls that read no "
               "Value: written in the descriptor, never acquired); plus each Call's committed interior words (the tap families)")
QUERY_OF_RECORD = Q_WORD_ID
QUERIES = (Q_WORD_ID, Q_MODULE_BODY_ID)


def partition_of(program_digest: str) -> dict[str, Any]:
    """The Program's partition of record: the `verity/partition/v1` object (Q_word v1, X = 16, W = 32) and its digest."""
    obj = {"format": PO.FORMAT, "program": program_digest, "query": PO.query(16, 32)}
    return {"object": obj, "digest": PO.digest(obj)}

# ---------------------------------------------------------------------------------------------------------------------------------
# protocol declarations: exact Definition ids (what a Definition IS for the protocol; never a pattern over names)
# ---------------------------------------------------------------------------------------------------------------------------------

#: the sampling event that ends an engine step for one request (verity_vllm.program.sampling_event + the explicit-argmax selector)
SAMPLING_EVENTS = frozenset(SAMPLING_EVENT_FAMILIES) | {"ArgmaxTokenSelect_v1"}
#: the sampling events that consume a registered runtime input (seed / splits) beside the logits
STOCHASTIC_SAMPLING_EVENTS = frozenset(SAMPLING_EVENT_FAMILIES) - set(GREEDY_SELECTS)
#: Definitions implemented by ONE FlashAttention launch per (layer, step) whose M1 stream the protocol requires; their binding record
#: carries the launch geometry (T, NH, KVH, D)
FLASH_ATTENTION_LAUNCHES = frozenset({"Attention_v1", "Attention_v2", "Attention_v3", "Attention_v4", "Attention_v5", "Attention_v10", "Attention_v11", "Attention_v18",
                                      "AttentionSoftcap_v1", "AttentionSoftcap_v2", "AttentionN_v1", "AttentionTP2_v1"})
#: the parameters of a `registered_values` Program's supplied inputs (`Population.supplied`): `values`, and `run` / `run_<i>` per request
SUPPLIED_PARAM = re.compile(r"^(values|run(_\d+)?)$")


def collective_kind(family: str) -> str | None:
    """The tensor-parallel collective a Definition family is, any world: "all_reduce" | "all_gather" | None -- the registry's composites
    decide (AllReduce2_v1 / AllReduce_v2{WORLD,N} / AllGather2_v1 / AllGather_v1{WORLD,N}); imported on first use (only a rank Program asks)."""
    from verity_vllm.program.frontend.collective_kinds import collective_kind as kind
    return kind(family)


def collective_trailing(family: str) -> int:
    """The operands a collective reads after its `parts` (`collective_kinds.trailing_operands`)."""
    from verity_vllm.program.frontend.collective_kinds import trailing_operands
    return trailing_operands(family)

def under(path: str, module: str) -> bool:
    """Is `path` registered under `module` in the module tree (`module` is a proper dotted prefix of `path`)?"""
    if not module:
        return False
    a, b = path.split("."), module.split(".")
    return len(a) > len(b) and a[: len(b)] == b


def member_name(call: CallView, port: Port, name: CallName) -> str:
    """The member of the Value `port` of `call`: its port name when the Definition returns several (`<hint>.<port>` when the Build
    recorded a hint, which names the Call: two multi-port Calls of one module are told apart by it), else the recorded hint, else `out`."""
    if len(call.port_table) > 1:
        return f"{name.hint}.{port.name}" if name.hint else port.name
    return name.hint or "out"


# ---------------------------------------------------------------------------------------------------------------------------------
# shared module instances
# ---------------------------------------------------------------------------------------------------------------------------------

def shared_instances(P: ProgramView, corr: Correspondence, record_fields: Mapping[ValueRef, str]) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """module -> (canonical module, invocation) for module bodies that are one shared instance, and the faults.

    Two bodies are one instance when they read the SAME non-empty set of registered parameters and have the same body signature (the
    (Definition, hint) of their Calls); the canonical body is the one a shared parameter is registered under (vLLM registers a shared
    module under the first path that holds it); invocation = the body's order among the group.  Bodies under one registered owner
    need only the same set of (Definition, hint), not the same counts: `dead_free` drops one invocation's dead rows, as the last
    layer's prompt queries before the last, never its body.  Without one, only bodies equal Call for Call are one instance.
    Fail-closed: a group with no or several canonical candidates is not aliased and is reported."""
    reads: dict[str, set[str]] = defaultdict(set)
    sig: dict[str, Counter] = defaultdict(Counter)
    first: dict[str, int] = {}
    for c in P.calls:
        m = corr.module_of(c.id)
        first.setdefault(m, c.id)
        sig[m][(c.definition, corr.name(c.id).hint)] += 1
        for v in param_reads(c):                          # record_fields keys are prescribed-input Values: only those reads can hit
            f = record_fields.get(v)
            if f is not None:
                reads[m].add(f)
    groups: dict[tuple[frozenset[str], tuple], list[str]] = defaultdict(list)
    for m, fs in reads.items():
        if fs and m:
            # keyed by (parameters read, body signature): a body with another signature reading the same parameter is a different
            # logical producer (a tied embedding read by the logits Gemm), never an invocation of this one
            groups[(frozenset(fs), frozenset(sig[m]))].append(m)
    out: dict[str, tuple[str, int]] = {}
    faults: list[str] = []
    for (fs, _sig), group in groups.items():
        by_count: dict[frozenset, list[str]] = defaultdict(list)
        for m in group:
            by_count[frozenset(sig[m].items())].append(m)
        for mods in [group] if sum(any(under(f, m) for f in fs) for m in group) == 1 else by_count.values():
            if len(mods) < 2:
                continue
            mods = sorted(mods, key=first.__getitem__)
            canon = [m for m in mods if any(under(f, m) for f in fs)]
            if len(canon) != 1:
                faults.append(f"shared parameters {sorted(fs)[:2]} read by {len(mods)} identical bodies with {len(canon)} registered owner(s): not aliased")
                continue
            c0 = canon[0]
            rest = [m for m in mods if m != c0]
            for k, m in enumerate([c0] + rest):
                out[m] = (c0, k)
    return out, faults


# ---------------------------------------------------------------------------------------------------------------------------------
# the request manifest
# ---------------------------------------------------------------------------------------------------------------------------------

DTYPE_OF_WIDTH = {16: "bf16", 32: "f32", 8: "w8", 64: "i64"}


def _dtype(width: int | None, *, weight: bool = False) -> str:
    if width is None:
        return "mixed"
    if weight:
        return {8: "w8", 32: "f32", 16: "bf16"}.get(width, f"w{width}")
    return DTYPE_OF_WIDTH.get(width, f"w{width}")


def _as_list(x: Any) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


@dataclass
class BridgeResult:
    manifest: dict[str, Any]
    required: RequiredValues
    unreturned: frozenset[int] = frozenset()     # `Population.unreturned`: dead Calls, which the word check skips like literals


# ---------------------------------------------------------------------------------------------------------------------------------
# the population: RequiredValues(P, Q_module_body_v1, policy) per Call -- what the manifest counts and what the Commit addresses
# ---------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallMember:
    """One required Value a Call produces, spelt as the manifest names it: `port`, `member`, `width` (elements per row), `dtype` and the
    manifest `family` (`tp_rank_partials` for a collective's rank-local partial)."""

    port: str
    member: str
    width: int
    dtype: str
    family: str = "instance_outputs"


class Population:
    """The required values of one request Program under the query of record and the policy (module docstring), with the facts every reader
    of them needs: the step of a Call (sampling events end steps), the record-typed parameters, the boundary, and -- per Call -- the
    required Values it produces under their manifest member names (`members_of`).  `request_manifest` groups these into identity rows;
    the Commit's replay (`sampled_replay.ProgramIndex`) addresses a Call's committed output by the same members, so the rows a manifest
    counts and the rows the replay reads are one computation.

    A RANK Program (`tp` = {"world": W >= 2, "rank": r}, result.json's) adds the rank policy: every collective's output is
    protocol-required, and so is its rank-local PARTIAL -- the one Value of this rank the collective reads (its other operands are
    `peers_<k>` Program inputs read by collectives only: the peer ranks' partials, received).  A partial produced in the collective's own
    module body is that producer's required Value, spelt `part_rank<r>` / `shard_rank<r>`, and so is every other port of the producer
    (the committer binds the Call's outputs together); a collective whose operand another body produced
    (a collective site reading a slice of it) names the partial itself, beside its output.

    `query` is the query of record (`Q_WORD_ID`: the boundary is every Call's) or `Q_MODULE_BODY_ID` (A/B only: module bodies).  Under
    Q_word a Value two Calls of one module body exchange is required too (`interior`, family `call_boundaries`; a fused-MoE block row's
    Values are committed as its `moe_block_stream` plane, `format.moe_plane_kind` under `format.moe_plane_modules`), spelt under
    `<module>/<fx node>` with its structural member: serving acquires module outputs, so such a Value needs a source of its own.  TENSOR-RETURN reads the module-body boundary under
    either query: it is what a module returns."""

    def __init__(self, P: ProgramView, corr: Correspondence, *, return_slots: bool = True, tp: Mapping[str, Any] | None = None,
                 query: str = QUERY_OF_RECORD) -> None:
        if query not in QUERIES:
            raise ValueError(f"unknown query {query!r}: one of {QUERIES}")
        self.P, self.corr, self.return_slots, self.query = P, corr, return_slots, query
        self.events = [c.id for c in P.calls if c.family in SAMPLING_EVENTS]
        if not self.events:
            raise ValueError("Program has no sampling event: no step boundary")
        self.record_fields: dict[ValueRef, str] = {}
        self.record_params: list[InputView] = []
        #: a `registered_values` Program's supplied inputs (its `values` and `run_<i>` fields): prescribed, never a weight, the prompt or a
        #: committed word -- registration binds them (the build's supplied layout), so no identity of the manifest names them
        self.supplied: dict[ValueRef, str] = {}
        for p in P.params:
            if SUPPLIED_PARAM.match(p.name):
                for port in p.port_table:
                    self.supplied[ValueRef(p.id, port.name)] = port.name
                continue
            if len(p.port_table) > 1 or p.type_repr[:1] == "{":
                self.record_params.append(p)
                for port in p.port_table:
                    self.record_fields[ValueRef(p.id, port.name)] = port.name
        prescribed = frozenset(self.record_fields) | frozenset(self.supplied)
        #: a rank Program's WITNESS input (`format.SCHEDULE_PARAM`): its readers' keys, which the replay binds to each reader's own
        #: committed rank -- never token ids, and no identity of the manifest names it
        self.witness: frozenset[ValueRef] = frozenset(ValueRef(p.id, port.name) for p in P.params if p.name == fmt.SCHEDULE_PARAM
                                                      for port in p.port_table)
        self.witness_readers: list[int] = [c.id for c in P.calls if any(r in self.witness for r in param_reads(c))] if self.witness else []
        self.world, self.rank = int((tp or {}).get("world") or 1), int((tp or {}).get("rank") or 0)
        self.collectives: dict[int, str] = {}              # collective Call -> "all_reduce" | "all_gather"
        self.operand: dict[int, ValueRef] = {}             # collective Call -> its rank-local operand
        self.partial_of: dict[ValueRef, int] = {}          # an operand produced in its collective's own body -> that collective
        self.carries: set[int] = set()                     # collectives naming their partial themselves (the operand is another body's)
        self.site: dict[int, str] = {}                     # a carried collective -> its site path (`Correspondence.site_path`)
        self.norm_site: dict[int, str] = {}                # a carried collective that is a norm's -> the norm module (`<norm site>`)
        self.peer_inputs: frozenset[ValueRef] = frozenset()
        self.tp_faults: list[str] = []
        for cid in self.witness_readers:
            if P.call(cid).family not in fmt.SCHEDULE_READERS:
                self.tp_faults.append(f"{P.call(cid).family} under {corr.implementation_path(corr.module_of(cid))} reads the "
                                      f"`{fmt.SCHEDULE_PARAM}` witness (only {sorted(fmt.SCHEDULE_READERS)} may: the replay binds a reader's "
                                      f"keys to its own committed rank)")
        if self.world > 1:
            self._rank_partials()
        self.literals = literals = literal_calls(P)
        self.B_module = B_module = module_body_boundary(P, corr, literals)
        if query == Q_WORD_ID:
            self.Q = partition_by(lambda c: c.id)
            self.B = B = boundary_without_literals(P, self.Q, literals)
        else:
            self.Q = Q_module_body_v1(P, corr)
            self.B = B = B_module
        step_of = self.step_of
        #: UNRETURNED: Calls whose Values no Call reads, the Program does not return and no policy below requires: nothing depends on
        #: them, so the word check skips them and the manifest lists their outputs (`unreturned_rows`).  A Call lowered to no runtime tensor
        #: (the DSA indexer's MQA logits and top-k while every key is selected) is neither return slot nor returned tensor either.
        word_B = B if query == Q_WORD_ID else boundary_without_literals(P, partition_by(lambda c: c.id), literals)
        dead = [c.id for c in P.calls if c.id not in literals and not any(ValueRef(c.id, p.name) in word_B.values for p in c.port_table)]
        self.unreturned: frozenset[int] = frozenset(dead) & corr.tensorless
        protocol: set[ValueRef] = set()
        if return_slots:
            # TENSOR-RETURN: the acquisition binds the TENSORS a module returns, and a tensor is one fx node's output -- every row the
            # Program unrolled it into, every port of its tuple.  When any Value of a node's output leaves its body (a boundary Value), the
            # module returned that tensor, so every Value of the node's output at that step is required (the final norm's rows the logits
            # never read, the residual slot of a fused norm nobody consumes).
            returned_nodes: set[tuple[int, str, str]] = set()
            for v in B_module.values:
                if v.call >= 0:
                    n = corr.name(v.call)
                    returned_nodes.add((step_of(v.call), n.module, n.node))
        for c in P.calls:
            if c.id in literals:            # never protocol-required, whatever node / hint it is named under (a pad guard's `<node>/eos` beside its I32Eq rows)
                continue
            if c.family in SAMPLING_EVENTS or c.id in self.collectives:         # a collective's output is required on every rank
                protocol.update(ValueRef(c.id, p.name) for p in c.port_table)
            elif return_slots and c.id not in self.unreturned:
                n = corr.name(c.id)
                if n.hint or (step_of(c.id), n.module, n.node) in returned_nodes:      # RETURN-SLOT (recorded) / TENSOR-RETURN (structural)
                    protocol.update(ValueRef(c.id, p.name) for p in c.port_table)
        # the rank-local partials, each with every other port of its producer (a row-parallel linear that reduces in its own body
        # returns its quantised input and scales beside the partial: `DeepGemmFp8Linear_v1`)
        protocol.update(ValueRef(v.call, p.name) for v in self.partial_of for p in P.call(v.call).port_table)
        self.unreturned = frozenset(cid for cid in dead if not any(ValueRef(cid, p.name) in protocol for p in P.call(cid).port_table))
        self.policy = Policy(prescribed_inputs=prescribed, protocol_required=frozenset(protocol))
        self.required = RequiredValues(B.values, self.policy.prescribed_inputs, self.policy.protocol_required)
        inside = [v for v in B.values if v.call >= 0 and v not in B_module.values and v not in self.policy.protocol_required]
        # a MoE block row's Value inside its body is committed as the row's `moe_block_stream` plane: no member (a member of the body's
        # `out` would be a second producer beside the block's own output, or beside its AllReduce's on a rank Program)
        planes = fmt.moe_plane_modules((c.family, corr.module_of(c.id)) for c in P.calls)
        self.plane_fed = frozenset(v for v in inside if (m := corr.module_of(v.call)) in planes
                                   and fmt.moe_plane_kind(P.call(v.call).family, m) is not None)
        self.interior = frozenset(inside) - self.plane_fed
        # (required_values() would recompute the boundary; the policy is validated the same way here)
        known_inputs = frozenset(P.inputs)
        for v in self.policy.prescribed_inputs:
            if v not in known_inputs:
                raise ValueError(f"policy prescribes {v}, which is not an input")
        self._ports_of: dict[int, list[str]] | None = None

    def _rank_partials(self) -> None:
        """The rank policy on a rank Program: the collectives, their rank-local operands (the partial is the operand's producer's Value when that
        producer is in the collective's body) and the received peer partials (non-record Program inputs every reader of which is a collective)."""
        P, corr = self.P, self.corr
        kinds: dict[str, str | None] = {}
        collective_only: dict[ValueRef, bool] = {}
        for c in P.calls:
            if c.family not in kinds:
                kinds[c.family] = collective_kind(c.family)
            k = kinds[c.family]
            for r in param_reads(c):
                if r not in self.record_fields and r not in self.supplied:
                    collective_only[r] = collective_only.get(r, True) and k is not None
            if k is None:
                continue
            self.collectives[c.id] = k
            parts = c.reads[: len(c.reads) - collective_trailing(c.family)]     # the fused norm's residual and weight follow its parts
            # the rank's own part: one Value, or the M row Values a flat collective (`AllReduceRule`: one Call over M rows) reshapes
            own = [r for r in parts if r.call >= 0]
            where = corr.implementation_path(corr.module_of(c.id))
            inside = {corr.module_of(r.call) == corr.module_of(c.id) for r in own}
            if not own or len(inside) != 1:
                self.tp_faults.append(f"{c.family} under {where}: {len(own)} rank-local operands"
                                      f"{', some produced in its body and some outside it' if own else ''} (one part expected)")
                continue
            self.operand[c.id] = own[0]
            if inside == {True}:
                if any(op in self.partial_of for op in own):
                    self.tp_faults.append(f"{P.call(own[0].call).family} under {where}: one Value is the rank-local partial of two collectives")
                    continue
                self.partial_of.update(dict.fromkeys(own, c.id))
            elif (len({p.elems for p in c.port_table}) != 1 if k == "all_reduce" else
                  len(c.port_table) != 1 or c.port_table[0].elems % self.world):
                self.tp_faults.append(f"{c.family} under {where}: an operand from another body and output ports "
                                      f"{[(p.name, p.elems) for p in c.port_table]} (all_reduce: outputs of one width; all_gather: one "
                                      f"output, a multiple of world {self.world}, expected)")
            else:
                self.carries.add(c.id)
                self.site[c.id] = corr.site_path(c.id)
        self.peer_inputs = frozenset(v for v, only in collective_only.items() if only)
        self._norm_sites()

    def _norm_sites(self) -> None:
        """The carried collectives that are a norm's (`collective_names.norm_sites`) leave the collective sites: their identities are
        the norm's, at invocation 0.  A carried collective it refuses is a fault."""
        self.norm_site, refused = norm_sites(self.P, self.corr, self.record_fields, self.carries, self.collectives, collective_trailing, self.step_of)
        for cid in self.norm_site:
            del self.site[cid]
        for cid in refused:
            self.tp_faults.append(f"{self.P.call(cid).family} at {self.corr.implementation_path(self.site[cid])}: not one norm's collective "
                                  f"(one `<norm>.weight` read, and no other collective of the step for that norm unless all are fused, expected)")

    def _partial(self, port: str, kind: str, width: int, dtype: str) -> CallMember:
        member = (fmt.TP_SHARD_MEMBER if kind == "all_gather" else fmt.TP_PARTIAL_MEMBER).format(rank=self.rank)
        return CallMember(port, member, width, dtype, "tp_rank_partials")

    def step_of(self, cid: int) -> int:
        """The step of a Call: the number of sampling events before it (the event itself belongs to the step it ends)."""
        return bisect.bisect_left(self.events, cid)

    def program_path(self, cid: int) -> str:
        """The Program module path the required Values of Call `cid` are spelt under: its site path for a collective SITE, `<module>/<fx
        node>` for a Call whose required Values are all Call boundaries inside its module body or whose member another fx node of its
        module produces at its step (`contested`), else its owning module.  `collective_sites` aliases a site path to its (enclosing
        module, k)."""
        if cid in self.interior_calls or cid in self.contested:
            return f"{self.corr.module_of(cid)}/{self.corr.name(cid).node}"
        return self.norm_site.get(cid) or self.site.get(cid) or self.corr.module_of(cid)

    @functools.cached_property
    def interior_calls(self) -> frozenset[int]:
        """The Calls all of whose required Values are `interior`."""
        return frozenset({v.call for v in self.interior} - {v.call for v in self.required.values if v.call >= 0 and v not in self.interior})

    @functools.cached_property
    def contested(self) -> frozenset[int]:
        """The Calls spelt at their fx node though they are spelt at their module otherwise: two fx nodes of one module produce a member
        of the same name at one step (a compressed attention's kv_score Gemm and wo_a einsum, both `out` of the attention module; a
        layer's two mHC post-pres).  Two producers never share an identity, and the binder tells them apart by Definition and port, in
        Program order.  The UNRETURNED Calls count, so an omission is never spelt as a required Value."""
        keyed: list[tuple[int, tuple[int, str, str]]] = []
        for cid in self.producers():
            if cid in self.interior_calls or cid in self.site or cid in self.norm_site:
                continue
            keyed += [(cid, (self.step_of(cid), self.corr.module_of(cid), cm.member)) for cm in self.members_of(cid)]
        for cid in self.unreturned:
            c = self.P.call(cid)
            keyed += [(cid, (self.step_of(cid), self.corr.module_of(cid), member_name(c, p, self.corr.name(cid)))) for p in c.port_table]
        nodes: dict[tuple[int, str, str], set[str]] = defaultdict(set)
        for cid, key in keyed:
            nodes[key].add(self.corr.name(cid).node)
        return frozenset(cid for cid, key in keyed if len(nodes[key]) > 1)

    def member(self, v: ValueRef) -> CallMember:
        """The manifest member of the required Value `v` of a (non-sampling) Call."""
        c = self.P.call(v.call)
        port = c.port(v.port)
        coll = self.partial_of.get(v)
        if coll is not None:
            return self._partial(port.name, self.collectives[coll], port.elems, _dtype(port.width))
        member = self.narrowed.get(v) or member_name(c, port, self.corr.name(c.id))
        if v in self.interior:
            return CallMember(port.name, member, port.elems, _dtype(port.width), "call_boundaries")
        if c.id in self.norm_site:                       # the norm's own outputs: its ports, or the one all-reduced tensor
            return CallMember(port.name, port.name if len(c.port_table) > 1 else NORM_SITE_MEMBER, port.elems, _dtype(port.width))
        return CallMember(port.name, member, port.elems, _dtype(port.width))

    @functools.cached_property
    def narrowed(self) -> dict[ValueRef, str]:
        """The required Values of the rows a Program narrowed, each -> its member: one fx node's Calls of one Definition at a step whose
        rows have several widths (`program.prune.narrowed` binds a linear whose output only some columns of are read at the columns
        read: the last layer's fused qkv on the prompt rows before the last, whose queries nothing reads, beside the last row's whole
        qkv).  The widest rows keep the member and each narrower width is its own, `format.NARROWED_MEMBER` (`out@1024`), so the key and
        value rows never stand for the whole rows or the reverse.  Calls of several Definitions under one name stay one member, which
        `request_manifest` refuses as two producers."""
        widths: dict[tuple[int, str, str, str], dict[int, list[ValueRef]]] = defaultdict(lambda: defaultdict(list))
        families: dict[tuple[int, str, str, str], set[str]] = defaultdict(set)
        for cid, ports in self._ports().items():
            c = self.P.call(cid)
            if c.family in SAMPLING_EVENTS or cid in self.collectives or cid in self.norm_site:
                continue
            n = self.corr.name(cid)
            for p in ports:
                v = ValueRef(cid, p)
                if v in self.plane_fed or v in self.partial_of:
                    continue
                port = c.port(p)
                key = (self.step_of(cid), n.module, n.node, member_name(c, port, n))
                widths[key][port.elems].append(v)
                families[key].add(c.family)
        out: dict[ValueRef, str] = {}
        for key, by_width in widths.items():
            if len(by_width) > 1 and len(families[key]) == 1:
                widest = max(by_width)
                for w, vs in by_width.items():
                    if w != widest:
                        out.update(dict.fromkeys(vs, fmt.NARROWED_MEMBER.format(member=key[3], width=w)))
        return out

    def _ports(self) -> dict[int, list[str]]:
        if self._ports_of is None:
            ports: dict[int, list[str]] = defaultdict(list)
            for v in sorted(self.required.values):
                if v.call >= 0:
                    ports[v.call].append(v.port)
            self._ports_of = ports
        return self._ports_of

    def producers(self) -> list[int]:
        """The Calls that produce at least one required Value, in body order (sampling events included)."""
        return sorted(self._ports())

    def members_of(self, cid: int) -> list[CallMember]:
        """The required Values Call `cid` produces, in port order ([] = its outputs are interior to its module body).  A sampling
        event's output is the protocol's `sampled_token_ids` (not a module member): [] here.  A collective whose rank-local operand another
        body produced names that partial after its output (width: the operand's per-rank share of the output)."""
        c = self.P.call(cid)
        if c.family in SAMPLING_EVENTS:
            return []
        out = [self.member(ValueRef(cid, p)) for p in self._ports().get(cid, ()) if ValueRef(cid, p) not in self.plane_fed]
        if cid in self.carries:
            port = c.port_table[0]
            width = port.elems if self.collectives[cid] == "all_reduce" else port.elems // self.world
            out.append(self._partial(port.name, self.collectives[cid], width, _dtype(port.width)))
        return out


def population(P: ProgramView, corr: Correspondence, *, return_slots: bool = True, tp: Mapping[str, Any] | None = None,
               query: str = QUERY_OF_RECORD) -> Population:
    """`Population(P, corr)`: the required values of one request Program under `query` + the policy (+ the rank policy on a rank Program)."""
    return Population(P, corr, return_slots=return_slots, tp=tp, query=query)


def _profile(res: Mapping[str, Any], profile: str | None) -> tuple[str, int | None, dict]:
    attn_obs = res.get("attention_impls_observed") or {}
    fa = None
    for v in attn_obs.values():
        fa = v.get("vllm_flash_attn_version")
        break
    if profile is None:
        from verity_vllm.program.frontend.target_profile import TargetProfile

        cc = ((res.get("target") or {}).get("declared") or {}).get("compute_capability") or [8, 9]
        fa3 = TargetProfile(compute_capability=(int(cc[0]), int(cc[1]))).effective_flash_attn_version() == 3
        profile = "hopper" if (fa == 3 or (fa is None and fa3)) else "ampere"
    return profile, fa, attn_obs


def hidden_family_of(P: ProgramView, profile: str) -> str | None:
    """The attention hidden-stream family the Program's flash-attention launches put in the manifest under `profile`; None when it has
    none (MLA / DSA sparse MLA: no `varlen_fwd`, so no hidden tap fires)."""
    if not any(c.family in FLASH_ATTENTION_LAUNCHES for c in P.calls):
        return None
    return "fa3_hidden_m1_stream" if profile == "hopper" else "fa2_hidden_m1_stream"


def query_header(profile: str, hidden_family: str | None, *, return_slots: bool, query: str = QUERY_OF_RECORD,
                 partition: Mapping[str, Any] | None = None, call_boundaries: bool = False) -> dict[str, Any]:
    """The manifest's query header; under Q_word, `partition` (`partition_of`) is the Program's `verity/partition/v1` object and digest,
    and `call_boundaries` says the manifest requires Values two Calls of one module body exchange."""
    word = query == Q_WORD_ID
    text = (Q_WORD_TEXT if word else Q_MODULE_BODY_TEXT) + f"; policy: prescribed = record-typed inputs (weights); protocol = sampling-event outputs" + \
        (" + recorded return slots + returned tensors (every Value of an fx node's output at a step once any of them crosses the module body boundary)"
         ", neither of which is a Call lowered to no runtime tensor whose Values nothing reads" if return_slots else "") + \
        "; unreturned = the Calls whose Values nothing reads, the Program does not return and the policy does not require, listed by identity" + \
        f"; hidden = {hidden_family or 'none'}; profile = {profile}"
    return {"query_id": query, "query_id_digest": hashlib.sha256(text.encode()).hexdigest(), "query_text": text,
            **({"partition": dict(partition)} if word and partition else {}),
            "engine": "q-word" if word else "v2-query", "boundary": "call" if word else "module-body", "profile": profile,
            "attention_hidden_family": hidden_family,
            "required_families": ["input_token_ids", "instance_outputs", "sampled_token_ids", "weights"] + ([hidden_family] if hidden_family else []) +
                                 (["call_boundaries"] if call_boundaries else []),
            "policy": {"prescribed_inputs": "every field of the Program's record-typed parameters",
                       "protocol_required": ["sampling-event outputs"] +
                       (["recorded return slots (RETURN-SLOT)", "returned tensors: all rows / ports of an fx node's output once one crosses (TENSOR-RETURN)",
                         "neither: a Call lowered to no runtime tensor whose Values no Call reads and the Program does not return"]
                        if return_slots else []),
                       "unreturned": "a Call whose Values no Call reads, the Program does not return and no policy requires: not required, "
                                     "skipped by the word check, each output listed under `unreturned` (UNRETURNED)"}}


def unreturned_rows(pop: Population, aliases: Mapping[str, tuple[str, int]], identity) -> list[dict[str, Any]]:
    """The UNRETURNED Calls' outputs, one row per (step, module, member) as a required one would be spelt (`identity`: the manifest's
    row maker): what the word check skips."""
    P, corr, impl = pop.P, pop.corr, pop.corr.implementation_path
    dead: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cid in sorted(pop.unreturned):
        c = P.call(cid)
        for k, port in enumerate(c.port_table):
            g = dead.setdefault((pop.step_of(cid), pop.program_path(cid), member_name(c, port, corr.name(cid))),
                                {"rows": 0, "width": port.elems, "dtype": _dtype(port.width), "spec": c.family, "port": port.name, "position": [cid, k]})
            g["rows"] += 1
    out = []
    for (step, mod, member), g in dead.items():
        canon, k = aliases.get(mod, (mod, 0))
        out.append(identity("instance_outputs", step, k, impl(canon), member, 0, g["rows"] * g["width"], **g,
                            **({"program_op_path": impl(mod)} if canon != mod else {})))
    return out


def group_rows(groups: Mapping[tuple[int, str, str], Mapping[str, Any]], aliases: Mapping[str, tuple[str, int]], impl, add,
               unmodelled: Counter) -> None:
    """One identity row per (step, module, member) group of Call-produced required Values (`request_manifest`'s; `add`: its row adder):
    `rows` x `width` elements under the module, or under its shared instance's canonical module at its invocation.  A group of width 0
    is unmodelled.  Where its Calls' Values lie (`Correspondence.rows`, `columns`): `runtime_rows`, and a narrowed group's `columns`."""
    for (step, mod, member), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if g["width"] <= 0:
            unmodelled[f"{g['family']} under {impl(mod)} ({member}: width 0)"] += 1
            continue
        rr, cols = runtime_rows(g["at"]), g["columns"]
        extras = {"rows": g["rows"], "width": g["width"], "dtype": g["dtype"], "spec": g["family"], "port": g["port"], "position": g["position"],
                  "fx_nodes": len(g["nodes"]), "boundary": {"crossing_rows": g["crossing"], "returned_rows": g["returned"]},
                  **({"narrowed_of": g["narrowed_of"]} if g["narrowed_of"] else {}),
                  **({"runtime_rows": rr} if rr else {}), **({"columns": list(next(iter(cols)))} if g["narrowed_of"] and len(cols) == 1 and None not in cols else {})}
        if mod in aliases:
            canon, k = aliases[mod]
            add(g["manifest_family"], step, k, impl(canon), member, 0, g["rows"] * g["width"], program_op_path=impl(mod), **extras)
        else:
            add(g["manifest_family"], step, 0, impl(mod), member, 0, g["rows"] * g["width"], **extras)


def cached_prefix_inputs(P: ProgramView, res: Mapping[str, Any]) -> frozenset[ValueRef]:
    """A warm request Program's `prefix` input (its derive's `res["prefix"]["input"]`, the wrapper's input index): rows 0..C-1 of every
    layer's key and value, the producing requests' committed rows (no identity of this request names them)."""
    pfx = res.get("prefix")
    if not isinstance(pfx, Mapping):
        return frozenset()
    p = P.params[int(pfx["input"])]
    return frozenset(ValueRef(p.id, port.name) for port in p.port_table)


def request_manifest(P: ProgramView, corr: Correspondence, res: Mapping[str, Any], *, request_id: str | int | None = None,
                     workload: Mapping[str, Any] | None = None, artifact: Mapping[str, Any] | None = None, program_dir: str | None = None,
                     rank: int = 0, ranks: int = 1, profile: str | None = None, return_slots: bool = True, norm_scales: Any = None, taps: Sequence[Any] = (), query: str = QUERY_OF_RECORD, program_digest: str | None = None) -> BridgeResult:
    """The v1-shaped manifest of ONE request Program (B=1 component) under `query` (+ `norm_scales`, + `taps`: `query.router_softmax` / `vocab_range` POLICY; + its partition given `program_digest`)."""
    pdig = (artifact or {}).get("program_digest") or res["program"]["digest"]
    profile, fa, attn_obs = _profile(res, profile)
    hidden_family = hidden_family_of(P, profile)
    tpinfo = res.get("tp") if isinstance(res.get("tp"), dict) else {}
    prog_world, prog_rank = int(tpinfo.get("world") or 1), int(tpinfo.get("rank") or 0)
    if (prog_world > 1 or ranks > 1) and (int(ranks) != prog_world or int(rank) != prog_rank):
        raise ValueError(f"rank Program mismatch: Program is rank {prog_rank} of world {prog_world}, manifest requested rank {rank} of {ranks}")
    sweep = (workload or {}).get("sweep") or {}
    wr = (workload or {}).get("requests") or []
    workload_request_id = wr[0].get("request_id") if wr and isinstance(wr[0], dict) else None
    request_id = fmt.request_label(request_id, workload)
    k_req = int(fmt.REQ_RE.match(request_id).group(1))
    if wr and k_req < len(wr) and isinstance(wr[k_req], dict) and str(wr[k_req].get("request_id")) != str(workload_request_id):
        workload_request_id = wr[k_req].get("request_id")
    arr = sweep.get("arrive_steps") or []
    arrive = int(arr[k_req] if k_req < len(arr) else (arr[0] if arr else 0)) if sweep else 0
    n_seq = 1
    # --- the query and the policy ---------------------------------------------------------------------------------------------------
    pop = (norm_scales.population if norm_scales else population)(P, corr, return_slots=return_slots, tp=({"world": prog_world, "rank": prog_rank} if prog_world > 1 else None), query=query)
    events, step_of, record_fields, record_params = pop.events, pop.step_of, pop.record_fields, pop.record_params
    B, policy, req, literals = pop.B, pop.policy, pop.required, pop.literals

    aliases, alias_faults = shared_instances(P, corr, record_fields)
    sites = collective_sites(pop)
    aliases.update(sites)
    impl = corr.implementation_path
    identities: list[dict[str, Any]] = []
    fam_counts: Counter = Counter()
    unmodelled: Counter = Counter()
    def identity(family: str, step: int | str, invocation: int, op_path: str, member: str, lo: int, hi: int | None, **extra) -> dict[str, Any]:
        row = {"family": family, "program_digest": pdig, "request_id": request_id, "step": step, "invocation": invocation,
               "op_path": op_path, "output_member": member, "element_range": [lo, hi], "rank": rank}
        if step != "*":
            row["engine_step"] = arrive + int(step)
        row.update(extra)
        return row

    def add(family: str, *args, **extra):
        identities.append(identity(family, *args, **extra))
        fam_counts[family] += 1

    # --- weights: every prescribed record field --------------------------------------------------------------------------------------
    for p in record_params:
        for port in sorted(p.port_table, key=lambda q: q.name):
            add("weights", "*", 0, impl(port.name), "weight", 0, None, dtype=_dtype(port.width, weight=True))

    # --- the Call-produced required Values, grouped per (step, module, member) -----------------------------------------------------
    groups: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cid in pop.producers():
        c = P.call(cid)
        n = corr.name(c.id)
        mod = pop.program_path(c.id)
        step = step_of(c.id)
        if c.family in SAMPLING_EVENTS:
            add("sampled_token_ids", step, 0, "runner.sampler", "sampled_token_ids", 0, n_seq, dtype="i64", spec=c.family)
            for r in c.reads:                                   # the runtime inputs the selector consumes (seed, splits): the sampler's
                if r.call < 0 and r not in record_fields and r not in pop.supplied:
                    pin = P.input(r.call)
                    add(f"sampler_{pin.name}", step, 0, "runner.sampler", pin.name, 0, n_seq, dtype=_dtype(pin.port_table[0].width), spec=c.family)
            continue
        for cm in pop.members_of(cid):
            carried = cm.family == "tp_rank_partials" and cid in pop.carries
            v = pop.operand[cid] if carried else ValueRef(cid, cm.port)
            member = cm.member
            g = groups.get((step, mod, member))
            if g is None:
                g = groups[(step, mod, member)] = {"rows": 0, "width": cm.width, "dtype": cm.dtype, "family": c.family, "spec": c.definition,
                                                   "manifest_family": cm.family, "nodes": set(), "port": cm.port,
                                                   "position": [cid, [p.name for p in c.port_table].index(cm.port)], "crossing": 0, "returned": 0,
                                                   "narrowed_of": member_name(c, c.port(cm.port), n) if v in pop.narrowed else None,
                                                   "at": [], "columns": set()}
            elif g["family"] != c.family or g["width"] != cm.width:
                unmodelled[f"{c.family} under {impl(mod)} ({member}: two producers {g['family']} {g['width']} / {c.family} {cm.width})"] += 1
                continue
            g["rows"] += 1
            g["nodes"].add(n.node)
            g["at"].append(None if carried else corr.rows.get(cid))
            g["columns"].add(corr.columns.get(cid))
            if v in B.values:
                g["crossing"] += 1
            if carried or v in policy.protocol_required:
                g["returned"] += 1

    group_rows(groups, aliases, impl, add, unmodelled)
    unreturned = unreturned_rows(pop, aliases, identity)

    # --- the token ids a module body consumes: prompt input or a sampling event's output -------------------------------------------
    ids_rows: Counter = Counter()
    event_set = frozenset(events)
    non_ids = pop.peer_inputs | pop.witness | cached_prefix_inputs(P, res)    # received peer partials, the witness, a warm Program's cached rows
    for c in P.calls:
        m = corr.module_of(c.id)
        if not m or c.family in SAMPLING_EVENTS:            # a selector's runtime inputs are the sampler's (above), wherever it sits (TP: `inner`)
            continue
        # a non-weight prescribed input (the prompt ids) or a sampling event's output among the reads
        if any(r not in record_fields and r not in non_ids and r not in pop.supplied for r in param_reads(c)) or reads_any_of(c, event_set):
            ids_rows[(step_of(c.id), m)] += 1
    for (step, mod), rows in sorted(ids_rows.items()):
        add("input_token_ids", step, 0, impl(mod), "in", 0, rows, rows=rows, width=1, dtype="i32")

    # --- the attention hidden stream: one launch per (attention body, step) ------------------------------------------------------
    attn_bodies: list[str] = []
    attn: dict[tuple[int, str], dict[str, Any]] = {}
    for c in P.calls:
        if c.family in FLASH_ATTENTION_LAUNCHES:
            m = corr.module_of(c.id)
            if m not in attn_bodies:
                attn_bodies.append(m)
            st = c.statics
            a = attn.setdefault((step_of(c.id), m), {"rows": 0, "end": 0, "T_max": 0, "NH": int(st.get("NH", 0)), "KVH": int(st.get("KVH", st.get("NH", 0))), "D": int(st.get("D", 0))})
            a["rows"] += 1
            a["end"] = max(a["end"], corr.rows.get(c.id, -1) + 1)     # the launch runs the query rows `dead_free` dropped (`runtime_rows`)
            a["T_max"] = max(a["T_max"], int(st.get("T", 0)))
    tap = "fa3.m1" if (hidden_family or "")[:3] == "fa3" else "fa2.m1"
    for (step, mod), a in sorted(attn.items(), key=lambda kv: (kv[0][0], attn_bodies.index(kv[0][1]))):
        layer, rows = attn_bodies.index(mod), max(a["rows"], a["end"])
        if hidden_family[:3] == "fa3":
            geo = fmt.fa3_geometry(a["T_max"], rows, a["NH"], a["KVH"], a["D"], n_seq)
        else:
            win = (attn_obs.get(impl(mod)) or attn_obs.get(mod) or {}).get("sliding_window")
            w0 = win[0] if isinstance(win, (list, tuple)) and win else (win if isinstance(win, int) else -1)
            geo = fmt.fa2_geometry(a["T_max"], rows, a["NH"], a["KVH"], a["D"], n_seq, windowed=(w0 is not None and int(w0) >= 0))
        HB, M, NB, BN, gname = geo
        add(hidden_family, step, 0, f"{tap}/L{layer}", gname, 0, fmt.mat_total_words(HB, M, NB, BN, a["D"]), dtype="u32",
            geometry={"HB": HB, "M": M, "NB": NB, "BN": BN, "D": a["D"], "seqlen_k": a["T_max"], "rows": rows}, attention_module=impl(mod))

    # --- MoE block planes (kernel geometry, v1's): per (MoE body, step) ----------------------------------------------------------------
    moe_layers: set[int] = set()
    lo = 0
    for step, ev in enumerate(events):                     # the step's Calls are the body slice up to and including its sampling event
        seg = P.calls[lo: ev + 1]
        lo = ev + 1
        for mod, g in fmt.moe_geometry({"spec": c.definition, "name": c.name} for c in seg).items():
            layer = fmt.layer_of(mod)
            if g["M"] <= 0 or g["TOPK"] <= 0:
                continue
            moe_layers.add(layer)
            for plane, nw, dtype in fmt.moe_identities(g):
                add("moe_block_stream", step, 0, f"moe/L{layer}", plane, 0, nw, dtype=dtype, rows=g["M"], topk=g["TOPK"], moe_module=impl(mod))

    for f in alias_faults + pop.tp_faults:
        unmodelled[f] += 1
    scales = norm_scales.kernel_rows(pop, add, impl, aliases) if norm_scales else None     # the norm-scale tap's rows: interior to the kernel norms
    tapped = [(t, t.kernel_rows(pop, add, impl, aliases)) for t in taps]                   # the other taps' rows: interior to their Calls
    identities.sort(key=fmt.row_key)
    q = query_header(profile, hidden_family, return_slots=return_slots, query=query, partition=partition_of(program_digest) if program_digest and query == Q_WORD_ID else None, call_boundaries=bool(fam_counts.get("call_boundaries")))
    if prog_world > 1:
        q["required_families"].append("tp_rank_partials")
        q["policy"]["protocol_required"].append(f"TP-04 (world {prog_world}): every collective output and its rank-local partial, with every "
                                                "port of the partial's producer")
        q["tp"] = {"world": prog_world, "collectives": len(pop.collectives), "peer_inputs": len(pop.peer_inputs)}
        if pop.witness:
            q["tp"]["witness_inputs"] = {fmt.SCHEDULE_PARAM: {"readers": len(pop.witness_readers),
                                                              "binding": "each reader's own committed rank (port 0)"}}
    if moe_layers:
        q["required_families"].append("moe_block_stream")
        q["moe_layers"] = sorted(moe_layers)
    if any(f[:8] == "sampler_" for f in fam_counts):
        q["required_families"] += sorted(f for f in fam_counts if f[:8] == "sampler_")
    alias_out = {impl(m): [impl(canon), k] for m, (canon, k) in aliases.items() if m != canon}
    man = {"schema": fmt.SCHEMA, "identity_key": list(fmt.IDENTITY_KEY), "program_digest": pdig,
           "program": {"dir": program_dir, "function": res["program"].get("function"), "wrapper": res.get("wrapper"), "LP": res.get("LP"), "T": res.get("T"),
                       "model_pin": res.get("model_pin"), "artifact_identity": (artifact or {}).get("identity"), "instances": len(P.calls),
                       "instances_digest": P.digest, "source": P.source},
           "correspondence": corr.provenance(),                 # which reader gave the owning modules: the runtime-correspondence/v1 record or the v1 annotations
           "workload": {"row_id": sweep.get("row_id"), "batch": sweep.get("batch"), "request_id": request_id, "workload_request_id": workload_request_id,
                        "arrive_step": arrive} if workload else None,
           "query": reduce(lambda h, ts: ts[0].header(h, ts[1]), tapped, norm_scales.header(q, scales) if norm_scales else q), "wrapper_prefix": corr.prefix, "op_path_aliases": alias_out, "target": ((res.get("target") or {}).get("declared") or {}),
           "flash_attn_version": fa, "ranks": ranks, "rank": rank,
           "steps": list(range(len(events))), "engine_steps": [arrive + s for s in range(len(events))], "requests": [request_id],
           "populations": {"identities": len(identities), "by_family": dict(fam_counts), "steps": len(events),
                           "attention_layers": len(attn_bodies), "units": len(B.units), "boundary_values": len(B.values),
                           "required_values": len(req.values), "literal_calls": len(literals), "unreturned_calls": len(pop.unreturned)},
           "complete": not unmodelled, "unmodelled": dict(unmodelled),
           "manifest_digest": fmt.manifest_digest(identities), "identities": identities, **fmt.unreturned_block(unreturned)}
    return BridgeResult(man, req, pop.unreturned)
