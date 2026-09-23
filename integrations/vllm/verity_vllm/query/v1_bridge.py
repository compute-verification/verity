"""The required values of a request Program under `Q_module_body_v1`, spelt as v1 `IDENTITY_KEY` rows.

What this module DECIDES (structure + correspondence, no by-name table):

* WHICH Values are required: `verity.verification.query.required_values(P, Q_module_body_v1, policy)` -- every Value that crosses a
  module-body boundary in the Program's own dataflow, plus the policy below.  v1's "promotion" of interior rows whose output another
  module reads (F-r19-int-20) is not a pass here: such a Value crosses the boundary and is required by definition.
* WHERE a Value lives: the producing Call's recorded owning module (correspondence), mapped to the implementation's path.
* WHAT the member is called, structurally: the port name when the Definition returns a tuple; else the recorded return-slot hint
  when the Build recorded one; else `out`.
* WHEN: the step of a Call = the number of sampling events before it (the sampling event ENDS a step).
* Shared module instances (`op_path_aliases`): two module bodies that read the SAME registered parameter and have the same body
  signature are one module instance called twice; canonical = the body the parameter is registered under; invocation = body order.

The POLICY (`Policy`, explicit, listed in the header as `query.policy`):

* prescribed inputs -- every field of the Program's record-typed parameters (the registered weights of record);
* protocol-required -- (a) the output of every sampling event (the committed sampled id), (b) RETURN-SLOT: every Value the Build
  recorded as a module's return slot (`<module>/<node>/<slot>[row]`), (c) TENSOR-RETURN: the acquisition binds the TENSORS a module
  returns, and a tensor is one fx node's output -- every row the Program unrolled it into, every port of its tuple; once any Value of
  a node's output at a step leaves its body, every Value of that node's output is required (the final norm's rows the logits never
  read, the residual slot of the final fused norm nobody consumes).  Switch (b)+(c) off (`return_slots=False`) to see the
  pure-dataflow population.

What this module cannot derive and therefore RECORDS (`NAMED_RESIDUALS`, emitted in the header, counted per row):

* v1 member spellings that come from vLLM's Python return convention, not from the Definition or the recorded hint (`Gemm_v1`
  under a Linear -> `"0"`: `LinearBase.forward` returns `(output, output_bias)`; `RoPE_v1` hint `query` -> `"0"`, `key` -> `"1"`);
* v1 identity ENCODINGS: the sampler rows under the fixed path `runner.sampler`, the token ids under the consuming module as `in`,
  the attention hidden stream under `fa2.m1/L<k>`, the weights as `<field>/weight`.

Kernel geometry (the FA hidden stream, the MoE block planes) and the identity encoding constants (`SCHEMA`, `IDENTITY_KEY`, the row
order, the digest) are the manifest format's (`verity_capture.commit.manifest_format`) -- they are what the bridge is a bridge TO, not
membership decisions.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from verity.verification.query import Policy, RequiredValues, ValueRef, boundary, required_values
from verity_capture.commit import manifest_format as fmt
from verity_vllm.ir.sampling_event import SAMPLING_EVENT_FAMILIES

from .correspondence import CallName, Correspondence
from .module_body import Q_MODULE_BODY_ID, Q_MODULE_BODY_TEXT, Q_module_body_v1
from .program_view import CallView, InputView, Port, ProgramView, param_reads, reads_any_of

__all__ = ["Residual", "NAMED_RESIDUALS", "SAMPLING_EVENTS", "STOCHASTIC_SAMPLING_EVENTS", "FLASH_ATTENTION_LAUNCHES", "COLLECTIVES",
           "member_name", "shared_instances", "request_manifest", "compose_global", "BridgeResult", "under"]

# ---------------------------------------------------------------------------------------------------------------------------------
# protocol declarations: exact Definition ids (what a Definition IS for the protocol; never a pattern over names)
# ---------------------------------------------------------------------------------------------------------------------------------

#: the sampling event that ends an engine step for one request (verity_vllm.ir.sampling_event + the explicit-argmax selector)
SAMPLING_EVENTS = frozenset(SAMPLING_EVENT_FAMILIES) | {"ArgmaxTokenSelect_v1"}
#: the sampling events that consume a registered runtime input (seed / splits) beside the logits
STOCHASTIC_SAMPLING_EVENTS = frozenset({"GumbelTokenSelect_v1", "GumbelTopPTokenSelect_v1"})
#: Definitions implemented by ONE FlashAttention launch per (layer, step) whose M1 stream the protocol requires; their binding record
#: carries the launch geometry (T, NH, KVH, D)
FLASH_ATTENTION_LAUNCHES = frozenset({"Attention_v1", "Attention_v2", "Attention_v3", "AttentionSoftcap_v1", "AttentionN_v1", "AttentionTP2_v1"})
#: tensor-parallel collectives (rank Programs)
COLLECTIVES = {"AllReduce2_v1": "all_reduce", "AllGather2_v1": "all_gather"}

# ---------------------------------------------------------------------------------------------------------------------------------
# named residuals: v1 spellings the structure does not give
# ---------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Residual:
    """A v1 member name the structure cannot derive.  `family`: exact Definition base id; `structural`: the member the structure gives
    (port name / recorded hint / `out`); `condition`: an extra structural predicate the residual applies under (`own_parameter`: the Call
    reads a registered parameter that is registered UNDER its owning module -- a Linear reading its own weight); `why`: the reason."""

    family: str
    structural: str
    v1_member: str
    why: str
    condition: str | None = None
    kind: str = "member"                         # member | encoding

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "structural": self.structural, "v1_member": self.v1_member, "condition": self.condition, "kind": self.kind, "why": self.why}


_LINEAR_WHY = ("vLLM `LinearBase.forward` returns `(output, output_bias)`; the collector binds slot 0.  The Program's Definition returns one "
               "array (`out`) and the Build recorded no return slot for the row -- the tuple slot is Python's, invisible to the Program")
NAMED_RESIDUALS: tuple[Residual, ...] = (
    Residual("Gemm_v1", "out", "0", _LINEAR_WHY, condition="own_parameter"),
    Residual("Gemm_v2", "out", "0", _LINEAR_WHY, condition="own_parameter"),
    Residual("BiasAdd_v1", "out", "0", _LINEAR_WHY + " (the biased linear's epilogue is the returned tensor)", condition="own_parameter"),
    Residual("ScaledMmFp8Block_v1", "out", "0", _LINEAR_WHY + " (blockwise-FP8 linear)", condition="own_parameter"),
    Residual("AllReduce2_v1", "out", "0", _LINEAR_WHY + " (the row-parallel linear returns the all-reduced tensor as slot 0)", condition="own_parameter"),
    Residual("RoPE_v1", "query", "0", "the Build records RoPE's return slots by keyword (`query` / `key`); v1 spells them as the tuple index of "
             "`RotaryEmbedding.forward -> (q, k)`.  Same slot, two spellings"),
    Residual("RoPE_v1", "key", "1", "as `query` -> `0`"),
    Residual("*", "in", "in", "v1 encodes the token ids a module consumes as `<consuming module>/in` (family input_token_ids): the Value is the "
             "prompt input or the previous step's sampled id, spelt at its consumer", kind="encoding"),
    Residual("*", "sampled_token_ids", "sampled_token_ids", "v1 encodes the sampling event's output under the fixed path `runner.sampler` (the "
             "root body has no module path)", kind="encoding"),
    Residual("*", "weight", "weight", "v1 encodes a registered weight as `<field path>/weight` with an open range, once per rank", kind="encoding"),
    Residual("*", "hidden", "g<HB>x<M>x<NB>x<BN>x<D>", "the attention M1 stream is a kernel-side stream under `fa2.m1/L<k>` (k = the attention "
             "body's ordinal); its member is the launch geometry (v1 `_fa2_geometry` / `_fa3_geometry`)", kind="encoding"),
)


def _residual_for(family: str, structural: str, own_parameter: bool) -> Residual | None:
    for r in NAMED_RESIDUALS:
        if r.kind != "member" or r.family != family or r.structural != structural:
            continue
        if r.condition == "own_parameter" and not own_parameter:
            continue
        return r
    return None


def under(path: str, module: str) -> bool:
    """Is `path` registered under `module` in the module tree (`module` is a proper dotted prefix of `path`)?"""
    if not module:
        return False
    a, b = path.split("."), module.split(".")
    return len(a) > len(b) and a[: len(b)] == b


def member_name(call: CallView, port: Port, name: CallName, *, own_parameter: bool) -> tuple[str, str, Residual | None]:
    """(v1 member, structural member, residual applied or None) for the Value `port` of `call`."""
    structural = port.name if len(call.port_table) > 1 else (name.hint or "out")
    r = _residual_for(call.family, structural, own_parameter)
    return (r.v1_member if r else structural), structural, r


# ---------------------------------------------------------------------------------------------------------------------------------
# shared module instances
# ---------------------------------------------------------------------------------------------------------------------------------

def shared_instances(P: ProgramView, corr: Correspondence, record_fields: Mapping[ValueRef, str]) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """module -> (canonical module, invocation) for module bodies that are one shared instance, and the faults.

    Two bodies are one instance when they read the SAME non-empty set of registered parameters and have the same body signature (the
    multiset of (Definition, hint) of their Calls); the canonical body is the one a shared parameter is registered under (vLLM registers a
    shared module under the first path that holds it); invocation = the body's order among the group.  Fail-closed: a group with no or
    several canonical candidates, or unequal signatures, is not aliased and is reported."""
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
            groups[(frozenset(fs), tuple(sorted(sig[m].items())))].append(m)
    out: dict[str, tuple[str, int]] = {}
    faults: list[str] = []
    for (fs, _sig), mods in groups.items():
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


@dataclass
class BridgeResult:
    manifest: dict[str, Any]
    required: RequiredValues
    residual_counts: Counter = field(default_factory=Counter)


def _profile(res: Mapping[str, Any], profile: str | None) -> tuple[str, int | None, dict]:
    attn_obs = res.get("attention_impls_observed") or {}
    fa = None
    for v in attn_obs.values():
        fa = v.get("vllm_flash_attn_version")
        break
    if profile is None:
        cc = ((res.get("target") or {}).get("declared") or {}).get("compute_capability") or [8, 9]
        profile = "hopper" if (fa == 3 or (fa is None and int(cc[0]) >= 9)) else "ampere"
    return profile, fa, attn_obs


def query_header(profile: str, hidden_family: str, *, return_slots: bool) -> dict[str, Any]:
    text = Q_MODULE_BODY_TEXT + f"; policy: prescribed = record-typed inputs (weights); protocol = sampling-event outputs" + \
        (" + recorded return slots + returned tensors (every Value of an fx node's output at a step once any of them crosses the body boundary)"
         if return_slots else "") + f"; hidden = {hidden_family}; profile = {profile}"
    return {"query_id": Q_MODULE_BODY_ID, "query_id_digest": hashlib.sha256(text.encode()).hexdigest(), "query_text": text,
            "engine": "v2-query", "boundary": "module-body", "profile": profile, "attention_hidden_family": hidden_family,
            "required_families": ["input_token_ids", "instance_outputs", "sampled_token_ids", "weights", hidden_family],
            "policy": {"prescribed_inputs": "every field of the Program's record-typed parameters",
                       "protocol_required": ["sampling-event outputs"] +
                       (["recorded return slots (RETURN-SLOT)", "returned tensors: all rows / ports of an fx node's output once one crosses (TENSOR-RETURN)"]
                        if return_slots else [])},
            "named_residuals": [r.as_dict() for r in NAMED_RESIDUALS]}


def request_manifest(P: ProgramView, corr: Correspondence, res: Mapping[str, Any], *, request_id: str | int | None = None,
                     workload: Mapping[str, Any] | None = None, artifact: Mapping[str, Any] | None = None, program_dir: str | None = None,
                     rank: int = 0, ranks: int = 1, profile: str | None = None, return_slots: bool = True) -> BridgeResult:
    """The v1-shaped manifest of ONE request Program (B=1 component) under Q_module_body_v1."""
    pdig = (artifact or {}).get("program_digest") or res["program"]["digest"]
    profile, fa, attn_obs = _profile(res, profile)
    hidden_family = "fa3_hidden_m1_stream" if profile == "hopper" else "fa2_hidden_m1_stream"
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
    events = [c.id for c in P.calls if c.family in SAMPLING_EVENTS]
    if not events:
        raise ValueError("Program has no sampling event: no step boundary")
    step_of = lambda cid: bisect.bisect_left(events, cid)   # noqa: E731  -- the event itself belongs to the step it ends

    record_fields: dict[ValueRef, str] = {}
    record_params: list[InputView] = []
    for p in P.params:
        if len(p.port_table) > 1 or p.type_repr[:1] == "{":
            record_params.append(p)
            for port in p.port_table:
                record_fields[ValueRef(p.id, port.name)] = port.name
    prescribed = frozenset(record_fields)
    Q = Q_module_body_v1(P, corr)
    B = boundary(P, Q)
    protocol: set[ValueRef] = set()
    if return_slots:
        # TENSOR-RETURN: the acquisition binds the TENSORS a module returns, and a tensor is one fx node's output -- every row the
        # Program unrolled it into, every port of its tuple.  When any Value of a node's output leaves its body (a boundary Value), the
        # module returned that tensor, so every Value of the node's output at that step is required (the final norm's rows the logits
        # never read, the residual slot of a fused norm nobody consumes).
        returned_nodes: set[tuple[int, str, str]] = set()
        for v in B.values:
            if v.call >= 0:
                n = corr.name(v.call)
                returned_nodes.add((step_of(v.call), n.module, n.node))
    for c in P.calls:
        if c.family in SAMPLING_EVENTS:
            protocol.update(ValueRef(c.id, p.name) for p in c.port_table)
        elif return_slots:
            n = corr.name(c.id)
            if n.hint or (step_of(c.id), n.module, n.node) in returned_nodes:      # RETURN-SLOT (recorded) / TENSOR-RETURN (structural)
                protocol.update(ValueRef(c.id, p.name) for p in c.port_table)
    policy = Policy(prescribed_inputs=prescribed, protocol_required=frozenset(protocol))
    req = RequiredValues(B.values, policy.prescribed_inputs, policy.protocol_required)
    # (required_values() would recompute the boundary; the policy is validated the same way here)
    known_inputs = frozenset(P.inputs)
    for v in policy.prescribed_inputs:
        if v not in known_inputs:
            raise ValueError(f"policy prescribes {v}, which is not an input")

    aliases, alias_faults = shared_instances(P, corr, record_fields)
    impl = corr.implementation_path

    identities: list[dict[str, Any]] = []
    fam_counts: Counter = Counter()
    unmodelled: Counter = Counter()
    residual_counts: Counter = Counter()

    def add(family: str, step: int | str, invocation: int, op_path: str, member: str, lo: int, hi: int | None, **extra):
        row = {"family": family, "program_digest": pdig, "request_id": request_id, "step": step, "invocation": invocation,
               "op_path": op_path, "output_member": member, "element_range": [lo, hi], "rank": rank}
        if step != "*":
            row["engine_step"] = arrive + int(step)
        row.update(extra)
        identities.append(row)
        fam_counts[family] += 1

    # --- weights: every prescribed record field --------------------------------------------------------------------------------------
    for p in record_params:
        for port in sorted(p.port_table, key=lambda q: q.name):
            add("weights", "*", 0, impl(port.name), "weight", 0, None, dtype=_dtype(port.width, weight=True))

    # --- the Call-produced required Values, grouped per (step, module, member) -----------------------------------------------------
    own_param: dict[int, bool] = {}
    for c in P.calls:
        m = corr.module_of(c.id)
        own_param[c.id] = any(under(record_fields[v], m) for v in param_reads(c) if v in record_fields)

    groups: dict[tuple[int, str, str], dict[str, Any]] = {}
    for v in sorted(req.values):
        if v.call < 0:
            continue
        c = P.call(v.call)
        n = corr.name(c.id)
        step = step_of(c.id)
        if c.family in SAMPLING_EVENTS:
            add("sampled_token_ids", step, 0, "runner.sampler", "sampled_token_ids", 0, n_seq, dtype="i64", spec=c.family)
            residual_counts["encoding:sampled_token_ids"] += 1
            for r in c.reads:                                   # the runtime inputs the selector consumes (seed, splits): the sampler's
                if r.call < 0 and r not in record_fields:
                    pin = P.input(r.call)
                    add(f"sampler_{pin.name}", step, 0, "runner.sampler", pin.name, 0, n_seq, dtype=_dtype(pin.port_table[0].width), spec=c.family)
            continue
        port = c.port(v.port)
        member, structural, resid = member_name(c, port, n, own_parameter=own_param[c.id])
        if resid is not None:
            residual_counts[f"{resid.family}:{resid.structural}->{resid.v1_member}"] += 1
        g = groups.get((step, n.module, member))
        if g is None:
            g = groups[(step, n.module, member)] = {"rows": 0, "width": port.elems, "dtype": _dtype(port.width), "family": c.family, "spec": c.definition,
                                                    "nodes": set(), "hinted": bool(n.hint), "structural": structural,
                                                    "crossing": 0, "returned": 0}
        elif g["family"] != c.family or g["width"] != port.elems:
            unmodelled[f"{c.family} under {impl(n.module)} ({member}: two producers {g['family']} {g['width']} / {c.family} {port.elems})"] += 1
            continue
        g["rows"] += 1
        g["nodes"].add(n.node)
        if v in B.values:
            g["crossing"] += 1
        if v in policy.protocol_required:
            g["returned"] += 1

    for (step, mod, member), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if g["width"] <= 0:
            unmodelled[f"{g['family']} under {impl(mod)} ({member}: width 0)"] += 1
            continue
        extras = {"rows": g["rows"], "width": g["width"], "dtype": g["dtype"], "spec": g["family"], "fx_nodes": len(g["nodes"]),
                  "boundary": {"crossing_rows": g["crossing"], "returned_rows": g["returned"], "structural_member": g["structural"]}}
        if mod in aliases:
            canon, k = aliases[mod]
            add("instance_outputs", step, k, impl(canon), member, 0, g["rows"] * g["width"], program_op_path=impl(mod), **extras)
        else:
            add("instance_outputs", step, 0, impl(mod), member, 0, g["rows"] * g["width"], **extras)

    # --- the token ids a module body consumes: prompt input or a sampling event's output -------------------------------------------
    ids_rows: Counter = Counter()
    event_set = frozenset(events)
    for c in P.calls:
        m = corr.module_of(c.id)
        if not m:
            continue
        # a non-weight prescribed input (the prompt ids) or a sampling event's output among the reads
        if any(r not in record_fields for r in param_reads(c)) or reads_any_of(c, event_set):
            ids_rows[(step_of(c.id), m)] += 1
    for (step, mod), rows in sorted(ids_rows.items()):
        add("input_token_ids", step, 0, impl(mod), "in", 0, rows, rows=rows, width=1, dtype="i32")
        residual_counts["encoding:in"] += rows

    # --- the attention hidden stream: one launch per (attention body, step) ------------------------------------------------------
    attn_bodies: list[str] = []
    attn: dict[tuple[int, str], dict[str, Any]] = {}
    for c in P.calls:
        if c.family in FLASH_ATTENTION_LAUNCHES:
            m = corr.module_of(c.id)
            if m not in attn_bodies:
                attn_bodies.append(m)
            st = c.statics
            a = attn.setdefault((step_of(c.id), m), {"rows": 0, "T_max": 0, "NH": int(st.get("NH", 0)), "KVH": int(st.get("KVH", st.get("NH", 0))), "D": int(st.get("D", 0))})
            a["rows"] += 1
            a["T_max"] = max(a["T_max"], int(st.get("T", 0)))
    tap = "fa3.m1" if hidden_family[:3] == "fa3" else "fa2.m1"
    for (step, mod), a in sorted(attn.items(), key=lambda kv: (kv[0][0], attn_bodies.index(kv[0][1]))):
        layer = attn_bodies.index(mod)
        if hidden_family[:3] == "fa3":
            geo = fmt.fa3_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], n_seq)
        else:
            win = (attn_obs.get(impl(mod)) or attn_obs.get(mod) or {}).get("sliding_window")
            w0 = win[0] if isinstance(win, (list, tuple)) and win else (win if isinstance(win, int) else -1)
            geo = fmt.fa2_geometry(a["T_max"], a["rows"], a["NH"], a["KVH"], a["D"], n_seq, windowed=(w0 is not None and int(w0) >= 0))
        HB, M, NB, BN, gname = geo
        add(hidden_family, step, 0, f"{tap}/L{layer}", gname, 0, fmt.mat_total_words(HB, M, NB, BN, a["D"]), dtype="u32",
            geometry={"HB": HB, "M": M, "NB": NB, "BN": BN, "D": a["D"], "seqlen_k": a["T_max"], "rows": a["rows"]}, attention_module=impl(mod))
        residual_counts["encoding:hidden"] += 1

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

    for f in alias_faults:
        unmodelled[f] += 1
    identities.sort(key=fmt.row_key)
    q = query_header(profile, hidden_family, return_slots=return_slots)
    if moe_layers:
        q["required_families"].append("moe_block_stream")
        q["moe_layers"] = sorted(moe_layers)
    if any(f[:8] == "sampler_" for f in fam_counts):
        q["required_families"] += sorted(f for f in fam_counts if f[:8] == "sampler_")
    q["residuals_applied"] = dict(sorted(residual_counts.items()))
    alias_out = {impl(m): [impl(canon), k] for m, (canon, k) in aliases.items() if m != canon}
    man = {"schema": fmt.SCHEMA, "identity_key": list(fmt.IDENTITY_KEY), "program_digest": pdig,
           "program": {"dir": program_dir, "function": res["program"].get("function"), "wrapper": res.get("wrapper"), "LP": res.get("LP"), "T": res.get("T"),
                       "model_pin": res.get("model_pin"), "artifact_identity": (artifact or {}).get("identity"), "instances": len(P.calls),
                       "instances_digest": P.digest, "source": P.source},
           "correspondence": corr.provenance(),                 # which reader gave the owning modules: the runtime-correspondence/v1 record or the v1 annotations
           "workload": {"row_id": sweep.get("row_id"), "batch": sweep.get("batch"), "request_id": request_id, "workload_request_id": workload_request_id,
                        "arrive_step": arrive} if workload else None,
           "query": q, "wrapper_prefix": corr.prefix, "op_path_aliases": alias_out, "target": ((res.get("target") or {}).get("declared") or {}),
           "flash_attn_version": fa, "ranks": ranks, "rank": rank,
           "steps": list(range(len(events))), "engine_steps": [arrive + s for s in range(len(events))], "requests": [request_id],
           "populations": {"identities": len(identities), "by_family": dict(fam_counts), "steps": len(events),
                           "attention_layers": len(attn_bodies), "units": len(B.units), "boundary_values": len(B.values),
                           "required_values": len(req.values)},
           "complete": not unmodelled, "unmodelled": dict(unmodelled),
           "manifest_digest": fmt.manifest_digest(identities), "identities": identities}
    return BridgeResult(man, req, residual_counts)


# ---------------------------------------------------------------------------------------------------------------------------------
# the global (B >= 2) composition -- v1's reading (TABLE-RULES global), over v2 components
# ---------------------------------------------------------------------------------------------------------------------------------

def compose_global(fixture: Mapping[str, Any], parts: Mapping[str, dict[str, Any]], reqs: list[dict[str, Any]], *, workload_digest: str | None,
                   rank: int = 0, ranks: int = 1) -> dict[str, Any]:
    """`parts` = {request label: component manifest}; `reqs` = the declared requests in (arrive_step, index) order with `request_id` = label."""
    component_digests = sorted({p["program_digest"] for p in parts.values()})
    pdig = str(workload_digest) if workload_digest else fmt.components_fold_digest(component_digests)
    pdig_kind = "workload" if workload_digest else "components-fold"
    by_engine: dict[int, list[tuple[str, int]]] = {}
    for r in reqs:
        for s in range(int(r["max_tokens"])):
            by_engine.setdefault(int(r.get("arrive_step", 0)) + s, []).append((r["request_id"], s))
    hidden_family = parts[reqs[0]["request_id"]]["query"]["attention_hidden_family"]
    ids: list[dict[str, Any]] = []
    seen_w: set = set()
    for rid, p in parts.items():
        for row in p["identities"]:
            if row["family"] == "weights":
                if (row["rank"], row["op_path"]) in seen_w:
                    continue
                seen_w.add((row["rank"], row["op_path"]))
                ids.append(dict(row, program_digest=pdig, component_digest=row["program_digest"]))
    engine_steps_out: list[dict[str, Any]] = []
    for e in sorted(by_engine):
        members = by_engine[e]
        rows_of: dict[str, int] = {}
        for rid, s in members:
            emb = next((x for x in parts[rid]["identities"] if x["family"] == "input_token_ids" and x["step"] == s), None)
            rows_of[rid] = int(emb["rows"]) if emb else 0
        attn_union: dict[str, dict[str, Any]] = {}
        for rid, s in members:
            for row in parts[rid]["identities"]:
                if row["step"] != s or row["family"] == "weights":
                    continue
                g = dict(row, program_digest=pdig, component_digest=row["program_digest"], engine_step=e)
                if row["family"] == hidden_family:
                    a = attn_union.setdefault(row["op_path"], {"requests": [], "n_seq": 0, "max_rows": 0, "T_max": 0, "module": row.get("attention_module")})
                    a["requests"].append(rid); a["n_seq"] += 1
                    a["max_rows"] = max(a["max_rows"], int(row["geometry"]["rows"])); a["T_max"] = max(a["T_max"], int(row["geometry"]["seqlen_k"]))
                else:
                    ids.append(g)
        for op, a in sorted(attn_union.items()):
            comp = parts[a["requests"][0]]
            pre = next(x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 0)
            dec = next((x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 1), None)
            D = int(pre["geometry"]["D"])
            NH = int(pre["geometry"]["HB"])
            KVH = int(dec["geometry"]["HB"]) if dec is not None and int(dec["geometry"]["M"]) > 1 else NH
            fn = fmt.fa3_geometry if hidden_family[:3] == "fa3" else fmt.fa2_geometry
            HB, M, NB, BN, gname = fn(a["T_max"], a["max_rows"] * a["n_seq"], NH, KVH, D, a["n_seq"])
            ids.append({"family": hidden_family, "program_digest": pdig, "request_id": "*", "requests": a["requests"], "step": "*", "engine_step": e, "invocation": 0,
                        "op_path": op, "output_member": gname, "element_range": [0, fmt.mat_total_words(HB, M, NB, BN, D)], "rank": rank, "dtype": "u32",
                        "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, "seqlen_k": a["T_max"], "rows": a["max_rows"] * a["n_seq"], "n_seq": a["n_seq"]},
                        "attention_module": a["module"]})
        engine_steps_out.append({"engine_step": e, "members": [{"request_id": rid, "step": s, "rows": rows_of[rid]} for rid, s in members],
                                 "rows": sum(rows_of.values()), "n_seq": len(members)})
    ids.sort(key=fmt.row_key)
    fam: Counter = Counter(r["family"] for r in ids)
    ref: dict[str, Any] = {}
    for r in ids:
        if r["family"] in ("input_token_ids", "sampled_token_ids") and r["family"] not in ref:
            ref[r["family"]] = {"op_path": r["op_path"], "output_member": r["output_member"], "per_row": 1}
    q0 = parts[reqs[0]["request_id"]]["query"]
    applied: Counter = Counter()
    for p in parts.values():
        applied.update(p["query"].get("residuals_applied") or {})
    return {"schema": fmt.SCHEMA, "identity_key": list(fmt.IDENTITY_KEY), "program_digest": pdig, "program_digest_kind": pdig_kind, "component_digests": component_digests,
            "global": True, "addressing": "request-local", "composition_reference": ref,
            "fixture": {k: v for k, v in fixture.items() if k in ("schema", "fixture", "model", "revision", "sampling", "expected_engine_steps", "case")}
                       | ({"row_id": fixture["sweep"].get("row_id"), "seed": fixture["sweep"].get("seed")} if isinstance(fixture.get("sweep"), dict) else {}),
            "requests_declared": [{k: r.get(k) for k in ("request_id", "workload_request_id", "index", "prompt_len", "max_tokens", "arrive_step")} for r in reqs],
            "components": {rid: {"program_digest": p["program_digest"], "steps": len(p["steps"]), "LP": p["program"]["LP"], "T": p["program"]["T"],
                                 "workload_request_id": next(r.get("workload_request_id") for r in reqs if r["request_id"] == rid),
                                 "correspondence": p.get("correspondence")} for rid, p in parts.items()},
            # the source of the owning modules across the components: one source (the record on every component, or the fallback on every one)
            "correspondence": {"source": sorted({(p.get("correspondence") or {}).get("source", "v1_annotations") for p in parts.values()}),
                               "digests": sorted({d for p in parts.values() if (d := (p.get("correspondence") or {}).get("digest"))})},
            "complete": all(p.get("complete", True) for p in parts.values()),
            "unmodelled": {k: v for p in parts.values() for k, v in (p.get("unmodelled") or {}).items()},
            "query": dict(q0, residuals_applied=dict(sorted(applied.items()))), "op_path_aliases": parts[reqs[0]["request_id"]]["op_path_aliases"],
            "ranks": ranks, "rank": rank,
            "engine_steps": engine_steps_out, "steps": sorted(by_engine), "requests": [r["request_id"] for r in reqs],
            "populations": {"identities": len(ids), "by_family": dict(fam), "engine_steps": len(by_engine), "requests": len(reqs)},
            "manifest_digest": fmt.manifest_digest(ids), "identities": ids}
