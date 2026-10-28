"""Coverage of record: the replay's reach reconciled against the query population, identity by identity
(`query_population`), the query families no row evaluates with the mechanism that checks them, and the families the
Commit value-checks by form (B) only (`form_b_families`)."""
from __future__ import annotations

from collections import Counter
from typing import Any

from verity_vllm.query import norm_scales as NS
from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay.replay_codes import NOT_YET
from verity.ml.definitions.numerics.rows import EVALUATORS
from verity_vllm.check.replay.index import _MOE_LAYER_RE, SAMPLER_OP, ProgramIndex
from verity_vllm.check.replay.opening import alias_table, resolve_op_path
from verity_vllm.check.replay.population import ATTENTION_ROW_FAMILIES, EXECUTED_PREFIX_RULE, NOT_EXECUTED_SCOPE
from verity_vllm.records.fields import RecordError
from verity_vllm.records.required_manifest import RequiredManifest

#: what `query_population` names a manifest it was handed (the replay is handed the parsed JSON, not its path)
HANDED = "the required-value manifest the replay was handed"


#: named in population() whenever a Program carries attention rows: what the row relation does not read (a Q_fine_v1 research item, not a
#: gap under the query of record)
ATTENTION_INTERIOR_MECHANISM = ("AttnBlock interiors (Q_fine_v1 units: s[c] dots, PV k-step chains, max chain, lane sums, epilogue) live in the "
                                "fa2/fa3 M1 export stream; the sampled replay evaluates the attention ROW (the Q_module_v1 unit) from its "
                                "committed q + K/V history to its committed output and does not read those interior words. "
                                "The row evaluator is driven by the row's DECLARED signature, not by an attention-specific hand list: "
                                "its inputs are exactly the args the Program row wires (the Definition's arity: q, K history, V history -- each "
                                "ref resolved to committed words at its producer's step; a row wiring another arity is unresolved by name), its "
                                "statics come from the spec text (T, NH, KVH, D, BN, and for Attention_v2 DOT / INV by id), and every member "
                                "the manifest requires of the row is compared (a member the evaluator does not produce = not evaluated, by name).")

#: the controller / guard families of a request Program's root body (the manifest header's `guard_rows` counts them)
GUARD_PREFIXES = ("LStepLive", "LSelect", "LBot", "LIsEOS", "LEmitNext")

#: the query families the exact replay has NO VU row for, with the mechanism of record that checks their values.  The
#: population itself is the manifest's (Q(P)); this names, per family, what covers it when no evaluator can -- a required family with no
#: entry here is reported `class: unstated` (fail closed on the record; never silently "covered").
QUERY_FAMILY_MECHANISM: dict[str, str] = {
    "weights": "registered inputs: the weights pin (served parameters == the checkpoint of record, by digest) + C1 coverage / C3 openings",
    "input_token_ids": "prescribed input: boundary linkage (the committed ids at the arrival step == the declared prompt; sampled id[s-1] -> id[s]) + C1 / C3",
    # the sampler families name what CONSUMES them, as input_token_ids' entry does: the committed seed / splits are the
    # inputs of the sampler VU (GumbelTopPTokenSelect_v1) whose exact replay in form (A) reads them by binding, and of stoch_recompute's
    # attribution (the served token re-derived from the committed logits + seed + splits) -- not only a linkage check on their values
    "sampler_seed": ("prescribed input: prescribed_input_linkage (committed seed == the declared SamplingParams.seed) + C1 / C3; "
                     "CONSUMED BY the sampler VU GumbelTopPTokenSelect_v1's exact replay, which reads the committed seed as its Philox key "
                     "(seed, pos) -- the committed sampled id is reproducible from the committed logits only with it (form (A) per pick; the "
                     "seed is a bound input of every sampled stratum) -- and by stoch_recompute's attribution (served token re-derived "
                     "from committed logits + seed + splits, every row)"),
    "sampler_splits": ("prescribed input: prescribed_input_linkage (committed S == SplitsFor_v1(live geometry), REQ-03) + C1 / C3; "
                       "CONSUMED BY the same relation (the sampler VU GumbelTopPTokenSelect_v1's exact replay) as its 6th operand: the top-p mask "
                       "kernel's split count S = SplitsFor_v1(|live(s)|, num_SMs), "
                       "derived, never a static (a wrong S fails form (A): S is a bound input of every sampled stratum) and by "
                       "stoch_recompute's attribution (served token re-derived from committed logits + seed + splits)"),
    "fa2_hidden_m1_stream": ("attention interior (the launch's M1 export stream): no VU row of its own; the attention VU it belongs to is state-linked "
                             "(NOT_YET) -- C1 / C3 + form (B) where present; form (A) over it is an engineering gap, not a domain"),
    "fa3_hidden_m1_stream": ("attention interior (the Hopper launch's M1 export stream): no VU row of its own; the attention VU it belongs to is "
                             "state-linked (NOT_YET) -- C1 / C3 + form (B) where present; form (A) over it is an engineering gap, not a domain"),
    "norm_scales": ("opt-in, never the record (query.norm_scales): a norm's scale, one f32 per row -- interior to the kernel norms' Calls (the "
                    "norm-scale tap) or an ATen-chain norm's rsqrt output: no VU row of its own under Q_module_body_v1; C1 / C3 here, and the "
                    "tapped kernels' scale == the IR model's scale bit for bit is the norm-tap exactness property (properties.norm_tap_exactness)"),
    "router_softmax": ("opt-in, never the record (query.router_softmax): the selection-order router's softmax per token row -- interior to the "
                       "router Call (the router-softmax tap): no VU row of its own under Q_module_body_v1; C1 / C3 here, and every tapped word == the "
                       "ordered router's committed value in the IR bit for bit is the router-tap exactness property (properties.router_tap_exactness)"),
    "vocab_range": ("opt-in, never the record (query.vocab_range): a TP rank's range values per token row -- interior to EmbeddingShard_v1 "
                    "(the vocabulary-range source): no VU row of its own under Q_module_body_v1; C1 / C3 here, and every value == the IR's is the "
                    "vocabulary-range exactness property (properties.vocab_range_exactness)"),
    "scale_products": ("additive (query.scale_products): a shared-scale FP8 GEMM's scale products per token row -- interior to "
                       "ScaledMmFp8BlockSharedScale_v1 (the scale-products source, F32Mul_v1 of the Call's own scale operands through the IR): "
                       "no VU row of its own under Q_module_body_v1; C1 / C3 here"),
}
#: the query families whose identities ARE produced by Program rows (a VU or a named non-evaluable row must address each of them)
QUERY_FAMILIES_WITH_ROWS = ("instance_outputs", "sampled_token_ids", "moe_block_stream", "tp_rank_partials", "call_boundaries")


def _row_produced(ident: dict[str, Any]) -> bool:
    """Whether a Program row must address the identity: the row families, and an ATen-chain norm's scale (its `RsqrtF32_v1` Call's own
    output, `NS.CALL_SCALES`) -- a kernel norm's scale is interior to its Call and stays a no-row family."""
    fam = str(ident.get("family"))
    if fam == NS.FAMILY:
        return NS.site_of(RM.spec_family(str(ident.get("spec") or ""))) == "aten"
    return fam in QUERY_FAMILIES_WITH_ROWS


def _address_family(op: str, member: str, a: dict[str, Any]) -> str:
    """The manifest family of a `population(..., addresses=)` record: recorded by the address rule that produced it (`manifest_family`,
    the same labels build_manifest assigns).  A record built without the label (a caller's own accumulator, e.g. the tests') is named by
    the address SCHEMA it was written in: the sampler op, a `moe/L<k>` plane address (`_MOE_LAYER_RE`, the moe_block_stream op_path
    vocabulary), else the member's own family (`manifest_format.alias_family`)."""
    fam = a.get("manifest_family")
    if fam:
        return str(fam)
    if op == SAMPLER_OP:
        return "sampled_token_ids"
    return "moe_block_stream" if _MOE_LAYER_RE.match(op) else RM.alias_family(member)


def _manifest_engine_name(manifest: RequiredManifest) -> str:
    """Which engine built a manifest: the traversal's header carries `engine` (schema + `traversal` = the class path), the
    inventory's carries none (byte-for-byte the older header) -- the guard header names it too where present."""
    eng = manifest.engine
    if isinstance(eng, str):
        return eng
    if eng is not None:
        return "traversal" if "traversal" in eng else str(eng.get("name") or "traversal")
    g = manifest.guard
    return str(g.get("engine")) if g is not None and g.get("engine") else "inventory"


def _header(man: RequiredManifest | None, key: str) -> Any:
    """A field of the manifest's query header; None without a manifest, a header (`acquire.served` writes none) or the field (the
    compiled manifest's header names its profile only)."""
    return man.query.get(key) if man is not None and man.query is not None else None


def controller_rows_of(programs: dict[str, Any], indices: dict[int, "ProgramIndex"] | None = None) -> dict[str, int]:
    """The replay's OWN count of guard / controller rows per request Program (spec family under GUARD_PREFIXES) -- the
    cross-check of the manifest header's `guard_rows`, so a stale or foreign header cannot drop or shrink the controller statement ('the
    collector cannot define its own population', applied to the controller)."""
    out: dict[str, int] = {}
    for rid, inst in programs.items():
        rows = inst.get("rows") if isinstance(inst, dict) else None
        if rows is None and indices is not None and id(inst) in indices:
            rows = indices[id(inst)].rows
        out[str(rid)] = sum(1 for r in (rows or []) if RM.spec_family(str(r.get("spec"))).startswith(GUARD_PREFIXES))
    return out


def query_population(manifest: RequiredManifest | dict[str, Any] | None, addresses: dict[tuple, dict], evaluated: set[tuple], *, ranks: list[int],
                     strata: dict[str, dict] | None = None, first_n: int = 6, executed_prefix: dict[str, dict] | None = None, controller_rows: dict[str, int] | None = None) -> dict[str, Any]:
    """Reconcile the replay's reach against the population of record, Q(P), IDENTITY BY IDENTITY.
    `manifest`: the required-value manifest of the row (Q_module(P): the module-boundary population over the Program, families selected by
    the canonical VU query; the traversal adapter's or the inventory's -- the same identities either way); None = no manifest supplied,
    the reconciliation is over the Program rows' own addresses and says so.
    `addresses`: `population(..., addresses=)` accumulator (manifest address -> class of the rows producing it); `evaluated`: the
    (request, step, op_path) triples with an evaluated pick.
    Returns the three record fields the row must carry (`query_total`, `sampled_strata`, `uncovered_by_name`) + `query_checks`:
      * query_total     -- |Q(P)| and its families (the authority: never the collector's, never the replay's own enumeration)
      * sampled_strata  -- what the exact replay REACHES (identities with an evaluable VU) and what it EVALUATED, as a share of Q(P)
      * uncovered_by_name -- every required identity the exact replay has no evaluable VU for, grouped by (family, spec, class) with the
                             mechanism of record that covers it instead; plus the query families with no row (weights, ids, streams)
      * query_checks    -- the fail-closed reconciliation: a VU whose address is NOT a required identity (the replay enumerating
                           something Q(P) does not require), a required identity NO row produces, a row count that disagrees with the
                           manifest's `rows`, a rank set that disagrees.  Any non-zero count is a finding on the record (`ok` False).
      * not_executed_by_rule -- `executed_prefix` (request -> `executed_prefix_of_record`): the identities of Q(P) at
                           request-relative steps >= n_exec are the third bucket beside reached / uncovered (query_total unchanged;
                           `accounted` = reached + uncovered + not-executed == |Q(P)|); a request whose prefix carries faults is NOT
                           classified and fails the check by name."""
    # a manifest missing a field its writers always write is refused by name: the reach is then reconciled against nothing, and
    # `query_checks` fails (`manifest_refused`), never a crash and never the unreconciled None of a replay handed no manifest
    refused: RecordError | None = None
    try:
        man = RequiredManifest.of(manifest, HANDED) if manifest is not None else None
    except RecordError as e:
        man, refused = None, e
    by_key: dict[tuple, dict] = {}
    fam_total: Counter = Counter()
    ranks_seen: set[int] = set()
    # shared modules (vLLM's one RotaryEmbedding instance called by every layer): the manifest addresses them at the FIRST layer's path
    # with invocation = layer order and records the per-layer paths in `op_path_aliases` (alias -> [canonical, invocation]) -- resolved
    # through the ONE table `CommittedStore.words` reads through as well
    aliases = alias_table(man)

    def _key(rid: str, st: int, op: str, mem: str) -> tuple:
        canon, inv = resolve_op_path(aliases, op)
        return (rid, st, canon, mem, inv)

    non_row_idents: list[dict] = []                                                # the no-row families, classified below
    keyed, keyed_not_exec = Counter(), Counter()                                   # row-produced identities of a no-row family (ATen norm scales)
    if man is not None:
        for ident in man.identities:
            fam = str(ident.get("family"))
            fam_total[fam] += 1
            ranks_seen.add(int(ident.get("rank", 0) or 0))
            if _row_produced(ident) and ident.get("step") != "*" and ident.get("request_id") != "*":
                keyed[fam] += fam not in QUERY_FAMILIES_WITH_ROWS
                by_key[(str(ident.get("request_id")), int(ident["step"]), str(ident.get("op_path")), str(ident.get("output_member")), int(ident.get("invocation", 0) or 0))] = ident
            elif executed_prefix:
                non_row_idents.append(ident)
    else:
        for (rid, st, op, mem), a in addresses.items():
            fam_total[_address_family(op, mem, a)] += 1
    total = sum(fam_total.values())
    # --- reach: identities with >= 1 evaluable VU; evaluated: identities under an evaluated (request, step, module) -------------------
    reached: Counter = Counter()
    reached_rows: Counter = Counter()                                              # Program-row addresses (a 2-member row addresses 2 identities)
    evald: Counter = Counter()
    not_exec: Counter = Counter()                                                  # identities at steps >= n_exec, by family
    not_exec_req: Counter = Counter()                                              # per request: every classified identity (rows + named inputs)
    not_exec_req_rows: Counter = Counter()                                         # per request: the classified ROW identities only
    req_max_step: dict[str, int] = {}                                              # per request: the last engine step Q(P) carries an identity at
    req_ids: Counter = Counter()                                                   # required row identities per request (manifest rows families)
    past_stop: list[dict] = []                                                     # an EVALUATED VU past a request's executed prefix: FAIL by name
    outside: list[dict] = []
    rows_disagree: list[dict] = []
    rows_annot: list[dict] = []
    n_outside = n_rows_disagree = n_rows_annot = n_missing = 0
    groups: dict[tuple, dict] = {}
    seen_keys: set[tuple] = set()
    for (rid, st, op, mem), a in sorted(addresses.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        k5 = _key(rid, st, op, mem)
        seen_keys.add(k5)
        ident = by_key.get(k5) if man is not None else None
        fam = str(ident["family"]) if ident is not None else _address_family(op, mem, a)
        if man is not None and ident is None:
            n_outside += 1
            if len(outside) < first_n:
                outside.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "spec": a["spec"], "class": a["class"]})
            # not a required identity: it is NOT counted toward the reach of Q(P) (the replay does not enlarge the population)
        elif ident is not None:
            # the identity's block height = element_range / width (digest-bearing) must be the number of Program rows producing it; the
            # `rows` annotation is compared beside it (a global manifest annotates every identity of the engine step with the request's
            # token count -- false on per-sequence members such as logits_processor/out: reported as an annotation finding, not a reach gap)
            lo, hi = (ident.get("element_range") or [0, None])[:2]
            width = ident.get("width")
            height = ((int(hi) - int(lo)) // int(width)) if (hi is not None and isinstance(width, int) and width > 0) else None
            if height is not None and height != int(a["rows"]):
                n_rows_disagree += 1
                if len(rows_disagree) < first_n:
                    rows_disagree.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "manifest_height": height, "program_rows": a["rows"]})
            if isinstance(ident.get("rows"), int) and height is not None and int(ident["rows"]) != height:
                n_rows_annot += 1
                if len(rows_annot) < first_n:
                    rows_annot.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "rows_annotation": ident["rows"], "height": height})
        if ident is not None:
            req_ids[str(rid)] += 1
            if int(st) > req_max_step.get(str(rid), -1):
                req_max_step[str(rid)] = int(st)
        ep = (executed_prefix or {}).get(str(rid))
        # `st` is the identity's / Program row's OWN step = request-relative already (the traversal annotates `engine_step` = arrive + step
        # beside it); the executed prefix's n_exec counts the request's own steps, so the comparison is direct -- the arrival enters only for the
        # batch-level planes below, whose one step is the ENGINE's
        if ep is not None and not ep.get("faults") and int(st) >= int(ep["n_exec"]) and (man is None or ident is not None):
            not_exec[fam] += 1                                                    # classified, not reached, not uncovered
            not_exec_req[str(rid)] += 1
            not_exec_req_rows[str(rid)] += 1
            keyed_not_exec[fam] += ident is not None and fam not in QUERY_FAMILIES_WITH_ROWS
            if (rid, st, op) in evaluated:
                past_stop.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "n_exec": ep["n_exec"]})
            continue
        if a["class"] == "vu":
            if man is None or ident is not None:
                reached[fam] += 1
                reached_rows[fam] += int(a["rows"])
                if (rid, st, op) in evaluated:
                    evald[fam] += 1
            continue
        sf = RM.spec_family(a["spec"])                                           # grouped by the spec FAMILY (Attention_v3), the shapes counted
        g = groups.setdefault((fam, sf, a["class"]), {"family": fam, "spec": sf, "class": a["class"], "why": a["why"], "identities": 0, "row_addresses": 0,
                                                       "shapes": set(), "first": []})
        g["identities"] += 1
        g["row_addresses"] += int(a["rows"])
        g["shapes"].add(a["spec"])
        if len(g["first"]) < 3:
            g["first"].append([rid, st, op, mem])
    # --- required identities no row produces (manifest / replay address disagreement) ----------------------------------------------------
    missing: list[dict] = []
    if man is not None:
        for k, ident in by_key.items():
            if k not in seen_keys:
                n_missing += 1
                if len(missing) < first_n:
                    missing.append({"request_id": k[0], "step": k[1], "op_path": k[2], "member": k[3], "invocation": k[4], "family": ident.get("family"),
                                    "spec": ident.get("spec")})
    # --- the class over the NO-ROW families: the rule names identities of a request at request-relative steps
    # >= n_exec, whatever their family -- a named input (`input_token_ids` at r0's steps past its stop) or any other per-request identity is
    # not executed by the same rule as the rows; a BATCH-LEVEL plane (request_id '*', `requests` list, ONE engine step) is not executed
    # only when EVERY listed request is past its own stop at that engine step (then nothing was committed there); otherwise the identity
    # exists with a smaller composition and is REQUIRED -- the per-request slots past the stop are counted beside it (the Commit's
    # binding coverage expands batch-level rows per request: its count reconciles against `request_slots_not_executed`, never N_excluded).
    kind_by_family: dict[str, str] = {}
    batch_level: dict[str, dict] = {}
    batch_slots_req: Counter = Counter()
    for ident in non_row_idents:
        fam = str(ident.get("family"))
        rid, st = ident.get("request_id"), ident.get("step")
        if rid != "*" and st != "*":
            kind_by_family.setdefault(fam, "named-input" if fam == "input_token_ids" else "per-request")
            ep = executed_prefix.get(str(rid))
            if ep is not None and not ep.get("faults") and int(st) >= int(ep["n_exec"]):
                not_exec[fam] += 1
                not_exec_req[str(rid)] += 1
            elif int(st) > req_max_step.get(str(rid), -1):
                req_max_step[str(rid)] = int(st)
        elif rid == "*" and isinstance(ident.get("requests"), list) and ident.get("engine_step") is not None:
            kind_by_family.setdefault(fam, "batch-level")
            e, listed = int(ident["engine_step"]), [str(r) for r in ident["requests"]]
            past = [r for r in listed if (ep := executed_prefix.get(r)) is not None and not ep.get("faults")
                    and e - int(ep.get("arrive", 0) or 0) >= int(ep["n_exec"])]
            # a (plane, listed request) pair is one CELL (the request's slices of the plane); the Commit's
            # per-request expansion counts cells -- `cells_not_executed` is the number it reconciles against
            b = batch_level.setdefault(fam, {"identities": 0, "identities_not_executed": 0, "identities_partially_executed": 0,
                                             "request_slots": 0, "request_slots_not_executed": 0, "cells_not_executed": 0})
            b["identities"] += 1
            b["request_slots"] += len(listed)
            b["request_slots_not_executed"] += len(past)
            b["cells_not_executed"] += len(past)
            for r in past:
                batch_slots_req[r] += 1
            if listed and len(past) == len(listed):
                b["identities_not_executed"] += 1
                not_exec[fam] += 1
            elif past:
                b["identities_partially_executed"] += 1
        else:
            kind_by_family.setdefault(fam, "static" if st == "*" else "unclassified")
    # --- the uncovered list: row groups by class, then the families with no rows at all -------------------------------------------------
    mech_of_class = {
        "state-linked": "C1 coverage + C3 openings + form (B) where present; the exact VU (K/V history from committed qkv outputs) is an engineering gap, not a domain",
        "no-evaluator": "C1 coverage + C3 openings + form (B) where present; a registered evaluator is the engineering gap (form (A) required)",
        "plane-fed": "the plane is read as its consumers' committed input; the producer's own relation has no evaluator yet (form (B) required meanwhile)",
        "interior-composed-v2": "C1 / C3 + form (B) required (no composed evaluator yet)",
        "mixed": "rows of two classes under one identity: named, resolve before any claim",
    }
    uncovered = []
    for g in sorted(groups.values(), key=lambda g: (-g["identities"], g["family"], g["spec"])):
        shapes = sorted(g.pop("shapes"))
        uncovered.append(dict(g, shapes=len(shapes), shape_example=shapes[0], mechanism=mech_of_class.get(g["class"], "unstated (fail closed)")))
    for fam, n in sorted(fam_total.items()):
        n_rowless = int(n) - keyed[fam] - (not_exec[fam] - keyed_not_exec[fam])
        if fam in QUERY_FAMILIES_WITH_ROWS or (keyed[fam] and not n_rowless):
            continue
        cls = "mechanism" if fam in QUERY_FAMILY_MECHANISM else "unstated"
        # the family's not-executed identities left the uncovered count for the class (reached + uncovered + not-executed == |Q(P)|)
        uncovered.append({"family": fam, "spec": None, "class": cls, "identities": n_rowless, "row_addresses": 0,
                          "why": "no VU row of its own (the query requires it; the exact replay has no relation to evaluate for it)",
                          "mechanism": QUERY_FAMILY_MECHANISM.get(fam, "UNSTATED: a required family with no named mechanism -- fail closed on the record")})
    # a LIFTED Program's controller rows (LStepLive / LSelect / LBot; Continuation_v1's LIsEOS<E> `is_eos`,
    # LEmitNext[H] `emit`, LSelect<w> `out` = the bits b_t / e_t / r_t / o_t / u_t) carry NO identity under Q_module_v1's rule (so
    # RequiredValues(Lift(P), Q) == RequiredValues(P, Q) row for row) -- the population statement NAMES them as uncovered with their
    # mechanism of record (the lifted record's R8 controller replay through the registered evaluators; the executed prefix from r_t, R3)
    # instead of letting them be silently absent.  identities 0 by rule, rows counted; a decision to give the bits identities is a Q change
    # (Q_module_v2: digest moves), recorded in Q-EXPANSION -- not made here.
    mq = man.query if man is not None else None
    controller = None
    hdr_rows = int(mq.get("guard_rows") or 0) if isinstance(mq, dict) else 0
    own_rows = {str(k): int(v) for k, v in (controller_rows or {}).items()}          # the replay's own count per request Program (c2)
    # the header vs the replay's own count of guard-prefixed rows: the required manifest of record is the traversal of
    # the WORKLOAD Program (manifest_verify.used_program_digest == the workload digest; one manifest, query_total.requests = all of them), so
    # its `guard_rows` is the SUM over the request Programs -- the compare is made AT THE HEADER'S SCOPE (sum(own) vs header), never per
    # request (per-request equality is a semantics the header does not have: B2 {r0: 7, r1: 7} under 14 AGREES; {10, 4} under 14 agrees;
    # under 7 both disagree).  A disagreement is a fault by name (a stale / foreign header cannot drop or shrink the statement); the
    # statement is emitted when EITHER side names controller rows, and it says which scope the compare used.
    own_sum = sum(own_rows.values())
    ctrl_scope = "header = the workload Program's traversal: guard_rows is the SUM over the request Programs (compared as sum(program_rows_by_request) vs header_rows)"
    ctrl_disagree = ([{"program_rows_sum": own_sum, "header_rows": hdr_rows, "program_rows_by_request": own_rows, "scope": ctrl_scope}]
                     if (mq is not None and own_rows and own_sum != hdr_rows) else [])
    if (isinstance(mq, dict) and (mq.get("lifted") or hdr_rows > 0)) or own_sum > 0:
        gf = mq.get("guard_families") if isinstance(mq, dict) and isinstance(mq.get("guard_families"), dict) else None
        controller = {"family": "controller", "spec": None, "class": "controller", "identities": 0, "row_addresses": 0,
                      "program_rows": (own_sum if own_rows else hdr_rows), "header_rows": hdr_rows, "program_rows_by_request": own_rows,
                      "compare_scope": ctrl_scope,
                      "rows_agree": ((not ctrl_disagree) if (own_rows and mq is not None) else None),   # None: no header to agree with
                      "families": (gf if gf is not None else "unstated (older manifest header: guard_rows only)"),
                      "why": ("the lifted Program's guard / controller rows carry no identity under Q_module_v1's rule: RequiredValues(Lift(P), Q) == "
                              "RequiredValues(P, Q) row for row; the bits are not sampled by C2 -- said here BY NAME"),
                      "mechanism": ("the lifted record's R8 controller replay through the registered evaluators (LIsEOS / LEmitNext / LSelect; pad "
                                    "Continuation_v1) + the executed prefix derived from r_t (R3, executed_prefix_of_record served=)")}
        uncovered.append(controller)
    n_reached, n_evald = sum(reached.values()), sum(evald.values())
    n_uncovered = sum(int(u["identities"]) for u in uncovered)
    n_not_exec = sum(not_exec.values())
    ep_faults = [{"request_id": r, "faults": e.get("faults")} for r, e in sorted((executed_prefix or {}).items()) if e.get("faults")]
    # the class is an EXCLUSION over the cap-length manifest of record: a classified request whose Q(P) carries NO
    # identity at its not-executed steps was CUT before it reached the population (a restricted manifest handed in as Q(P)) -- N_manifest
    # would shrink silently and the class count 0.  Named here, fails the reconciliation: the manifest of record is never restricted.
    manifest_cut: list[dict] = []
    if man is not None:
        for r, e in sorted((executed_prefix or {}).items()):
            if e.get("faults") or int(e["n_exec"]) >= int(e["cap"]):
                continue
            last = int(e["cap"]) - 1                                              # request-relative, like the identities' own `step`
            if req_max_step.get(str(r), -1) < last or int(not_exec_req.get(str(r), 0)) == 0:
                manifest_cut.append({"request_id": str(r), "manifest_last_step": req_max_step.get(str(r)), "cap_last_step": last,
                                     "n_exec": int(e["n_exec"]), "identities_at_not_executed_steps": int(not_exec_req.get(str(r), 0)),
                                     "why": "Q(P) carries no identity at the request's not-executed steps: the manifest was restricted before the "
                                            "population (the manifest of record stays cap-length; the class is the exclusion)"})
    # an identity at invocation >= 1 is addressed ONLY through `op_path_aliases` (a shared module's k-th call, at the
    # first layer's path); one that is the image of no alias entry is unreachable from every Program row -- the 960 + 960 on the TP2 B1 row
    # of record were this (the rank merge had dropped the table).  The CAUSE named beside the two symptom counts, fails the reconciliation.
    alias_images = {(str(c), int(k)) for c, k in aliases.values()}
    unreachable: list[dict] = []
    n_unreachable = 0
    if man is not None:
        for ident in man.identities:
            inv = int(ident.get("invocation", 0) or 0)
            if inv >= 1 and (str(ident.get("op_path")), inv) not in alias_images:
                n_unreachable += 1
                if len(unreachable) < first_n:
                    unreachable.append({"request_id": ident.get("request_id"), "step": ident.get("step"), "op_path": ident.get("op_path"),
                                        "invocation": inv, "family": ident.get("family"), "rank": ident.get("rank", 0)})
    checks = {"vus_outside_query": {"n": n_outside, "first": outside}, "identities_without_rows": {"n": n_missing, "first": missing},
              "identities_unreachable_by_alias": {"n": n_unreachable, "first": unreachable, "alias_entries": len(aliases),
                                                  "why": ("invocation >= 1 identities that no `op_path_aliases` entry maps a Program path to: the manifest "
                                                          "carries no (or a partial) alias table; the Program's per-layer rows of the shared module then "
                                                          "read as vus_outside_query and these identities as identities_without_rows"
                                                          if n_unreachable else None)},
              "rows_disagree": {"n": n_rows_disagree, "first": rows_disagree},
              "rows_annotation_inconsistent": {"n": n_rows_annot, "first": rows_annot,
                                               "note": "manifest annotation only (`rows` != element_range / width); digest-neutral, not a reach gap"},
              "ranks": {"replay": [int(r) for r in ranks], "manifest": sorted(ranks_seen) if man is not None else None,
                        "equal": (sorted(ranks_seen) == sorted(int(r) for r in ranks)) if man is not None else None},
              "accounted": (n_reached + n_uncovered + n_not_exec == total) if man is not None else None,
              "unstated_families": [u["family"] for u in uncovered if u["class"] == "unstated"],
              "controller_rows_disagree": {"n": len(ctrl_disagree), "first": ctrl_disagree[:first_n]},   # [R4 c2] header vs the replay's own count
              "executed_prefix_rule": {"applied": bool(executed_prefix), "faults": ep_faults, "evaluated_past_stop": {"n": len(past_stop), "first": past_stop[:first_n]},
                                       "manifest_cut_before_classification": {"n": len(manifest_cut), "first": manifest_cut[:first_n]}}}
    if refused is not None:
        checks["manifest_refused"] = {"field": refused.field, "why": str(refused)}
    checks["ok"] = (False if refused is not None else None if man is None else     # None = not reconciled (no Q(P) supplied), stated
                    bool(n_outside == 0 and n_missing == 0 and n_unreachable == 0 and n_rows_disagree == 0 and checks["ranks"]["equal"] is not False
                         and checks["accounted"] is not False and not checks["unstated_families"]
                         and not ep_faults and not past_stop and not manifest_cut and not ctrl_disagree))
    _qid0 = _header(man, "query_id") or "Q_module_v1"
    _noun = {"Q_module_v1": ("the module-boundary query of record", "Program row = module-boundary VU"),
             "Q_module_pad_v1": ("the declared padded-boundary query", "Program row = module-boundary VU + every slot unit of a padded MoE row"),
             "Q_fine_v1": ("the fine query (vu_canonical) through the RequiredValues path", "fine VU = relation-level partition unit"),
             "Q_fine_direct_v1": ("the directly-committed fine query", "fine VU ending at a materialised word + the module rows"),
             }.get(_qid0, ("the query of record", (_header(man, "unit") or "per its header")))
    src = ((f"required manifest: {_qid0}(P) -- {_noun[0]} over the Program (unit: {_noun[1]}; "
            "Q_fine_v1 selects the families, not the unit)" if man is not None
            else f"required manifest REFUSED ({refused}): nothing reconciled; the Program rows' own manifest addresses, NOT a Q(P) statement"
            if refused is not None
            else "NO required manifest supplied: the Program rows' own manifest addresses (the v2 query's population; NOT a Q(P) statement -- "
                 "weights / ids / hidden streams absent)"))
    # the guard's state travels with Q(P) onto the record: a manifest whose dataflow guard was not armed says so here
    guard = man.guard if man is not None else None
    guard_line = (("guard armed" if guard.get("armed") else "guard NOT armed") + f" ({guard.get('engine')})" if guard is not None
                  else ("guard state not on the manifest (its header predates the guard)" if man is not None
                        else "manifest refused" if refused is not None else "no manifest"))
    # the one-release cross-check (traversal of record, inventory beside it) travels onto the record and the scope line
    cc = man.cross_check if man is not None else None
    if cc is not None:
        guard_line += ("; cross-check " + ("AGREED" if cc.get("agree") else "DISAGREED") +
                       f" ({cc.get('primary_engine')} vs {cc.get('other_engine')}: {(cc.get('identities') or {}).get(cc.get('primary_engine'))} / "
                       f"{(cc.get('identities') or {}).get(cc.get('other_engine'))} identities)")
    elif man is not None:
        guard_line += f"; no cross-check on the manifest (one engine asked for by name: {_manifest_engine_name(man)})"
    # the query of record BY ID on the scope line (a manifest predating the id: named as such, never assumed)
    qid = (_header(man, "query_id") or "Q_module (older header, no query_id)") if man is not None else "Q"
    if total:
        scope = (f"{qid}(P) = {total} identities ({', '.join(f'{k} {v}' for k, v in sorted(fam_total.items()))}; {guard_line}); exact replay reaches {n_reached} "
                 f"({100.0 * n_reached / total:.1f}%), under an evaluated module {n_evald}")
    else:
        scope = f"{qid}(P) = 0 identities ({guard_line})"
    if uncovered:
        scope += "; uncovered by name: " + ", ".join(f"{u['spec'] or u['family']} x{u['identities']} ({u['class']})" for u in uncovered)
    if n_not_exec or ep_faults:
        scope += (f"; not-executed {n_not_exec} identities (" + ", ".join(f"{r}: a_r {e['a_r']} LAG {e['lag']} -> steps [{e['n_exec']}, {e['cap']})"
                                                                             for r, e in sorted((executed_prefix or {}).items()) if e.get('steps_not_executed')) + ")"
                  + ("; executed-prefix FAULTS: " + "; ".join(f"{f['request_id']}: {f['faults'][0]}" for f in ep_faults) if ep_faults else ""))
    return {"query_total": {"source": src, "identities": int(total), "by_family": {k: int(v) for k, v in sorted(fam_total.items())},
                            "manifest_digest": man.manifest_digest if man is not None else None,
                            "program_digest": man.program_digest if man is not None else None,
                            "query_digest": _header(man, "query_digest"),
                            "query_id": _header(man, "query_id"),                                                  # Q_module_v1
                            "query_id_digest": _header(man, "query_id_digest"),
                            "engine": (_manifest_engine_name(man) if man is not None else None),
                            "implemented_query": _header(man, "implemented_query"),
                            "unit": "Program row = module-boundary VU under Q_module_v1 (a `vus` count above is a count of those rows)",
                            "requests": man.requests if man is not None else None,
                            "guard": ({k: guard.get(k) for k in ("engine", "armed", "available", "fully_wired", "flags", "named_input") if k in guard}
                                      if guard is not None else None),
                            "cross_check": ({k: cc.get(k) for k in ("agree", "primary_engine", "other_engine", "digests", "identities", "complete", "why") if k in cc}
                                            if cc is not None else None),
                            "guard_rule": ("the 'missing family cannot vanish from both sides' guard (unnamed crossing / unnamed input / hidden-BN "
                                           "disagreement) is ARMED only when the traversal engine audited every request Program's operand runs; "
                                           "not armed = the population is the vocabulary's word alone, said here rather than read as clean")},
            "sampled_strata": {"identities_reached": int(n_reached), "identities_under_evaluated_module": int(n_evald), "row_addresses_reached": int(sum(reached_rows.values())),
                               "share_of_query": (round(n_reached / total, 4) if total else None),
                               "by_family_reached": {k: int(v) for k, v in sorted(reached.items())}, "by_family_under_evaluated_module": {k: int(v) for k, v in sorted(evald.items())},
                               "strata": (len(strata) if strata is not None else None),
                               "strata_evaluated": (sum(1 for v in strata.values() if int(v.get("evaluated") or 0) > 0) if strata is not None else None),
                               "rule": ("'complete' grades the sampled strata over the REACHED identities only; every required identity outside the "
                                        "reach is in uncovered_by_name (with its mechanism of record), never inside the grade. "
                                        "`identities_under_evaluated_module` counts every identity of a (request, step, module) with >= 1 evaluated pick "
                                        "-- one pick under a 2-member module counts 2; it is NOT a count of identities whose own elements were evaluated")},
            "uncovered_by_name": uncovered, "uncovered_identities": int(n_uncovered),
            "controller": controller,   # [R4] None on an un-lifted Program; on a lifted one the named statement above (also in uncovered_by_name)
            # the two flat keys the readers wait for, emitted from the SAME computation as the block below
            "not_executed_by_family": {k: int(v) for k, v in sorted(not_exec.items())},
            "not_executed_rule": {"rule": "not-executed(a_r, LAG)", "applied": bool(executed_prefix), "n": int(n_not_exec),
                                  "requests": {r: {"a_r": e["a_r"], "lag": e["lag"], "n_exec": e["n_exec"], "cap": e["cap"], "step_range": e.get("steps_not_executed"),
                                                   "n": int(not_exec_req.get(r, 0)), "faults": e.get("faults"), "mechanism": e.get("mechanism")}
                                               for r, e in sorted((executed_prefix or {}).items())},
                                  # WHICH mechanism produced the class, per request, BY NAME: the recurrence (one code path with
                                  # Continuation_v1) or the arithmetic (the served list / the unit absent) -- a reader sees the tie's status on the record
                                  "mechanism": {m: int(n) for m, n in sorted(Counter(str(e.get("mechanism") or "unstated").split(":")[0]
                                                                                                 for e in (executed_prefix or {}).values()).items())},
                                  "r3_one_mechanism": (bool(executed_prefix) and all(e.get("r3_one_mechanism") is True for e in executed_prefix.values())),
                                  "scope": NOT_EXECUTED_SCOPE},
            "not_executed_by_rule": {"rule": EXECUTED_PREFIX_RULE, "scope": NOT_EXECUTED_SCOPE, "applied": bool(executed_prefix), "identities": int(n_not_exec),
                                     "by_family": {k: int(v) for k, v in sorted(not_exec.items())},
                                     "totals": {"N_manifest": int(total) if man is not None else None,
                                                "N_required": (int(total) - int(n_not_exec)) if man is not None else None, "N_excluded": int(n_not_exec),
                                                # a batch-level family carries its plane-cell statement beside the identity-level counts (the key the MoE
                                                # reconciler reads): {identities: planes fully not executed, cells: (plane, request) cells past the stop}
                                                "by_family": {k: {"N_manifest": int(v), "N_required": int(v) - int(not_exec.get(k, 0)), "N_excluded": int(not_exec.get(k, 0)),
                                                                  **({"plane_cells_not_executed": {"identities": int(batch_level[k]["identities_not_executed"]),
                                                                                                   "cells": int(batch_level[k]["cells_not_executed"])}} if k in batch_level else {})}
                                                              for k, v in sorted(fam_total.items())} if man is not None else None},
                                     "by_request": {r: dict(e, identities=int(not_exec_req.get(r, 0)), excluded_unexecuted_identities=int(not_exec_req.get(r, 0)),
                                                        excluded_row_identities=int(not_exec_req_rows.get(r, 0)),
                                                        required_row_identities=int(req_ids.get(r, 0)) - int(not_exec_req_rows.get(r, 0)),
                                                        batch_level_slots_not_executed=int(batch_slots_req.get(r, 0)))
                                                    for r, e in sorted((executed_prefix or {}).items())},
                                     # the REACH of the class over every family of Q(P): kind per family (row | named-input |
                                     # per-request | batch-level | static) and the batch-level planes' per-request slots (identity-level
                                     # N_excluded counts a plane only when every listed request is past its stop; the Commit's per-request
                                     # expansion reconciles against request_slots_not_executed)
                                     "kind_by_family": {**{k: "row" for k in sorted(fam_total) if k in QUERY_FAMILIES_WITH_ROWS}, **dict(sorted(kind_by_family.items()))},
                                     "batch_level": {k: dict(v) for k, v in sorted(batch_level.items())}},
            "query_checks": checks, "scope": scope}


def _uncovered_mechanisms(all_vus: list[dict], not_ev_fam: Counter) -> list[dict]:
    """The Program's attention rows are evaluated on their committed boundary (the Q_module_v1 unit); the
    AttnBlock interiors inside them (Q_fine_v1 units) are not read -- the record names that with the row count, so a PASS bounds its claim."""
    n = sum(1 for v in all_vus if v["family"] in ATTENTION_ROW_FAMILIES) + sum(int(not_ev_fam.get(f, 0)) for f in ATTENTION_ROW_FAMILIES)
    if n == 0:
        return []
    return [{"mechanism": "attention block interiors (Q_fine_v1 units inside the evaluated Attention rows)", "rows": int(n),
             "query_of_record": "Q_module_v1", "unit": "the Attention row (module boundary): committed q + committed K/V history -> committed out",
             "why": ATTENTION_INTERIOR_MECHANISM,
             # what happens to the committed interior words meanwhile -- stated, not implied
             "committed_words_status": "fa2_hidden_m1_stream / fa3_hidden_m1_stream rows are COMMITTED (coverage + openings) and value-UNCHECKED by "
                                       "this replay; form (B) does not compare them either unless oracle_compare.compared_by_family names "
                                       "hidden_stream (the record's not_compared_by_family.hidden_stream is the count) -- Stage B (Q_fine_v1)"}]


def form_b_families(manifest: RequiredManifest | dict[str, Any] | None) -> dict[str, int]:
    """Producer family -> the manifest's required `instance_outputs` identities this replay has no registered evaluator for (outside
    NOT_YET, not a token selector).  The Commit value-checks those by form (B) only, which is complete only over a Match that snapshotted
    every step (commit_verdict._form_b_complete_over_values): the Match reads this to pick its snapshot steps."""
    out: Counter = Counter()
    for r in RequiredManifest.of(manifest, HANDED).identities if manifest is not None else ():
        if r.get("family") != "instance_outputs" or not r.get("spec"):
            continue
        fam = str(r["spec"]).split("{", 1)[0].strip()
        if fam not in EVALUATORS and fam not in NOT_YET and not RM.is_token_select(fam):
            out[fam] += 1
    return dict(out)
