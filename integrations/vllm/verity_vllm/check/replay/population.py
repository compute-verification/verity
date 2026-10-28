"""The VU population of a request's Program: every row instance with its stratum (spec, request, phase), minus the
Calls outside the executed prefix, which leave the pool into a named class before any draw (`executed_prefix_of_record`,
`split_not_executed`)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Callable, Iterable, Sequence

from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.check.replay.replay_codes import NOT_YET, why_class
from verity.ml.definitions.numerics.rows import EVALUATORS
from verity_vllm.program.lifting.continuation import executed_prefix_by_rule
from verity_vllm.check.replay.delegate import DELEGATES
from verity_vllm.check.replay.index import _MOE_LAYER_RE, SAMPLER_OP, ProgramIndex, const_word, node_runs
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.check.replay.evaluate import INTERIOR_DEPTH


#: the attention families whose ROW the replay evaluates; their block interiors are Q_fine_v1 units
ATTENTION_ROW_FAMILIES = ("Attention_v3", "Attention_v2", "Attention_v4", "Attention_v5", "Attention_v18", "AttentionSoftcap_v2")


def _phase(step: int, last_step: int) -> str:
    """Phase axis of the stratum key.  `last_step` = the request's LAST EXECUTED step: cap - 1, or -- under an executed
    prefix -- n_exec - 1, the lag step of an EOS-stopped request (the executed prefix's boundary), so that request's
    `last` stratum exists and is sampled deterministically like every other request's; steps past it are 'not-executed' (split out of the
    pool by name, never a 'last' or a 'mid')."""
    if step == 0:
        return "prefill"
    if step == last_step:
        return "last"
    if step > last_step:
        return "not-executed"
    return "first-decode" if step == 1 else "mid"


def phase_anchor(pi_n_steps: int, ep: dict | None) -> dict[str, Any]:
    """The step the `last` phase of a request is anchored to, NAMED: n_exec - 1 under a fault-free executed prefix that
    stops before the cap (the lag step), else cap - 1.  On the record per request (`strata_keys.phase_anchor`)."""
    cap_last = int(pi_n_steps) - 1
    if ep and not ep.get("faults") and int(ep["n_exec"]) < int(ep["cap"]):
        n_exec, lag = int(ep["n_exec"]), int(ep.get("lag") or 0)
        # under LAG 0 the last executed step is the EOS step, not a lag step -- the text says which
        last_is = "the lag step" if lag > 0 else "the EOS step (LAG 0: no lag step)"
        return {"last_step": n_exec - 1, "anchor": "n_exec - 1", "cap_last_step": cap_last, "n_exec": n_exec,
                "lag_step": (n_exec - 1 if lag > 0 else None),
                "why": (f"the request stopped at a_r {ep.get('a_r')} under LAG {lag}: steps [{n_exec}, {int(ep['cap'])}) are not executed; "
                        f"its last EXECUTED step {n_exec - 1} is {last_is} -- the `last` stratum is anchored there, not at the cap")}
    if ep and not ep.get("faults"):
        # an EOS stop at a_r = cap - LAG under LAG > 0 has n_exec == cap: the cap anchors, AND its last step IS a lag
        # step -- the flag follows the STOP (a_r < cap and LAG > 0), not n_exec < cap.
        n_exec, a_r, lag = int(ep["n_exec"]), int(ep.get("a_r") if ep.get("a_r") is not None else ep["cap"]), int(ep.get("lag") or 0)
        eos_stop_lag = a_r < int(ep["cap"]) and lag > 0
        return {"last_step": cap_last, "anchor": "cap - 1", "cap_last_step": cap_last, "n_exec": n_exec,
                "lag_step": (n_exec - 1 if eos_stop_lag else None),
                "why": (f"the request stopped at a_r {a_r} under LAG {lag} and its executed prefix reaches the cap (n_exec {n_exec} == cap): the cap "
                        f"anchors and its last step {n_exec - 1} is the lag step" if eos_stop_lag else "the request ran to its cap")}
    return {"last_step": cap_last, "anchor": "cap - 1", "cap_last_step": cap_last, "n_exec": None,
            "lag_step": None, "why": ("no executed prefix" if not ep else "executed prefix has faults: the cap anchors")}


def _lifetime(engine_step: int, ring_slots: int | None) -> str:
    """Storage-reuse axis: with a windowed collector the step's device window is ring slot (engine_step % ring_slots); from engine step
    ring_slots on, every committed identity occupies storage the run already committed (and released) under an earlier step's identities
    -- the post-overwrite / stale-window reads live there.  Unwindowed collectors: one value, stated."""
    if not ring_slots or int(ring_slots) <= 0:
        return "n/a (unwindowed storage)"
    return "recycled-window" if int(engine_step) >= int(ring_slots) else "fresh-window"


def moe_plane_alias_check(store: CommittedStore, *, max_named: int = 12) -> dict[str, Any]:
    """[FLUSH-03 fact of record] per (request, step, MoE layer) with BOTH planes committed: the committed `up` bytes differ
    from the committed `down` bytes.  Before FLUSH-03 the gather collector committed the same layer's down plane under the up name (vLLM's
    TritonExperts carves intermediate_cache1 = up and intermediate_cache3 = down out of ONE workspace; with no flush between the layer's two
    GEMM launches the staged up view was read after the w2 launch had overwritten it -- 22/22 same-layer pairs identical
    on the OLMoE records).  The comparison is over the COMMON PREFIX of the two request-local slices (min(len) bytes): when
    2I == H (OLMoE) that is the whole plane; when the widths differ the alias would still show as identical leading bytes (one base pointer,
    two row strides), so a prefix test catches it on every geometry.  Result: True = every compared pair differs; False = an identical pair
    (named: request, step, layer, bytes) -- the up plane is the down plane's alias; None = no MoE layer with both planes committed (dense
    model, or the planes not bound).  A FACT beside the replay (the plane GEMM relation already fails an aliased plane by name);
    the known-good root re-pin of the OLMoE reference stands on this line, not on the run reproducing itself."""
    layers: dict[tuple[str, int, int], dict[str, tuple[int, int]]] = {}
    for (rid, step, op, mem), es in store.idx.items():
        m = _MOE_LAYER_RE.match(str(op))
        if not m or mem not in ("up", "down"):
            continue
        rng = None
        for e in es:                                                            # the bound request-local range (every invocation under the key)
            r = (e.get("identity") or {}).get("element_range") or [0, None]
            if r[1] is not None:
                rng = (int(r[0]), int(r[1])) if rng is None else (min(rng[0], int(r[0])), max(rng[1], int(r[1])))
        if rng is not None:
            layers.setdefault((rid, int(step), int(m.group(1))), {})[mem] = rng
    compared, identical, errors, per_layer = 0, [], [], {}
    widths: dict[str, set[int]] = {"up": set(), "down": set()}
    for (rid, step, layer), mems in sorted(layers.items()):
        if "up" not in mems or "down" not in mems:
            continue
        got: dict[str, bytes] = {}
        for mem in ("up", "down"):
            lo, hi = mems[mem]
            w, why = store.words(rid, step, f"moe/L{layer}", mem, lo, hi)
            if w is None:
                if len(errors) < max_named:
                    errors.append({"request_id": rid, "step": step, "layer": layer, "plane": mem, "why": str(why)[:200]})
                break
            got[mem] = w.tobytes()
            widths[mem].add(hi - lo)
        if len(got) < 2:
            continue
        n = min(len(got["up"]), len(got["down"]))
        compared += 1
        pl = per_layer.setdefault(f"L{layer}", {"compared": 0, "identical": 0})
        pl["compared"] += 1
        if n > 0 and got["up"][:n] == got["down"][:n]:
            pl["identical"] += 1
            if len(identical) < max_named:
                identical.append({"request_id": rid, "step": step, "layer": layer, "bytes_compared": n,
                                  "sha256": hashlib.sha256(got["up"][:n]).hexdigest()[:16]})
    n_ident = sum(v["identical"] for v in per_layer.values())
    result = None if compared == 0 else (n_ident == 0)
    return {"schema": "verity-sweep/moe-plane-alias-check/v1", "result": result, "compared": compared, "identical_n": n_ident,
            "identical": identical, "read_errors_n": len(errors), "read_errors": errors, "per_layer": per_layer,
            "plane_words": {k: sorted(v) for k, v in widths.items()},
            "rule": "for every (request, step, MoE layer) with both planes committed: committed up bytes != committed down bytes over their common "
                    "prefix (min length); identical = the up plane is the down plane's alias (pre-FLUSH-03 defect); None = no MoE layer with both planes",
            "why": (None if result is not False else
                    f"{n_ident} of {compared} (request, step, layer) triples commit IDENTICAL up and down plane bytes -- the up plane is the down plane's "
                    f"alias (FLUSH-03): first {identical[0]['request_id']} step {identical[0]['step']} L{identical[0]['layer']}")}


def _interior_feeders_without_evaluator(pi: ProgramIndex, r: dict, *, depth: int = 0) -> set[str]:
    """The INTERIOR producers this row's VU would have to compose that
    this replay cannot evaluate: an `['n', producer, lo, hi]` input whose producer row claims no committed member (interior: the pre-bias
    Gemm under a BiasAdd) and whose family has no registered evaluator (Gemm_v2 on Hopper / batch-invariant Ada Programs), or a
    composable interior whose own inputs have that defect (bounded by INTERIOR_DEPTH).  Derived from the Program rows alone (families +
    member claims), never from the collector's emissions.  Empty when every interior input composes (Gemm_v1 under BiasAdd_v1)."""
    out: set[str] = set()
    for group in r.get("args") or []:
        for sl in node_runs(group):
            a, lo, hi = int(sl[1]), int(sl[2]), int(sl[3])
            if pi.member_at(a, lo, hi) is not None:
                continue                                                         # a committed member: read at replay, nothing to compose
            pr = pi.by_i.get(a)
            if const_word(pr) is not None:
                continue                                                         # a Program constant (Const32/64[..]): read from the text, nothing to compose
            pd = pi.describe(pr) if pr is not None else None
            if pd is None or pd["members"]:
                continue                                                         # unmodelled / a member-straddling slice: a replay-time fault, not a name
            pf = pd["family"]
            if pf not in EVALUATORS and pf not in DELEGATES:
                out.add(f"{pf} ({NOT_YET.get(pf, 'no registered evaluator')})")
            elif depth + 1 >= INTERIOR_DEPTH:
                out.add(f"{pf} (interior chain deeper than {INTERIOR_DEPTH})")
            else:
                out |= _interior_feeders_without_evaluator(pi, pr, depth=depth + 1)
    return out


def population(pi: ProgramIndex, rid: str, *, rank: int = 0, arrive: int = 0, ring_slots: int | None = None,
               addresses: dict[tuple, dict] | None = None, last_step: int | None = None,
               codes: dict[str, dict] | None = None) -> tuple[list[dict], Counter, Counter]:
    """Every Program row of one request is in the population: the evaluable VUs [{i, spec, family, phase, stratum}], the not-evaluable
    count by reason, and the not-evaluable count BY FAMILY (strata of the record by name, never dropped).
    `ring_slots`: the collector's window ring depth -- an identity committed at ENGINE step >= ring_slots lives in storage the ring
    already committed under another step's identities (window recycle): the `lifetime` stratum axis.
    `addresses`: an accumulator keyed by the row's MANIFEST address (request, request-relative step, op_path, member) ->
    {spec, family, class ('vu' | a why class), why, rows}: every row that produces a required identity is recorded at that identity,
    evaluable or not, so `query_population` can reconcile the replay's reach against Q(P) identity by identity (a row with no member --
    structure, interior -- produces no identity and is not recorded).
    `last_step`: the request's last EXECUTED step (`phase_anchor(...)["last_step"]`: n_exec - 1 under an executed prefix,
    the lag step of an EOS-stopped request); None = the cap's last step.  The POPULATION is the same either way (every row is enumerated);
    only the phase label of the stratum key moves.
    `codes`: an accumulator, not-evaluable reason -> its structured code (`replay_codes.reason_code`), what the verdict reads."""
    vus, not_ev, not_ev_fam = [], Counter(), Counter()
    last = pi.n_steps - 1 if last_step is None else int(last_step)

    def _not_ev(why: str, cls: str, fam: str, *, domain: bool = False, tags: tuple[str, ...] = ()) -> None:
        not_ev[why] += 1
        if codes is not None:
            codes[why] = RC.reason_code(cls, fam, domain=domain, tags=tags)

    def _addr(d: dict | None, fam: str, spec: str, cls: str, why: str | None) -> None:
        if addresses is None or d is None:
            return
        # [MOE-TP-02] a member committed under a second required name (the TP rank's MoeSum: part_rank<r> AND the moe/L<k>/sum plane) is
        # recorded at BOTH identities -- the same row, the same class
        # `manifest_family`: the manifest family the address belongs to (the label build_manifest assigns), read off the address rule that
        # produced it -- the sampler op, a plane description (`plane_kind`) / a plane copy (MOE-TP-02), else the member's own family
        # (`manifest_format.alias_family`: a rank-local partial vs an instance output) -- never parsed back out of the op_path string
        keys = [(str(d["op_path"]), str(m[0]), "moe_block_stream" if d.get("plane_kind") else None) for m in d["members"]] \
            + [(str(c["op_path"]), str(c["member"]), "moe_block_stream") for c in d.get("copies") or []]
        for op, mem, mfam in keys:
            k = (rid, int(d["step"]), op, mem)
            a = addresses.get(k)
            if a is None:
                mfam = "sampled_token_ids" if op == SAMPLER_OP else (mfam or RM.alias_family(mem))
                addresses[k] = {"spec": spec, "family": fam, "class": cls, "why": why, "rows": 1, "manifest_family": mfam}
            else:
                a["rows"] += 1
                if a["class"] != cls:                                            # one identity, rows of two classes: named, never averaged
                    a["class"], a["why"] = "mixed", f"{a['class']} + {cls}: {a['why']} | {why}"

    for r in pi.rows:
        fam = RM.spec_family(r["spec"])
        if fam in NOT_YET:
            if addresses is not None:
                try:
                    d = pi.describe(r)
                except Exception as e:  # noqa: BLE001
                    # its identity would be missing from the reconciliation unseen: a mapping fault by name, not the family's domain label
                    _not_ev(f"{fam} row address unmodelled (the manifest address rule raised: {type(e).__name__}: {e})"[:240], "mapping", fam)
                    not_ev_fam[f"{fam} (address unmodelled)"] += 1
                    continue
                _addr(d, fam, r["spec"], why_class(NOT_YET[fam]), NOT_YET[fam])
            _not_ev(NOT_YET[fam], why_class(NOT_YET[fam]), fam, domain=True)
            not_ev_fam[fam] += 1
            continue
        d = pi.describe(r)
        if d is not None and d.get("plane_error"):
            # a fused-MoE layer row whose row -> plane slice mapping the Program does not support (inconsistent geometry, an
            # ordinal outside the plane, a name index that disagrees with Program order): NAMED, class `mapping` -- a fault of the record's
            # Program shape, never a guess at another pair's words and never carried on a PASS.
            _not_ev(f"MoE plane row -> plane slice mapping failed: {d['plane_error']}", "mapping", fam)
            not_ev_fam[f"{fam} (plane mapping failed)"] += 1
            continue
        if fam not in EVALUATORS and fam not in DELEGATES:
            if d is None:
                _not_ev(f"unmodelled family {fam} (manifest incomplete -- coverage fails closed)", "mapping", fam)
                not_ev_fam[fam] += 1
            elif not d["members"]:
                not_ev_fam[f"{fam} (structure / interior: no identity)"] += 1        # Const*, BiasAdd, ...: counted, no relation to replay
            elif d.get("plane"):
                # a plane PRODUCER (router / expert GEMM / sum) whose relation has no evaluator in this version: its
                # plane IS read -- as the committed input of the rows it feeds (up -> the act SiluMul, sum -> the next fused norm) -- but
                # the plane's own production is not re-evaluated.  Named under the moe_block_stream class so the verdict's (B)-conditioned
                # population-gap rule keeps gating it until the evaluator lands (unit 2), never a silent 'no evaluator' domain exclusion.
                w = (f"no registered evaluator for {fam}: moe_block_stream plane producer ({d['plane']} plane read as its consumers' committed input; "
                     f"evaluator = MAN-06f v3 unit 2)")
                _not_ev(w, "plane-fed", fam, tags=("MAN-06f",))
                not_ev_fam[f"{fam} (moe_block_stream plane producer: no evaluator)"] += 1
                _addr(d, fam, r["spec"], "plane-fed", w)
            else:
                w = f"no registered evaluator for {fam}"
                _not_ev(w, "no-evaluator", fam)
                not_ev_fam[fam] += 1
                _addr(d, fam, r["spec"], "no-evaluator", w)
            continue
        if d is None:
            _not_ev(f"unmodelled family {fam}", "mapping", fam)
            not_ev_fam[fam] += 1
            continue
        if not d["members"]:                                                     # an evaluable family whose row has no
            not_ev_fam[f"{fam} (interior: composed into its epilogue's VU / a hinted row of the module is its returned tensor)"] += 1
            continue                                                             # committed member of its own: the pre-bias Gemm under a BiasAdd,
                                                                                 # the pre-cap logits Gemm -- interior, no VU, counted by name
        ife = _interior_feeders_without_evaluator(pi, r)
        if ife:
            # the VU's input is an INTERIOR (uncommitted) value of a fused producer
            # this replay cannot compose (the pre-bias Gemm_v2 under Qwen2.5's biased qkv_proj: BiasAdd_v1 is the producer of record, its
            # input is interior, Gemm_v2 has no evaluator): counted BY NAME as not evaluable -- the same shape as the MoE planes -- never
            # sampled as a stratum that stays 'not evaluated' (partial = FAIL closed on every biased-linear row).  The values ARE required
            # and covered by coverage + the C2 oracle-compare; the composed evaluator F(x, W, b) is the v2 follow-up.
            w = f"fed by interior producer(s) without an evaluator ({', '.join(sorted(ife))}): composition not possible in this replay (MAN-06g, v2)"
            _not_ev(w, "interior-composed-v2", fam, tags=("MAN-06g",))
            not_ev_fam[f"{fam} (fed by interior {', '.join(sorted(x.split(' (')[0] for x in ife))}: composed evaluator, v2)"] += 1
            _addr(d, fam, r["spec"], "interior-composed-v2", w)
            continue
        _addr(d, fam, r["spec"], "vu", None)
        st = d["step"]
        # stratum = (spec incl. shape, request, rank, phase, MODULE op_path): the module axis makes every (module, request, phase) cell
        # a stratum of its own, so a relabel / a wrong-value emission of ANY module at a request's prefill, first-decode or last step is
        # sampled deterministically (the cell holds one row group); only a mid-phase single-identity fault is sampled with probability
        # n / (steps in the phase) -- targets (`--replay-target`) pin those for the negatives of record
        life = _lifetime(arrive + st, ring_slots)
        sspec = _stratum_spec(fam, r["spec"], d.get("statics") or {})
        vus.append({"i": int(r["i"]), "spec": r["spec"], "family": fam, "request_id": rid, "step": st, "phase": _phase(st, last), "rank": int(rank),
                    "op_path": d["op_path"], "lifetime": life, "stratum": f"{sspec}|{rid}|rank{int(rank)}|{_phase(st, last)}|{d['op_path']}|{life}"})
    return vus, not_ev, not_ev_fam


#: the scope of the class, verbatim on the rule, on every record that carries it, and in the C2 schema doc
NOT_EXECUTED_SCOPE = ("NOT-EXECUTED-BY-RULE is an active-reference implementation classification only. In the lifted fixed Program the "
                      "corresponding positions are REQUIRED logical values assigned the canonical inactive representation tau(e) = bot; "
                      "there is no not-required-because-not-executed category.")

#: the executed prefix of a request, as the C2 population CLASSIFIES it (never as it shrinks by it).
EXECUTED_PREFIX_RULE = ("a request that served a_r < cap tokens (its served token a_r-1 a declared EOS id of the Program's stop rule) "
                        "executed exactly n_exec = min(a_r + LAG, cap) request-relative steps; a request served to its cap executed cap "
                        "(a length stop: no post-EOS row even if the cap token is an EOS). RequiredValues(P, Q_module_v1) stays the "
                        "cap-length manifest of record (fixed population across stopping points; digest unchanged): its identities "
                        "at request-relative steps >= n_exec are the named class not-executed(a_r, LAG) -- counted in query_total, "
                        "absent from the sampling pool, never 'not evaluated' and never silently absent; reached + uncovered + not-executed "
                        "== |Q(P)|. CONDITIONS: (a) a_r is READ from the Match record of record (G6 lengths_by_request / tokens.json) "
                        "and the Commit's own committed served length must EQUAL it by name -- the committed tokens count only because they "
                        "are linked to the served output and the Match account (C2 boundary linkage + tokens_equal + G6 agree); 'the committer "
                        "says EOS happened here' is never sufficient; (b) LAG is the Program's declared static (workload_lag_static), attested "
                        "declared == record_attests == tokens.json; (c) any committed row of the request at a step >= arrival + n_exec is "
                        "'execution past the stop' and FAILS. A request failing any condition is NOT classified: its prefix carries faults and "
                        "the record's query_checks fail by name. " + NOT_EXECUTED_SCOPE)


MECH_RECURRENCE = "recurrence(Continuation_v1): n_exec = |{t : r_t = 1}| over the served list (first EOS stops; A = LAG); min(a_r + LAG, cap) is the CHECK"
MECH_ARITH_NO_SERVED = ("arithmetic(a_r, LAG): n_exec = min(a_r + LAG, cap) -- the served list was not supplied, the class is not derived from r_t "
                        "(pass served= / eos_ids= to derive it)")


def executed_prefix_of_record(rid: str, *, a_r: int, lag: int, cap: int, arrive: int = 0, eos_declared: bool | None = None,
                              served_len_commit: int | None = None, linked: bool | None = None, lag_attested: bool | None = None,
                              rows_past_stop: int = 0, a_r_source: str, served: Sequence[int] | None = None, eos_ids: Iterable[int] | None = None,
                              rule: Callable[..., dict[str, Any]] | None = None, account_missing: Sequence[str] | None = None) -> dict[str, Any]:
    """Normalise one request's executed prefix for the population.  The CALLER (the Commit) supplies the facts of record -- `a_r` READ
    from the Match record, `served_len_commit` its own committed served length, `linked` (C2 boundary linkage + tokens_equal + G6 agree),
    `lag_attested` (declared == record_attests == tokens.json), `eos_declared` (served token a_r-1 is a declared EOS id), `rows_past_stop`
    (committed rows of the request at steps >= arrival + n_exec, from the Commit's coverage) -- this function only fixes the arithmetic and
    NAMES the faults: `faults` non-empty means the class is NOT applicable to this request and the record fails by name (conditions (a)(b)(c)
    of `check.executed_prefix`).  Every condition is fail-closed: an unasserted fact (None) is a fault, not a default.
    `a_r_source` is REQUIRED and names WHERE a_r was read (the Match record file + digest, as executed_prefix_facts passes it); it has
    no default, so a caller passing the Commit's own count cannot put 'match record' on the record by omission -- an own count must be
    labelled as such by the caller, and (a) then faults on `linked`.  [ONE mechanism] with `served` (the request's served token ids of
    record) and `eos_ids` (the Program's declared EOS ids) the class is DERIVED FROM r_t: `rule` (default Continuation_v1's
    `executed_prefix_by_rule`) runs the recurrence over the served list, `n_exec = |{t : r_t = 1}|`, and the arithmetic `min(a_r + LAG, cap)`
    is demoted to a CHECK (a disagreement is a fault by name); the rule's own problems (served past the first EOS, a last served token that is no declared EOS, over the cap) are
    faults.  Without `served` the arithmetic is the mechanism and is NAMED as such in `mechanism` (never silently the same thing)."""
    a_r, lag, cap, arrive = int(a_r), int(lag), int(cap), int(arrive)
    faults: list[str] = []
    if account_missing:
        # FIRST, so the verdict names the cause: a Match account leg the link needs is not beside the Commit
        # (e.g. match/capture/ moved to /vault under a queued Commit).  Still fail-closed -- (a) and (b) below fault too --
        # but an operator reads 'input missing: restore tokens.json' and not 'the Program's executed prefix is wrong'.
        faults.append(f"Match account leg(s) missing beside the Commit: {', '.join(str(x) for x in account_missing)} -- the link cannot hold on a "
                      "missing input (restore the Match record's file(s) and re-run the Commit; not a verdict on the Program) (executed-prefix conditions (a), (b))")
    if not isinstance(a_r_source, str) or not a_r_source.strip():
        faults.append("a_r_source unstated: the record must name where a_r was read (executed-prefix condition (a))")
    if lag not in (0, 1):
        faults.append(f"LAG {lag} outside the supported profile domain {{0, 1}}")
    if a_r < 0 or a_r > cap:
        faults.append(f"served length a_r {a_r} outside [0, cap {cap}]")
    n_exec = cap if a_r >= cap else min(a_r + lag, cap)                        # the arithmetic: the mechanism without a served list, else the CHECK
    mechanism, by_rule = MECH_ARITH_NO_SERVED, None
    if served is not None:
        rule = rule or executed_prefix_by_rule
        served_l = [int(t) for t in served]
        if len(served_l) != a_r:
            faults.append(f"served list of {len(served_l)} token(s) != a_r {a_r} ({a_r_source}): the recurrence would run over another prefix than the record's")
        if lag in (0, 1):
            blk = rule(served_l, cap=cap, A=lag, eos=[int(e) for e in (eos_ids or ())])
            by_rule = {k: blk.get(k) for k in ("executed_prefix_len", "eos_step", "lag_steps", "first_eos_index", "served_past_first_eos", "cap_exhausted", "eos_stop", "problems")}
            for pr in blk.get("problems") or []:
                faults.append(f"Continuation_v1 rule: {pr}")
            if not blk.get("problems") and len(served_l) == a_r:
                n_rule = int(blk["executed_prefix_len"])
                if n_rule != n_exec:
                    faults.append(f"the arithmetic min(a_r + LAG, cap) = {n_exec} disagrees with the recurrence |{{t : r_t = 1}}| = {n_rule} (R3: the rule is the mechanism, "
                                  "the arithmetic the check)")
                n_exec = n_rule
            mechanism = MECH_RECURRENCE
    if a_r < cap and eos_declared is not True:
        faults.append(f"a_r {a_r} < cap {cap} but served token a_r-1 is not asserted a declared EOS id (eos_declared={eos_declared}): "
                      "an early stop the Program's stop rule does not explain -- the not-executed class is NOT applicable, FAIL by name")
    if linked is not True:                                                       # (a)
        faults.append(f"a_r is not linked to the served output and the Match account (linked={linked}): C2 boundary linkage + tokens_equal + G6 "
                      "must agree before the committer's stop counts (condition (a))")
    if served_len_commit is None or int(served_len_commit) != a_r:
        faults.append(f"Commit served length {served_len_commit} != record a_r {a_r} (a Commit that stops earlier or later than the record fails by "
                      "name; it does not shrink or grow its required population)")
    if lag_attested is not True:                                                 # (b)
        faults.append(f"LAG rule not attested (declared == record_attests == tokens.json required; lag_attested={lag_attested}) (condition (b))")
    if int(rows_past_stop) > 0:                                                  # (c)
        faults.append(f"execution past the stop: {int(rows_past_stop)} committed row(s) of the request at steps >= arrival {arrive} + n_exec {n_exec} (condition (c))")
    return {"request_id": str(rid), "a_r": a_r, "a_r_source": a_r_source, "lag": lag, "cap": cap, "arrive": arrive, "n_exec": int(n_exec),
            "executed_prefix": int(n_exec), "steps_not_executed": [int(n_exec), cap] if n_exec < cap else None, "eos_declared": eos_declared,
            "served_len_commit": served_len_commit, "linked": linked, "lag_attested": lag_attested, "rows_past_stop": int(rows_past_stop),
            "account_missing": [str(x) for x in (account_missing or ())],
            # a faulted request produced NO class: the mechanism is 'not classified', never 'recurrence' beside a fault
            "mechanism": (mechanism if not faults else f"not classified: {len(faults)} fault(s) -- the class is not applicable to this request"),
            "r3_one_mechanism": (by_rule is not None and not faults), "by_rule": by_rule, "faults": faults}


def split_not_executed(vus: list[dict], executed_prefix: dict[str, dict] | None) -> tuple[list[dict], list[dict]]:
    """(rows the sampler may sample, rows at request-relative step >= n_exec of their request).  No `executed_prefix` -> nothing is split."""
    if not executed_prefix:
        return list(vus), []
    keep, out = [], []
    for v in vus:
        ep = executed_prefix.get(str(v["request_id"]))
        if ep is not None and not ep.get("faults") and int(v["step"]) >= int(ep["n_exec"]):
            out.append(v)
        else:
            keep.append(v)
    return keep, out


_T_STATIC = re.compile(r"\bT=\d+,")


def _stratum_spec(fam: str, spec: str, statics: dict[str, Any]) -> str:
    """The spec as the STRATUM key.  An attention row's spec carries its history length T (one value per token of
    the request), so 'spec incl. shape' would make every attention row of the run its own stratum -- an exhaustive replay of the family
    (~0.1 ms x T per row; B8 1024/128: 8 x 32k rows x up to 1152 committed reads each), not a sample.  The stratum keeps the numerical
    MECHANISM the length selects -- NB = ceil(T / BN), the number of key tiles visited, and `last=full|partial`: whether
    the last tile is FULL (T mod BN == 0: no masked lane) or PARTIAL (masked lanes -> exact zeros through exp2 and the lane sums) -- three
    chains (single tile / multi-tile / partial last tile) that NB alone does not keep apart: measured on llama L40S B8 256/32, 32 of 80
    NB-only strata mixed the one full-tile row (T = 128) with 127 partial ones, so the full-tile chain was picked with probability
    per_stratum / 128 where the spec-as-stratum key had made it certain.  The phase axis keeps prefill / first-decode / mid / last:
    `Attention_v2{NB=9,last=partial,NH=..,...}` x request x phase x module x lifetime; <= +1 stratum per formerly mixed stratum.  Every
    other family's spec is its own key (Gemm{K,N}, RMSNorm{N}, ... carry no per-row length)."""
    if fam not in ATTENTION_ROW_FAMILIES or "T" not in statics or "BN" not in statics:
        return spec
    try:
        t, bn = int(statics["T"]), max(1, int(statics["BN"]))
    except (TypeError, ValueError):
        return spec
    nb = -(-t // bn)
    return _T_STATIC.sub(f"NB={nb},last={'full' if t % bn == 0 else 'partial'},", spec, count=1)
