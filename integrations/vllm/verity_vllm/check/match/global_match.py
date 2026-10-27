"""cb_a.global_match/v1 -- the GLOBAL Match of a batched execution against one rooted workload Program (GM-01): the
runtime match check of record.  Glossary (Draft 3 §12): a fold 'instance' is a Call, 'attribution' is occurrence resolution, and the
capture log (RunHeader / StepStart / SampleEnd / StepLink rows) is execution evidence.

X-09 (`batch_decomp`) establishes, per request, that the request's projection of the batched eager fold is canonical-equal to its
request-wrapper component Program.  That is evidence INSIDE this check, not the verdict.  The global check consumes

  * the global workload Program `G` (`cb-a/global-program/v1`: shared prescribed inputs, one component per declared request, the
    declared arrival schedule and bounded output interfaces, the address-space convention) -- the built rooted Program, or,
    until it exists, `mock_global_program` assembled from the workload manifest + the Build's request components;
  * the batched record: `<oracle>/program.json` + `instances.jsonl` (the fold) and `<oracle>/capture/log.jsonl.gz` (RunHeader,
    StepStart, SampleEnd, StepLink: the observed chronology), optionally `<oracle>/capture/tokens.json` (the driver's realized
    schedule and generated tokens);

and establishes (each a named check with PASS / FAIL and evidence; the verdict is PASS iff every check passes):

  G1 requests_declared     the workload manifest, G and the record's RunHeader name the same request set; no request omitted or
                           duplicated; every request's prompt is the declared prompt (input association).
  G2 requests_observed     every declared request executed (has rows, a prefill, output events); no instance of the fold is left
                           unattributed; no request appears that was not declared.
  G3 subcircuits_x09       per request: projection canonical-equal to its component (X-09 sequence criterion) for every EXECUTED
                           step; a request that stopped early (EOS) is compared over its executed prefix of the BOUNDED component
                           and labelled `eos_truncated` with the unexecuted step count -- never rebuilt at its stopping length.
  G4 identity_map          request-local identity (request, local step, local index) <-> global fold node is a bijection: total on
                           the attributed fold, injective, and the local side is exactly the component's executed instance set.
  G5 shared_state          weights are ONE global prescribed input read by every request (same field map, no overlapping ranges);
                           every node operand of a request's instance is produced by the same request (0 cross-request reads) in the
                           same or an earlier engine step; the classes of cross-step producers are reported (KV cache, tokens).
  G6 chronology            declared arrival (engine-step units) == realized (driver) == observed (first StepStart with the request's
                           rows); rows contiguous from arrival to finish; every output event (SampleEnd over the request's logits
                           row) is one TokenSelect of the request in that engine step, the k-th event is the k-th generated token,
                           feeds the request's next input row (StepLink / StepStart.input_token_ids), and the event count respects
                           the bounded output interface (== max_tokens on `length`, < on EOS).
  G7 vu_population         the population the global query exposes (every instance of every component over its executed steps)
                           equals the fold's resolved population; probes on both sides of every request boundary inside a shared
                           engine step resolve to the right owner / local identity / spec.
  G8 within_step_order     M-ARRIVAL-ORDER: inside an engine step the order of one request's own instances must be its component's
                           order (checked by G3); the interleaving of DIFFERENT requests' rows is the runtime's batch layout
                           (StepStart.rows), which the Program does not prescribe -- reported, never re-sorted.  A request whose
                           own prefill instances come out permuted is FAIL here with the permutation characterised precisely.

Time unit of the chronology: the ENGINE STEP (one model forward), counted from 0 = the first forward of the run.  Ordering
convention: programmed dependencies and step indices only -- no statement about elapsed time is made or implied.

Request identity: the Program, the workload manifest and every check name requests `r<k>` (k = manifest position).
The record (RunHeader / StepStart rows / tokens.json / fold meta) names them by the manifest's UUID -- the id vLLM served them
under; the checker maps those through the global Program's `workload_request_id` table (G1 verifies the UUID set == manifest).

Within-step order: `within_step="dag"` (the DEFAULT of record) compares
a request's step as an order-insensitive Merkle DAG multiset with structural canonical addresses; `within_step="sequence"` (the r14-v1
verdicts) compares it as X-09's instance SEQUENCE (spec + wiring by position). The other criterion's verdict is recorded beside the
verdict of record (`alternate_verdict`) and never merged into it. The dag criterion uses X-09's chunked-prefill Merkle multiset
(`batch_decomp.dag_hashes`) and addresses every fold Call by the canonical index of its structurally identical component Call, so the fold's emission order (AttentionScope: one group per
distinct T in first-occurrence order over the batch) is recorded in G8 as batch layout, never used as an address and never
re-sorted; a consumer emitted before its producer, a wiring change or a missing/extra Call still FAILS G3.

Layout: this module holds the entry points (`check`, the CLI, the MATCH_IMPL dispatch) and the `_check` driver, which runs the
named checks in order over one `match_run.MatchRun`.  The checks live beside it: `declaration` (the global Program, the record's
engine facts, G1), `record` (the capture log, canonical request ids), `attribution` (fold instance -> request, G2), `per_request`
(G3, G4, G5), `chronology` (G6), `sampler_geometry` (the derived S G6 checks) and `engine_steps` (G7, G8).

PROTOCOL SUMMARY (Match stage: the record checker -- GM-01, the runtime match verdict of record for any B >= 1)
  Purpose: establish that THIS row's Build workload Program (GP-01: weights once, one component per declared request, declared arrivals,
    caps, stop rule + LAG) corresponds to the batched execution the Match stage recorded -- all requests, all executed steps, global I/O
    associations and chronology -- as eight named checks G1..G8, each PASS/FAIL with evidence; the verdict is PASS iff every check passes.
  Inputs: --gprog build_workload/workload_program.json (+ --workload wl.json), --oracle <match dir> (program.json + instances.jsonl =
    the fold, capture/log.jsonl.gz, capture/tokens.json), --within-step dag|sequence, [--derived dirs, --decomp-out match_decomp.json].
  Outputs: --out global_match.json {verdict, checks[{id, status, ...}], failed_checks, alternate_verdict, requests[{served_output_len,
    executed_prefix_len, eos_truncated, ...}], lag, achieved_concurrency, impl}, <out>_global_program.json; the brief on stdout.
    Exit 0 PASS, 3 FAIL, 4 X-09 names missing derived shapes (row_pod derives them and reruns once).
  Invariants: requests are named r<k> (manifest position) and the record's UUIDs are mapped through the Program's table (G1); a request
    that stopped early is compared over its executed prefix and must end on a DECLARED EOS id with the declared LAG (G6, fail closed
    without the declaration); X-09 is evidence inside G3, never the verdict; the other within-step criterion is recorded beside the verdict,
    never merged; MATCH_IMPL selects fast (the Match of record, the CLI's default) | record (the audit checker) -- an unknown name is an
    error, never a fallback; capture/tokens.json is required on a row of record.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

from verity_vllm import settings
from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import compiler_form as CF
from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.check.match import global_match_fast as GMF  # MATCH_IMPL=fast: compute-once memo of the pure per-request work; inert without the flag
from verity_vllm.check.match import instance_sequence as SEQ
from verity_vllm.check.match import program_loaders as PL
from verity_vllm.check.match.attribution import attribute_instances, requests_observed
from verity_vllm.check.match.chronology import chronology
from verity_vllm.check.match.declaration import GP_SCHEMA, from_gprog, mock_global_program, requests_declared, sha256_file
from verity_vllm.check.match.engine_steps import vu_population, within_step_order
from verity_vllm.check.match.match_run import MatchRun
from verity_vllm.check.match.per_request import check_requests
from verity_vllm.check.match.record import canonicalize_record, read_record, request_aliases
from verity_vllm.records.global_match import SCHEMA as GM_SCHEMA

COMPARATOR = "cb_a.global_match/v1"
RULES_VERSION = "r14-v1"
WITHIN_STEP_CRITERIA = ("sequence", "dag")


def match_ops(fast: bool) -> BD.Ops:
    """The Program operations check() runs and hands its X-09 run (the record's, or MATCH_IMPL=fast's); a component dir holding a fold loads as one."""
    ops = GMF.ops() if fast else BD.record_ops()
    return dataclasses.replace(ops, load_derived=lambda d, label: _load_component(ops, d, label))


def _load_component(ops: BD.Ops, d: str, label: str) -> SEQ.Prog:
    if os.path.exists(os.path.join(d, "program.json")):
        return ops.load_oracle(d, label)  # a fold record standing in for a derived component (tests)
    return ops.load_derived(d, label)


# ------------------------------------------------------------------------------------------------------ the check


DEFAULT_WITHIN_STEP = BD.DEFAULT_WITHIN_STEP   # "dag": the criterion of record from the release carrying 6a2cd7f5d


def check(gp: dict, oracle_dir: str, log_path: str | None = None, tokens_path: str | None = None, run_x09: bool = True,
          within_step: str = DEFAULT_WITHIN_STEP, record_alternate: bool = True,
          _loaded: dict | None = None, tokens_optional: bool = False) -> dict:
    """`_loaded` = {"oracle": SEQ.Prog, "components": {dir: SEQ.Prog}} decoded once by this process and shared with the X-09 run and
    the alternate-criterion run (the fold Prog is read-only after load; decoding it three times per
    invocation -- check, X-09, alternate -- put GM-01's peak RSS at 87 GB on the B16 1024/128 record). The criteria are unchanged.

    MATCH_IMPL=fast: the same body over global_match_fast's compute-once operations (`match_ops`); the outermost
    call releases their caches when it returns or raises, the nested alternate-criterion call keeps them."""
    fast = GMF.enabled()
    ops = match_ops(fast)
    if not fast:
        return _check(gp, oracle_dir, log_path, tokens_path, run_x09, within_step, record_alternate, _loaded, ops, fast=False,
                      tokens_optional=tokens_optional)
    GMF.enter()
    try:
        return _check(gp, oracle_dir, log_path, tokens_path, run_x09, within_step, record_alternate, _loaded, ops, fast=True,
                      tokens_optional=tokens_optional)
    finally:
        GMF.leave()


# two more `_loaded` keys, both the CLI's (row_pod's shared pipeline; the alternate run passes run_x09=False):
#   "x09_extra_derived": derived request-wrapper dirs X-09 may use BESIDE the global Program's components -- the shapes row_pod derives at
#                        Match time after an exit 4 (early EOS: served T_r < declared), `--derived`; the global Program itself is unchanged,
#                        G1-G8 see exactly what they saw;
#   "x09_document":      OUT -- the whole X-09 document decompose() returned (cb_a.batch_decomp's CLI schema): what row_pod's match.compare
#                        stage computed AGAIN in its own process (365 s at B8 1024/128); `--decomp-out` writes it.  Module globals would not
#                        do: under `python -m` the record path runs in `__main__` while the fast path imports this module a second time.
X09_DOCUMENT, X09_EXTRA_DERIVED = "x09_document", "x09_extra_derived"


def _check(gp: dict, oracle_dir: str, log_path: str | None, tokens_path: str | None, run_x09: bool, within_step: str,
           record_alternate: bool, _loaded: dict | None, ops: BD.Ops, fast: bool, tokens_optional: bool = False) -> dict:
    # tokens_optional rides the wrapper -> body split of mprof unit 4 unchanged
    t0 = time.time()
    _loaded = _loaded if _loaded is not None else {}
    # phase timestamps (elapsed + process CPU since the previous mark): telemetry beside the verdict, never
    # part of it; the differential harness reports them for the record and the candidate on the same artifacts
    phases: list[dict] = []
    _pt = [time.time(), time.process_time()]

    def mark(name: str) -> None:
        now, cpu = time.time(), time.process_time()
        phases.append({"name": name, "t_end_s": round(now - t0, 3), "dt_s": round(now - _pt[0], 3), "cpu_s": round(cpu - _pt[1], 3)})
        _pt[0], _pt[1] = now, cpu
    if within_step not in WITHIN_STEP_CRITERIA:
        raise ValueError(f"within_step must be one of {WITHIN_STEP_CRITERIA}, got {within_step!r}")
    log_path = log_path or os.path.join(oracle_dir, "capture", "log.jsonl.gz")
    tokens_path = tokens_path or os.path.join(oracle_dir, "capture", "tokens.json")
    tokens = json.load(open(tokens_path)) if os.path.exists(tokens_path) else None
    rec = read_record(log_path)
    mark("read_record")
    res = {"schema": GM_SCHEMA, "comparator": COMPARATOR, "rules_version": RULES_VERSION,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir, "log": log_path,
           "global_program": {"schema": gp.get("schema"), "builder": gp.get("builder"), "workload": gp.get("workload"),
                              "workload_digest": gp.get("workload_digest"), "n_requests": len(gp.get("requests") or []),
                              # whether this checker recomputed the Build artifact's digest from its descriptor (from_gprog)
                              "digest_verification": (gp.get("gprog_artifact") or {}).get("digest_verification"),
                              # component dirs the Build recorded on another host, resolved beside its artifact (relocate_components); G4
                              # cross-checks every loaded component's digest against the Build's claim, so this is traceability, not trust
                              "relocated_components": (gp.get("gprog_artifact") or {}).get("relocated_components") or {}},
           "chronology_unit": "engine_step", "request_id_convention": "r<k> (k = manifest position; record UUIDs mapped via workload_request_id)",
           "within_step_criterion": within_step,
           "within_step_note": ("sequence: X-09 instance sequence per request step (r14-v1 verdict)" if within_step == "sequence" else
                                "dag: criterion of record -- order-insensitive "
                                "Merkle DAG multiset per request step, canonical addressing by structural identity, dependency order enforced "
                                "(G8); the sequence verdict is recorded beside it under alternate_verdict"),
           "checks": [], "requests": []}
    if gp.get("schema") != GP_SCHEMA:
        res["verdict"] = "FAIL"
        res["reasons"] = [f"global program schema {gp.get('schema')!r} != {GP_SCHEMA!r}"]
        return res
    G = {r["request_id"]: r for r in gp["requests"]}

    # ---------------------------------------------------------------- the fold (meta channel of the record) + canonical request ids
    comp_dirs = [g["component"]["dir"] for g in gp["requests"] if (g.get("component") or {}).get("dir")]
    comp_dirs += [d for d in _loaded.get(X09_EXTRA_DERIVED) or () if d not in comp_dirs]   # shapes derived at Match time (--derived)
    B = _loaded.get("oracle")
    if B is None:
        B = _loaded["oracle"] = ops.load_oracle(oracle_dir, "oracle")
        CF.fold_in_form(B, comp_dirs)
    res["fold_form"] = getattr(B, "fold_form", None)
    mark("load_fold")
    alias = request_aliases(gp)
    # ---------------------------------------------------------------- X-09 evidence (whole run, as row_pod runs it)
    # Runs on the RAW fold, before canonicalize_record renames the fold rows' request ids to r<k> IN PLACE below: X-09 attributes
    # instances by the record's own ids (RunHeader.workload UUIDs) and now shares this process's decoded fold instead of decoding its own.
    if run_x09:
        try:
            x = BD.decompose(oracle_dir, comp_dirs, log_path, within_step=within_step, oracle=B,   # X-09 evidence under the same criterion
                             components=_loaded.setdefault("components", {}), ops=ops)     # (same decoded fold + components, same ops)
            _loaded[X09_DOCUMENT] = x   # the whole document (batch_decomp's CLI schema) for the CLI's --decomp-out
            res["x09"] = {"verdict": x["verdict"], "reasons": x["reasons"], "attribution": x["attribution"], "xreq_total": x.get("xreq_total"),
                          "within_step_criterion": x.get("within_step_criterion"),
                          "requests": [dict({k: e.get(k) for k in ("request_id", "LP", "T", "n_instances", "steps", "canonical_equal", "sequence_equal", "dag_multiset_equal", "criterion", "prefill_chunks")},
                                            request_id=alias.get(str(e.get("request_id")), e.get("request_id")), workload_request_id=e.get("request_id"))
                                       for e in x["requests"]], "seconds": x["seconds"],
                          "note": "X-09 batch decomposition run on the raw record (its own attribution); request ids mapped to r<k> here"}
        except Exception as e:  # noqa: BLE001 - evidence, not the verdict
            res["x09"] = {"verdict": "ERROR", "error": repr(e)[:300]}
        mark("x09")

    res["request_ids"] = canonicalize_record(rec, tokens, B.oracle_rows, alias)
    res["request_ids"]["table"] = {str(r.get("workload_request_id")): r["request_id"] for r in gp["requests"] if r.get("workload_request_id") is not None}

    run = MatchRun(gp=gp, oracle_dir=oracle_dir, tokens_path=tokens_path, within_step=within_step, fast=fast, ops=ops, loaded=_loaded,
                   tokens=tokens, tokens_optional=tokens_optional, rec=rec, res=res, fold=B, declared=G)
    facts, hdr_reqs, param_of, g_ids = requests_declared(run)
    mark("g1")
    att = attribute_instances(run, hdr_reqs)
    mark("attribution")
    present = requests_observed(run, att, g_ids)
    mark("g2")
    reqs = check_requests(run, att, param_of)
    mark("g3_g4_g5_per_request")
    chronology(run, facts, present, reqs.engine_step_of)
    mark("g6")
    by_step = vu_population(run, att, reqs)
    mark("g7")
    within_step_order(run, att, reqs, by_step)
    mark("g8")

    failed = [k["id"] for k in res["checks"] if k["status"] != "PASS"]
    res["verdict"] = "PASS" if not failed else "FAIL"
    res["failed_checks"] = failed
    res["reasons"] = [p for k in res["checks"] for p in k["problems"][:3]]
    res["eos_truncated_requests"] = [e["request_id"] for e in res["requests"] if e.get("eos_truncated")]
    # per truncated request: True only when served[-1] is a declared EOS id (the finalizer reads this)
    res["eos_verified"] = {e["request_id"]: e.get("eos_verified") for e in res["requests"] if "eos_verified" in e}
    res["seconds"] = round(time.time() - t0, 1)
    # both within-step verdicts stay in the artifact: the OTHER criterion is run on the same record (X-09 not repeated;
    # G1/G2/G6 do not depend on the criterion) and kept beside the verdict of record for traceability -- never merged into it
    if record_alternate:
        other = "sequence" if within_step == "dag" else "dag"
        try:
            alt = check(gp, oracle_dir, log_path, tokens_path, run_x09=False, within_step=other, record_alternate=False, _loaded=_loaded,
                        tokens_optional=tokens_optional)
            res["alternate_verdict"] = {"criterion": other, "verdict": alt["verdict"], "failed_checks": alt["failed_checks"],
                                        "checks": [(k["id"], k["status"]) for k in alt["checks"]], "reasons": alt["reasons"][:8],
                                        "requests": [{k: e.get(k) for k in ("request_id", "executed_steps_equal", "n_divergent_steps", "criterion")}
                                                     for e in alt["requests"]],
                                        "seconds": alt["seconds"],
                                        "note": f"the {other} criterion's verdict on the same record, recorded beside the verdict of record "
                                                f"({within_step})"}
        except Exception as e:  # noqa: BLE001 - traceability material, not the verdict
            res["alternate_verdict"] = {"criterion": other, "verdict": "ERROR", "error": repr(e)[:300]}
        mark("alternate_criterion")
    res["phases"] = phases
    return res


# ------------------------------------------------------------------------------------------------------ implementation dispatch

IMPL_ENV = "MATCH_IMPL"
IMPLS = ("record", "fast")
IMPL_DEFAULT = "fast"


def impl_module(name: str | None = None):
    """The checker implementation `MATCH_IMPL` selects: `fast` (`cb_a.global_match_fast`: the same criteria
    with compute-once memoisation, compact operand progressions and forked per-request legs -- the Match of record
    (`sweep/row_pod.sh` exports it for every row) and the CLI's default) or `record` (this module's un-memoised single-process body:
    the audit checker the differential harness compares against).  Both expose `check(gp, oracle_dir, log_path, tokens_path, run_x09,
    within_step, record_alternate)` with the same result schema; the differential harness (`verity_vllm.pipeline.match_diff`) runs
    both on the same artifacts and compares verdict, per-check outcomes, populations and mismatch locations.  An unknown name is an
    error, never a silent fallback to either implementation.

    The default moved from `record` to `fast`: a `global_match.py` invoked outside `row_pod.sh` (a by-hand rerun,
    another launcher) used to get the audit checker silently -- on the B64 llama32-1b row that is 8+ h single-core at 40-50 GB against
    ~4 min on 8 cores at 13.5 GB for the same verdict.  `MATCH_IMPL=record` still selects the audit checker explicitly."""
    name = (name if name is not None else settings.get(IMPL_ENV) or IMPL_DEFAULT).strip().lower() or IMPL_DEFAULT
    if name == "record":
        import sys
        return sys.modules[__name__]
    if name == "fast":
        from verity_vllm.check.match import global_match_fast
        return global_match_fast
    raise ValueError(f"{IMPL_ENV}={name!r}: not one of {IMPLS}")


def impl_banner(mod, gp: dict | None = None) -> str:
    """One stderr line stating, before the run, which implementation is about to run and how (workers, operand form) and which
    component dirs were relocated -- the two facts whose absence once cost an 8 h run (record checker by default,
    a Build's dangling component paths noticed only at load)."""
    name = getattr(mod, "IMPL_NAME", "record")
    env = settings.raw(IMPL_ENV)
    parts = [f"# global_match impl={name}", f"({IMPL_ENV}={env!r})" if env is not None else f"({IMPL_ENV} unset -> default {IMPL_DEFAULT})"]
    if name == "fast":
        parts.append(f"workers={GMF.workers(1 << 30)} ({GMF.WORKERS_ENV}={settings.raw(GMF.WORKERS_ENV) or 'default'})")
        parts.append(f"compact_args={settings.get(GMF.COMPACT_ENV)}")
    else:
        parts.append("single process, no memo (the audit checker; hours and tens of GB on a batched row -- MATCH_IMPL=fast is the Match of record)")
    reloc = ((gp or {}).get("gprog_artifact") or {}).get("relocated_components") or {}
    if reloc:
        parts.append(f"relocated_components={len(reloc)} (Build recorded {next(iter(reloc.values()))['from']!r} ...)")
    return " ".join(parts)


def impl_info(mod) -> dict:
    """What ran: implementation name, module, and the sha256 of its source (so two runs whose modules are byte-identical -- the
    harness's identity run -- are visibly so, and a candidate that changed is visibly not the record)."""
    f = getattr(mod, "__file__", None)
    digest = None
    if f and os.path.exists(f):
        digest = sha256_file(f)
    info = {"name": getattr(mod, "IMPL_NAME", "record"), "module": mod.__name__, "file": os.path.basename(f) if f else None,
            "source_sha256": digest, "candidate_stage": getattr(mod, "CANDIDATE_STAGE", None),
            # EVERY document -- the record checker's as much as the fast one's -- states, per fold directory
            # it loaded, the source (the record files, nothing beside them) and the STORED form read (instances.jsonl schema +
            # operand_refs_form), so a record document of a v1 store and one of a v2 store of the same capture are distinguishable by content
            "fold_load": dict(PL.LOAD_NOTES)}
    if getattr(mod, "IMPL_NAME", "record") == "fast":   # memo hit/miss counters + worker count: telemetry beside the verdict
        # after check() returned its caches are released (unit 4) -- the outermost run's counters live in last_stats
        info["stats"] = dict(GMF.last_stats) if GMF.last_stats else {k: int(v) for k, v in sorted(GMF.stats.items())}
        # the DECLARED configuration (workers cap, operand form, switches as read), beside the counters
        info["config"] = dict(GMF.config)
        # the same fold_load under the DECLARED configuration too: earlier fast documents of record named it there; readers keep working
        info["config"]["fold_load"] = dict(PL.LOAD_NOTES)
    return info


@dataclass(frozen=True, kw_only=True)
class Options:
    oracle: str = option("--oracle", help="Match dir: program.json + instances.jsonl + capture/log.jsonl.gz")
    global_program: str | None = option("--global-program", default=None, help="cb-a/global-program/v1 JSON; omit to assemble the mock from --workload + --component")
    gprog: str | None = option("--gprog", default=None, help="the Build's workload_program.json (GP-01) -> converted with from_gprog")
    workload: str | None = option("--workload", type=workload_arg, default=None, help="workload manifest (for the mock global Program)")
    component: list[str] = option("--component", action="append", default=[], help="request component dir (derive_step output); repeatable")
    log: str | None = option("--log", default=None)
    tokens: str | None = option("--tokens", default=None)
    tokens_optional: bool = option("--tokens-optional", default=False, action="store_true", help="accept a record without capture/tokens.json (G6 output-association legs skipped, " "LABELLED checks[G6].tokens_json=false; mechanism/fixture runs only, never a row of record)")
    no_x09: bool = option("--no-x09", default=False, action="store_true")
    within_step: str = option("--within-step", choices=WITHIN_STEP_CRITERIA, default=DEFAULT_WITHIN_STEP, help="within-engine-step correspondence of a request's instances: dag (default of record) | " "sequence (the r14-v1 criterion); the other criterion's verdict is recorded beside it (alternate_verdict)")
    no_alternate: bool = option("--no-alternate", default=False, action="store_true", help="do not run/record the other criterion's verdict beside the verdict of record")
    out: str | None = option("--out", default=None)
    decomp_out: str | None = option("--decomp-out", default=None, help="also write the X-09 batch decomposition document here (cb_a.batch_decomp's CLI " "schema, the same decompose() call whose verdict is this document's x09 block): row_pod's match_decomp.json " "without a second batch_decomp process over the same fold and components. Exit 4 when X-09 names " "missing derived shapes (batch_decomp's protocol: derive them and rerun), else GM-01's 0 / 3.")
    derived: list[str] = option("--derived", action="append", default=[], help="derived request-wrapper dir X-09 may use beside the global Program's " "components (repeatable): the shapes row_pod derived after an exit 4")


def main(a: Options) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    if a.decomp_out and a.no_x09:
        raise UsageError("--decomp-out needs the X-09 run (drop --no-x09)")
    loaded = {X09_EXTRA_DERIVED: list(a.derived)}   # shared with check(): the fold / components decoded once + the X-09 document
    if a.global_program:
        gp = json.load(open(a.global_program))
    elif a.gprog:
        gp = from_gprog(a.gprog, a.workload)
    else:
        if not a.workload:
            raise UsageError("--workload is required without --global-program")
        gp = mock_global_program(a.workload, a.component)
    mod = impl_module()   # MATCH_IMPL: fast (default, the Match of record) | record (the audit checker) -- never a silent fallback
    print(impl_banner(mod, gp), file=sys.stderr, flush=True)   # what is about to run, BEFORE the hours it may take (not after)
    res = mod.check(gp, a.oracle, a.log, a.tokens, run_x09=not a.no_x09, within_step=a.within_step, record_alternate=not a.no_alternate,
                    tokens_optional=a.tokens_optional, _loaded=loaded)   #
    res["impl"] = impl_info(mod)
    if (a.decomp_out or a.derived) and isinstance(res.get("x09"), dict) and loaded.get(X09_DOCUMENT) is not None:
        # a document produced under the shared pipeline (P1) states what its X-09 ran over (batch_decomp's own keys) and
        # the invocation, so two GM-01 documents of one row are distinguishable by content.  ONLY then: the record flow's document stays
        # byte for byte the checker of record's (match_diff's drift rule: check()'s document must equal the reference tree's strictly).
        x = loaded[X09_DOCUMENT]
        res["x09"].update(derived_available=x.get("derived_available"), missing_derived=x.get("missing_derived"), extra_derived=list(a.derived),
                          pipeline={"shared": bool(a.decomp_out), "decomp_out": a.decomp_out, "derived_args": list(a.derived)})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        gpp = os.path.splitext(a.out)[0] + "_global_program.json"
        json.dump(gp, open(gpp, "w"), indent=1)
    rc_x09 = 0
    if a.decomp_out:
        x = loaded.get(X09_DOCUMENT)
        if x is None or (res.get("x09") or {}).get("verdict") == "ERROR":   # X-09 did not produce a document: no file, loud
            print(json.dumps({"decomp_out": a.decomp_out, "error": "X-09 produced no document", "x09": res.get("x09")}, default=str), file=sys.stderr)
            return 3
        json.dump(x, open(a.decomp_out, "w"), indent=1, default=str)   # byte-for-byte what cb_a.batch_decomp's CLI writes for this run
        rc_x09 = 4 if x.get("verdict") == "MISSING_DERIVED" else 0   # X-09's own PASS/FAIL is read from the document (row_pod's summary), as today
    brief = {"verdict": res["verdict"], "comparator": res["comparator"], "rules_version": res["rules_version"], "within_step": res.get("within_step_criterion"),
             "impl": res["impl"], "phases": res.get("phases"),
             "alternate_verdict": {k: (res.get("alternate_verdict") or {}).get(k) for k in ("criterion", "verdict", "failed_checks")},
             "failed_checks": res.get("failed_checks"), "checks": [(k["id"], k["status"]) for k in res["checks"]], "reasons": res.get("reasons", [])[:8],
             "achieved_concurrency": res.get("achieved_concurrency"),
             "requests": [{k: e.get(k) for k in ("request_id", "workload_request_id", "LP", "n_instances", "local_steps", "executed_steps_equal", "eos_truncated",
                                                 "first_step", "output_events", "generated_length")} for e in res["requests"]], "seconds": res["seconds"]}
    print(json.dumps(brief, indent=1, default=str))
    if rc_x09 == 4:   # the shapes X-09 could not compare are not derived yet: row_pod derives them and reruns (the compare stage's exit 4)
        return 4
    return 0 if res["verdict"] == "PASS" else 3
