"""cb_a.global_match/v1 -- the GLOBAL Match of a batched execution against one rooted workload Program (R14 lane gmatch, GM-01): the
runtime match check of record.  Glossary (Draft 3 §12): a fold 'instance' is a Call, 'attribution' is occurrence resolution, and the
capture log (RunHeader / StepStart / SampleEnd / StepLink rows) is execution evidence.

X-09 (`batch_decomp`) establishes, per request, that the request's projection of the batched eager fold is canonical-equal to its
request-wrapper component Program.  That is evidence INSIDE this check, not the verdict.  The global check consumes

  * the global workload Program `G` (`cb-a/global-program/v1`: shared prescribed inputs, one component per declared request, the
    declared arrival schedule and bounded output interfaces, the address-space convention) -- lane gprog's rooted Program, or,
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

Request identity [R15 COORD ruling 2]: the Program, the workload manifest and every check name requests `r<k>` (k = manifest position).
The record (RunHeader / StepStart rows / tokens.json / fold meta) names them by the manifest's UUID -- the id vLLM served them
under; the checker maps those through the global Program's `workload_request_id` table (G1 verifies the UUID set == manifest).

Within-step order [R15 DECISION M-0021 -> COORD ruling M-0177]: `within_step="dag"` (DEFAULT of record from r15-int-2) compares
a request's step as an order-insensitive Merkle DAG multiset with structural canonical addresses; `within_step="sequence"` (the r14-v1
verdicts) compares it as X-09's instance SEQUENCE (spec + wiring by position). The other criterion's verdict is recorded beside the
verdict of record (`alternate_verdict`) and never merged into it. The dag criterion uses X-09's chunked-prefill Merkle multiset
(`batch_decomp.dag_hashes`) and addresses every fold Call by the canonical index of its structurally identical component Call, so the fold's emission order (AttentionScope: one group per
distinct T in first-occurrence order over the batch) is recorded in G8 as batch layout, never used as an address and never
re-sorted; a consumer emitted before its producer, a wiring change or a missing/extra Call still FAILS G3.

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
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""
from __future__ import annotations

import collections
import dataclasses
import gzip
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.config import UsageError, option
from verity_vllm.program.compact import n_max_producer as _n_max_producer, n_producers as _n_producers  # [R17 mfast]
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import global_match_fast as GMF  # [R17 mprof] MATCH_IMPL=fast: compute-once memo of the pure per-request work; inert without the flag
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  # [R15 stoch]

COMPARATOR = "cb_a.global_match/v1"
RULES_VERSION = "r14-v1"
GP_SCHEMA = "cb-a/global-program/v1"
WITHIN_STEP_CRITERIA = ("sequence", "dag")
_TAG_RE = re.compile(r'^\{"\$e":\s*"(\w+)"')


def request_id_of(index: int) -> str:
    """`r<k>` (COORD R15 ruling 2) -- the same function as global_program.request_id_of (kept local: this module must not import torch paths)."""
    return f"r{int(index)}"


# ------------------------------------------------------------------------------------------------------ the global Program


def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def component_shape(d: str) -> tuple[int, int] | None:
    """(LP, T) of a component dir: a derive_step dir (`construction_manifest.json`) or a fold record standing in for one."""
    return BD.derived_shape(d)


_VARIANT_RE = re.compile(r"^stoch-t(?P<t>[0-9.]+)-p(?P<p>[0-9.]+)(?:-k(?P<k>[0-9]+))?$")


def manifest_sampling(wl: dict, variant: str | None = None) -> dict:
    """The manifest's declared sampling policy as the header records it.  SOURCES, all read, none trusted alone [rev-pm M-0667 TIP N5]:
    (1) `variants[variant]` -- the table the DRIVER consumes (vllm_adapter.load_workload) and the header records: the value of record;
    (2) the variant LABEL (greedy | stoch-t<T>-p<P>[-k<K>]) parsed; (3) explicit `sampling.{temperature,top_p,top_k}`; (4) `sweep.sampling_params`.
    When two sources disagree on a key the result carries `conflict` (G1 fails closed, named): a manifest whose label says p0.95 while
    its table -- the one the run used -- says 0.9 is not one declaration.  `variant` overrides the workload-level selection (per-request pin)."""
    ms = dict(wl.get("sampling") or {})
    v = variant or ms.get("variant") or (wl.get("sweep") or {}).get("sampling")
    per_key: dict[str, dict[str, object]] = {}   # key -> {source: value}
    def put(src: str, k: str, val) -> None:
        if val is not None:
            per_key.setdefault(k, {})[src] = val
    table = (wl.get("variants") or {}).get(v) if isinstance(v, str) else None
    if isinstance(table, dict):
        for k in ("temperature", "top_p", "top_k"):
            put("variants_table", k, table.get(k))
    if v == "greedy":
        put("label", "temperature", 0.0)
    elif isinstance(v, str) and _VARIANT_RE.match(v):
        m = _VARIANT_RE.match(v)
        put("label", "temperature", float(m["t"])); put("label", "top_p", float(m["p"])); put("label", "top_k", int(m["k"] or 0))
    if variant is None:
        for src, d in (("sampling", ms), ("sweep.sampling_params", (wl.get("sweep") or {}).get("sampling_params") or {})):
            for k in ("temperature", "top_p", "top_k"):
                put(src, k, d.get(k))
    out: dict = {"variant": v} if v else {}
    conflict = []
    for k, srcs in per_key.items():
        vals = {src: float(val) for src, val in srcs.items()}
        distinct = sorted(set(round(x, 9) for x in vals.values()))
        if len(distinct) > 1:
            conflict.append(f"{k}: " + ", ".join(f"{src}={val!r}" for src, val in sorted(srcs.items())))
        # the driver's table is the value of record; else the label; else the explicit fields
        for src in ("variants_table", "label", "sampling", "sweep.sampling_params"):
            if src in srcs:
                out[k] = srcs[src]
                break
    if isinstance(v, str) and v != "greedy" and table is None and wl.get("variants") is not None:
        conflict.append(f"variant {v!r} is not in the manifest's `variants` table (the driver refuses such a manifest)")
    if conflict:
        out["conflict"] = conflict
    out["sources"] = {k: sorted(srcs) for k, srcs in per_key.items()}
    return out


def mock_global_program(workload_path: str, component_dirs: list[str], eos_token_ids: list[int] | None = None, lag: int | None = None) -> dict:
    """The rooted workload Program's INTERFACE assembled from the declared workload and the Build's request components, so the
    global check can run before lane gprog's builder lands.  It reads the manifest and the components only -- never the record.
    [R17-1] `lag` = the declared scheduler's post-EOS rows (0/1) the mock Program is specialized for; None = a pre-R17-1 Program that
    declares no lag (G6 then fails by name on any EOS stop)."""
    wl = json.load(open(workload_path))
    sw = wl.get("sweep") or {}
    arrive = sw.get("arrive_steps") or [0] * len(wl["requests"])
    comps = {}
    for d in component_dirs:
        sh = component_shape(d)
        if sh is not None:
            comps.setdefault(sh, d)
    reqs = []
    for i, r in enumerate(wl["requests"]):
        lp, cap = len(r["prompt_token_ids"]), int(r["max_tokens"])
        T = cap - 1
        reqs.append({"index": i, "request_id": request_id_of(i), "workload_request_id": r["request_id"], "prompt_len": lp, "max_tokens": cap, "T_bound": T,
                     "arrive_step": int(r.get("arrive_step") if r.get("arrive_step") is not None else (arrive[i] if i < len(arrive) else 0)),
                     "prompt_sha256": hashlib.sha256(json.dumps(r["prompt_token_ids"]).encode()).hexdigest()[:16],
                     "component": {"dir": comps.get((lp, T)), "LP": lp, "T": T},
                     "output_interface": {"unit": "engine_step", "max_events": cap, "event": "TokenSelect over the request's logits row"}})
    from verity_vllm.pipeline.global_program import MODELLED_ENGINE
    engine = dict(MODELLED_ENGINE)
    stop_rule: dict = {"eos_token_ids": sorted(set(int(x) for x in eos_token_ids)) if eos_token_ids else None}
    lag_block = None
    if lag is not None:
        if lag not in (0, 1):
            raise ValueError(f"lag must be 0 or 1, got {lag!r}")
        engine["async_scheduling"] = bool(lag)
        stop_rule.update({"eos_lag_rows": int(lag), "lag": int(lag), "eos_lag_source": "mock_global_program(lag=...) (R17-1 declared static)"})
        lag_block = {"value": int(lag), "async_scheduling": bool(lag), "source": "mock_global_program(lag=...)", "declared_explicitly": True}
    return {"schema": GP_SCHEMA, "builder": "cb_a.global_match.mock_global_program (manifest + Build components; gprog's rooted Program replaces this)",
            "workload": {"path": workload_path, "sha256": _sha256_file(workload_path), "n_requests": len(reqs)},
            # the mock models GP-01's eager conditions (M-0063 (1)); G1 fails closed on a Program without a declaration (M-0261)
            "applicability": {"execution": "bi-eager", "row_execution": "bi-eager", "engine": engine, "refusal": None},
            # [F-pm-24] the stop rule: the EOS ids an early stop must end in (None = not declared -> G6 fails closed on early stops)
            "stop_rule": stop_rule,
            "lag": lag_block,   # [R17-1] the declared LAG block (None = pre-R17-1 Program)
            "shared_inputs": [{"name": "weights", "role": "prescribed", "global_param": "the fold's Input nodes after the request-owned parameters "
                                                                                          "(one per leaf field of a mixed-width Record; one for bf16)"}],
            "requests": reqs,
            "chronology": {"unit": "engine_step (one model forward; 0 = first forward of the run)",
                           "arrival": "request r is admitted before engine step arrive_step[r]; admitted earlier only when every earlier "
                                      "request has finished (the engine would idle) -- the realized step is recorded and checked",
                           "output_event": "r's k-th output event = TokenSelect over r's logits row in the engine step that completes "
                                           "prompt+k tokens; it is r's input row in the next engine step r is scheduled (StepLink)",
                           "order": "programmed dependencies + step indices only; elapsed time is not represented"},
            "address_space": {"vu": "(request_id, local_step, local_index) <-> global fold node k (program.json node = n_input_nodes + instance)",
                              "values": "prompt_<i> parameters (one per request, manifest order) + weights parameter (field ranges)"}}


_LAG_STATIC_RE = re.compile(r",lag=(-?\d+)\}$")


def workload_lag_static(desc: dict) -> int | None:
    """[R17-1] the `lag` static of the workload Program AS THE CODEC WROTE IT, read from the descriptor (both places are inside the digest
    program_digest verifies): the codec inlines a specialization's binding record into its spec id in static-parameter order
    (`Workload_v1{requests=[...],shared=[...],lag=1}`, verity.ir.codec._spec_id), so `desc["program_fn"]` ends in `,lag=<n>}` and the
    root id `Root[Workload_v1{...,lag=<n>}]_v1` carries the same suffix; the two must agree.  A descriptor whose root definition carries an
    explicit `bindings.lag` (the mock form) is read from there.  None = a Program that declares no lag (every pre-R17-1 Build)."""
    ddefs = desc.get("definitions")
    root_id = desc.get("root")
    root_def = ddefs.get(root_id) if isinstance(ddefs, dict) and isinstance(root_id, str) else None
    root_def = root_def if isinstance(root_def, dict) else {}
    b = root_def.get("bindings")
    v_bind = int(b["lag"]) if isinstance(b, dict) and b.get("lag") is not None else None
    pf = desc.get("program_fn")
    m_pf = _LAG_STATIC_RE.search(pf) if isinstance(pf, str) and pf.startswith("Workload_v") else None
    m_root = re.search(r",lag=(-?\d+)\}\]_v\d+$", root_id) if isinstance(root_id, str) and root_id.startswith("Root[Workload_v") else None
    v_pf = int(m_pf.group(1)) if m_pf else None
    v_root = int(m_root.group(1)) if m_root else None
    if v_pf is not None and v_root is not None and v_pf != v_root:
        raise ValueError(f"descriptor program_fn lag static {v_pf} != root id lag static {v_root}: not one Program")
    v_ids = v_pf if v_pf is not None else v_root
    # [rev S-r17-03 n2] an explicit bindings.lag must agree with the spec-id suffixes when both are present
    if v_bind is not None and v_ids is not None and v_bind != v_ids:
        raise ValueError(f"descriptor root bindings.lag {v_bind} != the spec-id lag static {v_ids}: not one Program")
    return v_bind if v_bind is not None else v_ids


def relocate_components(derived: dict, artifact_path: str) -> dict:
    """Resolve GP-01's component dirs (`inputs.derived`: "lp,t" -> derive_step dir) beside the artifact when the recorded absolute
    paths do not exist here.

    A `vllm-build/v1` records the row directory of the pod that built it (`/workspace/cp/<sweep>/<row>/build_request_LP<lp>_T<t>`).
    Staged on another host, or symlinked into another attempt directory (`verity_vllm/acquire/pod_match_v2.sh`), those paths dangle
    while the same component dirs sit beside the artifact: `<dir(artifact)>/../<basename>` (the artifact lives in `<row>/build_workload/`).
    The substitution is safe because every loaded component's digest is cross-checked against the Build's claim (G4, `program_digest`
    of the component's own `descriptor.json.gz`): a relocation can only supply the very Program the Build named, never another.
    A path missing at both places is kept as recorded so the loader's FileNotFoundError names it.

    Returns {"derived": <resolved map>, "relocated": {"lp,t": {"from", "to"}}, "base": <dir tried>}."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(artifact_path)))
    out: dict = {}
    relocated: dict = {}
    for k, d in (derived or {}).items():
        if isinstance(d, str) and d and not os.path.isdir(d):
            cand = os.path.join(base, os.path.basename(d.rstrip("/")))
            if os.path.isdir(cand):
                relocated[k] = {"from": d, "to": cand}
                d = cand
        out[k] = d
    return {"derived": out, "relocated": relocated, "base": base}


def from_gprog(artifact_path: str, workload_path: str | None = None) -> dict:
    """Lane gprog's `workload_program.json` (GP-01, `verity/workload-program/v1`) -> this checker's interface.  The rooted Program's
    component dirs come from `inputs.derived` ("lp,t" -> derive_step dir); its address map and chronology are carried through and
    cross-checked (component digests, per-request instance counts, programmed steps)."""
    art = json.load(open(artifact_path))
    if not art.get("ok", True):
        raise ValueError(f"{artifact_path}: builder result not ok ({art.get('reason')}: {art.get('missing_derived')})")
    # [R15 global, rev-pm M-0112] the artifact's `workload_digest` field is a CLAIM: recompute it from the descriptor GP-01 wrote beside the
    # artifact (descriptor.json.gz) and refuse a mismatch, so a digest edited/zeroed in the Build's artifact cannot flow into this checker's
    # copy (global_match_global_program.json) and the finalizer's same-attempt rule.  Absent descriptor (an evidence dir pulled without the
    # 17-25 MB file) -> recorded as unverified, never as verified.
    digest_verification = {"verified": False, "why": "descriptor.json.gz not beside the artifact"}
    dpath = os.path.join(os.path.dirname(os.path.abspath(artifact_path)), "descriptor.json.gz")
    if os.path.exists(dpath):
        from verity.ir.codec import canonical_json, program_digest
        with gzip.open(dpath, "rb") as f:
            desc = json.loads(f.read())
        recomputed = program_digest(desc)
        if recomputed != art.get("workload_digest"):
            raise ValueError(f"{artifact_path}: workload_digest {str(art.get('workload_digest'))[:16]} != the digest of the descriptor beside it "
                             f"{recomputed[:16]} (descriptor.json.gz): the Build artifact is not the Program it names")
        dsha = hashlib.sha256(canonical_json(desc)).hexdigest()
        declared_sha = (art.get("descriptor") or {}).get("sha256")
        if declared_sha and declared_sha != dsha:
            raise ValueError(f"{artifact_path}: descriptor.sha256 {declared_sha[:16]} != the descriptor file's {dsha[:16]}")
        digest_verification = {"verified": True, "recomputed_from": dpath, "descriptor_sha256": dsha}
        # [R17-1] the artifact's `lag` block is a CLAIM beside the Program: the Program's own `lag` static (inside the digest just verified)
        # must carry the same value -- an edited block cannot make G6 consume another lag than the Program's
        static_lag = workload_lag_static(desc)
        claimed = (art.get("lag") or {}).get("value") if isinstance(art.get("lag"), dict) else None
        if static_lag is not None or claimed is not None:
            if static_lag is None or claimed is None or int(static_lag) != int(claimed):
                raise ValueError(f"{artifact_path}: lag.value {claimed!r} != the Program's Workload_v1 `lag` static {static_lag!r} (descriptor "
                                 f"program_fn {str(desc.get('program_fn'))[:40]}...{str(desc.get('program_fn'))[-24:]}, root "
                                 f"{str(desc.get('root'))[:32]}...{str(desc.get('root'))[-24:]}): the artifact's lag declaration is not the Program's (R17-1)")
            digest_verification["lag_static"] = int(static_lag)
    derived = (art.get("inputs") or {}).get("derived") or {}
    relocation = relocate_components(derived, artifact_path)
    derived = relocation["derived"]
    wl_path = workload_path or (art.get("inputs") or {}).get("manifest")
    wl = json.load(open(wl_path)) if wl_path and os.path.exists(wl_path) else None
    # prompts by manifest POSITION (the Program's r<k>); the manifest's own request_id is the workload (UUID) id
    prompts = {request_id_of(i): r["prompt_token_ids"] for i, r in enumerate((wl or {}).get("requests", []))}
    wl_uuids = {request_id_of(i): r.get("request_id") for i, r in enumerate((wl or {}).get("requests", []))}
    amap = {r["request_id"]: r for r in (art.get("address_map") or {}).get("requests", [])}
    chron = {r["request_id"]: r for r in (art.get("chronology") or {}).get("requests", [])}
    reqs = []
    for r in art["requests"]:
        lp, cap = int(r["prompt_len"]), int(r["cap"])
        rid = r["request_id"]
        if rid != request_id_of(r["index"]):
            raise ValueError(f"{artifact_path}: request {rid!r} at index {r['index']} is not the r<k> convention (COORD R15 ruling 2): "
                             "rebuild the workload Program with the R15 global_program")
        wrid = r.get("workload_request_id") or wl_uuids.get(rid)
        if wl is not None and wl_uuids.get(rid) is not None and wrid != wl_uuids[rid]:
            raise ValueError(f"{artifact_path}: {rid} carries workload_request_id {wrid!r} but the manifest at position {r['index']} has {wl_uuids[rid]!r}")
        d = derived.get(f"{lp},{cap - 1}")
        reqs.append({"index": r["index"], "request_id": rid, "workload_request_id": wrid, "prompt_len": lp, "max_tokens": cap, "T_bound": cap - 1,
                     "arrive_step": int(r["arrive_step"]),
                     "prompt_sha256": hashlib.sha256(json.dumps(prompts[rid]).encode()).hexdigest()[:16] if rid in prompts else None,
                     "component": {"dir": d, "LP": lp, "T": cap - 1, "program_digest": (r.get("component") or {}).get("program_digest"),
                                   "specialization": (r.get("component") or {}).get("specialization")},
                     "global_address": {k: amap[rid].get(k) for k in ("root_node", "gates", "n_instances", "rule")} if rid in amap else None,
                     "programmed_steps": (chron.get(rid) or {}).get("programmed_steps"),
                     "output_interface": {"unit": "engine_step", "max_events": cap, "event": "TokenSelect over the request's logits row"}})
    return {"schema": GP_SCHEMA, "builder": f"gprog GP-01 {art.get('schema')} workload_fn={art.get('workload_fn')}", "workload_digest": art.get("workload_digest"),
            "workload": {"path": wl_path, "sha256": art.get("manifest_digest"), "n_requests": len(reqs)},
            "shared_inputs": [{"name": "weights", "role": "prescribed", "global_param": f"root parameter {art.get('shared_param_index')} (gprog shared)"}],
            "requests": reqs, "chronology": art.get("chronology"), "address_space": art.get("address_map", {}).get("convention"),
            "applicability": art.get("applicability"),
            "stop_rule": art.get("stop_rule"),   # [F-pm-24] the declared EOS set (None on a pre-fix artifact -> G6 fails closed on early stops)
            "lag": art.get("lag"),               # [R17-1] the declared LAG block (None on a pre-R17-1 artifact -> G6 fails by name on EOS stops)
            "lengths": art.get("lengths"),
            "moe_construction": art.get("moe_construction"),   # [R17-42 route A (3)] GP-01's declared-and-bound construction block (None pre-R17-42)
            "gprog_artifact": {"path": artifact_path, "gates": art.get("gates"), "input_gates": art.get("input_gates"), "descriptor": art.get("descriptor"),
                               "digest_verification": digest_verification,
                               # component dirs resolved beside the artifact (recorded absolute paths absent here); {} when none moved
                               "relocated_components": relocation["relocated"]}}


# ------------------------------------------------------------------------------------------------------ the observed record


def read_record(log_path: str) -> dict:
    """RunHeader + per engine step: rows, input tokens, logits rows, sampled tokens; StepLinks."""
    header, steps, samples, links = None, {}, {}, []
    sampler_batch: dict[int, int] = {}   # [M-0169 derived S] engine step -> rows of the sampler's logits tensor (SampleEnd.logits_view shape[0])
    opener = gzip.open if log_path.endswith(".gz") else open
    with opener(log_path, "rt") as f:
        for line in f:
            m = _TAG_RE.match(line)
            tag = m.group(1) if m else None
            if tag == "StepStart":
                d = json.loads(line)
                s = int(d["step"])
                if s in steps:
                    raise ValueError(f"two StepStart records for step {s}")
                steps[s] = {"rows": [(r, int(p)) for r, p in d["rows"]], "input_token_ids": [int(x) for x in d.get("input_token_ids") or []],
                            "logits_rows": [(r, int(p)) for r, p in d["logits_rows"]]}
            elif tag == "SampleEnd":
                d = json.loads(line)
                samples[int(d["step"])] = [int(x) for x in d["sampled_token_ids"]]
                lv = (d.get("logits_view") or {}).get("$v") if isinstance(d.get("logits_view"), dict) else None
                if isinstance(lv, list) and len(lv) > 2 and isinstance(lv[2], list) and lv[2]:
                    sampler_batch[int(d["step"])] = int(lv[2][0])
            elif tag == "StepLink":
                d = json.loads(line)
                links.append({k: d.get(k) for k in ("step", "row", "request_id", "src_step", "relation", "value")})
            elif tag == "RunHeader":
                header = json.loads(line)
    if header is None:
        raise ValueError(f"{log_path}: no RunHeader")
    return {"header": header, "steps": steps, "samples": samples, "links": links, "sampler_batch": sampler_batch}


# ------------------------------------------------------------------------------------------------------ sampler geometry (derived S)

_S_STATIC_RE = re.compile(r"[{,]S=(\d+)(?=[,}])")
_TOPP_FAMILY = "GumbelTopPTokenSelect_v1"
_SPLITS_OPERAND = 5   # GumbelTopPTokenSelect_v1{V}(logits, top_p, seed, pos, temperature, splits): the derived S operand (stoch M-0254)


def top_p_split_count(spec: str, B=None, inst: dict | None = None) -> int | None:
    """The launch's top-p split count S carried by a fold TokenSelect instance.  [R16 stoch M-0254 / global M-0169 (4)] The value of
    the select's `Const32[S]` splits operand (compare.Prog._bind_splits keeps it in `B.splits_values[select node]`; the constant itself
    is bound to the Program's request-owned `splits` input entry), else the `S=<n>` static of the R15 form `GumbelTopPTokenSelect_v1{V,S}`
    (the fold's spec strings may carry a `prefix:`).  None when the instance carries no readable S."""
    if B is not None and inst is not None:
        v = getattr(B, "splits_values", {}).get(inst.get("k"))
        if v is not None:
            return int(v)
        # an unbound instance sequence (a Prog-shaped reader without compare's binding, stoch M-0268 (A)): the 6th operand must be ONE run
        # ("n", k, 0, 1) onto a `Const32[0x<S>]_v1` instance emitted for the select (fold._const, patterns._stage3 Const(32, S, "splits"))
        args = inst.get("args") or []
        if len(args) > _SPLITS_OPERAND and len(args[_SPLITS_OPERAND]) == 1:
            run = args[_SPLITS_OPERAND][0]
            if len(run) == 4 and run[0] == "n" and (run[2], run[3]) == (0, 1):
                j = (getattr(B, "k2i", None) or {}).get(run[1])
                if j is not None:
                    from verity_vllm.check.match.program_compare import const_value
                    cv = const_value(B.inst[j]["spec"])
                    if cv is not None and cv[0] == 32:
                        return int(cv[1])
    m = _S_STATIC_RE.search(spec.split(":", 1)[-1])
    return int(m.group(1)) if m else None


def scheduled_live_counts(requests: list[dict], present: dict[str, list[int]]) -> dict:
    """[R17 stoch, BRIEF §6 Stochastic / COORD R17-1] The SCHEDULED workload state per engine step, built from the per-request facts G6
    has just established -- never from the record's per-step row list.

    For every request r with rows: its sampler-active steps are the contiguous range
        E_r = [first_r, first_r + (prefill_chunks_r - 1) + a_r + lag_r)
    where first_r is its verified arrival (== the declared step, or admitted by the declared idle rule), prefill_chunks_r - 1 the chunk
    steps whose sample is discarded, a_r its answer length (the cap on a length stop; the position of the first declared EOS on a
    VERIFIED EOS stop) and lag_r the placeholder rows after an EOS stop = the attested AND declared scheduler lag (`eos_lag_rows`, R17-1:
    LAG is a declared specialization parameter; the post-EOS step is active internal computation whose row IS in the sampler batch).
    |live(t)| = #{r : t in E_r}.  A request whose G6 evidence is not established (a problem on it, an unverified early stop) leaves the
    state UNESTABLISHED at every step it touches -- the caller fails closed there instead of falling back to the observed rows.

    Returns {"live": {step: n}, "unestablished": {step: [rid..]}, "transitions": [...], "requests": {rid: {...}}}; a transition names
    the step, the live count before/after and its causes (arrivals; departures with the stop kind and the lag rows that ended)."""
    live: dict[int, int] = collections.defaultdict(int)
    unest: dict[int, list[str]] = collections.defaultdict(list)
    arrivals: dict[int, list[str]] = collections.defaultdict(list)
    departures: dict[int, list[dict]] = collections.defaultdict(list)
    per_req: dict[str, dict] = {}
    for e in requests:
        rid = e["request_id"]
        ps = present.get(rid) or []
        if not ps:
            continue
        first = int(e.get("first_step", ps[0]))
        chunks = int(e.get("prefill_chunks") or 1)
        a_r = e.get("answer_events")
        lag = int(e.get("lag_rows") or 0)
        finish = e.get("finish")
        ok = (a_r is not None and finish in ("length", "eos")
              and (finish != "eos" or e.get("eos_verified") is True)
              and not e.get("g6_problems"))
        if not ok:
            for s in ps:
                unest[s].append(rid)
            per_req[rid] = {"established": False, "observed_steps": [ps[0], ps[-1]]}
            continue
        n = (chunks - 1) + int(a_r) + lag
        end = first + n   # exclusive
        exp = list(range(first, end))
        if exp != ps:
            for s in sorted(set(exp) | set(ps)):
                unest[s].append(rid)
            per_req[rid] = {"established": False, "scheduled_steps": [first, end - 1], "observed_steps": [ps[0], ps[-1]],
                            "note": "scheduled sampler steps != observed rows"}
            continue
        for s in exp:
            live[s] += 1
        arrivals[first].append(rid)
        departures[end].append({"request": rid, "finish": finish, "lag_rows": lag, "answer_events": int(a_r), "prefill_chunks": chunks})
        per_req[rid] = {"established": True, "first_step": first, "last_step": end - 1, "prefill_chunks": chunks, "answer_events": int(a_r),
                        "lag_rows": lag, "finish": finish}
    trans: list[dict] = []
    prev = 0
    for s in sorted(set(live) | set(arrivals) | set(departures)):
        cur = live.get(s, 0)
        if cur != prev or arrivals.get(s) or departures.get(s):
            trans.append({"step": s, "live_from": prev, "live_to": cur, "arrivals": sorted(arrivals.get(s, [])),
                          "departures": sorted(departures.get(s, []), key=lambda d: d["request"])})
        prev = cur
    return {"live": dict(live), "unestablished": {s: sorted(v) for s, v in sorted(unest.items())}, "transitions": trans, "requests": per_req,
            "max_live": max(live.values()) if live else 0}


def sampler_geometry_check(gp: dict, B, engine_step_of: dict[int, int], steps: dict, sampler_batch: dict[int, int] | None = None,
                           scheduled: dict | None = None) -> dict:
    """[COORD M-0169 on M-0571 (3) / M-0579 (4)] The top-p split geometry as a WORKLOAD-level derived value, checked per engine step.

    vLLM picks the split count of `_apply_topp_split` on the HOST per sampler launch, S = splits_for(sampler batch, num_SMs)
    (`verity_vllm.program.registry.topp_split.splits_for`, the doubling loop) -- no kernel computes it, so the fold has no SplitsFor instance:
    only the launch's S on every stochastic top-p TokenSelect of that engine step.  The sampler batch at engine step t is |live(t)|,
    the requests with a logits row at t -- the declared chronology AND the verified EOS stops that G6 establishes above -- and num_SMs
    is the Program's declared target constant (`sampler_geometry.num_sms`, else `applicability.components.num_sms`).  Rule per engine
    step with at least one top-p TokenSelect: every such instance carries S == SplitsFor_v1(|live(t)|, num_SMs); the SampleEnd's own
    logits_view batch (when recorded) must equal |live(t)|.  A top-p fold against a Program that declares no num_SMs FAILS closed; an
    instance without a readable S FAILS by name.  Greedy folds (no top-p TokenSelect) have nothing to check (`checked` = 0).

    [R17 stoch] With `scheduled` (scheduled_live_counts: the state G6 derived from arrivals, served lengths, verified EOS stops and the
    declared lag), |live(t)| is the SCHEDULED count: the record's logits rows and the SampleEnd batch must both equal it (a sampler row
    the workload state does not schedule -- an undeclared request, a row past a stop, a missing lag row -- is named, and S is never
    allowed to follow the inflated batch), a top-p select on a step the state schedules nobody at is named, a scheduled sampling step
    without a top-p select is named, and a step whose state is unestablished fails closed.  The live->S transitions are reported with
    their causes (`transitions`).  Without `scheduled` the check reads the observed logits rows as before (R16 form)."""
    from verity_vllm.program.registry.topp_split import splits_for
    from verity_vllm.program.sampling_event import family_of
    per_step: dict[int | None, set[int | None]] = collections.defaultdict(set)
    n_topp = 0
    for i, r in enumerate(B.inst):
        sp = r["spec"]
        if family_of(sp) != _TOPP_FAMILY:
            continue
        n_topp += 1
        per_step[engine_step_of.get(i)].add(top_p_split_count(sp, B, r))
    sched_live = (scheduled or {}).get("live") if isinstance(scheduled, dict) else None
    info: dict = {"relation": "S_t = SplitsFor_v1(|live(t)|, num_SMs): registry/topp_split.splits_for (vLLM _apply_topp_split doubling loop), "
                              "|live(t)| = requests with a logits row at engine step t (declared chronology and verified EOS), num_SMs declared",
                  "top_p_token_selects": n_topp, "checked_steps": 0,
                  "live_source": ("scheduled workload state (G6: declared arrivals, served lengths = cap or verified EOS, declared lag rows, "
                                  "prefill chunks) -- the record's logits rows and SampleEnd batch must equal it" if sched_live is not None
                                  else "observed logits rows of the record (R16 form; no scheduled state passed)")}
    problems: list[str] = []
    if n_topp == 0:
        info["note"] = "no top-p TokenSelect in the fold: nothing to check"
        if scheduled:
            info["transitions"] = [dict(t) for t in scheduled.get("transitions", [])]
        return {"info": info, "problems": problems}
    sg = gp.get("sampler_geometry") if isinstance(gp.get("sampler_geometry"), dict) else {}
    nsm = sg.get("num_sms")
    if nsm is None:
        nsm = ((gp.get("applicability") or {}).get("components") or {}).get("num_sms")
    info["num_sms"] = nsm
    info["declared_by"] = "sampler_geometry" if sg.get("num_sms") is not None else ("applicability.components.num_sms" if nsm is not None else None)
    if nsm is None:
        problems.append(f"the fold has {n_topp} top-p TokenSelect(s) (S = {sorted(x for v in per_step.values() for x in v if x is not None)[:6]}) "
                        f"but the Program declares no num_SMs (sampler_geometry.num_sms / applicability.components.num_sms): "
                        f"S_t = SplitsFor_v1(|live(t)|, num_SMs) cannot be evaluated, fail closed (M-0571 (3))")
        return {"info": info, "problems": problems}
    nsm = int(nsm)
    mism: list[dict] = []
    hist: collections.Counter = collections.Counter()
    batch_dis: list[dict] = []
    sched_dis: list[dict] = []
    unest_steps: list[dict] = []
    unscheduled: list[dict] = []
    sched_unest = (scheduled or {}).get("unestablished") or {} if sched_live is not None else {}
    for s in sorted(per_step, key=lambda x: (x is None, x if x is not None else -1)):
        Ss = per_step[s]
        if s is None or s not in steps:
            problems.append(f"{len(Ss)} top-p TokenSelect S value(s) {sorted(x for x in Ss if x is not None)} on an engine step the record does not "
                            f"list (fold meta step {s!r}): the sampler batch is unknown, fail closed")
            continue
        live_obs = len(steps[s]["logits_rows"])
        sb = (sampler_batch or {}).get(s)
        if sched_live is not None:
            if sched_unest.get(s) or sched_unest.get(str(s)):
                unest_steps.append({"step": s, "requests": sched_unest.get(s) or sched_unest.get(str(s))})
                continue
            live = int(sched_live.get(s, sched_live.get(str(s), 0)))
            if live == 0:
                unscheduled.append({"step": s, "logits_rows": live_obs, "fold_S": sorted(x for x in Ss if x is not None)})
                continue
            if live != live_obs:
                sched_dis.append({"step": s, "scheduled_live": live, "logits_rows": live_obs})
            if sb is not None and int(sb) != live:
                batch_dis.append({"step": s, "scheduled_live": live, "logits_rows": live_obs, "sampler_batch": int(sb)})
        else:
            live = live_obs
            if sb is not None and int(sb) != live:
                batch_dis.append({"step": s, "logits_rows": live, "sampler_batch": int(sb)})
        exp = int(splits_for(live, nsm))
        hist[f"live={live}:S={exp}"] += 1
        if None in Ss:
            problems.append(f"engine step {s}: a top-p TokenSelect carries no readable split count S (neither a Const32[S] splits operand -- the 6th, "
                            f"bound to the Program's splits input -- nor an S static of the R15 form)")
        bad = sorted(x for x in Ss if x is not None and x != exp)
        if bad:
            mism.append({"step": s, "live": live, "expected_S": exp, "fold_S": bad})
        info["checked_steps"] += 1
    if sched_live is not None:
        # every step the workload state schedules a sampler batch at must carry a top-p select of the fold (a stochastic row samples at
        # every engine step; a scheduled step without one is a missing launch, not a greedy step)
        missing = sorted(int(s) for s, n in sched_live.items() if int(n) > 0 and int(s) not in per_step and int(s) in steps)
        if missing:
            problems.append(f"{len(missing)} engine step(s) the scheduled workload state samples at (live > 0) carry no top-p TokenSelect in the fold: "
                            f"{missing[:8]}{'...' if len(missing) > 8 else ''} -- a sampler launch the record lacks, fail closed")
        if unest_steps:
            problems.append(f"{len(unest_steps)} engine step(s) whose scheduled workload state is NOT established (a request's arrival / served "
                            f"length / EOS / lag evidence failed in G6): {unest_steps[:6]} -- |live(t)| cannot be derived, the record's own row "
                            f"count is not a substitute; fail closed")
        if unscheduled:
            problems.append(f"{len(unscheduled)} engine step(s) with a top-p TokenSelect where the scheduled workload state has NO live request: "
                            f"{unscheduled[:6]} -- a sampler launch outside the declared workload (undeclared request / rows past a stop)")
        if sched_dis:
            problems.append(f"{len(sched_dis)} engine step(s) whose logits rows != the scheduled live count {sched_dis[:6]}"
                            f"{'...' if len(sched_dis) > 6 else ''} -- the sampler batch is not the scheduled workload state (declared arrivals, "
                            f"served lengths to cap or verified EOS, declared lag rows); S must follow the scheduled count, never the observed batch")
        trans = []
        for t in (scheduled or {}).get("transitions", []):
            tt = dict(t)
            tt["S_from"] = int(splits_for(t["live_from"], nsm)) if t.get("live_from") else None
            tt["S_to"] = int(splits_for(t["live_to"], nsm)) if t.get("live_to") else None
            trans.append(tt)
        info["transitions"] = trans
        info["S_transitions"] = [{"step": t["step"], "live": t["live_to"], "S": t["S_to"]} for t in trans
                                 if t.get("S_to") is not None and t.get("S_from") != t.get("S_to")]
        info["scheduled_max_live"] = (scheduled or {}).get("max_live")
    if batch_dis:
        problems.append(f"{len(batch_dis)} engine step(s) whose SampleEnd logits_view batch != the "
                        f"{'scheduled live count' if sched_live is not None else 'record' + chr(39) + 's logits rows'}: {batch_dis[:6]} "
                        f"-- the sampler launch saw another batch than the chronology lists")
    if mism:
        problems.append(f"{len(mism)} engine step(s) whose top-p split count S is not SplitsFor_v1(|live(t)|, num_SMs={nsm}): {mism[:6]}"
                        f"{'...' if len(mism) > 6 else ''} -- the launch geometry disagrees with the derived value (M-0571 (3))")
    info.update({"steps_with_mismatch": len(mism), "live_to_S": dict(sorted(hist.items())),
                 "sampler_batch_recorded_steps": sum(1 for s in per_step if s is not None and (sampler_batch or {}).get(s) is not None)})
    return {"info": info, "problems": problems}


def request_aliases(gp: dict) -> dict[str, str]:
    """record request id (the manifest UUID vLLM served under, or an already-canonical id) -> the Program's `r<k>`."""
    alias: dict[str, str] = {}
    for r in gp["requests"]:
        rid = str(r["request_id"])
        alias[rid] = rid
        w = r.get("workload_request_id")
        if w is not None:
            alias[str(w)] = rid
    return alias


def canonicalize_record(rec: dict, tokens: dict | None, oracle_rows: list[dict] | None, alias: dict[str, str]) -> dict:
    """Rename every request reference of the record (RunHeader.workload, StepStart rows / logits rows, StepLinks, tokens.json, fold
    instance meta) to the canonical `r<k>` through `alias`, IN PLACE. Ids the table does not know are left as they are and returned
    (G1/G2 then report them as undeclared). The original id is kept as `workload_request_id` on header/tokens entries."""
    unknown: collections.Counter = collections.Counter()

    def m(x):
        if x is None:
            return None
        y = alias.get(str(x))
        if y is None:
            unknown[str(x)] += 1
            return str(x)
        return y

    for st in rec["steps"].values():
        st["rows"] = [(m(r), p) for r, p in st["rows"]]
        st["logits_rows"] = [(m(r), p) for r, p in st["logits_rows"]]
    for lk in rec["links"]:
        lk["request_id"] = m(lk.get("request_id"))
    wl = rec["header"].get("workload") or {}
    for r in wl.get("requests") or []:
        r["workload_request_id"] = r.get("request_id")
        r["request_id"] = m(r.get("request_id"))
    for c in wl.get("children") or []:
        c["request_id"], c["parent"] = m(c.get("request_id")), m(c.get("parent"))
    for r in (tokens or {}).get("requests", []) if tokens else []:
        r["workload_request_id"] = r.get("request_id")
        r["request_id"] = m(r.get("request_id"))
    for o in oracle_rows or []:
        if o.get("request") is not None:
            o["request"] = m(o["request"])
    return {"renamed": len(alias), "unknown_ids": dict(unknown)}


def match_ops(fast: bool) -> BD.Ops:
    """The Program operations check() runs and hands its X-09 run (the record's, or MATCH_IMPL=fast's); a component dir holding a fold loads as one."""
    ops = GMF.ops() if fast else BD.record_ops()
    return dataclasses.replace(ops, load_derived=lambda d, label: _load_component(ops, d, label))


def _load_component(ops: BD.Ops, d: str, label: str) -> C.Prog:
    if os.path.exists(os.path.join(d, "program.json")):
        return ops.load_oracle(d, label)  # a fold record standing in for a derived component (tests)
    return ops.load_derived(d, label)


class _Check:
    def __init__(self, res: dict, name: str, what: str):
        self.res, self.name, self.what = res, name, what
        self.problems: list[str] = []
        self.info: dict = {}

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def done(self) -> bool:
        self.res["checks"].append({"id": self.name, "what": self.what, "status": "FAIL" if self.problems else "PASS",
                                   "problems": self.problems[:20], "n_problems": len(self.problems), **self.info})
        return not self.problems


# ------------------------------------------------------------------------------------------ the record's engine facts (G1)

ENGINE_FACT_KEYS = ("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill",
                    "async_scheduling")   # [M-0213 / ruling 21] G6 reads the scheduler's EOS lag rows from THIS attested fact; no default


def eos_lag_rows(facts: dict) -> tuple[int | None, str]:
    """[COORD M-0213 / ruling 21 / DP-pad-02] how many rows the engine computes for a request AFTER its EOS sample.

    vLLM's async scheduling (``async_scheduling=True``) schedules step t+1 before step t's sample is known, so a request whose
    sample at step t is an EOS is still forwarded once more at t+1 (a placeholder row whose input is the EOS and whose sample is
    discarded); with ``async_scheduling=False`` there is no such row. The count is a derived engine fact read from the record's
    OWN attestation (RunHeader.profile.engine_kwargs / RunHeader.engine_kwargs / capture/versions.json, all locations agreeing)
    -- never guessed: a lag-0 record must carry the flag too, else it is indistinguishable from a lag-1 record with a deleted
    step. Returns (rows, source) with rows None when the record does not attest the flag (GM-01 refuses by name)."""
    vals = (facts.get("values") or {}).get("async_scheduling")
    if not vals:
        return None, "not attested"
    v = vals[0]
    if isinstance(v, bool):
        return (1 if v else 0), f"async_scheduling={v!r} ({', '.join((facts.get('sources') or {}).get('async_scheduling', []))})"
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return (1 if v.strip().lower() == "true" else 0), f"async_scheduling={v!r} ({', '.join((facts.get('sources') or {}).get('async_scheduling', []))})"
    return None, f"async_scheduling={v!r} is not a boolean"


def record_engine_facts(header: dict, oracle_dir: str | None = None) -> dict:
    """The execution conditions the record attests, gathered from EVERY location a recorder writes them [rev-pm M-0261 / F-pm-21]:
    the production recorder nests the effective LLM(**kw) under `RunHeader.profile.engine_kwargs` and the compilation facts under
    `RunHeader.profile.compilation` (vllm_adapter build_manifest); older/foreign headers put them at the top level; the run's
    `capture/versions.json` repeats the effective engine kwargs.  Returns {engine: {k: v}, compilation: {fact: v}, values: {k: [distinct v]},
    sources: {k: [locations]}, disagreements: {k: (a, b, src_a, src_b)}} -- the caller FAILS on any disagreement and when nothing is attested."""
    from verity_vllm.pipeline.global_program import compilation_facts
    header = header if isinstance(header, dict) else {}
    prof = header.get("profile") if isinstance(header.get("profile"), dict) else {}
    ek_top = header.get("engine_kwargs") if isinstance(header.get("engine_kwargs"), dict) else {}
    ek_prof = prof.get("engine_kwargs") if isinstance(prof.get("engine_kwargs"), dict) else {}
    ek_ver: dict = {}
    if oracle_dir:
        vp = os.path.join(oracle_dir, "capture", "versions.json")
        if os.path.exists(vp):
            try:
                v = json.load(open(vp))
                ek_ver = v.get("engine_kwargs") if isinstance(v.get("engine_kwargs"), dict) else {}
            except Exception:  # noqa: BLE001 -- an unreadable versions.json is not an attestation
                ek_ver = {}
    engine_sources = {"RunHeader.profile.engine_kwargs": ek_prof, "RunHeader.engine_kwargs": ek_top, "capture/versions.json engine_kwargs": ek_ver}
    comp_sources = {"RunHeader.profile.compilation": prof.get("compilation"), "RunHeader.compilation": header.get("compilation"),
                    "RunHeader.profile.engine_kwargs.compilation_config": ek_prof.get("compilation_config"),
                    "RunHeader.engine_kwargs.compilation_config": ek_top.get("compilation_config"),
                    "RunHeader.engine_kwargs.compilation": ek_top.get("compilation"),
                    "capture/versions.json engine_kwargs.compilation_config": ek_ver.get("compilation_config")}
    out = {"engine": {}, "compilation": {}, "values": {}, "sources": {}, "disagreements": {}}

    def _j(v):
        return json.dumps(v, sort_keys=True, default=str)

    def take(k, v, src):
        bucket = out["engine"] if k in ENGINE_FACT_KEYS else out["compilation"]
        if k in bucket:
            if _j(bucket[k]) != _j(v):
                out["disagreements"].setdefault(k, (bucket[k], v, out["sources"][k][0], src))
        else:
            bucket[k] = v
        vals = out["values"].setdefault(k, [])
        if all(_j(x) != _j(v) for x in vals):
            vals.append(v)
        out["sources"].setdefault(k, []).append(src)

    for src, ek in engine_sources.items():
        for k in ENGINE_FACT_KEYS:
            if k in ek and ek[k] is not None:
                take(k, ek[k], src)
    for src, blk in comp_sources.items():
        for k, v in compilation_facts(blk).items():
            take(k, v, src)
        # the recorder's compilation block also states enforce_eager (profile.compilation.enforce_eager) -- an engine fact
        if isinstance(blk, dict) and isinstance(blk.get("enforce_eager"), bool):
            take("enforce_eager", blk["enforce_eager"], src + ".enforce_eager")
    return out


# ------------------------------------------------------------------------------------------------------ the check


DEFAULT_WITHIN_STEP = BD.DEFAULT_WITHIN_STEP   # [COORD ruling M-0177, 20:49Z] "dag": the criterion of record from the release carrying 6a2cd7f5d


def _perm_match(A: C.Prog, P: C.Prog, s: int, ha: dict, hb: dict) -> list:
    """G8's Merkle (order-insensitive) matching of one request step: fold instance -> component index with the same DAG hash, chosen
    in emission order inside a hash class; None where no component instance is left with that hash.  `ha` / `hb` must be
    dag_hashes(A) / dag_hashes(P) of ONE process (the fast path's ids are per process, F-r17-07)."""
    ma = collections.defaultdict(list)
    for j, i in enumerate(A.steps[s]):
        ma[ha[i]].append(j)
    pi = []
    for i in P.steps[s]:
        js = ma.get(hb[i])
        pi.append(js.pop(0) if js else None)
    return pi


def _perm_eligible(A: C.Prog, P: C.Prog, s) -> bool:
    return not (s is None or s >= len(P.steps) or s >= len(A.steps) or len(A.steps[s]) != len(P.steps[s]))


def check(gp: dict, oracle_dir: str, log_path: str | None = None, tokens_path: str | None = None, run_x09: bool = True,
          within_step: str = DEFAULT_WITHIN_STEP, record_alternate: bool = True,
          _loaded: dict | None = None, tokens_optional: bool = False) -> dict:
    """`_loaded` = {"oracle": C.Prog, "components": {dir: C.Prog}} decoded once by this process and shared with the X-09 run and
    the alternate-criterion run ([R15 global, GM-01 memory]: the fold Prog is read-only after load; decoding it three times per
    invocation -- check, X-09, alternate -- put GM-01's peak RSS at 87 GB on the B16 1024/128 record). The criteria are unchanged.

    [R17 mprof] MATCH_IMPL=fast (BRIEF s5): the same body over global_match_fast's compute-once operations (`match_ops`); the outermost
    call releases their caches when it returns or raises (rev F-r17-06), the nested alternate-criterion call keeps them."""
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


# [COORD R17-28, mprof P1] two more `_loaded` keys, both the CLI's (row_pod's shared pipeline; the alternate run passes run_x09=False):
#   "x09_extra_derived": derived request-wrapper dirs X-09 may use BESIDE the global Program's components -- the shapes row_pod derives at
#                        Match time after an exit 4 (early EOS: served T_r < declared), `--derived`; the global Program itself is unchanged,
#                        G1-G8 see exactly what they saw;
#   "x09_document":      OUT -- the whole X-09 document decompose() returned (cb_a.batch_decomp's CLI schema): what row_pod's match.compare
#                        stage computed AGAIN in its own process (365 s at B8 1024/128); `--decomp-out` writes it.  Module globals would not
#                        do: under `python -m` the record path runs in `__main__` while the fast path imports this module a second time.
X09_DOCUMENT, X09_EXTRA_DERIVED = "x09_document", "x09_extra_derived"


def _check(gp: dict, oracle_dir: str, log_path: str | None, tokens_path: str | None, run_x09: bool, within_step: str,
           record_alternate: bool, _loaded: dict | None, ops: BD.Ops, fast: bool, tokens_optional: bool = False) -> dict:
    # [rel r17-int-2] tokens_optional (stoch F-r17-01) rides the wrapper -> body split of mprof unit 4 unchanged
    t0 = time.time()
    _loaded = _loaded if _loaded is not None else {}
    # [R17 mfast, BRIEF §5 step 1] phase timestamps (elapsed + process CPU since the previous mark): telemetry beside the verdict, never
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
    header, steps, samples = rec["header"], rec["steps"], rec["samples"]
    res = {"schema": "cb-a/global-match/v1", "comparator": COMPARATOR, "rules_version": RULES_VERSION,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir, "log": log_path,
           "global_program": {"schema": gp.get("schema"), "builder": gp.get("builder"), "workload": gp.get("workload"),
                              "workload_digest": gp.get("workload_digest"), "n_requests": len(gp.get("requests") or []),
                              # [rev-pm M-0112] whether this checker recomputed the Build artifact's digest from its descriptor (from_gprog)
                              "digest_verification": (gp.get("gprog_artifact") or {}).get("digest_verification"),
                              # component dirs the Build recorded on another host, resolved beside its artifact (relocate_components); G4
                              # cross-checks every loaded component's digest against the Build's claim, so this is traceability, not trust
                              "relocated_components": (gp.get("gprog_artifact") or {}).get("relocated_components") or {}},
           "chronology_unit": "engine_step", "request_id_convention": "r<k> (k = manifest position; record UUIDs mapped via workload_request_id)",
           "within_step_criterion": within_step,
           "within_step_note": ("sequence: X-09 instance sequence per request step (r14-v1 verdict)" if within_step == "sequence" else
                                "dag: criterion of record (COORD ruling M-0177 on DECISION M-0021/M-0031; approved by Daniel 21:33Z, M-0281) -- order-insensitive "
                                "Merkle DAG multiset per request step, canonical addressing by structural identity, dependency order enforced "
                                "(G8); the sequence verdict is recorded beside it under alternate_verdict"),
           "checks": [], "requests": []}
    if gp.get("schema") != GP_SCHEMA:
        res["verdict"] = "FAIL"
        res["reasons"] = [f"global program schema {gp.get('schema')!r} != {GP_SCHEMA!r}"]
        return res
    G = {r["request_id"]: r for r in gp["requests"]}

    # ---------------------------------------------------------------- the fold (meta channel of the record) + canonical request ids
    B = _loaded.get("oracle")
    if B is None:
        B = _loaded["oracle"] = ops.load_oracle(oracle_dir, "oracle")
    mark("load_fold")
    alias = request_aliases(gp)
    # ---------------------------------------------------------------- X-09 evidence (whole run, as row_pod runs it)
    # Runs on the RAW fold, before canonicalize_record renames the fold rows' request ids to r<k> IN PLACE below: X-09 attributes
    # instances by the record's own ids (RunHeader.workload UUIDs) and now shares this process's decoded fold instead of decoding its own.
    comp_dirs = [g["component"]["dir"] for g in gp["requests"] if (g.get("component") or {}).get("dir")]
    comp_dirs += [d for d in _loaded.get(X09_EXTRA_DERIVED) or () if d not in comp_dirs]   # [R17-28 P1] shapes derived at Match time (--derived)
    if run_x09:
        try:
            x = BD.decompose(oracle_dir, comp_dirs, log_path, within_step=within_step, oracle=B,   # X-09 evidence under the same criterion
                             components=_loaded.setdefault("components", {}), ops=ops)     # (same decoded fold + components, same ops)
            _loaded[X09_DOCUMENT] = x   # [R17-28 P1] the whole document (batch_decomp's CLI schema) for the CLI's --decomp-out
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

    # ---------------------------------------------------------------- G1: the declared request set, three ways
    c = _Check(res, "G1", "requests_declared: manifest == global Program == record header; none omitted/duplicated; prompts as declared; "
                          "record engine conditions inside the Program's declared applicability")
    # [COORD M-0063 (1)(3)] the Program declares the execution profile + engine conditions it models (GP-01 applicability); the record's
    # RunHeader carries the EFFECTIVE engine kwargs -- a run outside the declaration is not a Match for this Program (named)
    app = gp.get("applicability") or {}
    # [rev-pm BLOCKER M-0261 / F-pm-21, COORD M-0285 (c)] the recorder (vllm_adapter build_manifest -> RunHeader.profile) nests the
    # effective engine kwargs and the compilation facts under RunHeader.profile.*; read EVERY location the record may attest them
    # (profile.*, top level, capture/versions.json), require the locations to agree, and FAIL CLOSED when the record attests nothing
    facts = record_engine_facts(header, oracle_dir)
    eff, hc = facts["engine"], facts["compilation"]
    res["execution"] = {"declared_execution": app.get("execution"), "row_execution": app.get("row_execution"), "declared_engine": app.get("engine"),
                        "effective_engine": eff or None, "effective_compilation": hc or None,
                        "source": facts["sources"] or "not recorded (RunHeader.profile.engine_kwargs / RunHeader.engine_kwargs / capture/versions.json)"}
    # [R17-42 route A (3)] the fused-MoE CONSTRUCTION the Program is of (GP-01 `moe_construction` block: indexed-read = MOE-01 / padded =
    # MoeFfnFixedL_v1 per token) is a Program static -- declared-and-bound, NEVER attested: the record is construction-agnostic (the same
    # trace serves both Programs), so nothing of the record is compared with it here; it is REPORTED so the verdict names which
    # representation of the workload it is a Match of (PADDED-TABLE labels the row with its actual population, R17-45 (1)).  A Program that
    # binds a fused-MoE block without the block (pre-R17-42 Build) is labelled indexed-read by its Definitions, never assumed padded.
    _mc = gp.get("moe_construction") if isinstance(gp.get("moe_construction"), dict) else None
    res["moe_construction"] = ({"value": _mc.get("value"), "moe_block_present": _mc.get("moe_block_present"), "population_label": _mc.get("population_label"),
                                "declared_explicitly": _mc.get("declared_explicitly"), "attested": False,
                                "source": "GP-01 moe_construction block (TargetProfile.moe_construction, declared-and-bound; never attested by the record)"}
                               if _mc else {"value": None, "source": "the Program carries no moe_construction block (pre-R17-42 Build): its Definitions "
                                                                     "decide (MOE-01 ids = indexed-read); nothing declared, nothing assumed"})
    for k, (a, b, sa, sb) in facts["disagreements"].items():
        c.fail(f"the record's own sources disagree on {k}: {a!r} ({sa}) vs {b!r} ({sb}) -- the record does not attest one execution")
    if not app.get("engine"):
        c.fail("the workload Program declares no applicability.engine (pre-M-0063 artifact): the record's engine conditions cannot be "
               "checked against it -- rebuild the Program on a release carrying GP-01 applicability")
    else:
        if not eff:
            c.fail("the record does not attest its engine conditions (no enforce_eager / enable_prefix_caching at RunHeader.profile.engine_kwargs, "
                   "RunHeader.engine_kwargs or capture/versions.json engine_kwargs): applicability cannot be checked, fail closed (M-0261)")
        # [M-0213 / ruling 21] async_scheduling joins the compared keys only when the Program declares it (one source, two readers)
        for k in ("enforce_eager", "enable_prefix_caching") + (("async_scheduling",) if app["engine"].get("async_scheduling") is not None else ()):
            # [rev-pm M-0464] EACH declared key must be attested -- a record that omits one knob (prefix caching: vLLM V1 default ON)
            # does not get applicability by absence
            if eff and k not in facts["values"]:
                c.fail(f"the record does not attest {k}, which the Program's applicability declares ({k}={app['engine'].get(k)!r}): "
                       f"fail closed (M-0464)")
            for v in facts["values"].get(k, ()):
                if v != app["engine"].get(k):
                    c.fail(f"record engine {k}={v!r} is outside the Program's declared applicability ({k}={app['engine'].get(k)!r})")
    # [compiled M-0153 (4a)] a Program modelling a COMPILED execution declares its compilation facts; the record must attest the same
    # (RunHeader.profile.compilation / .compilation / engine_kwargs.compilation_config); a compiled Program against a header without them
    # FAILS (fail closed).  An EAGER Program against a record whose compilation facts attest cudagraphs / torch.compile FAILS too.
    from verity_vllm.pipeline.global_program import compilation_facts
    decl_comp = (app.get("engine") or {}).get("compilation") if app.get("engine") else None
    if decl_comp is not None:
        decl = compilation_facts(decl_comp)
        res["execution"]["declared_compilation"] = decl
        if not hc:
            c.fail("the Program models a compiled execution but the record carries no compilation facts (RunHeader.profile.compilation / "
                   "RunHeader.compilation / engine_kwargs.compilation_config) to compare with the declared ones")
        else:
            for k in ("mode", "cudagraph_mode", "capture_sizes", "backend"):
                if k in decl and k in hc and json.dumps(hc[k], sort_keys=True, default=str) != json.dumps(decl[k], sort_keys=True, default=str):
                    c.fail(f"record compilation {k}={hc[k]!r} is outside the Program's declared applicability ({k}={decl[k]!r})")
            if "cudagraph_mode" in decl and "cudagraph_mode" not in hc:
                c.fail("the record's compilation facts do not attest cudagraph_mode which the Program's applicability declares")
    elif app.get("engine") and hc:
        cg = hc.get("cudagraph_mode")
        if cg is not None and str(cg).upper() not in ("NONE", "0"):
            c.fail(f"record compilation cudagraph_mode={cg!r}: the record attests a compiled execution which the eager Program does not model")
        md = hc.get("mode")
        if md is not None and str(md).upper() not in ("0", "NONE", "NO_COMPILATION", "EAGER"):
            c.fail(f"record compilation mode={md!r}: the record attests a compiled execution which the eager Program does not model")
    if app.get("refusal"):
        c.fail(f"the workload Program refused this row's applicability: {app['refusal']}")
    g_ids = [r["request_id"] for r in gp["requests"]]
    dup_g = [k for k, n in collections.Counter(g_ids).items() if n > 1]
    if dup_g:
        c.fail(f"global Program lists requests more than once: {dup_g}")
    bad_conv = [r["request_id"] for r in gp["requests"] if r["request_id"] != request_id_of(r["index"])]
    if bad_conv:
        c.fail(f"global Program request ids are not r<k> at their manifest index (COORD R15 ruling 2): {bad_conv[:6]}")
    hdr_reqs, param_of, parent_of = BD.request_params(header)
    h_ids = [r["request_id"] for r in hdr_reqs]
    dup_h = [k for k, n in collections.Counter(h_ids).items() if n > 1]
    if dup_h:
        c.fail(f"record header lists requests more than once: {dup_h}")
    wl_ids = None
    wl_path = (gp.get("workload") or {}).get("path")
    wl = None
    if wl_path and os.path.exists(wl_path):
        wl = json.load(open(wl_path))
        # the manifest's positions are the Program's r<k>; its request_id column is the workload (UUID) id the record uses
        wl_ids = [request_id_of(i) for i, _ in enumerate(wl["requests"])]
        if sorted(wl_ids) != sorted(g_ids):
            c.fail(f"manifest requests {sorted(wl_ids)} != global Program requests {sorted(g_ids)}")
        sw_arr = (wl.get("sweep") or {}).get("arrive_steps") or []
        for i, w in enumerate(wl["requests"]):
            g = G.get(request_id_of(i))
            if g is None:
                continue
            if g.get("workload_request_id") is not None and str(w.get("request_id")) != str(g["workload_request_id"]):
                c.fail(f"{request_id_of(i)}: manifest workload request id {w.get('request_id')!r} != the Program's {g['workload_request_id']!r}")
            # [rev-pm M-0302 (a)] a manifest whose per-request fields were edited after the Build is named per request, not only by digest
            w_lp = w.get("prompt_len") if w.get("prompt_len") is not None else len(w.get("prompt_token_ids") or ())
            w_arr = w.get("arrive_step") if w.get("arrive_step") is not None else (sw_arr[i] if i < len(sw_arr) else 0)
            for name, wv, gv in (("prompt_len", w_lp, g.get("prompt_len")), ("max_tokens", w.get("max_tokens"), g.get("max_tokens")),
                                 ("arrive_step", w_arr, g.get("arrive_step"))):
                if wv is not None and gv is not None and int(wv) != int(gv):
                    c.fail(f"{request_id_of(i)}: manifest {name}={int(wv)} != the Program's declared {name}={int(gv)} (manifest edited after the Build)")
        want = (gp.get("workload") or {}).get("sha256")
        have = _sha256_file(wl_path)
        c.info["workload_sha256"] = have
        if want and want != have:
            c.fail("workload manifest sha256 differs from the global Program's")
        # [rev-pm M-0570 (2)] the manifest of record must be the manifest the RUN was captured with: the driver records its sha256 in
        # RunHeader.workload.sha256 -- a manifest edited after the run (seeds swapped, a seed deleted) composes the same Program and
        # passed G1 by ids/prompts/caps/arrivals alone; and per request the header's sampling fields are compared with the manifest's
        hw = (header.get("workload") or {}).get("sha256")
        # the driver (vllm_adapter.load_workload) hashes the manifest DOCUMENT canonically (sort_keys, compact separators), not the file
        have_doc = hashlib.sha256(json.dumps(wl, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        c.info["header_workload_sha256"] = hw
        c.info["workload_doc_sha256"] = have_doc
        if not hw:
            c.fail("the record header carries no workload.sha256: the manifest of record cannot be tied to the run (fail closed) [M-0570]")
        elif hw not in (have_doc, have):
            c.fail(f"manifest of record differs from the header: RunHeader.workload.sha256 {hw[:16]} != manifest {have_doc[:16]} "
                   f"(canonical document; file {have[:16]}) -- the run was captured with another manifest [rev-pm M-0570]")
        hdr_by_id = {r["request_id"]: r for r in hdr_reqs}
        for i, w in enumerate(wl["requests"]):
            h = hdr_by_id.get(request_id_of(i))
            if h is None:
                continue
            ws, hs = w.get("seed"), h.get("seed")
            if ws is None and hs is not None:
                c.fail(f"no declared seed for {request_id_of(i)} but the run header records seed={hs} [M-0570]")
            elif ws is not None and hs is None:
                c.fail(f"{request_id_of(i)}: manifest declares seed={ws} but the run header records no seed [M-0570]")
            elif ws is not None and int(ws) != int(hs):
                c.fail(f"{request_id_of(i)}: manifest declares seed={ws} but the run header records seed={hs} [M-0570]")
        m_s, h_s = manifest_sampling(wl), dict((header.get("workload") or {}).get("sampling") or {})
        c.info["sampling"] = {"manifest": m_s, "header": {k: h_s.get(k) for k in ("variant", "temperature", "top_p", "top_k") if k in h_s}}
        # [rev-pm M-0667 N5] the manifest's own sources (variants table = the driver's, label, explicit fields) must agree: fail closed
        for conf in m_s.get("conflict") or []:
            c.fail(f"manifest of record is not one sampling declaration -- {conf} [M-0667 N5]")
        for k in ("variant", "temperature", "top_p", "top_k"):
            mv, hv = m_s.get(k), h_s.get(k)
            if mv is None or hv is None:
                continue
            same = (str(mv) == str(hv)) if k == "variant" else (abs(float(mv) - float(hv)) <= 1e-9)
            if not same:
                c.fail(f"manifest of record declares sampling {k}={mv!r} but the run header records {k}={hv!r} [M-0570]")
        # a request pinning ANOTHER variant (vllm_adapter [ov-sampling]): its table entry vs the header's per-request fields
        for i, w in enumerate(wl["requests"]):
            h = hdr_by_id.get(request_id_of(i))
            rv = w.get("variant")
            if h is None or (rv is None and h.get("variant") is None):
                continue
            r_s = manifest_sampling(wl, variant=str(rv)) if rv is not None else m_s
            for conf in (r_s.get("conflict") or []) if rv is not None else []:
                c.fail(f"{request_id_of(i)}: manifest variant {rv!r} is not one declaration -- {conf} [M-0667 N5]")
            # [rev-pm BLOCKER M-0717 (i)] a manifest that pins a per-request variant against a header WITHOUT a per-request sampling
            # record is not agreement: the run was not pinned (mirror of the seed rule above) -- a pin added after the run fails closed
            if rv is not None and all(h.get(k) is None for k in ("variant", "temperature", "top_p", "top_k")):
                c.fail(f"{request_id_of(i)}: manifest pins variant {rv!r} for the request but the run header records no per-request "
                       f"sampling (variant/temperature/top_p/top_k) -- the pin was not what the run served [M-0717]")
                continue
            for k in ("variant", "temperature", "top_p", "top_k"):
                mv, hv = r_s.get(k), h.get(k)
                if mv is None and rv is not None and k != "variant":
                    # the driver's defaults for a table entry that omits the field (vllm_adapter: temperature 0.0, top_p 1.0, top_k 0)
                    mv = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}[k]
                if mv is None or hv is None:
                    continue
                same = (str(mv) == str(hv)) if k == "variant" else (abs(float(mv) - float(hv)) <= 1e-9)
                if not same:
                    c.fail(f"{request_id_of(i)}: manifest declares sampling {k}={mv!r} but the run header records {k}={hv!r} for the request [M-0570]")
    if sorted(h_ids) != sorted(g_ids):
        only_g, only_h = sorted(set(g_ids) - set(h_ids)), sorted(set(h_ids) - set(g_ids))
        c.fail(f"record header requests != global Program requests: only in Program {only_g}, only in record {only_h}")
    for r in hdr_reqs:
        g = G.get(r["request_id"])
        if g is None:
            continue
        if g.get("workload_request_id") is not None and r.get("workload_request_id") is not None and str(r["workload_request_id"]) != str(g["workload_request_id"]):
            c.fail(f"{r['request_id']}: record served workload request id {r['workload_request_id']!r}, the Program declares {g['workload_request_id']!r}")
        lp = len(r["prompt_token_ids"])
        if lp != g["prompt_len"]:
            c.fail(f"{r['request_id']}: record prompt length {lp} != declared {g['prompt_len']}")
        if int(r.get("max_tokens", -1)) != int(g["max_tokens"]):
            c.fail(f"{r['request_id']}: record max_tokens {r.get('max_tokens')} != declared {g['max_tokens']}")
        sha = hashlib.sha256(json.dumps(r["prompt_token_ids"]).encode()).hexdigest()[:16]
        if g.get("prompt_sha256") and sha != g["prompt_sha256"]:
            c.fail(f"{r['request_id']}: record prompt != declared prompt (input association)")
        if wl is not None:
            w = wl["requests"][g["index"]] if g["index"] < len(wl["requests"]) else None
            if w is not None and list(w["prompt_token_ids"]) != list(r["prompt_token_ids"]):
                c.fail(f"{r['request_id']}: record prompt != manifest prompt at position {g['index']}")
    c.info.update({"n_declared": len(g_ids), "n_record": len(h_ids), "record_ids_unknown": res["request_ids"]["unknown_ids"]})
    c.done()
    mark("g1")

    # ---------------------------------------------------------------- attribute every fold instance to its request
    n_in = B.n_input_nodes
    # [R16 global, F-fp8-GM1 / M-0823] the shared weights = the Input nodes no request owns (one per leaf field of a mixed-width Record:
    # FP8 435; one node for a homogeneous bf16 Record) -- never `n_input_nodes - 1`.  [R15 stoch] the request-owned inputs are the prompts
    # (one per header request) + the seeds (compare.Prog.seed_index).  A layout that does not hold is a G2 failure by name, not a guess.
    _seed_index_all = dict(getattr(B, "seed_index", {}))
    _peer_index_all = dict(getattr(B, "peer_index", {}))   # [R17 D90] TP rank fold: peers node -> request index (one per request)
    weights_nodes_error = None
    try:
        weights_nodes = frozenset(BD.weights_input_nodes(B, len(hdr_reqs), len(_seed_index_all), len(_peer_index_all)))
    except ValueError as e:
        weights_nodes, weights_nodes_error = frozenset(), str(e)   # G5 fails by name below; every weight read is then a cross-request read
    meta_of = {o["node"] + n_in: o for o in B.oracle_rows}
    per_req: dict[str, list[int]] = collections.OrderedDict()
    unattributed = []
    # [R19, F-r19-int-15] weight-only instances (every node operand a shared weight or an earlier weight-only instance: gemma2's RMSNorm
    # scale `1 + w`, 11235 on #57 B8 1024/128) carry no request's data.  They are SHARED -- owned by no request, read like a weight --
    # and belong to the leg of every request with rows in their engine step (BD.shared_legs: the request's own derive computes the node
    # inside its forward at that step).  The same rule X-09 (BD.decompose) applies, after the meta channel and before the row-count rule,
    # which at B>1 refused them ("rows=1 != tokens|logits") and at B=1 handed them to the one request by coincidence.
    shared_nodes: set[int] = set()            # node ids of the weight-only instances (an operand source `is_weight_only` accepts)
    shared_idx: list[tuple[int, int]] = []    # (instance index, engine step)
    _compact = getattr(B, "compact", False)
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        rid = o.get("request") if o else None
        if rid is None and o is not None:
            if BD.is_weight_only(r, weights_nodes, shared_nodes, _compact):
                shared_nodes.add(r["k"])
                shared_idx.append((i, int(o["step"])))
                continue
            st = steps.get(o.get("step"))
            if st is not None:
                rid, _how = BD.attribute(o, {"rows": st["rows"], "logits_rows": st["logits_rows"]})
        if rid is None:
            unattributed.append(i)
        else:
            per_req.setdefault(rid, []).append(i)
    shared_gi: frozenset[int] = frozenset(i for i, _ in shared_idx)
    if shared_idx:
        per_req = collections.OrderedDict(BD.shared_legs(per_req, shared_idx, steps))
    mark("attribution")

    # ---------------------------------------------------------------- G2: every declared request executed, nothing extra
    c = _Check(res, "G2", "requests_observed: every declared request has rows + output events + instances; 0 unattributed; no undeclared request")
    present: dict[str, list[int]] = collections.defaultdict(list)
    for s in sorted(steps):
        for rid in {r for r, _ in steps[s]["rows"]}:
            present[rid].append(s)
    for rid in g_ids:
        if rid not in present:
            c.fail(f"{rid}: declared but never scheduled (no StepStart rows)")
        if rid not in per_req:
            c.fail(f"{rid}: declared but no fold instance is attributed to it")
    for rid in set(present) | set(per_req):
        if rid not in G:
            c.fail(f"{rid}: executed / in the fold but not declared")
    if unattributed:
        c.fail(f"{len(unattributed)} fold instances attributed to no request")
    c.info.update({"n_instances": len(B.inst), "attributed": sum(len(v) for v in per_req.values()), "unattributed": len(unattributed),
                   "engine_steps": len(steps),
                   # [F-r19-int-15] weight-only instances: owned by no request, in the leg of every request with rows in their step
                   "weight_only_shared": len(shared_idx),
                   "weight_only_specs": dict(collections.Counter(B.inst[i]["spec"].split("{")[0] for i in shared_gi).most_common(6))})
    c.done()
    mark("g2")

    # ---------------------------------------------------------------- G3 + G4 + G5 + G7/G8 material: per request
    c3 = _Check(res, "G3", "subcircuits_x09: each request's projection canonical-equal to its component over the executed steps (eos-truncated labelled)")
    c4 = _Check(res, "G4", "identity_map: (request, local step, local index) <-> global node is a bijection over the attributed fold")
    c5 = _Check(res, "G5", "shared_state: one weights input for all requests (no overlapping fields); node operands same-request, producer step <= consumer step")
    glob2loc: dict[int, tuple[str, int, int]] = {}
    shared_loc: dict[int, dict[str, tuple[str, int, int]]] = {}   # [F-r19-int-15] weight-only instance -> {request: its local identity}
    loc_count = 0
    exec_pop = 0
    unexercised = 0
    A_cache: dict[str, C.Prog] = _loaded.setdefault("components", {})  # decoded components shared with the alternate-criterion run
    proj: dict[str, BD.Projection] = {}
    comps: dict[str, C.Prog] = {}
    xstep_specs: collections.Counter = collections.Counter()
    weight_readers: set[str] = set()
    if getattr(B, "oracle_wmap_overlaps", None):
        c5.fail(f"weights field map has {len(B.oracle_wmap_overlaps)} overlapping ranges (ambiguous shared-value identity)")
    engine_step_of: dict[int, int] = {}  # global instance index -> engine step
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        if o is not None and o.get("step") is not None:
            engine_step_of[i] = int(o["step"])
    # [R17 mfast, BRIEF §5 step 3 -- safe parallelization] The per-request work below is ONE function of the request's declaration, its
    # attributed instances (established above), and the shared READ-ONLY fold + component Programs: `leg(rid)` returns facts (the request
    # entry, the check problems in the order it met them, the local->component address map, population contributions, the request's
    # dependency-order facts for G8) and writes nothing shared.  `apply(out)` -- always in the parent, always in request order -- does the
    # global reconciliation: the identity map over EVERY request (two requests claiming one instance is named here), the population sums,
    # the problems appended to G3/G4/G5 in the same order the inline form appended them.  The checker of record runs leg+apply inline per
    # request.  MATCH_IMPL=fast runs the legs first (GMF.request_legs: forked workers over the parent's objects, copy-on-write -- never a
    # copy of the trace per process; a failed worker raises, never a silent fallback), then apply in request order: same function, same
    # inputs, same facts.  State-linked steps and cross-rank dependencies are NOT assumed independent: nothing here compares across
    # requests; every cross-request fact (shared inputs G5, interfaces G7, layout G8, population G7) stays in the single parent pass.
    # `pre` is keyed by POSITION in gp["requests"], not by request_id: a Program that lists one request twice (G1 names it) still gets one
    # entry per listed request, exactly as the inline loop produced -- the record's document is the same document.
    pre: list[tuple] = []   # position -> (g, ent, idxs, P, A | None, cd): Projections + components built in the parent, shared read-only
    for g in gp["requests"]:
        rid = g["request_id"]
        ent = {"request_id": rid, "workload_request_id": g.get("workload_request_id"), "index": g["index"], "LP": g["prompt_len"], "max_tokens": g["max_tokens"],
               "arrive_step_declared": g["arrive_step"]}
        idxs = per_req.get(rid)
        if not idxs:
            ent["error"] = "no attributed instances"
            pre.append((g, ent, None, None, None, None))
            continue
        pidx = param_of.get(rid)
        # [R15 stoch] the request's seed parameter (`seed_<i>`, fold._header) is its own input, not a cross-request read
        # (`Input64_v1` nodes in request order, compare.Prog.seed_index: the i-th seed node is request index i's)
        seed_index = dict(getattr(B, "seed_index", {}))
        my_seed = next((k for k, i in seed_index.items() if i == g["index"]), None)
        my_peers = next((k for k, i in _peer_index_all.items() if i == g["index"]), None)   # [R17 D90] the request's own peers node
        if fast:   # [R17 mprof] one Projection per request per process: X-09's (same construction key) / the verdict-of-record run's for the alternate
            P = GMF.projection(_loaded, rid, B, f"oracle[{rid}]", idxs, pidx, weights_nodes, my_seed, peer_node=my_peers)
        else:
            P = ops.projection(B, f"oracle[{rid}]", idxs, tok_node=pidx, weights_node=weights_nodes, seed_node=my_seed, peer_node=my_peers)
        P._all_seed_nodes = set(seed_index)
        proj[rid] = P
        cd = (g.get("component") or {}).get("dir")
        A = None
        if cd:
            A = A_cache.get(cd)
            if A is None:
                A = A_cache[cd] = ops.load_derived(cd, f"component[{rid}]")
            comps[rid] = A
        pre.append((g, ent, idxs, P, A, cd))

    def leg(k: int) -> dict:
        """The request's own facts: canonical operand sources (P.canon), chronology of its producers, comparison with its component
        (compare_steps; under `dag` the Merkle matching per executed step -> canonical addresses).  Reads shared objects, writes none."""
        g, ent, idxs, P, A, cd = pre[k]
        rid = g["request_id"]
        out = {"rid": rid, "ent": ent, "fails": [], "P_steps": None, "local_addr": None, "A_steps_len": None, "exec_pop": 0, "unexercised": 0,
               "weight_reader": False, "xstep": collections.Counter(), "dep_viol": [], "perm": {}}
        fails = out["fails"]
        if idxs is None:
            return out
        for i_local in range(len(P.inst)):
            P.canon(i_local)  # resolves every operand source once: fills P.xreq (cross-request reads) before the checks below
        n_ts = sum(1 for r in P.inst if _is_sampling_event(r["spec"]))   # [R15 stoch] any sampling-event family
        LP = g["prompt_len"]
        prefill_steps = sorted(s for s in steps if any(r == rid and pos < LP for r, pos in steps[s]["rows"]))
        chunks = max(1, len(prefill_steps))
        T_obs = n_ts - chunks
        ent.update({"n_instances": len(idxs), "local_steps": len(P.steps), "token_selects": n_ts, "prefill_chunks": chunks, "T_observed": T_obs})
        if P.xreq:
            fails.append(("G5", f"{rid}: {sum(P.xreq.values())} cross-request operand reads {dict(P.xreq)}"))
        # G8's dependency-order facts of THIS request (DECISION M-0021 (4): every producer emitted before its consumer inside a request
        # step) -- the same scan G8 made over every request's projection, done here, AFTER the G5 count above:
        # every canon() call of the record counts cross-request sources again, and the record's count is the first pass's
        for s_, loc in enumerate(P.steps):
            for pos, i in enumerate(loc):
                for a in P.canon(i, False):
                    for t in a:
                        if t[0] != "N":
                            continue
                        mx = _n_max_producer(P.steps, t)   # O(1) for a progression whose producers are monotone; all runs otherwise
                        if mx >= i:
                            for prod in _n_producers(P.steps, t):   # one record per violating RUN, as the record emits
                                if prod >= i:
                                    out["dep_viol"].append({"request_id": rid, "local_step": s_, "emission": pos, "consumer": i, "producer": prod})
        # producer chronology: every node operand's producer is this request's instance in the same or an earlier engine step
        late = 0
        compact = getattr(B, "compact", False)
        if fast and not compact:   # [R17 mprof unit 5] run form: the same count with node -> producer step resolved once per fold
            late, _xs, _wr = GMF.producer_chronology(P, idxs, B, engine_step_of, weights_nodes)
            out["xstep"].update(_xs)
            if _wr:
                out["weight_reader"] = True
        else:
          for i_local, r in enumerate(P.inst):
            gi = idxs[i_local]
            cs = engine_step_of.get(gi)
            for a in r["args"]:
                for run in a:   # record: (space, idx, lo, hi), one per run; compact: (space, (idx0, lo, hi), (dI, dL, dH), count) = `count` runs
                    if run[0] != "n":
                        continue
                    if compact:
                        dI, n_runs = run[2][0], run[3]
                        idxs_ = (run[1][0],) if dI == 0 else range(run[1][0], run[1][0] + n_runs * dI, dI)
                        w = n_runs if dI == 0 else 1
                    else:
                        idxs_, w = (run[1],), 1
                    for idx in idxs_:
                        if idx in B.inputs:
                            if idx in weights_nodes:
                                out["weight_reader"] = True
                            continue
                        j = B.k2i.get(idx)
                        ps = engine_step_of.get(j) if j is not None else None
                        if ps is not None and cs is not None and ps > cs:
                            late += w
                        if ps is not None and cs is not None and ps < cs:
                            out["xstep"][r["spec"].split("{")[0]] += w
        if late:
            fails.append(("G5", f"{rid}: {late} operands produced in a LATER engine step than their consumer"))
        # component
        if not cd:
            fails.append(("G3", f"{rid}: no component Program in the global Program for shape (LP {LP}, T_bound {g.get('T_bound')})"))
            return out
        ent["component"] = {"dir": cd, "digest": A.digest, "steps": len(A.steps), "instances": len(A.inst),
                            "LP": g["component"].get("LP"), "T": g["component"].get("T")}
        want_d = g["component"].get("program_digest")
        if want_d and want_d != A.digest:
            fails.append(("G4", f"{rid}: the global Program's component digest {want_d[:16]} != the loaded component's {A.digest[:16]}"))
        ga = g.get("global_address")
        if ga:
            ent["global_address"] = ga
            if ga.get("n_instances") is not None and int(ga["n_instances"]) != len(A.inst):
                fails.append(("G4", f"{rid}: address map declares {ga['n_instances']} instances for the request, the component has {len(A.inst)}"))
        n_exec = len(P.steps)
        if n_exec > len(A.steps):
            fails.append(("G3", f"{rid}: executed {n_exec} local steps but the bounded component has {len(A.steps)} (output interface exceeded)"))
        cmp = ops.compare_steps(A, P, unfold_a=False)
        exec_steps = cmp["steps"][:n_exec]
        bad_seq = [s for s in exec_steps if not s["spec_seq_equal"] or (s.get("wiring") or {}).get("operands_different")]
        tail = cmp["steps"][n_exec:]
        ent["eos_truncated"] = bool(tail)
        ent["unexecuted_steps"] = len(tail)
        out["unexercised"] = sum(len(A.steps[s]) for s in range(n_exec, len(A.steps)))
        ent["sequence_equal_steps"] = n_exec - len(bad_seq)
        ent["operands"] = {"equal": sum((s.get("wiring") or {}).get("operands_equal", 0) for s in exec_steps),
                           "different": sum((s.get("wiring") or {}).get("operands_different", 0) for s in exec_steps)}
        # canonical address of every executed fold instance: (rid, s, j) with j the COMPONENT's index of the corresponding instance.
        #   sequence: j = emission position (X-09 sequence criterion: the fold's order IS the component's order, else G3 fails);
        #   dag:      j = the index of the structurally identical component instance (equal Merkle DAG hash: spec, form, out type,
        #             operands with node sources named by their producer's hash) -- the emission order is never an address.
        local_addr: list[list[int | None]] = []
        if within_step == "dag":
            ha, hb = ops.dag_hashes(A), ops.dag_hashes(P)
            bad_dag, dag_info = [], []
            for s in range(min(n_exec, len(A.steps))):
                by_hash: dict[str, list[int]] = collections.defaultdict(list)
                for j, ia in enumerate(A.steps[s]):
                    by_hash[ha[ia]].append(j)
                classes_gt1 = sum(1 for v in by_hash.values() if len(v) > 1)
                addr, unmatched = [], []
                for il in P.steps[s]:
                    js = by_hash.get(hb[il])
                    if js:
                        addr.append(js.pop(0))
                    else:
                        addr.append(None)
                        unmatched.append(il)
                leftover = sum(len(v) for v in by_hash.values())
                permuted = sum(1 for j, x in enumerate(addr) if x is not None and x != j)
                if unmatched or leftover:
                    spec = lambda Q, i: Q.inst[i]["spec"].split("{")[0]  # noqa: E731
                    bad_dag.append({"step": s, "n_a": len(A.steps[s]), "n_b": len(P.steps[s]), "unmatched_fold": len(unmatched), "unmatched_component": leftover,
                                    "unmatched_fold_specs": dict(collections.Counter(spec(P, i) for i in unmatched).most_common(6)),
                                    "unmatched_component_specs": dict(collections.Counter(spec(A, A.steps[s][j]) for v in by_hash.values() for j in v).most_common(6)),
                                    "forward_refs_in_fold": sum(1 for i in unmatched for a in P.canon(i, False) for t in a
                                                                for prod in _n_producers(P.steps, t) if prod >= i)})
                dag_info.append({"step": s, "permuted": permuted, "duplicate_hash_classes": classes_gt1,
                                 "duplicate_hash_class_sizes": dict(collections.Counter(len(v) for v in by_hash.values() if len(v) > 1))})
                local_addr.append(addr)
            for s in range(len(A.steps), n_exec):   # executed beyond the bounded component: no address (G3 already fails)
                local_addr.append([None] * len(P.steps[s]))
            bad = bad_dag
            ent["executed_steps_equal"] = not bad
            ent["n_divergent_steps"] = len(bad)
            ent["criterion"] = "dag (order-insensitive Merkle multiset per request step; X-09's chunked-prefill criterion) over executed steps" + \
                               (" of the bounded component (eos-truncated)" if tail else "")
            sizes: collections.Counter = collections.Counter()
            for d in dag_info:
                sizes.update({int(k): v for k, v in d["duplicate_hash_class_sizes"].items()})
            # hash classes of multiplicity > 1 (structurally identical instances inside one request step, e.g. the same op over two equal
            # KV blocks): the bijection is chosen in emission order inside the class -- reported with the class sizes (rev-pm M-0042 (3))
            ent["within_step"] = {"permuted_instances": sum(d["permuted"] for d in dag_info), "steps_with_permutation": sum(1 for d in dag_info if d["permuted"]),
                                  "duplicate_hash_classes": sum(d["duplicate_hash_classes"] for d in dag_info),
                                  "duplicate_hash_class_sizes": {str(k): v for k, v in sorted(sizes.items())},
                                  "sequence_also_equal": not bad_seq}
            if bad:
                ent["first_divergence"] = bad[0]
                fails.append(("G3", f"{rid}: {len(bad)} of {n_exec} executed steps not DAG-equal to the component (first: local step {bad[0]['step']}, "
                                    f"{bad[0]['unmatched_fold']} fold / {bad[0]['unmatched_component']} component instances unmatched)"))
            if fast:
                # [F-r17-07] G8's per-step Merkle matching of every step G8 may name (its address differs from the emission position
                # anywhere), computed HERE with this process's hash ids -- the leg hands back the matching (facts), never hash dicts:
                # the fast path's dag ids are per process and a worker's must not meet the parent's or another worker's
                for s, addr in enumerate(local_addr):
                    if any(x is None or x != pos for pos, x in enumerate(addr)) and _perm_eligible(A, P, s):
                        out["perm"][s] = _perm_match(A, P, s, ha, hb)
        else:
            bad = bad_seq
            ent["executed_steps_equal"] = not bad
            ent["n_divergent_steps"] = len(bad)
            ent["criterion"] = "sequence (X-09) over executed steps" + (" of the bounded component (eos-truncated)" if tail else "")
            if bad:
                ent["first_divergence"] = {"step": bad[0]["step"], "n_a": bad[0]["n_a"], "n_b": bad[0]["n_b"], "spec_seq_equal": bad[0]["spec_seq_equal"],
                                           "wiring_examples": (bad[0].get("wiring") or {}).get("examples", [])[:3]}
                fails.append(("G3", f"{rid}: {len(bad)} of {n_exec} executed steps not canonical-equal (first: local step {bad[0]['step']})"))
                if fast and _perm_eligible(A, P, bad[0]["step"]):   # [F-r17-07] G8 names the first divergence: its matching, computed here
                    s0 = bad[0]["step"]
                    out["perm"][s0] = _perm_match(A, P, s0, ops.dag_hashes(A), ops.dag_hashes(P))
            local_addr = [list(range(len(loc))) for loc in P.steps]
        if P.xreq:
            fails.append(("G3", f"{rid}: cross-request reads"))
        out.update({"P_steps": P.steps, "local_addr": local_addr, "A_steps_len": [len(s) for s in A.steps],
                    "exec_pop": sum(len(A.steps[s]) for s in range(min(n_exec, len(A.steps))))})
        return out

    checks_by_id = {"G3": c3, "G4": c4, "G5": c5}
    dep_viol: list[dict] = []   # every request's dependency-order violations, in request order (G8 names the first)
    perm_facts: dict[str, dict] = {}   # [F-r17-07] rid -> {local step: G8's Merkle matching}, computed by the leg in ITS process

    def apply(out: dict) -> None:
        """Global reconciliation of one request's facts, in request order: problems to their checks in the order met, the identity map
        over all requests (an instance with two local identities is named), the population sums, the request entry on the record."""
        nonlocal loc_count, exec_pop, unexercised
        rid, ent = out["rid"], out["ent"]
        for cid, msg in out["fails"]:
            checks_by_id[cid].fail(msg)
        if out["weight_reader"]:
            weight_readers.add(rid)
        xstep_specs.update(out["xstep"])
        dep_viol.extend(out["dep_viol"])
        if out.get("perm"):
            perm_facts[rid] = out["perm"]
        if out["P_steps"] is None:
            res["requests"].append(ent)
            return
        idxs, local_addr, A_len = per_req[rid], out["local_addr"], out["A_steps_len"]
        # identity map: local (s, j) <-> global node
        for s, loc in enumerate(out["P_steps"]):
            if s < len(A_len) and A_len[s] != len(loc):
                c4.fail(f"{rid}: local step {s} has {len(loc)} executed instances but the component has {A_len[s]}")
            for pos, il in enumerate(loc):
                gi = idxs[il]
                j = local_addr[s][pos] if s < len(local_addr) and pos < len(local_addr[s]) else None
                if j is None:
                    if within_step == "dag":
                        c4.fail(f"{rid}: fold instance {gi} (local step {s}, emission {pos}) has no structurally identical component instance: no canonical address")
                    j = pos  # emission position: keeps the map total for reporting (the step is already a G3 failure)
                key = (rid, s, j)
                if gi in shared_gi:
                    # [F-r19-int-15] a weight-only instance has one local identity PER request whose leg holds it (never a bijection
                    # with the global node): kept apart from glob2loc, so G7's owner blocks and G8's layout skip it like an unowned node
                    shared_loc.setdefault(gi, {})[rid] = key
                    loc_count += 1
                    continue
                if gi in glob2loc:
                    c4.fail(f"global instance {gi} carries two local identities {glob2loc[gi]} and {key}")
                glob2loc[gi] = key
                loc_count += 1
        exec_pop += out["exec_pop"]
        unexercised += out["unexercised"]
        res["requests"].append(ent)

    order = list(range(len(pre)))
    if fast:
        n_workers = GMF.workers(sum(1 for k in order if pre[k][2] is not None))
        GMF.stats["request_workers"] = max(GMF.stats.get("request_workers", 0), n_workers)   # telemetry (impl block), never the verdict
        legs = GMF.request_legs(order, leg, n_workers, weight=lambda k: len(pre[k][2] or ()))
        for out in legs:
            apply(out)
    else:
        for k in order:
            apply(leg(k))
    n_distinct = len(set().union(*per_req.values())) if per_req else 0   # every attributed fold instance once (a shared one sits in several legs)
    if len(glob2loc) + len(shared_loc) != n_distinct:
        c4.fail(f"identity map covers {len(glob2loc) + len(shared_loc)} of {n_distinct} attributed instances (not total)")
    c4.info.update({"mapped": len(glob2loc), "attributed": sum(len(v) for v in per_req.values()), "distinct": n_distinct,
                    "weight_only_shared": len(shared_loc), "weight_only_local_identities": sum(len(v) for v in shared_loc.values())})
    c3.done()
    c4.done()
    missing_w = [rid for rid in per_req if rid not in weight_readers]
    if missing_w:
        c5.fail(f"requests that never read the shared weights input: {missing_w}")
    if weights_nodes_error:
        c5.fail(f"shared weights input nodes could not be identified from the fold's parameter layout: {weights_nodes_error}")
    c5.info.update({"weights_param": (min(weights_nodes) if weights_nodes else None),   # first weights Input node (== the node for bf16)
                    # [R16 global, F-fp8-GM1] the weights as a node range: one node (homogeneous bf16 Record) or one per leaf field (FP8)
                    "weights_input_nodes": {"n": len(weights_nodes), "first": min(weights_nodes), "last": max(weights_nodes)} if weights_nodes else None,
                    "weight_fields": len(getattr(B, "oracle_wmap", []) or []),
                    "cross_step_producer_classes": dict(xstep_specs.most_common(8))})
    c5.done()
    mark("g3_g4_g5_per_request")

    # ---------------------------------------------------------------- G6: chronology
    c = _Check(res, "G6", "chronology: declared == realized == observed arrival; contiguous rows; output events = TokenSelects, k-th = k-th token, feed the next row; bounded")
    # [F-r17-01, rev M-0035] the driver's tokens.json is PART of the record: without it every output-association leg (driver-served token k
    # == the recorder's k-th TokenSelect, served length == the driver's, the R17-1 two-length claims, the tokens.json lag attestation) is
    # silently skipped.  Fail closed by name; `tokens_optional=True` (CLI --tokens-optional) accepts a record without it for mechanism /
    # fixture runs, LABELLED in checks[G6].tokens_json=false -- never a row of record (row_pod always produces the file).
    if tokens is None and not tokens_optional:
        c.fail(f"the record has no {os.path.relpath(tokens_path, oracle_dir)}: the output-association legs (driver-served tokens vs the recorder's "
               f"TokenSelects, served/executed length claims, tokens.json lag attestation) cannot run; a record without it is refused unless the caller "
               f"passes --tokens-optional (labelled, never a row of record) (F-r17-01)")
    tok_req = {r["request_id"]: r for r in (tokens or {}).get("requests", [])} if tokens else {}
    _sr = gp.get("stop_rule") or {}
    eos_ids = (set(int(x) for x in _sr["eos_token_ids"]) if _sr.get("eos_token_ids")
               else (set() if _sr.get("ignore_eos") else None))   # ignore_eos: declared EMPTY set -> every early stop fails, named
    # [R17-1 / DP-pad-02 (b); supersedes the ruling-21 reader] LAG -- the rows the scheduler computes after an EOS stop -- is the PROGRAM's
    # declared specialization parameter (GP-01 `lag.value` == the Workload_v1 `lag` static == stop_rule.lag/eos_lag_rows; from_gprog verified
    # the block against the descriptor).  G6 consumes THAT value and nothing else; the record's own attestation (RunHeader.profile
    # .engine_kwargs.async_scheduling, the engine's RESOLVED read-back) must AGREE with it (G1 compares the engine fact; here the lag it
    # implies) -- a disagreement or a record that does not attest FAILS by name.  Nothing is ever inferred from the trace: a Program that
    # declares no lag (pre-R17-1 Build) cannot have an EOS stop checked (FAIL by name per request); when no request stops early the lag is
    # not exercised and the record is labelled so (lag.exercised false), never silently accepted as a lag-N record.
    rec_lag, rec_lag_src = eos_lag_rows(facts)                       # what the RECORD attests (for the agreement check + the label)
    _lag_block = gp.get("lag") if isinstance(gp.get("lag"), dict) else None
    decl_lag = (_lag_block or {}).get("value")
    if decl_lag is None:
        decl_lag = _sr.get("lag", _sr.get("eos_lag_rows"))
    decl_lag = int(decl_lag) if decl_lag is not None else None
    decl_src = (_lag_block or {}).get("source") or _sr.get("eos_lag_source") or "stop_rule.eos_lag_rows"
    lag_declared = decl_lag is not None
    if lag_declared and decl_lag not in (0, 1):
        c.fail(f"the Program declares lag {decl_lag!r}, not in {{0, 1}}: not a supported scheduler specialization (R17-1)")
        lag_declared = False
    if lag_declared and rec_lag is not None and decl_lag != rec_lag:
        c.fail(f"the Program is specialized for LAG {decl_lag} (async_scheduling={bool(decl_lag)}: {decl_src}) but the record attests "
               f"{rec_lag_src} -> LAG {rec_lag}: the record is not an execution of the declared scheduler (R17-1; the header's LAG-{rec_lag} "
               f"attestation on a LAG-{decl_lag} Program FAILS by name, never re-read from the trace)")
    if lag_declared and rec_lag is None:
        c.fail(f"the Program is specialized for LAG {decl_lag} (async_scheduling={bool(decl_lag)}: {decl_src}) but the record does not attest "
               f"async_scheduling ({rec_lag_src}; RunHeader.profile.engine_kwargs / RunHeader.engine_kwargs / capture/versions.json): the "
               f"declared scheduler cannot be checked against the record, refuse by name (R17-1)")
    # the lag G6 CONSUMES: the Program's declaration (None = pre-R17-1 Program: no rule to apply; every EOS stop then fails by name below)
    eos_lag = decl_lag if lag_declared else None
    eos_lag_src = (f"Program lag {decl_lag} ({decl_src}); record attests {rec_lag_src}" if lag_declared
                   else f"Program declares no lag (pre-R17-1 Build); record attests {rec_lag_src} (NOT used as the rule)")
    c.info["stop_rule"] = {"eos_token_ids": sorted(eos_ids) if eos_ids else None, "source": _sr.get("source"),
                           "rule": "an early stop must end in a declared EOS id; else FAIL (F-pm-24)",
                           "eos_lag_rows": eos_lag, "eos_lag_source": eos_lag_src, "eos_lag_rows_declared": decl_lag,
                           "lag_rule": "R17-1: after an EOS stop exactly LAG (the Program's declared static) more row(s) run, each fed the preceding "
                                       "sample, its sample fed nowhere and absent from generated_token_ids (executed_prefix_len = a_r + LAG, "
                                       "served_output_len = a_r); the record must attest the same async_scheduling; undeclared -> refuse by name",
                           "evidence": "the recorder's SampleEnd token, cross-checked against the driver's tokens.json (output association); "
                                       "the fold TokenSelect VALUE is Commit territory (C3 openings of the sampled ids), not Match's"}
    c.info["lag"] = {"declared": decl_lag, "declared_source": decl_src if lag_declared else None, "record_attests": rec_lag, "record_source": rec_lag_src,
                     "agree": (decl_lag == rec_lag) if (lag_declared and rec_lag is not None) else None,
                     "exercised": False, "eos_stop_requests": [], "lag_rows_observed": {}, "variable_lag": False,
                     "lengths": gp.get("lengths") or ((gp.get("chronology") or {}).get("lengths") if isinstance(gp.get("chronology"), dict) else None)}
    # [R17-1] the driver's tokens.json carries the same recorded fact (doc.lag.value from resolved_async_scheduling) and per request its
    # served_output_len / executed_prefix_len claims: one record, one fact -- a tokens.json whose lag disagrees with the header's attestation
    # is a record that does not attest one execution (named); the per-request claims are checked against the observed rows below
    _tok_lag = ((tokens or {}).get("lag") or {}).get("value") if isinstance((tokens or {}).get("lag"), dict) else None
    if _tok_lag is not None and rec_lag is not None and int(_tok_lag) != int(rec_lag):
        c.fail(f"the record's own sources disagree on the scheduler lag: tokens.json lag.value {_tok_lag} vs {rec_lag_src} -> LAG {rec_lag}: "
               f"not one attested execution (R17-1)")
    c.info["lag"]["tokens_json_lag"] = _tok_lag
    _lag_obs: dict[str, int] = {}                                    # EOS-stopped request -> rows served after its EOS (for the variable-lag finding)
    idle_rule = 0
    for g in gp["requests"]:
        rid = g["request_id"]
        ent = next(e for e in res["requests"] if e["request_id"] == rid)
        _n_problems_before = len(c.problems)
        ps = present.get(rid)
        if not ps:
            continue
        first, last = ps[0], ps[-1]
        ent.update({"first_step": first, "finish_step": last})
        if ps != list(range(first, last + 1)):
            c.fail(f"{rid}: rows not contiguous over engine steps {first}..{last} (preempted / resumed: not in the Program)")
        decl = int(g["arrive_step"])
        tr = tok_req.get(rid) or {}
        realized = tr.get("arrive_step_realized")
        if realized is None:
            realized = tr.get("first_step")
        ent["arrive_step_realized"] = realized
        if realized is not None and int(realized) != first:
            c.fail(f"{rid}: driver realized arrival {realized} != observed first engine step {first}")
        if first != decl:
            # the declared rule: admitted early only if every earlier-arriving request had finished (engine would idle)
            earlier = [q for q in gp["requests"] if (q["arrive_step"], q["index"]) < (decl, g["index"])]
            if first < decl and earlier and all(present.get(q["request_id"], [-1])[-1] < first for q in earlier):
                idle_rule += 1
                ent["arrival_note"] = f"admitted at {first} < declared {decl}: every earlier request had finished (declared idle rule)"
            else:
                c.fail(f"{rid}: declared arrival step {decl} != observed {first}")
        pst = g.get("programmed_steps")
        if pst and int(pst.get("prefill", decl)) != first and "arrival_note" not in ent:
            c.fail(f"{rid}: programmed prefill step {pst.get('prefill')} != observed {first}")
        if pst and pst.get("decode") and ent.get("prefill_chunks", 1) == 1:
            d0 = int(pst["decode"][0]) + (first - decl)
            if len(ps) > 1 and ps[1] != d0:
                c.fail(f"{rid}: programmed first decode step {d0} != observed {ps[1]}")
        if tr.get("first_step") is not None and int(tr["first_step"]) != first:
            c.fail(f"{rid}: driver first_step {tr['first_step']} != observed {first}")
        # (the driver's finish_step is compared below, once the request's accepted lag rows after an EOS stop are known)
        # output events
        LP = g["prompt_len"]
        events = []  # (engine step, sampled token, logits position)
        for s in ps:
            lr = steps[s]["logits_rows"]
            js = [j for j, (r, _p) in enumerate(lr) if r == rid]
            if len(js) > 1:
                c.fail(f"{rid}: {len(js)} logits rows in engine step {s}")
            for j in js:
                pos = lr[j][1]
                samp = samples.get(s)
                if samp is None or j >= len(samp):
                    c.fail(f"{rid}: no SampleEnd token for engine step {s} logits row {j}")
                    continue
                events.append((s, samp[j], pos))
        served = [(s, t) for s, t, pos in events if pos >= LP - 1]      # a chunk's discarded sample has pos < LP-1
        ent["output_events"] = len(served)
        ent["output_event_steps"] = [s for s, _ in served][:6] + (["..."] if len(served) > 6 else [])
        ent["output_event_steps_last"] = served[-1][0] if served else None
        n_ts = ent.get("token_selects")
        if n_ts is not None and n_ts != len(events):
            c.fail(f"{rid}: {n_ts} TokenSelect instances in the fold but {len(events)} sampled logits rows in the record")
        gen = tr.get("generated_token_ids")
        cap = int(g["max_tokens"])
        # [COORD M-0213 (1)(2) / ruling 21 / DP-pad-02] the request's served rows are its answer a_r = the events up to and including
        # the FIRST declared EOS (or all of them on a length stop) plus, after an EOS stop ONLY, exactly the placeholder rows the
        # record's attested scheduler computes (eos_lag_rows: async_scheduling=True -> 1, False -> 0, unattested -> refuse by name).
        # Every accepted lag row's input is the preceding sample (the feed-forward check below), its own sample is fed nowhere and
        # is absent from the driver's generated_token_ids; the fold holds a TokenSelect for it (token_selects == events). Anything
        # else -- more rows than attested, fewer (a deleted step), a lag row after a length stop -- FAILS by name.
        first_eos = next((j for j, (_s, t) in enumerate(served) if eos_ids and int(t) in eos_ids), None)
        # an EOS sampled AT the cap is a length stop (vLLM checks max_tokens before the EOS; the scheduler knows the length ahead: no
        # placeholder row); an EOS before the cap is an EOS stop and the lag rule applies
        eos_stop = first_eos is not None and first_eos + 1 < cap
        # a length stop's served output is AT MOST the cap: rows beyond it are execution past the cap (named below), never output
        a_r = first_eos + 1 if eos_stop else min(len(served), cap)
        lag_rows = len(served) - a_r
        ent["answer_events"] = a_r
        ent["lag_rows"] = lag_rows
        # [R17-1] the two lengths, kept distinct on the record: what was DESIGNATED as output vs what RAN
        ent["served_output_len"] = a_r
        ent["executed_prefix_len"] = len(served)
        ent["cap_exhausted"] = (a_r == cap)
        if eos_stop:
            _lag_obs[rid] = lag_rows
            if lag_declared and (rec_lag is None or rec_lag != decl_lag):
                # the record-level disagreement (named once above) makes THIS request's EOS stop unacceptable too: its post-EOS rows cannot be
                # read as the declared lag when the record does not attest the Program's scheduler -- the request is not established (no
                # downstream state, stoch's scheduled live included, is built on it)
                c.fail(f"{rid}: EOS stop at engine step {served[first_eos][0]} on a record whose attested scheduler ({rec_lag_src}) is not the "
                       f"Program's declared LAG {decl_lag}: the {lag_rows} post-EOS row(s) are not accepted as the declared lag (R17-1)")
        # the driver's own claims for the two lengths (tokens.json, R17-1 schema: served = len(generated_token_ids), executed = served + the
        # record's lag on an EOS stop) must be what the record's rows show -- a claim beside the rows, named when it disagrees
        _tq = tok_req.get(rid) or {}
        for _k, _v in (("served_output_len", a_r), ("executed_prefix_len", len(served))):
            if _tq.get(_k) is not None and int(_tq[_k]) != int(_v):
                c.fail(f"{rid}: the driver's tokens.json claims {_k} {_tq[_k]} but the record's rows show {_v} (served_output_len {a_r}, "
                       f"executed_prefix_len {len(served)}, cap {cap}): the record's own claim disagrees with its rows (R17-1)")
        answer = [t for _, t in served[:a_r]]
        if gen is not None and answer != list(gen)[:a_r]:
            c.fail(f"{rid}: sampled tokens per engine step {answer[:8]} != driver's generated_token_ids {list(gen)[:8]} (output association)")
        if gen is not None and len(list(gen)) != a_r:
            c.fail(f"{rid}: the driver's generated_token_ids has {len(list(gen))} tokens but the request's answer is {a_r} event(s) "
                   f"(up to its first declared EOS): {'the rows after the EOS are not output (a lag row sample reported)' if len(list(gen)) > a_r else 'output tokens the record never sampled'} (M-0213)")
        if len(served) > cap:
            # [R17-1, rev M-0027 (4)] REQUIRED from the trace, not assumed: a length stop runs no row after the cap under ANY lag (vLLM knows the
            # length ahead; the scheduler computes a post-EOS row only for an EOS stop) -- an executed row past the cap is execution past cap
            c.fail(f"{rid}: execution past cap -- {len(served)} output events executed but the declared max_tokens is {cap} (executed_prefix_len "
                   f"{len(served)} > served_output_len {a_r} = cap; a length stop computes no post-cap row under any LAG, the Program's LAG "
                   f"{eos_lag if eos_lag is not None else 'undeclared'} applies to EOS stops only) (output interface; R17-1)")
        if ps and a_r != cap:
            ent["finish"] = "eos" if a_r < cap else "?"
            if tr.get("finish_reason") not in (None, "stop"):
                c.fail(f"{rid}: {a_r} events < max_tokens but driver finish_reason {tr.get('finish_reason')!r}")
            # [rev-pm ALERT M-0537 / F-pm-24] the driver's finish_reason is the prover's word; the EOS itself is the evidence: the last
            # ANSWER token (tied to a SampleEnd + the driver's generated_token_ids by the output association above) must be one of the
            # Program's declared EOS ids. A request that stopped early on any other token (dropped, aborted, stop string, a prover that
            # skipped the tail) is NOT an EOS truncation and FAILS -- whatever finish_reason says. No declared set -> fail closed.
            last_tok = served[a_r - 1][1] if served and a_r else None
            if eos_ids is None:
                c.fail(f"{rid}: stopped after {a_r} < max_tokens {cap} events but the Program declares no EOS set "
                       f"(stop_rule.eos_token_ids): the truncation cannot be verified, fail closed (F-pm-24)")
                ent["eos_verified"] = False
            elif last_tok is None or int(last_tok) not in eos_ids:
                c.fail(f"{rid}: stopped after {a_r} < max_tokens {cap} events on token {last_tok}, not a declared EOS id "
                       f"{sorted(eos_ids)[:6]} (driver finish_reason {tr.get('finish_reason')!r}): an early stop without EOS is not modelled (F-pm-24)")
                ent["eos_verified"] = False
            else:
                ent["eos_verified"] = True
                ent["eos_token"] = int(last_tok)
        else:
            ent["finish"] = "length"
        # [LIFTING-SPEC v0.3 s7 [P1] (i), the other direction; R17-1] executed activity may not run PAST the answer by more than the
        # Program's DECLARED lag: executed_prefix_len == a_r + LAG exactly on an EOS stop (an EOS is among the served tokens), == a_r on a
        # length stop (vLLM knows the length ahead: no placeholder row). Under a declared ignore_eos (empty set) there is nothing to pass.
        finish_step_expected = last
        if eos_stop:
            eos_step = served[first_eos][0]
            finish_step_expected = eos_step
            ent["eos_step"] = eos_step
            if eos_lag is None:
                c.fail(f"{rid}: stopped on the declared EOS id {int(served[first_eos][1])} at engine step {eos_step} with {lag_rows} more "
                       f"row(s) executed after it, but the Program declares no LAG ({eos_lag_src}): the post-EOS rows cannot be checked against "
                       f"a declaration -- refuse by name, nothing is inferred from the trace or read from the header as the rule; rebuild the "
                       f"workload Program with the declared async-scheduling specialization (R17-1)")
                ent["eos_verified"] = False
            elif lag_rows > eos_lag:
                c.fail(f"{rid}: execution past EOS -- output event {first_eos} is the declared EOS id {int(served[first_eos][1])} but "
                       f"{lag_rows} more row(s) were executed where the Program's declared scheduler ({eos_lag_src}) computes LAG {eos_lag} after an EOS "
                       f"(executed_prefix_len {len(served)} > served_output_len {a_r} + LAG {eos_lag}): the Program calls the request finished there "
                       f"(LIFTING-SPEC s7 [P1]; R17-1)")
                ent["eos_verified"] = False
                ent["eos_passed_at_event"] = first_eos
            elif lag_rows < eos_lag:
                c.fail(f"{rid}: the Program's declared scheduler ({eos_lag_src}) computes LAG {eos_lag} post-EOS row(s) after an EOS stop but "
                       f"{lag_rows} were executed after the EOS at engine step {eos_step} (executed_prefix_len {len(served)} < served_output_len {a_r} + "
                       f"LAG {eos_lag}): a deleted step -- the record is not one execution of the declared engine (R17-1)")
                ent["eos_verified"] = False
            else:
                for k in range(a_r, len(served)):
                    s_k, t_k = served[k]
                    # the lag row's own sample is fed nowhere: no later row of this request at the position after it
                    pos_k = [pos for s, _t, pos in events if s == s_k and pos >= LP - 1][0]
                    fed_later = [s for s in ps if s > s_k and any(r == rid and p == pos_k + 1 for r, p in steps[s]["rows"])]
                    if fed_later:
                        c.fail(f"{rid}: the lag row at engine step {s_k} (after the EOS at {eos_step}) sampled {int(t_k)} which is fed to the "
                               f"request's row at position {pos_k + 1} in engine step {fed_later[0]}: not a placeholder row (M-0213)")
                        ent["eos_verified"] = False
                    if gen is not None and len(list(gen)) > k and int(list(gen)[k]) == int(t_k):
                        ent["eos_verified"] = False   # already failed above (gen longer than a_r): the leak is named there
        elif lag_rows:
            # a length stop with rows past the cap (a_r = cap < executed): already named 'execution past cap' above; the explicit rule for the record
            c.fail(f"{rid}: {lag_rows} row(s) executed after a length stop: the scheduler computes no placeholder row when the cap ends the request "
                   f"(M-0213; R17-1: LAG applies to EOS stops only)")
        if ent.get("eos_verified") and eos_lag:
            ent["eos_lag_rows_accepted"] = lag_rows
        # the driver's finish_step names the step the request FINISHED at: the EOS step on an EOS stop (the lag row, if any, comes after
        # it -- pad's fixture M-0162: finish_step 10, EOS at 10, lag row at 11), the last row on a length stop
        if tr.get("finish_step") is not None and int(tr["finish_step"]) != finish_step_expected:
            c.fail(f"{rid}: driver finish_step {tr['finish_step']} != observed {finish_step_expected}"
                   + (f" (the EOS step; the request's last served row is the lag row at {last})" if finish_step_expected != last else ""))
        # each event feeds the request's next input row
        fed = 0
        for (s, t, pos), s_next in zip(events, ps[1:] if ps else []):
            rows_next = steps[s_next]["rows"]
            toks_next = steps[s_next]["input_token_ids"]
            k = [i for i, (r, p) in enumerate(rows_next) if r == rid and p == pos + 1]
            if pos + 1 < LP:
                continue  # the next row is still prompt (chunked prefill): the chunk's sample is discarded
            if not k:
                c.fail(f"{rid}: no input row at position {pos + 1} in engine step {s_next} for the token sampled at step {s}")
            elif toks_next and toks_next[k[0]] != t:
                c.fail(f"{rid}: token sampled at step {s} ({t}) != input row token at step {s_next} ({toks_next[k[0]]})")
            else:
                fed += 1
        ent["events_fed_forward"] = fed
        for lk in rec["links"]:
            if lk["request_id"] == rid and lk.get("relation") == "eq" and lk.get("value") is not None:
                ev = [t for s, t, _ in events if s == lk["src_step"]]
                if ev and ev[0] != lk["value"]:
                    c.fail(f"{rid}: StepLink value {lk['value']} at step {lk['step']} != sampled {ev[0]} at src_step {lk['src_step']}")
        if ent.get("T_observed") is not None and ent["T_observed"] != len(served) - 1:
            c.fail(f"{rid}: fold T_observed {ent['T_observed']} != output events - 1 ({len(served) - 1})")
        ent["generated_length"] = len(served)
        # [rev-pm M-0622 TIP (a)] eos_verified is a verdict on THIS request's G6 evidence: any G6 problem on the request (an output
        # association failure -- SampleEnd token != driver token -- a chronology mismatch, ...) makes it False, never True beside a FAIL
        if len(c.problems) > _n_problems_before and ent.get("eos_verified") is True:
            ent["eos_verified"] = False
            ent["eos_verified_note"] = f"{len(c.problems) - _n_problems_before} G6 problem(s) on this request: the EOS evidence is not accepted"
            ent.pop("eos_lag_rows_accepted", None)   # [R17-1] no lag row is 'accepted' on a request whose EOS evidence is not
        ent["g6_problems"] = len(c.problems) - _n_problems_before   # [R17 stoch] the scheduled workload state is built from established requests only
    # achieved concurrency (BRIEF §11: a configured limit is not proof the run exercised that many simultaneous requests): the
    # observed number of distinct requests with rows per engine step, against the programmed one from the declaration alone
    # (members with a_r <= e <= a_r + cap_r - 1); observed < programmed is explained only by EOS-truncated requests
    obs_per_step = {s: len({r for r, _ in steps[s]["rows"]}) for s in steps}
    prog_per_step: dict[int, int] = collections.defaultdict(int)
    for g in gp["requests"]:
        for e in range(int(g["arrive_step"]), int(g["arrive_step"]) + int(g["max_tokens"])):
            prog_per_step[e] += 1
    obs_max = max(obs_per_step.values()) if obs_per_step else 0
    prog_max = max(prog_per_step.values()) if prog_per_step else 0
    conc = {"programmed_max": prog_max, "observed_max": obs_max, "declared_requests": len(gp["requests"]),
            "observed_steps_at_max": sum(1 for v in obs_per_step.values() if v == obs_max),
            "observed_histogram": dict(sorted(collections.Counter(obs_per_step.values()).items())),
            "programmed_histogram": dict(sorted(collections.Counter(prog_per_step.values()).items())),
            "engine_steps_observed": len(steps), "engine_steps_programmed": len(prog_per_step)}
    # [F-pm-24] the shortfall is explained only by EOS-VERIFIED truncation: an unverified early stop does not unlock it
    trunc = [e for e in res["requests"] if e.get("eos_truncated") or (e.get("finish") == "eos")]
    if obs_max != prog_max and not (trunc and all(e.get("eos_verified") is True for e in trunc)):
        c.fail(f"achieved concurrency {obs_max} != programmed {prog_max} with no EOS-verified truncated request "
               f"({len(trunc)} truncated, {sum(1 for e in trunc if e.get('eos_verified') is True)} EOS-verified): declared schedule not exercised")
    # [COORD M-0571 (1)] per ENGINE STEP: the waiver applies only to steps AFTER a verified EOS. A request that reached its cap or
    # stopped on a verified EOS occupies exactly the steps it was observed in; one that stopped early WITHOUT a verified EOS was
    # programmed to continue for (cap - served) more steps -- those steps are named here as "declared schedule not exercised"
    expected_per_step: dict[int, int] = collections.defaultdict(int)
    for e in res["requests"]:
        rid = e["request_id"]
        ps_r = present.get(rid) or []
        occ = set(ps_r)
        if e.get("finish") == "eos" and e.get("eos_verified") is not True and ps_r:
            short = int(e.get("max_tokens") or 0) - int(e.get("output_events") or 0)
            occ.update(range(ps_r[-1] + 1, ps_r[-1] + 1 + max(0, short)))
        for st in occ:
            expected_per_step[st] += 1
    bad_steps = [st for st in sorted(set(expected_per_step) | set(obs_per_step)) if obs_per_step.get(st, 0) != expected_per_step.get(st, 0)]
    if bad_steps:
        culprits = [e["request_id"] for e in res["requests"] if e.get("finish") == "eos" and e.get("eos_verified") is not True]
        c.fail(f"engine steps {bad_steps[:8]}{'...' if len(bad_steps) > 8 else ''}: live requests observed "
               f"{[obs_per_step.get(st, 0) for st in bad_steps[:8]]} != programmed {[expected_per_step.get(st, 0) for st in bad_steps[:8]]} -- the shortfall "
               f"is not after a verified EOS ({culprits} stopped early without one): declared schedule not exercised (M-0571 (1))")
    conc["steps_short_of_programmed"] = len(bad_steps)
    conc["verified_eos_steps"] = sorted({int(e["output_event_steps_last"]) for e in res["requests"] if e.get("eos_verified") is True and e.get("output_event_steps_last") is not None})
    # [COORD M-0169 / M-0571 (3)] the top-p split count S is a WORKLOAD-level derived value of the chronology this check established:
    # S_t = SplitsFor_v1(|live(t)|, num_SMs) per engine step; every top-p TokenSelect of the fold at t must carry it (the launch's S,
    # host-computed by vLLM -- no kernel, no fold instance of its own). Nothing to check on a greedy fold; a top-p fold against a Program
    # without a declared num_SMs fails closed.
    # [R17 stoch, BRIEF §6 / COORD R17-1] |live(t)| is the SCHEDULED workload state -- declared arrivals, served lengths (the cap, or the
    # verified EOS position), the declared lag rows after an EOS stop (the post-EOS step is active computation: its row is in the sampler
    # batch) and the prefill chunks -- derived from the per-request facts established above; the record's logits rows and the SampleEnd
    # batch must equal it at every step. The observed row count is never the source of |live(t)|: an unscheduled sampler row would
    # otherwise inflate the batch and S would follow it. The live->S transitions and their causes go on the record.
    scheduled = scheduled_live_counts(res["requests"], present)
    geo = sampler_geometry_check(gp, B, engine_step_of, steps, rec.get("sampler_batch"), scheduled)
    for p in geo["problems"]:
        c.fail(p)
    # [R17-1] the lag finding of the record as a whole: exercised iff some request stopped on an EOS before its cap; VARIABLE lag (two
    # EOS-stopped requests of ONE record with different post-EOS row counts) is evidence the profile is not the constant-lag scheduler
    # the Program declares -- surfaced by name (a G6 FAIL + res.alerts entry; the lane posts ALERT + a DECISIONS-PENDING entry), never
    # absorbed into whichever value most requests show
    li = c.info["lag"]
    li["exercised"] = bool(_lag_obs)
    li["eos_stop_requests"] = sorted(_lag_obs)
    li["lag_rows_observed"] = dict(sorted(_lag_obs.items()))
    distinct = sorted(set(_lag_obs.values()))
    li["variable_lag"] = len(distinct) > 1
    if li["variable_lag"]:
        c.fail(f"VARIABLE LAG: the EOS-stopped requests of this record show different post-EOS row counts {li['lag_rows_observed']} under one "
               f"declared scheduler (Program LAG {decl_lag}, record attests {rec_lag_src}): not a constant-lag profile -- surface (ALERT + "
               f"DECISIONS-PENDING), never fold into a constant-lag model (R17-1)")
        res.setdefault("alerts", []).append({"kind": "variable_lag", "check": "G6", "lag_rows_observed": li["lag_rows_observed"], "declared": decl_lag,
                                             "record_attests": rec_lag, "action": "ALERT + DECISIONS-PENDING entry (R17-1)"})
    if not lag_declared and li["exercised"]:
        li["note"] = "pre-R17-1 Program (no lag declared) with EOS stops: every EOS-stopped request failed by name above; rebuild with the declared lag"
    elif not lag_declared:
        li["note"] = ("pre-R17-1 Program (no lag declared); no request stopped before its cap, so the lag parameter was not exercised on this record "
                      "-- the record is labelled, not accepted as a lag-N record")
    c.info.update({"arrivals_by_idle_rule": idle_rule, "step_links": len(rec["links"]), "tokens_json": bool(tokens), "achieved_concurrency": conc,
                   "generated_lengths": {e["request_id"]: e.get("generated_length") for e in res["requests"]}, "sampler_geometry": geo["info"],
                   "scheduled_live": {"max_live": scheduled["max_live"], "transitions": scheduled["transitions"],
                                      "unestablished_steps": len(scheduled["unestablished"]),
                                      "requests": scheduled["requests"]},
                   "lengths_by_request": {e["request_id"]: {"served_output_len": e.get("served_output_len"), "executed_prefix_len": e.get("executed_prefix_len"),
                                                            "cap": e.get("max_tokens"), "cap_exhausted": e.get("cap_exhausted")} for e in res["requests"]}})
    res["achieved_concurrency"] = conc
    res["sampler_geometry"] = geo["info"]
    res["scheduled_live"] = c.info["scheduled_live"]
    res["lag"] = dict(li)   # [R17-1] the finalizer's fields: declared / record_attests / exercised / variable_lag / lag_rows_observed / lengths
    c.done()
    mark("g6")

    # ---------------------------------------------------------------- G7: VU population + boundary probes
    c = _Check(res, "G7", "vu_population: global query population (components over executed steps) == attributed fold; boundary probes resolve")
    attributed = sum(len(v) for v in per_req.values())
    if exec_pop != attributed:
        c.fail(f"global query exposes {exec_pop} VUs over the executed steps but the fold holds {attributed} attributed instances")
    probes = failures = 0
    loc2glob = {v: k for k, v in glob2loc.items()}
    by_step: dict[int, list[int]] = collections.defaultdict(list)
    for gi, s in engine_step_of.items():
        by_step[s].append(gi)
    for s in sorted(by_step):
        gis = sorted(by_step[s])
        owners = []
        block_last: list[int] = []
        if fast:   # [R17 mprof] same blocks; the last attributed instance of each block kept while building them (O(1) per boundary below)
            owners, block_last = GMF.g7_owner_blocks(gis, glob2loc)
        else:
            for gi in gis:
                o = glob2loc.get(gi)
                if o is None:
                    continue
                if not owners or owners[-1][0] != o[0]:
                    owners.append((o[0], gi))
        active = {o for o, _ in owners}
        if len(active) < 2:
            continue
        # every boundary between consecutive owner blocks in emission order: probe the last VU before and the first after
        for bi, ((oa, ga), (ob, gb)) in enumerate(zip(owners, owners[1:])):
            before = block_last[bi] if fast else max(g for g in gis if g < gb and glob2loc.get(g, (None,))[0] == oa)
            for gi in (before, gb):
                probes += 1
                loc = glob2loc[gi]
                rid, ls, lj = loc
                back = loc2glob.get(loc)
                A = comps.get(rid)
                ok = back == gi and rid in per_req and A is not None and ls < len(A.steps) and lj < len(A.steps[ls]) \
                    and A.inst[A.steps[ls][lj]]["spec"] == B.inst[gi]["spec"] and (meta_of.get(B.inst[gi]["k"]) or {}).get("request") in (None, rid)
                if not ok:
                    failures += 1
                    if failures <= 10:
                        c.fail(f"probe engine step {s} node {B.inst[gi]['k']}: address {loc} does not resolve to its owner/spec")
    c.info.update({"population_query": exec_pop, "population_fold": attributed, "unexercised_declared": unexercised,
                   "boundary_probes": probes, "boundary_probe_failures": failures})
    c.done()
    mark("g7")

    # ---------------------------------------------------------------- G8: within-step order (M-ARRIVAL-ORDER)
    c = _Check(res, "G8", "within_step_order: each request's own instances in its component order inside every engine step; cross-request interleaving = recorded runtime batch layout"
               if within_step == "sequence" else
               "within_step_order: emission order inside an engine step recorded as batch layout (canonical addresses are structural, DECISION M-0021); "
               "dependency order is G3's (a consumer before its producer is DAG-unequal)")
    layout = []
    nonmono_steps = {}
    for s in sorted(by_step):
        gis = sorted(by_step[s])
        seq = [glob2loc.get(gi) for gi in gis]
        reqs_in_rows = list(dict.fromkeys(r for r, _ in steps.get(s, {}).get("rows", [])))
        emission_reqs = list(dict.fromkeys(o[0] for o in seq if o))
        last_j: dict[tuple[str, int], int] = {}
        nonmono = 0
        for o in seq:
            if o is None:
                continue
            rid, ls, lj = o
            k = (rid, ls)
            if k in last_j and lj < last_j[k]:
                nonmono += 1
            last_j[k] = lj
        if nonmono:
            nonmono_steps[s] = nonmono
            if within_step == "sequence":
                c.fail(f"engine step {s}: {nonmono} instances emitted out of their request-local order")
        if len(reqs_in_rows) > 1:
            layout.append({"engine_step": s, "batch_rows_order": reqs_in_rows[:8], "emission_first_seen": emission_reqs[:8],
                           "phase": "mixed" if any(p < G[r]["prompt_len"] for r, p in steps[s]["rows"] if r in G) and
                           any(p >= G[r]["prompt_len"] for r, p in steps[s]["rows"] if r in G) else "uniform",
                           **({"emitted_out_of_canonical_order": nonmono} if nonmono else {})})
    # [rev-pm M-0171 TIP] a request's local steps are segmented at ITS TokenSelects (compare.Prog._segment, the rc3 rule); the fold rows
    # carry the engine step of every instance.  The two must agree: a local step lies inside ONE engine step, and one engine step holds
    # at most one TokenSelect-terminated local step of the request (its sampler is the last of its instances there), in increasing
    # engine-step order.  A mis-emitted sampler is named here (it would otherwise re-segment the request and surface as unrelated counts).
    seg_bad = []
    for ent in res["requests"]:
        rid = ent["request_id"]
        if rid not in proj:
            continue
        P_, gl = proj[rid], per_req[rid]
        closed: dict[int, int] = {}      # engine step -> local step whose TokenSelect closed it
        prev_es = None
        for s_, loc in enumerate(P_.steps):
            ess = sorted({engine_step_of[gl[i]] for i in loc if gl[i] in engine_step_of})
            has_ts = bool(loc) and P_.inst[loc[-1]]["spec"].startswith("TokenSelect_v1")
            if len(ess) > 1:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess, "problem": "the local step spans engine steps"})
            elif ess and prev_es is not None and ess[0] < prev_es:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess, "problem": f"engine step order decreases (after {prev_es})"})
            elif ess and has_ts and ess[0] in closed:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess,
                                "problem": f"second TokenSelect of the request in engine step {ess[0]} (local step {closed[ess[0]]} already ended there)"})
            if ess and has_ts:
                closed.setdefault(ess[-1], s_)
            if ess:
                prev_es = ess[-1]
    if seg_bad:
        c.fail(f"{len(seg_bad)} request-local step(s) whose TokenSelect segmentation disagrees with the fold rows' engine steps "
               f"(first: {seg_bad[0]}) -- a mis-emitted sampler, not a layout")
    c.info["segmentation"] = {"rule": "request-local steps = the request's instances segmented at its TokenSelects (compare.Prog._segment); "
                                      "cross-checked against the engine step the fold rows record for every instance",
                              "disagreements": seg_bad[:6]}
    # a request whose step is a permutation of its component's (same DAG multiset, different sequence) -- characterise it: under
    # `sequence` it is the G3 failure (first_divergence); under `dag` it is the recorded layout of every step with a permutation
    perms = []
    perm_targets = []
    # dependency order inside the fold's emission: every producer before its consumer.  Under `sequence` a violation is already a G3
    # sequence/wiring failure; under `dag` G8 must FAIL on it explicitly (DECISION M-0021 (4): "every producer is emitted before its
    # consumer" -- the recorded permutation is a layout only while the dependency order holds).
    # `dep_viol`: collected per request by `leg` (the same scan over every request's projection -- P.canon(i) of every instance of every
    # local step, a node source whose producer is emitted at or after its consumer -- in request order, right where the canonical
    # sources were resolved); G8 names it here
    if dep_viol:
        c.fail(f"{len(dep_viol)} consumer-before-producer emissions inside a request step (first: {dep_viol[0]}) -- not a layout permutation")
    c.info["dependency_order_violations"] = len(dep_viol)
    for ent in res["requests"]:
        rid = ent["request_id"]
        if rid not in proj or rid not in comps:
            continue
        if within_step == "sequence":
            if ent.get("executed_steps_equal") is False:
                perm_targets.append((ent, rid, (ent.get("first_divergence") or {}).get("step")))
        else:
            A_, P_ = comps[rid], proj[rid]
            for s_ in range(min(len(P_.steps), len(A_.steps))):
                if any((glob2loc.get(per_req[rid][il]) or shared_loc.get(per_req[rid][il], {}).get(rid) or (None, None, None))[2] != pos
                       for pos, il in enumerate(P_.steps[s_])):
                    perm_targets.append((ent, rid, s_))
    for ent, rid, s in perm_targets[:32]:
        A, P = comps[rid], proj[rid]
        if not _perm_eligible(A, P, s):
            continue
        # Merkle (order-insensitive) matching of the step's instances: B instance -> A index with the same DAG hash -- the leg's
        # matching when it ran (F-r17-07: computed where the hashes were), else this process's hashes on both sides
        pi = (perm_facts.get(rid) or {}).get(s)
        if pi is None:
            pi = _perm_match(A, P, s, ops.dag_hashes(A), ops.dag_hashes(P))
        same_dag = all(x is not None for x in pi)
        moved = [(j, x) for j, x in enumerate(pi) if x != j]
        es = engine_step_of.get(per_req[rid][P.steps[s][0]])
        # which computations were displaced: the specs of the A instances that moved by more than the induced shift, with their
        # Attention T (= query position + 1) against the positions of the OTHER requests' rows in the same engine step
        offs = collections.Counter(x - j for j, x in moved if x is not None)
        big = [(j, x) for j, x in moved if x is not None and abs(x - j) > 2]
        displaced = collections.Counter(A.inst[A.steps[s][x]]["spec"] for _, x in big)
        t_re = re.compile(r"\{T=(\d+)")
        disp_pos = sorted({int(m.group(1)) - 1 for _, x in big for m in [t_re.search(A.inst[A.steps[s][x]]["spec"])] if m})
        others = sorted({p for r_, p in steps.get(es, {}).get("rows", []) if r_ != rid}) if es is not None else []
        if within_step == "sequence":
            statement = ("the request's own instances of this step are exactly its component's computations (Merkle-identical DAG multiset) "
                         "emitted in a different order: the displaced instances are moved to the head of each layer's attention list; the "
                         "request-local order is prescribed by the component Program, the emission order of a prefill co-scheduled with other "
                         "requests' rows is a property of the executed kernel/fold, not derivable from the declared workload -> refused, "
                         "not re-sorted (M-ARRIVAL-ORDER)" if same_dag else
                         "the request's instances of this step are NOT a permutation of the component's (different DAG): a structural difference")
        else:
            statement = ("emission order of the request's instances inside this engine step differs from its component's canonical layout "
                         "(fold encoding: AttentionScope groups the batch's rows by distinct T in first-occurrence order); the instances, their "
                         "wiring and dependency order are the component's (Merkle-identical), each is addressed by its canonical component index; "
                         "recorded as batch layout, not re-sorted (DECISION M-0021)" if same_dag else
                         "NOT a permutation of the component's step (different DAG): reported by G3")
        perms.append({"request_id": rid, "local_step": s, "engine_step": es, "n": len(pi), "moved": len(moved), "same_dag_multiset": same_dag,
                      "unmatched": sum(1 for x in pi if x is None), "offset_histogram": dict(offs.most_common(6)),
                      "displaced_specs": dict(displaced.most_common(6)), "displaced_query_positions": disp_pos[:16],
                      "coscheduled_other_positions": others[:16], "permutation_head": pi[:8], "statement": statement})
        if within_step == "sequence":
            c.fail(f"{rid}: local step {s} (engine step {es}) is a permutation of its component's step ({len(moved)}/{len(pi)} moved; displaced query "
                   f"positions {disp_pos[:6]} vs co-scheduled others' positions {others[:6]})" if same_dag
                   else f"{rid}: local step {s} differs structurally from its component ({sum(1 for x in pi if x is None)} unmatched)")
    c.info.update({"shared_steps": len(layout), "batch_layout": layout[:6] + ([{"...": len(layout) - 6}] if len(layout) > 6 else []),
                   "steps_with_emission_out_of_canonical_order": len(nonmono_steps),
                   "order_source": "request-local canonical order: the component Program; cross-request row interleaving and (under dag) the "
                                   "emission order: runtime batch layout / fold encoding (StepStart.rows), recorded not prescribed",
                   "permutations": perms})
    c.done()
    mark("g8")

    failed = [k["id"] for k in res["checks"] if k["status"] != "PASS"]
    res["verdict"] = "PASS" if not failed else "FAIL"
    res["failed_checks"] = failed
    res["reasons"] = [p for k in res["checks"] for p in k["problems"][:3]]
    res["eos_truncated_requests"] = [e["request_id"] for e in res["requests"] if e.get("eos_truncated")]
    # [F-pm-24 / report M-0554] per truncated request: True only when served[-1] is a declared EOS id (the finalizer reads this)
    res["eos_verified"] = {e["request_id"]: e.get("eos_verified") for e in res["requests"] if "eos_verified" in e}
    res["seconds"] = round(time.time() - t0, 1)
    # [COORD M-0177] both within-step verdicts stay in the artifact: the OTHER criterion is run on the same record (X-09 not repeated;
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
                                                f"({within_step}); COORD ruling M-0177 / M-0031"}
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
    """[R17 mfast, COORD R17-7.2] The checker implementation `MATCH_IMPL` selects: `fast` (`cb_a.global_match_fast`: the same criteria
    with compute-once memoisation, compact operand progressions and forked per-request legs -- the Match of record since R17
    (`sweep/row_pod.sh` exports it for every row) and the CLI's default) or `record` (this module's un-memoised single-process body:
    the audit checker the differential harness compares against).  Both expose `check(gp, oracle_dir, log_path, tokens_path, run_x09,
    within_step, record_alternate)` with the same result schema; the differential harness (`verity_vllm.pipeline.match_diff`) runs
    both on the same artifacts and compares verdict, per-check outcomes, populations and mismatch locations.  An unknown name is an
    error, never a silent fallback to either implementation.

    [lane/vllm-v2-gm-scale] The default moved from `record` to `fast`: a `global_match.py` invoked outside `row_pod.sh` (a by-hand rerun,
    another launcher) used to get the audit checker silently -- on the B64 llama32-1b row that is 8+ h single-core at 40-50 GB against
    ~4 min on 8 cores at 13.5 GB for the same verdict.  `MATCH_IMPL=record` still selects the audit checker explicitly."""
    name = (name if name is not None else os.environ.get(IMPL_ENV, IMPL_DEFAULT)).strip().lower() or IMPL_DEFAULT
    if name == "record":
        import sys
        return sys.modules[__name__]
    if name == "fast":
        from verity_vllm.check.match import global_match_fast
        return global_match_fast
    raise ValueError(f"{IMPL_ENV}={name!r}: not one of {IMPLS}")


def impl_banner(mod, gp: dict | None = None) -> str:
    """One stderr line stating, before the run, which implementation is about to run and how (workers, operand form) and which
    component dirs were relocated -- the two facts whose absence cost the v2 acquisition lane an 8 h run (record checker by default,
    a Build's dangling component paths noticed only at load)."""
    name = getattr(mod, "IMPL_NAME", "record")
    env = os.environ.get(IMPL_ENV)
    parts = [f"# global_match impl={name}", f"({IMPL_ENV}={env!r})" if env is not None else f"({IMPL_ENV} unset -> default {IMPL_DEFAULT})"]
    if name == "fast":
        parts.append(f"workers={GMF.workers(1 << 30)} ({GMF.WORKERS_ENV}={os.environ.get(GMF.WORKERS_ENV) or 'default'})")
        parts.append(f"compact_args={os.environ.get(GMF.COMPACT_ENV, '1').strip() != '0'}")
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
        digest = _sha256_file(f)
    info = {"name": getattr(mod, "IMPL_NAME", "record"), "module": mod.__name__, "file": os.path.basename(f) if f else None,
            "source_sha256": digest, "candidate_stage": getattr(mod, "CANDIDATE_STAGE", None),
            # [F2, rev M-0500 (2) / S-r17-30 n1] EVERY document -- the record checker's as much as the fast one's -- states, per fold directory
            # it loaded, the source (the record files, nothing beside them -- F-r17-12) and the STORED form read (instances.jsonl schema +
            # operand_refs_form), so a record document of a v1 store and one of a v2 store of the same capture are distinguishable by content
            "fold_load": dict(C.LOAD_NOTES)}
    if getattr(mod, "IMPL_NAME", "record") == "fast":   # memo hit/miss counters + worker count: telemetry beside the verdict
        # [rev F-r17-09] after check() returned its caches are released (unit 4) -- the outermost run's counters live in last_stats
        info["stats"] = dict(GMF.last_stats) if GMF.last_stats else {k: int(v) for k, v in sorted(GMF.stats.items())}
        # [rev S-r17-17 condition] the DECLARED configuration (workers cap, operand form, switches as read), beside the counters
        info["config"] = dict(GMF.config)
        # the same fold_load under the DECLARED configuration too: the fast documents of record (int-2..int-5) named it there; readers keep working
        info["config"]["fold_load"] = dict(C.LOAD_NOTES)
    return info


@dataclass(frozen=True, kw_only=True)
class Options:
    oracle: str = option("--oracle", help="Match dir: program.json + instances.jsonl + capture/log.jsonl.gz")
    global_program: str | None = option("--global-program", default=None, help="cb-a/global-program/v1 JSON; omit to assemble the mock from --workload + --component")
    gprog: str | None = option("--gprog", default=None, help="lane gprog's workload_program.json (GP-01) -> converted with from_gprog")
    workload: str | None = option("--workload", default=None, help="workload manifest (for the mock global Program)")
    component: list[str] = option("--component", action="append", default=[], help="request component dir (derive_step output); repeatable")
    log: str | None = option("--log", default=None)
    tokens: str | None = option("--tokens", default=None)
    tokens_optional: bool = option("--tokens-optional", default=False, action="store_true", help="accept a record without capture/tokens.json (G6 output-association legs skipped, " "LABELLED checks[G6].tokens_json=false; mechanism/fixture runs only, never a row of record) [F-r17-01]")
    no_x09: bool = option("--no-x09", default=False, action="store_true")
    within_step: str = option("--within-step", choices=WITHIN_STEP_CRITERIA, default=DEFAULT_WITHIN_STEP, help="within-engine-step correspondence of a request's instances: dag (default of record, COORD ruling M-0177/M-0031) | " "sequence (the r14-v1 criterion); the other criterion's verdict is recorded beside it (alternate_verdict)")
    no_alternate: bool = option("--no-alternate", default=False, action="store_true", help="do not run/record the other criterion's verdict beside the verdict of record")
    out: str | None = option("--out", default=None)
    decomp_out: str | None = option("--decomp-out", default=None, help="[COORD R17-28, P1] also write the X-09 batch decomposition document here (cb_a.batch_decomp's CLI " "schema, the same decompose() call whose verdict is this document's x09 block): row_pod's match_decomp.json " "without a second batch_decomp process over the same fold and components. Exit 4 when X-09 names " "missing derived shapes (batch_decomp's protocol: derive them and rerun), else GM-01's 0 / 3.")
    derived: list[str] = option("--derived", action="append", default=[], help="[P1] derived request-wrapper dir X-09 may use beside the global Program's " "components (repeatable): the shapes row_pod derived after an exit 4")


def main(a: Options) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    if a.decomp_out and a.no_x09:
        raise UsageError("--decomp-out needs the X-09 run (drop --no-x09)")
    loaded = {X09_EXTRA_DERIVED: list(a.derived)}   # shared with check(): the fold / components decoded once + the X-09 document (P1)
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
                    tokens_optional=a.tokens_optional, _loaded=loaded)   # [rel r17-int-2 merge: stoch F-r17-01 --tokens-optional + mfast MATCH_IMPL dispatch; mprof P1 _loaded]
    res["impl"] = impl_info(mod)
    if (a.decomp_out or a.derived) and isinstance(res.get("x09"), dict) and loaded.get(X09_DOCUMENT) is not None:
        # [rev S-r17-27 n1] a document produced under the shared pipeline (P1) states what its X-09 ran over (batch_decomp's own keys) and
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
