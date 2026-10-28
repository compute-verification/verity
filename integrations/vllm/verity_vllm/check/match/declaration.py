"""The declared side of GM-01: the global workload Program (the Build's GP-01 artifact, or the mock assembled from the manifest), the
execution conditions the record attests, and G1 (requests_declared) comparing the two with the record header."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from verity_vllm.config import workload_file
from verity_vllm.observe import capture_log as CL
from verity_vllm.check.match import derived_index as DI
from verity_vllm.check.match.match_run import MatchRun, NamedCheck
from verity_vllm.program import prefix_plan as PP
from verity_vllm.records.fields import RecordError
from verity_vllm.records.workload import Workload

GP_SCHEMA = "cb-a/global-program/v1"


def request_id_of(index: int) -> str:
    """`r<k>` -- the same function as global_program.request_id_of (kept local: this module must not import torch paths)."""
    return f"r{int(index)}"


# ------------------------------------------------------------------------------------------------------ the global Program


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def component_shape(d: str) -> tuple[int, ...] | None:
    """(LP, T[, C]) of a component dir: a derive_step dir (`construction_manifest.json`) or a fold record standing in for one."""
    return DI.derived_shape(d)


_VARIANT_RE = re.compile(r"^stoch-t(?P<t>[0-9.]+)-p(?P<p>[0-9.]+)(?:-k(?P<k>[0-9]+))?$")


def manifest_sampling(wl: Workload | Mapping[str, Any], variant: str | None = None) -> dict:
    """The manifest's declared sampling policy as the header records it.  SOURCES, all read, none trusted alone:
    (1) `variants[variant]` -- the table the DRIVER consumes (vllm_adapter.load_workload) and the header records: the value of record;
    (2) the variant LABEL (greedy | stoch-t<T>-p<P>[-k<K>]) parsed; (3) explicit `sampling.{temperature,top_p,top_k}`; (4) `sweep.sampling_params`.
    When two sources disagree on a key the result carries `conflict` (G1 fails closed, named): a manifest whose label says p0.95 while
    its table -- the one the run used -- says 0.9 is not one declaration.  `variant` overrides the workload-level selection (per-request pin)."""
    w = Workload.of(wl, "workload")
    sweep = w.sweep or {}
    v = variant or w.sampling_variant or sweep.get("sampling")
    per_key: dict[str, dict[str, object]] = {}   # key -> {source: value}
    def put(src: str, k: str, val) -> None:
        if val is not None:
            per_key.setdefault(k, {})[src] = val
    table = w.variants.get(v) if isinstance(v, str) else None
    if table is not None:
        for k in ("temperature", "top_p", "top_k"):
            put("variants_table", k, table.get(k))
    if v == "greedy":
        put("label", "temperature", 0.0)
    elif isinstance(v, str) and _VARIANT_RE.match(v):
        m = _VARIANT_RE.match(v)
        put("label", "temperature", float(m["t"])); put("label", "top_p", float(m["p"])); put("label", "top_k", int(m["k"] or 0))
    if variant is None:
        s = w.sampling
        explicit = {} if s is None else {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k}
        for src, d in (("sampling", explicit), ("sweep.sampling_params", sweep.get("sampling_params", {}))):
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
    if isinstance(v, str) and v != "greedy" and table is None:
        conflict.append(f"variant {v!r} is not in the manifest's `variants` table (the driver refuses such a manifest)")
    if conflict:
        out["conflict"] = conflict
    out["sources"] = {k: sorted(srcs) for k, srcs in per_key.items()}
    return out


def mock_global_program(workload_path: str, component_dirs: list[str], eos_token_ids: list[int] | None = None, lag: int | None = None) -> dict:
    """The rooted workload Program's INTERFACE assembled from the declared workload and the Build's request components, so the
    global check can run without a built `workload_program.json`.  It reads the manifest and the components only -- never the record.
    `lag` = the declared scheduler's post-EOS rows (0/1) the mock Program is specialized for; None = an older Program that
    declares no lag (G6 then fails by name on any EOS stop)."""
    wl = Workload.load(workload_file(workload_path))
    comps = {}
    for d in component_dirs:
        sh = component_shape(d)
        if sh is not None:
            comps.setdefault(sh, d)
    reqs = []
    for i, r in enumerate(wl.requests):
        if r.prompt_token_ids is None:
            raise RecordError(wl.where, f"requests[{i}].prompt_token_ids", "is absent (a lengths-only plan): the Program declares each prompt")
        lp, cap = len(r.prompt_token_ids), r.max_tokens
        T = cap - 1
        reqs.append({"index": i, "request_id": request_id_of(i), "workload_request_id": r.request_id, "prompt_len": lp, "max_tokens": cap, "T_bound": T,
                     "arrive_step": wl.arrive_step(i),
                     "prompt_sha256": hashlib.sha256(json.dumps(list(r.prompt_token_ids)).encode()).hexdigest()[:16],
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
        stop_rule.update({"eos_lag_rows": int(lag), "lag": int(lag), "eos_lag_source": "mock_global_program(lag=...) (declared static)"})
        lag_block = {"value": int(lag), "async_scheduling": bool(lag), "source": "mock_global_program(lag=...)", "declared_explicitly": True}
    return {"schema": GP_SCHEMA, "builder": "cb_a.global_match.mock_global_program (manifest + Build components; the Build's rooted Program replaces this)",
            "workload": {"path": workload_path, "sha256": sha256_file(workload_path), "n_requests": len(reqs)},
            # the mock models GP-01's eager conditions; G1 fails closed on a Program without a declaration
            "applicability": {"execution": "bi-eager", "row_execution": "bi-eager", "engine": engine, "refusal": None},
            # the stop rule: the EOS ids an early stop must end in (None = not declared -> G6 fails closed on early stops)
            "stop_rule": stop_rule,
            "lag": lag_block,   # the declared LAG block (None = the Program declares none)
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
    """The `lag` static of the workload Program AS THE CODEC WROTE IT, read from the descriptor (both places are inside the digest
    program_digest verifies): the codec inlines a specialization's binding record into its spec id in static-parameter order
    (`Workload_v1{requests=[...],shared=[...],lag=1}`, verity.ml.circuits.codec._spec_id), so `desc["program_fn"]` ends in `,lag=<n>}` and the
    root id `Root[Workload_v1{...,lag=<n>}]_v1` carries the same suffix; the two must agree.  A descriptor whose root definition carries an
    explicit `bindings.lag` (the mock form) is read from there.  None = a Program that declares no lag (every Build from before the static)."""
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
    # an explicit bindings.lag must agree with the spec-id suffixes when both are present
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
    """The builder's `workload_program.json` (GP-01, `verity/workload-program/v1`) -> this checker's interface.  The rooted Program's
    component dirs come from `inputs.derived` ("lp,t" -> derive_step dir); its address map and chronology are carried through and
    cross-checked (component digests, per-request instance counts, programmed steps).  The manifest it was built from
    (`workload_path`, else `inputs.manifest`) is read through the workload record, and one that declares no prompt for a request (a
    lengths-only plan) is refused by name."""
    art = json.load(open(artifact_path))
    if not art.get("ok", True):
        raise ValueError(f"{artifact_path}: builder result not ok ({art.get('reason')}: {art.get('missing_derived')})")
    # the artifact's `workload_digest` field is a CLAIM: recompute it from the descriptor GP-01 wrote beside the
    # artifact (descriptor.json.gz) and refuse a mismatch, so a digest edited/zeroed in the Build's artifact cannot flow into this checker's
    # copy (global_match_global_program.json) and the finalizer's same-attempt rule.  Absent descriptor (an evidence dir pulled without the
    # 17-25 MB file) -> recorded as unverified, never as verified.
    digest_verification = {"verified": False, "why": "descriptor.json.gz not beside the artifact"}
    dpath = os.path.join(os.path.dirname(os.path.abspath(artifact_path)), "descriptor.json.gz")
    if os.path.exists(dpath):
        from verity.ml.circuits.codec import canonical_json, program_digest
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
        # the artifact's `lag` block is a CLAIM beside the Program: the Program's own `lag` static (inside the digest just verified)
        # must carry the same value -- an edited block cannot make G6 consume another lag than the Program's
        static_lag = workload_lag_static(desc)
        claimed = (art.get("lag") or {}).get("value") if isinstance(art.get("lag"), dict) else None
        if static_lag is not None or claimed is not None:
            if static_lag is None or claimed is None or int(static_lag) != int(claimed):
                raise ValueError(f"{artifact_path}: lag.value {claimed!r} != the Program's Workload_v1 `lag` static {static_lag!r} (descriptor "
                                 f"program_fn {str(desc.get('program_fn'))[:40]}...{str(desc.get('program_fn'))[-24:]}, root "
                                 f"{str(desc.get('root'))[:32]}...{str(desc.get('root'))[-24:]}): the artifact's lag declaration is not the Program's")
            digest_verification["lag_static"] = int(static_lag)
    derived = (art.get("inputs") or {}).get("derived") or {}
    relocation = relocate_components(derived, artifact_path)
    derived = relocation["derived"]
    wl_path = workload_path or (art.get("inputs") or {}).get("manifest")
    wl = Workload.load(wl_path) if wl_path and workload_file(wl_path).is_file() else None
    declared = () if wl is None else wl.requests
    for i, q in enumerate(declared):
        if q.prompt_token_ids is None:
            raise RecordError(wl.where, f"requests[{i}].prompt_token_ids", "is absent (a lengths-only plan): the Program declares each prompt")
    # prompts by manifest POSITION (the Program's r<k>); the manifest's own request_id is the workload (UUID) id
    prompts = {request_id_of(i): list(q.prompt_token_ids) for i, q in enumerate(declared)}
    wl_uuids = {request_id_of(i): q.request_id for i, q in enumerate(declared)}
    amap = {r["request_id"]: r for r in (art.get("address_map") or {}).get("requests", [])}
    chron = {r["request_id"]: r for r in (art.get("chronology") or {}).get("requests", [])}
    plan = PP.plan(wl.dump()) if wl is not None else None
    reqs = []
    for r in art["requests"]:
        lp, cap = int(r["prompt_len"]), int(r["cap"])
        rid = r["request_id"]
        if rid != request_id_of(r["index"]):
            raise ValueError(f"{artifact_path}: request {rid!r} at index {r['index']} is not the r<k> convention: "
                             "rebuild the workload Program with the current global_program")
        wrid = r.get("workload_request_id") or wl_uuids.get(rid)
        if wl is not None and wl_uuids.get(rid) is not None and wrid != wl_uuids[rid]:
            raise ValueError(f"{artifact_path}: {rid} carries workload_request_id {wrid!r} but the manifest at position {r['index']} has {wl_uuids[rid]!r}")
        pfx = r.get("prefix")
        p = plan[r["index"]] if plan is not None and r["index"] < len(plan) else None
        if p is not None and pfx != ({"rows": p["cached"], "producers": [request_id_of(q) for q in p["producers"]]} if p["cached"] else None):
            raise ValueError(f"{artifact_path}: {rid} declares the prefix-cache hit {pfx!r}, the manifest's arrival chronology and prompts give "
                             f"{p!r} (program.prefix_plan)")
        C = int((pfx or {}).get("rows") or 0)
        d = derived.get(PP.shape_key(PP.shape(lp, cap - 1, C)))
        reqs.append({"index": r["index"], "request_id": rid, "workload_request_id": wrid, "prompt_len": lp, "max_tokens": cap, "T_bound": cap - 1,
                     "arrive_step": int(r["arrive_step"]), **({"prefix": pfx} if pfx else {}),
                     "prompt_sha256": hashlib.sha256(json.dumps(prompts[rid]).encode()).hexdigest()[:16] if rid in prompts else None,
                     "component": {"dir": d, "LP": lp, "T": cap - 1, **({"C": C} if C else {}), "program_digest": (r.get("component") or {}).get("program_digest"),
                                   "specialization": (r.get("component") or {}).get("specialization")},
                     "global_address": {k: amap[rid].get(k) for k in ("root_node", "gates", "n_instances", "rule")} if rid in amap else None,
                     "programmed_steps": (chron.get(rid) or {}).get("programmed_steps"),
                     "output_interface": {"unit": "engine_step", "max_events": cap, "event": "TokenSelect over the request's logits row"}})
    return {"schema": GP_SCHEMA, "builder": f"gprog GP-01 {art.get('schema')} workload_fn={art.get('workload_fn')}", "workload_digest": art.get("workload_digest"),
            "workload": {"path": wl_path, "sha256": art.get("manifest_digest"), "n_requests": len(reqs)},
            "shared_inputs": [{"name": "weights", "role": "prescribed", "global_param": f"root parameter {art.get('shared_param_index')} (gprog shared)"}],
            "requests": reqs, "chronology": art.get("chronology"), "address_space": art.get("address_map", {}).get("convention"),
            "applicability": art.get("applicability"),
            "stop_rule": art.get("stop_rule"),   # the declared EOS set (None on a pre-fix artifact -> G6 fails closed on early stops)
            "lag": art.get("lag"),               # the declared LAG block (None on an older artifact -> G6 fails by name on EOS stops)
            "lengths": art.get("lengths"),
            "moe_construction": art.get("moe_construction"),   # GP-01's declared-and-bound construction block (None on an older artifact)
            "gprog_artifact": {"path": artifact_path, "gates": art.get("gates"), "input_gates": art.get("input_gates"), "descriptor": art.get("descriptor"),
                               "digest_verification": digest_verification,
                               # component dirs resolved beside the artifact (recorded absolute paths absent here); {} when none moved
                               "relocated_components": relocation["relocated"]}}


# ------------------------------------------------------------------------------------------ the record's engine facts (G1)

ENGINE_FACT_KEYS = ("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill",
                    "async_scheduling")   # G6 reads the scheduler's EOS lag rows from THIS attested fact; no default


def eos_lag_rows(facts: dict) -> tuple[int | None, str]:
    """How many rows the engine computes for a request AFTER its EOS sample.

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
    """The execution conditions the record attests, gathered from EVERY location a recorder writes them:
    the production recorder nests the effective LLM(**kw) under `RunHeader.profile.engine_kwargs` and the compilation facts under
    `RunHeader.profile.compilation` (vllm_adapter build_manifest); older/foreign headers put them at the top level; the run's
    `capture/versions.json` repeats the effective engine kwargs.  Returns {engine: {k: v}, compilation: {fact: v}, values: {k: [distinct v]},
    sources: {k: [locations]}, disagreements: {k: (a, b, src_a, src_b)}, and unreadable: {location: error} only when a location is present
    but unreadable} -- the caller FAILS on any disagreement, on an unreadable location and when nothing is attested."""
    from verity_vllm.pipeline.global_program import compilation_facts
    header = header if isinstance(header, dict) else {}
    prof = header.get("profile") if isinstance(header.get("profile"), dict) else {}
    ek_top = header.get("engine_kwargs") if isinstance(header.get("engine_kwargs"), dict) else {}
    ek_prof = prof.get("engine_kwargs") if isinstance(prof.get("engine_kwargs"), dict) else {}
    ek_ver: dict = {}
    unreadable: dict[str, str] = {}
    if oracle_dir:
        vp = os.path.join(oracle_dir, "capture", "versions.json")
        if os.path.exists(vp):
            try:
                v = json.load(open(vp))
                ek_ver = v.get("engine_kwargs") if isinstance(v.get("engine_kwargs"), dict) else {}
            except Exception as e:  # noqa: BLE001 -- an unreadable versions.json is not an attestation
                unreadable["capture/versions.json"] = repr(e)[:200]
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
    if unreadable:
        out["unreadable"] = unreadable
    return out


def requests_declared(run: MatchRun) -> tuple[dict, list[dict], dict, list[str]]:
    """G1: the declared request set, three ways (manifest, global Program, record header), and the record's engine conditions inside
    the Program's declared applicability.  Returns (the record's engine facts, the header's requests, request -> its prompt parameter,
    the Program's request ids)."""
    res, gp, header, G = run.res, run.gp, run.header, run.declared
    c = NamedCheck(res, "G1", "requests_declared: manifest == global Program == record header; none omitted/duplicated; prompts as declared; "
                             "record engine conditions inside the Program's declared applicability")
    facts = _execution_conditions(c, res, gp, header, run.oracle_dir)
    g_ids = [r["request_id"] for r in gp["requests"]]
    dup_g = [k for k, n in collections.Counter(g_ids).items() if n > 1]
    if dup_g:
        c.fail(f"global Program lists requests more than once: {dup_g}")
    bad_conv = [r["request_id"] for r in gp["requests"] if r["request_id"] != request_id_of(r["index"])]
    if bad_conv:
        c.fail(f"global Program request ids are not r<k> at their manifest index: {bad_conv[:6]}")
    hdr_reqs, param_of, parent_of = CL.request_params(header)
    h_ids = [r["request_id"] for r in hdr_reqs]
    dup_h = [k for k, n in collections.Counter(h_ids).items() if n > 1]
    if dup_h:
        c.fail(f"record header lists requests more than once: {dup_h}")
    wl = _manifest_of_record(c, gp, G, header, hdr_reqs, g_ids)
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
            w = wl.requests[g["index"]] if g["index"] < len(wl.requests) else None
            if w is not None and w.prompt_token_ids is None:
                c.fail(f"{r['request_id']}: the manifest has no prompt at position {g['index']} (a lengths-only plan): the record's prompt is "
                       "tied to no declared one")
            elif w is not None and list(w.prompt_token_ids) != list(r["prompt_token_ids"]):
                c.fail(f"{r['request_id']}: record prompt != manifest prompt at position {g['index']}")
    c.info.update({"n_declared": len(g_ids), "n_record": len(h_ids), "record_ids_unknown": res["request_ids"]["unknown_ids"]})
    c.done()
    return facts, hdr_reqs, param_of, g_ids


def _execution_conditions(c: NamedCheck, res: dict, gp: dict, header: dict, oracle_dir: str) -> dict:
    """The engine and compilation conditions the record attests, against the Program's declared applicability; returns the facts."""
    # the Program declares the execution profile + engine conditions it models (GP-01 applicability); the record's
    # RunHeader carries the EFFECTIVE engine kwargs -- a run outside the declaration is not a Match for this Program (named)
    app = gp.get("applicability") or {}
    # the recorder (vllm_adapter build_manifest -> RunHeader.profile) nests the
    # effective engine kwargs and the compilation facts under RunHeader.profile.*; read EVERY location the record may attest them
    # (profile.*, top level, capture/versions.json), require the locations to agree, and FAIL CLOSED when the record attests nothing
    facts = record_engine_facts(header, oracle_dir)
    eff, hc = facts["engine"], facts["compilation"]
    res["execution"] = {"declared_execution": app.get("execution"), "row_execution": app.get("row_execution"), "declared_engine": app.get("engine"),
                        "effective_engine": eff or None, "effective_compilation": hc or None,
                        "source": facts["sources"] or "not recorded (RunHeader.profile.engine_kwargs / RunHeader.engine_kwargs / capture/versions.json)"}
    # the fused-MoE CONSTRUCTION the Program is of (GP-01 `moe_construction` block: indexed-read = MOE-01 / padded =
    # MoeFfnFixedL_v1 per token) is a Program static -- declared-and-bound, NEVER attested: the record is construction-agnostic (the same
    # trace serves both Programs), so nothing of the record is compared with it here; it is REPORTED so the verdict names which
    # representation of the workload it is a Match of (PADDED-TABLE labels the row with its actual population).  A Program that
    # binds a fused-MoE block without the block (an older Build) is labelled indexed-read by its Definitions, never assumed padded.
    _mc = gp.get("moe_construction") if isinstance(gp.get("moe_construction"), dict) else None
    res["moe_construction"] = ({"value": _mc.get("value"), "moe_block_present": _mc.get("moe_block_present"), "population_label": _mc.get("population_label"),
                                "declared_explicitly": _mc.get("declared_explicitly"), "attested": False,
                                "source": "GP-01 moe_construction block (TargetProfile.moe_construction, declared-and-bound; never attested by the record)"}
                               if _mc else {"value": None, "source": "the Program carries no moe_construction block (an older Build): its Definitions "
                                                                     "decide (MOE-01 ids = indexed-read); nothing declared, nothing assumed"})
    for k, (a, b, sa, sb) in facts["disagreements"].items():
        c.fail(f"the record's own sources disagree on {k}: {a!r} ({sa}) vs {b!r} ({sb}) -- the record does not attest one execution")
    for src, err in (facts.get("unreadable") or {}).items():
        c.fail(f"the record's {src} is present but unreadable ({err}): its engine conditions cannot be read, fail closed")
    if not app.get("engine"):
        c.fail("the workload Program declares no applicability.engine (an older artifact): the record's engine conditions cannot be "
               "checked against it -- rebuild the Program on a release carrying GP-01 applicability")
    else:
        if not eff:
            c.fail("the record does not attest its engine conditions (no enforce_eager / enable_prefix_caching at RunHeader.profile.engine_kwargs, "
                   "RunHeader.engine_kwargs or capture/versions.json engine_kwargs): applicability cannot be checked, fail closed")
        # async_scheduling joins the compared keys only when the Program declares it (one source, two readers)
        for k in ("enforce_eager", "enable_prefix_caching") + (("async_scheduling",) if app["engine"].get("async_scheduling") is not None else ()):
            # EACH declared key must be attested -- a record that omits one knob (prefix caching: vLLM V1 default ON)
            # does not get applicability by absence
            if eff and k not in facts["values"]:
                c.fail(f"the record does not attest {k}, which the Program's applicability declares ({k}={app['engine'].get(k)!r}): "
                       f"fail closed")
            for v in facts["values"].get(k, ()):
                if v != app["engine"].get(k):
                    c.fail(f"record engine {k}={v!r} is outside the Program's declared applicability ({k}={app['engine'].get(k)!r})")
    # a Program modelling a COMPILED execution declares its compilation facts; the record must attest the same
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
    return facts


def _manifest_of_record(c: NamedCheck, gp: dict, G: dict, header: dict, hdr_reqs: list[dict], g_ids: list[str]) -> Workload | None:
    """The workload manifest against the Program (ids, per-request fields, sha256) and the record header (the manifest the run was
    captured with, seeds, sampling).  Returns the manifest, or None when the Program names none that exists here or it is not a
    workload record (named)."""
    wl_path = (gp.get("workload") or {}).get("path")
    wl = None
    if wl_path and workload_file(wl_path).is_file():
        try:
            wl = Workload.load(workload_file(wl_path))
        except RecordError as e:
            c.fail(f"the workload manifest is not a workload record: {e}")
            return None
        # the manifest's positions are the Program's r<k>; its request_id column is the workload (UUID) id the record uses
        wl_ids = [request_id_of(i) for i in range(len(wl.requests))]
        if sorted(wl_ids) != sorted(g_ids):
            c.fail(f"manifest requests {sorted(wl_ids)} != global Program requests {sorted(g_ids)}")
        for i, w in enumerate(wl.requests):
            g = G.get(request_id_of(i))
            if g is None:
                continue
            if g.get("workload_request_id") is not None and w.request_id != str(g["workload_request_id"]):
                c.fail(f"{request_id_of(i)}: manifest workload request id {w.request_id!r} != the Program's {g['workload_request_id']!r}")
            # a manifest whose per-request fields were edited after the Build is named per request, not only by digest
            for name, wv, gv in (("prompt_len", w.prompt_length, g.get("prompt_len")), ("max_tokens", w.max_tokens, g.get("max_tokens")),
                                 ("arrive_step", wl.arrive_step(i), g.get("arrive_step"))):
                if wv is not None and gv is not None and int(wv) != int(gv):
                    c.fail(f"{request_id_of(i)}: manifest {name}={int(wv)} != the Program's declared {name}={int(gv)} (manifest edited after the Build)")
        want = (gp.get("workload") or {}).get("sha256")
        have = sha256_file(wl_path)
        c.info["workload_sha256"] = have
        if want and want != have:
            c.fail("workload manifest sha256 differs from the global Program's")
        # the manifest of record must be the manifest the RUN was captured with: the driver records its sha256 in
        # RunHeader.workload.sha256 -- a manifest edited after the run (seeds swapped, a seed deleted) composes the same Program and
        # passed G1 by ids/prompts/caps/arrivals alone; and per request the header's sampling fields are compared with the manifest's
        hw = (header.get("workload") or {}).get("sha256")
        # the driver (vllm_adapter.load_workload) hashes the manifest DOCUMENT canonically (sort_keys, compact separators), not the file
        have_doc = wl.doc_sha256
        c.info["header_workload_sha256"] = hw
        c.info["workload_doc_sha256"] = have_doc
        if not hw:
            c.fail("the record header carries no workload.sha256: the manifest of record cannot be tied to the run (fail closed)")
        elif hw not in (have_doc, have):
            c.fail(f"manifest of record differs from the header: RunHeader.workload.sha256 {hw[:16]} != manifest {have_doc[:16]} "
                   f"(canonical document; file {have[:16]}) -- the run was captured with another manifest")
        hdr_by_id = {r["request_id"]: r for r in hdr_reqs}
        for i, w in enumerate(wl.requests):
            h = hdr_by_id.get(request_id_of(i))
            if h is None:
                continue
            ws, hs = w.seed, h.get("seed")
            if ws is None and hs is not None:
                c.fail(f"no declared seed for {request_id_of(i)} but the run header records seed={hs}")
            elif ws is not None and hs is None:
                c.fail(f"{request_id_of(i)}: manifest declares seed={ws} but the run header records no seed")
            elif ws is not None and int(ws) != int(hs):
                c.fail(f"{request_id_of(i)}: manifest declares seed={ws} but the run header records seed={hs}")
        m_s, h_s = manifest_sampling(wl), dict((header.get("workload") or {}).get("sampling") or {})
        c.info["sampling"] = {"manifest": m_s, "header": {k: h_s.get(k) for k in ("variant", "temperature", "top_p", "top_k") if k in h_s}}
        # the manifest's own sources (variants table = the driver's, label, explicit fields) must agree: fail closed
        for conf in m_s.get("conflict") or []:
            c.fail(f"manifest of record is not one sampling declaration -- {conf}")
        for k in ("variant", "temperature", "top_p", "top_k"):
            mv, hv = m_s.get(k), h_s.get(k)
            if mv is None or hv is None:
                continue
            same = (str(mv) == str(hv)) if k == "variant" else (abs(float(mv) - float(hv)) <= 1e-9)
            if not same:
                c.fail(f"manifest of record declares sampling {k}={mv!r} but the run header records {k}={hv!r}")
        # a request pinning ANOTHER variant (vllm_adapter): its table entry vs the header's per-request fields
        for i, w in enumerate(wl.requests):
            h = hdr_by_id.get(request_id_of(i))
            rv = w.variant
            if h is None or (rv is None and h.get("variant") is None):
                continue
            r_s = manifest_sampling(wl, variant=rv) if rv is not None else m_s
            for conf in (r_s.get("conflict") or []) if rv is not None else []:
                c.fail(f"{request_id_of(i)}: manifest variant {rv!r} is not one declaration -- {conf}")
            # a manifest that pins a per-request variant against a header WITHOUT a per-request sampling
            # record is not agreement: the run was not pinned (mirror of the seed rule above) -- a pin added after the run fails closed
            if rv is not None and all(h.get(k) is None for k in ("variant", "temperature", "top_p", "top_k")):
                c.fail(f"{request_id_of(i)}: manifest pins variant {rv!r} for the request but the run header records no per-request "
                       f"sampling (variant/temperature/top_p/top_k) -- the pin was not what the run served")
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
                    c.fail(f"{request_id_of(i)}: manifest declares sampling {k}={mv!r} but the run header records {k}={hv!r} for the request")
    return wl
