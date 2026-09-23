"""LIFTING-SPEC v0.5.1 A7 — the IDENTITY-LEVEL correspondence table active-reference ↔ lifted (revb M-0042 F2: "row counts alone are
not proof"; BRIEF §7 (a)).  **lifted-padded**, `rules_version = lifted-v1`; never a panel cell.

For ONE request of a record, every identity of `RequiredValues(P)` — the Commit's own required-manifest generator
(`verity_capture.commit.required_manifest.build_manifest`, torch-free, the population the Commit checked coverage against; its digest is
compared with the Commit verdict's `required_manifest.manifest_digest`, so the population is the Commit's, not this tool's) — over EVERY
step `s ∈ [0, cap)` of the fixed Program, gets one row:

    step, engine_step, family, op_path, output_member, element_range, n              -- the identity (TABLE-RULES key)
    assignment      active | bot | shared        -- the pre-challenge assignment: `s < a_r + LAG` (A2), weights are shared (step -1)
    active_reference bound | unbound | n/a       -- does the Commit's binding map carry it (coverage rule: same key, containing range)
    lifted_word     "0||v" | 2^w (= enc(⊥))      -- the word class the lifted record half holds there (A3.4): the observed value widened
                                                    at an active position, the constant enc(⊥) at an inactive one; NEVER read from a trace
    dtype, dtype_bits, w, w_lifted, lifted_bits  -- the storage dtype (information) and the PROGRAM's leaf width w (from the producing
                                                    spec's return type when the identity names one, else from the dtype), w+1, n·(w+1)
    guard_live, tok_in                            -- the guard's gates of the identity's step: live_s = LStepLive(tok_in_{s-1}) ∈ {Active(1),
                                                    Active(0)} (never ⊥, A3.2), tok_in_s ∈ {Active, ⊥}; steps 0..LAG are unguarded

The table has `cap · |identities per step| + |weights|` rows whatever the stopping point; two runs of one Build with different early stops
have the SAME rows modulo `assignment` / `active_reference` (`compare`).  What a lifted-padded demonstration reads off it (A7 (a)(ii)/(iv)):
the inactive suffix is PRESENT (bot rows with `enc(⊥)`, `unbound` in today's Commit), the widths changed (`w → w+1` on every row), and no
active identity is unbound (a coverage hole).  Today's Commit binds the executed steps only, so `bound_bot = 0` and `unbound_bot = |bot|`;
a LIFTED Commit binds every bot identity with `enc(⊥)` (`bound_bot = |bot|`) and R6 checks the words.

    python -m verity_vllm.lifting.correspondence build --build-request <dir> --commit <dir> [--pair 0] --commit-request-id r0 --a-r 11 --cap 32 --lag 1 --out t.jsonl.gz
    python -m verity_vllm.lifting.correspondence compare a.jsonl.gz b.jsonl.gz --out cmp.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from verity_vllm.registry import lifted as L

SCHEMA = "verity/lifted-correspondence/v1"

#: storage dtype → bits (information only: `dtype_bits`); the PROGRAM's leaf width `w` comes from the producing spec when the identity names one
DTYPE_BITS = {"bf16": 16, "bfloat16": 16, "f16": 16, "float16": 16, "f32": 32, "float32": 32, "u32": 32, "uint32": 32, "i32": 32, "int32": 32,
              "i64": 64, "int64": 64, "u8": 8, "uint8": 8, "i8": 8, "int8": 8, "fp8_e4m3": 8, "fp8_e5m2": 8, "bool": 1}
#: the Program's leaf width for the families whose identity names no spec: token ids are `Value<32>` leaves of the Program (the prompt
#: parameter `Array<LP, Value<32>>`, `TokenSelect_v1`'s return) whatever the engine's storage dtype; the FA2/FA3 hidden stream is u32 words
FAMILY_W = {"input_token_ids": 32, "sampled_token_ids": 32, "fa2_hidden_m1_stream": 32, "fa3_hidden_m1_stream": 32, "sampler_seed": 64, "sampler_splits": 32}


def load_program_tolerant(build_request_dir: str | Path) -> dict[str, Any]:
    """`required_manifest.load_program`, tolerating a copied row without `result.json` (the R16 evidence copies carry descriptor / instances /
    artifact only): the digest comes from `artifact.json`, `attention_impls_observed` / `target` are absent and the manifest SAYS so
    (`result_json: reconstructed`) — the digest of the identities is what pins it to the Commit's."""
    from verity_capture.commit import required_manifest as RM
    d = str(build_request_dir)
    if os.path.exists(os.path.join(d, "result.json")):
        prog = RM.load_program(d)
        prog["result_json"] = "present"
        return prog
    with gzip.open(os.path.join(d, "instances.json.gz"), "rt") as f:
        inst = json.load(f)
    if os.path.exists(os.path.join(d, "artifact.json")):
        art = json.load(open(os.path.join(d, "artifact.json")))
        pdig, src = art["program_digest"], "artifact.json program_digest"
    else:  # a synthetic row: the component's own digest (what the Build's artifact.json records for a real one)
        from verity_vllm.ir import workload as W
        from verity_vllm.lifting.record import REGISTRY, load_descriptor
        art = None
        params = [(str(n), str(t)) for n, t in (inst.get("params") or [])] or None
        pdig, src = W.request_component(load_descriptor(d), REGISTRY, params).program_digest, "request_component(descriptor).program_digest (no artifact.json)"
    res = {"program": {"digest": pdig}, "attention_impls_observed": {}, "target": {}}
    return {"instances": inst, "result": res, "artifact": art, "dir": d, "result_json": f"reconstructed ({src}; no attention_impls_observed / target)"}


def required_identities(build_request_dir: str | Path, *, request_id: str = "r0", arrive: int = 0, batch: int = 1, profile: str | None = None,
                        engine: str | None = None) -> dict[str, Any]:
    """`RequiredValues(P)` of one request component over every step of the fixed Program — the Commit's generator, run here.  `engine`:
    reqval's `inventory` (the cached serialisation) or `traversal` (`required_values.Traversal`; the one that unlifts `Lifted[X]` rows), else
    the tree's default (`required_manifest.manifest_engine`)."""
    from verity_capture.commit import required_manifest as RM
    prog = load_program_tolerant(build_request_dir)
    wl = {"sweep": {"arrive_steps": [int(arrive)], "batch": int(batch)}} if (arrive or batch > 1) else None
    m = RM.build_manifest(prog, wl, request_id=request_id, profile=profile, engine=engine)
    m["result_json"] = prog.get("result_json")
    return m


def workload_identities(wp: dict, comp_dirs: dict[str, str | Path], *, profile: str | None = None) -> dict[str, Any]:
    """`RequiredValues(P)` of GP-01's WORKLOAD Program (B ≥ 2): the Commit's `build_global_manifest` over the Build's own declaration
    (`workload_program.json` `requests[]`: request_id / index / prompt_len / cap / arrive_step; `workload_digest`), one loaded component per
    `r<k>` (`comp_dirs`).  Its `manifest_digest` is what the Commit verdict's `required_manifest.manifest_digest` must equal."""
    from verity_capture.commit import required_manifest as RM
    fixture = {"fixture": wp.get("manifest_digest"), "requests": [
        {"request_id": r.get("request_id"), "index": int(r.get("index", k)), "prompt_len": int(r["prompt_len"]), "max_tokens": int(r["cap"]),
         "arrive_step": int(r.get("arrive_step", 0))} for k, r in enumerate(wp["requests"])]}
    programs = {str(rid): load_program_tolerant(d) for rid, d in comp_dirs.items()}
    m = RM.build_global_manifest(fixture, programs, workload_digest=wp.get("workload_digest"), profile=profile)
    m["result_json"] = {rid: p.get("result_json") for rid, p in programs.items()}
    return m


def split_by_request(manifest: dict) -> dict[str, dict]:
    """The global manifest's identities per `r<k>` (the shared `*` rows — weights, the packed hidden stream — go to EVERY request's view,
    labelled shared by `correspondence`)."""
    out: dict[str, dict] = {}
    # the weights are ONE registered input of the whole workload whatever request id the generator stamped on their rows (the global
    # manifest labels them with the first request's): shared by every request's view, as the Commit binds them
    shared = [i for i in manifest["identities"] if str(i.get("request_id")) == "*" or i.get("family") == "weights"]
    for i in manifest["identities"]:
        rid = str(i.get("request_id"))
        if rid == "*" or i.get("family") == "weights":
            continue
        out.setdefault(rid, {"program_digest": manifest.get("program_digest"), "manifest_digest": manifest.get("manifest_digest"),
                             "query": manifest.get("query"), "result_json": manifest.get("result_json"), "identities": []})["identities"].append(i)
    for rid in out:
        out[rid]["identities"].extend(shared)
    return out


def spec_leaf_widths(desc: dict, params: list[tuple[str, str]] | None) -> dict[str, list[int]]:
    """spec id → the distinct leaf widths of that node's return type in the Program (`Value<w>` leaves), from the decoded component body."""
    from verity_vllm.ir import workload as W
    from verity.ir.codec import _spec_id
    from verity_vllm.lifting.record import REGISTRY, _leaf_widths
    comp = W.request_component(desc, REGISTRY, params)
    out: dict[str, set[int]] = {}
    for n in comp.fn.body.nodes:
        sid = _spec_id(n.fn)
        ws = _leaf_widths(n.out)
        out.setdefault(sid, set()).update(ws)
        out.setdefault(sid.split("{", 1)[0], set()).update(ws)  # the family (`Gemm_v1`): what the required manifest's `spec` names
    return {k: sorted(v) for k, v in out.items()}


def identity_w(ident: dict, widths_by_spec: dict[str, list[int]] | None) -> tuple[int | None, str]:
    """The Program's leaf width of an identity: the producing spec's unique return leaf width when the identity names a spec the body has;
    else the family rule (token ids / hidden stream); else the dtype's bits (weights: bf16 → 16, f32 → 32).  `(None, why)` when none applies."""
    spec = ident.get("spec")
    if spec and widths_by_spec is not None:
        ws = widths_by_spec.get(spec)
        if ws and len(ws) == 1:
            return ws[0], "spec"
        if ws:
            return None, f"spec {spec[:40]} returns leaves of several widths {ws}"
    fam = str(ident.get("family"))
    if fam in FAMILY_W:
        return FAMILY_W[fam], "family"
    dt = str(ident.get("dtype") or "")
    if dt in DTYPE_BITS:
        return DTYPE_BITS[dt], "dtype"
    return None, f"no width rule for family {fam!r} dtype {dt!r}"


def _step_key(x: Any) -> int | str:
    """`step` of an identity: an int, or the wildcard `*` / `-1` of the shared weights (kept as the string it is; never coerced)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return x
    s = str(x)
    return int(s) if s.lstrip("-").isdigit() else s


def bound_index(bmap: dict | None, commit_rid: str) -> dict[tuple, list[tuple[int, int | None]]]:
    """(step, invocation, op_path, member) → the committed element ranges of `commit_rid` (and of the shared `*` request)."""
    idx: dict[tuple, list[tuple[int, int | None]]] = {}
    for e in (bmap or {}).get("entries") or []:
        i = e.get("identity") or {}
        rid = str(i.get("request_id", "r0"))
        if rid not in (commit_rid, "*"):
            continue
        lo, hi = (i.get("element_range") or [0, None])[:2]
        idx.setdefault((_step_key(i.get("step", -1)), int(i.get("invocation", 0)), str(i.get("op_path")), str(i.get("output_member"))), []).append((int(lo), None if hi is None else int(hi)))
    return idx


def is_bound(idx: dict, ident: dict) -> bool:
    key = (_step_key(ident.get("step", -1)), int(ident.get("invocation", 0)), str(ident.get("op_path")), str(ident.get("output_member")))
    lo, hi = (ident.get("element_range") or [0, None])[:2]
    for blo, bhi in idx.get(key, []):
        if blo <= int(lo) and (bhi is None or hi is None or bhi == -1 or int(hi) <= bhi):
            return True
    # member `*` = any member of that op_path / invocation (the coverage rule's wildcard)
    for blo, bhi in idx.get((key[0], key[1], key[2], "*"), []):
        return True
    return False


#: batch-level hidden rows (request `*` + a co-scheduled `requests` list): the Commit binds them per (request, step) as per-sequence planes
HIDDEN_FAMILIES = ("fa2.m1/", "fa3.m1/", "moe/")
BINDING_RULE_COMMIT = "commit (verity_capture.commit.binding.load_required_manifest -> coverage_check on the same manifest and map)"
BINDING_RULE_LOCAL = "local (is_bound: exact (step, invocation, op_path, member) key or member '*'; synthetic maps only -- not the Commit's rule)"


def is_plane(ident: dict) -> bool:
    """A packed per-step plane row of the global manifest: request `*`, a co-scheduled `requests` list, a hidden-stream op, an engine step."""
    return (str(ident.get("request_id")) == "*" and isinstance(ident.get("requests"), list) and bool(ident["requests"])
            and str(ident.get("op_path")).startswith(HIDDEN_FAMILIES) and ident.get("engine_step") is not None)


def plane_widths(manifest: dict, widths_by_spec_by_request: dict[str, dict[str, list[int]]] | None = None) -> dict:
    """F-r17b-18 (revb): the packed per-step planes of a GLOBAL manifest, each counted ONCE at its whole-launch extent, by family/dtype and
    by engine step -- the workload-level widths of what the per-request tables only see slices of (`widths.planes_seen`).  A plane's width
    is the family rule (hidden streams) or the dtype's; `widths_by_spec_by_request` is consulted for a plane row that names a spec."""
    by_w: dict[str, dict[str, Any]] = {}
    by_step: dict[int, dict[str, int]] = {}
    unknown: dict[str, int] = {}
    n_planes = 0
    for ident in manifest["identities"]:
        if not is_plane(ident):
            continue
        n_planes += 1
        rng = ident.get("element_range") or [0, None]
        n = (int(rng[1]) - int(rng[0])) if (rng[1] is not None and rng[1] != -1) else None
        w, w_src = None, "none"
        for rk in [str(x) for x in ident["requests"]] + [None]:
            w, w_src = identity_w(ident, (widths_by_spec_by_request or {}).get(rk) if rk is not None else None)
            if w is not None:
                break
        key = f"{ident.get('family')}/{ident.get('dtype')}"
        if w is None:
            unknown[key] = unknown.get(key, 0) + 1
            continue
        bw = by_w.setdefault(key, {"family": ident.get("family"), "dtype": ident.get("dtype"), "w": w, "w_lifted": w + 1, "w_source": w_src,
                                   "planes": 0, "elements": 0, "lifted_bits": 0, "payload_bits": 0, "bot_word": L.enc_bot(w)})
        bw["planes"] += 1
        bw["elements"] += n or 0
        bw["lifted_bits"] += (n or 0) * (w + 1)
        bw["payload_bits"] += (n or 0) * w
        e = int(ident["engine_step"])
        st = by_step.setdefault(e, {"planes": 0, "elements": 0, "lifted_bits": 0, "requests": 0})
        st["planes"] += 1
        st["elements"] += n or 0
        st["lifted_bits"] += (n or 0) * (w + 1)
        st["requests"] = max(st["requests"], len(ident["requests"]))
    return {"planes": n_planes, "by_family_dtype": by_w, "w_unknown": unknown,
            "lifted_bits_total": sum(v["lifted_bits"] for v in by_w.values()), "payload_bits_total": sum(v["payload_bits"] for v in by_w.values()),
            "by_engine_step": {str(e): by_step[e] for e in sorted(by_step)},
            "rule": "each packed plane once, at its whole-launch extent (every co-scheduled request's slice); the per-request tables exclude them "
                    "from their own widths (widths.excludes_planes) and report their view under widths.planes_seen"}


def _pop_key(rid: Any, st: Any, inv: Any, op: Any, mem: Any, lo: Any, hi: Any, rk: Any) -> tuple:
    hi = -1 if hi is None else int(hi)
    return (str(rid), int(st), int(inv), str(op), str(mem), int(lo), hi, int(rk or 0))


def commit_coverage(manifest: dict, bmap: dict, *, requests: list[str], caps: dict[str, int], arrivals: dict[str, int],
                    request_steps: dict[str, Any] | None = None) -> dict[str, Any]:
    """THE Commit's binding rule, run here on the SAME manifest and binding map the Commit used (revb F-r17b-15: never a second rule):
    `binding.load_required_manifest` expands the manifest's rows into the per-(request, engine step) population exactly as `commit_delta`
    does (a batch-level hidden row -> one op-level identity per co-scheduled request, member `*`; a weights row -> the registered input),
    `binding.coverage_check` decides bound / missing per expanded identity.  Returns the per-identity verdict keyed by
    (request, engine_step, invocation, op_path, member, lo, hi, rank), the coverage numbers, and the expansion header.

    The expansion is over the Program's CAP SHAPE (`RequiredValues(P)` = `RequiredValues(Lift(P))`, the fixed population): that is the
    lifted reading -- a bot identity of an early-stopped request is in the population and today's active-reference Commit leaves it unbound
    (SPEC A3.6 N8 / R3e).  `request_steps` = {r<i>: the engine steps the request EXECUTED} (moe's `executed_prefix.request_steps_of_record`,
    R17-1 / F-moe-r17-03 aad4bd3ea) additionally runs the Commit's RULE OF RECORD -- the requirement placed over each request's executed
    prefix, the cap-shaped rows past it counted as `programmed_not_executed_n`, not required -- and returns it under `of_record` beside the
    cap-shaped reading (`available` False, named, on a binding.py without the kwarg).  Same manifest, same map, the Commit's own code both times."""
    import inspect
    import tempfile
    from verity_capture.commit import binding as B
    horizon = max(int(arrivals.get(k, 0)) + int(caps[k]) for k in caps)
    doc = {k: v for k, v in manifest.items() if k != "result_json"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(doc, f)
        path = f.name
    of_record: dict[str, Any] | None = None
    try:
        pop, hdr = B.load_required_manifest(path, manifest["program_digest"], requests=list(requests), steps=horizon, ranks=[0], arrivals=dict(arrivals))
        if request_steps is not None:
            if "request_steps" in inspect.signature(B.load_required_manifest).parameters:
                pop_r, hdr_r = B.load_required_manifest(path, manifest["program_digest"], requests=list(requests), steps=horizon, ranks=[0], arrivals=dict(arrivals),
                                                        request_steps={str(k): [int(s) for s in v] for k, v in request_steps.items()})
                cov_r = B.coverage_check(pop_r, bmap, max_report=len(pop_r) + 1)
                of_record = {"available": True, "rule": "the Commit's rule of record (R17-1 / F-moe-r17-03): load_required_manifest(request_steps=executed prefix of "
                                                        "record) -> coverage_check; rows of a request past its executed prefix are programmed_not_executed, not required",
                             "request_steps_of_record": hdr_r.get("request_steps_of_record"), "programmed_not_executed_n": hdr_r.get("programmed_not_executed_n"),
                             "coverage": {k: cov_r.get(k) for k in ("ok", "expected_n", "bound_n", "missing_n", "ambiguous_n", "address_mismatch_n", "rules_version")},
                             "missing_by_request": _count_by_request(cov_r.get("missing") or [])}
            else:
                of_record = {"available": False, "why": "verity_capture.commit.binding.load_required_manifest has no `request_steps` (tree without moe's F-moe-r17-03 "
                                                       "aad4bd3ea): the Commit's rule of record cannot be run here; the cap-shaped reading stands alone"}
    finally:
        os.unlink(path)
    cov = B.coverage_check(pop, bmap, max_report=len(pop) + 1)
    missing = {_pop_key(m["request_id"], m["step"], m["invocation"], m["op_path"], m["output_member"], *m["element_range"], m.get("rank", 0)) for m in cov["missing"]}
    bound: dict[tuple, bool] = {}
    for ident in pop:
        _pd, rid, st, inv, op, mem, (lo, hi), rk = ident
        k = _pop_key(rid, st, inv, op, mem, lo, hi, rk)
        bound[k] = k not in missing
    return {"bound": bound, "rule": BINDING_RULE_COMMIT, "horizon": horizon, "arrivals": dict(arrivals),
            "coverage": {k: cov.get(k) for k in ("ok", "expected_n", "bound_n", "missing_n", "ambiguous_n", "address_mismatch_n", "rules_version")},
            "expansion": {k: hdr.get(k) for k in ("hidden_batch_rows_op_level", "engine_step_rebased_rows", "global", "rows_n", "manifest_digest_verified")},
            "shape": "cap (the fixed population RequiredValues(P) == RequiredValues(Lift(P)); the lifted reading)",
            "of_record": of_record}


def _count_by_request(missing: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in missing:
        rid = str(m.get("request_id"))
        out[rid] = out.get(rid, 0) + 1
    return dict(sorted(out.items()))


def _composition_step(manifest: dict, engine_step: int, rid: str) -> tuple[int | None, int | None]:
    """(request-local step, rows) of `rid` at `engine_step` from the global manifest's DECLARED composition (`engine_steps[e].members`), else
    (None, None) -- never inferred from the collector."""
    for es in manifest.get("engine_steps") or []:
        if isinstance(es, dict) and int(es.get("engine_step", -1)) == int(engine_step):
            for m in es.get("members") or []:
                if str(m.get("request_id")) == str(rid):
                    return (int(m["step"]) if m.get("step") is not None else None), (int(m["rows"]) if m.get("rows") is not None else None)
    return None, None


def _declared_slot(manifest: dict, engine_step: int, rid: str, plane_requests: list[str]) -> tuple[int, int | None, list[int]]:
    """(sequence index of `rid`, its rows, the declared rows of every member in order) at `engine_step` from the manifest's DECLARED composition
    (`engine_steps[e].members`, the Program x workload -- the LIFTED fixed plane's slot order); falls back to the plane row's own `requests`
    order with unknown rows when the manifest declares no composition."""
    for es in manifest.get("engine_steps") or []:
        if isinstance(es, dict) and int(es.get("engine_step", -1)) == int(engine_step):
            mem = es.get("members") or []
            order = [str(m.get("request_id")) for m in mem]
            if str(rid) in order:
                rows = [int(m["rows"]) if m.get("rows") is not None else None for m in mem]
                i = order.index(str(rid))
                return i, rows[i], rows
    return plane_requests.index(str(rid)), None, [None] * len(plane_requests)


_GEOM = re.compile(r"^(?:s(?P<seq>\d+)/)?g(?P<HB>\d+)x(?P<M>\d+)x(?P<NB>\d+)(?:x(?P<BN>\d+)x(?P<D>\d+))?(?:/(?P<part>[A-Za-z0-9_]+))?$")


def parse_plane_member(member: str) -> dict[str, Any] | None:
    """`[s<j>/]g<HB>x<M>x<NB>[x<BN>x<D>][/<part>]` -> {seq (int | None: a packed whole-launch plane), HB, M, NB, BN, D, part}; None if not a plane member."""
    m = _GEOM.match(str(member or ""))
    if not m:
        return None
    d = {k: (int(v) if v is not None and k != "part" else v) for k, v in m.groupdict().items()}
    return d


def record_composition(bmap: dict | None, manifest: dict) -> dict[str, Any] | None:
    """F-r17b-23 (revb): the RECORD's per-engine-step composition of the packed launch, read from the binding map's entries of the manifest's
    `composition_reference` members (per-token `input_token_ids` op/member: a request's `element_offset` = its first row of the launch, its
    `element_range` width = its rows; per-sequence `sampled_token_ids`: `element_offset` = its sequence index) -- what the collector actually
    committed at each step (the Commit's own composition_check reads the same entries), NEVER the manifest's declared composition.  Returns
    {"source", "steps": {e: {"n_seq", "rows_total", "order": [rid by row], "members": {rid: {"seq_index", "row_lo", "row_hi", "rows"}}}}}
    or None when the map carries no reference entries at all (named by the caller)."""
    if bmap is None:
        return None
    ref = manifest.get("composition_reference") or {}
    tok = ref.get("input_token_ids") or {"op_path": "model.embed_tokens", "output_member": "in"}
    smp = ref.get("sampled_token_ids") or {"op_path": "runner.sampler", "output_member": "sampled_token_ids"}
    steps: dict[int, dict[str, Any]] = {}
    seq_of: dict[tuple[int, str], int] = {}
    for e in bmap.get("entries") or []:
        if e.get("kind") not in (None, "leaf"):
            continue
        i = e.get("identity") or {}
        rid, st = str(i.get("request_id")), i.get("step")
        if st is None or rid == "*":
            continue
        op, mem = str(i.get("op_path")), str(i.get("output_member"))
        if op == str(tok.get("op_path")) and mem == str(tok.get("output_member")):
            lo, hi = (i.get("element_range") or [0, None])
            rows = (int(hi) - int(lo)) if hi is not None and hi != -1 else None
            off = int(e.get("element_offset") or 0)
            shape = (e.get("encoding") or {}).get("shape") or []
            d = steps.setdefault(int(st), {"members": {}, "rows_total": None, "n_seq": None})
            d["members"][rid] = {"row_lo": off, "row_hi": (off + rows) if rows is not None else None, "rows": rows}
            if shape:
                d["rows_total"] = int(shape[0]) if d["rows_total"] is None else max(d["rows_total"], int(shape[0]))
        elif op == str(smp.get("op_path")) and mem == str(smp.get("output_member")):
            seq_of[(int(st), rid)] = int(e.get("element_offset") or 0)
            shape = (e.get("encoding") or {}).get("shape") or []
            d = steps.setdefault(int(st), {"members": {}, "rows_total": None, "n_seq": None})
            if shape:
                d["n_seq"] = int(shape[0]) if d["n_seq"] is None else max(d["n_seq"], int(shape[0]))
    if not steps:
        return None
    for st, d in steps.items():
        order = sorted(d["members"], key=lambda r: (d["members"][r]["row_lo"], r))
        for j, rid in enumerate(order):
            d["members"][rid]["seq_index"] = seq_of.get((st, rid), j)
        for (s2, rid), j in seq_of.items():
            if s2 == st and rid not in d["members"]:
                d["members"][rid] = {"row_lo": None, "row_hi": None, "rows": None, "seq_index": j}
        d["order"] = sorted(d["members"], key=lambda r: (d["members"][r]["seq_index"], r))
        if d["n_seq"] is None:
            d["n_seq"] = len(d["members"])
        if d["rows_total"] is None:
            d["rows_total"] = sum(m["rows"] or 0 for m in d["members"].values()) or None
    return {"source": f"binding map entries of composition_reference input_token_ids {tok.get('op_path')}/{tok.get('output_member')} (row offset + rows) "
                      f"and sampled_token_ids {smp.get('op_path')}/{smp.get('output_member')} (sequence index) -- the record's composition, not the manifest's",
            "steps": steps}


def record_planes(bmap: dict | None) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """The committed plane entries of the binding map by (op_path, engine step): the tap's per-sequence planes `s<j>/g…/<part>` at a
    multi-sequence step, the packed whole-launch plane `g…` otherwise; each with its element count (the encoding's shape) -- the committed
    geometry F-r17b-23 asks the slice to be placed in."""
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for e in (bmap or {}).get("entries") or []:
        i = e.get("identity") or {}
        op = str(i.get("op_path"))
        if not op.startswith(HIDDEN_FAMILIES) or i.get("step") is None:
            continue
        mem = str(i.get("output_member"))
        g = parse_plane_member(mem)
        if g is None:
            continue
        key = (op, int(i["step"]), mem)
        if key in seen:
            continue  # the same plane bound to every co-scheduled request (op-level rule): one committed object
        seen.add(key)
        shape = (e.get("encoding") or {}).get("shape") or []
        numel = 1
        for x in shape:
            numel *= int(x)
        out.setdefault((op, int(i["step"])), []).append({"member": mem, "seq": g["seq"], "M": g["M"], "part": g["part"], "numel": numel if shape else None,
                                                         "shape": list(shape), "leaf_lo": e.get("leaf_lo"), "stream_offset": e.get("stream_offset")})
    return out


def plane_slice(ident: dict, manifest: dict, commit_rid: str, *, rcomp: dict | None, rplanes: dict | None) -> dict[str, Any]:
    """F-r17b-23 (revb): this request's SLICE of a packed per-step plane, placed twice -- in the LIFTED fixed plane (the manifest's declared
    geometry and composition: slot `[i*M, (i+1)*M)` rows of `n_seq` slots of `M` rows, `elements_per_row = n / rows`) and in the COMMITTED
    plane of the record (the binding map's composition: `[row_lo, row_hi)` of `rows_total` rows, the per-sequence planes `s<j>/…` or the packed
    plane that carry it), or `committed.present False` -- 'absent (committed n_seq = k)' -- for a request the record did not run at that step,
    with the bot slot's size from the lifted fixed shape.  Never a row count standing for a correspondence."""
    e = int(ident["engine_step"])
    n = None
    rng = ident.get("element_range") or [0, None]
    if rng[1] is not None and rng[1] != -1:
        n = int(rng[1]) - int(rng[0])
    preq = [str(x) for x in ident.get("requests") or []]
    geom = dict(ident.get("geometry") or {})
    pm = parse_plane_member(str(ident.get("output_member")))
    # the fa2 geometry's `M` is the KERNEL's query-tile rows -- equal to the rows per sequence at prefill, NOT at decode under GQA (SmolLM2
    # `g6x3x1`: M = 3 q-heads per kv head with 1 row per sequence; OLMoE `g16x1x1x64x128`: M = 1 = rows) -- so the slot is never sized by it:
    # it is recorded as `M_geometry`; the slot comes from the declared COMPOSITION (`engine_steps[e]`: rows_total, n_seq, members' rows)
    M_geom = int(geom.get("M") if geom.get("M") is not None else (pm["M"] if pm else 0)) or None
    n_seq = int(geom.get("n_seq") if geom.get("n_seq") is not None else len(preq)) or None
    i, rows_r, all_rows = _declared_slot(manifest, e, commit_rid, preq)
    rows_decl = int(geom["rows"]) if geom.get("rows") is not None else (sum(all_rows) if all(r is not None for r in all_rows) and all_rows else None)
    epr = (n // rows_decl) if (n is not None and rows_decl) else None
    if all_rows and all(r is not None for r in all_rows) and rows_decl is not None and sum(all_rows) == rows_decl:
        # the members' declared rows tile the plane exactly (no pad rows): slot i = the cumulative rows of the members before it
        slot_lo = sum(all_rows[:i])
        slot_w: int | None = all_rows[i]
        slot_rule = "cumulative declared member rows (sum(rows) == rows_total: no pad rows)"
    elif rows_decl is not None and n_seq and rows_decl % n_seq == 0:
        # equal slots of rows_total / n_seq rows (a request's own rows first, the rest pad -- OLMoE step 0: 31 own + 1 pad of 32)
        slot_w = rows_decl // n_seq
        slot_lo = i * slot_w
        slot_rule = "rows_total / n_seq equal slots (pad rows after a member's own rows)"
    else:
        slot_lo, slot_w, slot_rule = None, None, "unplaceable: the declared rows neither tile the plane nor divide into n_seq equal slots (named)"
    slot_hi = (slot_lo + slot_w) if (slot_lo is not None and slot_w is not None) else None
    slot = {"seq_index": i, "slot_rows": slot_w, "M_geometry": M_geom, "M_geometry_equals_slot_rows": ((M_geom == slot_w) if (M_geom is not None and slot_w is not None) else None),
            "n_seq": n_seq, "rows_total": rows_decl, "elements_per_row": epr,
            "row_range": [slot_lo, slot_hi], "own_rows": ([slot_lo, slot_lo + rows_r] if (rows_r is not None and slot_lo is not None) else None),
            "pad_rows": ((slot_w - rows_r) if (slot_w is not None and rows_r is not None) else None),
            "element_range": ([slot_lo * epr, slot_hi * epr] if (epr is not None and slot_lo is not None) else None),
            "elements": (slot_w * epr) if (slot_w is not None and epr is not None) else None, "slot_rule": slot_rule,
            "source": "declared composition engine_steps[e] (rows_total, n_seq, members' rows in order) + the plane's n -- the lifted fixed plane; "
                      "the fa2 geometry M is recorded, never load-bearing"}
    M = slot_w  # the bot slot's rows (below) are the slot's
    committed: dict[str, Any]
    if rcomp is None:
        committed = {"present": None, "why": "no composition-reference entries in the binding map: the committed slice cannot be placed (named)"}
    else:
        cs = (rcomp.get("steps") or {}).get(e)
        if cs is None:
            committed = {"present": False, "why": f"absent (the record committed no composition at engine step {e})", "n_seq": 0, "order": [],
                         "bot_slot_elements": slot["elements"]}
        elif commit_rid not in cs["members"]:
            committed = {"present": False, "why": f"absent (committed n_seq = {cs['n_seq']}: {cs['order']})", "n_seq": cs["n_seq"], "order": cs["order"],
                         "rows_total": cs["rows_total"], "bot_slot_elements": slot["elements"], "bot_slot_rows": M}
        else:
            m = cs["members"][commit_rid]
            planes = (rplanes or {}).get((str(ident.get("op_path")), e)) or []
            own = [p for p in planes if p["seq"] is not None and p["seq"] == m.get("seq_index")]
            packed = [p for p in planes if p["seq"] is None]
            # F-r17b-26 (revb): the carrier's extent is the COMMITTED plane's (tile-rounded: SmolLM2 per-sequence m1 plane 3,584 elements for
            # 3,564 = 28 x 128), named as its own field; the slice's `elements` are its rows x the lifted plane's elements per row (the
            # payload), `carrier_pad_elements` the difference -- never a padded extent standing for the slice
            if own:
                numel = sum(p["numel"] or 0 for p in own)
                epr_l = slot["elements_per_row"]
                elems = (m["rows"] * epr_l) if (m.get("rows") is not None and epr_l is not None) else None
                committed = {"present": True, "n_seq": cs["n_seq"], "rows_total": cs["rows_total"], "order": cs["order"], "seq_index": m.get("seq_index"),
                             "row_range": [m["row_lo"], m["row_hi"]], "rows": m["rows"], "carrier": "per-sequence planes", "members": [p["member"] for p in own],
                             "elements": (elems if elems is not None else numel), "elements_per_row": epr_l,
                             "element_range_in_carrier": ([0, elems] if elems is not None else [0, numel]),
                             "carrier_elements": numel, "carrier_pad_elements": ((numel - elems) if (elems is not None and numel) else None),
                             "carrier_extent_consistent": ((numel >= elems) if (elems is not None and numel) else None),
                             "M_geometry_committed": own[0]["M"], "M_geometry_equals_rows": (own[0]["M"] == m["rows"]) if m.get("rows") is not None else None}
            elif packed:
                numel = sum(p["numel"] or 0 for p in packed)
                epr_l = slot["elements_per_row"]
                epr_c = epr_l if epr_l is not None else ((numel // cs["rows_total"]) if (cs.get("rows_total") and numel) else None)
                elems = (m["rows"] * epr_c) if (m.get("rows") is not None and epr_c is not None) else None
                whole = (cs["rows_total"] * epr_c) if (cs.get("rows_total") and epr_c is not None) else None
                committed = {"present": True, "n_seq": cs["n_seq"], "rows_total": cs["rows_total"], "order": cs["order"], "seq_index": m.get("seq_index"),
                             "row_range": [m["row_lo"], m["row_hi"]], "rows": m["rows"], "carrier": "packed plane", "members": [p["member"] for p in packed],
                             "elements": elems, "elements_per_row": epr_c,
                             "element_range_in_carrier": ([m["row_lo"] * epr_c, m["row_hi"] * epr_c] if (epr_c is not None and m.get("row_lo") is not None
                                                                                                          and m.get("row_hi") is not None) else None),
                             "carrier_elements": numel, "carrier_pad_elements": ((numel - whole) if (whole is not None and numel) else None),
                             "carrier_extent_consistent": ((numel >= whole) if (whole is not None and numel) else None),
                             "M_geometry_committed": packed[0]["M"], "M_geometry_equals_rows": (packed[0]["M"] == m["rows"]) if m.get("rows") is not None else None}
            else:
                committed = {"present": False, "why": f"the record ran the request at step {e} (rows {[m['row_lo'], m['row_hi']]} of {cs['rows_total']}) but the "
                                                      f"binding map has no plane of {ident.get('op_path')} at that step", "n_seq": cs["n_seq"], "order": cs["order"],
                             "row_range": [m["row_lo"], m["row_hi"]], "rows": m["rows"]}
    return {"lifted_slot": slot, "committed_slice": committed}


def guard_of_step(s: int, n_exec: int, lag: int) -> tuple[str, str]:
    """The guard's gates at step `s` (input guard, A2 / A3.2): steps `0..LAG` are unguarded (unconditionally live: event 0 and the LAG
    placeholder steps); for `s > LAG`, `live_s = LStepLive(tok_in_{s-1})` is `Active(1)` iff the step is executed (`s < n_exec`) else
    `Active(0)` — never ⊥; `tok_in_s` is the Active previous token iff live, else ⊥."""
    if s <= lag:
        return "unguarded", "unguarded"
    return ("Active(1)", "Active") if s < n_exec else ("Active(0)", "bot")


def correspondence(manifest: dict, bmap: dict | None, *, commit_rid: str, a_r: int, cap: int, lag: int, arrive: int = 0,
                   widths_by_spec: dict[str, list[int]] | None = None, commit_cov: dict | None = None) -> tuple[list[dict], dict]:
    """The rows and their summary for one request.  `a_r` is the served prefix (R2, from the committed tokens); `n_exec = min(a_r + LAG, cap)`.

    Binding column: with `commit_cov` (`commit_coverage` on the FULL manifest and the Commit's map) every row's bound / unbound is THE
    Commit's own verdict for that identity (F-r17b-15 (1)); without it and with a `bmap`, the local key match (synthetic maps), named in the
    summary as such.  A batch-level hidden row (request `*`, co-scheduled `requests`, `HIDDEN_FAMILIES` op) is a PACKED per-step plane: it
    has no assignment of its own -- this request's (request, step) slice of it is one row of THIS table, taking the request's assignment at
    that step (`0||v` executed, `enc(bot)` inactive; SPEC A3.6) and the Commit's op-level per-request binding (F-r17b-15 (2)); the plane's
    other requests' slices are rows of THEIR tables."""
    lag = L.check_lag(lag)
    n_exec = L.expected_served(a_r, cap, lag)
    if not manifest.get("identities"):
        raise ValueError("the required-manifest generator found NO identities for this Program (a synthetic body without module names, or a "
                         "derive without instances.json.gz rows): no correspondence table, nothing is inferred")
    idx = bound_index(bmap, commit_rid) if bmap is not None else None
    registered = any(e.get("kind") == "registered-input" for e in (bmap or {}).get("entries") or []) if bmap is not None else None
    cbound = (commit_cov or {}).get("bound") if commit_cov else None
    binding_rule = None if idx is None else (BINDING_RULE_COMMIT if cbound is not None else BINDING_RULE_LOCAL)
    rows: list[dict] = []
    per_step: dict[int, dict[str, int]] = {}
    by_w: dict[str, dict[str, Any]] = {}
    plane_w: dict[str, dict[str, Any]] = {}  # F-r17b-18: the packed planes this request sees, apart from its own widths
    w_unknown: dict[str, int] = {}
    counts = {"rows": 0, "active": 0, "bot": 0, "shared": 0, "bound_active": 0, "unbound_active": 0, "bound_bot": 0, "unbound_bot": 0,
              "bound_shared": 0, "unbound_shared": 0, "plane_slices": 0, "plane_slices_active": 0, "plane_slices_bot": 0, "planes_not_this_request": 0,
              "commit_key_unresolved": 0, "plane_slices_committed_present": 0, "plane_slices_committed_absent": 0, "plane_slices_committed_unplaced": 0}
    # F-r17b-23: the record's composition and committed planes, read ONCE from the binding map (None without a map: named per row)
    rcomp = record_composition(bmap, manifest) if bmap is not None else None
    rplanes = record_planes(bmap) if bmap is not None else None
    slice_bits = 0
    slice_elements = 0
    for ident in manifest["identities"]:
        step = ident.get("step")
        s = int(step) if isinstance(step, int) else -1
        rng = ident.get("element_range") or [0, None]
        n = (int(rng[1]) - int(rng[0])) if (rng[1] is not None and rng[1] != -1) else None
        rid = str(ident.get("request_id"))
        op = str(ident.get("op_path"))
        inv = int(ident.get("invocation", 0) or 0)
        rk = int(ident.get("rank", 0) or 0)
        plane = None
        if rid == "*" and isinstance(ident.get("requests"), list) and ident["requests"] and op.startswith(HIDDEN_FAMILIES) and ident.get("engine_step") is not None:
            # a packed per-step plane: this request's slice, or not this request's row at all
            if commit_rid not in [str(x) for x in ident["requests"]]:
                counts["planes_not_this_request"] += 1
                continue
            e = int(ident["engine_step"])
            cs, crows = _composition_step(manifest, e, commit_rid)
            s = cs if cs is not None else e - int(arrive)
            sl = plane_slice(ident, manifest, commit_rid, rcomp=rcomp, rplanes=rplanes)
            plane = {"packed_plane": True, "plane_requests": [str(x) for x in ident["requests"]], "plane_member": str(ident.get("output_member")),
                     "slice_rows": crows, "slice_step_source": "composition" if cs is not None else "engine_step - arrive",
                     # F-r17b-23: the WHOLE launch's extent stays visible, but the row's element_range / n / lifted_bits below are this request's SLOT
                     "plane_whole_launch": {"element_range": list(rng), "n": n, "lifted_bits": None},
                     "lifted_slot": sl["lifted_slot"], "committed_slice": sl["committed_slice"]}
            assign = "active" if s < n_exec else "bot"
            if sl["lifted_slot"]["element_range"] is not None:
                row_rng, row_n = list(sl["lifted_slot"]["element_range"]), sl["lifted_slot"]["elements"]
            else:
                row_rng, row_n = list(rng), n
        elif s < 0 or rid == "*":
            assign = "shared"
            row_rng, row_n = list(rng), n
        else:
            assign = "active" if s < n_exec else "bot"
            row_rng, row_n = list(rng), n
        w, w_src = identity_w(ident, widths_by_spec)
        if idx is None:
            bound, ref = None, "n/a"
        elif cbound is not None:
            # the Commit's verdict for the identity the Commit expanded this row into (binding.load_required_manifest's key)
            if ident.get("family") == "weights" or op == "weights":
                k = _pop_key("*", -1, 0, "weights", "registered", 0, 0, rk)
            elif plane is not None:
                k = _pop_key(commit_rid, int(ident["engine_step"]), inv, op, "*", rng[0], rng[1], rk)
            else:
                k = _pop_key(commit_rid, s + int(arrive), inv, op, ident.get("output_member"), rng[0], rng[1], rk)
            if k in cbound:
                bound = cbound[k]
            else:
                bound = False
                counts["commit_key_unresolved"] += 1
            ref = ("registered" if bound else "unbound") if (ident.get("family") == "weights" or op == "weights") else ("bound" if bound else "unbound")
        elif ident.get("family") == "weights":
            # the weights are bound as ONE registered-input digest over the whole group (binding-map/1 kind `registered-input`, request `*`),
            # never per tensor: a per-tensor identity is "registered" when that entry exists, and it is shared by every request
            bound = registered
            ref = "registered" if registered else "unbound"
        else:
            bound = is_bound(idx, ident)
            ref = "bound" if bound else "unbound"
        live, tok_in = guard_of_step(s, n_exec, lag) if s >= 0 else ("shared", "shared")
        row = {"request_id": commit_rid, "step": s if s >= 0 else None, "engine_step": (int(ident["engine_step"]) if plane is not None else (arrive + s)) if s >= 0 else None,
               "family": ident.get("family"),
               # F-r17b-28 (revb): the INVOCATION (a shared module's call index -- RotaryEmbedding: canonical model.layers.0.self_attn.rotary_emb,
               # invocation = layer, 30 identities per member per step on SmolLM2 / 16 on OLMoE) and the rank are part of the identity: without
               # them 30 rows are byte-identical and not addressable; both are in the row and in the key
               "invocation": inv, "rank": rk,
               "op_path": ident.get("op_path"), "output_member": ident.get("output_member"), "element_range": row_rng, "n": row_n,
               "assignment": assign, "active_reference": ref,
               "lifted_word": ("0||v" if assign != "bot" else L.enc_bot(w)) if w is not None else None,
               "dtype": ident.get("dtype"), "dtype_bits": DTYPE_BITS.get(str(ident.get("dtype") or ""), None),
               "w": w, "w_source": w_src, "w_lifted": (w + 1) if w is not None else None,
               "lifted_bits": (row_n * (w + 1)) if (w is not None and row_n is not None) else None,
               "guard_live": live, "tok_in": tok_in, **(plane or {})}
        if plane is not None:
            row["plane_whole_launch"]["lifted_bits"] = (n * (w + 1)) if (w is not None and n is not None) else None
            cp = row["committed_slice"].get("present")
            counts["plane_slices_committed_present" if cp is True else ("plane_slices_committed_absent" if cp is False else "plane_slices_committed_unplaced")] += 1
        rows.append(row)
        counts["rows"] += 1
        counts[assign] += 1
        if plane is not None:
            counts["plane_slices"] += 1
            counts[f"plane_slices_{assign}"] += 1
        if ref != "n/a":
            counts[f"{'bound' if bound else 'unbound'}_{assign}"] += 1
        if s >= 0:
            ps = per_step.setdefault(s, {"identities": 0, "bound": 0, "unbound": 0, "elements": 0, "lifted_bits": 0, "plane_slices": 0, "plane_lifted_bits": 0,
                                         "plane_whole_launch_lifted_bits": 0, "plane_slices_committed_absent": 0})
            ps["identities"] += 1
            if plane is not None:
                # F-r17b-23: this request's SLOT of the plane (its rows of the lifted fixed plane) -- and the whole launch's extent beside it;
                # neither enters this request's elements / lifted_bits (F-r17b-18: the planes are counted once at the workload level)
                ps["plane_slices"] += 1
                ps["plane_lifted_bits"] += row["lifted_bits"] or 0
                ps["plane_whole_launch_lifted_bits"] += row["plane_whole_launch"]["lifted_bits"] or 0
                if row["committed_slice"].get("present") is False:
                    ps["plane_slices_committed_absent"] += 1
            else:
                ps["elements"] += n or 0
                ps["lifted_bits"] += row["lifted_bits"] or 0
            if ref != "n/a":
                ps["bound" if bound else "unbound"] += 1
        key = f"{ident.get('family')}/{ident.get('dtype')}"
        if w is None:
            w_unknown[key] = w_unknown.get(key, 0) + 1
        elif plane is not None:
            # F-r17b-18 (revb): a plane's n / lifted_bits are the WHOLE launch's (every co-scheduled request's slice), so they never enter
            # this request's widths -- summing per-request totals would count each shared plane once per extra co-scheduled request.  Kept
            # apart here (this request's VIEW of the planes) and counted ONCE at the workload level (a7_correspondence_workload.plane_widths).
            # F-r17b-23: the request's own SLOT bits beside them (slice_*): the slots of the co-scheduled requests tile the plane exactly
            pw = plane_w.setdefault(key, {"family": ident.get("family"), "dtype": ident.get("dtype"), "w": w, "w_lifted": w + 1, "w_source": w_src,
                                          "planes": 0, "plane_slices_bot": 0, "elements": 0, "lifted_bits": 0, "slice_elements": 0, "slice_lifted_bits": 0,
                                          "committed_slices_absent": 0, "bot_word": L.enc_bot(w)})
            pw["planes"] += 1
            pw["elements"] += n or 0
            pw["lifted_bits"] += row["plane_whole_launch"]["lifted_bits"] or 0
            pw["slice_elements"] += row_n or 0
            pw["slice_lifted_bits"] += row["lifted_bits"] or 0
            slice_bits += row["lifted_bits"] or 0
            slice_elements += row_n or 0
            if assign == "bot":
                pw["plane_slices_bot"] += 1
            if row["committed_slice"].get("present") is False:
                pw["committed_slices_absent"] += 1
        else:
            bw = by_w.setdefault(key, {"family": ident.get("family"), "dtype": ident.get("dtype"), "w": w, "w_lifted": w + 1, "w_source": w_src,
                                       "identities": 0, "elements": 0, "lifted_bits": 0, "bot_identities": 0, "bound_bot_identities": 0,
                                       "bot_word": L.enc_bot(w)})
            bw["identities"] += 1
            bw["elements"] += n or 0
            bw["lifted_bits"] += row["lifted_bits"] or 0
            if assign == "bot":
                bw["bot_identities"] += 1  # (e) on MoE: the router's leaves (moe_block_stream) at bot steps -- how many the Commit binds is named per family
                if ref == "bound":
                    bw["bound_bot_identities"] += 1
            if bw["w"] != w:
                bw["w"] = None  # mixed: named below
    steps_present = sorted(per_step)
    per_step_ids = sorted({v["identities"] for s, v in per_step.items() if s >= 1})
    bot_steps = [s for s in range(n_exec, cap)]
    summary = {
        "schema": SCHEMA, "label": L.LABEL, "rules_version": L.LIFTED_RULES_VERSION, "spec": "LIFTING-SPEC v0.5.1 A7",
        "request_id": commit_rid, "a_r": a_r, "cap": cap, "lag": lag, "n_exec": n_exec, "arrive": arrive,
        "program_digest": manifest.get("program_digest"), "manifest_digest": manifest.get("manifest_digest"),
        "manifest_engine": (manifest.get("query") or {}).get("engine") or "inventory", "result_json": manifest.get("result_json"),
        "binding_rule": binding_rule,
        "commit_coverage": ({k: (commit_cov or {}).get(k) for k in ("coverage", "expansion", "horizon", "arrivals")} if cbound is not None else None),
        "counts": counts,
        # F-r17b-28 (revb): the table is identity-level only if every row is one identity under the FULL key (with invocation + rank)
        "identity_key": "request_id, step, family, op_path, invocation, rank, output_member, element_range",
        "unique_keys": unique_keys(rows),
        # F-r17b-15 (2): this request's slices of the packed per-step planes, by assignment -- the inactive ones are demonstration (d)'s object
        "plane_slices": {"n": counts["plane_slices"], "active": counts["plane_slices_active"], "bot": counts["plane_slices_bot"],
                         "bound_bot": sum(1 for r in rows if r.get("packed_plane") and r["assignment"] == "bot" and r["active_reference"] == "bound"),
                         "unbound_active": sum(1 for r in rows if r.get("packed_plane") and r["assignment"] == "active" and r["active_reference"] == "unbound"),
                         "steps_bot": sorted({r["step"] for r in rows if r.get("packed_plane") and r["assignment"] == "bot"})[:64],
                         # F-r17b-23: the slices placed in the COMMITTED plane (the record's composition) vs absent there (the record did not run
                         # the request at the step: committed n_seq excludes it) vs unplaceable (no map / no composition reference)
                         "committed_present": counts["plane_slices_committed_present"], "committed_absent": counts["plane_slices_committed_absent"],
                         "committed_unplaced": counts["plane_slices_committed_unplaced"],
                         "committed_absent_are_exactly_bot": ((counts["plane_slices_committed_absent"] == counts["plane_slices_bot"]
                                                              and all(r["committed_slice"].get("present") is False for r in rows if r.get("packed_plane") and r["assignment"] == "bot")
                                                              and all(r["committed_slice"].get("present") is True for r in rows if r.get("packed_plane") and r["assignment"] == "active"))
                                                             if (bmap is not None and rcomp is not None and counts["plane_slices"]) else None),
                         "active_slices_placed_in_committed_plane": ((counts["plane_slices_committed_present"] == counts["plane_slices_active"])
                                                                     if (bmap is not None and rcomp is not None and counts["plane_slices"]) else None),
                         "slice_source": {"lifted_slot": "manifest geometry + declared composition (the lifted fixed plane)",
                                          "committed_slice": (rcomp or {}).get("source") if rcomp else "n/a (no binding map / no composition reference)"},
                         "mixed_steps_first": [{"engine_step": r["engine_step"], "op_path": r["op_path"], "assignment": r["assignment"],
                                                "lifted_slot_rows": r["lifted_slot"]["row_range"], "lifted_slot_elements": r["lifted_slot"]["element_range"],
                                                "committed": ({"rows": r["committed_slice"].get("row_range"), "of_rows": r["committed_slice"].get("rows_total"),
                                                               "n_seq": r["committed_slice"].get("n_seq"), "carrier": r["committed_slice"].get("carrier"),
                                                               "members": (r["committed_slice"].get("members") or [])[:2]}
                                                              if r["committed_slice"].get("present") else r["committed_slice"].get("why"))}
                                               for r in rows if r.get("packed_plane") and r["step"] in (0, n_exec - 1, n_exec) and str(r["op_path"]).endswith("/L0")][:6],
                         "rule": "SPEC A3.6: a packed plane's (request, step) slice takes the request's assignment at that step; bound by the Commit's "
                                 "op-level per-request rule (>= 1 plane of that op / step bound to the request), never a second rule; F-r17b-23: the "
                                 "slice is placed in the lifted fixed plane (slot) AND in the committed plane (the record's composition), or named absent"},
        "steps": {"present": len(steps_present), "expected": cap, "all_present": steps_present == list(range(cap)),
                  "identities_per_decode_step": per_step_ids, "uniform_per_decode_step": len(per_step_ids) <= 1,
                  "active": list(range(0, n_exec)), "bot": bot_steps, "transition": (n_exec if n_exec < cap else None)},
        "per_step": {str(s): {**per_step[s], "assignment": ("active" if s < n_exec else "bot"), **dict(zip(("guard_live", "tok_in"), guard_of_step(s, n_exec, lag)))}
                     for s in steps_present},
        "widths": {"by_family_dtype": by_w, "w_unknown": w_unknown, "w_lifted_max": max((v["w_lifted"] for v in by_w.values()), default=None),
                   "bot_words": sorted({v["bot_word"] for v in by_w.values() if v["bot_identities"]}),
                   "lifted_bits_total": sum(v["lifted_bits"] for v in by_w.values()),
                   "payload_bits_total": sum(v["elements"] * v["w"] for v in by_w.values() if v["w"] is not None),
                   "w_max_lifted": L.W_MAX_LIFTED, "w_max_active_reference": L.W_MAX,
                   "excludes_planes": True,
                   # F-r17b-18: the planes this request is a slice of, at the plane's (whole-launch) extent -- NOT this request's bits, and
                   # NOT to be summed across requests: the workload-level document counts each plane once (plane_widths)
                   "planes_seen": {"by_family_dtype": plane_w, "planes": sum(v["planes"] for v in plane_w.values()),
                                   "lifted_bits_whole_launch": sum(v["lifted_bits"] for v in plane_w.values()),
                                   # F-r17b-23: this request's own SLOTS of those planes (Σ over the co-scheduled requests' slots = the planes once)
                                   "slice_elements_total": slice_elements, "slice_lifted_bits_total": slice_bits,
                                   "note": "each plane's n / lifted_bits are the whole launch's (every co-scheduled request's slice); shared with the "
                                           "co-scheduled requests -- counted once at the workload level, never added to lifted_bits_total; slice_* are "
                                           "this request's slots of them (F-r17b-23)"}},
        "padding_exercised": bool(bot_steps),  # False on a length stop (a_r == cap): the population rule holds trivially, no bot row to show
        "checks": {
            # (a)(ii): the bot rows EXIST in the table -- None on a length stop (nothing to be present; A7 (a) evidence needs a record with a stop)
            "inactive_suffix_present": (all(s in per_step for s in bot_steps) if bot_steps else None),
            "no_active_identity_unbound": (counts["unbound_active"] == 0) if idx is not None else None,  # a coverage hole of the executed population
            "bot_population_bound": (counts["bound_bot"] == counts["bot"] and counts["bot"] > 0) if (idx is not None and bot_steps) else None,
            "every_row_widened": all(r["w_lifted"] == (r["w"] or 0) + 1 for r in rows if r["w"] is not None) and not w_unknown,
            # F-r17b-28: identity-level means one row per identity under the full key (invocation + rank included)
            "every_row_one_identity": len({_key(r) for r in rows}) == len(rows),
        },
        "bot_rows_first": [r for r in rows if r["assignment"] == "bot"][:12],
    }
    summary["digest"] = table_digest(rows)
    return rows, summary


def table_digest(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def write_table(rows: Iterable[dict], path: str | Path, summary: dict | None = None) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "wt") as f:  # type: ignore[operator]
        if summary is not None:
            f.write(json.dumps({"$summary": summary}, default=str) + "\n")
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return str(p)


def read_table(path: str | Path) -> tuple[dict | None, list[dict]]:
    op = gzip.open if str(path).endswith(".gz") else open
    summary, rows = None, []
    with op(path, "rt") as f:  # type: ignore[operator]
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "$summary" in d:
                summary = d["$summary"]
            else:
                rows.append(d)
    return summary, rows


def _key(r: dict) -> tuple:
    """The identity of a row: request, step, family, op_path, INVOCATION, RANK, member, element range (F-r17b-28: a shared module's call index
    is part of the identity; without it the 30 RotaryEmbedding rows of a SmolLM2 step collapse to one key)."""
    return (r["request_id"], r["step"], r["family"], r["op_path"], int(r.get("invocation") or 0), int(r.get("rank") or 0), r["output_member"], tuple(r["element_range"]))


def _key_json(k: tuple) -> list:
    return [*k[:7], list(k[7])]


def unique_keys(rows: Sequence[dict]) -> dict[str, Any]:
    """Table invariant (F-r17b-28): every row is one identity -- len(rows) == len(set(keys)); the duplicates, if any, are named by key."""
    seen: dict[tuple, int] = {}
    for r in rows:
        k = _key(r)
        seen[k] = seen.get(k, 0) + 1
    dup = [(_key_json(k), c) for k, c in seen.items() if c > 1]
    return {"rows": len(rows), "unique_keys": len(seen), "every_row_one_identity": len(seen) == len(rows), "duplicate_keys_n": len(dup), "duplicate_keys_first": dup[:8]}


def compare(a: Sequence[dict], b: Sequence[dict], sa: dict | None = None, sb: dict | None = None) -> dict:
    """A7 (a): two records of ONE Build with different stopping points have the SAME identity population (same rows modulo `assignment`
    and `active_reference`) and differ in assignment on EXACTLY the steps between the two `n_exec`."""
    ka = {_key(r): r for r in a}
    kb = {_key(r): r for r in b}
    only_a, only_b = sorted(set(ka) - set(kb), key=str), sorted(set(kb) - set(ka), key=str)
    same_keys = set(ka) & set(kb)
    struct_diff = [k for k in same_keys if {x: ka[k][x] for x in ("n", "w", "w_lifted", "lifted_bits", "dtype")} != {x: kb[k][x] for x in ("n", "w", "w_lifted", "lifted_bits", "dtype")}]
    assign_diff_steps = sorted({k[1] for k in same_keys if ka[k]["assignment"] != kb[k]["assignment"] and k[1] is not None})
    na, nb = (sa or {}).get("n_exec"), (sb or {}).get("n_exec")
    want = sorted(range(min(na, nb), max(na, nb))) if (na is not None and nb is not None) else None
    ua, ub = unique_keys(a), unique_keys(b)
    return {"schema": SCHEMA + "/compare", "label": L.LABEL, "rules_version": L.LIFTED_RULES_VERSION,
            "rows": {"a": len(a), "b": len(b)}, "keys": {"a": len(ka), "b": len(kb)},
            "every_row_one_identity": {"a": ua["every_row_one_identity"], "b": ub["every_row_one_identity"]},
            "key": "request_id, step, family, op_path, invocation, rank, output_member, element_range (F-r17b-28: judged over every row, one identity each)",
            "population_equal": not only_a and not only_b and not struct_diff and ua["every_row_one_identity"] and ub["every_row_one_identity"],
            "only_in_a": [_key_json(k) for k in only_a[:8]], "only_in_b": [_key_json(k) for k in only_b[:8]], "structural_differences": [_key_json(k) for k in struct_diff[:8]],
            "a_r": {"a": (sa or {}).get("a_r"), "b": (sb or {}).get("a_r")}, "n_exec": {"a": na, "b": nb},
            "program_digest_equal": (sa or {}).get("program_digest") == (sb or {}).get("program_digest") if sa and sb else None,
            "manifest_digest_equal": (sa or {}).get("manifest_digest") == (sb or {}).get("manifest_digest") if sa and sb else None,
            "assignment_differs_on_steps": assign_diff_steps, "expected_differing_steps": want,
            "assignment_differs_exactly_between_the_stops": (assign_diff_steps == want) if want is not None else None,
            "ok": (not only_a and not only_b and not struct_diff and ua["every_row_one_identity"] and ub["every_row_one_identity"]
                   and (want is None or assign_diff_steps == want))}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--build-request", required=True)
    b.add_argument("--commit", default=None)
    b.add_argument("--pair", type=int, default=0)
    b.add_argument("--commit-request-id", default="r0")
    b.add_argument("--a-r", type=int, required=True, help="the served prefix (R2 of lifted_record)")
    b.add_argument("--cap", type=int, required=True)
    b.add_argument("--lag", type=int, required=True, help="the DECLARED lag (R0 of lifted_record; never inferred here)")
    b.add_argument("--arrive", type=int, default=0)
    b.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        sa, ra = read_table(a.a)
        sb, rb = read_table(a.b)
        rep = compare(ra, rb, sa, sb)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: rep[k] for k in ("population_equal", "assignment_differs_on_steps", "expected_differing_steps", "ok")}))
        return 0 if rep["ok"] else 1
    from verity_vllm.lifting.record import load_descriptor
    bmap = None
    if a.commit:
        bp = Path(a.commit) / f"binding_map_p{a.pair}.json"
        if bp.exists():
            bmap = json.loads(bp.read_text())
        elif bp.with_suffix(".json.gz").exists():
            with gzip.open(bp.with_suffix(".json.gz"), "rt") as f:
                bmap = json.load(f)
    man = required_identities(a.build_request, request_id=a.commit_request_id, arrive=a.arrive)
    params = None
    ip = Path(a.build_request) / "instances.json.gz"
    if ip.exists():
        with gzip.open(ip, "rt") as f:
            params = [(str(n), str(t)) for n, t in (json.load(f).get("params") or [])] or None
    wbs = spec_leaf_widths(load_descriptor(a.build_request), params)
    rows, summ = correspondence(man, bmap, commit_rid=a.commit_request_id, a_r=a.a_r, cap=a.cap, lag=a.lag, arrive=a.arrive, widths_by_spec=wbs)
    write_table(rows, a.out, summ)
    print(json.dumps({"rows": summ["counts"], "checks": summ["checks"], "digest": summ["digest"][:16], "manifest_digest": (summ["manifest_digest"] or "")[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
