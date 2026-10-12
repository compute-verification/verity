"""GP-01 driver: build the workload Program of a declared manifest from the derived request-wrapper Programs.

    verity-vllm global-program --workload <manifest.json> --derived <dir> [--derived <dir> ...] --out <dir>

Inputs (pre-execution only): the request manifest (request ids, prompt lengths, caps, `arrive_step`, the row's declared `target`
block) and derive_step request artifacts (`descriptor.json.gz` + `construction_manifest.json` / `instances.json.gz`) — one per
distinct (prompt_len, cap-1) shape, exactly the shapes X-09's Build derives. Nothing from an execution record is read. Exit 4 lists
the shapes still to derive.  [R17-1] `LAG` (the declared scheduler's post-EOS rows, `TargetProfile.lag()` of the manifest's target) is a
REQUIRED static of the workload Program (`Workload_v1{requests, shared, lag}`): the digest differs per value and GM-01 G6 consumes it.

Writes: `workload_program.json` (artifact: digest, components, address map, chronology), `descriptor.json.gz` (the workload
Program's descriptor, decodable with verity.ml.circuits.codec), `address_map.json`, `chronology.json`.

Layout 1.1 (`verity_vllm.pipeline.layout`): beside the absolute `inputs.manifest` / `inputs.derived` paths of the build machine, the
artifact carries a `layout` block with the same paths relative to the artifact root (`--artifact-root`, default the parent of
`--out`, i.e. the row dir `build_workload/` sits in), so a Build fetched elsewhere resolves its components without being laid out
at the original path.  Readers go through `build_paths.component_dir` / `manifest_path`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

from verity.ml.circuits.codec import canonical_json, descriptor_size
from verity.ml.circuits.defs import REGISTRY
from verity_vllm.config import option
from verity_vllm.pipeline import layout as build_paths
from verity_vllm.program import workload as W
from verity_vllm.program.registry.catalog import load as _load_registry

# the request descriptors decode through the registry the Match reads (`program_compare._registry`): every registry module loaded
for _e in _load_registry()[1]:
    print(f"[global_program] registry module unavailable ({_e}): components naming its Definitions will not decode", file=sys.stderr)

from verity_vllm.check.match.batch_decomp import derived_shape
from verity_vllm.pipeline.components import (component_of, component_provenance, follow_components, load_component, load_desc,  # noqa: F401
                                             recorded_component_digest, registration_refusal, wrapper_params)


def request_id_of(index: int) -> str:
    """[R15 COORD 19:25Z ruling 2] the request id everywhere (manifest, binding map, Program subcircuit names, records) is `r<k>`,
    k = the request's 0-based position in the workload manifest. The manifest's `request_id` field (a UUID, the id vLLM serves the
    request under) is the WORKLOAD request id: kept beside as `workload_request_id`, never used as the Program's request identity."""
    return f"r{int(index)}"


def manifest_requests(m: dict) -> list[dict]:
    reqs = m.get("requests") or (m.get("workload") or {}).get("requests") or []
    out = []
    for i, r in enumerate(reqs):
        idx = int(r.get("index", i))
        if idx != i:
            raise ValueError(f"manifest request {i} declares index {idx}: requests must be listed in index order (r<k> = position k)")
        wrid = r.get("request_id") or r.get("id") or None
        lp = r.get("prompt_len") or len(r.get("prompt_token_ids") or ())
        cap = r.get("max_tokens") or (r.get("sampling_params") or {}).get("max_tokens")
        if not lp or not cap:
            raise ValueError(f"manifest request {wrid or i}: no prompt_len / max_tokens")
        out.append({"request_id": request_id_of(i), "workload_request_id": wrid, "index": i, "prompt_len": int(lp), "cap": int(cap),
                    "arrive_step": int(r.get("arrive_step", 0) or 0)})
    return out


def execution_of_workload(m: dict) -> str:
    """The row's declared execution label: `execution` / `sweep.execution` when present, else the row-id suffix (`…__bi-eager`,
    `…__bi-compiled`, `…__bi-eager-fa2`) -- the same rule as exec's EXEC-02 `execution_of_workload` (M-0078), kept local so the
    Build does not import the adapter."""
    ex = m.get("execution") or (m.get("sweep") or {}).get("execution")
    if ex:
        return str(ex)
    rid = (m.get("sweep") or {}).get("row_id") or m.get("row_id") or ""
    return rid.rsplit("__", 1)[-1] if "__" in rid else ""


# engine conditions the workload Program models (COORD M-0063 (1)): the components are eager per-request Programs (X-09) over one shared
# weights input; no cross-request cache reuse (G5 requires 0 cross-request reads); a request's prefill may be chunked over engine steps
# (X-09's chunked-prefill criterion) -- so chunked prefill is modelled, prefix caching is not.
MODELLED_ENGINE = {"enforce_eager": True, "enable_prefix_caching": False, "enable_chunked_prefill": "modelled (a request's prefill rows may span engine steps)"}

# [compiled M-0153 (4a) / M-0141] a COMPILED execution is modelled only when every component declares it (applicability.execution
# 'bi-compiled…' AND an agreed `compilation` block {mode, cudagraph_mode, capture_sizes, …} written by derive_step from the TargetProfile);
# the engine conditions then say enforce_eager False + those facts, and GM-01 G1 compares the record's header against them.
COMPILATION_FACT_KEYS = ("mode", "cudagraph_mode", "capture_sizes", "backend", "use_inductor_graph_partition", "compile_cache",
                         # [compiled M-0219] the record-level pin of the Inductor output identity (control process): carried into
                         # global_match.json execution.effective_compilation when the header attests it, NOT compared with the Build
                         # (derive_step traces eagerly; the pin is an attempt-to-attempt condition the report/commit compare)
                         "kernel_set_sha256", "call_set_sha256", "wrapper_norm_set_sha256")
_FACT_ALIASES = {"compilation_mode": "mode"}   # compiled's TargetProfile.modelled_engine() names it compilation_mode (M-0174)


def compilation_facts(block: dict | None) -> dict:
    """The compilation facts of an applicability `engine`/`compilation` block, key-normalised (compilation_mode -> mode); {} if none."""
    if not isinstance(block, dict):
        return {}
    out = {}
    for k, v in block.items():
        kk = _FACT_ALIASES.get(k, k)
        if kk in COMPILATION_FACT_KEYS and v is not None:
            out[kk] = _MODE_NAMES.get(str(v), v) if kk == "mode" else v
    return out


# vLLM CompilationMode: the recorder's profile.compilation writes the enum VALUE as a string ("0" on an eager record), compiled's
# TargetProfile.modelled_engine() the NAME ("VLLM_COMPILE"); one spelling so the record's own sources are comparable (M-0261)
_MODE_NAMES = {"0": "NONE", "1": "STOCK_TORCH_COMPILE", "2": "DYNAMO_TRACE_ONCE", "3": "VLLM_COMPILE"}


def eos_lag_rows_declared(engine: dict | None) -> int | None:
    """[M-0213 / ruling 21 -> R17-1] the post-EOS rows a DECLARED engine block computes: async_scheduling True -> 1, False -> 0, undeclared /
    non-boolean -> None.  Under R17-1 GP-01 always declares it (`workload_lag`), so None here means a pre-R17-1 Program: GM-01 G6 then FAILS
    by name on any EOS stop (nothing is read from the record as the rule) and labels a record without an EOS stop `lag.exercised false`."""
    if not isinstance(engine, dict):
        return None
    v = engine.get("async_scheduling")
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return 1 if v.strip().lower() == "true" else 0
    return None


def workload_target(manifest: dict):
    """[R17-1] The row's declared TargetProfile as GP-01 reads it: the manifest's `target` block (the ONE declared target source; row_pod
    passes the same block to derive_step as --target and every GPU stage builds its engine for it) over the default profile.  A GP-01
    compose has no GPU, so cc / SMs are whatever the block says (they do not enter the lag); an unknown field is refused by name."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    t = manifest.get("target") if isinstance(manifest.get("target"), dict) else {}
    unknown = set(t) - set(TargetProfile.__dataclass_fields__)
    if unknown:
        raise ValueError(f"workload target: unknown TargetProfile fields {sorted(unknown)}")
    d = TargetProfile.default().to_json()
    d.update({k: v for k, v in t.items() if v is not None})
    return TargetProfile.from_json(d)


def component_declared_target(d: str) -> dict | None:
    """derive_step's `target.declared` (result.json, v3) for a component dir: the TargetProfile the component was DERIVED under; None
    when the artifact does not carry one (pre-v3)."""
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return None
    try:
        t = json.load(open(p)).get("target")
    except Exception:  # noqa: BLE001
        return None
    return t.get("declared") if isinstance(t, dict) and isinstance(t.get("declared"), dict) else None


def workload_lag(manifest: dict, shapes: dict) -> dict:
    """[R17-1 / DP-pad-02 (b)] The workload Program's REQUIRED `lag` declaration: LAG = TargetProfile.lag() of the row's declared target
    (the resolved async-scheduling configuration of the serving profile; 1 under async scheduling, 0 synchronous), with the source and
    the implementation version pinned.  Coupling checks (fail closed, named): every component's derive-time `target.declared` must
    resolve to the SAME value (a component derived for a synchronous profile does not compose into an async workload); `refusal` names
    the first disagreement.  The record's resolved read-back is compared by GM-01 G1/G6 -- never read here (a Build reads no execution)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    try:
        tp = workload_target(manifest)
    except (TypeError, ValueError) as e:
        return {"schema": "verity/workload-program/v1/lag", "value": None, "refusal": f"the manifest's target block does not parse as a TargetProfile: {e}"}
    decl = tp.async_scheduling_declaration()
    out = {"schema": "verity/workload-program/v1/lag", **decl,
           "rule": "R17-1: LAG is a required declared specialization parameter of the workload Program (static `lag` of Workload_v1, in the "
                   "digest), coupled to the serving profile's RESOLVED async_scheduling; Build and GM-01 consume this declaration, the "
                   "record header must attest the same resolved value (G1), never inferred from the trace; the post-EOS step under LAG=1 "
                   "is active internal computation whose sampled token is not a served output",
           "components": {}, "refusal": None}
    for s, d in sorted(shapes.items()):
        ct = component_declared_target(d)
        key = f"{s[0]},{s[1]}"
        if ct is None:
            out["components"][key] = {"dir": d, "declared_target": None, "lag": None, "note": "component carries no target.declared (pre-v3 derive): not checked"}
            continue
        try:
            unknown = set(ct) - set(TargetProfile.__dataclass_fields__)
            if unknown:
                raise ValueError(f"unknown TargetProfile fields {sorted(unknown)}")
            cp = TargetProfile.from_json(ct)
        except (TypeError, ValueError) as e:
            out["refusal"] = f"component (LP={s[0]}, T={s[1]}) {d}: target.declared does not parse as a TargetProfile: {e}"
            return out
        out["components"][key] = {"dir": d, "async_scheduling": cp.effective_async_scheduling(), "lag": cp.lag(),
                                  "declared_explicitly": cp.async_scheduling is not None, "vllm_revision": cp.vllm_revision}
        if cp.lag() != decl["value"]:
            out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d} was derived under a target resolving async_scheduling={cp.effective_async_scheduling()} "
                              f"(LAG {cp.lag()}) but the workload's declared target resolves async_scheduling={decl['async_scheduling']} (LAG {decl['value']}): "
                              f"one workload Program cannot be specialized for two schedulers (R17-1)")
            return out
        if cp.vllm_revision != tp.vllm_revision:
            out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d} was derived for vLLM {cp.vllm_revision} but the workload's target pins "
                              f"{tp.vllm_revision}: the lag declaration is coupled to the pinned implementation (R17-1)")
            return out
    return out


#: [R17-42] the Definition ids (name prefixes of `spec_histogram` keys) that identify each construction's binding of the fused-MoE block
_MOE_CONSTRUCTION_SPECS = {"indexed-read": ("MoeExpertGemm_v1", "MoeExpertGemmW_v1", "MoeExpertGemm_v2", "MoeExpertGemmW_v2", "MoeRouterTopK_v1", "MoeRouterTopKNorm_v1", "MoeSum_v1"),
                           "indexed-read-ordered": ("MoeExpertGemm_v1", "MoeExpertGemmW_v1", "MoeExpertGemm_v2", "MoeExpertGemmW_v2", "MoeRouterTopKOrdered_v1", "MoeRouterTopKOrderedNorm_v1", "MoeSum_v1"),
                           "padded": ("MoeFfnFixedL_v1",)}


def component_spec_histogram(d: str) -> dict | None:
    """derive_step's `spec_histogram` (result.json): Definition id -> call count of the component's Program; None when absent."""
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return None
    try:
        h = json.load(open(p)).get("spec_histogram")
    except Exception:  # noqa: BLE001
        return None
    return h if isinstance(h, dict) else None


def workload_moe_construction(manifest: dict, shapes: dict) -> dict:
    """[R17-42 / R17-45 (a) route A (3)] The workload Program's `moe_construction` header block: the fused-MoE construction the row's
    declared target binds (`TargetProfile.effective_moe_construction()`: `indexed-read[-ordered]` = MOE-01, `padded` = one `MoeFfnFixedL_v1` per token),
    a Program STATIC that is declared-and-bound and NEVER attested (the record is construction-agnostic; two Programs of the same trace differ
    in digest by construction).  Coupling checks (fail closed, named): every component's derive-time `target.declared` resolves to the SAME
    construction as the manifest's target, and every component that binds a fused-MoE block binds it with THAT construction's Definitions
    (its `spec_histogram` names `MoeFfnFixedL_v1` under padded / the MOE-01 ids under indexed-read, never the other family's); an explicitly
    declared `padded` on a workload with no MoE block anywhere is an idle declaration and is refused (nothing was constructed that way).
    `moe_block_present` labels the workload; `population_label` is what PADDED-TABLE reports the row under (R17-45 (1): actual population)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    try:
        tp = workload_target(manifest)
    except (TypeError, ValueError) as e:
        return {"schema": "verity/workload-program/v1/moe_construction", "value": None, "refusal": f"the manifest's target block does not parse as a TargetProfile: {e}"}
    decl = tp.moe_construction_declaration()
    out = {"schema": "verity/workload-program/v1/moe_construction", **decl,
           "rule": "R17-42 / R17-45 (a): the fused-MoE construction is a declared-and-bound Program static (TargetProfile.moe_construction; "
                   "indexed-read = MOE-01, padded = MoeFfnFixedL_v1 per token with the hand-authored lift LMoeFfnL_v1 of record); the Build "
                   "binds the declared construction only, the digest differs by construction on the same trace, GP-01 refuses components of "
                   "another construction by name; never attested by the record, never inferred from it (moe M-1066 (2a))",
           "components": {}, "moe_block_present": False, "refusal": None}
    own, other = _MOE_CONSTRUCTION_SPECS[decl["value"]], {s for v in _MOE_CONSTRUCTION_SPECS.values() for s in v} - set(_MOE_CONSTRUCTION_SPECS[decl["value"]])
    for s, d in sorted(shapes.items()):
        key = f"{s[0]},{s[1]}"
        ct = component_declared_target(d)
        hist = component_spec_histogram(d) or {}
        names = {k.split("{", 1)[0] for k in hist}
        has_own, has_other = sorted(names & set(own)), sorted(names & set(other))
        comp = {"dir": d, "moe_construction": None, "binds_declared_construction": bool(has_own), "binds_other_construction": has_other}
        if ct is not None:
            try:
                unknown = set(ct) - set(TargetProfile.__dataclass_fields__)
                if unknown:
                    raise ValueError(f"unknown TargetProfile fields {sorted(unknown)}")
                cp = TargetProfile.from_json(ct)
            except (TypeError, ValueError) as e:
                out["refusal"] = f"component (LP={s[0]}, T={s[1]}) {d}: target.declared does not parse as a TargetProfile: {e}"
                return out
            comp["moe_construction"] = cp.effective_moe_construction()
            comp["declared_explicitly"] = cp.moe_construction is not None
            if cp.effective_moe_construction() != decl["value"]:
                out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d} was derived under moe_construction={cp.effective_moe_construction()!r} but the "
                                  f"workload's declared target says {decl['value']!r}: one workload Program is of ONE construction (R17-42)")
                out["components"][key] = comp
                return out
        else:
            comp["note"] = "component carries no target.declared (pre-v3 derive): the construction is read from its Definitions only"
        if has_other:
            out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d} binds the fused-MoE block with {has_other} -- the "
                              f"{'/'.join(k for k, v in _MOE_CONSTRUCTION_SPECS.items() if set(has_other) & set(v))} construction -- while the workload declares "
                              f"{decl['value']!r}: refused by name (a Program is of one construction; the digest would be another row's)")
            out["components"][key] = comp
            return out
        out["components"][key] = comp
        out["moe_block_present"] = out["moe_block_present"] or bool(has_own)
    if decl["value"] == "padded" and not out["moe_block_present"]:
        out["refusal"] = ("the target declares moe_construction='padded' but no component binds a fused-MoE block (no MoeFfnFixedL_v1 anywhere): "
                          "an idle declaration is not a construction -- refused (a dense workload has nothing to pad; do not declare it)")
        return out
    out["population_label"] = (("padded fixed-slot MoE: |Q(Lift(P_fixed-slot))|, labelled with its ACTUAL population (R17-45 (1))" if decl["value"] == "padded"
                                else f"{decl['value']} MoE (MOE-01): |Q(P)|") if out["moe_block_present"] else "no fused-MoE block (dense workload)")
    return out


def modelled_engine(execution: str, comps: dict) -> tuple[dict, str | None]:
    """The engine conditions the workload Program models for the components' agreed execution label; (engine, refusal).

    Eager: GP-01's own conditions (MODELLED_ENGINE). Compiled ('…compiled…', compiled M-0153 (4a) / M-0174): the components' declared
    `applicability.engine` blocks (derive_step writes profile.modelled_engine(): enforce_eager False + compilation_mode/backend/cudagraph_mode/…)
    or their `applicability.compilation` dicts must AGREE and carry compilation facts (at least cudagraph_mode); the workload engine block
    is then enforce_eager False + `compilation` = the normalised facts (what GM-01 G1 compares with the record header) + GP-01's own
    conditions (no prefix caching; chunked prefill modelled)."""
    ex = str(execution or "")
    # [compiled M-0174 (a)] the components' agreed applicability.engine block (derive_step: profile.modelled_engine()) is carried verbatim
    # into the workload engine block (claim text, program_derivation, attention note...); GP-01's own compose-level conditions stay
    # authoritative for what GP-01 models (no prefix caching; chunked prefill modelled) and enforce_eager follows the execution label.
    blocks = [a.get("engine") for a in comps.values() if a and isinstance(a.get("engine"), dict)]
    encs_b = {json.dumps(b, sort_keys=True, default=str) for b in blocks}
    if len(encs_b) > 1:
        return dict(MODELLED_ENGINE), f"components disagree on applicability.engine {sorted(encs_b)}"
    declared = json.loads(encs_b.pop()) if encs_b else {}
    if declared.get("enable_prefix_caching") is True:
        return dict(MODELLED_ENGINE), "a component's applicability.engine declares enable_prefix_caching true, which GP-01 does not model"
    base = {**declared, **MODELLED_ENGINE}
    if "compiled" not in ex:
        if declared.get("enforce_eager") is False:
            return dict(MODELLED_ENGINE), f"components declare the eager execution {ex!r} but an engine block with enforce_eager false"
        return ({**base, "enforce_eager": True}, None)
    facts = []
    for a in comps.values():
        if not a:
            continue
        f = compilation_facts(a.get("engine")) or compilation_facts(a.get("compilation"))
        facts.append(f)
    if not facts or any(not f or "cudagraph_mode" not in f for f in facts):
        return ({**base, "enforce_eager": False, "compilation": None},
                f"components declare the compiled execution {ex!r} but no compilation facts (applicability.engine / applicability.compilation with "
                f"cudagraph_mode, compilation_mode|mode, capture_sizes, backend): the Program cannot state the compiled execution it models")
    if declared.get("enforce_eager") is True:
        return ({**base, "enforce_eager": False, "compilation": None},
                f"components declare the compiled execution {ex!r} but an engine block with enforce_eager true")
    encs = {json.dumps(f, sort_keys=True, default=str) for f in facts}
    if len(encs) > 1:
        return ({**base, "enforce_eager": False, "compilation": None}, f"components disagree on the compilation facts {sorted(encs)}")
    return ({**base, "enforce_eager": False, "compilation": json.loads(encs.pop())}, None)


def component_applicability(d: str) -> dict:
    """derive_step's `applicability` block (result.json / artifact) for a component dir."""
    for name in ("result.json", "artifact.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            j = json.load(open(p))
            a = j.get("applicability") or (j.get("artifact") or {}).get("applicability") or {}
            if a:
                return a
    return {}


def workload_applicability(m: dict, comps: dict) -> dict:
    """The workload Program's declared applicability = the components' (all must agree on the execution profile) + the engine
    conditions above + the row's execution label.  `refusal` names why the Program does NOT model this row (fail closed)."""
    execs = {str(a.get("execution")) for a in comps.values() if a}
    row_ex = execution_of_workload(m)
    keys = ("execution", "numerics_profile", "gemm_target", "tuned_gemm_arch_family", "tuned_gemm_table_sha256", "attention_backend",
            "flash_attn_version_applied", "sampler", "num_sms", "vllm_version", "torch_version", "triton_version", "engine")
    agreed, disagreed = {}, {}
    for k in keys:
        vals = {json.dumps(a.get(k), sort_keys=True, default=str) for a in comps.values() if a}
        if len(vals) == 1:
            agreed[k] = json.loads(vals.pop())
        elif vals:
            disagreed[k] = sorted(vals)
    refusal = None
    engine = dict(MODELLED_ENGINE)
    if not execs or execs == {"None"}:
        refusal = "components declare no execution profile (derive_step applicability.execution missing): cannot certify the row's execution"
    elif len(execs) > 1:
        refusal = f"components declare different execution profiles {sorted(execs)}"
    elif row_ex and row_ex not in execs:
        refusal = f"row execution {row_ex!r} is not modelled: the components model {sorted(execs)} only (eager Programs are not evidence for a compiled row)"
    elif disagreed:
        refusal = f"components disagree on applicability fields {sorted(disagreed)}"
    else:
        engine, refusal = modelled_engine(sorted(execs)[0], comps)
    return {"schema": "verity/workload-program/v1/applicability", "execution": sorted(execs)[0] if len(execs) == 1 else sorted(execs),
            "row_execution": row_ex, "engine": engine, "components": agreed, "components_disagree": disagreed or None,
            "refusal": refusal}


def component_sampling(d: str) -> dict:
    """derive_step's `sampling` block for a component dir (result.json `sampling`, else the construction manifest's structural input)."""
    for name, path in (("result.json", ("sampling",)), ("construction_manifest.json", ("structural_inputs", "sampling"))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                j = json.load(open(p))
            except Exception:  # noqa: BLE001
                continue
            for k in path:
                j = (j or {}).get(k)
            if j:
                return j
    return {}


def declared_live_counts(reqs: list[dict]) -> list[int]:
    """The DECLARED chronology's live count per engine step: n_live(e) = #{r : arrive_r <= e < arrive_r + cap_r} (every request to its cap)."""
    horizon = max((int(r["arrive_step"]) + int(r["cap"]) for r in reqs), default=0)
    n = [0] * horizon
    for r in reqs:
        for e in range(int(r["arrive_step"]), int(r["arrive_step"]) + int(r["cap"])):
            n[e] += 1
    return n


def sampler_geometry(app: dict, reqs: list[dict]) -> dict | None:
    """[COORD M-0169 on M-0571 (3)] The workload Program's declaration of the top-p split geometry as a WORKLOAD-level derived value:
    S_t = SplitsFor_v1(|live(t)|, num_SMs), with |live(t)| the requests holding a logits row at engine step t (the declared chronology
    and the verified EOS stops GM-01 G6 establishes) and num_SMs the components' agreed target constant (`applicability.num_sms`).
    vLLM computes S on the host per sampler launch (`registry/topp_geometry.splits_for`), so no fold instance carries the relation: GM-01
    evaluates it per engine step against every top-p TokenSelect's launch S (`global_match.sampler_geometry_check`).  Declared for a
    stochastic sampler only (None for greedy); a stochastic sampler without an agreed num_SMs is a named refusal (the relation cannot be
    evaluated).  The declared schedule (every request to its cap) is carried for the reader; the check uses the REALIZED live count.
    Not part of the Program digest (a declaration beside `applicability`, like `stop_rule`)."""
    comps = app.get("components") or {}
    samp = comps.get("sampler") if isinstance(comps.get("sampler"), dict) else {}
    if (samp.get("kind") or "greedy") != "stochastic":
        return None
    nsm = comps.get("num_sms")
    if nsm is None:
        nsm = (samp.get("geometry") or {}).get("num_sms") if isinstance(samp.get("geometry"), dict) else None
    out = {"schema": "verity/workload-program/v1/sampler-geometry", "relation": "SplitsFor_v1", "num_sms": nsm,
           "rule": "S_t = SplitsFor_v1(|live(t)|, num_SMs) per engine step t; |live(t)| = the requests with a logits row at t (declared chronology "
                   "and verified EOS stops, G6); every top-p TokenSelect of the fold at t carries S_t (its launch S); GM-01 fails by name otherwise",
           "source": "verity_vllm.program.registry.topp_split.splits_for (vLLM _apply_topp_split doubling loop); num_SMs = applicability.components.num_sms",
           "component_splits": samp.get("splits")}
    if nsm is None:
        out["refusal"] = ("stochastic sampler but the components declare no num_SMs (applicability.num_sms): S_t = SplitsFor_v1(|live(t)|, num_SMs) "
                          "cannot be evaluated -- the Program does not model this row's sampler geometry (M-0571 (3))")
        return out
    from verity_vllm.program.registry.topp_geometry import splits_for
    live = declared_live_counts(reqs)
    out["declared_live_counts"] = live
    out["declared_schedule"] = [int(splits_for(n, int(nsm))) if n > 0 else None for n in live]
    out["declared_schedule_note"] = "S per engine step when every request runs to its cap; a verified EOS lowers |live(t)| from that step on and the check follows the record"
    return out


def sampler_schedule_binding(reqs: list[dict], shapes: dict[tuple[int, int], str]) -> dict:
    """[rev-pm M-0560 (1) on DECISION M-0529 (A)] A stochastic component whose sampler split count S is a per-EVENT schedule computed from
    the declared chronology (derive_step `sampling.geometry.request` = the arrival it was computed for, `sampling.splits` = [S_0..S_T]) is
    the Program of ONE request: compose must not reuse it for a request with the same shape (LP, T) but another arrival (its S-schedule
    differs and its Match would fail for the wrong reason).  Checked here per binding: (a) every request bound to the component arrives
    at the step the component declares; (b) a list schedule has T+1 entries; (c) the component's declared live counts are the manifest's
    (a component derived against another manifest of the same shape does not compose); (d) a constant S (the single-request form,
    SplitsFor_v1(1, num_SMs)) binds only in a manifest of one request.  Refusal is named; greedy and derived-S components bind by shape."""
    out = {"checked": {}, "refusal": None}
    for s, d in shapes.items():
        samp = component_sampling(d)
        geom, splits = samp.get("geometry") or {}, samp.get("splits")
        req = geom.get("request")
        if type(splits) is int and len(reqs) != 1:
            out["refusal"] = f"component (LP={s[0]}, T={s[1]}) {d}: S = {splits} is the single-request constant, but this manifest has {len(reqs)} requests (S_t follows the live set: the derived-S component)"
            return out
        if not req and not isinstance(splits, list):
            continue
        bound = [r for r in reqs if (r["prompt_len"], r["cap"] - 1) == s]
        rec = {"component": d, "declared_arrive_step": (req or {}).get("arrive_step"), "bound_requests": [r["request_id"] for r in bound],
               "splits": splits if not isinstance(splits, list) else f"schedule[{len(splits)}]"}
        out["checked"][f"{s[0]},{s[1]}"] = rec
        if req is not None:
            for r in bound:
                if int(r["arrive_step"]) != int(req.get("arrive_step", -1)):
                    out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d} declares its per-event S-schedule for the request arriving at engine "
                                      f"step {req.get('arrive_step')} but the manifest binds {r['request_id']} (arrive_step {r['arrive_step']}) to it: "
                                      f"the same shape with another arrival has another S-schedule (M-0529 (A)); one component per request here")
                    return out
        if isinstance(splits, list) and len(splits) != s[1] + 1:
            out["refusal"] = f"component (LP={s[0]}, T={s[1]}) {d}: S-schedule has {len(splits)} entries, the request has T+1 = {s[1] + 1} sampling events"
            return out
        dl = geom.get("declared_live_counts")
        if dl is not None and [int(x) for x in dl] != (n_live := declared_live_counts(reqs)):
            out["refusal"] = (f"component (LP={s[0]}, T={s[1]}) {d}: its S-schedule was computed from live counts {list(dl)[:12]}... but this "
                              f"manifest's declared chronology has {n_live[:12]}...: the component was derived against another workload")
            return out
    return out


def stop_rule(checkpoint_dir: str | None, eos_token_ids: list[int] | None, manifest: dict | None = None) -> dict:
    """[rev-pm ALERT M-0537 / F-pm-24] The STOP RULE the workload Program declares for a request that ends before its cap: the model's
    EOS token ids, read from the checkpoint the verifier pinned (config.json / generation_config.json `eos_token_id`, int or list) or
    given explicitly.  The bounded request Program has no EOS semantics of its own, so GM-01 needs this declaration to verify that an
    early stop IS an EOS stop (served[-1] in the set) and not a dropped / aborted / stop-string request recorded as `finish_reason=stop`.
    Not part of the Program digest (a declaration beside `applicability`); a Program without it makes GM-01 FAIL CLOSED on any
    request that stopped early."""
    ids: set[int] = set(int(x) for x in (eos_token_ids or []))
    sources: dict[str, list[int]] = {}
    if eos_token_ids:
        sources["--eos-token-ids"] = sorted(ids)
    # the manifest's sampling block may add per-workload stop ids (vLLM stop_token_ids) or switch EOS off (ignore_eos: every request
    # must then reach its cap -- any early stop FAILS); both are declared here so GM-01 reads one rule [rev-pm M-0548 (2), M-0560 (2)]
    samp = (manifest or {}).get("sampling") or {}
    if checkpoint_dir:
        for fn in ("generation_config.json", "config.json"):
            p = os.path.join(checkpoint_dir, fn)
            if os.path.exists(p):
                try:
                    v = json.load(open(p)).get("eos_token_id")
                except Exception as e:  # noqa: BLE001
                    sources[fn] = f"unreadable: {e!r}"[:120]
                    continue
                if v is None:
                    continue
                vs = [int(x) for x in (v if isinstance(v, list) else [v])]
                sources[fn] = sorted(vs)
                ids.update(vs)
    # [COORD M-0571 (1)] the EOS set pinned INTO the manifest is the declaration of record when present; a checkpoint given beside it
    # must agree (a manifest pinned from another tokenizer/generation config is refused by name), else the checkpoint is the source
    if samp.get("eos_token_ids") is not None:
        ms = sorted({int(x) for x in samp["eos_token_ids"]})
        sources["manifest.sampling.eos_token_ids"] = ms
        ck = sorted({v for k, v in sources.items() if k in ("config.json", "generation_config.json") and isinstance(v, list) for v in v})
        if ck and set(ck) != set(ms):
            return {"schema": "verity/workload-program/v1/stop-rule", "eos_token_ids": None, "source": sources, "checkpoint_dir": checkpoint_dir,
                    "refusal": f"manifest sampling.eos_token_ids {ms} != the pinned checkpoint's eos_token_id set {ck}: the manifest was pinned "
                               f"from another tokenizer/generation config"}
        ids = set(ms) | {int(x) for x in (eos_token_ids or [])}
    if samp.get("stop_token_ids"):
        st = [int(x) for x in samp["stop_token_ids"]]
        sources["manifest.sampling.stop_token_ids"] = sorted(st)
        ids.update(st)
    if samp.get("ignore_eos"):
        return {"schema": "verity/workload-program/v1/stop-rule", "eos_token_ids": [], "ignore_eos": True, "source": sources or None,
                "checkpoint_dir": checkpoint_dir, "rule": "manifest sampling.ignore_eos: every request must reach max_tokens; any early stop FAILS G6"}
    if not ids:
        return {"schema": "verity/workload-program/v1/stop-rule", "eos_token_ids": None, "source": sources or None, "checkpoint_dir": checkpoint_dir,
                "note": "no EOS set declared: GM-01 fails closed on any request that stops before its max_tokens (F-pm-24)"}
    return {"schema": "verity/workload-program/v1/stop-rule", "eos_token_ids": sorted(ids), "source": sources, "checkpoint_dir": checkpoint_dir,
            "rule": "a request may stop before max_tokens only when its last served token is one of eos_token_ids (finish_reason stop/None); "
                    "stop strings, aborts and drops are not modelled and FAIL G6"}


def weights_of_record_binding(components: dict[str, dict]) -> dict:
    """[COORD M-0614 (1) / tp M-0632 / manbind MAN-09 M-0638] The WEIGHTS OF RECORD per component: manbind's `weights_of_record.json`
    (root_of_record = H(the served tensors composed from the PINNED safetensors shards, in Program order; for a rank Program of TP world W,
    that rank's shard slices) written beside the component by row_pod / tp2's Build.  GP-01 does not compute a second root (one function,
    one owner); it CARRIES each component's record into the workload Program, and REFUSES when the records cannot all be one workload's:
      (a) a record whose program_digest is not the component's recorded digest (a stale record beside a re-derived component);
      (b) two components of the same rank with different roots (the compose would span two checkpoints);
      (c) a record that is not complete (a field with no composition rule) -- fail closed, named.
    A component with no record is carried as null (the row's Commit then fails 'weights unpinned', MAN-09), never invented.
    `components`: {"LP,T": {"dir", "recorded_digest"}}."""
    per: dict[str, dict | None] = {}
    by_rank: dict[int, tuple[str, str]] = {}
    for key, c in sorted(components.items()):
        p = os.path.join(c["dir"], "weights_of_record.json")
        if not os.path.exists(p):
            per[key] = None
            continue
        try:
            rec = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            return {"refusal": f"component {key}: weights_of_record.json unreadable ({e})", "components": per, "per_rank": None}
        tp = rec.get("tp") if isinstance(rec.get("tp"), dict) else {}
        rank = int(tp.get("rank") or 0)
        ent = {"root_of_record": rec.get("root_of_record"), "program_digest": rec.get("program_digest"), "tp": {"world": int(tp.get("world") or 1), "rank": rank},
               "complete": bool(rec.get("complete")), "n_pinned": rec.get("n_pinned"), "n_fields": rec.get("n_fields"),
               "unresolved": rec.get("unresolved") or [], "schema": rec.get("schema"), "checkpoint_dir": (rec.get("checkpoint") or {}).get("dir")}
        per[key] = ent
        rd = c.get("recorded_digest")
        # [rev-pm M-0717 (ii)] a record without program_digest cannot be tied to the component (the stale check could never fire): refused
        if not ent["program_digest"]:
            return {"refusal": f"component {key}: weights_of_record.json carries no program_digest -- it cannot be tied to the component "
                               f"(recorded digest {str(rd)[:16] if rd else None})", "components": per, "per_rank": None}
        if rd and str(ent["program_digest"]) != str(rd):
            return {"refusal": f"component {key}: weights_of_record.json is for program_digest {str(ent['program_digest'])[:16]}, the component's "
                               f"recorded digest is {str(rd)[:16]} (stale record beside a re-derived component)", "components": per, "per_rank": None}
        if not ent["complete"]:
            return {"refusal": f"component {key}: weights of record incomplete -- {len(ent['unresolved'])} Program weight field(s) without a "
                               f"composition rule: {[str(u)[:80] for u in ent['unresolved'][:5]]} (MAN-09: fail closed)", "components": per, "per_rank": None}
        if ent["unresolved"]:   # [M-0717 (ii)] complete:true beside unresolved != [] is a self-contradicting record
            return {"refusal": f"component {key}: weights_of_record.json says complete but lists {len(ent['unresolved'])} unresolved field(s) "
                               f"{[str(u)[:80] for u in ent['unresolved'][:5]]}: self-contradicting record, refused", "components": per, "per_rank": None}
        if not ent["root_of_record"]:
            return {"refusal": f"component {key}: weights_of_record.json carries no root_of_record", "components": per, "per_rank": None}
        # [rev F-r16-06, M-0056] a root is carried only when it was composed from the PINNED checkpoint shards (W1): `checkpoint.all_shards_pinned`
        # must be True (None = no shard pins recorded, False = a shard differs from its pin) and the record must not describe LIVE parameters
        # (a `source` starting with "live" is the Commit side's re-registration digest, never the weights of record) -- the finalizer applies the
        # same guard to weights_of_record*.json files and trusts workload_program.json's roots as-is, so the guard has to hold at the source.
        ck = rec.get("checkpoint") if isinstance(rec.get("checkpoint"), dict) else {}
        ent["all_shards_pinned"] = ck.get("all_shards_pinned")
        ent["source"] = rec.get("source")
        if str(rec.get("source") or "").lower().startswith("live"):
            return {"refusal": f"component {key}: weights_of_record.json describes live parameters (source {str(rec.get('source'))[:40]!r}), not the "
                               f"pinned checkpoint -- not a weights of record (W1)", "components": per, "per_rank": None}
        if ck.get("all_shards_pinned") is not True:
            return {"refusal": f"component {key}: weights_of_record.json root was not composed from pinned shards "
                               f"(checkpoint.all_shards_pinned={ck.get('all_shards_pinned')!r}; MAN-09 W1: only a root over the pinned checkpoint is of record)",
                    "components": per, "per_rank": None}
        if rank in by_rank and by_rank[rank][1] != ent["root_of_record"]:
            return {"refusal": f"components {by_rank[rank][0]} and {key} (rank {rank}) carry different weights roots "
                               f"{by_rank[rank][1][:16]} != {ent['root_of_record'][:16]}: one workload Program cannot span two checkpoints",
                    "components": per, "per_rank": None}
        by_rank.setdefault(rank, (key, ent["root_of_record"]))
    have = [k for k, v in per.items() if v]
    return {"refusal": None, "components": per,
            "per_rank": {str(r): v[1] for r, v in sorted(by_rank.items())} if have else None,
            "complete": bool(have) and len(have) == len(per),
            "note": ("every component carries manbind's weights of record; Commit compares each served parameter with it by name (MAN-09)"
                     if have and len(have) == len(per) else
                     f"{len(per) - len(have)} of {len(per)} components carry no weights_of_record.json: the row's Commit fails 'weights unpinned' (MAN-09)")}


def _refused(out: str, res: dict) -> dict:
    json.dump(res, open(os.path.join(out, "workload_program.json"), "w"), indent=1)
    return res


def build(workload_path: str, derived_dirs: list[str], out: str, checkpoint_dir: str | None = None, eos_token_ids: list[int] | None = None,
          artifact_root: str | None = None, early: dict[str, dict] | None = None, canon: dict | None = None) -> dict:
    """`artifact_root`: the dir the `vllm-build/v1` tree is rooted at (default: the parent of `out`); the `layout` block's relative paths
    are taken from it (module docstring).  `early`, `canon`: components `follow_components` already loaded, and their definition set."""
    t0 = time.time()
    artifact_root = os.path.abspath(artifact_root) if artifact_root else os.path.dirname(os.path.abspath(out))
    raw = open(workload_path, "rb").read()
    m = json.loads(raw)
    mdigest = hashlib.sha256(raw).hexdigest()
    reqs = manifest_requests(m)
    # bind each manifest request to the derived shape (LP, cap-1) — derived once per distinct shape, reused for equal shapes
    shapes: dict[tuple[int, int], str] = {}
    for d in derived_dirs:
        s = derived_shape(d)
        if s is None:
            raise ValueError(f"{d}: not a derive_step request artifact (no structural_inputs LP/T)")
        if s in shapes and shapes[s] != d:
            raise ValueError(f"two derived dirs for shape {s}: {shapes[s]} and {d}")
        shapes[s] = d
    need = sorted({(r["prompt_len"], r["cap"] - 1) for r in reqs})
    missing = [s for s in need if s not in shapes]
    os.makedirs(out, exist_ok=True)
    if missing:
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "missing_derived", "missing_derived": missing,
               "hint": "derive_step --wrapper request --LP <lp> --T <t> --workload <manifest> for each"}
        return _refused(out, res)
    # [COORD M-0063 (1)] the Program declares the execution profile + engine conditions it models and REFUSES a row it does not model
    app = workload_applicability(m, {s: component_applicability(shapes[s]) for s in need})
    if app["refusal"]:
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "applicability", "message": app["refusal"], "applicability": app,
               "manifest_sha256": mdigest}
        return _refused(out, res)
    # [rev-pm M-0560 (1)] a component carrying a per-request S-schedule (stochastic top-p, DECISION M-0529 (A)) binds only to the
    # request whose declared arrival it was computed for; a shape shared by two arrivals does not compose with one component
    sched = sampler_schedule_binding(reqs, {s: shapes[s] for s in need})
    # [M-0571 (3)] a component derived under stoch's EXPERIMENTAL per-event S (STOCH_DECLARED_CHRONOLOGY_S=1, sampling.geometry.experimental)
    # never composes into a canonical workload Program: refused unless this compose is itself marked experimental by the same variable
    exp = {f"{s[0]},{s[1]}": (component_sampling(shapes[s]).get("geometry") or {}).get("experimental") for s in need}
    exp = {k: v for k, v in exp.items() if v}
    if exp and os.environ.get("STOCH_DECLARED_CHRONOLOGY_S") != "1":
        sched["refusal"] = (f"components {sorted(exp)} were derived under the EXPERIMENTAL per-event split schedule "
                            f"(STOCH_DECLARED_CHRONOLOGY_S=1; ruling M-0571 (3): S is a workload-level derived value, not built yet): "
                            f"not composable into a canonical workload Program -- set the same variable to compose mechanism evidence")
    if sched["refusal"]:
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "sampler_schedule", "message": sched["refusal"], "sampler_schedule": sched,
               "manifest_sha256": mdigest}
        return _refused(out, res)
    # [R17-1 / DP-pad-02 (b)] LAG: the REQUIRED specialization parameter, declared from the row's target (the resolved async-scheduling
    # configuration of the serving profile, implementation pinned) BEFORE the Program is composed -- it is a static of Workload_v1 and
    # enters the digest; the components' derive-time targets must resolve to the same value (fail closed, named)
    lag = workload_lag(m, {s: shapes[s] for s in need})
    if lag.get("refusal") or lag.get("value") not in W.LAG_VALUES:
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "lag", "message": lag.get("refusal") or f"lag {lag.get('value')!r} not in {W.LAG_VALUES}",
               "lag": lag, "manifest_sha256": mdigest}
        return _refused(out, res)
    app["engine"] = {**app["engine"], "async_scheduling": bool(lag["async_scheduling"])}   # G1 compares it with the record's resolved read-back
    # [R17-42 route A (3)] the fused-MoE CONSTRUCTION: declared by the row's target, bound by the components (their Definitions), coupled here
    # (one construction per workload Program, fail closed, named); a header block, NOT an applicability.engine key -- never attested
    moec = workload_moe_construction(m, {s: shapes[s] for s in need})
    if moec.get("refusal"):
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "moe_construction", "message": moec["refusal"], "moe_construction": moec,
               "manifest_sha256": mdigest}
        return _refused(out, res)
    comps: dict[tuple[int, int], W.RequestComponent] = {}
    comp_secs = {}
    comp_provenance = {}
    canon = {} if canon is None else canon   # [rev-pm M-0123 F(1)] this compose's shared-definition set, never another compose's in the same process
    for s in need:
        d = shapes[s]
        pre = (early or {}).get(d) or load_component(d, canon)
        # [rev-pm M-0123 E] the component descriptor is checked against the digest the derive RECORDED for it (result.json program.digest /
        # artifact.json program_digest): a descriptor edited after the derive composes to another workload digest silently otherwise
        prov = pre["prov"]
        if prov["refusal"]:
            res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "component_provenance", "message": prov["refusal"], "component": d,
                   "manifest_sha256": mdigest}
            return _refused(out, res)
        # [hopper HOP-03 M-0276] a component derived with a declared LAUNCH CONTEXT (FA3 tile = launch property) carries it in its
        # applicability (`launch_context`, present only when declared): recorded per component here -- it is the component's own
        # (each shape has its own per-step launch max), so it is NOT an agreed workload_applicability key
        lc = component_applicability(d).get("launch_context")
        if lc is not None:
            prov["launch_context"] = lc
        comp_provenance[f"{s[0]},{s[1]}"] = prov
        comps[s] = component_of(pre)
        comp_secs[f"{s[0]},{s[1]}"] = pre["secs"]
    if why := registration_refusal([shapes[s_] for s_ in need]):   # the requests share one registration (`shared.values`, the tables)
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "registration", "message": why, "manifest_sha256": mdigest}
        return _refused(out, res)
    wreqs = [W.WorkloadRequest(r["request_id"], r["index"], r["prompt_len"], r["cap"], r["arrive_step"], comps[(r["prompt_len"], r["cap"] - 1)],
                               T=r["cap"] - 1) for r in reqs]
    t2 = time.time()
    wl = W.WorkloadProgram(wreqs, manifest_digest=mdigest, lag=int(lag["value"]))
    art = wl.artifact()
    art["lag"] = {**lag, "value": wl.lag}   # the declaration block (source, pinned implementation, per-component coupling) beside the static
    art["moe_construction"] = moec         # [R17-42] the construction block (declared value, family, per-component coupling, population label)
    # the manifest's UUIDs beside the Program's r<k> ids (the record's RunHeader/StepStart name requests by the UUID; the global
    # checker maps them through this table -- G1 checks the sets agree)
    wrids = {r["request_id"]: r["workload_request_id"] for r in reqs}
    for r in art["requests"]:
        r["workload_request_id"] = wrids.get(r["request_id"])
    art["request_ids"] = {"convention": "r<k>, k = 0-based position in the workload manifest (COORD R15 ruling 2)",
                          "workload_request_ids": wrids}
    art["applicability"] = app
    if exp:
        art["experimental"] = {"reason": "per-event split schedule from the DECLARED chronology (DECISION M-0529 (A)); mechanism evidence, "
                                         "not canonical (ruling M-0571 (3))", "components": exp}
    # [M-0614 (1) / M-0632 / MAN-09] the weights of record per component (per rank for TP), carried, never recomputed here
    wor = weights_of_record_binding({k: {"dir": v["dir"], "recorded_digest": v.get("recorded_digest")} for k, v in comp_provenance.items()})
    if wor["refusal"]:
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "weights_of_record", "message": wor["refusal"], "weights_of_record": wor,
               "manifest_sha256": mdigest}
        return _refused(out, res)
    art["weights_of_record"] = wor
    art["stop_rule"] = stop_rule(checkpoint_dir, eos_token_ids, m)   # [F-pm-24] the declared EOS set GM-01 verifies early stops against
    if art["stop_rule"].get("refusal"):
        res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "stop_rule", "message": art["stop_rule"]["refusal"], "stop_rule": art["stop_rule"],
               "manifest_sha256": mdigest}
        return _refused(out, res)
    # [R17-1 / DP-pad-02 (b)] the rows the declared scheduler computes AFTER an EOS stop are the Program's `lag` static (above); the stop
    # rule carries the same value under both names (eos_lag_rows: the ruling-21 reader's key; lag: R17-1's) -- one source, and GM-01 G6
    # consumes THIS declaration (never the record's attestation as the rule; G1 checks the record attests the same resolved value)
    art["stop_rule"]["eos_lag_rows"] = wl.lag
    art["stop_rule"]["lag"] = wl.lag
    art["stop_rule"]["eos_lag_source"] = ("Workload_v1 static `lag` = TargetProfile.lag() of the row's declared target (" + lag["source"] + "); "
                                          "applicability.engine.async_scheduling carries the resolved value the record must attest (G1/G6)")
    assert eos_lag_rows_declared(app.get("engine")) == wl.lag, "applicability.engine.async_scheduling and the lag static disagree"
    # [COORD M-0169 / M-0571 (3)] a stochastic row declares its top-p split geometry as a workload-level derived value (S_t from the live
    # count and the declared num_SMs); GM-01 evaluates it per engine step. Greedy rows carry no declaration (None -> omitted).
    sgeo = sampler_geometry(app, reqs)
    if sgeo is not None:
        art["sampler_geometry"] = sgeo
        if sgeo.get("refusal"):
            res = {"schema": W.SCHEMA + "/result", "ok": False, "reason": "sampler_geometry", "message": sgeo["refusal"], "sampler_geometry": sgeo,
                   "manifest_sha256": mdigest}
            return _refused(out, res)
    desc = wl.descriptor
    with gzip.open(os.path.join(out, "descriptor.json.gz"), "wb") as f:
        f.write(cj := canonical_json(desc))
    art["descriptor"] = {"bytes": len(cj) if "annotations" not in desc else descriptor_size(desc), "sha256": hashlib.sha256(cj).hexdigest(),
                         "n_definitions": len(desc["definitions"])}
    art["inputs"] = {"manifest": workload_path, "manifest_sha256": mdigest, "derived": {f"{k[0]},{k[1]}": v for k, v in shapes.items() if k in comps},
                     # per component: the descriptor's recomputed digest == the derive's recorded program_digest (checked above), descriptor sha
                     "component_provenance": comp_provenance,
                     "sampler_schedule": sched["checked"] or None,   # [M-0560 (1)] per-request S-schedule components and their bound request
                     "execution_record_read": False}
    # layout 1.1: the same paths relative to the artifact root, so a fetched Build resolves its components wherever it lands
    art["layout"] = build_paths.layout_block(artifact_root, workload_path, art["inputs"]["derived"])
    art["cost"] = {"component_secs": comp_secs, "compose_secs": round(time.time() - t2, 2), "total_secs": round(time.time() - t0, 2)}
    art["ok"] = True
    json.dump(art, open(os.path.join(out, "workload_program.json"), "w"), indent=1)
    json.dump(art["address_map"], open(os.path.join(out, "address_map.json"), "w"), indent=1)
    json.dump(art["chronology"], open(os.path.join(out, "chronology.json"), "w"), indent=1)
    return art


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload")
    derived: list[str] = option("--derived", action="append", default=[])
    out: str = option("--out")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="the pinned checkpoint: its config.json / generation_config.json eos_token_id " "become the Program's declared stop rule (F-pm-24)")
    eos_token_ids: str | None = option("--eos-token-ids", default=None, help="comma-separated EOS token ids declared explicitly (union with --checkpoint-dir's)")
    artifact_root: str | None = option("--artifact-root", default=None, help="the dir the vllm-build/v1 tree is rooted at (holds build_workload/ and the " "build_request*/ dirs); the `layout` block's paths are relative to it. Default: the parent of --out")
    follow: float | None = option("--follow", type=float, default=None,
                                  help="started beside the derives: load each component as its derive finishes, waiting up to this many seconds")
    follow_done: str | None = option("--follow-done", default=None, help="with --follow: a file whose existence means every derive has ended")
    follow_stop: str | None = option("--follow-stop", default=None, help="with --follow: a file whose existence means the Build failed: compose nothing")


def main(a: Options) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    eos = [int(x) for x in a.eos_token_ids.split(",") if x.strip()] if a.eos_token_ids else None
    early, canon = None, {}
    if a.follow:
        reqs = manifest_requests(json.load(open(a.workload)))
        early = follow_components(a.derived, {(r["prompt_len"], r["cap"] - 1) for r in reqs}, canon, timeout=a.follow, done=a.follow_done,
                                  stop=a.follow_stop)
        if early is None:
            return 5
    res = build(a.workload, a.derived, a.out, checkpoint_dir=a.checkpoint_dir, eos_token_ids=eos, artifact_root=a.artifact_root, early=early,
                canon=canon)
    if not res.get("ok"):
        print(json.dumps({k: res[k] for k in ("reason", "message", "missing_derived") if k in res}))
        return 4
    print(json.dumps({"workload_digest": res["workload_digest"], "gates": res["gates"], "input_gates": res["input_gates"],
                      "requests": len(res["requests"]), "descriptor_bytes": res["descriptor"]["bytes"], "cost": res["cost"],
                      "eos_token_ids": (res.get("stop_rule") or {}).get("eos_token_ids"),
                      "lag": (res.get("lag") or {}).get("value"), "async_scheduling": (res.get("lag") or {}).get("async_scheduling"),
                      "moe_construction": (res.get("moe_construction") or {}).get("value"),
                      "moe_block_present": (res.get("moe_construction") or {}).get("moe_block_present")}))
    return 0
