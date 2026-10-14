"""DERIVE-VLLM-1 step (4)-(5): export a wrapper around the ACTUAL pinned vLLM model class and derive it with
`verity_vllm.program.frontend.derive` + the per-op vLLM bindings. Wrappers (`vllm_meta.make_wrapper`): `step` (one engine step, M tokens at
pos0), `request` (prefill LP + T decode steps, flat), `padded` (the fixed padded rectangle prefill | decode 1..D_MAX with liveness
values). Writes the derive report (provenance: every rule application), the program's descriptor + digest, a per-spec histogram,
the instance sequence (comparator input), and the inputs trace.

    PYTHONPATH=. verity-vllm build --out /workspace/cp/cba/step_decode --wrapper step --M 1 --pos0 8
    PYTHONPATH=. verity-vllm build --out /workspace/cp/cba/request_8_4 --wrapper request --LP 8 --T 4
    PYTHONPATH=. verity-vllm build --out /workspace/cp/cba/padded_8_4 --wrapper padded --LP 8 --T 4 --eos 0
"""

from __future__ import annotations

import contextlib
import copy
import functools
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass

from verity_vllm.config import option
from verity_vllm.engine import env as engine_env, hooks
from verity_vllm.engine.profiles.transport import workload_custom_all_reduce_off
from verity_vllm.program.boolean import derived
from verity_vllm.program.frontend.inputs_trace import InputsTrace
from verity_vllm.pipeline.tp.links import module_in_model_vocabulary, read_tp_links, same_module_path, write_tp_links  # noqa: F401 (read through here)


def _spec_histogram(program) -> dict[str, int]:
    from verity.ml.circuits.codec import _spec_id
    h: dict[str, int] = {}
    for n in program.fn.body.nodes:
        if n.form in ("call", "primitive"):
            k = _spec_id(n.fn)
            h[k] = h.get(k, 0) + 1
        else:
            k = f"{n.form}:{_spec_id(n.fn)}"
            h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))


def _instance_rows(program):
    """The flat body as (spec, argument items) rows — the comparator's unit (compare.py), one at a time.  An argument is stored in the
    progressions form (instances v2, `compact.file_items_from_refs`, the fold writer's): O(rows) for a scattered Strided where `runs()`
    is O(leaves) — the GLM-5.3 MLA-absorb transposed-weight read is ~1e6 leaves per operand.  Every reader expands back to the
    per-leaf run lists, bit-identically (`compact.expanded_file_args` at `program_view.iter_instance_rows` and the raw doc loaders)."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.program.compact import file_items_from_refs
    for k, n in enumerate(program.fn.body.nodes):
        yield {"i": k, "spec": descriptor_id(n.fn), "form": n.form, "name": n.name, "args": [file_items_from_refs(a.refs) for a in n.args]}


def _instance_sequence(program) -> list[dict]:
    return list(_instance_rows(program))


def _write_instances(f, head: dict, rows, chunk: int = 4096) -> None:
    """`json.dump({**head, "rows": list(rows)}, f)`, the same text, written a chunk of rows at a time with `json.dumps` (its C encoder;
    `json.dump` to a file always runs the Python one) and without the rows list."""
    f.write(json.dumps(head)[:-1] + (', "rows": [' if head else '"rows": ['))
    batch, first = [], True
    for r in rows:
        batch.append(r)
        if len(batch) == chunk:
            f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
            batch, first = [], False
    if batch:
        f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
    f.write("]}")


#: `instances.json.gz`'s gzip level.  Its readers parse its text, and no digest or published output reads its bytes
#: (`research_outputs.NEVER`).  A long context's text is mostly the attention Calls' key and value runs (B1 4096/512: 8.6 GB), which
#: level 6 compresses in under a third of level 9's CPU time, 2% larger, and which inflates as fast at either level.
INSTANCES_GZ_LEVEL = 6


def _write_instances_file(path: str, head: dict, rows) -> None:
    with gzip.open(path, "wt", compresslevel=INSTANCES_GZ_LEVEL) as f:
        _write_instances(f, head, rows)


#: the builder's own sources (hashed into `construction_version`): the export shims, the meta instantiation / wrappers, this driver,
#: the Triton interception, the vLLM bindings + vocabulary + derive core, and the applicability contract itself.
_CONSTRUCTION_SOURCES = ("verity_vllm/program/frontend/export_compat.py", "verity_vllm/program/frontend/vllm_meta.py",
                         "verity_vllm/pipeline/build.py", "verity_vllm/program/frontend/triton_capture.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/__init__.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/attention.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/cache_pad.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/collectives.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/elementwise.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/fp8.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/fused_norm.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/norm_chain.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/observations.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/operands.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/pins.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/triton_launches.py",
                         "verity_vllm/program/frontend/rules/vllm_bindings/views.py",
                         "verity_vllm/program/frontend/rules/vocab.py", "verity_vllm/program/frontend/rules/triton_iface.py",
                         "verity_vllm/program/frontend/rules/vllm_iface.py", "verity_vllm/program/frontend/derive.py", "verity_vllm/program/frontend/target_profile.py",
                         "verity_vllm/program/frontend/targets.py",
                         # [R13 tp] the TP construction scaffolding (group stub, collective bus, verity_tp ops) and the TP2 Definitions
                         "verity_vllm/program/frontend/export_ops.py", "verity_vllm/program/registry/b1_tp2.py",
                         # [R15 stoch] the stochastic token-select rule (the sampler Definitions are registry digests in the descriptor); the fused-MoE rules
                         "verity_vllm/program/frontend/rules/vllm_sampling.py", "verity_vllm/program/frontend/rules/vllm_moe.py",
                         "verity_vllm/program/frontend/rules/vllm_moe_fp8.py", "verity_vllm/program/frontend/rules/vllm_moe_router.py",
                         "verity_vllm/program/frontend/rules/vllm_dsa.py",
                         # `TargetProfile.ir == "boolean"`: the derived Program on bits (the Boolean Definitions are digests in the descriptor)
                         "verity_vllm/program/boolean.py",
                         # the vLLM switches the meta instantiation pins (engine.env.EXPORT)
                         "verity_vllm/engine/env.py",
                         # the derive's report and its relabel per TP rank (`--emit-ranks`), and the tp_links.json writer
                         "verity_vllm/program/frontend/report.py", "verity_vllm/program/frontend/relabel.py",
                         "verity_vllm/pipeline/tp/emit_ranks.py", "verity_vllm/pipeline/tp/links.py")


def construction_version(rules, root: str | None = None) -> dict:
    """Identity of the CONSTRUCTION (not of the target, not of the host): schema, the binding rules (id + version), and the sha256 of
    the builder's source files, read from the integration root `root` (default: the one `verity_vllm` is imported from). Two hosts
    running the same tree get the same value; an edited rule/shim gets another. A listed source that is not there raises."""
    from verity_vllm.config import ROOT
    root = root or str(ROOT)
    h = hashlib.sha256()
    files = []
    for rel in _CONSTRUCTION_SOURCES:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"construction_version: source {rel} is not under the integration root {root}")
        with open(p, "rb") as f:
            b = f.read()
        files.append({"file": rel, "sha256": hashlib.sha256(b).hexdigest()})
        h.update(rel.encode() + b"\0" + b + b"\0")
    return {"schema": "cb-a/aot-artifact/v1", "mechanism": "torch.export(strict=False) on meta + verity_vllm.program.frontend.derive (profile mode)",
            "rules": [{"id": r.rule_id, "version": r.info.version} for r in rules], "sources_sha256": h.hexdigest(), "files": files}


def engine_dtype(model_config) -> str:
    """The dtype the engine runs the checkpoint's weights in: vLLM's quantization method when the checkpoint is quantized
    (`model_config.quantization`, "fp8" for an FP8 checkpoint), else the model dtype ("bfloat16")."""
    if model_config.quantization:
        return str(model_config.quantization)
    return str(model_config.dtype).removeprefix("torch.")


#: [R13 X-05] the token-select vocabulary of this builder: sampling policy kind -> the rule family that binds it.  A row whose policy is
#: not a key here is REFUSED before export (`reason: "sampler"`) — never built as the greedy Program (F-dse-08: fail-open Build).
SAMPLERS_BOUND = {"greedy": "ArgmaxTokenSelect_v1 (aten.argmax over the bf16 logits row -> TokenSelect_v1{V})",
                  # [R15 stoch] temperature > 0 with top-p, an explicit per-request seed and no top-k: the wrapper authors ONE
                  # `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p, pos, splits)` per sampled step; the rule binds it to
                  # `GumbelTopPTokenSelect_v1{V,S}` (registry.sampling: f32 widening -> div.full.f32 temperature -> the S-split top-p mask
                  # `_topp_sb_{stats,step,mask}_kernel` -> Gumbel-max keyed `tl.randint(seed, pos)` -> first maximum).  The seed is a runtime
                  # INPUT of the Program; temperature / top_p / pos are constants; S = SplitsFor_v1(|live(t)|, num_SMs) is a DERIVED input
                  # (`splits`, request-owned, one entry per sampling event; `sampler_geometry`), or the constant SplitsFor_v1(1, num_SMs)
                  # of a single-request workload -- the 6th operand, never a static.
                  "stochastic": "GumbelTopPTokenSelect[vllm]_v3 (vllm_iface.gumbel_topp_token_select -> GumbelTopPTokenSelect_v1{V} with seed + derived S Program inputs, or GumbelTopPTokenSelect_v2{V,S} with the seed input and the constant S of a single-request workload: its keep word as word gates)"}


def declared_live_counts(reqs: list[dict]) -> list[int]:
    """[R15 stoch, DECISION M-0529 (A)] The DECLARED chronology's live sampler batch per engine step: n_live(e) = #{r : arrive_r <= e <
    arrive_r + cap_r} — every request is sampled once per engine step from its arrival (prefill sample at e = arrive_r) until it has produced
    `cap_r` tokens (the `length` stop; vLLM's sampler sees one logits row per scheduled request, chunked prefills included).  Index e."""
    horizon = max((int(r.get("arrive_step", 0) or 0) + _cap_of(r) for r in reqs), default=0)
    n = [0] * horizon
    for r in reqs:
        a, c = int(r.get("arrive_step", 0) or 0), _cap_of(r)
        for e in range(a, a + c):
            n[e] += 1
    return n


def _cap_of(r: dict) -> int:
    return int(r.get("max_tokens") or (r.get("sampling_params") or {}).get("max_tokens") or 0)


def sampler_geometry(sampling: dict, target, LP: int | None = None, T: int | None = None) -> dict | None:
    """[R15 stoch; R16 coord M-0169] The launch geometry of the pinned stochastic sampler for this row, or the refusal (`unsupported`
    dict) when the policy has no Definition.  On success `sampling["splits"]` is "derived" (several requests) or the constant S (one
    request, below) and `sampling["geometry"]` records the RELATION -- {num_sms, relation SplitsFor_v1, live workload-derived,
    sampler_batch B, pipeline} -- and nothing per request, so every component of a workload declares the same sampler applicability
    (GP-01 agreement by construction).

    vLLM d9105ea80 `apply_top_k_top_p` for p-only requests and a sampler batch <= 64 runs `_apply_topp_split` with
    S = splits_for(batch, num_sms) splits (the largest power of two S <= 32 with S * batch <= num_sms; vLLM's doubling loop); the masked
    row's fp32 value can depend on S (the partials' summation order).  RULING M-0571 (3) / M-0169, amended for single-request workloads
    (below): S is a WORKLOAD-level DERIVED value -- S_t = SplitsFor_v1(|live(t)|, num_SMs), |live(t)| = the requests with a sampled
    logits row at engine step t (declared chronology and verified EOS stops) -- and the request Program takes it as a request-owned
    INPUT `splits` (int32 [T+1], one entry per sampling event; `vllm_meta.StochasticRequest`), the 6th operand of
    `GumbelTopPTokenSelect_v1{V}`.  The Program declares NO S and NO schedule: the Match binds each event's entry to the split kernels'
    constexpr and checks the relation per engine step (GM-01 G6 'sampler geometry'; `bench.stoch_recompute --build`).  R15's static S /
    declared-chronology schedule (`STOCH_DECLARED_CHRONOLOGY_S`) is gone: a Program that declared S could only be right by assumption
    (every request runs to its cap).

    The amendment for single-request workloads (the Project's decision record dates it): a workload of ONE request has |live(t)| = 1
    at every sampling event it has (a request that stops has no later event), so S = SplitsFor_v1(1, num_SMs) exactly, with no
    assumption about the schedule.  Its Program takes S as a per-event CONSTANT (`sampling["splits"]` = that int; `Const32[S]` the
    select's 6th operand, computed here from the target's SM count, never hard-coded) and has no `splits` input; the circuit then
    anchors S itself.  A workload of several requests keeps the derived input above."""
    from verity_vllm.program.registry.topp_geometry import SPLIT_MAX_BATCH, splits_for, splits_for_v1
    kind = sampling.get("kind")
    if kind != "stochastic":
        return None
    def refuse(hint: str, reason: str = "sampler", **detail) -> dict:
        return {"reason": reason, "phase": "applicability", "node": None, "module_path": None, "hint": hint,
                "detail": {"sampling": {k: v for k, v in sampling.items() if k != "workload"}, "samplers_bound": SAMPLERS_BOUND, **detail}}
    if int(sampling.get("top_k", 0) or 0) != 0:
        return refuse(f"top_k = {sampling.get('top_k')}: vLLM's top-k path (`_topk_topp_kernel` / `_apply_topk_topp`) has no exact Definition; "
                      f"only the p-only split pipeline (`_apply_topp_split`) is modelled (TopPMask_v1{{V}})", missing=["TopKMask"])
    if not sampling.get("seeded", False):
        return refuse("the policy has no explicit per-request seed: vLLM draws a random seed when `SamplingParams.seed` is None "
                      "(SamplingStates.add_request), so the RNG stream is not a declared input and the sampled token is not reproducible "
                      "from the manifest — declare `seed` per request (sweep `sampling_seeds`)", missing=["declared seed"])
    reqs = []
    wl = sampling.get("workload")
    if wl:
        try:
            w = json.load(open(wl))
            reqs = list(w.get("requests") or [])
        except Exception as e:  # noqa: BLE001
            return refuse(f"cannot read the workload manifest {wl!r} for the sampler batch: {e!r}", reason="sampler_geometry")
    B = max(1, len(reqs))
    if B > SPLIT_MAX_BATCH:
        return refuse(f"sampler batch {B} > _SPLIT_MAX_BATCH {SPLIT_MAX_BATCH}: vLLM runs `_topk_topp_kernel`, not the split pipeline", reason="sampler_geometry")
    if getattr(target, "num_sms", None) is None:
        return refuse("the target profile declares no num_sms: S_t = SplitsFor_v1(|live(t)|, num_SMs) has no num_SMs to be derived with",
                      reason="sampler_geometry")
    nsm = int(target.num_sms)
    if wl and len(reqs) == 1:                      # the manifest's one request: the only live request at every event
        S = int(splits_for_v1(1, nsm))
        geom = {"sampler_batch": 1, "num_sms": nsm, "pipeline": "_apply_topp_split", "relation": "SplitsFor_v1",
                "rule": "S = SplitsFor_v1(1, num_SMs) at every sampling event: the workload's one request is the only live one",
                "live": "single request", "splits": S, "ruling": "the derived-S ruling, amended for single-request workloads",
                "constant": f"Const32[{S}]: the select's 6th operand at every sampling event (no `splits` input)"}
        sampling["splits"] = S
        sampling["geometry"] = geom
        return None
    geom = {"sampler_batch": B, "num_sms": nsm, "pipeline": "_apply_topp_split", "relation": "SplitsFor_v1",
            "rule": "S_t = SplitsFor_v1(|live(t)|, num_SMs); |live(t)| = requests with a sampled logits row at engine step t (workload-derived)",
            "live": "workload-derived", "splits_range_1_to_B": [int(splits_for(B, nsm)), int(splits_for(1, nsm))],
            "input": "splits: int32 [T+1] (request-owned Program Input; entry s = the split count of sampling event s), after the seed"}
    sampling["splits"] = "derived"
    sampling["geometry"] = geom
    return None


def workload_engine_args(workload: str | None, tp: int) -> dict:
    """The engine arguments a TP row's Build takes from its workload: ``disable_custom_all_reduce`` as the row's engines run it
    (`engine.profiles.transport.workload_custom_all_reduce_off`).  None at TP1 or without a workload: `vllm_meta`'s default then."""
    if tp < 2 or workload is None:
        return {}
    return {"disable_custom_all_reduce": workload_custom_all_reduce_off(json.load(open(workload)))}


def sampling_policy(arg: str | None, workload: str | None) -> dict:
    """[R13 X-05] The row's sampling policy as a STRUCTURAL INPUT of the construction (BRIEF: prompt length, max_tokens, sampling kind).

    Sources, in order: `--workload <manifest>` (the row's manifest: `sweep.sampling` + `sweep.sampling_params` from `sweep/workload.py`,
    else the legacy `sampling.variant` -> `variants[...]`, else per-request `variant` / `seed` fields), and/or `--sampling <policy>` with
    `greedy` | `stoch-t<T>-p<P>[-k<K>]` | a JSON object.  Both given -> they must agree (refused otherwise).  Neither given -> greedy,
    recorded as `source: "default"` (a caller without a manifest declares greedy by omission; the row procedure passes the manifest).
    Returns {"kind": "greedy"} or {"kind": "stochastic", "temperature", "top_p", "top_k", "seeded", "label"} (+ "source", "mixed")."""
    def _norm(params: dict, label: str | None = None) -> dict:
        t = float(params.get("temperature", 1.0))
        if t == 0.0:
            return {"kind": "greedy"}
        return {"kind": "stochastic", "temperature": t, "top_p": float(params.get("top_p", 1.0)), "top_k": int(params.get("top_k", 0) or 0),
                "seeded": bool(params.get("seeded", True)), "label": label or f"stoch-t{t:g}-p{float(params.get('top_p', 1.0)):g}"}

    def _parse_arg(s: str) -> dict:
        s = s.strip()
        if s.startswith("{"):
            return _norm(json.loads(s))
        if s == "greedy":
            return {"kind": "greedy"}
        import re
        m = re.fullmatch(r"stoch-t([0-9.]+)-p([0-9.]+)(?:-k([0-9]+))?", s)
        if not m:
            raise ValueError(f"unknown sampling policy {s!r} (greedy | stoch-t<T>-p<P>[-k<K>] | JSON)")
        return _norm({"temperature": float(m.group(1)), "top_p": float(m.group(2)), "top_k": int(m.group(3) or 0)}, s)

    def _from_manifest(path: str) -> dict:
        w = json.load(open(path))
        sw = w.get("sweep") or {}
        if sw.get("sampling") is not None:
            pol = _norm(dict(sw.get("sampling_params") or {}), sw["sampling"]) if sw["sampling"] != "greedy" else {"kind": "greedy"}
            if sw.get("sampling_seeds"):
                pol["seeded"] = True
            return pol
        variants = w.get("variants") or {}
        top = (w.get("sampling") or {}).get("variant")
        pols = []
        for r in w.get("requests", []):
            v = r.get("variant", top)
            p = variants.get(v) if v is not None else None
            if p is None:
                p = {"temperature": 0.0} if r.get("seed") is None and v in (None, "greedy") else {"temperature": 1.0}
            pols.append(_norm(dict(p), v))
        if not pols:
            return _norm(dict(variants.get(top) or {"temperature": 0.0}), top) if top else {"kind": "greedy"}
        kinds = {json.dumps(p, sort_keys=True) for p in pols}
        if len(kinds) > 1:
            return {"kind": "mixed", "policies": [json.loads(k) for k in sorted(kinds)]}
        return pols[0]

    pol_w = _from_manifest(workload) if workload else None
    pol_a = _parse_arg(arg) if arg else None
    if pol_w is not None and pol_a is not None:
        a, b = ({k: v for k, v in p.items() if k not in ("label", "source")} for p in (pol_w, pol_a))
        if a != b:
            raise ValueError(f"--sampling {arg!r} disagrees with the manifest's sampling policy {pol_w} ({workload})")
    pol = pol_w if pol_w is not None else (pol_a if pol_a is not None else {"kind": "greedy"})
    pol = dict(pol)
    pol["source"] = "workload" if pol_w is not None else ("arg" if pol_a is not None else "default")
    if workload:
        pol["workload"] = workload
    return pol


def applicability_conditions(profile, applied: dict, attention_impls: dict, export_facts: dict, sampling: dict | None = None, tp: dict | None = None,
                             launch: dict | None = None) -> dict:
    """The facts READ BACK from the environment/model/export that the Program's Definition selection rests on — the conditions under
    which the artifact may be reused (`target_profile.check_reuse`). Every value is observed, none is a literal."""
    from verity_vllm.program.frontend.rules.vllm_bindings import GEMM_PINS, RMS_PINS, tuned_gemm_table_sha256
    from verity_vllm.program.frontend.gemm_targets import try_gemm_target_for
    # [R13 ampere] the registered GEMM specialization the construction pinned (verity_vllm.program.frontend.gemm_targets, selected by the target's
    # capability + the arch family vLLM applied): its key and k16 step are conditions of reuse; None (with the refusal) when unregistered.
    gt, gt_refusal = try_gemm_target_for(profile.compute_capability, applied.get("tuned_gemm_arch_family"))
    from verity_vllm.program.frontend import targets as TG
    try:   # [R13 hopper] the target-keyed Definition selection the Program rests on (frontend.targets): step, GEMM/attention kinds, FA version
        tb = TG.describe(profile)
        target_bindings = {k: tb[k] for k in ("tensor_core_family", "dot", "gemm_kind", "attention_kind", "flash_attn_version_effective", "tuned_gemm_arch_family")}
    except TG.UnregisteredTarget as e:
        target_bindings = {"unregistered": str(e)}
    impls = sorted({str(v.get("impl")) for v in attention_impls.values()}) if attention_impls else []
    favs = sorted({str(v.get("vllm_flash_attn_version")) for v in attention_impls.values()}) if attention_impls else []
    backs = sorted({str(v.get("backend")) for v in attention_impls.values()}) if attention_impls else []
    samp = {k: v for k, v in (sampling or {"kind": "greedy"}).items() if k not in ("source", "workload", "label")}
    cond = {"numerics_profile": profile.numerics_profile, "attention_impl": impls, "attention_backend": backs, "vllm_flash_attn_version": favs,
            # [X-05] the sampler the Program binds is an applicability condition: a greedy artifact is not reusable for a stochastic row
            "sampler": samp, "token_select": SAMPLERS_BOUND.get(samp.get("kind")),
            "fused_add_rms_norm_dispatch": (export_facts or {}).get("fused_add_rms_norm_dispatch"),
            "vllm_version": applied.get("vllm_version"), "torch_version": applied.get("torch_version"), "triton_version": applied.get("triton_version"),
            "tuned_gemm_arch_family": applied.get("tuned_gemm_arch_family"), "tuned_gemm_table_sha256": tuned_gemm_table_sha256(),
            "gemm_target": gt.key if gt else None, "gemm_target_dot": gt.dot if gt else None, "gemm_target_refusal": gt_refusal,
            # [X-01] the SM count is the ONE TargetProfile fact of the persistent-GEMM grid (scheduling, not numerics): recorded here as vLLM
            # read it back through the shim (= what every bound launch showed as NUM_SMS; equals profile.num_sms, which the identity carries too).
            "num_sms": applied.get("num_sms"),
            # [R13 hopper] the declared attention knob + the version read back from the layers, and the target-keyed bindings selected
            "flash_attn_version_declared": profile.flash_attn_version, "flash_attn_version_applied": applied.get("flash_attn_version"),
            "execution": profile.execution_label, "target_bindings": target_bindings,
            "gemm_kernel_src_sha256": GEMM_PINS["src_sha256"], "rms_kernel_src_sha256": RMS_PINS["src_sha256"]}
    if tp and tp.get("world", 1) > 1:
        # [R13 tp] a rank Program of a TP world is reusable only for that (world, rank) on the admitted collective path (pynccl, custom AR off).
        # Key present ONLY at world > 1: TP1 artifact identities are unchanged.
        cond["tp"] = {"world": tp["world"], "rank": tp["rank"], "collective_backend": tp.get("collective_backend"), "engine_args_required": tp.get("engine_args_required")}
    if launch:
        # [R15 hopper HOP-03] a request Program built under a declared launch context is reusable only for a batch that co-schedules it the same
        # way (FA3's tile reads the launch's max_seqlen_q).  Key present ONLY when a context was declared: B1 / undeclared identities are unchanged.
        cond["launch_context"] = {"max_seqlen_q_by_step": {str(k): v for k, v in sorted((launch.get("max_seqlen_q_by_step") or {}).items())},
                                  "conflict": bool(launch.get("conflict"))}
    return cond


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def run(out: str, M: int, pos0: int, profile: str, backend: str, intercept: bool = True, wrapper: str = "step", LP: int = 8, T: int = 4, eos: int = 0, model_id: str | None = None, revision: str | None = None, max_model_len: int = 2048,
        target=None, sampling: dict | None = None, tp: int = 1, rank: int = 0, launch: dict | None = None, emit=None,
        engine_args: dict | None = None) -> dict:
    """`emit`: an `emit_ranks.Emit` (`--emit-ranks`): the other ranks' files are written from this rank's derivation.  `engine_args`:
    the row's engine arguments the meta config takes (`workload_engine_args`)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    sampling = dict(sampling) if sampling is not None else {"kind": "greedy", "source": "default"}
    if tp < 1 or not (0 <= rank < tp):
        raise ValueError(f"--tp {tp} --rank {rank}")
    if target is None:   # [X-01] no declared target: the target is THIS host's GPU, queried once (multi_processor_count, capability); GPU-free host -> default
        target = TargetProfile.from_device() if _cuda_available() else TargetProfile.default()
        res_target_source = "device" if _cuda_available() else "default"
    else:
        res_target_source = "declared"
    if target.numerics_profile != profile:
        target = target.with_overrides(numerics_profile=profile)  # `--profile` IS the vocabulary field of the target profile (one source)
    res: dict = {"schema": "cb-a/derive_step/v3", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wrapper": wrapper, "M": M, "pos0": pos0,
                 "LP": LP, "T": T, "eos": eos, "sampling": sampling, "profile": profile, "target": {"declared": target.to_json(), "declared_source": res_target_source},
                 # [R13 tp] the tensor-parallel structural inputs: world size and WHICH rank this Program describes (rank-local computation,
                 # this rank's weight shards as its committed weight inputs, the other rank's collective partials as peer inputs)
                 "tp": {"world": int(tp), "rank": int(rank)},
                 # [R15 hopper HOP-03, M-0225] the LAUNCH CONTEXT (structural input): per request step the varlen launch's max_seqlen_q over every
                 # co-scheduled sequence (frontend.targets.parse_launch_context).  FA3's tile rule reads it; None = own rows (B1 semantics, digests unchanged)
                 "launch_context": launch}
    try:
        res["code_rev"] = json.load(open("EXPORT.json"))
    except Exception:  # noqa: BLE001
        res["code_rev"] = None
    # the profile's environment is applied BEFORE vLLM is imported (vllm_meta's setdefaults are the fallback for callers without a profile)
    engine_env.apply_target(target.env())
    res["ok"] = False
    try:
        # [X-05] applicability declares the sampler: a policy outside the builder's token-select vocabulary is refused HERE, before any
        # export — the greedy Program is not "the" Program of a stochastic row (F-dse-08).
        if sampling.get("kind") not in SAMPLERS_BOUND:
            res["unsupported"] = {"reason": "sampler", "phase": "applicability", "node": None, "module_path": None,
                                  "hint": (f"the row's sampling policy is {sampling} (source: {sampling.get('source')}); this builder binds the token-select "
                                           f"kinds {sorted(SAMPLERS_BOUND)} only ({'; '.join(f'{k}: {v}' for k, v in SAMPLERS_BOUND.items())}). A stochastic "
                                           f"row needs a token-select Definition for the pinned BI sampler and a rule that emits it — not the greedy argmax. "
                                           f"[X-06] `GumbelTokenSelect_v1{{V}}(logits, seed, pos, temp)` (verity_vllm.program.registry.sampling; seed/pos/temperature as "
                                           f"bound inputs) is exact to `_temperature_kernel` -> `_gumbel_sample_kernel` for top_p == 1 / no top_k; the top-p "
                                           f"mask (`topk_topp_triton._topp_sb_{{stats,step,mask}}_kernel`) has no exact Definition (F-dense-02) and no "
                                           f"wrapper authors the stochastic select yet."),
                                  "detail": {"sampling": sampling, "samplers_bound": SAMPLERS_BOUND,
                                             "definitions_available": {"GumbelTokenSelect_v1": "temperature + Gumbel-max, top_p == 1, no top_k (X-06)"},
                                             "missing": ["TopPMask (F-dense-02)", "wrapper authoring of the stochastic select (Bet A)",
                                                         "eager resolver patterns for _temperature_kernel / stochastic _gumbel_sample_kernel"]}}
            raise _Refused()
        # [R15 stoch] the stochastic policy's Definition coverage and launch geometry (S), refused with the reason when not a static
        geo_refusal = sampler_geometry(sampling, target, LP=LP if wrapper == "request" else None, T=T if wrapper == "request" else None)
        if geo_refusal is not None:
            res["unsupported"] = geo_refusal
            raise _Refused()
        res["sampling"] = sampling
        if res_target_source == "declared":   # the declared target decides the platform, not the host's NVML (a GPU-free Build host has none)
            from verity_vllm.program.frontend.export_compat import select_cuda_platform
            print(f"[derive_step] declared target {target.sm_arch} x{tp}: vLLM platform {select_cuda_platform(tp)}", file=sys.stderr)
        _construct(out, res, target, M, pos0, profile, backend, intercept, wrapper, LP, T, eos, model_id, revision, max_model_len, t_start, sampling,
                   tp=tp, rank=rank, launch=launch, emit=emit, engine_args=engine_args)
    except _Refused:
        pass
    res = _finish(out, res, t_start)
    if emit is not None:
        from verity_vllm.pipeline.tp.emit_ranks import finish
        finish(emit, out, res, _write_manifest)
    return res


_WRAPPER_NAMES = {"step": "vLLMStep", "request": "vLLMRequest", "padded": "vLLMPaddedRequest"}


class _Refused(Exception):
    """The construction refused before derive (target profile mismatch, unbound sampler); `res["unsupported"]` carries the verbatim reason."""


def _construct(out, res, target, M, pos0, profile, backend, intercept, wrapper, LP, T, eos, model_id, revision, max_model_len, t_start, sampling,
               tp: int = 1, rank: int = 0, launch: dict | None = None, emit=None, engine_args: dict | None = None) -> None:
    from verity_vllm.program.frontend.target_profile import ArtifactIdentity, profile_mismatches
    with InputsTrace(os.path.join(out, "inputs.json")), contextlib.ExitStack() as tp_stack:
        from verity_vllm.program.frontend import vllm_meta as vm
        from verity_vllm.program.frontend.export_compat import apply_target_profile, host_facts, read_back_applied
        from verity_vllm.program.frontend.triton_capture import TritonLaunchLog
        import torch
        from verity_vllm.program.frontend.derive import derive_report
        from verity_vllm.program.frontend.rules.vllm_bindings import (VLLM_BINDING_RULES, vllm_ruleset, observe_attention_impls, observe_export_facts,
                                                            observe_target_profile, declared_derived_accounting, clear_observations,
                                                            observe_allreduce_settings, observe_launch_context)
        clear_observations()
        observe_target_profile(target)
        observe_launch_context(launch)
        res["target"]["host"] = host_facts()
        t0 = time.time()
        with apply_target_profile(target):  # ONE application of the declared target for config, instantiation, export and derive
            # [R13 hopper] `TargetProfile.engine_args()` = the attention knobs the target applies (`attention_backend`, `attention_config.flash_attn_version`)
            cfg = vm.make_vllm_config(model=model_id or vm.PIN["model"], revision=revision or vm.PIN["revision"], max_model_len=max_model_len,
                                      **target.engine_args(), **(engine_args or {}), tensor_parallel_size=tp)
            res["model_pin"] = {"model": model_id or vm.PIN["model"], "revision": revision or vm.PIN["revision"], "dtype": engine_dtype(cfg.model_config), "tp": tp,
                                "rank": rank, "max_model_len": max_model_len}
            vm.ensure_distributed(cfg, backend)
            bus = None
            if tp > 1:
                # [R13 tp] rank `rank` of `tp`: vLLM's TP group := TPStub (shard sizes + collectives -> CollectiveBus) for instantiation, export and derive
                from verity_vllm.program.frontend.export_ops import CollectiveBus, bind_vocab_parallel_embeddings, tp_group_override
                env_now = engine_env.snapshot()
                enabled = lambda name: str(env_now.get(name, "1")).strip().lower() not in ("0", "false", "no", "off")  # noqa: E731
                allreduce_settings = {
                    "custom_all_reduce": not bool(cfg.parallel_config.disable_custom_all_reduce),
                    "symmetric_memory": enabled("VLLM_ALLREDUCE_USE_SYMM_MEM"),
                    "flashinfer": enabled("VLLM_ALLREDUCE_USE_FLASHINFER"),
                }
                observe_allreduce_settings(allreduce_settings)
                bus = CollectiveBus(rank=rank, world=tp, settings=allreduce_settings)
                tp_stack.enter_context(tp_group_override(rank, tp, bus))
                car_off = bool(cfg.parallel_config.disable_custom_all_reduce)
                res["tp"]["engine_args_required"] = {"tensor_parallel_size": tp, "disable_custom_all_reduce": car_off}
                # the workload's transport as the row set it (`row_tp.TpRow.p2p_env`) -- recorded, not consulted here
                res["tp"]["env_required"] = {k: v for k, v in env_now.items() if k == "NCCL_P2P_DISABLE"}
                res["tp"]["allreduce_settings"] = allreduce_settings
                res["tp"]["collective_backend"] = ("pynccl (custom all-reduce disabled; BI pins NCCL tree/Simple/1 channel)" if car_off else
                                                   "vLLM's custom all-reduce where it admits the tensor, else pynccl")
            from verity_vllm.engine import sparse_mla
            sparse_mla.arm()   # FA3's sparse-MLA split pin, armed as a serving process arms it before vLLM builds the model
            model = vm.instantiate_meta(cfg)
            # [R15 fp8] loader parity: a checkpoint that ships a redundant `lm_head.weight` under `tie_word_embeddings` is untied by
            # vLLM for loading and RE-TIED after it iff the two tensors are equal (Qwen3-4B-Instruct-2507-FP8); the same decision from the
            # pinned checkpoint's bytes, so the Program has the served model's one shared root (see `vm.retie_lm_head_as_the_loader_does`)
            res["lm_head_tying"] = vm.retie_lm_head_as_the_loader_does(model, cfg)
            res["mla_absorbed_weights"] = vm.mla_absorbed_weights_as_the_loader_does(model, cfg)
            if tp > 1:
                res["tp"]["embedding_shards"] = bind_vocab_parallel_embeddings(model, tp_stack.enter_context(hooks.Hooks()))
                res["tp"]["sharded_params"] = _sharded_param_summary(model)
            res["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
            res["hf_config"] = {k: v for k, v in cfg.model_config.hf_config.to_dict().items()
                                if k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size",
                                         "rms_norm_eps", "tie_word_embeddings", "head_dim", "max_position_embeddings")}
            res["param_device"] = str(next(model.parameters()).device)
            res["cuda_available"] = torch.cuda.is_available()
            res["target"]["consulted"] = vm.consulted_facts(cfg)
            # READ BACK what vLLM saw during instantiation (never a literal); declared != applied -> refuse
            applied = read_back_applied()
            applied["attention_backend"] = res["target"]["consulted"].get("attention_backend_knob")
            # [R15 fp8] the kernels vLLM selected: the FP8 linears' (declared by `fp8_block_gemm`), the fused-MoE experts' (by `moe_backend`)
            from verity_vllm.program.frontend.target_profile import applied_kernels
            applied.update(applied_kernels(target, model))
            # derived (observed) selection fields: the instantiated layers' attention impl; the fused-norm dispatch (after export)
            res["attention_impls_observed"] = vm.attention_impls(model)
            # [R13 hopper] READ BACK the FA version the selector gave the layers (one value over all layers, else None = mismatch): the
            # declared `flash_attn_version` (or the selector's default for the declared capability) must be what was instantiated
            favs = {v.get("vllm_flash_attn_version") for v in res["attention_impls_observed"].values() if v.get("kind") != "mla"}
            fav_ints = {int(x) for x in favs if isinstance(x, int) and not isinstance(x, bool)}
            if fav_ints or len(favs) > 1:        # FlashAttention layers (or a mixed tree): judged here.  A backend that carries no FA
                applied["flash_attn_version"] = fav_ints.pop() if (len(fav_ints) == 1 and len(favs) == 1) else None
            # else: no layer is a FlashAttention impl (e.g. TritonAttentionImpl) -> not an FA-version question; the attention rule refuses
            # the unregistered impl verbatim (numerics_unregistered) instead of a mismatch on a fact the backend does not have
            res["target"]["applied"] = applied
            bad = profile_mismatches(target, applied)
            if bad:
                res["unsupported"] = {"reason": "target_profile_mismatch", "phase": "instantiation", "hint": "; ".join(bad)}
                raise _Refused()
            observe_attention_impls(res["attention_impls_observed"])
            log = TritonLaunchLog()
            mod, inputs = vm.make_wrapper(wrapper, model, pos0=pos0, M=M, LP=LP, T=T, eos_id=eos, sampling=sampling)
            ntok = M if wrapper == "step" else LP
            if tp > 1:
                # [R13 tp] probe: one meta forward records the rank's collectives in call order -> the PEER INPUTS of the Program (the other
                # rank's partials, ordinary Inputs, bound by tp_links.json to the other rank's committed partials); then the traced wrapper
                probe = vm.probe_collectives(mod, inputs, cfg, ntok, bus, profile=target)
                res["tp"]["collectives"] = {"n": len(probe), "by_kind": _count([p["kind"] for p in probe]),
                                            "by_shape": _count([f'{p["kind"]}{tuple(p["shape"])}' for p in probe]), "record": probe}
                mod, inputs = vm.with_peers(mod, inputs, bus)
            applied_export: dict = {}
            ep = vm.export_module(mod, inputs, cfg, ntok, intercept_triton=intercept, launch_log=log, profile=target, applied_out=applied_export)
            res["export"] = {"secs": round(time.time() - t0, 2), "n_nodes": len(list(ep.graph.nodes)), "op_histogram": vm.op_histogram(ep),
                             "triton_kernels": sorted({l["kernel"] for l in log.launches}), "n_triton_launches": len(log.launches)}
            res["export_facts"] = vm.export_facts(res["export"]["op_histogram"], res["export"]["triton_kernels"])
            observe_export_facts(res["export_facts"])
            # READ BACK inside the export context: an export-phase value that diverged from the declared one (the R9 a1 shape: a
            # second capability applied only around the export) is refused, never recorded as the declared value
            applied_export["attention_backend"] = applied["attention_backend"]
            if "flash_attn_version" in applied:                  # absent when no layer is a FlashAttention impl (judged by the attention rule)
                applied_export["flash_attn_version"] = applied["flash_attn_version"]
            # [R15 fp8] the layers' kernel objects, read back AGAIN after the export (a kernel swapped around the export would show here)
            applied_export.update(applied_kernels(target, model))
            res["target"]["applied_after_export"] = applied_export
            bad = profile_mismatches(target, applied_export)
            if bad:
                res["unsupported"] = {"reason": "target_profile_mismatch", "phase": "export", "hint": "; ".join(bad)}
                raise _Refused()
            res["declared_derived"] = declared_derived_accounting(target, applied)
            # derive: the wrapper module IS the source, derived from the export above (same module, same example input, same shims)
            t1 = time.time()
            session: list | None = [] if emit is not None else None
            with _derive_context(vm, cfg, ntok, target, model):
                # the rank coordinate goes to the runtime correspondence only (`annotations["correspondence"].rank`): the Program is unchanged
                rep = derive_report(mod, inputs, mode="profile", profile=profile, rules=vllm_ruleset(), name=f"{_WRAPPER_NAMES[wrapper]}[{res['model_class'].rsplit('.', 1)[-1]}]",
                                    check_registry=True, rank=(tp, rank) if tp > 1 else None, exported=ep, registered_values=bool(target.registered_values),
                                    session=session)
            rep = derived(rep) if target.ir == "boolean" else rep
        res["derive_secs"] = round(time.time() - t1, 2)
        res["bindings_rules"] = [r.info.to_json() for r in VLLM_BINDING_RULES]
        res["ok"] = rep.ok
        if not rep.ok:
            res["unsupported"] = rep.unsupported.to_json()
            res["rules_applied_before_refusal"] = rep.provenance.get("rules_applied_before_refusal")
            res["n_applications_before_refusal"] = len(rep.provenance.get("applications_before_refusal", []))
            json.dump(rep.provenance, open(os.path.join(out, "derive-report.json"), "w"), indent=1, default=str)
        else:
            # [lane vllm-tp-n] the model's path INSIDE the traced wrapper, by object identity (`WithPeers.inner` -> wrapper `.model`):
            # the derive's module paths are the wrapper's `named_modules()` vocabulary, the recorder's are `model.named_modules()`;
            # the root is recorded as DATA so no reader spells the wrapper prefix
            model_root = next((n for n, m in mod.named_modules() if m is model), None) if tp > 1 else None
            write = functools.partial(_write_program, target=target, applied=applied, sampling=sampling, launch=launch, tp=tp, model_root=model_root)
            pre = copy.deepcopy(res) if emit is not None else None      # rank r's `res` before its writer: the template's, at rank r
            write(out, res, rep, rank=rank)
            if emit is not None:
                from verity_vllm.pipeline.tp import emit_ranks
                from verity_vllm.program.frontend.export_ops import rank_embedding_shards
                rep = None                                                 # the template's descriptor and correspondence: freed
                emit_ranks.emit(emit, out, pre, session[0] if session else None, shards=lambda r: rank_embedding_shards(model, r),
                                context=lambda: _derive_context(vm, cfg, ntok, target, model, apply_target=True),
                                post=derived if target.ir == "boolean" else None, write=write)


def _write_program(out: str, res: dict, rep, *, target, applied: dict, sampling: dict, launch: dict | None, tp: int, rank: int,
                   model_root: str | None) -> None:
    """A successful derive's files under `out` (descriptor, derive report, instances, tp_links, artifact) and its `res` fields."""
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_vllm.correspondence.runtime import NAMESPACE as CORR_NS, correspondence_digest
    from verity_vllm.program.frontend.rules.vllm_bindings import VLLM_BINDING_RULES
    from verity_vllm.program.frontend.target_profile import ArtifactIdentity
    program, supplied = rep.program, ({"supplied": s} if (s := rep.provenance.get("supplied")) else {})   # a registered_values derive's
    desc = {**rep.descriptor, "annotations": dict(program.annotations)} if rep.descriptor is not None else encode_program(program)  # annotations re-read: the derive adds provenance after its digest
    digest = hashlib.sha512(canon := canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()  # program_digest's text
    # runtime correspondence: carried in the descriptor's annotations (excluded from program_digest), digested apart
    corr_digest = correspondence_digest(desc)
    assert corr_digest == rep.correspondence_digest, (corr_digest, rep.correspondence_digest)
    raw = json.dumps(desc, sort_keys=True).encode()
    with gzip.open(os.path.join(out, "descriptor.json.gz"), "wb") as f:
        f.write(raw)
    corr_doc = (desc.get("annotations") or {}).get(CORR_NS) or {}
    res["program"] = {"function": program.fn.id, "digest": digest, "gates": program.gates, "input_gates": program.input_gates,
                      "n_root_nodes": len(program.fn.body.nodes), "descriptor_bytes": len(canon),
                      "descriptor_sha256": hashlib.sha256(raw).hexdigest(), "dead_gates": rep.dead.total if rep.dead else None,
                      "correspondence_digest": corr_digest,
                      "correspondence": {"schema": corr_doc.get("schema"), "namespace": CORR_NS, "calls": len(corr_doc.get("calls") or ()),
                                         "occurrences": len(corr_doc.get("occurrences") or ()), "rank": corr_doc.get("rank")} if corr_doc else None, **supplied}
    res["spec_histogram"] = _spec_histogram(program)
    res["rules_applied"] = rep.provenance["rules_applied"]
    res["n_applications"] = len(rep.provenance["applications"])
    res["weights"] = {"n_fields": len(rep.provenance["weights"]), "roles": rep.provenance.get("weight_roles")}
    if tp > 1:
        res["tp"]["links"] = write_tp_links(out, res, program, rep, tp, rank, model_root=model_root)
    res["mutation_effects"] = rep.provenance["mutation_effects"]
    res["kv_rewrites"] = len(rep.provenance.get("kv_rewrites") or []) if isinstance(rep.provenance, dict) else None
    json.dump(rep.to_json(), open(os.path.join(out, "derive-report.json"), "w"), indent=1, default=str)
    _write_instances_file(os.path.join(out, "instances.json.gz"),
                          {"schema": "cb-a/instance-sequence/v2", "digest": digest, "correspondence_digest": corr_digest,
                           "params": [[n, str(t)] for n, t in program.fn.body.params] if hasattr(program.fn.body, "params") else None, **supplied},
                          _instance_rows(program))
    # the artifact wrapper: identity = H(construction version || target profile || applicability || Program digest); digest unchanged
    art = ArtifactIdentity.build(construction_version(VLLM_BINDING_RULES), target,
                                 applicability_conditions(target, applied, res["attention_impls_observed"], res["export_facts"], sampling, res.get("tp"), launch), digest)
    res["artifact"] = art.to_json()
    # beside program_digest, NOT inside the identity: the correspondence is data about the Program, not part of it
    res["artifact"]["correspondence_digest"] = corr_digest
    json.dump(res["artifact"], open(os.path.join(out, "artifact.json"), "w"), indent=1, sort_keys=True)


@contextlib.contextmanager
def _derive_context(vm, cfg, ntok: int, target, model, *, apply_target: bool = False):
    """The contexts the derive runs in; `apply_target`: with the declared target applied again (a relabel, after the construction's)."""
    import torch
    from verity_vllm.program.frontend.export_compat import apply_target_profile, export_compat
    from verity_vllm.program.frontend.triton_capture import intercept_triton_launches
    with contextlib.ExitStack() as st:
        if apply_target:
            st.enter_context(apply_target_profile(target))
        with torch.no_grad(), vm.fake_forward_context(cfg, ntok), torch.device("meta"), export_compat(target, model), intercept_triton_launches():
            yield


def _count(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))


def _sharded_param_summary(model) -> dict:
    """[R13 tp] the rank's parameter shapes AS INSTANTIATED at this rank (the sharded weight inputs the Program commits to), keyed by the
    vLLM layer class: `{class: {shape: count}}` — the independent check that the shard shapes match the engine worker's loaded shards."""
    import torch
    by_cls: dict = {}
    n_params = n_elems = 0
    for name, m in model.named_modules():
        own = list(m.named_parameters(recurse=False))
        if not own:
            continue
        key = type(m).__name__
        for pn, p in own:
            n_params += 1
            n_elems += p.numel()
            by_cls.setdefault(key, {}).setdefault(f"{pn}{tuple(p.shape)}", 0)
            by_cls[key][f"{pn}{tuple(p.shape)}"] += 1
    return {"n_params": n_params, "n_elements": n_elems, "by_class": by_cls}


def _finish(out: str, res: dict, t_start: float) -> dict:
    import resource
    res["wall_secs_total"] = round(time.time() - t_start, 2)
    res["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)  # Linux: KB
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1, default=str)
    _write_manifest(out, res)
    return res


def _write_manifest(out: str, res: dict) -> None:
    """`construction_manifest.json`: what the construction read (inputs trace summary), which rules fired, the declared
    facts it relied on, wall/RSS, and the program identity — for the independent validator."""
    tr = {}
    try:
        tr = json.load(open(os.path.join(out, "inputs.json")))
    except Exception:  # noqa: BLE001
        pass
    m = {"schema": "cb-a/construction_manifest/v3", "utc": res.get("utc"), "wrapper": res.get("wrapper"),
         # [X-05] `sampling` is the row's declared policy (manifest / --sampling / default-by-omission), never a literal
         "structural_inputs": {"LP": res.get("LP"), "T": res.get("T"), "M": res.get("M"), "pos0": res.get("pos0"), "eos": res.get("eos"),
                               "sampling": res.get("sampling"), "tp": (res.get("tp") or {}).get("world", 1), "rank": (res.get("tp") or {}).get("rank", 0),
                               "launch_context": res.get("launch_context")},
         # [R13 tp] world > 1: the rank's collectives (peer-input signature), embedding shard, engine args + env the admitted TP path requires,
         # the sharded parameter shapes and the cross-rank binding summary (full record: result.json `tp`, tp_links.json)
         "tp": ({k: v for k, v in res["tp"].items() if k != "collectives"} | {"collectives": {k: v for k, v in res["tp"].get("collectives", {}).items() if k != "record"}}
                if (res.get("tp") or {}).get("world", 1) > 1 else None),
         "model": {"class": res.get("model_class"), "hf_config": res.get("hf_config"), "pin": res.get("model_pin"),
                   # [R15 fp8] `hf_config.tie_word_embeddings` above is what vLLM SET for loading (False when the checkpoint ships an
                   # lm_head); the loader's re-tie decision, reproduced from the checkpoint's bytes, is the served model's fact
                   "lm_head_tying": res.get("lm_head_tying")},
         "code_rev": res.get("code_rev"), "profile": res.get("profile"),
         # v3: `target.declared` = the TargetProfile (the only source of every applied value); `target.applied` / `applied_after_export` =
         # read back from vLLM inside the applied shim; `target.host` = the construction host (informational, in no identity);
         # `target.consulted` = config facts + env split into consulted / ignored (dead in this pin) / non-vLLM
         "target": res.get("target"), "artifact": res.get("artifact"),
         "declared": (res.get("declared_derived") or {}).get("declared"), "derived": (res.get("declared_derived") or {}).get("derived"),
         "observed": {"attention_impls": res.get("attention_impls_observed"), "export_facts": res.get("export_facts")},
         "construction": {"mechanism": "torch.export(strict=False) of the vLLM model class on meta + verity_vllm.program.frontend.derive (profile mode)",
                          "export": res.get("export"), "derive_secs": res.get("derive_secs"), "wall_secs_total": res.get("wall_secs_total"),
                          "max_rss_mb": res.get("max_rss_mb"), "cuda_available": res.get("cuda_available"), "param_device": res.get("param_device"),
                          "real_inference": False, "weights_loaded": False},
         "rules_applied": res.get("rules_applied"), "n_applications": res.get("n_applications"), "bindings_rules": res.get("bindings_rules"),
         "inputs_trace": {"n_open_events": tr.get("n_open_events"), "distinct_paths": tr.get("distinct_paths"), "forbidden_hits": tr.get("forbidden_hits"),
                          "site_packages_by_package": tr.get("site_packages_by_package"),
                          "non_site_packages_paths": [p for p in tr.get("non_site_packages_paths", []) if not p.startswith("/usr/lib/python")]},
         "program": res.get("program"), "spec_histogram": res.get("spec_histogram"), "ok": res.get("ok"), "unsupported": res.get("unsupported"),
         "files": ["descriptor.json.gz (verity.ml.circuits descriptor v1)", "instances.json.gz (instance sequence: spec + operand run progressions)", "derive-report.json", "inputs.json", "result.json",
                   "artifact.json (verity/aot-artifact/v1: identity + construction_version + target_profile + applicability + program_digest)"]}
    if res.get("relabeled_from"):
        m["relabeled_from"] = res["relabeled_from"]
    json.dump(m, open(os.path.join(out, "construction_manifest.json"), "w"), indent=1, default=str)


@dataclass(frozen=True, kw_only=True)
class Options:
    out: str = option("--out")
    M: int = option("--M", type=int, default=1)
    pos0: int = option("--pos0", type=int, default=8)
    profile: str = option("--profile", default="b1-eager-v3")
    backend: str = option("--backend", default="gloo")
    wrapper: str = option("--wrapper", default="step", choices=("step", "request", "padded"))
    LP: int = option("--LP", type=int, default=8)
    T: int = option("--T", type=int, default=4)
    eos: int = option("--eos", type=int, default=0)
    model: str | None = option("--model", default=None, help="HF repo id (default: the B0 pin)")
    revision: str | None = option("--revision", default=None)
    max_model_len: int = option("--max-model-len", type=int, default=2048)
    target: str | None = option("--target", default=None,
                                help="TargetProfile: JSON overrides of the default (e.g. '{\"compute_capability\": [9, 0]}') or a path to a JSON profile")
    sampling: str | None = option("--sampling", default=None,
                                  help="[X-05] the row's sampling policy: greedy | stoch-t<T>-p<P>[-k<K>] | JSON (must agree with --workload if both)")
    workload: str | None = option("--workload", default=None,
                                  help="[X-05] the row's workload manifest; its sampling policy becomes the structural input `sampling`")
    tp: int = option("--tp", type=int, default=1,
                     help="[R13 tp] tensor-parallel world size (structural input); the Program describes ONE rank")
    rank: int = option("--rank", type=int, default=0, help="[R13 tp] which rank's Program to construct (0..tp-1)")
    emit_ranks: str | None = option("--emit-ranks", default=None,
                                    help="other ranks (comma list) whose files are written from this rank's derivation (pipeline.tp.emit_ranks)")
    emit_root: str | None = option("--emit-root", default=None, help="with --emit-ranks: rank r's files go to <root>/rank<r>/<name of --out>")
    launch_max_seqlen_q: str | None = option("--launch-max-seqlen-q", default=None,
                                             help="[R15 hopper HOP-03] the LAUNCH CONTEXT this request Program is built under (structural input): the varlen launch's "
                                             "max_seqlen_q over every co-scheduled sequence, per request step -- int N (step 0 = a prefill launch whose longest prompt "
                                             "has N rows), JSON list [n0, n1, ...] (per step), or 'conflict' (requests of this shape are co-scheduled differently: "
                                             "refused under FA3, whose tile reads it; recorded and inert under FA2).  Omitted = own rows (a batch of one).  "
                                             "row_pod.sh derives it from the workload (verity_vllm.pipeline.launch_context) for B >= 2.")


def main(a: Options) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    try:
        from verity_vllm.program.frontend.target_profile import parse_target_arg
        from verity_vllm.program.frontend.targets import parse_launch_context
        from verity_vllm.pipeline.tp.emit_ranks import parse_ranks
        emit = parse_ranks(a.emit_ranks, a.tp, a.rank, a.emit_root) if a.emit_ranks else None
        res = run(a.out, a.M, a.pos0, a.profile, a.backend, wrapper=a.wrapper, LP=a.LP, T=a.T, eos=a.eos, model_id=a.model, revision=a.revision,
                  max_model_len=a.max_model_len, target=parse_target_arg(a.target) if a.target else None, sampling=sampling_policy(a.sampling, a.workload),
                  tp=a.tp, rank=a.rank, launch=parse_launch_context(a.launch_max_seqlen_q), emit=emit,
                  engine_args=workload_engine_args(a.workload, a.tp))
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        os.makedirs(a.out, exist_ok=True)
        json.dump({"ok": False, "exception": type(e).__name__, "message": str(e)[:4000], "traceback_tail": tb[-6000:]}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        print(f"[derive_step] CRASH {type(e).__name__}: {str(e)[:800]}", file=sys.stderr)
        print(tb[-3000:], file=sys.stderr)
        return 2
    summary = {k: res.get(k) for k in ("ok", "unsupported", "program", "rules_applied", "n_applications", "derive_secs")}
    summary["target"] = {k: (res.get("target") or {}).get(k) for k in ("declared", "applied")}
    summary["artifact_identity"] = (res.get("artifact") or {}).get("identity")
    print(json.dumps(summary, indent=1, default=str))
    if res.get("ok"):
        print(json.dumps(res["spec_histogram"], indent=1))
    return 0 if res.get("ok") else 1
