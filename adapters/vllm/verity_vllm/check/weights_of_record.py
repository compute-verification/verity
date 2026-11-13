"""The WEIGHTS OF RECORD.

A Commit's `weights` registration hashes whatever model the committer served; comparing it with the live parameters re-registered at
replay (O5) is self-consistency, not authentication: a served model that differs from the checkpoint in one element (a flipped weight)
replays bit-exactly against itself.  C2 requires the registered inputs to be authenticated by their CANONICAL reference; for the
weights that reference is the CHECKPOINT OF RECORD -- the safetensors shards whose sha256 `manifests/checkpoints.json` pins --
not the live engine.

This module derives, ENGINE-FREE and on CPU, one digest per weight field the Program's `weights` parameter names, from the shard bytes:

  * the field's tensor is COMPOSED from the checkpoint tensors by the implementation's stacking rule (the served model fuses
    `q|k|v -> qkv_proj`, `gate|up -> gate_up_proj`, the experts' `gate|up -> w13_weight[e]` / `down -> w2_weight[e]` (served on the
    FusedMoE layer or on its `routed_experts` submodule, vLLM 0.28; Mixtral's `w1|w3 -> w13`, `w2 -> w2`); a tied
    `lm_head.weight` is `embed_tokens.weight`; everything else is 1:1 under the export wrapper prefix); a rank Program of a TP world
    gets the rank's shard (per-part dim-0 slices for the fused column-parallel fields, the dim-1 slice for row-parallel fields,
    replicated norms unchanged; FP8 expert scales on vLLM's refined grid first, `fp8_moe_scales`);
  * in the SERVED element width the Program declares (`Value<16>` = bf16): a checkpoint stored in another dtype is cast with the
    implementation's round-to-nearest-even (`torch.Tensor.to(bfloat16)` semantics), and the cast is recorded by name;
  * digest = sha256 of the composed row-major bytes; root_of_record = sha256 over the ordered (field, digest) list, an ANALYTIC field
    (the rotary table) entering as `analytic:<sha256 of its exact config.json inputs>` -- so the root is a function of (the Program's
    weights struct, the checkpoint bytes, config.json) and of nothing on the deriving host.  [Schema v2: v1 keyed the analytic
    entry by the numpy-fp32 inv_freq bytes, which differ by host CPU dispatch (AVX512 SVML vs scalar libm: cp-r1 vs cp-a100 gave two
    roots for one Llama-3.2-1B record) -- the pin VERDICT never depended on it (check_cos_sin is a bounded check).]

Program fields that are NOT checkpoint tensors (the attention `_k_scale/_v_scale/_q_scale/_prob_scale` `Value<32>` engine constants,
and on a quantized checkpoint vLLM's `attn.k_scale/v_scale/q_scale/prob_scale` loader placeholders -- F-fp8-PIN1) are pinned from the
engine's declared configuration by `_engine_constant` (f32 1.0 / the f32 -1.0 sentinel under kv_cache_dtype auto; the rotary
table analytically) or listed BY NAME as `unpinned` when no rule applies;
a field with no composition rule is `unresolved`, named, and the record's `complete` is False (fail closed: a Program whose weights
this module cannot derive from the checkpoint has no weights of record).

The Commit side (`commit_delta --weights-of-record / --checkpoint-dir`) sha256s the live parameters the committer registered (the same
tensors the weights root covers) and compares field by field (`check`): any difference, or a Program field missing from the served
model, is a named FAIL of the row (`commit_verdict`: weights differ from the checkpoint of record).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import config
from verity_vllm.check import derived_weights, fp8_ue8m0
from verity_vllm.check.checkpoint_index import CheckpointIndex, sha256_file, shard_pin  # noqa: F401  (sha256_file: the record's shard hash)
from verity_vllm.check.component_programs import (
    component_program_dirs,
    load_program_head,
)

# handed on: other modules and tests read the composition and the engine constants through this module
from verity_vllm.check.composed_weights import EXPERTS as EXPERTS
from verity_vllm.check.composed_weights import EXPERTS_MIXTRAL as EXPERTS_MIXTRAL
from verity_vllm.check.composed_weights import EXPERTS_RE as EXPERTS_RE
from verity_vllm.check.composed_weights import FUSED as FUSED
from verity_vllm.check.composed_weights import FUSED_W13 as FUSED_W13
from verity_vllm.check.composed_weights import _compose_experts as _compose_experts
from verity_vllm.check.composed_weights import _f32_to_bf16_bits as _f32_to_bf16_bits
from verity_vllm.check.composed_weights import _rank_rows as _rank_rows
from verity_vllm.check.composed_weights import compose
from verity_vllm.check.composed_weights import to_served as to_served
from verity_vllm.check.engine_constants import (
    ENGINE_CONSTANT_RE,
    ENGINE_ROTARY_RE,
    PLACEHOLDER_RE,
    _engine_constant,
    analytic_key,
)
from verity_vllm.check.engine_constants import (
    ENGINE_KV_PLACEHOLDER_RE as ENGINE_KV_PLACEHOLDER_RE,
)
from verity_vllm.check.engine_constants import ENGINE_KV_SCALE_RE as ENGINE_KV_SCALE_RE
from verity_vllm.check.engine_constants import (
    kv_scale_remap_sources as kv_scale_remap_sources,
)
from verity_vllm.check.param_digests import param_digests
from verity_vllm.config import UsageError, option
from verity_vllm.log import init_logger
from verity_vllm.query.manifest import format as MF

__all__ = ["CheckpointIndex", "ENGINE_CONSTANT_RE", "EXPERTS_RE", "FUSED", "Options", "PARAM", "SCHEMA", "build_record",
           "check", "compose", "derivation_provenance", "live_param_digests", "main", "program_weight_fields",
           "rank_composer", "record_shard_pin", "served_record", "stamp_of_record_set", "strip_prefix", "weight_param",
           "weights_of_record"]

logger = init_logger(__name__)

SCHEMA = "verity/commit/weights-of-record/v2"       # v2: host-independent root (analytic entries keyed by their exact inputs)
SCHEMA_V1 = "verity/commit/weights-of-record/v1"    # v1 roots differ per deriving host for models with a rotary table; verdict unaffected
FULL_DIGEST_RE = re.compile(r"[0-9a-f]{128}")       # a Program digest (sha512 hex): compared only whole, a prefix names no Program
PARAM = "weights"                                   # the Program parameter (a struct) the record is of, and the window circuit's registered value


def weight_param(label: str, words: int) -> str:
    """The registered parameter of weight ``label`` (its name in the weights record) read in rows of ``words`` u16 words:
    one registration a weight and row width, so every read's base and steps are whole rows of it (`interface._reads_of`)
    and its pin is a root over the weight's own rows (`interface.registered_reads`), the committer's hiding root for it
    (`program.predict.window_weights.weights_registration`); a pin of `PARAM` pins none of them (`interface.input_count`)."""
    return f"{PARAM}/{label}@{words}"


def strip_prefix(field: str, prefix: str) -> str:
    return field[len(prefix):] if prefix and field.startswith(prefix) else field


def rank_composer(*, checkpoint_dir: str, checkpoints_manifest: str | None = None, model: str | None = None, world: int = 1,
                  rank: int = 0) -> Callable[[str, list[int], int], np.ndarray]:
    """`compose` at a TP rank over the checkpoint at `checkpoint_dir`: (served name, shape, bits) -> raw words.  Its shards are not hashed
    again here: the rank's weights pin checks them."""
    ck = CheckpointIndex(checkpoint_dir, checkpoints_manifest=checkpoints_manifest or _default_manifest(), model=model, verify_shards=False)
    return lambda name, shape, bits: compose(name, shape, bits, ck, world=world, rank=rank)[0]


def program_weight_fields(inst: dict[str, Any]) -> list[tuple[str, list[int], int]]:
    """(field, shape, value bits) for every field of the Program's `weights` struct parameter, in declaration order."""
    from verity_vllm.check.replay.index import parse_struct
    wparam = next((t for n, t in inst.get("params", []) if n == PARAM), None)
    if not wparam or not str(wparam).startswith("{"):
        return []
    return [(name, shape, bits) for name, _count, shape, bits in parse_struct(str(wparam))]


def root_of_fields(fields: list[dict[str, Any]]) -> str:
    """root_of_record under the CURRENT schema from a record's `fields`: the ordered (field, digest | analytic:<analytic_key>) list."""
    ordered = []
    for f in fields:
        if f.get("digest"):
            ordered.append([f["field"], f["digest"]])
        elif f.get("pinned") == "analytic" and isinstance(f.get("analytic"), dict):
            a = f["analytic"]
            ordered.append([f["field"], "analytic:" + (a.get("inputs_sha256") or analytic_key(a))])
    return hashlib.sha256((SCHEMA + "\n" + json.dumps(ordered, separators=(",", ":"))).encode()).hexdigest()


def reroot(record: dict[str, Any]) -> str:
    """The root a record's fields give under the CURRENT schema -- a reader: the record is not changed.  For a current-schema record this
    equals its root_of_record (a self-check that the root is recomputable from the fields); for a v1 record it is the root v2 would have
    derived from the same fields (two v1 roots of one Llama-3.2-1B, cp-r1 / cp-a100, re-root to one)."""
    return root_of_fields(record.get("fields") or [])


def diff_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two records FIELD BY FIELD (the comparison report uses for same-model workloads across pods under schema v1):
    fields only on one side; fields whose digest / shape / bits / analytic KEY differ (the weights of record differ); fields that differ
    only in host-side information (analytic.inv_freq_sha256, the host note, sources/rule text) -- named separately, they do not make
    the weights differ.  `same_weights_of_record` = no one-sided field, no differing digest/shape/bits/analytic key; then reroot(a) ==
    reroot(b) whatever the two schemas were."""
    fa = {f["field"]: f for f in (a.get("fields") or [])}
    fb = {f["field"]: f for f in (b.get("fields") or [])}
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    differ: list[dict[str, Any]] = []
    host_only: list[dict[str, Any]] = []
    for name in sorted(set(fa) & set(fb)):
        x, y = fa[name], fb[name]
        keys = [k for k in ("digest", "shape", "bits", "pinned") if x.get(k) != y.get(k)]
        ax, ay = x.get("analytic"), y.get("analytic")
        if isinstance(ax, dict) or isinstance(ay, dict):
            kx = analytic_key(ax) if isinstance(ax, dict) else None
            ky = analytic_key(ay) if isinstance(ay, dict) else None
            if kx != ky:
                keys.append("analytic inputs")
            elif isinstance(ax, dict) and isinstance(ay, dict) and ax.get("inv_freq_sha256") != ay.get("inv_freq_sha256"):
                host_only.append({"field": name, "keys": ["analytic.inv_freq_sha256"],
                                  "a": str(ax.get("inv_freq_sha256"))[:16], "b": str(ay.get("inv_freq_sha256"))[:16]})
        if keys:
            differ.append({"field": name, "keys": keys, "a": {k: x.get(k) for k in keys if k != "analytic inputs"}, "b": {k: y.get(k) for k in keys if k != "analytic inputs"}})
    same = not only_a and not only_b and not differ
    ra, rb = reroot(a), reroot(b)
    return {"schema": [a.get("schema"), b.get("schema")], "root_of_record": [a.get("root_of_record"), b.get("root_of_record")],
            "n_fields": [len(fa), len(fb)], "only_a": only_a, "only_b": only_b, "differ": differ, "host_only": host_only,
            "equal_fields": len(set(fa) & set(fb)) - len(differ), "reroot": [ra, rb], "same_weights_of_record": same and ra == rb}


def weights_of_record(program: dict[str, Any], ck: CheckpointIndex, *, prefix: str | None = None, kv_cache_dtype: str = "auto",
                      fields: list[tuple[str, list[int], int]] | None = None,
                      on_field: Callable[[str, np.ndarray], None] | None = None) -> dict[str, Any]:
    """The record: per Program weight field the digest of the checkpoint-composed served tensor; root over the ordered list.
    `fields` (field, shape, bits) replaces the Program's `weights` struct parameter (`served_record`); `on_field(field, words)`
    is given each field it digests, its raw words row-major as digested."""
    t0 = time.perf_counter()
    inst, res = program["instances"], program.get("result") or {}
    tp = res.get("tp")
    if not isinstance(tp, dict):
        tp = {}
    world, rank = int(tp.get("world") or 1), int(tp.get("rank") or 0)
    fields = program_weight_fields(inst) if fields is None else fields
    if prefix is None:
        named = MF.wrapper_prefix_of(inst.get("rows", []), res.get("attention_impls_observed") or [])
        # the rows' answer when it is the checkpoint's, else the prefix (up to two segments: GLM-5.3's `inner.model.`) under which the
        # most fields are checkpoint tensors 1:1 -- determined from the checkpoint, not assumed
        def hits(p: str) -> int:
            return sum(1 for f, _s, _b in fields if ck.has(strip_prefix(f, p)))
        cands = sorted({".".join(f.split(".")[:k]) + "." for f, _s, _b in fields for k in (1, 2) if f.count(".") >= k} | {"", named})
        prefix = max(cands, key=lambda p: (hits(p), p == named, -len(p)))
    out_fields: list[dict[str, Any]] = []
    unpinned: list[str] = []
    unresolved: list[dict[str, str]] = []
    n_bytes = 0
    analytic_fields: list[str] = []
    dims = {f: (s, b) for f, s, b in fields}       # DeepGEMM linears and MLA's absorbed kv_b: vLLM's load-time ue8m0 rewrite (fp8_ue8m0)
    roles = fp8_ue8m0.build_roles(program.get("dir"))
    rw = fp8_ue8m0.Rewrites(list(dims), roles, lambda f: compose(strip_prefix(f, prefix), *dims[f], ck, world=world, rank=rank), ck.config)
    for field, shape, bits in fields:
        live = strip_prefix(field, prefix)
        if ENGINE_CONSTANT_RE.search(field) and not ck.has(live):
            try:
                ent = _engine_constant(field, live, shape, bits, ck, kv_cache_dtype)
            except ValueError as e:
                unresolved.append({"field": field, "why": str(e)[:300]})
                out_fields.append({"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": None, "pinned": False, "why": str(e)[:300]})
                continue
            if ent is None or ent.get("pinned") is False:
                unpinned.append(field)
                out_fields.append(ent or {"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": None, "pinned": False,
                                          "why": "engine constant (not a checkpoint tensor)"})
            else:
                if ent.get("pinned") == "analytic":
                    analytic_fields.append(field)
                out_fields.append(ent)
            continue
        try:
            arr, note = (rw.compose(field, shape) if rw.covers(field) else derived_weights.compose(live, shape, bits, roles.get(field), ck, world, rank)
                         or compose(live, shape, bits, ck, world=world, rank=rank))
        except (KeyError, ValueError) as e:
            unresolved.append({"field": field, "why": str(e)[:300]})
            out_fields.append({"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": None, "pinned": False, "why": str(e)[:300]})
            continue
        b = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)      # arr.tobytes()'s bytes, not copied again
        n_bytes += b.nbytes
        if on_field is not None:
            on_field(field, arr)
        out_fields.append({"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": hashlib.sha256(b).hexdigest(),
                           "pinned": True, **note})
    root = root_of_fields(out_fields)
    n_pinned = sum(1 for f in out_fields if f.get("digest") or f.get("pinned") == "analytic")
    shards_ok = all(v.get("pinned") is True for v in ck.pin["shards"].values()) if ck.pin["shards"] else None
    program_digest = str(res.get("program_digest") or (inst.get("digest") or ""))
    return {"schema": SCHEMA, "program_digest": program_digest, "program_dir": program.get("dir"), "wrapper_prefix": prefix,
            "tp": {"world": world, "rank": rank}, "checkpoint": {"dir": ck.dir, "config_tie_word_embeddings": bool(ck.config.get("tie_word_embeddings", False)),
                                                                   **ck.pin, "all_shards_pinned": shards_ok},
            "root_of_record": root, "fields": out_fields, "n_fields": len(fields), "n_pinned": n_pinned, "bytes": n_bytes,
            "n_analytic": len(analytic_fields), "analytic_fields": analytic_fields,      # pinned by the frozen analytic table (bounded), no byte digest
            "engine_assumptions": {"kv_cache_dtype": str(kv_cache_dtype or "auto")},
            "unpinned": unpinned, "unresolved": unresolved, "complete": not unresolved,
            "of_record": (not unresolved) and shards_ok is True,          # [W1] composed from the PINNED shards, every field resolved
            "why": None if not unresolved else f"{len(unresolved)} Program weight field(s) have no checkpoint tensor of record: " + ", ".join(u["field"] for u in unresolved[:8]),
            "wall_s": round(time.perf_counter() - t0, 3)}


# ---------------------------------------------------------------------------------------------------------------------- live side

def live_param_digests(model, *, include_buffers: bool = True, raw_for: re.Pattern[str] | None = ENGINE_ROTARY_RE) -> dict[str, Any]:
    """sha256 of every live parameter (and buffer) of the served model, by `named_parameters()` name: the SAME tensors the committer's
    weights root covers (register_weights), hashed as raw contiguous bytes (bf16 words row-major).  Tensors whose name matches `raw_for`
    (default: the rotary cos_sin table) also keep their raw words under `raw[name]` (numpy uint16, in memory only -- not for the row
    record) so `check` can run the analytic bound on them.  Hashed a chunk at a time on a thread pool (`param_digests`)."""
    return param_digests(model, include_buffers=include_buffers, raw_for=raw_for)


def _check_analytic(f: dict[str, Any], raw: np.ndarray | None) -> dict[str, Any]:
    """The served rotary table against the frozen analytic table's bound (analytic.check_cos_sin)."""
    from verity_vllm.program import model as analytic
    a = f.get("analytic") or {}
    rep: dict[str, Any] = {"field": f["field"], "served_name": f["served_name"], "table": a.get("table"), "expected_shape": a.get("expected_shape")}
    if raw is None:
        return rep | {"status": "missing", "why": "served table words not captured (live_param_digests raw_for) or not bf16"}
    if a.get("table") != "cos_sin":
        return rep | {"status": "error", "why": f"no validator for analytic table {a.get('table')!r}"}
    C = dict(a.get("C") or {})
    try:
        r = analytic.check_cos_sin(C, np.ascontiguousarray(raw, dtype=np.uint16))
    except Exception as e:                                        # noqa: BLE001 -- the reason is the evidence
        return rep | {"status": "error", "why": f"check_cos_sin raised {type(e).__name__}: {e}"[:300]}
    keep = {k: r.get(k) for k in ("status", "shape", "shape_ok", "reason", "n_elements", "n_diff", "max_ulp", "n_outside_bound", "max_excess", "first_offending", "ulp_histogram")}
    if r.get("status") != "pass":
        keep["why"] = r.get("reason") or f"{r.get('n_outside_bound')} of {r.get('n_elements')} served words outside the analytic bound (max ulp {r.get('max_ulp')})"
    return rep | keep


def check(record: dict[str, Any], live: dict[str, Any], *, program_digests_of_record: set[str] | list[str] | None = None,
          checkpoint_of_record: dict[str, Any] | None = None, kv_cache_dtype: str | None = None) -> dict[str, Any]:
    """Field by field: the checkpoint-composed digest of record == the served parameter's digest.  Any difference or a Program field
    missing from the served model -> result False, NAMED.  Fields the record leaves unpinned are listed, not compared.

    The record is OF RECORD only when every shard it was composed from carries the sha256 `manifests/checkpoints.json`
    pins: an unpinned / differently-hashed shard (a copy outside the manifest, a substituted file) -> result False, named -- 'served ==
    whatever is on disk' is not the pin.  [W2] with `program_digests_of_record` (the Build digest of record / the manifest's component
    digests) a record built for another Program is refused by name; with `checkpoint_of_record` ({repo, revision}) a record whose
    manifest entry is another checkpoint is refused by name."""
    lp = live.get("params") or {}
    mismatched, missing, compared = [], [], 0
    ck = record.get("checkpoint") or {}
    shard_why: list[str] = []
    for fn, v in (ck.get("shards") or {}).items():
        if v.get("pinned") is None:
            ent = ck.get("entry")
            shard_why.append(f"shard {fn}: no checkpoints.json entry for {ck.get('dir')}" + (f" ({ent.get('repo')} @ {str(ent.get('revision'))[:10]})" if ent else " (repo/revision unknown)"))
        elif v.get("pinned") is not True:
            shard_why.append(f"shard {fn} sha256 {str(v.get('sha256'))[:16]} != manifests/checkpoints.json pin {str(v.get('pinned_sha256'))[:16]}")
    if not (ck.get("shards") or {}):
        shard_why.append("no shard pin recorded (record built with --no-verify-shards?)")
    shards_ok = ck.get("all_shards_pinned") is True and not shard_why
    prog_why = None
    if program_digests_of_record is not None:
        pd = str(record.get("program_digest") or "")
        ofr = {str(x) for x in program_digests_of_record}
        stamped = record.get("of_record_set") if isinstance(record.get("of_record_set"), dict) else None
        st = {str(x) for x in (stamped.get("program_digests") or [])} if stamped is not None else set()
        if short := sorted(x[:16] for x in ofr | st | ({pd} if pd else set()) if not FULL_DIGEST_RE.fullmatch(x)):
            prog_why = f"Program digests not full sha512 (128 hex), compared only whole: {short}"
        elif stamped is not None:
            # a record stamped with the WHOLE of-record set is accepted iff that set equals the run's: a record of another
            # row -- more, fewer or other components -- is refused by name; the stamp itself must be sound (every component's struct equal)
            if stamped.get("why"):
                prog_why = f"record's of-record stamp is not sound: {str(record.get('of_record_why') or stamped.get('why'))[:200]}"
            elif not st:
                prog_why = "record's of-record stamp is empty"
            else:
                miss_run, miss_rec = sorted(x[:16] for x in ofr - st), sorted(y[:16] for y in st - ofr)
                if miss_run or miss_rec:
                    prog_why = (f"record's of-record set != the run's: run digests not on the record {miss_run}; record digests not in the run {miss_rec}")
        elif not pd or pd not in ofr:
            prog_why = f"record built for Program {pd[:16] or '?'} != Program of record {sorted(x[:16] for x in ofr)}"
    ent_why = None
    if checkpoint_of_record:
        ent = ck.get("entry") or {}
        want_repo, want_rev = checkpoint_of_record.get("repo"), str(checkpoint_of_record.get("revision") or "")
        if want_repo and ent.get("repo") != want_repo:
            ent_why = f"record's checkpoint entry {ent.get('repo')} != the workload's model {want_repo}"
        elif want_rev and not str(ent.get("revision") or "").startswith(want_rev[:10]):
            ent_why = f"record's checkpoint revision {str(ent.get('revision'))[:10]} != the workload's pin {want_rev[:10]}"
    # the record's engine assumption must be the engine's: a record that pinned the kv scales as 1.0 (kv_cache_dtype auto)
    # does not certify an engine running a quantized kv cache (its scales are other values) -- refused by name, never re-assumed
    kv_why = None
    rec_kvd = str((record.get("engine_assumptions") or {}).get("kv_cache_dtype") or "auto")
    if kv_cache_dtype is not None and str(kv_cache_dtype) != rec_kvd:
        kv_why = f"record assumes kv_cache_dtype={rec_kvd}; the engine runs kv_cache_dtype={kv_cache_dtype} (kv scales not of record)"
    analytic_checked: list[dict[str, Any]] = []
    raws = live.get("raw") or {}
    for f in record.get("fields", []):
        cands = [f["served_name"], f["field"]]
        if f["served_name"] == "lm_head.weight":
            cands.append("model.embed_tokens.weight")
        if f.get("pinned") == "analytic":
            got = next((lp[c] for c in cands if c in lp), None)
            if got is None:
                missing.append(f["field"])
                continue
            compared += 1
            analytic_checked.append(_check_analytic(f, next((raws[c] for c in cands if c in raws), None)))
            continue
        if not f.get("digest"):
            continue
        got = next((lp[c] for c in cands if c in lp), None)
        if got is None:
            missing.append(f["field"])
            continue
        compared += 1
        if str(got.get("digest")) != str(f["digest"]):
            mismatched.append({"field": f["field"], "served_name": f["served_name"], "of_record": f["digest"][:16], "served": str(got.get("digest"))[:16],
                               "shape": f.get("shape"), "served_dtype": got.get("dtype")})
    # a loader placeholder (`attn.k_scale`, no underscore) the engine deleted after loading is absent by construction, not
    # missing: moved out of `missing` by name iff the record assumes kv_cache_dtype auto AND the `attn._k_scale` the kernels read was
    # itself found and compared (so the scale that reaches the served values IS pinned).
    placeholders_deleted: list[str] = []
    if rec_kvd == "auto" and missing:
        # the layers whose engine-constant `attn._x_scale` (the scale the kernels read) IS among the served parameters and was compared
        layers_with_served_scale = {f["field"].rsplit(".attn._", 1)[0] + ".attn." for f in record.get("fields", [])
                                    if ENGINE_CONSTANT_RE.search(f["field"]) and ".attn._" in f["field"] and any(c in lp for c in (f["served_name"], f["field"]))}
        still_missing = []
        for name in missing:
            m = PLACEHOLDER_RE.match(name)
            if m and m.group(1) in layers_with_served_scale:
                placeholders_deleted.append(name)
            else:
                still_missing.append(name)
        missing = still_missing
    analytic_failed = [a for a in analytic_checked if a.get("status") != "pass"]
    ok = (record.get("complete", False) and shards_ok and prog_why is None and ent_why is None and kv_why is None and not mismatched and not missing
          and not analytic_failed)
    why = None
    if not record.get("complete", False):
        why = "weights of record incomplete: " + str(record.get("why"))
    elif not shards_ok:
        why = "checkpoint shards are not the pinned ones (W1): " + "; ".join(shard_why[:4])
    elif prog_why:
        why = prog_why
    elif ent_why:
        why = ent_why
    elif kv_why:
        why = kv_why
    elif mismatched:
        why = f"{len(mismatched)} served weight(s) differ from the checkpoint of record: " + ", ".join(m["served_name"] for m in mismatched[:8])
    elif missing:
        why = f"{len(missing)} Program weight field(s) not found among the served parameters: " + ", ".join(missing[:8])
    elif analytic_failed:
        why = f"{len(analytic_failed)} engine table(s) outside the frozen analytic bound: " + "; ".join(f"{a['served_name']}: {a.get('why')}" for a in analytic_failed[:4])
    unpinned = list(record.get("unpinned") or [])
    return {"result": bool(ok), "root_of_record": record.get("root_of_record"), "program_digest": record.get("program_digest"),
            "program_of_record": bool(prog_why is None) if program_digests_of_record is not None else None,
            "of_record_set": ((record.get("of_record_set") or {}).get("program_digests") if isinstance(record.get("of_record_set"), dict) else None),
            "checkpoint_entry": ck.get("entry"), "checkpoint_entry_of_record": bool(ent_why is None) if checkpoint_of_record else None,
            "compared": compared, "mismatched": mismatched, "missing": missing, "unpinned": unpinned,
            "placeholders_deleted": placeholders_deleted,
            "placeholders_deleted_note": ((f"{len(placeholders_deleted)} Fp8KVCacheMethod loader placeholder(s) (attn.q/k/v/prob_scale) pinned as the -1.0 "
                                          "sentinel on the record are not served: vLLM's process_weights_after_loading deletes them after weight loading; "
                                          "the attn._x_scale the kernels read are compared") if placeholders_deleted else None),
            "analytic": analytic_checked, "engine_assumptions": record.get("engine_assumptions"),
            "unpinned_note": (("engine constants left unpinned by name are not authenticated by this pin (form (B) only)") if unpinned else
                              "every declared weight field is pinned: checkpoint bytes by digest, kv scales as the engine constant, the rotary table by the frozen analytic bound"),
            "unresolved": list(record.get("unresolved") or []), "shards": ck.get("shards"),
            "all_shards_pinned": ck.get("all_shards_pinned"), "shards_ok": bool(shards_ok),
            "method": "sha256(served parameter bytes) == sha256(checkpoint-composed tensor of record) per Program weight field, shards pinned by "
                      "manifests/checkpoints.json sha256, record bound to the Program of record; engine constants: kv scales "
                      "== f32 1.0 under kv_cache_dtype auto, rotary table within analytic.check_cos_sin's bound",
            "why": why}


# ---------------------------------------------------------------------------------------------------------------------------- CLI

def _default_manifest() -> str | None:
    """The tree's checkpoint pins (`config.CHECKPOINTS`); None on a tree that holds none."""
    return str(config.CHECKPOINTS) if config.CHECKPOINTS.exists() else None


def record_shard_pin(checkpoint_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None,
                     revision: str | None = None) -> dict[str, Any] | None:
    """The shard pin `build_record` takes with the same arguments: taken once by a TP driver, passed to every rank as `shard_pin`.
    None when the checkpoint or manifest can't be read: each rank then takes the pin itself and its record names what failed."""
    try:
        return shard_pin(checkpoint_dir, checkpoints_manifest=checkpoints_manifest or _default_manifest(), model=model, revision=revision)
    except (OSError, ValueError) as e:
        logger.warning("no shard pin of %s was taken, so each rank takes its own: %s: %s", checkpoint_dir, type(e).__name__, e)
        return None


def build_record(program_dir: str, checkpoint_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None,
                 revision: str | None = None, verify_shards: bool = True, kv_cache_dtype: str = "auto",
                 shard_pin: dict[str, Any] | None = None) -> dict[str, Any]:
    """The record of one Program dir.  `shard_pin` (`record_shard_pin` of the same checkpoint and manifest) is adopted instead of hashing
    every shard again; a pin of other shards or another manifest is refused.  The Program's rows are streamed, never expanded."""
    prog = load_program_head(program_dir)
    ck = CheckpointIndex(checkpoint_dir, checkpoints_manifest=checkpoints_manifest or _default_manifest(), model=model, revision=revision,
                         verify_shards=verify_shards, pin=shard_pin)
    return weights_of_record(prog, ck, kv_cache_dtype=kv_cache_dtype)


def served_weight_fields(program: dict[str, Any]) -> list[tuple[str, list[int], int]]:
    """(field, shape, 16) for every weight a served Build's Program reads (`acquire.served`: a served parameter or buffer by name, read
    as bf16 words by the Definition bound to it), in first-read order.  A weight read at two shapes is refused."""
    seen: dict[str, list[int]] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if "weight" in x and seen.setdefault(x["weight"], list(x["shape"])) != list(x["shape"]):
                raise ValueError(f"{x['weight']} is read at {seen[x['weight']]} and at {list(x['shape'])}")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(program)
    return [(n, s, 16) for n, s in seen.items()]


def served_record(build: dict[str, Any], checkpoint_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None,
                  revision: str | None = None, kv_cache_dtype: str = "auto") -> dict[str, Any]:
    """The record of a served Build: its Program's weight reads (`served_weight_fields`, by the served names: no wrapper prefix),
    bound to the Build's program digest."""
    ck = CheckpointIndex(checkpoint_dir, checkpoints_manifest=checkpoints_manifest or _default_manifest(), model=model, revision=revision)
    return weights_of_record({"instances": {}, "result": {"program_digest": build["program_digest"]}, "dir": None}, ck, prefix="",
                             kv_cache_dtype=kv_cache_dtype, fields=served_weight_fields(build["program"]))


def weights_struct(program: dict[str, Any]) -> list[list[Any]]:
    """The weights STRUCT of a Program -- [field, shape, bits] in declaration order -- the model-level object the record is a function of
    (no checkpoint read).  Two Programs of one model at different sequence shapes have the same struct."""
    return [[f, list(shape), int(bits)] for f, shape, bits in program_weight_fields(program["instances"])]


def stamp_of_record_set(record: dict[str, Any], programs_root: str, *, manifest: str | None = None) -> dict[str, Any]:
    """Stamp the WHOLE of-record set on the record instead of one arbitrary component's digest:
    every component Program under `programs_root` (build_request_LP*_T* + build_request) + the workload digest (build_workload/
    workload_program.json) + the manifest's program / component digests.  Each component's weights struct must EQUAL the record's
    (field list, shapes, bits): a component whose struct differs is named and the record is not of record (a record derived from one
    shape never silently stands for a Program of another struct).  `check()` then requires the run's of-record set to equal the stamp."""
    my_struct = [[f["field"], list(f["shape"]), int(f["bits"])] for f in record.get("fields", [])]
    # the manifest's digests first: they define which Program dirs under the root are OF RECORD (a default-shape build_request that is
    # not a workload component is struct-checked but its digest does not join the set -- the run's set never carries it)
    mpath = manifest or str(Path(programs_root) / "manifest.json")
    manifest_digests: list[str] = []
    why: list[str] = []
    if os.path.exists(mpath):
        try:
            m = json.loads(Path(mpath).read_text())
            manifest_digests = [str(x) for x in ([m.get("program_digest")] + list(m.get("component_digests") or [])
                                                 + [c.get("program_digest") for c in (m.get("components") or {}).values() if isinstance(c, dict)]) if x]
        except Exception as e:  # noqa: BLE001
            why.append(f"manifest {os.path.basename(mpath)} unreadable ({type(e).__name__})")
    digests: set[str] = set(manifest_digests)
    comps: list[dict[str, Any]] = []
    dirs = component_program_dirs(programs_root)
    shaped = [d for d in dirs if os.path.basename(d) != "build_request"]
    for d in dirs:
        try:
            prog = MF.load_program(d)
        except Exception as e:  # noqa: BLE001
            why.append(f"{os.path.basename(d)}: not loadable ({type(e).__name__}: {str(e)[:80]})")
            continue
        res = prog.get("result") or {}
        pd = str(res.get("program_digest") or (prog["instances"].get("digest") or ""))
        st = weights_struct(prog)
        same = st == my_struct
        in_manifest = bool(pd) and pd in manifest_digests
        # of record: a shaped component dir (build_request_LP*_T*), or build_request when it is the only Program (B1) or the manifest names it
        of_rec = bool(pd) and (os.path.basename(d) != "build_request" or not shaped or in_manifest)
        comps.append({"dir": os.path.basename(d), "program_digest": pd, "LP": res.get("LP"), "T": res.get("T"), "struct_equal": same,
                      "n_fields": len(st), "of_record": of_rec, "in_manifest": in_manifest})
        if not same:
            mine = {f for f, _s, _b in my_struct}
            theirs = {f for f, _s, _b in st}
            diff = sorted(mine ^ theirs)[:4] or sorted(f for (f, s, b) in st if [f, s, b] not in my_struct)[:4]
            why.append(f"{os.path.basename(d)} ({pd[:16]}): weights struct differs from the record's ({len(st)} vs {len(my_struct)} fields; e.g. {diff})")
        elif of_rec:
            digests.add(pd)
    wl = Path(programs_root) / "build_workload" / "workload_program.json"
    workload_digest = None
    if wl.exists():
        try:
            workload_digest = str(json.loads(Path(wl).read_text()).get("workload_digest") or "") or None
        except Exception as e:  # noqa: BLE001
            why.append(f"build_workload/workload_program.json unreadable ({type(e).__name__})")
    if workload_digest:
        digests.add(workload_digest)
    rec_pd = str(record.get("program_digest") or "")
    if short := sorted(x[:16] for x in digests | ({rec_pd} if rec_pd else set()) if not FULL_DIGEST_RE.fullmatch(x)):
        why.append(f"Program digests not full sha512 (128 hex): {short}")
    elif rec_pd and rec_pd not in digests:
        why.append(f"the record's own Program {rec_pd[:16]} is not of record under {os.path.basename(str(programs_root))} (derived from a non-component)")
    if not comps:
        why.append(f"no component Program under {programs_root}")
    record["of_record_set"] = {"program_digests": sorted(digests), "components": comps, "workload_digest": workload_digest,
                               "manifest_digests": sorted(set(manifest_digests)), "programs_root": str(programs_root), "why": why or None,
                               "rule": "every component Program's weights struct == the record's; the run's of-record set must equal this set"}
    if why:
        record["of_record"] = False
        record["of_record_why"] = "; ".join(why)[:600]
    return record


def derivation_provenance(program_dir: str, checkpoint_dir: str, checkpoints_manifest: str | None, argv: list[str] | None, *,
                          supersedes: str | None = None) -> dict[str, Any]:
    """[Record repairs are LOGGED DERIVATIONS] the provenance line of a record written by this CLI: tool, the code that
    ran it (the repo's HEAD when this file sits in a git tree, else null, with `code_source` naming what failed), the SOURCE FILES the derivation read (the Program's instances.json.gz
    + result.json, the checkpoint dir + checkpoints.json) and the time; `supersedes` names the record this one replaces (row_pod moves a
    stale / older record aside before re-deriving).  Informational: never read by check() -- the digests are the evidence."""
    code = None
    code_source = None
    unread: list[str] = []
    try:
        # the release tree on a pod has no .git: ship.sh's EXPORT.json names the shipped commit (experiment.code_version reads it first,
        # then git); the first pod records carried code null for that reason
        from verity_vllm.pipeline.experiment import code_version
        cv = code_version(config.ROOT)
        code, code_source = cv.get("commit") or None, cv.get("source")
        if code and cv.get("dirty_patch_sha256"):
            code_source = f"{code_source} (dirty patch {str(cv['dirty_patch_sha256'])[:12]})"
    except Exception as e:  # noqa: BLE001
        code = None
        unread.append(f"code_version raised {type(e).__name__}: {e}")
    if code is None:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            code = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
            code_source = "git" if code else None
        except Exception as e:  # noqa: BLE001
            code = None
            unread.append(f"git rev-parse raised {type(e).__name__}: {e}")
    if unread:
        code_source = (f"git ({'; '.join(unread)})" if code else f"unreadable: {'; '.join(unread)}")[:240]
    return {"tool": "verity_vllm.check.weights_of_record", "code": code, "code_source": code_source, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": {"program_dir": os.path.abspath(program_dir), "program_files": ["instances.json.gz", "result.json"],
                        "checkpoint_dir": os.path.abspath(checkpoint_dir), "checkpoints_manifest": os.path.abspath(checkpoints_manifest) if checkpoints_manifest else "repo default"},
            "argv": list(argv) if argv is not None else sys.argv[1:], "supersedes": os.path.abspath(supersedes) if supersedes else None}


@dataclass(frozen=True, kw_only=True)
class Options:
    """weights of record: per-field digests of the checkpoint-composed served weights"""
    program_dir: str | None = option("--program-dir", default=None, help="the Program of record (instances.json.gz + result.json)")
    programs_root: str | None = option("--programs-root", default=None, help="a row dir: derive from its first component Program and stamp EVERY component's digest + the workload digest as the of-record set (each component's weights struct must equal the record's)")
    manifest: str | None = option("--manifest", default=None, help="the row's required manifest (default <programs-root>/manifest.json): its digests join the stamp")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="the pinned checkpoint snapshot (safetensors shards + config.json); required to derive")
    diff: list[str] | None = option("--diff", nargs=2, metavar=("A.json", "B.json"), default=None, help="READER (no derivation, nothing written): compare two records field by field and re-root both under the current schema; exit 0 when they pin the same weights of record (same-model rows across pods under schema v1)")
    checkpoints: str | None = option("--checkpoints", default=None, help="manifests/checkpoints.json (default: the repo's)")
    model: str | None = option("--model", default=None, help="HF repo id, to select the checkpoints.json entry when --checkpoint-dir is not its local_path")
    revision: str | None = option("--revision", default=None)
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true", help="skip the shard sha256 (the record then says pinned: null)")
    kv_cache_dtype: str = option("--kv-cache-dtype", default="auto", help="the engine's kv_cache_dtype the record assumes (auto: kv scales pinned as f32 1.0)")
    out: str | None = option("--out", default=None, help="write the record here (default: <program-dir>/weights_of_record.json)")
    supersedes: str | None = option("--supersedes", default=None, help="the record this derivation replaces (moved aside by the caller), named in `derived`")


def main(a: Options) -> int:
    if a.diff:
        A, B = (json.loads(Path(x).read_text()) for x in a.diff)
        d = diff_records(A, B)
        print(f"weights_of_record --diff: schema {d['schema'][0]} vs {d['schema'][1]}; roots as recorded {str(d['root_of_record'][0])[:16]} / "
              f"{str(d['root_of_record'][1])[:16]}; fields {d['n_fields'][0]} / {d['n_fields'][1]}; equal {d['equal_fields']}, differ {len(d['differ'])}, "
              f"only A {len(d['only_a'])}, only B {len(d['only_b'])}, host-only {len(d['host_only'])}")
        for x in d["differ"][:20]:
            print(f"  DIFFER {x['field']}: {', '.join(x['keys'])}  A {json.dumps(x['a'], default=str)[:80]}  B {json.dumps(x['b'], default=str)[:80]}")
        for x in d["only_a"][:10]:
            print(f"  ONLY A {x}")
        for x in d["only_b"][:10]:
            print(f"  ONLY B {x}")
        for x in d["host_only"]:
            print(f"  host-only {x['field']}: analytic.inv_freq_sha256 {x['a']}.. vs {x['b']}.. (the deriving host's numpy value; not in the v2 root)")
        print(f"  re-root under {SCHEMA}: {d['reroot'][0][:16]} / {d['reroot'][1][:16]} -> same weights of record: {d['same_weights_of_record']}")
        return 0 if d["same_weights_of_record"] else 1
    given = a.program_dir or a.programs_root
    if not given:
        raise UsageError("one of --program-dir / --programs-root is required")
    if not a.checkpoint_dir:
        raise UsageError("--checkpoint-dir is required to derive a record")
    pdir = a.program_dir
    if not pdir:
        cands = component_program_dirs(given)
        if not cands:
            print(f"weights_of_record: no component Program (build_request*/instances.json.gz) under {a.programs_root}")
            return 2
        pdir = cands[0]
    rec = build_record(pdir, a.checkpoint_dir, checkpoints_manifest=a.checkpoints, model=a.model, revision=a.revision,
                       verify_shards=not a.no_verify_shards, kv_cache_dtype=a.kv_cache_dtype)
    if a.programs_root:
        stamp_of_record_set(rec, a.programs_root, manifest=a.manifest)
    out = a.out or os.path.join(a.programs_root or pdir, "weights_of_record.json")
    rec["derived"] = derivation_provenance(pdir, a.checkpoint_dir, a.checkpoints, None, supersedes=a.supersedes)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"weights_of_record: root {rec['root_of_record'][:16]} fields {rec['n_pinned']}/{rec['n_fields']} pinned ({rec['n_analytic']} analytic), unpinned {len(rec['unpinned'])}, "
          f"unresolved {len(rec['unresolved'])}, shards pinned {rec['checkpoint'].get('all_shards_pinned')}, complete {rec['complete']}, "
          f"of_record {rec['of_record']} -> {out}")
    if rec.get("of_record_set"):
        ors = rec["of_record_set"]
        print(f"  of-record set: {len(ors['program_digests'])} digests ({len(ors['components'])} components"
              f"{', workload ' + ors['workload_digest'][:16] if ors.get('workload_digest') else ''}); struct equal on all: {not ors.get('why')}"
              + (f" -- {rec.get('of_record_why')}" if ors.get("why") else ""))
    if rec["unresolved"]:
        for u in rec["unresolved"][:20]:
            print(f"  UNRESOLVED {u['field']}: {u['why']}")
    return 0 if rec["of_record"] else 2
