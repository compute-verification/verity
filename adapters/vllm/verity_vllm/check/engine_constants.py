"""The weight fields no checkpoint carries: the engine's constants, pinned from the declared configuration.

A Program's `weights` struct names fields the engine creates itself: the attention kv-cache scales, vLLM's loader
placeholders for them on a quantized checkpoint, and the rotary cos/sin table.  `_engine_constant` gives the record's entry
for such a field (pinned exactly, pinned `analytic`, or unpinned by name with the reason); `kv_scale_remap_sources` names the
checkpoint tensors vLLM renames onto a served kv scale; `analytic_key` is the key an analytic field enters the record's
root under.  `weights_of_record` calls these for a field that matches `ENGINE_CONSTANT_RE` and that the checkpoint lacks.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from verity_vllm.check.checkpoint_index import CheckpointIndex
from verity_vllm.check.composed_weights import to_served

#: Program weight fields that are engine constants, not checkpoint tensors (listed by name; not pinned by this module): the attention
#: kv-cache scales (f32 1.0 without kv quantization) and the rotary cos/sin table (a function of config.json: rope_theta, rope_scaling,
#: head_dim, max_position_embeddings -- derivable, v2).  Their effect on the served values is under the Match oracle (form B).
#: a QUANTIZED checkpoint's attention layers also carry vLLM's loader placeholders `attn.q_scale / k_scale /
#: v_scale / prob_scale` (no underscore): `Fp8KVCacheMethod.create_weights` -> `KVCacheScaleParameter()` = the f32 sentinel -1.0,
#: "overwritten when loading weights" IF the checkpoint ships kv scales (then `compose` finds them 1:1: the `not ck.has(live)` guard),
#: created by `torch.tensor(-1.0)` at the MODEL dtype (bf16 -> `Value<16>` in the Program; F-fp8-PIN2: the measured Qwen3-4B-FP8 B1/B8
#: Programs declare them Value<16>, `_k_scale` Value<32>) -- pinned as -1.0 in the declared width,
#: otherwise left at the sentinel; with the kv cache unquantized (kv_cache_dtype auto) `process_weights_after_loading` enforces
#: `_k_scale = _v_scale = 1.0` and never touches the placeholders, which no kernel reads (Qwen3-4B-FP8 B1: 144 such fields, 723 - 435
#: checkpoint - 144 `_x_scale`).  Pinned EXACTLY as the sentinel under kv_cache_dtype auto (a served value other than -1.0 means the
#: loader wrote it from somewhere that is not the pinned checkpoint); any other kv_cache_dtype: unpinned by name (the scale is then live).
#: vLLM d9105ea80's `BaseKVCacheMethod.process_weights_after_loading` ends every branch with `del layer.k_scale /
#: v_scale / q_scale / prob_scale` (kv_cache.py:106-109, 209-212; the early-return branch asserts they are already gone): the placeholders
#: exist only between `create_weights` and the end of weight loading, so the served model has NO parameter to compare the record's
#: pinned sentinel against.  #74 (Qwen3-4B-FP8 B8, H100): 579/579 compared equal, 144 placeholders "not found among the served
#: parameters", 3/3 pairs FAIL with every other check green.  A deleted parameter cannot reach a served value; `check_served` therefore
#: reports a placeholder field absent from the served model as `placeholders_deleted` (by name, counted, in the result), not as
#: `missing` -- only for fields matching PLACEHOLDER_RE, only under the record's kv_cache_dtype=auto assumption, and only when the
#: `_x_scale` the kernels do read is itself among the compared fields.  A placeholder that IS served is compared exactly as before.
ENGINE_CONSTANT_RE = re.compile(r"([._]attn\._?(k|v|q|prob)_scale|\.rotary_emb\.cos_sin_cache)$")
PLACEHOLDER_RE = re.compile(r"^(.*\.attn\.)(k|v|q|prob)_scale$")
ENGINE_KV_SCALE_RE = re.compile(r"[._]attn\._(k|v|q|prob)_scale$")
ENGINE_KV_PLACEHOLDER_RE = re.compile(r"\.attn\.(k|v|q|prob)_scale$")       # [F-fp8-PIN1] Fp8KVCacheMethod loader placeholders
ENGINE_ROTARY_RE = re.compile(r"\.rotary_emb\.cos_sin_cache$")


def kv_scale_remap_sources(served_name: str) -> list[str]:
    """The checkpoint names vLLM's `weight_utils.maybe_remap_kv_scale_name` renames onto a served
    `<layer>.attn.<x>_scale` (x in k, v, q, prob) at load: kv_cache_scheme checkpoints (`<layer>.<x>_scale`, `<layer>.<x>_proj.<x>_scale`),
    modelopt (`<layer>.<x>_proj.output_scale`), the legacy fused `<layer>.kv_scale` (k and v).  `<layer>` = the served name up to `.attn.`.
    Ordered; the caller pins 1:1 from the ONE source present and refuses two.  `attn._<x>_scale` (an engine constant) -> []."""
    m = re.search(r"^(.*)\.attn\.(k|v|q|prob)_scale$", served_name)
    if not m:
        return []
    layer, x = m.group(1), m.group(2)
    out = [f"{layer}.{x}_scale", f"{layer}.{x}_proj.{x}_scale", f"{layer}.{x}_proj.output_scale"]
    if x in ("k", "v"):
        out.append(f"{layer}.kv_scale")
    return out


def _engine_constant(field: str, live: str, shape: list[int], bits: int, ck: CheckpointIndex, kv_cache_dtype: str) -> dict[str, Any] | None:
    """The engine constants the Program declares as weights but no checkpoint carries, pinned from
    config.json / the engine's declared configuration instead of left unpinned:
      * attention kv-cache scales `attn._{k,v,q,prob}_scale`: without kv-cache quantization (kv_cache_dtype auto) vLLM creates them as
        f32 1.0 -- pinned EXACTLY (digest of the f32 word); any other kv_cache_dtype has its scales in the checkpoint (then `compose`
        finds them) or is left unpinned BY NAME with the dtype in the why (never assumed).
      * the rotary table `rotary_emb.cos_sin_cache` [MAX_POS, ROT] bf16: an analytic table (`acquire.plan.analytic_table_of_buffer`,
        checked by analytic.check_cos_sin, gate I9) -- the served words must round from values within the fp32
        pipeline's error bound of the analytic function of config.json (theta, D/ROT, MAX_POS, rope_scaling); there is no byte digest
        (CUDA libm != numpy libm at the last bf16 ulp), so the field is pinned 'analytic' and `check` runs the bound on the served
        words.  A rope_scaling type without an analytic table stays unpinned by name.
    Returns the field entry, or None when the field is not an engine constant."""
    base = {"field": field, "served_name": live, "shape": shape, "bits": bits}
    if ENGINE_KV_SCALE_RE.search(field):
        kvd = str(kv_cache_dtype or "auto")
        if kvd != "auto":
            return base | {"digest": None, "pinned": False, "why": f"kv-cache scale under kv_cache_dtype={kvd} without a checkpoint tensor (no rule; not assumed 1.0)"}
        if bits != 32 or list(shape) not in ([], [1]):
            return base | {"digest": None, "pinned": False, "why": f"kv-cache scale declared {shape} x Value<{bits}>: the engine constant is one f32 word"}
        arr = np.ones(shape or [], dtype=np.float32)
        return base | {"digest": hashlib.sha256(arr.tobytes()).hexdigest(), "pinned": True, "sources": ["engine:kv_cache_dtype=auto"],
                       "rule": "engine constant: kv-cache scale == f32 1.0 (kv_cache_dtype auto -- no kv quantization; vLLM Attention.__init__)"}
    if ENGINE_KV_PLACEHOLDER_RE.search(field):
        # The remap-aware compose: BEFORE the placeholder rule, look the checkpoint up under
        # vLLM's REMAP SOURCES for this layer's scale (weight_utils.maybe_remap_kv_scale_name renames them to attn.<x>_scale at load): a
        # shipped scale is a checkpoint tensor, written by the loader and read by the kernel -- pinned 1:1 from its source as one f32 word
        # under ANY kv_cache_dtype; two sources for one scale, or a source that is not one word, -> unresolved (fail closed).  Only a
        # checkpoint with NO source for this scale reaches the placeholder rule below (the sentinel, which such a checkpoint would fail by
        # name anyway).  The exact-name `ck.has(served)` case never arrives here (compose pins it first).
        srcs = kv_scale_remap_sources(live)
        found = [n for n in srcs if ck.has(n)]
        if found:
            if len(found) > 1:
                raise ValueError(f"{field}: {len(found)} checkpoint kv-scale sources remap onto it ({found}): ambiguous, unresolved")
            src = found[0]
            # [F-fp8-PIN2 on the remap branch] the loader copies the checkpoint scale INTO the placeholder Parameter, which was created at the
            # model dtype (torch.tensor(-1.0), no dtype) -- so the served word is the scale cast to the DECLARED width (bf16 on a bf16 model,
            # RNE by copy_; f32 where the Program declares Value<32>); composed the same way; another width has no rule -> unresolved
            if int(bits) not in (16, 32) or list(shape) not in ([], [1]):
                raise ValueError(f"{field}: declared {shape} x Value<{bits}> but the checkpoint kv scale {src} is one bf16/f32 word: unresolved")
            a = ck.array(src)
            if a.size != 1:
                raise ValueError(f"{field}: checkpoint kv scale {src} has {a.size} elements, the served scale is one word: unresolved")
            words, cast = to_served(a.reshape(shape or []), ck.dtype(src), int(bits))
            return base | {"digest": hashlib.sha256(np.ascontiguousarray(words).tobytes()).hexdigest(), "pinned": True, "sources": [src],
                           "rule": f"kv scale (remapped from checkpoint {src} -> {live}; vLLM maybe_remap_kv_scale_name; one word in the declared Value<{bits}>)",
                           **({"cast": cast} if cast else {})}
        # vLLM Fp8KVCacheMethod.create_weights: `KVCacheScaleParameter()` = `torch.tensor(-1.0)` with NO
        # dtype -> the sentinel is created at the model's default dtype (bf16 under `set_default_torch_dtype(model dtype)`: the served
        # Qwen3-4B-FP8 word is bf16 0xBF80, and the Program declares the field `Value<16>`; `_k_scale` is an explicit f32 1.0, `Value<32>`).
        # Pinned as -1.0 IN THE DECLARED WIDTH (one word, shape [] or [1]); a declared width other than 16/32 has no rule -> unpinned by name.
        # With no kv scale in the checkpoint and the kv cache unquantized it is never written after construction and no kernel reads it.
        kvd = str(kv_cache_dtype or "auto")
        if kvd != "auto":
            return base | {"digest": None, "pinned": False, "why": f"kv-scale loader placeholder under kv_cache_dtype={kvd} without a checkpoint tensor (no rule; not assumed)"}
        if int(bits) not in (16, 32) or list(shape) not in ([], [1]):
            return base | {"digest": None, "pinned": False, "why": f"kv-scale loader placeholder declared {shape} x Value<{bits}>: KVCacheScaleParameter is one bf16/f32 word"}
        word = np.array([0xBF80], dtype=np.uint16) if int(bits) == 16 else np.array([-1.0], dtype=np.float32)      # -1.0 in the declared width
        arr = np.broadcast_to(word, (1,)).astype(word.dtype)
        wdt = "bf16" if int(bits) == 16 else "f32"
        return base | {"digest": hashlib.sha256(arr.tobytes()).hexdigest(), "pinned": True,
                       "sources": [f"engine:Fp8KVCacheMethod.create_weights (KVCacheScaleParameter torch.tensor(-1.0) at the model dtype -> {wdt}), kv_cache_dtype=auto"],
                       "rule": f"engine constant: Fp8KVCacheMethod loader placeholder == {wdt} -1.0 sentinel in the declared width (no kv scale in the checkpoint; kv_cache_dtype auto -- read by no kernel; F-fp8-PIN1/PIN2)"}
    if ENGINE_ROTARY_RE.search(field):
        from verity_vllm.program import model as analytic
        try:
            C, _prov = analytic.config_of(ck.config)
            inv = analytic._inv_freq(C)                      # raises NotImplementedError for a rope_scaling type without a table
            expected = [int(C["MAX_POS"]), int(analytic.rotary_dim(C))]
        except NotImplementedError as e:
            return base | {"digest": None, "pinned": False, "why": f"rotary table: {e}"[:300]}
        except (KeyError, ValueError, TypeError) as e:
            return base | {"digest": None, "pinned": False, "why": f"rotary table: config.json lacks the analytic inputs ({e})"[:300]}
        if bits != 16 or list(shape) != expected:
            raise ValueError(f"{field}: declared rotary table {shape} x Value<{bits}> != analytic [MAX_POS, ROT] = {expected} x bf16 from config.json")
        keep = {k: C.get(k) for k in ("family", "H", "NH", "KVH", "D", "ROT", "ROPE_THETA", "MAX_POS", "rope_scaling") if k in C}
        # the root's key for this field: the EXACT inputs of the analytic function (config.json values, the declared shape) -- never the
        # fp32 inv_freq bytes numpy computes here (np.power dispatches to SVML on AVX512 hosts: 7 of 32 exponents differ by one ulp
        # between cp-r1 and cp-a100 -> two roots for one record); inv_freq_sha256 stays on the record as the deriving host's value
        inputs_key = analytic_key({"table": "cos_sin", "expected_shape": expected, "C": keep})
        return base | {"digest": None, "pinned": "analytic", "sources": ["config.json:rope_theta,head_dim,max_position_embeddings,rope_scaling"],
                       "analytic": {"table": "cos_sin", "validator": "analytic.check_cos_sin (root_policy ANALYTIC_TABLES['cos_sin'], gate I9)",
                                    "expected_shape": expected, "inputs_sha256": inputs_key,
                                    "inv_freq_sha256": hashlib.sha256(np.ascontiguousarray(inv, dtype=np.float32).tobytes()).hexdigest(),
                                    "inv_freq_note": "the deriving host's numpy-fp32 inv_freq (host-dependent at the last ulp: not part of the root)",
                                    "C": keep},
                       "rule": "engine constant: rotary cos_sin table = analytic function of config.json, served words within the fp32 pipeline's error bound (no byte digest: CUDA libm)"}
    return None


def analytic_key(a: dict[str, Any]) -> str:
    """The root's key for an analytic field: sha256 of the canonical JSON of its EXACT inputs -- table name, the declared shape, the config.json
    values C the analytic function reads (schema v2).  Recomputable from any record's `analytic` entry, v1 included (v1 records carry
    table / expected_shape / C; they keyed the root by the host's inv_freq bytes instead)."""
    return hashlib.sha256(json.dumps({"table": a.get("table"), "expected_shape": a.get("expected_shape"), "C": a.get("C")},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
