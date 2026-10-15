"""The WEIGHTS OF RECORD (R15 manbind MAN-09; rev-cb M-0610, coord M-0614 (1)).

A Commit's `weights` registration hashes whatever model the committer served; comparing it with the live parameters re-registered at
replay (O5) is self-consistency, not authentication: a served model that differs from the checkpoint in one element (rev-cb `flipweight`)
replays bit-exactly against itself.  DECISIONS-2 §4 requires the registered inputs to be authenticated by their CANONICAL reference; for the
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
    weights struct, the checkpoint bytes, config.json) and of nothing on the deriving host.  [R16 pin, schema v2: v1 keyed the analytic
    entry by the numpy-fp32 inv_freq bytes, which differ by host CPU dispatch (AVX512 SVML vs scalar libm: cp-r1 vs cp-a100 gave two
    roots for one Llama-3.2-1B record, exec M-0178) -- the pin VERDICT never depended on it (check_cos_sin is a bounded check).]

Program fields that are NOT checkpoint tensors (the attention `_k_scale/_v_scale/_q_scale/_prob_scale` `Value<32>` engine constants,
and on a quantized checkpoint vLLM's `attn.k_scale/v_scale/q_scale/prob_scale` loader placeholders -- F-fp8-PIN1) are pinned from the
engine's declared configuration by `_engine_constant` (MAN-09c: f32 1.0 / the f32 -1.0 sentinel under kv_cache_dtype auto; the rotary
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
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses import dataclass

from verity_vllm.check import fp8_moe_scales, fp8_ue8m0
from verity_vllm.config import UsageError, option
from verity_vllm.check.tp_shards import replicated_qkv, shard_dim as _shard_dim, shard_dim0 as _shard_dim0

SCHEMA = "verity/commit/weights-of-record/v2"       # v2: host-independent root (analytic entries keyed by their exact inputs)
SCHEMA_V1 = "verity/commit/weights-of-record/v1"    # v1 roots differ per deriving host for models with a rotary table; verdict unaffected
FULL_DIGEST_RE = re.compile(r"[0-9a-f]{128}")       # a Program digest (sha512 hex): compared only whole, a prefix names no Program

#: safetensors dtype -> (numpy view dtype, element bytes)
_ST_DTYPE = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "F64": (np.float64, 8),
             "I64": (np.int64, 8), "I32": (np.int32, 4), "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
             "BOOL": (np.uint8, 1), "F8_E4M3": (np.uint8, 1), "F8_E5M2": (np.uint8, 1)}

#: fused (stacked along dim 0) fields of the served model <- the checkpoint's parts, in stacking order; `{s}` = the field suffix
#: (`weight`, `bias`, `weight_scale_inv`, ...).  The rule applies only when the checkpoint has no tensor of the fused name itself
#: (Phi-3 checkpoints store `qkv_proj` / `gate_up_proj` already fused).
FUSED = {"self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
         "self_attn.wqkv": ("self_attn.wq", "self_attn.wk", "self_attn.wv"), "shared_experts.gate_up_proj": ("shared_experts.gate_proj", "shared_experts.up_proj"),
         "self_attn.fused_qkv_a_proj": ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"), "indexer.wk_weights_proj": ("indexer.wk", "indexer.weights_proj")}
#: fused-MoE experts: `experts.w13_weight[e] = gate|up of expert e`, `experts.w2_weight[e] = down of expert e` (vLLM FusedMoE: RoutedExperts
#: _load_w13 puts shard w1 = gate in the first half, w3 = up in the second; _load_w2 puts w2 = down).  The checkpoint's per-expert part
#: names are HF's gate_proj|up_proj|down_proj (OLMoE, Qwen3-MoE, DeepSeek) or Mixtral's w1|w3|w2 (`ckpt_names=("w1","w2","w3")`):
#: whichever the checkpoint carries for expert 0 is the composition's source set; neither -> refused by name.
EXPERTS = {"w13": ("gate_proj", "up_proj"), "w2": ("down_proj",)}
EXPERTS_MIXTRAL = {"w13": ("w1", "w3"), "w2": ("w2",)}
#: [F-r16-15, rev M-0131 / coord M-0130 (c)] the served stacks live on the FusedMoE layer (`<container>.experts.w13_weight`, vLLM <= 0.11) or
#: on its RoutedExperts submodule (`<container>.experts.routed_experts.w13_weight`, vLLM 0.28 on the pods): the OPTIONAL `routed_experts`
#: segment is the only extra segment accepted; the container (`mlp.experts`, Mixtral's `block_sparse_moe.experts`) is where the
#: checkpoint's `<container>.<e>.<part>.<suffix>` tensors live.  Any other segment: no composition rule (refused by name, never guessed).
EXPERTS_RE = re.compile(r"^(.*\.experts)(?:\.(routed_experts))?\.(w13|w2)_(\w+)$")
#: Program weight fields that are engine constants, not checkpoint tensors (listed by name; not pinned by this module): the attention
#: kv-cache scales (f32 1.0 without kv quantization) and the rotary cos/sin table (a function of config.json: rope_theta, rope_scaling,
#: head_dim, max_position_embeddings -- derivable, v2).  Their effect on the served values is under the Match oracle (form B).
#: [R16 fp8, F-fp8-PIN1] a QUANTIZED checkpoint's attention layers also carry vLLM's loader placeholders `attn.q_scale / k_scale /
#: v_scale / prob_scale` (no underscore): `Fp8KVCacheMethod.create_weights` -> `KVCacheScaleParameter()` = the f32 sentinel -1.0,
#: "overwritten when loading weights" IF the checkpoint ships kv scales (then `compose` finds them 1:1: the `not ck.has(live)` guard),
#: created by `torch.tensor(-1.0)` at the MODEL dtype (bf16 -> `Value<16>` in the Program; F-fp8-PIN2: the measured Qwen3-4B-FP8 B1/B8
#: Programs declare them Value<16>, `_k_scale` Value<32>) -- pinned as -1.0 in the declared width,
#: otherwise left at the sentinel; with the kv cache unquantized (kv_cache_dtype auto) `process_weights_after_loading` enforces
#: `_k_scale = _v_scale = 1.0` and never touches the placeholders, which no kernel reads (Qwen3-4B-FP8 B1: 144 such fields, 723 - 435
#: checkpoint - 144 `_x_scale`).  Pinned EXACTLY as the sentinel under kv_cache_dtype auto (a served value other than -1.0 means the
#: loader wrote it from somewhere that is not the pinned checkpoint); any other kv_cache_dtype: unpinned by name (the scale is then live).
#: [R19, F-r19-int-11] vLLM d9105ea80's `BaseKVCacheMethod.process_weights_after_loading` ends every branch with `del layer.k_scale /
#: v_scale / q_scale / prob_scale` (kv_cache.py:106-109, 209-212; the early-return branch asserts they are already gone): the placeholders
#: exist only between `create_weights` and the end of weight loading, so the served model has NO parameter to compare the record's
#: pinned sentinel against.  #74 (Qwen3-4B-FP8 B8, H100): 579/579 compared equal, 144 placeholders "not found among the served
#: parameters", 3/3 pairs FAIL with every other check green.  A deleted parameter cannot reach a served value; `check_served` therefore
#: reports a placeholder field absent from the served model as `placeholders_deleted` (by name, counted, in the result), not as
#: `missing` -- only for fields matching PLACEHOLDER_RE, only under the record's kv_cache_dtype=auto assumption, and only when the
#: `_x_scale` the kernels do read is itself among the compared fields.  A placeholder that IS served is compared exactly as before.
ENGINE_CONSTANT_RE = re.compile(r"([._]attn\._?(k|v|q|prob)_scale|\.rotary_emb\.cos_sin_cache)$")
PLACEHOLDER_RE = re.compile(r"^(.*\.attn\.)(k|v|q|prob)_scale$")


# ------------------------------------------------------------------------------------------------------------------ checkpoint index

def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class CheckpointIndex:
    """Every tensor of every safetensors shard under `ckpt_dir`: name -> (shard, dtype, shape, byte range); shards memory-mapped on demand.
    With `checkpoints_manifest` (manifests/checkpoints.json) the shard sha256s are verified against the pinned entry by name."""

    def __init__(self, ckpt_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None, revision: str | None = None,
                 verify_shards: bool = True):
        self.dir = ckpt_dir
        self.tensors: dict[str, tuple[str, str, list[int], int, int]] = {}
        self.shards: list[str] = []
        self.config = {}
        cfg = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(cfg):
            self.config = json.load(open(cfg))
        for fn in sorted(os.listdir(ckpt_dir)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(ckpt_dir, fn)
            hdr, base = read_safetensors_header(path)
            self.shards.append(fn)
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                lo, hi = meta["data_offsets"]
                self.tensors[name] = (fn, meta["dtype"], [int(x) for x in meta["shape"]], base + int(lo), base + int(hi))
        if not self.shards:
            raise FileNotFoundError(f"no *.safetensors shard under {ckpt_dir}")
        self.pin: dict[str, Any] = {"manifest": checkpoints_manifest, "entry": None, "shards": {}}
        if verify_shards:
            self._verify_shards(checkpoints_manifest, model, revision)

    def _verify_shards(self, manifest: str | None, model: str | None, revision: str | None) -> None:
        entry = None
        if manifest and os.path.exists(manifest):
            ck = json.load(open(manifest))
            for c in ck.get("checkpoints", []):
                lp = str(c.get("local_path") or "")
                if (lp and os.path.realpath(lp) == os.path.realpath(self.dir)) or \
                        (model and c.get("repo") == model and (not revision or str(c.get("revision", "")).startswith(revision[:10]))):
                    entry = c
                    break
        self.pin["entry"] = None if entry is None else {"repo": entry.get("repo"), "revision": entry.get("revision")}
        for fn in self.shards:
            sha = sha256_file(os.path.join(self.dir, fn))
            pinned = ((entry or {}).get("files") or {}).get(fn, {}).get("sha256") if entry else None
            self.pin["shards"][fn] = {"sha256": sha, "pinned_sha256": pinned,
                                      "pinned": (sha == pinned) if pinned else None}

    def has(self, name: str) -> bool:
        return name in self.tensors

    def array(self, name: str) -> np.ndarray:
        fn, dt, shape, lo, hi = self.tensors[name]
        npdt, isz = _ST_DTYPE[dt]
        with open(os.path.join(self.dir, fn), "rb") as f:     # a positioned read: a whole-shard memmap keeps every page it touched resident
            raw = f.read(hi - lo) if f.seek(lo) == lo else b""
        n = 1
        for s in shape:
            n *= s
        if len(raw) != n * isz:
            raise ValueError(f"{name}: {len(raw)} bytes for shape {shape} dtype {dt}")
        return np.frombuffer(raw, dtype=npdt).reshape(shape) if shape else np.frombuffer(raw, dtype=npdt).reshape(())

    def dtype(self, name: str) -> str:
        return self.tensors[name][1]


# -------------------------------------------------------------------------------------------------------------------- composition

def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 (the implementation's `.to(torch.bfloat16)`), as u16 words; NaN keeps a quiet payload."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = (u & 0x7F800000) == 0x7F800000
    nan &= (u & 0x007FFFFF) != 0
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    r = np.where(nan, ((u >> 16) | 0x0040).astype(np.uint16), r)
    return r


def to_served(arr: np.ndarray, src_dtype: str, bits: int) -> tuple[np.ndarray, str | None]:
    """The checkpoint tensor in the served element width; (array of raw words, cast note | None)."""
    if bits == 16:
        if src_dtype == "BF16":
            return arr.view(np.uint16), None
        if src_dtype in ("F16", "F32", "F64"):
            return _f32_to_bf16_bits(arr.astype(np.float32)), f"{src_dtype}->bf16 (RNE)"
    if bits == 32:
        if src_dtype == "F32":
            return arr.view(np.uint32), None
        if src_dtype in ("BF16",):
            return (arr.view(np.uint16).astype(np.uint32) << 16), "bf16->f32 (exact)"
        if src_dtype in ("F16", "F64"):
            return arr.astype(np.float32).view(np.uint32), f"{src_dtype}->f32"
    if bits == 8 and src_dtype in ("F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL"):
        return arr.view(np.uint8), None
    if bits == 64 and src_dtype in ("I64", "F64"):
        return arr.view(np.uint64), None
    raise ValueError(f"no served-width rule for checkpoint dtype {src_dtype} -> Value<{bits}>")


def strip_prefix(field: str, prefix: str) -> str:
    return field[len(prefix):] if prefix and field.startswith(prefix) else field


def compose(name: str, shape: list[int], bits: int, ck: CheckpointIndex, *, world: int = 1, rank: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """The served model's tensor `name` (wrapper prefix already stripped) of `shape` / `bits`, composed from the checkpoint.
    Returns (raw words row-major, note {sources, rule, cast, shard})."""
    note: dict[str, Any] = {"sources": [], "rule": None, "cast": None, "shard": None}
    tied = bool(ck.config.get("tie_word_embeddings", False))

    def served(src: str) -> np.ndarray:
        a, cast = to_served(ck.array(src), ck.dtype(src), bits)
        if cast:
            note["cast"] = cast
        note["sources"].append(src)
        return a

    parts: list[np.ndarray] = []
    if ck.has(name):
        note["rule"] = "1:1"
        parts = [served(name)]
        if world > 1:
            # a checkpoint that stores the fused tensor itself (Phi-3): the rank shard is PER PART, so split it by the config's geometry
            fused = next((k for k in FUSED if re.search(r"\." + re.escape(k) + r"\.\w+$", name)), None)
            if fused and (fused.endswith("qkv_proj") or fused == "self_attn.wqkv"):
                c = ck.config
                hq, hkv = int(c.get("num_attention_heads", 0)), int(c.get("num_key_value_heads") or c.get("num_attention_heads", 0))
                d = int(c.get("head_dim") or (int(c.get("hidden_size", 0)) // max(hq, 1)))
                if hq and d and parts[0].shape[0] == (hq + 2 * hkv) * d:
                    a = parts[0]
                    parts = [a[:hq * d], a[hq * d:(hq + hkv) * d], a[(hq + hkv) * d:]]
                    note["rule"] = "1:1 fused tensor split q|k|v by config geometry"
            elif fused:                                                                  # gate_up: two equal halves
                a = parts[0]
                parts = [a[: a.shape[0] // 2], a[a.shape[0] // 2:]]
                note["rule"] = "1:1 fused tensor split gate|up"
    elif name == "lm_head.weight" and tied and ck.has("model.embed_tokens.weight"):
        note["rule"] = "tied lm_head <- model.embed_tokens.weight"
        parts = [served("model.embed_tokens.weight")]
    elif name == "normalizer" or name.endswith(".normalizer"):
        # Gemma-2's embedding normalizer (dense M-0619 preflight on the real Gemma-2-2B Program: the ONE unresolved field of 316): a
        # non-persistent 0-d buffer `torch.tensor(hidden_size ** 0.5)` created at the served dtype in vLLM's Gemma2Model.__init__ -- no
        # checkpoint tensor, a function of config.json's hidden_size alone.  The frozen root policy already classifies it with the
        # rotary table (root_policy.ANALYTIC_TABLES["normalizer"]) and pins its value EXACTLY: the Python float sqrt(H) rounded once
        # (nearest-even) to bf16 (analytic.normalizer_word).  Composed here from config.json, so the served buffer is compared like a
        # weight, not left unpinned; any other declared width / shape has no rule -> unresolved (fail closed).
        if bits != 16 or list(shape) not in ([], [1]):
            raise ValueError(f"{name}: the config-derived normalizer is one bf16 word; the Program declares shape {list(shape)} x Value<{bits}>")
        hidden = int(ck.config.get("hidden_size") or 0)
        if hidden <= 0:
            raise KeyError(f"{name}: config.json carries no hidden_size to derive the normalizer from")
        from verity_vllm.program.model import normalizer_word
        note["rule"] = f"config-derived: bf16(sqrt(hidden_size={hidden})) rounded once RNE (root_policy analytic-table 'normalizer', analytic.normalizer_word)"
        note["sources"].append("config.json:hidden_size")
        out = np.array(normalizer_word(hidden), dtype=np.uint16).reshape(shape)
        return np.ascontiguousarray(out), note                  # replicated on every TP rank: no shard rule applies
    else:
        m = EXPERTS_RE.match(name)
        if m:
            container, routed, which, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
            e_n = int(shape[0]) if shape else 0
            # the checkpoint's per-expert part names, decided by expert 0 (HF gate_proj|up_proj|down_proj, else Mixtral w1|w3|w2)
            parts_of = next((tbl for tbl in (EXPERTS, EXPERTS_MIXTRAL) if all(ck.has(f"{container}.0.{p}.{suffix}") for p in tbl[which])), None)
            if parts_of is None and e_n:
                raise KeyError(f"{name}: no per-expert checkpoint tensors {container}.0.{{{'|'.join(EXPERTS[which] + EXPERTS_MIXTRAL[which])}}}.{suffix} "
                               f"(fused experts {which}: the composition needs expert 0's parts under the container the served name declares)")
            parts_of = parts_of or EXPERTS
            note["rule"] = (f"fused experts {which}{' (vLLM RoutedExperts submodule)' if routed else ''} <- " + "|".join(parts_of[which])
                            + f" per expert ({container}.<e>.<part>.{suffix})")
            stack = []
            for e in range(e_n):
                srcs = [f"{container}.{e}.{p}.{suffix}" for p in parts_of[which]]
                if not all(ck.has(s) for s in srcs):
                    raise KeyError(f"{name}: expert {e} sources missing: {[s for s in srcs if not ck.has(s)]}")
                ps = fp8_moe_scales.on_rank_grid([served(s) for s in srcs], srcs, which, ck, world, note)
                if world > 1:
                    ps = [_shard_dim0(p, world, rank) for p in ps] if which == "w13" else [_shard_dim(p, 1, world, rank) for p in ps]
                    note["shard"] = f"rank {rank}/{world} ({'per-part dim 0' if which == 'w13' else 'dim 1'})"
                stack.append(np.concatenate(ps, axis=0) if len(ps) > 1 else ps[0])
            out = np.stack(stack, axis=0) if stack else np.zeros(shape, dtype=np.uint16)
            if list(out.shape) != list(shape):
                raise ValueError(f"{name}: composed shape {list(out.shape)} != Program shape {shape}")
            return np.ascontiguousarray(out), note
        for fused, srcs in FUSED.items():
            m2 = re.match(r"^(.*)\." + re.escape(fused) + r"\.(\w+)$", name)
            if m2:
                base, suffix = m2.group(1), m2.group(2)
                names = [f"{base}.{s}.{suffix}" for s in srcs]
                if all(ck.has(s) for s in names):
                    note["rule"] = f"fused {fused} <- " + "|".join(srcs) + " (dim 0)"
                    parts = [served(s) for s in names]
                break
    if not parts:
        raise KeyError(f"{name}: no checkpoint tensor and no composition rule")
    kv = replicated_qkv(name, shape, parts, ck.config, world, rank)
    if kv:
        parts, note["shard"] = kv
    elif world > 1:
        full = [int(sum(p.shape[0] for p in parts))] + list(parts[0].shape[1:])
        if list(shape) == full:
            note["shard"] = f"rank {rank}/{world}: replicated"
        elif len(shape) >= 1 and shape[0] * world == full[0] and list(shape[1:]) == full[1:]:
            parts = [_shard_dim0(p, world, rank) for p in parts]
            note["shard"] = f"rank {rank}/{world}: per-part dim-0 slice"
        elif len(shape) >= 2 and shape[1] * world == full[1] and shape[0] == full[0]:
            parts = [_shard_dim(p, 1, world, rank) for p in parts]
            note["shard"] = f"rank {rank}/{world}: dim-1 slice"
        else:
            raise ValueError(f"{name}: Program shape {shape} is not a rank {rank}/{world} shard of the composed {full}")
    out = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    if list(out.shape) != list(shape):
        raise ValueError(f"{name}: composed shape {list(out.shape)} != Program shape {shape}")
    return np.ascontiguousarray(out), note


def program_weight_fields(inst: dict[str, Any]) -> list[tuple[str, list[int], int]]:
    """(field, shape, value bits) for every field of the Program's `weights` struct parameter, in declaration order."""
    from verity_vllm.check.replay.index import parse_struct
    wparam = next((t for n, t in inst.get("params", []) if n == "weights"), None)
    if not wparam or not str(wparam).startswith("{"):
        return []
    return [(name, shape, bits) for name, _count, shape, bits in parse_struct(str(wparam))]


ENGINE_KV_SCALE_RE = re.compile(r"[._]attn\._(k|v|q|prob)_scale$")
ENGINE_KV_PLACEHOLDER_RE = re.compile(r"\.attn\.(k|v|q|prob)_scale$")       # [F-fp8-PIN1] Fp8KVCacheMethod loader placeholders
ENGINE_ROTARY_RE = re.compile(r"\.rotary_emb\.cos_sin_cache$")


def kv_scale_remap_sources(served_name: str) -> list[str]:
    """[rev M-0489 / M-0516 F-r16-21 residual] the checkpoint names vLLM's `weight_utils.maybe_remap_kv_scale_name` renames onto a served
    `<layer>.attn.<x>_scale` (x in k, v, q, prob) at load: kv_cache_scheme checkpoints (`<layer>.<x>_scale`, `<layer>.<x>_proj.<x>_scale`),
    modelopt (`<layer>.<x>_proj.output_scale`), the legacy fused `<layer>.kv_scale` (k and v).  `<layer>` = the served name up to `.attn.`.
    Ordered; the caller pins 1:1 from the ONE source present and refuses two.  `attn._<x>_scale` (the MAN-09c engine constant) -> []."""
    m = re.search(r"^(.*)\.attn\.(k|v|q|prob)_scale$", served_name)
    if not m:
        return []
    layer, x = m.group(1), m.group(2)
    out = [f"{layer}.{x}_scale", f"{layer}.{x}_proj.{x}_scale", f"{layer}.{x}_proj.output_scale"]
    if x in ("k", "v"):
        out.append(f"{layer}.kv_scale")
    return out


def _engine_constant(field: str, live: str, shape: list[int], bits: int, ck: CheckpointIndex, kv_cache_dtype: str) -> dict[str, Any] | None:
    """[MAN-09c, rev-cb M-0655 W3] the engine constants the Program declares as weights but no checkpoint carries, pinned from
    config.json / the engine's declared configuration instead of left unpinned:
      * attention kv-cache scales `attn._{k,v,q,prob}_scale`: without kv-cache quantization (kv_cache_dtype auto) vLLM creates them as
        f32 1.0 -- pinned EXACTLY (digest of the f32 word); any other kv_cache_dtype has its scales in the checkpoint (then `compose`
        finds them) or is left unpinned BY NAME with the dtype in the why (never assumed).
      * the rotary table `rotary_emb.cos_sin_cache` [MAX_POS, ROT] bf16: the frozen root policy's analytic table (root_policy
        ANALYTIC_TABLES['cos_sin'] = analytic.check_cos_sin, gate I9) -- the served words must round from values within the fp32
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
        # [rev M-0489 (2) / M-0516 F-r16-21 residual: the remap-aware compose] BEFORE the placeholder rule, look the checkpoint up under
        # vLLM's REMAP SOURCES for this layer's scale (weight_utils.maybe_remap_kv_scale_name renames them to attn.<x>_scale at load): a
        # shipped scale is a checkpoint tensor, written by the loader and read by the kernel -- pinned 1:1 from its source as one f32 word
        # under ANY kv_cache_dtype; two sources for one scale, or a source that is not one word, -> unresolved (fail closed).  Only a
        # checkpoint with NO source for this scale reaches the placeholder rule below (the sentinel, which such a checkpoint would fail by
        # name anyway: rev M-0516).  The exact-name `ck.has(served)` case never arrives here (compose pins it first).
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
        # [R16 fp8 F-fp8-PIN1 / F-fp8-PIN2] vLLM Fp8KVCacheMethod.create_weights: `KVCacheScaleParameter()` = `torch.tensor(-1.0)` with NO
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
    derived from the same fields (exec M-0178: two v1 roots of one Llama-3.2-1B, cp-r1 / cp-a100, re-root to one)."""
    return root_of_fields(record.get("fields") or [])


def diff_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare two records FIELD BY FIELD (the comparison report uses for same-model rows across pods under schema v1, COORD ruling 25):
    fields only on one side; fields whose digest / shape / bits / analytic KEY differ (the weights of record differ); fields that differ
    only in host-side information (analytic.inv_freq_sha256, the host note, sources/rule text) -- named separately, they do not make
    the weights differ.  `same_weights_of_record` = no one-sided field, no differing digest/shape/bits/analytic key; then reroot(a) ==
    reroot(b) whatever the two schemas were."""
    fa = {f["field"]: f for f in (a.get("fields") or [])}
    fb = {f["field"]: f for f in (b.get("fields") or [])}
    only_a = sorted(set(fa) - set(fb)); only_b = sorted(set(fb) - set(fa))
    differ: list[dict[str, Any]] = []; host_only: list[dict[str, Any]] = []
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


def weights_of_record(program: dict[str, Any], ck: CheckpointIndex, *, prefix: str | None = None, kv_cache_dtype: str = "auto") -> dict[str, Any]:
    """The record: per Program weight field the digest of the checkpoint-composed served tensor; root over the ordered list."""
    t0 = time.perf_counter()
    inst, res = program["instances"], program.get("result") or {}
    tp = res.get("tp") if isinstance(res.get("tp"), dict) else {}
    world, rank = int(tp.get("world") or 1), int(tp.get("rank") or 0)
    fields = program_weight_fields(inst)
    if prefix is None:
        from verity_vllm.query.manifest import format as MF
        prefix = MF.wrapper_prefix_of(inst.get("rows", []), res.get("attention_impls_observed") or [])
        if not prefix:
            # no attention row to determine the wrapper from (e.g. a weights-only check): the prefix under which the most fields are
            # checkpoint tensors 1:1 -- determined from the checkpoint, not assumed
            def hits(p: str) -> int:
                return sum(1 for f, _s, _b in fields if ck.has(strip_prefix(f, p)))
            cands = sorted({f.split(".")[0] + "." for f, _s, _b in fields if "." in f} | {""})
            prefix = max(cands, key=lambda p: (hits(p), -len(p)))
    out_fields: list[dict[str, Any]] = []
    unpinned: list[str] = []
    unresolved: list[dict[str, str]] = []
    n_bytes = 0
    analytic_fields: list[str] = []
    dims = {f: (s, b) for f, s, b in fields}       # DeepGEMM linears and MLA's absorbed kv_b: vLLM's load-time ue8m0 rewrite (fp8_ue8m0)
    rw = fp8_ue8m0.Rewrites(list(dims), fp8_ue8m0.build_roles(program.get("dir")), lambda f: compose(strip_prefix(f, prefix), *dims[f], ck, world=world, rank=rank), ck.config)
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
            arr, note = rw.compose(field, shape) if rw.covers(field) else compose(live, shape, bits, ck, world=world, rank=rank)
        except (KeyError, ValueError) as e:
            unresolved.append({"field": field, "why": str(e)[:300]})
            out_fields.append({"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": None, "pinned": False, "why": str(e)[:300]})
            continue
        b = arr.tobytes()
        n_bytes += len(b)
        out_fields.append({"field": field, "served_name": live, "shape": shape, "bits": bits, "digest": hashlib.sha256(b).hexdigest(),
                           "pinned": True, **note})
    root = root_of_fields(out_fields)
    n_pinned = sum(1 for f in out_fields if f.get("digest") or f.get("pinned") == "analytic")
    shards_ok = all(v.get("pinned") is True for v in ck.pin["shards"].values()) if ck.pin["shards"] else None
    program_digest = str((res.get("program_digest") or (inst.get("digest") or "")))
    return {"schema": SCHEMA, "program_digest": program_digest, "program_dir": program.get("dir"), "wrapper_prefix": prefix,
            "tp": {"world": world, "rank": rank}, "checkpoint": {"dir": ck.dir, "config_tie_word_embeddings": bool(ck.config.get("tie_word_embeddings", False)),
                                                                   **ck.pin, "all_shards_pinned": shards_ok},
            "root_of_record": root, "fields": out_fields, "n_fields": len(fields), "n_pinned": n_pinned, "bytes": n_bytes,
            "n_analytic": len(analytic_fields), "analytic_fields": analytic_fields,      # [MAN-09c] pinned by the frozen analytic table (bounded), no byte digest
            "engine_assumptions": {"kv_cache_dtype": str(kv_cache_dtype or "auto")},
            "unpinned": unpinned, "unresolved": unresolved, "complete": not unresolved,
            "of_record": (not unresolved) and shards_ok is True,          # [W1] composed from the PINNED shards, every field resolved
            "why": None if not unresolved else f"{len(unresolved)} Program weight field(s) have no checkpoint tensor of record: " + ", ".join(u["field"] for u in unresolved[:8]),
            "wall_s": round(time.perf_counter() - t0, 3)}


# ---------------------------------------------------------------------------------------------------------------------- live side

def live_param_digests(model, *, include_buffers: bool = True, raw_for: "re.Pattern[str] | None" = ENGINE_ROTARY_RE) -> dict[str, Any]:
    """sha256 of every live parameter (and buffer) of the served model, by `named_parameters()` name: the SAME tensors the committer's
    weights root covers (register_weights), hashed as raw contiguous bytes (bf16 words row-major).  Tensors whose name matches `raw_for`
    (default: the rotary cos_sin table) also keep their raw words under `raw[name]` (numpy uint16, in memory only -- not for the row
    record) so `check` can run the analytic bound on them [MAN-09c]."""
    import torch
    t0 = time.perf_counter()
    out: dict[str, dict[str, Any]] = {}
    raws: dict[str, np.ndarray] = {}
    seen: dict[int, str] = {}
    items = list(model.named_parameters())
    if include_buffers:
        items += list(model.named_buffers())
    for name, p in items:
        if not isinstance(p, torch.Tensor) or p.numel() == 0:
            continue
        if id(p) in seen:
            out[name] = dict(out[seen[id(p)]], alias_of=seen[id(p)])
            if seen[id(p)] in raws:
                raws[name] = raws[seen[id(p)]]
            continue
        seen[id(p)] = name
        flat = p.detach().contiguous().reshape(-1)
        raw = (flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat).cpu().numpy()
        out[name] = {"digest": hashlib.sha256(memoryview(raw).cast("B")).hexdigest(), "dtype": str(p.dtype).replace("torch.", ""),
                     "shape": list(p.shape), "bytes": int(raw.nbytes)}
        if raw_for is not None and raw_for.search(name) and p.dtype == torch.bfloat16:
            raws[name] = np.ascontiguousarray(raw).view(np.uint16).reshape(list(p.shape))
    return {"params": out, "raw": raws, "n": len(out), "wall_s": round(time.perf_counter() - t0, 3)}


def _check_analytic(f: dict[str, Any], raw: np.ndarray | None) -> dict[str, Any]:
    """[MAN-09c] the served rotary table against the frozen analytic table's bound (analytic.check_cos_sin)."""
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

    [rev-cb M-0655 W1] the record is OF RECORD only when every shard it was composed from carries the sha256 `manifests/checkpoints.json`
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
            # [rev-cb M-0733] a record stamped with the WHOLE of-record set is accepted iff that set equals the run's: a record of another
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
    # [MAN-09c] the record's engine assumption must be the engine's: a record that pinned the kv scales as 1.0 (kv_cache_dtype auto)
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
    # [F-r19-int-11] a loader placeholder (`attn.k_scale`, no underscore) the engine deleted after loading is absent by construction, not
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
                                          "the attn._x_scale the kernels read are compared (F-r19-int-11)") if placeholders_deleted else None),
            "analytic": analytic_checked, "engine_assumptions": record.get("engine_assumptions"),
            "unpinned_note": (("engine constants left unpinned by name are not authenticated by this pin (form (B) only)") if unpinned else
                              "every declared weight field is pinned: checkpoint bytes by digest, kv scales as the engine constant, the rotary table by the frozen analytic bound"),
            "unresolved": list(record.get("unresolved") or []), "shards": ck.get("shards"),
            "all_shards_pinned": ck.get("all_shards_pinned"), "shards_ok": bool(shards_ok),
            "method": "sha256(served parameter bytes) == sha256(checkpoint-composed tensor of record) per Program weight field, shards pinned by "
                      "manifests/checkpoints.json sha256, record bound to the Program of record (MAN-09 + W1/W2); engine constants: kv scales "
                      "== f32 1.0 under kv_cache_dtype auto, rotary table within analytic.check_cos_sin's bound (MAN-09c)",
            "why": why}


# ---------------------------------------------------------------------------------------------------------------------------- CLI

def _default_manifest() -> str | None:
    from verity_vllm.config import CHECKPOINTS
    for p in (CHECKPOINTS, Path.cwd() / "manifests" / "checkpoints.json"):
        if p.exists():
            return os.path.abspath(p)
    return None


def build_record(program_dir: str, checkpoint_dir: str, *, checkpoints_manifest: str | None = None, model: str | None = None,
                 revision: str | None = None, verify_shards: bool = True, kv_cache_dtype: str = "auto") -> dict[str, Any]:
    from verity_vllm.query.manifest import format as MF
    prog = MF.load_program(program_dir)
    ck = CheckpointIndex(checkpoint_dir, checkpoints_manifest=checkpoints_manifest or _default_manifest(), model=model, revision=revision,
                         verify_shards=verify_shards)
    return weights_of_record(prog, ck, kv_cache_dtype=kv_cache_dtype)


def weights_struct(program: dict[str, Any]) -> list[list[Any]]:
    """The weights STRUCT of a Program -- [field, shape, bits] in declaration order -- the model-level object the record is a function of
    (no checkpoint read).  Two Programs of one model at different sequence shapes have the same struct."""
    return [[f, list(shape), int(bits)] for f, shape, bits in program_weight_fields(program["instances"])]


def component_program_dirs(programs_root: str) -> list[str]:
    """Every component Program dir under a row dir: build_request_LP*_T* + the default-shape build_request (with an instances.json.gz)."""
    root = Path(programs_root)
    out = sorted(str(p) for p in root.glob("build_request_LP*_T*") if (p / "instances.json.gz").exists())
    if (root / "build_request" / "instances.json.gz").exists():
        out.append(str(root / "build_request"))
    return out


def stamp_of_record_set(record: dict[str, Any], programs_root: str, *, manifest: str | None = None) -> dict[str, Any]:
    """[rev-cb M-0733 hardening of hopper M-0723] Stamp the WHOLE of-record set on the record instead of one arbitrary component's digest:
    every component Program under `programs_root` (build_request_LP*_T* + build_request) + the workload digest (build_workload/
    workload_program.json) + the manifest's program / component digests.  Each component's weights struct must EQUAL the record's
    (field list, shapes, bits): a component whose struct differs is named and the record is not of record (a record derived from one
    shape never silently stands for a Program of another struct).  `check()` then requires the run's of-record set to equal the stamp."""
    from verity_vllm.query.manifest import format as MF
    my_struct = [[f["field"], list(f["shape"]), int(f["bits"])] for f in record.get("fields", [])]
    # the manifest's digests first: they define which Program dirs under the root are OF RECORD (a default-shape build_request that is
    # not a workload component is struct-checked but its digest does not join the set -- the run's set never carries it)
    mpath = manifest or str(Path(programs_root) / "manifest.json")
    manifest_digests: list[str] = []
    why: list[str] = []
    if os.path.exists(mpath):
        try:
            m = json.load(open(mpath))
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
            diff = sorted((mine ^ theirs))[:4] or sorted(f for (f, s, b) in st if [f, s, b] not in my_struct)[:4]
            why.append(f"{os.path.basename(d)} ({pd[:16]}): weights struct differs from the record's ({len(st)} vs {len(my_struct)} fields; e.g. {diff})")
        elif of_rec:
            digests.add(pd)
    wl = Path(programs_root) / "build_workload" / "workload_program.json"
    workload_digest = None
    if wl.exists():
        try:
            workload_digest = str(json.load(open(wl)).get("workload_digest") or "") or None
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
                               "rule": "every component Program's weights struct == the record's; the run's of-record set must equal this set (rev-cb M-0733)"}
    if why:
        record["of_record"] = False
        record["of_record_why"] = "; ".join(why)[:600]
    return record


def derivation_provenance(program_dir: str, checkpoint_dir: str, checkpoints_manifest: str | None, argv: list[str] | None, *,
                          supersedes: str | None = None) -> dict[str, Any]:
    """[COORD R16 M-0038 (2): record repairs are LOGGED DERIVATIONS] the provenance line of a record written by this CLI: tool, the code that
    ran it (the repo's HEAD when this file sits in a git tree, else null), the SOURCE FILES the derivation read (the Program's instances.json.gz
    + result.json, the checkpoint dir + checkpoints.json) and the time; `supersedes` names the record this one replaces (row_pod moves a
    stale / pre-MAN-09c record aside before re-deriving).  Informational: never read by check() -- the digests are the evidence."""
    import subprocess
    import time as _t

    code = None
    code_source = None
    try:
        # the release tree on a pod has no .git: ship.sh's EXPORT.json names the shipped commit (experiment.code_version reads it first,
        # then git); the int-1 wave-1 records carried code null for that reason (exec r16_0413 build_request/weights_of_record.json)
        from verity_vllm.config import ROOT
        from verity_vllm.pipeline.experiment import code_version
        cv = code_version(ROOT)
        code, code_source = cv.get("commit") or None, cv.get("source")
        if code and cv.get("dirty_patch_sha256"):
            code_source = f"{code_source} (dirty patch {str(cv['dirty_patch_sha256'])[:12]})"
    except Exception:  # noqa: BLE001
        code = None
    if code is None:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            code = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
            code_source = "git" if code else None
        except Exception:  # noqa: BLE001
            code = None
    return {"tool": "verity_vllm.check.weights_of_record", "code": code, "code_source": code_source, "at": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
            "sources": {"program_dir": os.path.abspath(program_dir), "program_files": ["instances.json.gz", "result.json"],
                        "checkpoint_dir": os.path.abspath(checkpoint_dir), "checkpoints_manifest": os.path.abspath(checkpoints_manifest) if checkpoints_manifest else "repo default"},
            "argv": list(argv) if argv is not None else sys.argv[1:], "supersedes": os.path.abspath(supersedes) if supersedes else None}


@dataclass(frozen=True, kw_only=True)
class Options:
    """weights of record: per-field digests of the checkpoint-composed served weights (MAN-09)"""
    program_dir: str | None = option("--program-dir", default=None, help="the Program of record (instances.json.gz + result.json)")
    programs_root: str | None = option("--programs-root", default=None, help="[M-0733] a row dir: derive from its first component Program and stamp EVERY component's " "digest + the workload digest as the of-record set (each component's weights struct must equal the record's)")
    manifest: str | None = option("--manifest", default=None, help="the row's required manifest (default <programs-root>/manifest.json): its digests join the stamp")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="the pinned checkpoint snapshot (safetensors shards + config.json); required to derive")
    diff: list[str] | None = option("--diff", nargs=2, metavar=("A.json", "B.json"), default=None, help="READER (no derivation, nothing written): compare two records field by field and re-root both under the current schema; " "exit 0 when they pin the same weights of record (COORD R16 ruling 25: same-model rows across pods under schema v1)")
    checkpoints: str | None = option("--checkpoints", default=None, help="manifests/checkpoints.json (default: the repo's)")
    model: str | None = option("--model", default=None, help="HF repo id, to select the checkpoints.json entry when --checkpoint-dir is not its local_path")
    revision: str | None = option("--revision", default=None)
    no_verify_shards: bool = option("--no-verify-shards", default=False, action="store_true", help="skip the shard sha256 (the record then says pinned: null)")
    kv_cache_dtype: str = option("--kv-cache-dtype", default="auto", help="[MAN-09c] the engine's kv_cache_dtype the record assumes (auto: kv scales pinned as f32 1.0)")
    out: str | None = option("--out", default=None, help="write the record here (default: <program-dir>/weights_of_record.json)")
    supersedes: str | None = option("--supersedes", default=None, help="[COORD R16 M-0038 (2)] the record this derivation replaces (moved aside by the caller), named in `derived`")


def main(a: Options) -> int:
    if a.diff:
        A, B = (json.load(open(x)) for x in a.diff)
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
    if not a.program_dir and not a.programs_root:
        raise UsageError("one of --program-dir / --programs-root is required")
    if not a.checkpoint_dir:
        raise UsageError("--checkpoint-dir is required to derive a record")
    pdir = a.program_dir
    if not pdir:
        cands = component_program_dirs(a.programs_root)
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
