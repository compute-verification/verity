"""A served weight's words, composed from the checkpoint's tensors.

`compose` gives the served model's tensor of a name, shape and element width from the checkpoint: 1:1 where the checkpoint
holds the name, by the stacking rule where the served model fuses parts (`FUSED`, `FUSED_W13`, and the fused-MoE experts,
`EXPERTS`, `EXPERTS_MIXTRAL`, `EXPERTS_RE`), a tied `lm_head.weight` from `embed_tokens.weight`, and at a rank of a TP world
the rank's shard.  `to_served` casts a checkpoint tensor to the served element width as the implementation rounds.
`weights_of_record` digests what `compose` returns, and its docstring says what the record is.  CPU only, no engine.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from verity_vllm.check import fp8_moe_scales, fp8_ue8m0
from verity_vllm.check.checkpoint_index import CheckpointIndex
from verity_vllm.check.tp_shards import kv_heads_replicated, replicated_qkv, rows_of
from verity_vllm.check.tp_shards import shard_dim as _shard_dim
from verity_vllm.check.tp_shards import shard_dim0 as _shard_dim0

#: fused (stacked along dim 0) fields of the served model <- the checkpoint's parts, in stacking order; `{s}` = the field suffix
#: (`weight`, `bias`, `weight_scale_inv`, ...).  The rule applies only when the checkpoint has no tensor of the fused name itself
#: (Phi-3 checkpoints store `qkv_proj` / `gate_up_proj` already fused).  DeepSeek-V4's (`stacked_params_mapping`) are `attn.fused_wqa_wkv`,
#: `compressor.fused_wkv_wgate`, and its shared experts' gate|up from their w1|w3 (`FUSED_W13`).
FUSED = {"self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
         "self_attn.wqkv": ("self_attn.wq", "self_attn.wk", "self_attn.wv"), "shared_experts.gate_up_proj": ("shared_experts.gate_proj", "shared_experts.up_proj"),
         "self_attn.fused_qkv_a_proj": ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"), "indexer.wk_weights_proj": ("indexer.wk", "indexer.weights_proj"),
         "attn.fused_wqa_wkv": ("attn.wq_a", "attn.wkv"), "compressor.fused_wkv_wgate": ("compressor.wkv", "compressor.wgate")}
FUSED_W13 = {"shared_experts.gate_up_proj": ("shared_experts.w1", "shared_experts.w3")}
#: fused-MoE experts: `experts.w13_weight[e] = gate|up of expert e`, `experts.w2_weight[e] = down of expert e` (vLLM FusedMoE: RoutedExperts
#: _load_w13 puts shard w1 = gate in the first half, w3 = up in the second; _load_w2 puts w2 = down).  The checkpoint's per-expert part
#: names are HF's gate_proj|up_proj|down_proj (OLMoE, Qwen3-MoE, DeepSeek) or Mixtral's w1|w3|w2 (`ckpt_names=("w1","w2","w3")`):
#: whichever the checkpoint carries for expert 0 is the composition's source set; neither -> refused by name.
EXPERTS = {"w13": ("gate_proj", "up_proj"), "w2": ("down_proj",)}
EXPERTS_MIXTRAL = {"w13": ("w1", "w3"), "w2": ("w2",)}
#: the served stacks live on the FusedMoE layer (`<container>.experts.w13_weight`, vLLM <= 0.11) or
#: on its RoutedExperts submodule (`<container>.experts.routed_experts.w13_weight`, vLLM 0.28 on the pods): the OPTIONAL `routed_experts`
#: segment is the only extra segment accepted; the container (`mlp.experts`, Mixtral's `block_sparse_moe.experts`) is where the
#: checkpoint's `<container>.<e>.<part>.<suffix>` tensors live.  Any other segment: no composition rule (refused by name, never guessed).
EXPERTS_RE = re.compile(r"^(.*\.experts)(?:\.(routed_experts))?\.(w13|w2)_(\w+)$")


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
    if bits == 32 and src_dtype == "I64":
        if arr.size and (arr.min() < -(1 << 31) or arr.max() >= 1 << 31):
            raise ValueError("an I64 checkpoint tensor served as i32 holds a value outside int32")
        return arr.astype(np.int32).view(np.uint32), "i64->i32 (every value in range)"
    if bits == 8 and src_dtype in ("F8_E4M3", "F8_E5M2", "F8_E8M0", "I8", "U8", "BOOL"):
        return arr.view(np.uint8), None
    if bits == 64 and src_dtype in ("I64", "F64"):
        return arr.view(np.uint64), None
    raise ValueError(f"no served-width rule for checkpoint dtype {src_dtype} -> Value<{bits}>")


def _rank_rows(srcs: list[str], served, ck: CheckpointIndex, world: int, rank: int) -> list[np.ndarray]:
    """The rank's dim-0 slice of each whole checkpoint tensor `srcs`, read as its row range.  A source whose rows do not split by the
    world is refused after the whole read of every source, the order in which composing from whole tensors meets the two faults."""
    rows, err = [], None
    for s in srcs:
        try:
            rows.append(rows_of(ck.tensors[s][2][0], world, rank))
        except ValueError as e:
            err = err or e
    if err is not None:
        for s in srcs:
            served(s)
        raise err
    return [served(s, rows=r) for s, r in zip(srcs, rows)]


def _compose_experts(name: str, m: re.Match, shape: list[int], ck: CheckpointIndex, world: int, rank: int, note: dict[str, Any],
                     served) -> tuple[np.ndarray, dict[str, Any]]:
    """The fused-MoE stack `name` (EXPERTS_RE match `m`): per expert its checkpoint parts, on the rank's scale grid, the rank's slice."""
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
    # w13's rank share is a row range of each part (block scales excepted: they move onto the rank's grid first); w2's is a column
    # range, which touches every page of the tensor, so it is read whole
    by_rows = world > 1 and which == "w13" and suffix != "weight_scale_inv"
    stack = []
    for e in range(e_n):
        srcs = [f"{container}.{e}.{p}.{suffix}" for p in parts_of[which]]
        if not all(ck.has(s) for s in srcs):
            raise KeyError(f"{name}: expert {e} sources missing: {[s for s in srcs if not ck.has(s)]}")
        if by_rows and all(ck.tensors[s][2] for s in srcs):
            ps = _rank_rows(srcs, served, ck, world, rank)
        else:
            ps = fp8_moe_scales.on_rank_grid([served(s) for s in srcs], srcs, which, ck, world, note)
            if world > 1:
                ps = [_shard_dim0(p, world, rank) for p in ps] if which == "w13" else [_shard_dim(p, 1, world, rank) for p in ps]
        if world > 1:
            note["shard"] = f"rank {rank}/{world} ({'per-part dim 0' if which == 'w13' else 'dim 1'})"
        stack.append(np.concatenate(ps, axis=0) if len(ps) > 1 else ps[0])
    out = np.stack(stack, axis=0) if stack else np.zeros(shape, dtype=np.uint16)
    if list(out.shape) != list(shape):
        raise ValueError(f"{name}: composed shape {list(out.shape)} != Program shape {shape}")
    return np.ascontiguousarray(out), note


def compose(name: str, shape: list[int], bits: int, ck: CheckpointIndex, *, world: int = 1, rank: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """The served model's tensor `name` (wrapper prefix already stripped) of `shape` / `bits`, composed from the checkpoint.
    Returns (raw words row-major, note {sources, rule, cast, shard})."""
    note: dict[str, Any] = {"sources": [], "rule": None, "cast": None, "shard": None}
    tied = bool(ck.config.get("tie_word_embeddings", False))

    def served(src: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        a, cast = fp8_ue8m0.served_words(ck, src, bits, rows) or to_served(ck.array(src, rows=rows), ck.dtype(src), bits)
        if cast:
            note["cast"] = cast
        note["sources"].append(src)
        return a

    parts: list[np.ndarray] = []
    whole: list[str] = []               # the parts are these whole checkpoint tensors, read below (a dim-0 shard: the rank's rows only)
    if ck.has(name):
        note["rule"] = "1:1"
        # a checkpoint that stores the fused tensor itself (Phi-3): the rank shard is PER PART, so split it by the config's geometry
        fused = next((k for k in FUSED if re.search(r"\." + re.escape(k) + r"\.\w+$", name)), None) if world > 1 else None
        if not fused:
            whole = [name]
        elif fused.endswith("qkv_proj") or fused == "self_attn.wqkv":
            parts = [served(name)]
            c = ck.config
            hq, hkv = int(c.get("num_attention_heads", 0)), int(c.get("num_key_value_heads") or c.get("num_attention_heads", 0))
            d = int(c.get("head_dim") or (int(c.get("hidden_size", 0)) // max(hq, 1)))
            if hq and d and parts[0].shape[0] == (hq + 2 * hkv) * d:
                a = parts[0]
                parts = [a[:hq * d], a[hq * d:(hq + hkv) * d], a[(hq + hkv) * d:]]
                note["rule"] = "1:1 fused tensor split q|k|v by config geometry"
        else:                                                                        # gate_up: two equal halves
            a = served(name)
            parts = [a[: a.shape[0] // 2], a[a.shape[0] // 2:]]
            note["rule"] = "1:1 fused tensor split gate|up"
    elif name == "lm_head.weight" and tied and ck.has("model.embed_tokens.weight"):
        note["rule"] = "tied lm_head <- model.embed_tokens.weight"
        whole = ["model.embed_tokens.weight"]
    elif name == "normalizer" or name.endswith(".normalizer"):
        # Gemma-2's embedding normalizer (on the real Gemma-2-2B Program, the ONE unresolved field of 316): a
        # non-persistent 0-d buffer `torch.tensor(hidden_size ** 0.5)` created at the served dtype in vLLM's Gemma2Model.__init__ -- no
        # checkpoint tensor, a function of config.json's hidden_size alone.  It is an analytic table like the rotary table
        # (`acquire.plan.analytic_table_of_buffer`), and its value is pinned EXACTLY: the Python float sqrt(H) rounded once
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
            return _compose_experts(name, m, shape, ck, world, rank, note, served)
        for fused, srcs in (*FUSED.items(), *FUSED_W13.items()):
            m2 = re.match(r"^(.*)\." + re.escape(fused) + r"\.(\w+)$", name)
            names = [f"{m2.group(1)}.{s}.{m2.group(2)}" for s in srcs] if m2 else []
            if names and all(ck.has(s) for s in names):
                note["rule"] = f"fused {fused} <- " + "|".join(srcs) + " (dim 0)"
                whole = names
                break
    shapes = [ck.tensors[s][2] for s in whole]
    full = [int(sum(s[0] for s in shapes))] + list(shapes[0][1:]) if shapes and all(shapes) else None
    # a DeepGEMM grouped einsum weight (`is_bmm`: DeepSeek-V4's wo_a and its scales) is the rank's g*r rows viewed as [g, r, ...]
    rows = [shape[0] * shape[1], *shape[2:]] if full and len(shape) == len(full) + 1 else list(shape)
    if world > 1 and full and not kv_heads_replicated(name, rows, shapes, ck.config, world) and rows != full and len(rows) >= 1 and rows[0] * world == full[0] \
            and rows[1:] == full[1:]:
        ps = _rank_rows(whole, served, ck, world, rank)
        out = (np.concatenate(ps, axis=0) if len(ps) > 1 else ps[0]).reshape(shape)
        note["shard"] = f"rank {rank}/{world}: per-part dim-0 slice" + (f" viewed as {list(shape)}" if rows != list(shape) else "")
        if list(out.shape) != list(shape):
            raise ValueError(f"{name}: composed shape {list(out.shape)} != Program shape {shape}")
        return np.ascontiguousarray(out), note
    parts = parts or [served(s) for s in whole]
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
    if list(out.shape) == rows != list(shape):
        out = out.reshape(shape)
    if list(out.shape) != list(shape):
        raise ValueError(f"{name}: composed shape {list(out.shape)} != Program shape {shape}")
    return np.ascontiguousarray(out), note
