"""The bytes a native committer stages, bounded before a run: one engine step's bytes in the arena (`step_bound_bytes`, from the
acquired modules and the workload envelope, with the arena block's size) and the whole run's stream under the workload's declared
schedule (`schedule_stream_bound`).  Each is a function of the model's config, the acquired modules and the workload alone."""
from __future__ import annotations

from typing import Any

from verity_vllm.commit.committer.acquired import GPU_CHUNK_BYTES, _mro_names
from verity_vllm.engine.profiles import layer_classes as LC

# ---------------------------------------------------------------------------------------------------------------------------------
# The arena block is sized from the WORKLOAD ENVELOPE at arm time, never below 512 MiB.  One engine step must fit in one block (the
# native collector has no overflow chaining); a step larger than the block is refused mid-step ("arena block too small for this step"),
# which leaves the step root incomplete.  The size changes nothing about what is bound, the packed layout, lifetimes or openings, so
# roots for a workload do not depend on it.
_ARENA_ROUND = 64 << 20
_ARENA_SAFETY = 1.25            # decode steps: measured 2.50 MB vs bound 2.72 MB (Llama-3.2-1B, 48/16)
_ARENA_SAFETY_PREFILL = 1.5     # prefill steps: measured 108.0 MB vs analytic 103.8 MB at 48 tokens (the M1 stream packs more than the
                                # StreamLayout formula per launch); the block is the MAX of both, so roots never depend on the factor
_ARENA_MIN = 512 << 20


def _cfg(model: Any, *names: str, default: int | None = None) -> int | None:
    cfg = getattr(model, "config", None)
    for n in names:
        v = getattr(cfg, n, None)
        if isinstance(v, int) and v > 0:
            return v
    return default


def moe_bytes_per_token(model: Any, moe_layers: int) -> int:
    """Bytes per token per step the MoE block source binds: per MoE layer TOPK x (f32 weight + i32 id) + TOPK x (2I + I + H) bf16
    pair rows + the H bf16 sum (`moe_source.MoeBlockSource`)."""
    if not moe_layers:
        return 0
    hidden = _cfg(model, "hidden_size", default=0) or 0
    inter = _cfg(model, "moe_intermediate_size", "intermediate_size", default=0) or 0
    topk = _cfg(model, "num_experts_per_tok", default=1) or 1
    return int(moe_layers) * (topk * 8 + topk * (2 * inter + inter + hidden) * 2 + hidden * 2)


def schedule_stream_bound(asz: dict[str, Any], reqs: list[dict[str, Any]], sampling: dict[str, Any] | None = None) -> dict[str, Any]:
    """The bytes the collector binds over the WHOLE RUN under the workload's DECLARED schedule (each
    request prefills at its `arrive_step` in one step, then decodes one token per step for `max_tokens`), with the same per-token / per-
    sequence / per-step widths and the same FA2/FA3 hidden-m1 launch geometry as `step_bound_bytes` (`asz` = its record).  The envelope
    bound `retained_bytes_bound` is a worst case over any packing (B x max_prompt tokens of prefill, every sequence padded to the longest);
    for a known workload this is the stream of record: per step, and the hidden family's share (the bytes `--retain-exclude fa3_hidden_m1`
    leaves on the device).  Conservative where vLLM's actual packing is unknown: a step with prefill AND decode sequences pads every
    sequence of its hidden launch to the step's longest query (the FA3 varlen slab); chunked prefill is not modelled (one step per prompt)."""
    from verity_vllm.acquire.sources.hidden_source import (
        CHUNK_WORDS,
        fa2_kblock_n,
        fa3_tile,
        mat_total_words,
    )

    md = asz.get("model")                 # a step bound without its model: one head of 64, one layer, no hidden-value rows
    if not md:
        md = {}
    heads, kv_heads = int(md.get("heads") or 1), int(md.get("kv_heads") or 1)
    head_dim, layers = int(md.get("head_dim") or 64), int(md.get("layers") or 1)
    hidden_m1 = bool(md.get("hidden_m1"))
    fa3 = str(md.get("attn_backend") or "fa2") == "fa3"
    g = max(1, heads // kv_heads)
    per_token, per_seq, per_step = int(asz["per_token_bytes"]), int(asz["per_seq_bytes"]), int(asz["per_step_bytes"])

    def hidden_words(HB: int, M: int, NB: int, BN: int) -> int:
        return -(-mat_total_words(HB, M, NB, BN, head_dim) // CHUNK_WORDS) * CHUNK_WORDS

    def launch_words(n_seq: int, seqlen_q: int, seqlen_k: int) -> int:
        if not hidden_m1 or n_seq <= 0:
            return 0
        if fa3:
            if head_dim not in (64, 128):
                return 0
            _, bn = fa3_tile(head_dim, seqlen_q, g, 1)
            return hidden_words(n_seq * kv_heads, seqlen_q * g, (seqlen_k + bn - 1) // bn, bn)
        bn = fa2_kblock_n(head_dim, 1) if head_dim in (64, 96, 128, 256) else 128
        if seqlen_q == 1 and heads > kv_heads:
            return hidden_words(n_seq * kv_heads, g, (seqlen_k + bn - 1) // bn, bn)
        return hidden_words(n_seq * heads, seqlen_q, (seqlen_k + bn - 1) // bn, bn)

    smp_max = int((sampling or {}).get("max_tokens") or 1)
    rq = []
    for r in reqs:
        ids = r.get("prompt_token_ids")       # a declared request gives its prompt's ids, or only how many they are
        n_p = len(ids) if ids else int(r.get("prompt_len") or 1)
        rq.append((int(r.get("arrive_step") or 0), int(n_p), int(r.get("max_tokens") or smp_max)))
    horizon = max((a + n for a, _p, n in rq), default=0)
    steps: list[dict[str, int]] = []
    total = hidden_total = hidden_prefill_total = 0
    for s in range(horizon):
        pre = [(p, n) for a, p, n in rq if a == s]                                 # prefilling this step
        dec = [(p + (s - a), n) for a, p, n in rq if a < s < a + n]                 # decoding: kv length so far
        if not pre and not dec:
            continue
        tokens = sum(p for p, _n in pre) + len(dec)
        n_seq = len(pre) + len(dec)
        m_q = max([p for p, _n in pre] + ([1] if dec else []))
        k_max = max([p for p, _n in pre] + [k for k, _n in dec])
        hid = layers * 4 * launch_words(n_seq, m_q, k_max)
        b = tokens * per_token + n_seq * per_seq + per_step + hid
        steps.append({"step": s, "tokens": tokens, "seqs": n_seq, "prefill_seqs": len(pre), "bytes": b, "hidden_bytes": hid})
        total += b
        hidden_total += hid
        if pre:
            hidden_prefill_total += hid
    big = max(steps, key=lambda d: d["bytes"]) if steps else None
    return {"requests": len(rq), "steps": len(steps), "horizon": horizon, "run_bytes": total, "hidden_bytes": hidden_total,
            "hidden_prefill_bytes": hidden_prefill_total, "largest_step": big,
            "prefill_steps": [d for d in steps if d["prefill_seqs"]][:16], "decode_step_bytes_max": max([d["bytes"] for d in steps if not d["prefill_seqs"]] or [0]),
            "model": "declared schedule: one prefill step per arrive_step group, one decode token per step; hidden launch padded to the step's longest query"}


def step_bound_bytes(model: Any, modules: list[tuple[str, Any]], envelope: dict[str, Any], *, hidden_m1: bool, attn_backend: str = "fa2", moe_layers: int = 0,
                     chunk: int = GPU_CHUNK_BYTES) -> dict[str, Any]:
    """Upper bound of the bytes ONE engine step binds into the arena, from the acquired-module set (the required-value manifest of the
    native path: forward outputs of ACQUIRE_CLASSES, per token) and the workload envelope (`batch`, `max_prompt`, `max_new`,
    `max_num_batched_tokens`, `max_model_len`).  Prefill: the admitted prompt tokens of one step (<= max_num_batched_tokens, <= B x max
    prompt) x bytes/token + per-sequence logits + the FA2 M1 hidden stream of every launch (superlinear in the sequence length: the
    worst case packs whole max-length prompts into the step).  Decode: B tokens x bytes/token + hidden stream at kv length max_model_len."""
    import torch

    hidden = _cfg(model, "hidden_size", default=0) or 0
    heads = _cfg(model, "num_attention_heads", default=1) or 1
    kv_heads = _cfg(model, "num_key_value_heads", default=heads) or heads
    head_dim = _cfg(model, "head_dim", default=None) or (hidden // max(1, heads))
    inter = _cfg(model, "intermediate_size", "moe_intermediate_size", default=4 * hidden) or 4 * hidden
    layers = _cfg(model, "num_hidden_layers", default=1) or 1
    vocab = _cfg(model, "vocab_size", default=32000) or 32000
    per_token = 0            # bytes per token of module outputs
    per_seq = 0              # bytes per sequence per step (logits, sampled ids)
    per_step = 0             # fixed per-step bytes (chunk padding of every acquired tensor)
    widths: dict[str, int] = {}
    # A module SHARED by every layer (Llama's one `rotary_emb`, registered under layer 0) is acquired once per
    # INVOCATION -- `layers` times per step -- not once per module: count it per layer, else the bound is `(layers - 1) x width` per token
    # short (76,800 B/token on Llama-3.2-1B, 1-3 % of a prefill step).
    n_rope = sum(1 for _n, _m in modules if LC.ROTARY in _mro_names(_m))
    rope_mult = max(1, -(-layers // n_rope)) if n_rope else 1
    no_parameter: Any = object()            # a module with no parameter counts two bytes an element
    for name, m in modules:
        names = _mro_names(m)
        p = next(m.parameters(), no_parameter)
        isz = 2 if p is no_parameter else p.element_size()
        w = 0
        if LC.LOGITS in names:
            per_seq += vocab * 4
        elif LC.SAMPLER in names:
            per_seq += 8
        elif LC.EMBEDDING in names:
            w = int(getattr(m, "embedding_dim", hidden) or hidden) * isz + 4          # + the input ids (ACQUIRE_INPUT_CLASSES)
        elif names & LC.LINEAR:
            out = getattr(m, "output_size_per_partition", None) or getattr(m, "output_size", None)
            if not out and hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
                out = int(m.weight.shape[0])
            w = int(out or hidden) * isz
        elif names & LC.NORMS:
            hs = int(m.weight.numel()) if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor) else hidden
            w = 2 * hs * isz                                                            # fused add+norm returns (out, residual)
        elif names & LC.GATED_ACTS:
            w = inter * 2
        elif LC.ROTARY in names:
            w = (heads + kv_heads) * head_dim * 2 * rope_mult                           # returns (q, k); once per layer when shared
        elif LC.ATTENTION in names:
            w = heads * head_dim * 2
        else:
            w = hidden * 2
        per_token += w
        per_step += chunk
        cls = next((c for c in (LC.LOGITS, LC.SAMPLER, LC.EMBEDDING, LC.RMS_NORM, LC.ROTARY, LC.ATTENTION, LC.SILU_AND_MUL, LC.LINEAR_BASE) if c in names), "other")
        widths[cls] = widths.get(cls, 0) + w
    moe_pt = moe_bytes_per_token(model, moe_layers)
    if moe_pt:
        per_token += moe_pt
        per_step += moe_layers * 6 * chunk
        widths["MoeBlock"] = moe_pt
    B = max(1, int(envelope.get("batch") or 1))
    max_prompt = max(1, int(envelope.get("max_prompt") or 1))
    max_new = max(1, int(envelope.get("max_new") or 1))
    max_model_len = int(envelope.get("max_model_len") or (max_prompt + max_new))
    mnbt = envelope.get("max_num_batched_tokens")
    T_prefill = min(int(mnbt) if mnbt else B * max_prompt, B * max_prompt)

    # FA2: the M1 stream is arithmetic in the kernel tile (kBlockN, head_dim) -- 128/64 for hd 64, 64/hd otherwise (num_splits==1 kernel).
    # FA3: under FA3 (attn_backend "fa3") the slab is (sequence, KV head) and the rows are PackGQA-packed (M = seqlen_q * g); the tile
    # is tile_size_fwd_sm90's (hd 64: 192x128 prefill / 64x192 one-warpgroup decode; hd 128: 128x128 / 64x128) -- hidden_source.fa3_tile.
    from verity_vllm.acquire.sources.hidden_source import (
        CHUNK_WORDS,
        fa2_kblock_n,
        fa3_tile,
        mat_total_words,
    )
    g = max(1, heads // kv_heads)
    fa3 = attn_backend == "fa3"

    def hidden_words(HB: int, M: int, NB: int, BN: int) -> int:
        return -(-mat_total_words(HB, M, NB, BN, int(head_dim)) // CHUNK_WORDS) * CHUNK_WORDS

    def launch_words(n_seq: int, seqlen_q: int, seqlen_k: int) -> int:
        if fa3:
            if int(head_dim) not in (64, 128):
                return 0
            _, bn = fa3_tile(int(head_dim), seqlen_q, g, 1)
            return hidden_words(n_seq * kv_heads, seqlen_q * g, (seqlen_k + bn - 1) // bn, bn)
        bn = fa2_kblock_n(int(head_dim), 1) if int(head_dim) in (64, 96, 128, 256) else 128
        if seqlen_q == 1 and heads > kv_heads:            # FA2 decode "swapped" kernel: rows = the query heads of a KV group
            return hidden_words(n_seq * kv_heads, g, (seqlen_k + bn - 1) // bn, bn)
        return hidden_words(n_seq * heads, seqlen_q, (seqlen_k + bn - 1) // bn, bn)

    # prefill: the tapped launch streams HB = (sequences in the step) x heads rows blocks of M = max_seqlen_q rows each, i.e. EVERY
    # sequence of the step is padded to the step's longest prompt (hidden_source.varlen_fwd: HB = b*h, M = max_seqlen_q, NB from
    # max_seqlen_k; FA3: slab (sequence, KV head), M = max_seqlen_q * g).  Worst case for T_prefill admitted tokens: one max-length prompt +
    # (B-1) one-token prompts -> B x heads x max_prompt rows.  (A "whole prompts" packing (T_prefill // max_prompt
    # sequences) under-bounds a B=8 Phi-3 256-token prefill by 2x -> "arena block too small for this step" mid-run; the bound is the stream
    # geometry itself.)
    M_pref = min(max_prompt, T_prefill)
    seqs_prefill = min(B, max(1, T_prefill - M_pref + 1))
    hid_prefill = layers * 4 * launch_words(seqs_prefill, M_pref, M_pref) if hidden_m1 else 0
    prefill = T_prefill * per_token + seqs_prefill * per_seq + per_step + hid_prefill
    hid_decode = layers * 4 * launch_words(B, 1, max_model_len) if hidden_m1 else 0
    decode = B * per_token + B * per_seq + per_step + hid_decode
    bound = max(prefill, decode)
    rnd = lambda n: -(-int(n) // _ARENA_ROUND) * _ARENA_ROUND  # noqa: E731
    decode_block = max(_ARENA_ROUND, rnd(decode * _ARENA_SAFETY))
    prefill_block = max(decode_block, rnd(prefill * _ARENA_SAFETY_PREFILL))
    block = max(_ARENA_MIN, prefill_block)   # single-size equivalent (explicit --arena-mb comparison)
    # the GPU tree keeps every step's used bytes for lazy openings (existing lifetime): upper bound over the whole workload
    retained = prefill * (-(-B * max_prompt // max(1, T_prefill))) + decode * max_new
    return {"per_token_bytes": per_token, "per_seq_bytes": per_seq, "per_step_bytes": per_step, "widths": widths,
            "rope_per_layer": True, "rope_modules": n_rope, "rope_invocations_per_step": n_rope * rope_mult,
            "prefill_tokens": T_prefill, "prefill_step_bytes": prefill, "prefill_hidden_bytes": hid_prefill,
            "decode_step_bytes": decode, "decode_hidden_bytes": hid_decode, "step_bound_bytes": bound, "block_bytes": block,
            "prefill_block_bytes": prefill_block, "decode_block_bytes": decode_block,
            "retained_bytes_bound": retained, "envelope": {"batch": B, "max_prompt": max_prompt, "max_new": max_new,
                                                           "max_num_batched_tokens": mnbt, "max_model_len": max_model_len},
            "model": {"hidden": hidden, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "inter": inter, "layers": layers,
                      "vocab": vocab, "hidden_m1": bool(hidden_m1), "attn_backend": attn_backend, "moe_layers": int(moe_layers), "moe_bytes_per_token": moe_pt}}
