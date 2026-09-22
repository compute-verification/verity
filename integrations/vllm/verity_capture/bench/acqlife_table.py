"""R10 acq-life — build `lifetimes.json` + `lifetimes.md` from the measured passes of `acqlife_lifetimes.py` (raw/lt_nohold, raw/lt_hold,
raw/lt_nohold_fa2) plus the SOURCE-INSPECTED overwriter of every class (vLLM d9105ea80, Llama architecture, BI=1 eager).  Laptop-side.

    python -m verity_capture.bench.acqlife_table --raw out/gen/r10/acq-life/raw --out out/gen/r10/acq-life
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# hook order inside one decoder layer L (12 hooks; +1 `fa2.m1/L<k>` after rotary_emb/k when the M1 source is attached)
LAYER_HOOKS = ["input_layernorm/0 (normed)", "input_layernorm/1 (residual)", "qkv_proj/0", "rotary_emb/0 (q)", "rotary_emb/1 (k)", "[fa2.m1 stream]",
               "attn", "o_proj/0", "post_attention_layernorm/0 (normed)", "post_attention_layernorm/1 (residual)", "gate_up_proj/0", "act_fn", "down_proj/0"]

# class -> (manifest rows, producer op [source], in-place overwriter [source], first free (ref drop) [source], collection point, primitive)
ANNOT = {
    "embed_tokens/in (R1b input ids)": dict(rows="R1b", producer="runner prepare_inputs: `input_ids.gpu[:n]` view of the PERSISTENT runner buffer (H2D copy_ non_blocking, stream 0)",
        inplace="next step's prepare_inputs `input_ids.copy_` (H2D) overwrites the same buffer", freed="never (persistent buffer)",
        point="any time before the next step's prepare_inputs; step-end batch OK", primitive="stream-0 copy enqueued before the next step (no ref needed)"),
    "embed_tokens out (R2b)": dict(rows="R2b", producer="VocabParallelEmbedding.forward (embedding gather), becomes `residual` for layer 0",
        inplace="layer-0 post_attention_layernorm: `fused_add_rms_norm(input, residual)` writes residual := x+res IN PLACE (llama.py:328)", freed="end of forward (residual chain)",
        point="≤ layer-0 o_proj hook (P1 of layer 0)", primitive="stream-0 copy enqueued before the layer-0 post-attn fused norm"),
    "layer0 input_layernorm out (R2c, Triton, not fused)": dict(rows="R2c", producer="RMSNorm.forward_cuda → rms_norm_batch_invariant (Triton `_rms_norm_kernel`, residual=None) → NEW tensor",
        inplace="none", freed="when layer-0 self_attn returns (`hidden_states` rebinds, llama.py:325) → allocator reuse", point="≤ layer-0 o_proj hook (P1)", primitive="stream-0 copy; a held ref protects it to step end"),
    "input_layernorm/normed (R2d/R2e fused)": dict(rows="R2d", producer="`fused_add_rms_norm(input=down_proj_out, residual)`: normed written INTO the down_proj output buffer (same storage; llama.py:324)",
        inplace="none after production (it is the writer's output)", freed="when self_attn returns (llama.py:325) → block reused by a later Linear/attn/act output alloc (measured hook+12…17)",
        point="≤ o_proj hook (P1) without refs; step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "input_layernorm/residual (R2d/R2e fused)": dict(rows="R2e", producer="same fused kernel: residual := bf16(x+res) written IN PLACE into the residual buffer",
        inplace="post_attention_layernorm fused norm of the SAME layer rewrites the residual buffer (llama.py:328) — measured hook+6 in both modes", freed="end of forward",
        point="≤ o_proj hook (P1)", primitive="stream-0 copy enqueued before the post-attn fused norm; a ref does NOT protect it"),
    "qkv_proj out (R2a GEMM)": dict(rows="R2a", producer="QKVParallelLinear → BI `matmul_persistent` output (`torch.empty`, batch_invariant.py:294)",
        inplace="`ops.rotary_embedding(positions, q, k)` rotates q,k IN PLACE inside this buffer (rotary_embedding/base.py:244) — the very next kernel; measured hook+1 both modes",
        freed="when LlamaAttention.forward returns", point="its own hook (P0) — zero-hook window", primitive="stream-0 copy inside the qkv_proj forward hook; nothing else is legal"),
    "rotary_emb/q (R2g RoPE, view of qkv)": dict(rows="R2g", producer="in-place RoPE; the hook output is a VIEW of the qkv buffer (prefill: non-contiguous, stride 960)",
        inplace="none after RoPE (attention reads q; KV-cache write reads k,v)", freed="qkv buffer freed when LlamaAttention.forward returns → reused within 3–10 hooks (measured)",
        point="≤ o_proj hook (P1) without refs; step end with a held ref", primitive="stream-0 copy (strided for prefill) or hold the qkv storage"),
    "rotary_emb/k (R2g RoPE, view of qkv)": dict(rows="R2g", producer="as q", inplace="none after RoPE", freed="as q", point="≤ o_proj hook (P1)", primitive="as q"),
    "attn out (R2h)": dict(rows="R2h", producer="Attention layer output buffer (`torch.zeros/empty` in the attention wrapper; FA2 writes `out=`)",
        inplace="none", freed="when LlamaAttention.forward returns (after o_proj) → reused by the mlp allocs within 2–6 hooks (measured)",
        point="≤ o_proj hook (P1) without refs; step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "o_proj out (R2a GEMM)": dict(rows="R2a", producer="RowParallelLinear → BI matmul output",
        inplace="post_attention_layernorm `fused_add_rms_norm(input=o_proj_out, …)` writes the NORMED output into this buffer (llama.py:328) — the very next kernel; measured hook+1 both modes",
        freed="(becomes the normed buffer)", point="its own hook (P1) — zero-hook window", primitive="stream-0 copy inside the o_proj forward hook"),
    "post_attention_layernorm/normed (R2d/R2e fused)": dict(rows="R2d", producer="fused kernel writes the normed output INTO the o_proj output buffer",
        inplace="none after production", freed="when mlp returns (llama.py:329) → reused by the next layer's Linear outputs within 5–7 hooks (measured)",
        point="≤ down_proj hook (P2) without refs; step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "post_attention_layernorm/residual (R2d/R2e fused)": dict(rows="R2e", producer="fused kernel: residual := bf16(x+res) IN PLACE",
        inplace="next layer's input_layernorm fused norm rewrites the residual buffer (llama.py:324) — measured hook+4 both modes", freed="end of forward",
        point="≤ down_proj hook (P2)", primitive="stream-0 copy enqueued before the next layer's input fused norm; a ref does NOT protect it"),
    "gate_up_proj out (R2a GEMM)": dict(rows="R2a", producer="MergedColumnParallelLinear → BI matmul output",
        inplace="none", freed="`x = maybe_fused_act_quant(...)` rebinds x (llama.py:120) BEFORE down_proj → the down_proj output alloc splits the freed block; measured hook+2 (down_proj hook)",
        point="≤ act_fn hook (P2a) without refs; down_proj hook / step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "act_fn out (R2f SiLU*mul)": dict(rows="R2f", producer="SiluAndMul.forward_cuda `torch.empty` (activation.py:145) + kernel",
        inplace="none", freed="when down_proj returns → reused within 2–4 hooks (measured)", point="≤ down_proj hook (P2) without refs; step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "down_proj out (R2a GEMM)": dict(rows="R2a", producer="RowParallelLinear → BI matmul output",
        inplace="next layer's input_layernorm `fused_add_rms_norm(input=down_proj_out, …)` writes the normed output into it (llama.py:324; final `model.norm` for layer 29) — measured hook+1 both modes",
        freed="(becomes the normed buffer)", point="its own hook (P2) — zero-hook window", primitive="stream-0 copy inside the down_proj forward hook"),
    "final norm/normed (R2d/R2e fused)": dict(rows="R2d", producer="`model.norm` fused kernel (writes into the layer-29 down_proj buffer)",
        inplace="none", freed="end of step; block reused by the NEXT step's o_proj alloc (measured next_step_end)", point="step end (P3)", primitive="stream-0 copy at step end"),
    "final norm/residual (R2d/R2e fused)": dict(rows="R2e", producer="`model.norm` fused kernel: residual IN PLACE",
        inplace="none (last writer)", freed="when LlamaModel.forward returns → the sampler's gather `hidden_states[logits_indices]` (model_runner.py:1466) takes the block — measured hook+1 (logits hook) without refs; survives with a ref",
        point="≤ logits_processor hook (P3) without refs; step end with a held ref", primitive="stream-0 copy (ref optional)"),
    "logits (R2a lm_head)": dict(rows="R2a", producer="LogitsProcessor (lm_head GEMM) output", inplace="none (`logits.to(float32)` is a new tensor)",
        freed="after sampling; block reused by the NEXT step's logits alloc (measured next_step_end)", point="step end (P3)", primitive="stream-0 copy at step end"),
    "sampled_token_ids (R2i)": dict(rows="R2i", producer="`runner.sample()` return (Sampler output → int32; D2H later on the runner's async copy stream when enabled — not active in this config)",
        inplace="none", freed="two steps later (measured step+2_end)", point="step end (P3)", primitive="stream-0 copy at step end"),
    "FA2.M1 export stream (R5-R8)": dict(rows="R5–R8", producer="tapped `varlen_fwd` writes the canonical i32 stream into the committer's PER-GEOMETRY buffer (hidden_source.py `_buffer`)",
        inplace="the NEXT launch with the same geometry = the next layer's attention rewrites the same buffer — measured hook+13 (13 hooks/layer with the tap)", freed="never (persistent per geometry)",
        point="≤ o_proj hook of the same layer (P1); or per-layer buffers → step end", primitive="stream-0 copy after the launch, before the next layer's launch; per-layer buffers cost 30× the buffer (132 MB prefill_155, 0.85 MB decode)"),
}

GROUPS = [
    ("P0 = qkv_proj hook", "forced by RoPE-in-place (zero-hook window)", ["qkv_proj out (R2a GEMM)", "input_layernorm/normed (R2d/R2e fused)", "input_layernorm/residual (R2d/R2e fused)"],
     "layer 0 adds: embed_tokens/in, embed_tokens out, layer0 input_layernorm out"),
    ("P1 = o_proj hook", "forced by the post-attn fused norm writing into o_proj out (zero-hook window)", ["o_proj out (R2a GEMM)", "rotary_emb/q (R2g RoPE, view of qkv)", "rotary_emb/k (R2g RoPE, view of qkv)", "attn out (R2h)", "FA2.M1 export stream (R5-R8)"],
     "the in_ln residual/normed may ride here instead of P0 (windows open until post_attn norm / self_attn return)"),
    ("P2a = act_fn hook", "gate_up out is freed before down_proj (its block is re-split by the down_proj output alloc)", ["gate_up_proj out (R2a GEMM)", "act_fn out (R2f SiLU*mul)"],
     "with a held ref to gate_up, P2a merges into P2"),
    ("P2 = down_proj hook", "forced by the next layer's input fused norm writing into down_proj out (zero-hook window)", ["down_proj out (R2a GEMM)", "post_attention_layernorm/normed (R2d/R2e fused)", "post_attention_layernorm/residual (R2d/R2e fused)"],
     "act_fn out may ride here (alive until down_proj returns)"),
    ("P3 = step end (after sample())", "no in-place writer before the next step", ["final norm/normed (R2d/R2e fused)", "final norm/residual (R2d/R2e fused)", "logits (R2a lm_head)", "sampled_token_ids (R2i)"],
     "final residual needs a held ref or the logits hook (the sampler gather reuses its block at hook+1)"),
]


def load(raw: Path, name: str) -> tuple[dict, dict]:
    d = raw / name
    return json.loads((d / "classes.json").read_text()), json.loads((d / "summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="out/gen/r10/acq-life/raw")
    ap.add_argument("--out", default="out/gen/r10/acq-life")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    c_nh, s_nh = load(raw, "lt_nohold")
    c_h, s_h = load(raw, "lt_hold")
    c_fa, s_fa = load(raw, "lt_nohold_fa2")
    classes = {}
    order = ["embed_tokens/in (R1b input ids)", "embed_tokens out (R2b)", "layer0 input_layernorm out (R2c, Triton, not fused)",
             "input_layernorm/normed (R2d/R2e fused)", "input_layernorm/residual (R2d/R2e fused)", "qkv_proj out (R2a GEMM)",
             "rotary_emb/q (R2g RoPE, view of qkv)", "rotary_emb/k (R2g RoPE, view of qkv)", "FA2.M1 export stream (R5-R8)", "attn out (R2h)", "o_proj out (R2a GEMM)",
             "post_attention_layernorm/normed (R2d/R2e fused)", "post_attention_layernorm/residual (R2d/R2e fused)", "gate_up_proj out (R2a GEMM)",
             "act_fn out (R2f SiLU*mul)", "down_proj out (R2a GEMM)", "final norm/normed (R2d/R2e fused)", "final norm/residual (R2d/R2e fused)",
             "logits (R2a lm_head)", "sampled_token_ids (R2i)"]
    for cls in order:
        src = c_fa if cls.startswith("FA2") else c_nh
        nh = src[cls]
        h = c_h.get(cls)
        an = ANNOT[cls]
        inplace_boundary = (h["first_change_earliest"] if h else "n/a (per-geometry buffer; a ref cannot protect it)") if not cls.startswith("FA2") else nh["first_change_earliest"]
        inplace_lost = bool(h and h["lost_by_single_end_of_step_copy"]["tensors"]) or cls.startswith("FA2")
        reuse_lost = nh["lost_by_single_end_of_step_copy"]["tensors"] > 0 and not inplace_lost
        verdict = ("copy-at-hook-required (in-place overwrite)" if inplace_lost
                   else "batchable-until-<collection point> (allocator reuse only; a held ref extends it to step end)" if reuse_lost
                   else "batchable-until-step_end" if nh["verdict"] == "batchable-until-step_end" else "persistent(≥2 steps)")
        classes[cls] = {
            "manifest_rows": an["rows"], "tensors_per_step": nh["per_step"], "tensors_run": nh["tensors"], "bytes_run": nh["bytes_total"],
            "bytes_prefill": nh["bytes_prefill"], "bytes_decode_step": nh["bytes_per_step_decode"], "dtype": nh["dtypes"], "shape_decode": nh["shapes_decode"], "shape_prefill": nh["shapes_prefill"],
            "producing_stream": "0 (legacy default = compute stream; measured at every hook)",
            "produced_at": an["producer"],
            "in_place_overwriter_source": an["inplace"],
            "first_change_hold_mode_measured": {"earliest": h["first_change_earliest"], "latest": h["first_change_latest"], "hist": h["first_change_hist"]} if h else None,
            "storage_release_source": an["freed"],
            "first_change_no_ref_measured": {"earliest": nh["first_change_earliest"], "latest": nh["first_change_latest"], "hist": nh["first_change_hist"],
                                             "reused_by_collected_tensor": nh["reused_by_collected"], "reuse_examples": nh["reused_by_examples"]},
            "lost_by_single_end_of_step_copy": {"no_refs": nh["lost_by_single_end_of_step_copy"], "refs_held": h["lost_by_single_end_of_step_copy"] if h else nh["lost_by_single_end_of_step_copy"]},
            "earliest_safe_consumer_read": "after the producing kernel completes on stream 0 (event after the hook, or stream order) — the arena copy is the read",
            "safe_window": inplace_boundary, "legal_collection_point": an["point"], "ordering_primitive": an["primitive"], "verdict": verdict,
        }
    totals = {
        "native_rows_no_refs": s_nh["single_end_of_step_copy_would_lose"], "native_rows_refs_held": s_h["single_end_of_step_copy_would_lose"],
        "with_fa2_stream_no_refs": s_fa["single_end_of_step_copy_would_lose"],
        "native_rows_total": {"tensors": s_nh["tensors_total"], "bytes": s_nh["bytes_total"]}, "with_fa2_total": {"tensors": s_fa["tensors_total"], "bytes": s_fa["bytes_total"]},
    }
    for k in ("native_rows_no_refs", "native_rows_refs_held", "with_fa2_stream_no_refs"):
        totals[k] = {kk: vv for kk, vv in totals[k].items() if kk != "per_step"}
    groups = []
    for name, why, members, note in GROUPS:
        n = sum(classes[m]["tensors_per_step"] for m in members if m in classes and not m.startswith("FA2"))
        b = sum(classes[m]["bytes_decode_step"] for m in members if m in classes and not m.startswith("FA2"))
        groups.append({"collection_point": name, "why_forced": why, "members": members, "tensors_per_decode_step_native": n, "bytes_per_decode_step_native": int(b), "note": note,
                       "ordering": "all members produced on stream 0 and unmodified/unfreed at this hook (measured both modes); ONE packed copy (pointer table → arena) or one cudaMemcpyAsync per member, enqueued on stream 0 inside this hook; no event needed for correctness (same stream); the consumer (GPU tree, side stream) waits on the step-end event recorded on stream 0"})
    doc = {"schema": "verity/acq-life/lifetimes/v1", "label": "measured on cp-acq2 (L40S) + source-inspected vLLM d9105ea80; workload B0 B1 C256 (15 model steps: 1 prefill 155 tok + 14 decode)",
           "hooks_per_step": {"native": 365, "with_fa2_m1": 395}, "layer_hook_order": LAYER_HOOKS, "streams": s_nh["streams"],
           "modes": {"no_refs": "pointer-only collector (the R9 fast path holds no references): in-place writes AND allocator block reuse change the bytes",
                     "refs_held": "collector holds a Tensor reference for 2 steps: only in-place writes can change the bytes"},
           "single_end_of_step_copy_would_lose": totals, "classes": classes, "batching_groups": groups,
           "raw": ["raw/lt_nohold", "raw/lt_hold", "raw/lt_memhist(2)", "raw/lt_nohold_fa2"], "tokens_identical_to_control_in_every_pass": all(s["tokens_identical_to_control"] for s in (s_nh, s_h, s_fa))}
    (out / "lifetimes.json").write_text(json.dumps(doc, indent=1) + "\n")

    L = []
    L.append("# Lifetimes of the collected interface values — measured (cp-acq2, L40S) + source-inspected (vLLM d9105ea80) — for owner 1\n")
    L.append(f"Workload `B0-cov_b0_b1_c256` (SmolLM2-135M, B1, C256, BI=1 eager bf16): 15 model steps carry hooks (1 prefill of 155 tokens + 14 decode; the 16th engine step only finishes the request). "
             f"365 hooks/step native (395 with the FA2 M1 stream), all on **CUDA stream 0** (legacy default = the compute stream; every hook, every pass; the runner's side streams are not created in this config). "
             f"Tokens identical to control in every instrumented pass. Two passes per value: **no-refs** (the R9 fast path's situation — nothing is held, so in-place writes AND allocator block reuse count) and **refs-held** (a Tensor reference is kept for two steps — only in-place writes count). "
             f"Boundaries are named `hook+k` = changed by the time the k-th later hook fired (the op between is named from source). Raw: `raw/lt_*/tensors.jsonl` (one row per tensor per step), `classes.json`, `summary.json`.\n")
    L.append("## What a single end-of-step copy would have lost (MEASURED, bytes compared at step end against the hook-time snapshot)\n")
    L.append("| collector situation | tensors lost / total | bytes lost / total | share |")
    L.append("|---|---:|---:|---:|")
    for k, lbl, tot in (("native_rows_no_refs", "native rows, no refs (R9 fast path)", "native_rows_total"), ("native_rows_refs_held", "native rows, refs held (in-place writes only)", "native_rows_total"), ("with_fa2_stream_no_refs", "native rows + FA2 M1 stream, no refs (complete reference)", "with_fa2_total")):
        t, tt = totals[k], totals[tot]
        L.append(f"| {lbl} | {t['tensors']:,} / {tt['tensors']:,} | {t['bytes']:,} / {tt['bytes']:,} | {t['bytes'] / tt['bytes']:.1%} |")
    L.append("\n\"One copy at the end of the step\" binds the wrong bits for 98–99 % of the interface bytes without references and for 31 % (33.1 MB/run: qkv/o_proj/down_proj outputs, both fused-norm residual outputs, the embedding) even with every reference held — those are in-place overwrites that no reference protects.\n")
    L.append("## Per-class table (one row per value class = producer op; 30 layers × per-step counts)\n")
    L.append("| class (manifest row) | n/step · B/decode step · B/prefill | produced at | in-place overwriter (source) → measured boundary, refs held | storage release (source) → measured boundary, no refs | stream | legal collection point | ordering primitive | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cls in order:
        c = classes[cls]
        hm = c["first_change_hold_mode_measured"]
        hm_s = f"**{hm['earliest']}**" + (f"…{hm['latest']}" if hm and hm["latest"] != hm["earliest"] else "") if hm else "n/a"
        nr = c["first_change_no_ref_measured"]
        nr_s = f"**{nr['earliest']}**" + (f"…{nr['latest']}" if nr["latest"] != nr["earliest"] else "")
        L.append(f"| {cls} ({c['manifest_rows']}) | {c['tensors_per_step']:g} · {int(c['bytes_decode_step']):,} · {c['bytes_prefill']:,} | {c['produced_at']} | {c['in_place_overwriter_source']} → {hm_s} | {c['storage_release_source']} → {nr_s} | 0 | {c['legal_collection_point']} | {c['ordering_primitive']} | **{c['verdict'].split(' (')[0]}** |")
    L.append("\nReading the boundaries: hook+1 after qkv_proj = the RoPE kernel; hook+1 after o_proj = the post-attention fused norm; hook+1 after down_proj = the next layer's input fused norm; hook+6 after input_layernorm/residual = the same layer's post-attention norm; hook+4 after post_attention_layernorm/residual = the next layer's input norm; hook+7 after the embedding = layer-0's post-attention norm. "
             "Every other class survives ≥ 2 steps when a reference is held and dies within 2–17 hooks without one — allocator reuse by the next `matmul_persistent` output (`torch.empty`, batch_invariant.py:294), the SiLU output (activation.py:145), the attention output buffer, or the sampler's gather (model_runner.py:1466). "
             "`_version` counters are unusable here (inference tensors); the caching allocator's history (`raw/lt_memhist2`) confirms the reallocating frames.\n")
    L.append("## Batching groups for owner 1 (windows overlapping at one legal collection point; decode step, native rows)\n")
    L.append("| collection point | why it is forced | members (all produced on stream 0, unmodified and still allocated at this hook) | tensors / bytes per decode step | note |")
    L.append("|---|---|---|---:|---|")
    for g in groups:
        L.append(f"| **{g['collection_point']}** | {g['why_forced']} | {'; '.join(g['members'])} | {g['tensors_per_decode_step_native']:g} / {g['bytes_per_decode_step_native']:,} | {g['note']} |")
    L.append("\n**Minimum legal launch count without touching producers: 4 collection points per layer (P0, P1, P2a, P2) = 120 launches/step + 1 at step end, vs 365 (395) today; holding Tensor references (native: retain the `TensorImpl`/storage from the hook until the packed copy is enqueued) removes P2a and lets every ride-along class wait for P1/P2 → 3 forced points per layer = 90 + 1 launches/step.** "
             "The three forced points are dictated by three zero-window in-place overwrites (RoPE after qkv_proj; fused norm after o_proj; fused norm after down_proj); no primitive other than *enqueue the copy on stream 0 before the next kernel* is correct for them — an event on a side stream cannot help because the overwriter is on the same stream. "
             "Held references are legal and cheap (they delay a block free by ≤ 1 layer: +~5 KB per layer decode, +~1 MB per layer prefill of transient pool use) but they change the allocator's block choices, not numerics. "
             "Ordering for the consumer is unchanged from R9: one event recorded on stream 0 at step end, waited on by the GPU-tree side stream; the arena block is released only after the retained copy's event (the R9 race). "
             "The FA2 M1 stream needs its copy before the next layer's launch (P1 of the same layer) unless 30 per-layer buffers are kept (132 MB for prefill_155, growing with sequence length; 0.85 MB for decode) — that trade-off is recorded, not taken.\n")
    L.append("## Streams\n")
    L.append(f"`{json.dumps(s_nh['streams'])}` — every collected value is produced on the legacy default stream (handle 0); the R9 collector copies on `torch.cuda.current_stream()` = the same stream; the GPU tree runs on the worker's side stream after `wait_event(step event)`. No cross-stream producer exists for the collected classes in this configuration (the runner's `async_output_copy_stream` is None here).\n")
    (out / "lifetimes.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'lifetimes.md'} and lifetimes.json; classes {len(classes)}; groups {len(groups)}")


if __name__ == "__main__":
    main()
