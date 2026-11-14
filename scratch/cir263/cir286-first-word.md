**First word. Both forms should be registered Definitions at the served statics, and nothing new is needed; the capture decides it. Card 0 of node A is booked Sat 14 Nov 10:00Z to 10:12Z. Every one of item 4's small-statics parts, and the head GEMM, is needed at served sizes by a draft row, and by the target's rows the same.** [read: pin `d9105ea80` `common/kernels.py`, `deepseek_v32/attention.py`, GLM-5.3's config, the capture map; nothing run yet]

**`fused_norm_rope` at `has_indexer=False`.** Programs 0 and 3 return at once: no index-K write and no top-k fill, so the top-k buffer keeps what the last full layer wrote. Programs 1 and 2 are the with-indexer kernel's code:
- `DsaRMSNorm_v1{N=2048, EPS=1e-05}` on q_c (q_a_layernorm) and `DsaRMSNorm_v1{N=512, EPS=1e-05}` on kv_c (kv_a_layernorm);
- `RoPEHeadGptJ_v2{D=64}` on k_pe;
- the MLA cache row `[kv_c_out | k_pe_out]` in bf16 (the served cache dtype is `auto`), a copy with no gates.

**`fused_q` at `has_indexer=False`.** The served query is bf16 (`quantize_mqa=False`), so programs 1 and 2 return. Program 0 is `RoPEHeadGptJ_v2{D=64}` on each of the 8 local heads' q_pe. The grid is (tokens, 3, 4), against (tokens, 3, 32) with the indexer.

**What is new: the target runs these forms too.** GLM-5.3's config has `index_topk_freq` 4 at offset 3, so 58 of its 78 layers are "shared". vLLM builds those layers with no indexer (`attention.py:161-195`), and they call both kernels with `has_indexer=False` (`:308-327`). The DSA capture of record compared only `has_indexer=True`.

**The capture.** One script compares both forms with these Definitions:
- every output word, on served operands and on the RMSNorm and subnormal-RoPE probes the two existing captures use;
- the top-k buffer and the index cache left untouched;
- the cache rows;
- one Triton specialisation of each kernel at `HAS_INDEXER` false.

Each form compiles separately, so its words could still differ. If they do, that form gets a new Definition.

**The card row.** Node A, card 0, cores 0-11, Sat 14 Nov 10:00Z to 10:12Z, the first free hours (between vllm's two all-card rows). One launch between 10:00Z and 10:04Z, only on an idle card with a free lease, off the card by 10:12Z. If the capture is not ready by 09:50Z, I release the row and book card 1 from 14:15Z, the next free card.

**Item 4 at served sizes, TP 8.** None of these is draft-only: the target's MoE layers need each at the same sizes. [read: the capture map's ops, at GLM-5.3's config]
- `EmbeddingShard_v1` at 19,360 rows of 6,144.
- `FlashInferFp8Linear_v1` (under 32 tokens) and `DeepGemmFp8Linear_v1` (from 32 tokens), at these (K, N):
  - `fused_qkv_a` (6144, 2624);
  - `q_b` (2048, 2048);
  - `o_proj` (2048, 6144);
  - the indexer's `wq_b` (2048, 4096), draft step 1 only;
  - the shared expert's `gate_up` (6144, 512) and `down` (256, 6144).
  - `kv_b` (512, 3584) only if a draft step takes MLA's prefill path, which I have not read.
- `SiluMulFp8BlockQuant_v1` at the routed experts' 256 with groups of 128, and `SiluMul_v2` at the shared expert's 256.
- `MoeSum_v3` at TOPK 8, H 6144, and `ResidualAdd_v1` at 6144.
- `AllGather_v1` at 8 ranks over 154,880 (a rearrangement with no gates, per the map).
- The head GEMM `Gemm_v2{K=6144, N=19360}`.
- Not needed: the dense MLP's shapes. The draft layer is an MoE layer (layer 78, past `first_k_dense_replace` 3).

Least sure: that the 10:00Z row holds. It needs vllm's 08:45Z row off every card by 10:04Z.
