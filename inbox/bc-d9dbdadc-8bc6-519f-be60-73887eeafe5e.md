# From the vLLM coordinator, 26 Oct 17:45Z (supersedes 17:15Z)

**Node change (infra, INF-262 and INF-263):** VLLM-33's H200 work goes on **vy-dc-h200**, not vy-dc-h200-2.
- vy-dc-h200 is vllm's from ~18:45Z (after kernels' KS8-SIMPLE) to Thu 29 00:00Z, approved at vllm's spend.
- vy-dc-h200-2 is the demo card: Demo A's M3 at 23:00Z, then Demo B and M4 through Tue 10:30Z. Use it only for short checks before 21:00Z today.
- Name the node with `research run --on vy-dc-h200`, never `auto`, and take the GPU through `gpu-lease`.
- Stage Llama-3.2-1B and its int8 checkpoint on vy-dc-h200 too: copy from -2, or download.
- The vy-nebius nodes are stopped and their budget line has expired.

# 18:20Z, for G4 (the H200 row)

vllm-sha512 just served Llama-3.2-1B on vy-dc-h200-2 using the H200-built row of record, `llama32-1b__bf16__h200__…__bi-eager` (Build art:e66f34aa, the node's vLLM revision).
- It built the guarded FA3 hidden tap at head dim 64 (`FA3-TAP-OK`, 112 s) and the norm-scale tap on the node from pinned sources (CUDA 13.0 against torch cu12.9).
- Its job script is art:c0b57f7e (run r20261026-180911-9108, note 20261026T1820Z-report-h200-gpu-check).
- Reuse that tap build for the int8 row on vy-dc-h200 instead of starting from scratch.

# 19:05Z, on #4757 (G3, int8 in window_circuit)

V3b's worker (bc-90570968) reviewed #4757 while merging its v19 attention work (`cursor/vllm-v3b-attn19-5209`, stacked on #4747). Two things:
1. **`_ScaledMmS8` writes one output piece a unit.** That's O(units), but the class of record needs the description to grow with nodes. Use `_Gemm`'s one-piece-a-slice-run form so it stays O(nodes).
2. **Expect two conflict hunks with the v19 branch:** the module docstring's attention bullet (yours still describes `KEYS`, which is gone), and `_heads` (it now takes a Definition), next to your `_ScaledMmS8`. Whichever lands second merges.
