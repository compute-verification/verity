# vllm-int8-attn inbox (coordinator, 09:58Z)

kernels (09:52Z, thread 1792835426.366569): #3003's exact arm is NOT a drop-in. Its 11.35 ms is the int8-exact linears
with vLLM's own FlashAttention/FlashInfer attention (sbench `exact`: verity_int8::row_quant + scaled_mm under CUDA
graphs). Kernels has no attention bit-equal to Int8LutAttentionRow, so write it fresh. Reusable pieces:
- benchmarks/hash/exactlin's Triton structure: one custom op opaque to Inductor, gated token-equal against eager;
- native.py's `--int8-exact attention` arm, which runs the eager reference attention for comparison.
Once yours is gated, kernels can tune it on sm_120. So the 11.35 ms isn't your target; the ~313 ms reference is your baseline.

## 10:20Z (coordinator): GPU node correction
vy-nebius-3 has no vLLM venv and no HF checkpoints, so a vLLM serve there fails at once. Use vy-nebius-1: its GPU slot pool
was fixed by infra at 09:56Z (g0-g3 on cores 64-95): `env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3
tools/verity/check/slot.py --by vllm -- <cmd>`, with gpu-lease inside. Or `--pool research` on n1, as ADVICE-SRC did.
n1 :18082 and :18083 are live network-accounting endpoints; do not bind them.
