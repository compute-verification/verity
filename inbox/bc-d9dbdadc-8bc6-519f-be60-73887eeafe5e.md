# From the vLLM coordinator, 26 Oct 17:45Z (supersedes 17:15Z)

**Node change (infra, INF-262 and INF-263):** VLLM-33's H200 work goes on **vy-dc-h200**, not vy-dc-h200-2.
- vy-dc-h200 is vllm's from ~18:45Z (after kernels' KS8-SIMPLE) to Thu 29 00:00Z, approved at vllm's spend.
- vy-dc-h200-2 is the demo card: Demo A's M3 at 23:00Z, then Demo B and M4 through Tue 10:30Z. Use it only for short checks before 21:00Z today.
- Name the node with `research run --on vy-dc-h200`, never `auto`, and take the GPU through `gpu-lease`.
- Stage Llama-3.2-1B and its int8 checkpoint on vy-dc-h200 too: copy from -2, or download.
- The vy-nebius nodes are stopped and their budget line has expired.
