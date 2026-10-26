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

# 18:45Z, one more piece for the served path (from VLLM-113, #4753)

#4753 makes registered reads count only when the window's registration records the pinned root (`registered_roots`, from `verity-vllm advice --weights-of-record FILE`). serve-openai and `llama1b_serve_seal.sh` don't pass it yet, so every window served today counts its weights reads as free.

Add `--weights-of-record` to the served path for the window: serve-openai's advice per window, or the seal's registration. The record comes from `window_circuit.weights_record` over the checkpoint's pinned shards (`manifests/checkpoints.json`). Stack it on #4753's head or after it lands. It's small, but the window's leak count depends on it.

# 21:15Z, for G2 (from vllm-composed's dry rerun r20261026-204546-2d7c)

serve-openai on vy-dc-h200 ran with `async_scheduling` **true**, while #4770's int8 row declares `async_scheduling: false`, which G2 needs. φ records what the engine reports, so the two disagree today. Before G2's GPU check, find where serve-openai sets it (the engine args from the row, or vLLM's default), make the engine follow the row, and refuse a mismatch by name. Fold that into #4773 or a small PR of its own.

# 21:35Z, #4779's base changed

#4753 was rebuilt directly on main at **aa52fe479a**: three commits, without #4594 or #4573, so its SHAs are new. #4779 still contains the old 6303f5cfdb and #4594. After #4753 lands, rebase #4779 onto main, not onto the old head, and then mark it. #4753's mark waits on #4747, #4757, #4780, #4795, #4790 and #4594 landing.
