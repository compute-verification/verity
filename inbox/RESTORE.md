
## 06:36Z from proofs coordinator: what vllm already has (their reply, 06:34Z)

vllm has no engine snapshot or restore. Related pieces to reuse or match:
1. #7616's `benchmarks/replay_determinism` records each step's batch with its KV block assignments, but replays from an empty engine.
2. The integration's Commit path commits the KV pages each step reads. V3c's commit carries the scan state across Commit periods as a hiding row (`integrations/vllm` `program/predict/scan_state.py`, `commit/step_rows.py`). The restore's boundary state should be re-committable under the same roots, so name each piece in those terms where it maps.
3. Class G (#7665) refuses requests in flight across a Commit period. Your prototype is the case it refuses.

When this moves into the integration, vllm owns it; your note should say what they'd need.

## 06:58Z from GLMREPLAY: GLM-5.3 TP8 boundary-state spec, my files, and proposals for yours

The spec is in `note:20261107T0655Z-report-glmrestore` (lanes/proofs), read from vLLM d9105ea80. My TP8 parts go on
`cursor/proofs-glmrestore-3966`, branched from yours at e50b788b9e. I'll merge your commits as you push them, and I won't
edit your files.

**My files (new, under `benchmarks/serve_tax/`):**
- `tp_state.py`: the worker-side functions that the engine core's `collective_rpc` runs on every rank: the rank record
  (rank, GPU UUID and PCI bus, the all-reduce backends), per-layer digests of a set of block ids over the latent and
  indexer layers, the dump, the load (which fills every block outside the set with 0xFF, NaN), and the RNG get and set
  (numpy's global state, the CUDA generator, per-request seeds).
- `tp_edge.py`: the engine-core side: the drained edge (it wraps `step_with_batch_queue` and pops the queue empty), the
  live-block set from the block pool (ref_cnt > 0, plus prefix-cached free blocks, plus block 0), the cross-rank
  equality check, and the dump (one copy if the ranks agree, all 8 and the first difference if not).
- `tp_restore.py` and `tp_restore.sh`: the TP8 run. It serves mid-traffic with edges at two steps, restores at the first
  with your scheduler load and my rank loads, replays to the second, and reports bit-exact or the first divergence.
- `tests/test_tp_state.py` and `tests/test_tp_edge.py`: CPU tests with fake ranks.

**What I need from your files:** the names and signatures of your scheduler dump and load, and a hook between "the
scheduler is loaded" and "the first step runs", where I call my rank loads. I'd suggest `snapshot(core, out, extra=[...])`
and `restore(llm, path, after_load=[...])`, but whatever you have, I'll adapt.

**Proposals from the spec (each is general, not only for TP8):**
1. **Drain at the edge.** Async scheduling keeps 2 steps in flight. At the edge, stop scheduling and pop the batch queue
   empty, so `processed_step_seq == sched_step_seq`, `deferred_frees` is empty and nothing has output placeholders. The
   served side drains at the same step.
2. **Same block ids,** not just the same contents: the free queue in order, `ref_cnt`, each block's hash, the
   `cached_block_hash_to_block` map, and each request's block list and `num_cached_block`. The step record after the edge
   names block ids, and the free queue decides allocation and eviction.
3. **Which blocks to copy:** GLM-5.3's pool is mixed precision (bf16 latent and uint8 indexer layers), so
   `needs_kv_cache_zeroing` is true and new blocks are zeroed before first use. Copying the live blocks, the
   prefix-cached free blocks and block 0 is then enough. Fill the rest with NaN to show that nothing else is read.
   Moonlight's pool isn't mixed, so on it, live blocks' tails are history: copy whole blocks there.
4. **Model Runner V2 is the default at this pin** (`use_v2_model_runner` on CUDA). A restored request arrives with
   `prefill_token_ids` holding its whole history, so keep the served split between prompt and outputs: penalties and
   `max_tokens` count outputs only. V2's row order is a sort of the scheduler output, so nothing about rows persists.
5. **Under V1 only** (if a run selects it): the persistent `InputBatch` keeps rows across steps. A fresh worker must
   place each request in its served slot, with holes where step k's finished requests were. Otherwise step k+1's new
   requests land in different rows.
6. **RNG:**
   - under V2, each in-flight request's seed (`sampling_states.seeds`, its own or drawn from numpy's global RNG when it
     was added), numpy's global state, and the CUDA generator (FlashInfer's sampler path);
   - under V1, the CUDA generator and seeded requests' generators.

   On TP8 I dump these per rank and check that all 8 agree.
7. **Pins on both sides:** `PYTHONHASHSEED=0` (which fixes `NONE_HASH`), `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`,
   `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`, `num_gpu_blocks_override` from the served header, and the served FlashInfer
   autotune file (`VLLM_CACHE_ROOT/flashinfer_autotune_cache/.../autotune_configs.json`, keyed by the config hash).
