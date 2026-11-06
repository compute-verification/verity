# Replay determinism

Whether a served window's recorded schedule, replayed, gives the bytes it served without batch invariance
(`VLLM_BATCH_INVARIANT` unset), and whether a request served alone gives the bytes it got inside its batch. One GPU, one
MoE model, vLLM at the integration's pin with the integration's engine profile (`verity_vllm.engine.env.ENGINE`).

The model is Moonlight-16B-A3B-Instruct (`replay_det.py`'s `MODEL` and `REVISION`), bf16, 32 GB: a `DeepseekV3ForCausalLM`
that vLLM runs through the same model file as GLM-5.3 (its DeepSeek-V2 file), with MLA attention (kv_lora_rank 512) and a
sigmoid router with a score-correction bias (`noaux_tc`, one group, top 6 of 64, two shared experts). It differs from
GLM-5.3 in having no sparse-attention indexer, bf16 weights in place of FP8 blocks, and, on one GPU, no tensor-parallel
collectives.

## What runs

`run.sh` under `research run`, each phase a fresh process on the same GPU under one `gpu-lease 1`:

- the window: `N` requests (48) cut from this tree's Markdown in the model's chat format, prompts log-uniform from 16 to
  6,000 tokens, outputs from 8 to 512, half greedy and half sampled at temperature 0.8 with a seed each, Poisson
  arrivals; every request ignores EOS, so the schedule doesn't depend on the tokens;
- `serve`: the window served at its arrival times, recording each arrival's step and each step's batch: rows in the
  runner's order with each one's request, first position and length, and the KV blocks each request gained;
- `replay1`, `replay2`: the window replayed with each request added before its recorded step, fingerprinting every row;
  `replay_plain` the same without any hooks, for tokens;
- `alone`: each request served by itself;
- `compare.py`: schedules, tokens, and every row's fingerprints between the phases.

The engine is eager, with chunked prefill at 2,048 tokens a step, prefix caching off, synchronous scheduling, a fixed KV
block count, `PYTHONHASHSEED=0` and request ids without vLLM's random suffix, so a replay can reproduce the schedule; the
compare refuses to compare values unless it did.

A row's fingerprint is the wrapping int64 sum of its 32-bit words times fixed odd keys, so a single changed bit changes
it. Fingerprints are taken of every module output of the model (each layer's norms, projections, rotary embedding, MLA
attention, router logits, shared and routed experts, the decoder layer), of the top-k weights and ids each MoE router
selects, and of every logits row, with its top 8 kept as values.

## Commands

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/replay_determinism/run.sh fetch
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=ID \
    -- bash benchmarks/replay_determinism/run.sh
~~~

`fetch` downloads the model to `MODEL_DIR` (default `/workspace/research/scratch/replay-determinism/moonlight`) and
refuses unless 50 GB and 100,000 inodes stay free on `/workspace`. The run writes the window (`workload.json`), each phase's steps, tokens, timing
and logits rows, `lease.json` (the lease's wall time, its GPU-seconds) and `compare.json` and `compare.md` to `$OUT`; the
per-row fingerprints stay in `SCRATCH` under `WORK`. `N`, `PHASES`, `PY` (a Python with vLLM at the pin), `WORK` and
`LEASE_MIN` are settable.

`compare.md` answers:

- whether replay matches replay, and served matches replay: schedule, tokens, every row's module outputs, logits rows;
- alone against in-batch, over the positions whose inputs agree: the rows bit-identical, by prefill and decode and, for
  prefill, by whether the row's chunk had the same boundaries; for the rest the first module output that differs, by op
  and by layer; the rows whose router picks another expert; logits rows and argmax; tokens per request;
- each op's batch variance: over the rows where everything the op reads (for MLA attention, the keys of every earlier
  position too) is bit-identical alone and in the batch, how many rows of its output still differ;
- the GPU-seconds of each phase.
