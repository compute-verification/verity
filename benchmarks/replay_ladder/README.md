# The replay ladder

Does replaying a served window under its recorded schedule give the served tokens on models other than the one it was
first measured on? On each model of a ladder of pinned checkpoints (`ladder.LADDER`: dense, MoE and FP8 MoE at one GPU,
a dense model at TP2 and an FP8 MoE at TP4), `ladder.py` serves one saturated window with the scheduler's step trace,
replays it several times on compiled vLLM at the integration's pin (d9105ea80), and serves a sample of its requests alone.
It runs `benchmarks/serve_tax`'s `record` arm and `replay.py` unchanged, so every model sees the same server, workload
and replay.

For each model:

- `serve`: one burst of 800 requests at 1,000 req/s on stock compiled vLLM measures the model's capacity; then the
  `record` arm serves serve_tax's workload (C4 prompts from the whole shard, each opening with the checkpoint's own
  document start and blank line, most behind one of eight shared prefixes, Poisson arrivals, every request decoding to
  its cap) at 1.45 times that capacity for 60 s: a saturated window of mixed arrivals, prefill chunks beside decodes.
- `replay`: in a fresh process, the window replayed under its recorded schedule, compiled, compared with serving step by
  step and token by token, and from the second replay on with the first replay too. `run.sh` runs three replays on
  vLLM's compile cache (the first fills it: the server's entries don't match an offline engine's, so it compiles) and
  one that compiles from empty caches (`VLLM_DISABLE_COMPILE_CACHE=1`, fresh Inductor and Triton directories); each
  replay's summary says which. Replays differing from each other at the same schedule are run-to-run variation.
- The arms: `run.sh` serves and replays each model twice, `pinned` with Inductor's reduction tiling fixed
  (`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`; unpinned, a fused add + RMSNorm kernel benchmarks two tilings on its first
  launch in each process and keeps the faster, which changes its sum order) and `unpinned` without, three replays, which
  say whether the model needs the pin. Each arm compiles into its own caches, since a graph compiled unpinned keeps both
  tilings when a pinned process loads it.
- `alone`: 64 of the window's requests, evenly spaced in arrival order, each served by itself on an empty prefix cache in
  one compiled engine, compared with serving and with the first replay: how far a request's tokens depend on its batch,
  which is why a replay needs the recorded schedule.
- `summary`: each model's window, replays and alone sample in summary.json and summary.md. A row of output is one
  request's token at one position, and its share exact is the window's output tokens less those that differ, over the
  window's output tokens.

`coverage_table.py` renders the coverage table, one row per model with its Circuit, Replay (pinned), Replay unpinned and
Proved status, from a sidecar that backs every cell with its run or art ids, and fills each ladder model's Replay cells
from a summary.json's two arms.

## Running it

On a node with the pinned vLLM venv and the ladder's checkpoints (`integrations/vllm/verity_vllm/ops/pod_bootstrap.sh
--cpu --cases ROLE,...` installs both on a RunPod pod and checks every shard), under `research run`, with the C4 shard
`benchmarks/ppl/datasets.json` pins sent with `--send`:

~~~sh
research run --on NODE --project verity --source . --cwd source --send c4-val-00000.json.gz -- \
  bash benchmarks/replay_ladder/run.sh qwen3-8b llama31-8b olmoe-1b-7b
python benchmarks/replay_ladder/coverage_table.py render COVERAGE.json --ladder OUT/summary.json --out DIR
~~~

`run.sh` takes `ARMS` (`pinned unpinned`), `REPLAYS` (3), `FRESH` (1) and `ALONE` (64) for the pinned arm,
`UNPINNED_REPLAYS` (3), and `SPAN_S`, `OVER` and `BURST` for the window
(`ladder.py serve --docs` caps the shard's documents a plan takes prompts from). A TP2 or TP4
model needs a node with that many GPUs. FP8 models run DeepGEMM, which JIT-compiles with the newest CUDA toolkit under
`/usr/local` (12.8 or newer; a RunPod image's 12.4 can't, so install `cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-cccl-12-9`
first), unless `DEEP_GEMM=0`; `engine_env.txt` records the choice.
