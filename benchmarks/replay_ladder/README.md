# The replay ladder

Does replaying a served window under its recorded schedule give the served tokens on models other than the one it was
first measured on? On each model of a ladder of pinned checkpoints (`ladder.LADDER`: dense, MoE and FP8 MoE at one GPU,
a dense model at TP2, an FP8 MoE at TP4 and GLM-5.3, an FP8 MoE with DeepSeek sparse attention, at TP8), `ladder.py` serves one saturated window with the scheduler's step trace,
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
- The arms (`LADDER_ARM`, declared in `settings.py`): `run.sh` serves and replays each model twice. Unpinned, Inductor
  times candidates on the device and keeps the fastest: a kernel's configs on its first launch in each process, such as
  a norm's tilings, and at compile time whether to fuse kernels. Two processes can choose differently, and a
  reduction's config sets the order of its sums. `pinned` makes every one of those timings the same in every process of
  the server and of each replay (`inductor_pin.py`, a vLLM general plugin in the server's processes), so each choice
  keeps Inductor's first candidate. Inductor's own deterministic mode can't do this at the integration's torch: Dynamo
  turns it off after a process's first compile, and while it holds it refuses the combo-kernel timing vLLM's compile
  runs. `unpinned` runs three replays without the pin, which say whether the model needs it. `rblock` (on request) sets
  `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`, which drops the halved-R0_BLOCK candidate but leaves a reduction's other
  candidates to the timing. Each arm compiles into its own caches, since a compile cache keeps the choices made where it
  was written. The capacity server's processes and each replay's log the configs their reductions kept at run time
  (`tilings.PID.jsonl`); the summary counts the reductions that kept more than one config over serving and the
  replays.
- `alone`: 64 of the window's requests, evenly spaced in arrival order, each served by itself on an empty prefix cache in
  one compiled engine, compared with serving and with the first replay: how far a request's tokens depend on its batch,
  which is why a replay needs the recorded schedule.
- `summary`: each model's window, replays and alone sample in summary.json and summary.md. A row of output is one
  request's token at one position, and its share exact is the window's output tokens less those that differ, over the
  window's output tokens.

`status_table.py` renders the per-model status table, one row per model with its Circuit, Replay (pinned), Replay unpinned and
Proved status, from a sidecar that backs every cell with its run or art ids, and fills each ladder model's Replay cells
from a summary.json's two arms.

## Running it

On a node with the pinned vLLM venv and the ladder's checkpoints (`adapters/vllm/verity_vllm/ops/pod_bootstrap.sh
--cpu --cases ROLE,...` installs both on a RunPod pod and checks every shard), under `research run`, with the C4 shard
`benchmarks/ppl/datasets.json` pins sent with `--send`:

~~~sh
research run --on NODE --project verity --source . --cwd source --send c4-val-00000.json.gz -- \
  bash benchmarks/replay_ladder/run.sh qwen3-8b llama31-8b olmoe-1b-7b
python benchmarks/replay_ladder/status_table.py render STATUS.json --ladder OUT/summary.json --out DIR
~~~

`run.sh` takes `ARMS` (`pinned unpinned`; `rblock` too on request), `REPLAYS` (3), `FRESH` (1) and `ALONE` (64) for the pinned arm,
`UNPINNED_REPLAYS` (3) for each other arm, `RBLOCK0_ARMS` (`pinned` to run the pinned arm at
`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` too, as the integration's engines do; each arm's `arm_env.txt` records it),
`LOGPROBS` (`1`: each replay records its sampled tokens' logprobs, and from the second on compares them with the
first's bit for bit, `replay.py --logprobs`), `FLASHINFER` (`1`: CUDA_HOME's nvcc on PATH and FlashInfer's sampler at
vLLM's default, so vLLM uses FlashInfer as a stock install with a CUDA toolkit does, its autotune and, at TP > 1 on Hopper
or Blackwell, its fused all-reduce + RMSNorm included; otherwise FlashInfer is off, since the pin installs neither
flashinfer-cubin nor nvcc, and so is its sampler, whose JIT needs the toolkit's curand headers), and
`SPAN_S`, `OVER` and `BURST` for the window (`ladder.py serve --docs` caps the shard's documents a plan takes prompts
from). `engine_env.txt` records `has_flashinfer`, and each arm's `autotune.sha256` the digest of every FlashInfer
autotune cache after serving and after each replay. A TP2, TP4 or TP8 model needs a node with that many GPUs. GLM-5.3's window stays inside 2,048 tokens of
context (the plan's longest request is 1,024 + 256), where its sparse attention's top-k keeps every key.

`prod.sh MODEL` runs the same serve and replays on a window shaped like real serving instead (`prod_plan.py`): vLLM's
benchmark_serving ShareGPT set (the serve tax's pinned file, `serve_tax.py dataset chat --fetch`: the node's copy,
downloaded where there is none, its SHA-512 checked), multi-turn ShareGPT
conversations whose turns share their prefixes, and C4 prompts of 8,192 to 32,768 tokens, each through the model's chat
template, half greedy and half at temperature 0.7 with a per-request seed, every request stopping at its end of text or
its cap; the servers' context limit is `MAX_MODEL_LEN` (40,960), and `FLASHINFER` is 1 unless set. Each arm's `kinds.json` breaks every replay's differing
requests down by kind, with each one's context at its first difference, and counts the window's requests whose context
passed 2,048 keys, the DSA indexer's top-k: a difference that starts within it can't come from the indexer's selection. With `LADDER_ROUTE_CAPTURE=DIR`, a directory holding
the route capture's plugin module and the settings registry beside it, both servers also load that plugin and write their routes under
`capacity/route_capture` and `record/route_capture`; replays never load it. FP8 models run DeepGEMM, which JIT-compiles with the newest CUDA toolkit under
`/usr/local` (12.8 or newer; a RunPod image's 12.4 can't, so install `cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-cccl-12-9`
first, and `libcurand-dev-12-9 libcublas-dev-12-9 cuda-nvrtc-dev-12-9` for `FLASHINFER=1`), unless `DEEP_GEMM=0`;
`engine_env.txt` records the choice.

`SERVED=DIR`, an earlier run's `ARM/MODEL` directory, has `run.sh` replay that run's window instead of serving one (one
model, with `ARMS` the arm it was served in). Where a DSA model's replays part, `LADDER_TOPK_PROBE` (`topk_probe.py`)
says: each replay's engine processes wrap vLLM's prefill top-k (`top_k_per_row_prefill`) and record, per call and
device, checksums of the indexer logits it read, the set of indices it selected and their order (`replay-I/topk`);
`sort` also sorts each row's indices before attention reads them, and `record` only records. `run.sh` compares each
replay after the first with the first, call by call (`topk-I.json`: per device, the first call each checksum differs at).
The first call whose logits agree while its set differs is a selection that varies on the same input.
`LADDER_TOPK_PROBE_REAL=module:function` names a replacement top-k for the probe to call and record instead of vLLM's.
Decode's top-k isn't probed.
