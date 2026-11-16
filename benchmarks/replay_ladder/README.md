# The replay ladder

SPOT's proof server replays the requests an audit samples, under the schedule serving recorded, and a replay is of use
only if it gives back the served tokens exactly. This benchmark asks whether it does on more than one model: on each
model of a ladder of pinned checkpoints (`ladder.LADDER`: dense, MoE and FP8 MoE at one GPU, a dense model at TP2, an
FP8 MoE at TP4 and GLM-5.3 at TP8), `ladder.py` serves one saturated period, replays it several times on compiled vLLM
at the adapter's pin, and serves a sample of its requests alone, then reports each replay's share of output tokens
exact. It runs `benchmarks/serve_tax`'s server, workload and replay unchanged. `ladder.py`'s and `run.sh`'s headers say
what each step and arm does; `inductor_pin.py`'s says why the `pinned` arm exists.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/replay_ladder
~~~

It runs the ladder's tests on the CPU (the pin's plugin, the plan, the summary, the status table and the top-k probe)
in a few seconds; the ladder itself needs GPUs, below.

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
