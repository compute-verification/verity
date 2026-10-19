# maxtput: the Commit at vLLM's maximum throughput

The hash loop (`benchmarks/hash/loop/`) measures the Commit at a fixed batch (B32). This measures it where vLLM is fastest:
native compiled vLLM at saturation on one GPU, against the Commit's arms on the same load, each as a hash-loop workload row.

- `maxtput.py`, the driver: `verity_vllm.LLM` compiled, vLLM's scheduler defaults for the card (max_num_seqs 1024,
  max_num_batched_tokens 16384, gpu_memory_utilization 0.9), prefix caching off, every request at t=0 in served_commit's mixed
  rule at `vllm bench serve`'s random lengths (1024 in, 128 out). Each arm runs in its own engine process, interleaved by cycle,
  on one KV pool, and its tokens are gated against stock's. Its docstring has the measures (the steady window, TTFT, TPOT, and
  the saturation evidence).
- `job.sh`, one run on one GPU: the Build pin (`served-pin seed/check`, as `served_commit/prof.sh`), an nvidia-smi log, the
  driver, then `row.py`.
- `row.py`: the run's result, one result per arm (`arm=maxtput:<arm>`) for `benchmarks/hash/loop/plot.py`, and `--table`, a
  row's published arms from the store.
- `launch.sh`: the three cards from the pushed HEAD through `research run`, then the per-arm results published and labeled.
  The arms are circuits' committer (`served_commit`'s arms, `harness/env.sh` and the committer's GPU tests), which lands
  separately, so a run goes from a branch that merges its iterations (`cursor/hash-commit-loop-15ad`) with this directory;
  `launch.sh` refuses a tree without them.
- `build_job.sh`: `served-match` alone, for a Build of record on a model that has none (run it from a tree whose `served-match`
  accepts the card).

```text
bash benchmarks/hash/maxtput/launch.sh 11                                        # Qwen3-8B, every card
MACHINES=vy-nebius-2 MAXTPUT_ARGS="--kv-bytes 58000000000" bash benchmarks/hash/maxtput/launch.sh 11
uv run --with matplotlib python3 benchmarks/hash/loop/plot.py --table             # the rows, from the store
python3 benchmarks/hash/maxtput/row.py --table --row 11                           # every arm's absolutes and saturation evidence
```

On the RTX PRO 6000 (96 GB) the full Commit doesn't fit beside stock's KV pool at 0.9, so its runs pin every engine's pool
lower (`--kv-bytes`), and a stock-only run at vLLM's own sizing gives the native number. A pinned B200 run takes its Build's
attention backend (`--attention-backend FLASH_ATTN`); vLLM's default there is FlashInfer.

`XCHECK=1` checks the driver's stock engine against vLLM's own entry point: `vllm bench throughput` at the same engine settings
on its random dataset at 1024/128 exactly, before and after the driver run on the same fixed load (`--load fixed`), compared over
the whole run (`driver_vs_vllm_bench_pct` in the run's result; the launch line is in `launch.sh`). vLLM's figure samples at
temperature 1 where the driver is greedy, and its timer starts before it adds the requests.
