# serve_bandwidth

Where a served mix's DRAM read bandwidth goes on the RTX PRO 6000: vLLM serving Llama-3.1 (8B on one GPU, 70B at TP 8)
in FP8 under `kv_filler/serve.py`'s bursty trace, sampled by nsys's GPU metrics (DRAM read and write throughput, about
10 kHz), with the CUDA kernels and one NVTX range an engine step.

- `mix.py`: the served mix, its steps named by what they schedule (prompt tokens, decodes, context).
- `timeline.py`: each sample's bytes by phase (`prefill`, `decode`, `host` between steps, `idle`) and kernel class
  (`gemm`, `attention`, `other`, `none`), against the card's measured read rate R, and the modeled registered bytes
  (weights a step, KV a context token) against the measured reads.
- `run.sh`: both under nsys on a node, once for each max-batched-tokens of `MBT`.

~~~bash
uv run research run --queue --gpus 1 --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
  --max-min 30 -- bash benchmarks/serve_bandwidth/run.sh
MODEL=/workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/<rev> TP=8 SECS=15  # the 70B, --gpus 8
~~~
