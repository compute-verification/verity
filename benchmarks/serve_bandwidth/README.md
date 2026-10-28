# serve_bandwidth

Where a served mix's DRAM read bandwidth goes on the RTX PRO 6000: vLLM serving Llama-3.1 (8B on one GPU, 70B at TP 8)
in FP8 under `kv_filler/serve.py`'s bursty trace, sampled by nsys's GPU metrics (DRAM read and write throughput, about
10 kHz), with the CUDA kernels and one NVTX range an engine step.

**Status: maintained.** Its suite runs in `check`, `budget.py` and `timeline.py` read `benchmarks/kv_filler`'s step
profiles and kernel classes, and `benchmarks/stacked_share/share.py` takes its served trace from `mix.py`'s `mix.json`.
Decided by `rg serve_bandwidth` over `verity/`, `benchmarks/`, `examples/`, `tests/` and `integrations/`.

- `mix.py`: the served mix, its steps named by what they schedule (prompt tokens, decodes, context).
- `timeline.py`: each sample's bytes by phase (`prefill`, `decode`, `gap` between steps, `idle`) and kernel class
  (`gemm`, `attention`, `other`, `none`; phases only when nsys traced no kernels, as for TP workers), against the
  card's measured read rate R, and the modeled registered bytes (weights a step, KV a context token) against the
  measured reads.
- `run.sh`: both under nsys on a node, once for each max-batched-tokens of `MBT`; `REDO=<a run's out/>` analyzes a
  run's kept reports again (CPU only); `SCRIPT=gemm_dtypes.py` times the linears in BF16, FP8 and int8 instead.
- `budget.py`: the encoded stack's added time over stock by term (weights, attention, KV encode, receipts), from
  `kv_filler/step_profile.py`'s `step_profile.json` of a stock run and encoded runs at the same shapes.

~~~bash
uv run research run --queue --gpus 1 --project verity --source . --cwd source --mem-gb 64 --max-min 30 \
  -- bash benchmarks/serve_bandwidth/run.sh
MODEL=/workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/<rev> TP=8 SECS=15  # the 70B, --gpus 8
~~~
