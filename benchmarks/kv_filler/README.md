# benchmarks/kv_filler

The filler top-up beside vLLM serving: how close a period's registered reads come to the read rate R (η, reads over
R·T) when a top-up reads idle registered memory beside the serving workload, and what the top-up costs inference.
`topup.cu` is the top-up (it includes `benchmarks/bandwidth/tagger.cu`, readers in `readers.cu` and the fill in
`fill.cu`, whose reference is `fill_ref.py`). `bench.py` times it beside serving phases, `serve.py` serves Llama-3.1-8B
under a bursty trace with it, `step_profile.py` profiles one decode step, and `fill.py` measures the readers' rate.
`plan.py` is the filler a pass of the bandwidth audit (PoUB, `verity/core/protocols/remote_bandwidth`) reads in the vLLM
integration's store, `green.py` the SM partition, and `lag.py`, `analyze.py` and `eta.py` price the conversion lag and η.

**Status: active.** `benchmarks/serve_bandwidth` imports it (its suite's inputs and `sys.path`), so do
`benchmarks/stacked_share/share.py` and `benchmarks/vllm_encoded/bench_arm.py` (`green.py`, the top-up), and
`benchmarks/harness` builds `topup.cu`. Decided by `rg kv_filler` over `verity/`, `benchmarks/`, `examples/`, `tests/`
and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/kv_filler` runs what needs no GPU: the fill's reference, the
lag's replay, the filler plan, the analysis, and the serving and profiling scripts' pure parts. `run.sh` builds `topup.cu` and
`benchmarks/word_codec/kv.cu` and runs `SCRIPT` (default `bench.py`) on one GPU.

**Reads and writes:** `bench.py` writes `bench.json`, `serve.py` `serve-<arm>.json`, `step_profile.py`
`step_profile.json` and `fill.py` `fill-<cmd>.json`, each with its log, into `--out`; `analyze.py` reads those into
`conversion.json`, and `eta.py` into `eta.json`. **Backs:** no Lean guarantee or claim; its numbers are serving costs.
