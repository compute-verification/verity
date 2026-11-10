# benchmarks/kv_filler

The filler top-up beside vLLM serving: how close a period's registered reads come to the read rate R (η, reads over
R·T) when a top-up reads idle registered memory beside the serving workload, and what the top-up costs inference.
`topup.cu` is the top-up (it includes `benchmarks/codec_tensor/sha512.cuh`, readers in `readers.cu` and the fill in
`fill.cu`, whose reference is `fill_ref.py`). `bench.py` times it beside serving phases.
`plan.py` is the filler a pass of the bandwidth audit (PoUB, `verity/core/protocols/remote_bandwidth`) reads in the vLLM
integration's store, `green.py` the SM partition, and `lag.py`, `analyze.py` and `eta.py` price the conversion lag and η.

**Status: active.** `archive/benchmarks/stacked_share/share.py` and `benchmarks/vllm_encoded/bench_arm.py` import it (`green.py`,
the top-up), and `benchmarks/harness` builds `topup.cu`. Decided by `rg kv_filler` over `verity/`, `benchmarks/`,
`examples/`, `tests/` and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/kv_filler` runs what needs no GPU: the fill's reference, the
lag's replay, the filler plan and the analysis.

**Reads and writes:** `bench.py` writes `bench.json` with its log into `--out`; `analyze.py` writes
`conversion.json`, and `eta.py` `eta.json`. **Backs:** no Lean guarantee or claim; its numbers are serving costs.
