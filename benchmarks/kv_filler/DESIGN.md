# benchmarks/kv_filler: design

`topup.cu` is the top-up (it includes `verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/sha512.cuh`, readers in `readers.cu` and the fill in
`fill.cu`, whose reference is `fill_ref.py`). `bench.py` times it beside serving phases, `serve.py` serves Llama-3.1-8B
under a bursty trace with it.
`green.py` is the SM partition, and `lag.py` and `analyze.py` price the conversion lag.

## Who builds and reads it

**Status: active.** `benchmarks/stacked_share/share.py` and `benchmarks/vllm_encoded/bench_arm.py` import it (`green.py`,
the top-up), and `benchmarks/harness` builds `topup.cu`. Decided by `rg kv_filler` over `verity/`, `benchmarks/`,
`demos/`, `tests/` and `adapters/`.
