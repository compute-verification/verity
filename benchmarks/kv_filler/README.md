# benchmarks/kv_filler

The filler top-up beside vLLM serving: how close a period's registered reads come to the read rate R (η, reads over
R·T) when a top-up reads idle registered memory beside the serving workload, and what the top-up costs inference.
`topup.cu` is the top-up (it includes `benchmarks/codec_tensor/sha512.cuh`, readers in `readers.cu` and the fill in
`fill.cu`, whose reference is `fill_ref.py`). `bench.py` times it beside serving phases, `serve.py` serves Llama-3.1-8B
under a bursty trace with it.
`green.py` is the SM partition, and `lag.py` and `analyze.py` price the conversion lag.

**Status: active.** `benchmarks/stacked_share/share.py` and `benchmarks/vllm_encoded/bench_arm.py` import it (`green.py`,
the top-up), and `benchmarks/harness` builds `topup.cu`. Decided by `rg kv_filler` over `verity/`, `benchmarks/`,
`demos/`, `tests/` and `adapters/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/kv_filler` runs what needs no GPU: the fill's reference, the
lag's replay, the analysis, and the serving script's pure parts. `bench.py` and `serve.py` run on one GPU with a build
of `topup.cu` and one of `benchmarks/word_codec/kv.cu` at p512 (`--topup-lib`, `--kv-lib`); no script here builds the
two.

**Reads and writes:** `bench.py` writes `bench.json` and `serve.py` `serve-<arm>.json`, each with its log, into `--out`;
`analyze.py` reads those into `conversion.json`. **Backs:** no Lean guarantee or claim; its numbers are serving costs.
