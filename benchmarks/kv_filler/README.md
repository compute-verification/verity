# The filler top-up beside vLLM serving

While vLLM serves, a top-up can read idle registered memory beside the serving workload. This benchmark measures how
close a period's registered reads then come to the read rate R (η, reads over R·T), and what the top-up costs
inference. Its numbers are
serving costs: it backs no Lean guarantee or claim.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/kv_filler
~~~

It runs what needs no GPU: the fill's reference, the lag's replay, the analysis, and the serving script's pure parts.

| Script | Needs | Writes into `--out` |
| --- | --- | --- |
| `bench.py` | one GPU, a build of `topup.cu` (`--topup-lib`) and one of `verity/protocols/resource_accounting/memory/poub/prover/word_codec/kv.cu` at p512 (`--kv-lib`); no script here builds the two | `bench.json`, with its log |
| `serve.py` | the same; it serves Llama-3.1-8B under a bursty trace with the top-up | `serve-<arm>.json`, with its log |
| `analyze.py` | those outputs | `conversion.json` |

[`DESIGN.md`](DESIGN.md) says what each file is and which benchmarks import or build this one.
