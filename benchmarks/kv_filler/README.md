# benchmarks/kv_filler

The filler top-up beside vLLM serving: how close a period's registered reads come to the read rate R (η, reads over
R·T) when a top-up reads idle registered memory beside the serving workload, and what the top-up costs inference.
`topup.cu` is the top-up (it includes `benchmarks/codec_tensor/sha512.cuh`, readers in `readers.cu` and the fill in
`fill.cu`, whose reference is `fill_ref.py`). `bench.py` times it beside serving phases, `serve.py` serves Llama-3.1-8B
under a bursty trace with it, `step_profile.py` profiles one decode step, and `fill.py` measures the readers' rate.
`plan.py` is the filler a pass of the bandwidth audit (PoUB, `verity/protocols/poub/verifier`) reads in the vLLM
integration's store, `green.py` the SM partition, and `lag.py`, `analyze.py` and `eta.py` price the conversion lag and η.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/kv_filler` | The suite, on the CPU: the fill's reference, the lag's replay, the filler plan, the analysis, and the pure parts of the serving and profiling scripts. It writes no result. |
| `bash benchmarks/kv_filler/run.sh`, with `SCRIPT`, `ARGS`, `MAXMIN` and `ENV_DIR` | The benchmark on one GPU: it builds the top-up (`topup.cu`) and the KV codec (`benchmarks/word_codec/kv.cu`), runs the tests, then runs `SCRIPT` (`bench.py` by default) with `ARGS` under a lease of one GPU. It writes to `$RESEARCH_RUN_DIR/out`. |
| `topup.cu`, built as a shared library | The top-up, with the fill (`fill.cu`) and the readers (`readers.cu`) it includes. `bench.py`'s `Topup(path: str)` opens the library: `start(...)` runs the top-up thread over the given regions, `stop()` ends it and `log()` returns what it read. `mode_of(name: str) -> int` maps a mode's name to its number. |
| `green.partition(tu, torch, fill_sms: int)` and `green.spread_checks(tu, torch, step, new_stream, rec: dict) -> dict` (`green.py`) | The SM partition: the top-up on `fill_sms` SMs of its own and the engine's step on the rest, and the record of which SMs each stream's kernels ran on. |

Read from other benchmarks, and from nothing outside `benchmarks/`: the bandwidth arms of `benchmarks/vllm_encoded/`
import `green.py` and `bench.py`'s `Topup`, `mode_of`, `FILL_WARPS` and `RUN_SEED`, and `benchmarks/harness/` builds
`topup.cu` for them.

**Status: active.** `benchmarks/stacked_share/share.py` and `benchmarks/vllm_encoded/bench_arm.py` import it (`green.py`,
the top-up), and `benchmarks/harness` builds `topup.cu`. Decided by `rg kv_filler` over `verity/`, `benchmarks/`,
`workloads/`, `tests/` and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/kv_filler` runs what needs no GPU: the fill's reference, the
lag's replay, the filler plan, the analysis, and the serving and profiling scripts' pure parts. `run.sh` builds `topup.cu` and
`benchmarks/word_codec/kv.cu` and runs `SCRIPT` (default `bench.py`) on one GPU.

**Reads and writes:** `bench.py` writes `bench.json`, `serve.py` `serve-<arm>.json`, `step_profile.py`
`step_profile.json` and `fill.py` `fill-<cmd>.json`, each with its log, into `--out`; `analyze.py` reads those into
`conversion.json`, and `eta.py` into `eta.json`. **Backs:** no Lean guarantee or claim; its numbers are serving costs.
