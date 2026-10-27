# benchmarks/kv_lifetime

KV page lifetimes under a chat workload on one vLLM engine, and the KV-credit frontier they give under a lifetime cap L:
for each period T and cap L, the stock of encoded KV the bandwidth audit (PoUB, `verity/core/protocols/remote_bandwidth`)
would have to charge and the share of live KV it could credit, at each encode hardness E. `lifetimes.py`'s `Recorder`
logs each page's allocation, prefix hit, share and release in vLLM's page pool; `serve.py` serves the workload with it;
`analyze.py` turns the events and steps into the frontier's inputs; `frontier.py` prints the frontier; `encode_rate.py`
is `kv.cu`'s encode rate in 4 KiB chunks, the honest cost of appending and re-keying KV.

**Status: retire candidate.** `rg kv_lifetime` over `verity/`, `benchmarks/`, `examples/`, `tests/`, `integrations/`
and `Security/` finds nothing outside this directory: no importer, runner or citation. The audit states its lifetime cap
on its own (`audit.lifetime_windows`), and the suite declares no inputs, so only its own tests run it.

**Run:** `uv run tools/verity/check/suites.py benchmarks/kv_lifetime` tests the lifetime analysis and the frontier on
the CPU. `run.sh` runs `serve.py` and `analyze.py` on one GPU, and `encode_rate.py` runs under
`benchmarks/word_codec/run.sh` with `KV=1`.

**Reads and writes:** `run.sh` reads the ShareGPT V3 dataset and Llama-3.1-8B-Instruct from Hugging Face, and
`serve.py` writes the events and steps; `analyze.py` writes `analysis.json` beside them; `frontier.py` reads it and a
step model (`--models`) and prints a table, or writes JSON with `--json`. **Backs:** no Lean guarantee or claim.
