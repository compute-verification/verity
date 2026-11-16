# tc_probe_fp4: Blackwell's FP4 and FP8 tensor-core instructions against their models

This is the Blackwell counterpart of [`tc_probe`](../tc_probe/README.md). Each probe runs an instruction on its GPU,
raw bits in and out, and compares every output word with the instruction's pinned model in `verity/circuits/silicon`
(`BLACKWELL_SM120_NVF4`, `BLACKWELL_SM120_MXF4`, `GroupSum`). Where no model is pinned yet, a capture decides between
named candidates. Every mismatch is recorded, and nothing is tuned to fit.

Beside the probes, price microbenchmarks time each multiplier against dense NVFP4. `f1_rates.py`: every sm_120a
multiplier priced per nonzero product against dense NVFP4, the measurement behind the premise `fp4-tile-only/sm120`.

Everything runs on sm_120a (RTX 5090, RTX PRO 6000) unless its file says sm_100a (B200). Each file's docstring says
what it probes or prices, and `probe.py`'s lists every instruction it takes.

## Try it

~~~text
uv run python benchmarks/tc_probe_fp4/w1_price.py --out DIR --fake-silicon                 # CPU dry run
~~~

This runs the price microbenchmark's driver against fake silicon, with no GPU, in seconds: it prints a rate for each
row and writes `DIR/w1_price.json`. On the card itself, from the repository root after `uv sync --all-packages`, with nvcc (and
`cuobjdump` for `--sass-check`):

~~~text
uv run python benchmarks/tc_probe_fp4/probe.py --instruction nvf4 --out DIR --sass-check --sweep --n-random 4096
uv run python benchmarks/tc_probe_fp4/w1_price.py --out DIR --expect-uuid GPU-UUID
uv run python benchmarks/tc_probe_fp4/f1_rates.py build --out DIR                          # ptxas, the timed library, SASS gates
OUT=DIR bash benchmarks/tc_probe_fp4/f8_sm100.sh                                           # on a B200
~~~

`tool.py` holds the store declarations (`tc_probe_fp4`, `tc_price_fp4`, `tc_rates_fp4`), which say which parameters
key a result; the evidence store's tool registry loads it by name. The tests, under `tests/benchmarks/tc_probe_fp4/`,
are the CPU side, which builds and runs no kernel.
