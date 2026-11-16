# prover/cost: design

What the budget model's B counts, and how a class circuit's size and its window's commit are priced. `README.md` says
what the cost model is, and each script's docstring what it plans and its flags.

## What B counts

B is the budget model's unit, GPU-seconds of overhead per plain GPU-second over a window, and it is the serve-time
total (everything is committed at serve time): every
scale's serve-time commitments under the PoUW scheme (`--scheme`: `floor`, the measured s8 A-operand floor alone;
`h1`, `h1-bf16`, `pearl-c`, each scheme's rows a token a layer) and the store-block receipts of
PoUB, which every plan pays; then proving and the verifier's CPU (at its price ratio) for the sampled proof blocks,
the commitment of every proof block's outputs, replay of the requests the sample meets (serving keeps only the requests' tokens and
the weights, so each is re-run whole from its prompt to build its sampled proof blocks' witnesses), and the zero-work subcircuits at
their floor.

## The class circuit's size and its window's commit

`../audit/class_size.py` sizes a class circuit in ANDs and Flock blocks, and `benchmarks/prove_path/class_rows.py` counts its rows by what fills them and
their SHA-512 compressions (`--probe S,NP,O` lowers one Program at a small class). `window_cost.py` prices a
window's commit from that census, or from a probe projected to a class (`--project 8,1024,64`, which reproduces the exact
census proof block for proof block), against its target of 0.20 to 0.25 of the served window. It reports in ANDs (58,120 a
compression) and in seconds (the window's rows on the GPU's hashing rates, the docket on the CPU's, which
`--measure-cpu` measures). It covers each lever alone and the plan stacked, the pad pass at three traffics, the
docket's bytes and files, and totals over a week, a month and a year. With `--stream SIZING` it also prices serving's
own stream commit (rows-sha512, made while it serves, before the window's commit) and its seal, from serving's stream of
record (`native_host.schedule_stream_bound`), whole and without the attention interiors (`fa*.m1`, lever F), which nothing
reads after the seal:

~~~text
python benchmarks/prove_path/window_cost.py PROBE.json --target h200 --project 8,1024,64 \
    [--measure-cpu SCRATCH] [--stream tests/adapters/vllm/tests/pipeline/fixtures/admission/b16.json] [--out FILE]
~~~
