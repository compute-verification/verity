# The perplexity bench (`verity bench ppl`)

The perplexity bench is one standard bench for interventions on the matmuls of a served model: it measures what
perplexity a change to the linears or to attention costs, against bf16 and against plain int8, on fixed windows, seeds
and references, and reports nothing for a run whose gates fail. An intervention is one function, and an arm is a TOML
file that names a chain of them.

## Try it

On any machine, print the arms in `specs/`:

~~~sh
uv run verity bench ppl tables arms
~~~

It prints a Markdown table: each arm's scope, seeds, calibration, the plain int8 arm it is reported against, and its
description. The other commands, `run.py`'s docstring has in full:

~~~text
verity bench ppl submit --model llama-3.1-8b --set c4 --ctx 2048 adopted        # a `research run` on vy-nebius-2
verity bench ppl eval --model M --set S --ctx T --out DIR SPEC...                 # the same, on a GPU node
verity bench ppl report DIR/results.json...                                       # a run's tables
verity bench ppl baseline [DIR/results.json...]                                   # the locked baseline (and a check)
verity bench ppl steps | specs                                                    # what an arm can be made of
verity bench ppl tables [steps|arms|windows|models|baseline]...                   # this file's tables, as Markdown
verity bench ppl metrics DIR/results.json                                         # its metrics, through the kit
~~~

**On one H200**, from a clone, Llama 3.2 1B (the compute benchmark's model) on C4 at 2048, as shipped and with its
linears as Pearl-C's served kernel computes them (`pearl-c-linears`), reported as `metrics.toml`'s three metrics:

~~~text
research run --on NODE --gpus 1 --project verity --source . --cwd source --max-min 90 -- env WORK=/workspace bash benchmarks/ppl/h200.sh
~~~

`h200.sh` takes an idle H100 or H200 (under `gpu-lease` where the node has it), the venv and HF cache
`benchmarks/compute/serve.sh` uses (`$WORK/venv312`, bootstrapped as serve.sh does when absent; `$WORK/hf`, where the
checkpoint is fetched once, 2.5 GB), builds the sm_90a cubin (`benchmarks/harness/setup/pearl_c_sm90.sh`), fetches the
C4 shard from the hub into the same cache (its sha256 held to `datasets.json`'s), runs `run.py eval ... pearl-c-linears`
and then `run.py metrics`. `env GPU=I CORES=A-B` pins its lease to card I and those cores, for which the command drops
`--gpus 1` (research's own lease would take any card and leave the script none to pin).

`pearl-c-linears` is the compute benchmark's served arithmetic (`pearl-c-h100-v1-h3` in
`benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml`): its one step, `linear.pearl_c`, calls the
served engine's own kernel, `verity_torch.pouw.PearlCDeviceRun.gemm` on the sm_90a cubin, with x and W in BF16, so
every decoder linear is the FP8 E4M3 forming with Pearl's noise, the K32 chain and the noise peeled off that serving
runs; `lm_head` and attention stay bf16, as served (PoUW's executor sits at vLLM's `LinearBase` only). The bench keeps
q, k and v, and gate and up, as separate products, where vLLM fuses them: each output column's arithmetic is the same,
and the noise's seeds differ. No quantizer here is reused for it: the `fp` and `pc-fp8-*` arms scale by amax / 448 with
no noise and no peel, which is not Pearl-C. Its gate, `pearl_c`, holds every call of `linear.pearl_c` to the served
kernel's y being finite. `llama-3.2-1b` is `benchmarks/compute/load.toml`'s model at its revision, which a test holds
it to.

`run.py metrics RESULTS.json` (`report.metrics`) reports a passed run of `pearl-c-linears` through the benchmark kit:
`ppl.json`, and in the run's `result.json` `compute-accounting/perplexity-shipped` (the bf16 reference),
`compute-accounting/perplexity-pearl-c` (the mean over its seeds) and `compute-accounting/perplexity-cost-pearl-c`
(their difference in percent of bf16's), each with the model, its revision, the card, the set and the context as
fields. A run whose gates failed reports none, and a results file of another run is a reprint, which writes none.

This file holds no table: `run.py tables` (`tables.py`) prints each one from the code and data it describes, the
registered steps, the arms in `specs/`, the window lists, the models and the locked baseline, so a new step, arm,
model or locked point edits only its own file. `tests/benchmarks/ppl/test_run.py` fails on a table here.

## Adding an intervention

A new intervention on a matmul is one function. Put it in `steps.py`, or in any file passed with `--plugin FILE.py`:

~~~python
from spec import linear_step

@linear_step("clip", phase="real")
def clip(mm, nxt, *, sigmas: float = 6.0):
    """Clip each activation row at `sigmas` standard deviations before the quantizer."""
    lim = mm.x.std(dim=1, keepdim=True) * sigmas
    return nxt(mm.replace(x=mm.x.clamp(-lim, lim)))
~~~

and name it in a spec: `[[linear]]` `step = "clip"` `sigmas = 4.0`, before `int8`. The bench parses and type-checks its
parameters against the signature (keyword-only, after `mm, nxt`), runs it on every matmul it applies to, gates the
integer products after it, and reports the arm against bf16 and plain int8 like any other.

A step gets the matmul `mm` and `nxt`, the rest of the chain, and returns the product. Steps run in three phases, in
order:

1. `real` steps transform the real operands, and may post-process the product.
2. One `quantize` step puts both operands on the integer grid and scales the integer product the rest returns.
3. `integer` steps act on the integer operands.

The chain ends in the exact integer product, gated, or the FP32 product when there is no quantizer. What `mm` offers
(`matmul.MM`):

- `mm.x` and `mm.weight()`: the operands. W carries every earlier step's weight map, applied lazily.
- `mm.replace(x=..., w=f | tensor)`: new operands. A callable `w` maps the weight.
- `mm.integer(x, w)`: the quantizer's hand-off to the integer stage.
- `mm.cached(name, fn)`: once per matmul and chain prefix, shared across windows and arms (and across seeds with
  `coinless=True`, when fn reads none of the step's own coins).
- `mm.shared(key, fn)`: once per run.
- `mm.coins(shape)`: fresh bits from the arm's seed, the layer and the window.
- `mm.checking`: this is the check window. `mm.gate(name, ok)` records a gate; `mm.int_product(x, w)` is a gated exact product.
- `mm.kind`, `mm.name`, `mm.layer`, `mm.window`, `mm.seed`, `mm.extra`: where the matmul is. Attention's extra holds
  the KV group, the query rows, the causal mask, and `rep`, `s` and `d`.

Declare `seeded=True` (or a function of the parameters) when the step's effect depends on the seed, so the arm runs
every seed, and `calib=True` when it reads `mm.run.calib`. A `check=` function refuses bad parameters at parse time.
`source=` names what the step comes from (`run.py tables steps`' last column).
Register attention steps with `@attention_step(name, on=("qk", "pv"))`.

## Baseline

The adopted design (dither, w64 peel, psink, kc, SmoothQuant α 0.5, weights ±63), its halves, the design without
SmoothQuant, and the plain int8 arms, locked in `baseline.json` per point the bench has run. The lock's `tonight` is the
same arm as the harnesses the bench replaced measured it on the same windows (a layer-wise `linears_layerwise.py` on C4,
`attention.py` on WikiText-2), read from their runs' `results.jsonl`.
