# The perplexity bench (`verity bench ppl`)

One standard bench for interventions on the matmuls of a served model: what perplexity a change to the linears or to
attention costs, against bf16 and against plain int8, on fixed windows, seeds and references. `run.py` is the runner;
its docstring lists the commands.

~~~text
verity bench ppl submit --model llama-3.1-8b --set c4 --ctx 2048 adopted        # a `research run` on vy-nebius-2
verity bench ppl eval --model M --set S --ctx T --out DIR SPEC...                 # the same, on a GPU node
verity bench ppl eval ... --device cpu --part I/N --out DIR SPEC...               # part I of N windows, on a CPU's cores
verity bench ppl merge --out DIR PART_DIR...                                      # the parts as one run's results.json
verity bench ppl report DIR/results.json...                                       # a run's tables
verity bench ppl baseline [DIR/results.json...]                                   # the locked baseline (and a check)
verity bench ppl steps | specs                                                    # what an arm can be made of
verity bench ppl tables [steps|arms|windows|models|baseline]...                   # this file's tables, as Markdown
~~~

This file holds no table: `run.py tables` (`tables.py`) prints each one from the code and data it describes, the
registered steps, the arms in `specs/`, the window lists, the models and the locked baseline, so a new step, arm,
model or locked point edits only its own file. `tests/test_run.py` fails on a table here.

## Methodology

Fixed in `methodology.py`; `run.py eval` refuses any other window list, context or seed.

- **Eval sets.** Held-out C4 validation (`allenai/c4` `en/c4-validation.00000-of-00008.json.gz`, its first 3000
  documents) and WikiText-2 test (`wikitext-2-raw-v1`, its 4358 lines). `datasets.json` pins both files by sha256 and
  evidence-store id; `run.py submit` fetches them and sends them with the run.
- **Windows.** A set's documents joined by blank lines, tokenized without special tokens, cut into consecutive windows of
  T tokens, each led by the tokenizer's BOS if it has one (Llama's does, Qwen2.5's doesn't). Window i is tokens
  `[i * body, (i + 1) * body)` of the stream. `windows.lock.json` pins every list's sha256 per tokenizer, and a run
  refuses another.

  `run.py tables windows` lists each context's windows and scored tokens.

  Every position of a window but the first is scored. Perplexity is exp of the mean NLL over all scored tokens.
- **Calibration** (SmoothQuant's channel maxima, GPTQ's second moments): C4 windows 100–115 at 2048, the bf16 stream's
  inputs to each layer. They are disjoint from every eval window at every context.
- **Seeds.** An arm whose result depends on its coins (dither, random rotations, peels) runs seeds 0, 1 and 2. A
  deterministic arm runs once.
- **The reference** is the bf16 model with PyTorch SDPA attention (`F.scaled_dot_product_attention`, causal, GQA),
  never eager: HF's eager bf16 path rounds QKᵀ to bf16 before the scale, and Qwen2.5's q and k biases make the scores
  large, so its eager bf16 is 15% worse (WikiText-2 at 2K: 7.300 against SDPA's 6.355 and FP32's 6.354).
  The engine's bf16 layers are HF's `LlamaDecoderLayer` / `Qwen2DecoderLayer` with SDPA op for op, with HF's RoPE
  tables (computed on the CPU), and the `hf_reference` gate checks both bit for bit on every layer. TF32 is off.
- **Batching.** The bf16 reference and calibration streams run 4 consecutive windows per forward. Arms run one window
  per forward, since their coins are drawn per window.
- **The report** (`report.py`), per (model, set, context): each arm's perplexity as the mean over its seeds with the
  seeds' range, its Δ% against bf16 and against its scope's plain int8 arm, and the significance rule's verdict.
- **Plain int8**, for each scope: `int8-linears` (rotated, one scale per token and per output row, rounded to ±127),
  `int8-attention` (rotated, one limb, Q, K, P and V rounded) and `int8-all` (both). Every run includes the plain arm
  of each spec's scope.
- **Significance rule** (`methodology.compare`). Arm A is better than arm B (`<`) only when both hold:
  - their seed ranges of perplexity are disjoint;
  - a paired sign test over windows, on each window's seed-mean NLL, favours A at one-sided α = 0.025: at least 13 of
    16 windows.

  With fewer than 6 windows (32K has 4), the test needs every window, and the verdict is marked weak. Otherwise the
  verdict is `≈`. Seeds vary the coins and windows vary the text, so a verdict needs both to agree.

## Interventions: the spec

An arm is a TOML file: a list of steps for the linears (`[[linear]]`: q, k, v, o, gate, up and down, each y = x Wᵀ) and
one for attention (`[[attention]]`: S = Q Kᵀ and O = P V). Each step names a registered function and its parameters.
`layers = [...]` and, on linears, `linears = [...]` restrict where a step applies. The output head (`lm_head`) takes
only the linear steps that name it in `linears`, without calibration or `layers`. `specs/adopted.toml`:

~~~toml
description = "the adopted design end to end"

[[linear]]
step = "smooth"            # SmoothQuant, alpha 0.5, o_proj's factor shared per KV group (foldable)
alpha = 0.5
kv_shared = true

[[linear]]
step = "rotate"            # one orthogonal Q per k on x and W

[[linear]]
step = "int8"              # activations dithered, weights on ±63
act = "dither"
weight_top = 63

[[attention]]
step = "rotate"            # a random Hadamard R of the head dimension

[[attention]]
step = "center_keys"       # kc: K minus its per-channel mean

[[attention]]
step = "int8"              # one limb; K and V dithered; psink: key 0 its own P block
q_top = 63
k = "dither"
v = "dither"
p_sink = true

[[attention]]
step = "peel"              # w64 peel on Q
operand = "q"
width = 64

[[attention]]
step = "peel"              # w64 peel on P, served as P - 64 + E
operand = "p"
width = 64
offset = 64
~~~

`run.py tables steps` lists the built-in steps: each one's parameters, phase, the matmuls it acts on, its docstring's
first paragraph and its source (`source=` where it is registered). `run.py steps` prints
their parameters and docs.

`run.py tables arms` lists every arm in `specs/`: its scope, whether it runs every seed, whether it reads calibration,
the plain int8 arm it is reported against, and its `description`. A new arm is its file, and its `description` says
what it is.

`specs/` holds the three plain int8 arms, the adopted design (`adopted`) with its halves (`adopted-linears`,
`adopted-attention`) and `adopted-no-smooth`, and the variants measured against them. A design in the design registry
(`benchmarks/pouw/machine_model/data/designs/`) names its arm as `ppl`. `attention.int8`'s `softmax = "lut"` is the
integer softmax of `benchmarks/pouw/edited_gemm/fattn.cu`, which `steps.lut_softmax` states. `dither_plot.py` draws
perplexity against the dither's size per side choice, and `frontier_plot.py` perplexity against core ε
(`machine-model eps --checking full`) under each reading of the table route, each with its table. A SPEC is an arm's
name in `specs/` or, with its suffix, any TOML file's path.

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

## Gates

Built in, and a run whose gates fail reports nothing: `results.json` holds the gates and no perplexity, and `run.py`
exits 1.

- `int8_range`: every integer product's operands within int8, on every call.
- `grid_range`: instead of `int8_range`, for a quantizer that declares a wider grid (block-scaled E2M1, ±180): every
  operand within that bound, on every call.
- `accumulator`: each product's s32 accumulator, bounded a priori by k max|x| max|w|, on every call (a wider grid's
  product, FP32 chunks summed in float64, its wide operand split into base-16 limbs when chunks would be short: within
  2^53).
- `emulation`: every call of `emulate`: each limb plane exact in its format and the recombined product equal to the
  int8 one it replaces.
- `product_exact`: on the check window (window 0), every layer, each integer product against a float64 recomputation
  on 32 rows (linears) or 64 (attention).
- `peel`: on the check window, every layer, KV group and P block (and every linear with `linear.peel`), each peel
  identity. The served product, peeled, must
  equal the direct one exactly.
- `lut`: for the integer softmax, on every call, A within uint32 and m − S within 32 bits (fattn.cu's `umulhi`).
- `hf_reference`: on the check window's batch, every layer, the engine's bf16 layer output and RoPE tables bit-equal
  to HF transformers' own decoder layer on the same input. A run where this never ran fails.
- `windows` and `tokens`: the window lists are the lock's, and every stream scored every token.

## Models

`models.py` pins each snapshot by revision under the node's HF cache (`run.py tables models` lists them).

GLM-5.3 (`glm.py`) is read from its block-FP8 checkpoint, streamed: `pins/glm-5.3.json` pins every file's sha256, and
each safetensors shard is pulled from the hub when a layer needs it, checked and deleted once read (`stream.py`), so
its 705 GB need no copy on the node. Its bf16 reference is the FP8 weights dequantized (codes times their 128 × 128
scales, in FP32, rounded to bf16) with HF's `GlmMoeDsaDecoderLayer` op for op (MLA, the sigmoid router in FP32, eager
experts), checked by `hf_reference`. It runs at 2048 only: DSA's indexer keeps index_topk = 2048 keys per query, so at
T ≤ 2048 attention is dense causal, and longer windows are refused. Arms reach its linears, every routed expert's
included, on the rows routed to it; attention stays bf16. `fp8-block` is the checkpoint as vLLM serves it (its own
E4M3 weights, activations per token and group of 128), and `int8-channel127-v3` the int8 W8A8 linears H1 serves, as
recipe `verity-int8-channel127/v3` makes them: each weight's row codes from its exact float64 dequant (`int8.exact`),
the LM head int8 too, and kv_b_proj bf16.

The engine (`engine.py`, a layer-wise evaluation) reads one layer's weights at a time from the safetensors with
its own reader. Every stream of every window, the reference, calibration and each (arm, seed), goes through that layer
before the next is read. So a model of any size fits one GPU, every arm shares one pass over the weights, and
`accelerate` is never used. More GPUs split the windows: contiguous groups of 4, one process per leased GPU, merged
after. A sharded run gives each window the NLL a single-GPU run gives it. `eval --part I/N` runs one such part alone,
as a run of its own (a small model's on a CPU node's cores, `--device cpu`), and `merge` makes the parts one results.json.

## Results

`run.py submit` runs the eval through `research run` on vy-nebius-2, under a gpu-lease of the model's GPUs (at most 2),
reading clocks and never setting them. The run's `out/` holds `results.json` (schema `ppl-bench/v1`: per arm and seed
the perplexity, mean NLL and per-window NLLs, the gates, divergence from the reference), `nll.npz` (every scored
token's NLL), `table.md`, the shard logs and `gpu.txt`. The run publishes itself as an Attempt; cite it by run id, and
put findings about it on labels (`research data label`).

## Baseline

The adopted design (dither, w64 peel, psink, kc, SmoothQuant α 0.5, weights ±63), its halves, the design without
SmoothQuant, and the plain int8 arms, locked in `baseline.json` per point the bench has run. The lock's `tonight` is the
same arm as the harnesses the bench replaced measured it on the same windows (a layer-wise `linears_layerwise.py` on C4,
`attention.py` on WikiText-2), read from their runs' `results.jsonl`.

Two measurements of an arm reproduce each other when their seed ranges, widened by 0.1% (`methodology.reproduces`,
`REPRODUCE_FLOOR`), overlap. `run.py baseline RESULTS...` checks new results against the lock that way, and
`run.py baseline --lock RESULTS...` replaces their points in `baseline.json`. `run.py tables baseline` (or `run.py
baseline` alone) prints the lock as a table: per point and arm, the bench's perplexity with its seeds' range, its Δ%
against bf16 and against plain int8, the earlier harnesses' measurement and method, their Δ%, whether they reproduce,
and each point's runs. The floor covers what moves a deterministic arm between harnesses:

- RoPE tables. The earlier harnesses built them on the GPU (`LlamaRotaryEmbedding(config, device=cuda)`), whose
  `inv_freq` differs from the CPU's in 5 of 64 entries. HF's `from_pretrained` computes them on the CPU, as the bench
  does. HF's own `LlamaForCausalLM` (SDPA, 4 windows per forward) is the bench's bf16 end to end on Llama C4 2K, 9.15834,
  and with GPU-built tables it is the earlier harnesses' 9.15903.
- Kernel shapes. A bf16 GEMM's rounding depends on its rows: the same model at 1 window per forward is 9.16174
  (+0.04%). The bench's arms run 1 window per forward; the earlier linears harness ran 4.
- The coins. The earlier linears harness drew a linear's coins per batch of 4 windows, the bench per window, so seeded
  arms agree in distribution, not bit for bit.
- The reference attention under a linears-only arm: the earlier attention harness's was eager (Llama) or FP32
  (Qwen2.5), marked.

## Tests

`uv run tools/verity/check/suites.py benchmarks/ppl` runs the CPU tests on tiny random Llama, Qwen2 and block-FP8 GLM models: spec
parsing, the window recipe and lock, the significance rule, the engine's bf16 stream against an independent float64
forward, every gate on the adopted arms, gates that fail on sabotage reporting nothing, sharding, seeds, and HF parity
where transformers is importable.
