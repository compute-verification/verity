# The perplexity bench (`verity bench ppl`)

One standard bench for interventions on the matmuls of a served model: what perplexity a change to the linears or to
attention costs, against bf16 and against plain int8, on fixed windows, seeds and references. It joins tonight's three
harnesses (16 Oct: D4-PPL's linears, D4-ATTN's attention, D4-70B's layer-wise `linears_layerwise.py`) into one.
`run.py` is the runner; its docstring lists the commands.

~~~text
verity bench ppl submit --model llama-3.1-8b --set c4 --ctx 2048 adopted        # a `research run` on vy-nebius-2
verity bench ppl eval --model M --set S --ctx T --out DIR SPEC...                 # the same, on a GPU node
verity bench ppl report DIR/results.json...                                       # the tables
verity bench ppl baseline [DIR/results.json...]                                   # the locked baseline (and a check)
verity bench ppl steps | specs                                                    # what an arm can be made of
~~~

## Methodology

Fixed in `methodology.py`; `run.py eval` refuses any other window list, context or seed.

- **Eval sets.** Held-out C4 validation (`allenai/c4` `en/c4-validation.00000-of-00008.json.gz`, its first 3000
  documents) and WikiText-2 test (`wikitext-2-raw-v1`, its 4358 lines). `datasets.json` pins both files by sha256 and
  evidence-store id; `run.py submit` fetches them and sends them with the run.
- **Windows.** A set's documents joined by blank lines, tokenized without special tokens, cut into consecutive windows of
  T tokens, each led by the tokenizer's BOS if it has one (Llama's does, Qwen2.5's doesn't). Window i is tokens
  `[i * body, (i + 1) * body)` of the stream. `windows.lock.json` pins every list's sha256 per tokenizer, and a run
  refuses another; Llama's C4 lists at 2K are tonight's `c4val2048x16.npy` and `c4cal2048x16.npy` byte for byte.

  | context | windows | scored tokens (per arm and seed) |
  |---:|---:|---:|
  | 2048 | 0–15 | 32,752 |
  | 8192 | 0–15 | 131,056 |
  | 32768 | 0–3 | 131,068 |

  Every position of a window but the first is scored. Perplexity is exp of the mean NLL over all scored tokens.
- **Calibration** (SmoothQuant's channel maxima, GPTQ's second moments): C4 windows 100–115 at 2048, the bf16 stream's
  inputs to each layer. They are disjoint from every eval window at every context.
- **Seeds.** An arm whose result depends on its coins (dither, random rotations, peels) runs seeds 0, 1 and 2. A
  deterministic arm runs once.
- **The reference** is the bf16 model with PyTorch SDPA attention (`F.scaled_dot_product_attention`, causal, GQA),
  never eager: HF's eager bf16 path rounds QKᵀ to bf16 before the scale, and Qwen2.5's q and k biases make the scores
  large, so its eager bf16 is 15% worse (WikiText-2 at 2K: 7.300 against SDPA's 6.355 and FP32's 6.354, D4-ATTN).
  The engine's bf16 layers are HF's `LlamaDecoderLayer` / `Qwen2DecoderLayer` with SDPA op for op, with HF's RoPE
  tables (computed on the CPU), and the `hf_reference` gate checks both bit for bit on every layer. TF32 is off.
- **Batching.** The bf16 reference and calibration streams run 4 consecutive windows per forward, as tonight's
  harnesses did. Arms run one window per forward, since their coins are drawn per window.
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
`layers = [...]` and, on linears, `linears = [...]` restrict where a step applies. `specs/adopted.toml`:

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

The built-in steps (`run.py steps` prints their parameters and docs):

| step | phase | what it does | from |
|---|---|---|---|
| `linear.smooth(alpha, kv_shared)` | real | SmoothQuant, x / m and W m; `kv_shared` is the foldable variant | D4-70B |
| `linear.rotate(kind)` | real | x Q and W Q; `q` is exact_grid's rotation, `h` the flat Hadamard | D4-PPL |
| `linear.int8(act, act_top, weight_top, group, gptq, density)` | quantize | activations per token, rounded or dithered (`density` < 1: dithered on a coin-chosen fraction of every k64 word's positions, rounded elsewhere); weights per row or per `group` inputs, capped at `weight_top`, rounded or GPTQ | D4-PPL, D4-70B, LOWBIT-DESIGN |
| `attention.rotate()` | real | Q R, K R, V R and the output R⁻¹ | D4-ATTN |
| `attention.center_keys()` | real | kc | D4-ATTN |
| `attention.int8(q_top, k, k_top, p_top, p_block, p_sink, v, v_top, v_scale)` | quantize | one limb; K and V rounded or dithered; P per row or per key block, `p_sink` | D4-ATTN |
| `attention.peel(operand, width, offset)` | integer | low-rank noise on Q or P, peeled exactly | D4-ATTN |
| `linear.grid(grid, act, weight, block, scales, density)` | quantize | both operands on a low-bit grid (`fp4c`, `fp4`, `int5`), per token and per row, each rounded, dithered or `sr`; `block = 16` adds NVFP4's integer block scales over k (`scales`: `odd`, the UE4M3 odd integers 1–15, or `int`); `density` < 1 coins only a coin-chosen fraction of every k64 word's positions | D4-LOWBIT, LOWBIT-DESIGN |
| `linear.peel(alphabet, width, top)` | integer | the int8 design's weight-side coin: W + U served, U low-rank (period 32 in k, {0,1} or ±1..±width), peeled exactly and gated on the check window, with \|W + U\| ≤ top | LOWBIT-DESIGN |
| `attention.grid(grid, q, k, p, v, p_sink)` | quantize | Q, K, P and V on a low-bit grid, each rounded or dithered (P's bits only where key ≤ query) | D4-LOWBIT |

`specs/` holds the arms in use: the three plain int8 arms, and the adopted design (`adopted`) with its halves
(`adopted-linears`, tonight's `d63-sm50-smkv`; `adopted-attention`, tonight's `int8-rot-eqp=w64-poff64-psink-kc-kv=dith`)
and `adopted-no-smooth` (tonight's WikiText end to end). The low-bit arms put the adopted design's linears (SmoothQuant
0.5, rotated) on a grid, as integers: `fp4c` is E2M1's contiguous region (±2 in steps of 0.5, times 2: −4..4), `fp4`
E2M1's every value (times 2: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12) and `int5` the integers −16..16 that E4M3 holds exactly.
`GRID-MODE` (`fp4c-dither`, …, `int5-round`) keeps the adopted int8 attention, and `GRID-dither-act` dithers only the
activations, rounding the weights as the adopted linears do; `GRID-MODE-all` puts attention on the
grid too, with no key blocks; `fp4c-dither-all-kv` dithers only K and V there, as the adopted attention does, with
`p_sink`. A dithered operand serves the level below x / s or the one above it by a fresh fair bit (on a uniform grid,
floor(x / s) + b), the weights' bits drawn once per seed and layer; `GRID-sr` rounds the linears' operands up with
probability the fraction instead (unbiased, a uniform coin). `nvfp4-MODE` puts the full E2M1 grid in blocks of 16
with odd UE4M3 block scales (the integer operands q·σ reach 180, past int8, so their product takes the exact FP32-chunk
route under the `grid_range` gate), `nvfp4-int-MODE` the same with every integer scale 1–15, `nvfp4c-MODE` the
contiguous region in the same blocks, `nvfp4-qN`, `nvfp4c-qN` and `int5-qN` dither a coin-chosen 1/N of every k64
word's positions on both operands, and `fp8-peel-wW[-gptq]` is the int8 design on E4M3's exact integers: activations
dithered on −16..16, weights on ±W (rounded or GPTQ) served as W + U and peeled; `fp8-peel-w15-qN` dithers the
activations on 1/N of the positions only (U still coins every position).
Any other file passes as a path.

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
- `mm.cached(name, fn)`: once per matmul and chain prefix, shared across windows and arms.
- `mm.shared(key, fn)`: once per run.
- `mm.coins(shape)`: fresh bits from the arm's seed, the layer and the window.
- `mm.checking`: this is the check window. `mm.gate(name, ok)` records a gate; `mm.int_product(x, w)` is a gated exact product.
- `mm.kind`, `mm.name`, `mm.layer`, `mm.window`, `mm.seed`, `mm.extra`: where the matmul is. Attention's extra holds
  the KV group, the query rows, the causal mask, and `rep`, `s` and `d`.

Declare `seeded=True` (or a function of the parameters) when the step's effect depends on the seed, so the arm runs
every seed, and `calib=True` when it reads `mm.run.calib`. A `check=` function refuses bad parameters at parse time.
Register attention steps with `@attention_step(name, on=("qk", "pv"))`.

## Gates

Built in, and a run whose gates fail reports nothing: `results.json` holds the gates and no perplexity, and `run.py`
exits 1.

- `int8_range`: every integer product's operands within int8, on every call.
- `grid_range`: instead of `int8_range`, for a quantizer that declares a wider grid (block-scaled E2M1, ±180): every
  operand within that bound, on every call.
- `accumulator`: each product's s32 accumulator, bounded a priori by k max|x| max|w|, on every call.
- `product_exact`: on the check window (window 0), every layer, each integer product against a float64 recomputation
  on 32 rows (linears) or 64 (attention).
- `peel`: on the check window, every layer, KV group and P block (and every linear with `linear.peel`), each peel
  identity. The served product, peeled, must
  equal the direct one exactly.
- `hf_reference`: on the check window's batch, every layer, the engine's bf16 layer output and RoPE tables bit-equal
  to HF transformers' own decoder layer on the same input. A run where this never ran fails.
- `windows` and `tokens`: the window lists are the lock's, and every stream scored every token.

## Models

`models.py` pins each snapshot by revision under the node's HF cache:

| model | snapshot | GPUs |
|---|---|---:|
| `llama-3.1-8b` | `unsloth/Llama-3.1-8B` @ `3f0d51f8` | 1 |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B` @ `d1497293` | 1 |
| `llama-3.1-70b` | `unsloth/Meta-Llama-3.1-70B` @ `1b730665` | 2 |

The engine (`engine.py`, tonight's layer-wise evaluation) reads one layer's weights at a time from the safetensors with
its own reader. Every stream of every window, the reference, calibration and each (arm, seed), goes through that layer
before the next is read. So a model of any size fits one GPU, every arm shares one pass over the weights, and
`accelerate` is never used. More GPUs split the windows: contiguous groups of 4, one process per leased GPU, merged
after. A sharded run gives each window the NLL a single-GPU run gives it.

## Results

`run.py submit` runs the eval through `research run` on vy-nebius-2, under a gpu-lease of the model's GPUs (at most 2),
reading clocks and never setting them. The run's `out/` holds `results.json` (schema `ppl-bench/v1`: per arm and seed
the perplexity, mean NLL and per-window NLLs, the gates, divergence from the reference), `nll.npz` (every scored
token's NLL), `table.md`, the shard logs and `gpu.txt`. The run publishes itself as an Attempt; cite it by run id, and
put findings about it on labels (`research data label`).

## Baseline

The adopted design (dither, w64 peel, psink, kc, SmoothQuant α 0.5, weights ±63), its halves, the design without
SmoothQuant, and the plain int8 arms, locked in `baseline.json` per point the bench has run. "tonight" is the same arm
as 16 Oct's harnesses measured it on the same windows (D4-70B's `linears_layerwise.py` on C4, D4-ATTN's `attention.py`
on WikiText-2), read from their runs' `results.jsonl`.

Two measurements of an arm reproduce each other when their seed ranges, widened by 0.1% (`methodology.reproduces`,
`REPRODUCE_FLOOR`), overlap. `run.py baseline RESULTS...` checks new results against the lock that way, and
`run.py baseline --lock RESULTS...` replaces their points and rewrites the table below. The floor covers what moves a
deterministic arm between harnesses:

- RoPE tables. Tonight's harnesses built them on the GPU (`LlamaRotaryEmbedding(config, device=cuda)`), whose
  `inv_freq` differs from the CPU's in 5 of 64 entries. HF's `from_pretrained` computes them on the CPU, as the bench
  does. HF's own `LlamaForCausalLM` (SDPA, 4 windows per forward) is the bench's bf16 end to end on Llama C4 2K, 9.15834,
  and with GPU-built tables it is tonight's 9.15903 (`r20261016-182515-16e9`).
- Kernel shapes. A bf16 GEMM's rounding depends on its rows: the same model at 1 window per forward is 9.16174
  (+0.04%). The bench's arms run 1 window per forward; D4-70B's ran 4.
- The coins. D4-70B drew a linear's coins per batch of 4 windows, the bench per window, so seeded arms agree in
  distribution, not bit for bit.
- The reference attention under a linears-only arm: D4-ATTN's were eager (Llama) or FP32 (Qwen2.5), marked.

<!-- baseline -->
| model · set · context | arm | perplexity | Δ% vs bf16 | Δ% vs plain int8 | tonight | tonight's method | Δ% vs tonight | reproduces |
|---|---|---:|---:|---:|---:|---|---:|---|
| llama-3.1-8b · c4 · 2K (16 windows) | bf16 | 9.1583 | - | - | 9.1590 | `bf16` | -0.01 | yes |
|  | int8-linears | 9.1690 | +0.12 | - | 9.1753 | `r127` | -0.07 | yes |
|  | int8-attention | 9.2644 (9.2579–9.2702) | +1.16 | - | - | - | - | - |
|  | int8-all | 9.2782 (9.2758–9.2798) | +1.31 | - | - | - | - | - |
|  | adopted-linears | 9.1931 (9.1882–9.1985) | +0.38 | +0.26 | 9.1991 | `d63-sm50-smkv` | -0.07 | yes |
|  | adopted-attention | 9.2093 (9.1966–9.2165) | +0.56 | -0.59 | 9.2007 | `bf16-at` | +0.09 | yes |
|  | adopted-no-smooth | 9.2580 (9.2524–9.2610) | +1.09 | -0.22 | 9.2575 | `d63-at` | +0.01 | yes |
|  | adopted | 9.2402 (9.2296–9.2468) | +0.89 | -0.41 | 9.2334 | `d63-sm50-smkv-at` | +0.07 | yes |
| llama-3.1-8b · wikitext2 · 2K (16 windows) | bf16 | 6.4686 | - | - | 6.4674 | `sdpa` | +0.02 | yes |
|  | int8-linears | 6.4733 | +0.07 | - | 6.4741 | `bf16-lin8r` (eager attention) | -0.01 | yes |
|  | int8-attention | 6.5447 (6.5421–6.5477) | +1.18 | - | 6.5447 (6.5421–6.5477) | `int8-rot` | +0.00 | yes |
|  | int8-all | 6.5498 (6.5469–6.5527) | +1.26 | - | 6.5480 (6.5450–6.5507) | `int8-rot-lin8r` | +0.03 | yes |
|  | adopted-linears | 6.4912 (6.4897–6.4931) | +0.35 | +0.28 | - | - | - | - |
|  | adopted-attention | 6.4975 (6.4959–6.5008) | +0.45 | -0.72 | - | - | - | - |
|  | adopted-no-smooth | 6.5168 (6.5163–6.5171) | +0.75 | -0.50 | 6.5211 (6.5201–6.5230) | `int8-rot-lin8rd-eqp=w64-poff64-psink-kc-kv=dith` | -0.07 | yes |
|  | adopted | 6.5202 (6.5160–6.5234) | +0.80 | -0.45 | - | - | - | - |
| qwen2.5-7b · wikitext2 · 2K (16 windows) | bf16 | 6.3538 | - | - | 6.3553 | `sdpa` | -0.02 | yes |
|  | int8-linears | 6.3583 | +0.07 | - | 6.3577 | `ref-lin8r` (FP32 attention) | +0.01 | yes |
|  | int8-attention | 6.3940 (6.3895–6.3964) | +0.63 | - | - | - | - | - |
|  | int8-all | 6.3971 (6.3934–6.4002) | +0.68 | - | 6.3970 (6.3951–6.3983) | `int8-rot-lin8r` | +0.00 | yes |
|  | adopted-linears | 6.3767 (6.3717–6.3798) | +0.36 | +0.29 | - | - | - | - |
|  | adopted-attention | 6.3642 (6.3625–6.3659) | +0.16 | -0.47 | 6.3642 | `int8-rot-eqp=w64-poff64-psink-kc-kv=dith-s1` | +0.00 | yes |
|  | adopted-no-smooth | 6.3900 (6.3833–6.3956) | +0.57 | -0.11 | 6.3869 (6.3852–6.3891) | `int8-rot-lin8rd-eqp=w64-poff64-psink-kc-kv=dith` | +0.05 | yes |
|  | adopted | 6.3865 (6.3813–6.3905) | +0.51 | -0.17 | - | - | - | - |
| qwen2.5-7b · wikitext2 · 8K (16 windows) | bf16 | 5.9517 | - | - | 5.9517 | `sdpa` | +0.00 | yes |
|  | int8-linears | 5.9546 | +0.05 | - | 5.9536 | `ref-lin8r` (FP32 attention) | +0.02 | yes |
|  | int8-attention | 6.0252 (6.0236–6.0273) | +1.24 | - | - | - | - | - |
|  | int8-all | 6.0289 (6.0271–6.0314) | +1.30 | - | 6.0296 (6.0282–6.0312) | `int8-rot-lin8r` | -0.01 | yes |
|  | adopted-linears | 5.9668 (5.9661–5.9678) | +0.25 | +0.20 | - | - | - | - |
|  | adopted-attention | 5.9634 (5.9622–5.9643) | +0.20 | -1.03 | - | - | - | - |
|  | adopted-no-smooth | 5.9792 (5.9778–5.9804) | +0.46 | -0.82 | 5.9793 (5.9773–5.9814) | `int8-rot-lin8rd-eqp=w64-poff64-psink-kc-kv=dith` | +0.00 | yes |
|  | adopted | 5.9789 (5.9774–5.9801) | +0.46 | -0.83 | - | - | - | - |

Runs:

- llama-3.1-8b · c4 · 2K: bench `r20261016-181132-a47c`; tonight `r20261016-151826-4e17`
- llama-3.1-8b · wikitext2 · 2K: bench `r20261016-182558-c0e6`; tonight `r20261016-065318-8035`, `r20261016-091629-d9e7`, `r20261016-121921-ec96`
- qwen2.5-7b · wikitext2 · 2K: bench `r20261016-183113-01ee`; tonight `r20261016-105148-9add`, `r20261016-110904-30f8`, `r20261016-121921-ec96`
- qwen2.5-7b · wikitext2 · 8K: bench `r20261016-183927-f944`; tonight `r20261016-110904-30f8`, `r20261016-121921-ec96`
<!-- /baseline -->

## Tests

`uv run tools/verity/check/suites.py benchmarks/ppl` runs the CPU tests on tiny random Llama and Qwen2 models: spec
parsing, the window recipe and lock, the significance rule, the engine's bf16 stream against an independent float64
forward, every gate on the adopted arms, gates that fail on sabotage reporting nothing, sharding, seeds, and HF parity
where transformers is importable.
