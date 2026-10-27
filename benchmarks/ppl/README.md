# The perplexity bench (`verity bench ppl`)

One standard bench for interventions on the matmuls of a served model: what perplexity a change to the linears or to
attention costs, against bf16 and against plain int8, on fixed windows, seeds and references. It joins tonight's three
harnesses (16 Oct: D4-PPL's linears, D4-ATTN's attention, D4-70B's layer-wise `linears_layerwise.py`) into one.
`run.py` is the runner; its docstring lists the commands.

~~~text
verity bench ppl submit --model llama-3.1-8b --set c4 --ctx 2048 adopted        # a `research run` on vy-nebius-2
verity bench ppl eval --model M --set S --ctx T --out DIR SPEC...                 # the same, on a GPU node
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
  refuses another; Llama's C4 lists at 2K are tonight's `c4val2048x16.npy` and `c4cal2048x16.npy` byte for byte.

  `run.py tables windows` lists each context's windows and scored tokens.

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
first paragraph and the harnesses and notes it comes from (`source=` where it is registered). `run.py steps` prints
their parameters and docs.

`run.py tables arms` lists every arm in `specs/`: its scope, whether it runs every seed, whether it reads calibration,
the plain int8 arm it is reported against, and its `description`. A new arm is its file, and its `description` says
what it is and where it comes from; the account below, by family, runs to PPL-COMPILE and grows no further.

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
SKIP's arms (note:20261017T0340Z-draft-skip): `adopted-qkd` (Q Kᵀ dithered on both operands, no Q peel),
`adopted-bd` (the linears too), `adopted-bd-lm` (and lm_head), `co` (CO-Dither's served values at r = k/2: 7-bit
sides plus fresh bits), `co32c` (CHECKED-STRIP's: 7-bit weights and 8-bit activations on the linears, `co`'s
attention), `co32c-pv` (`co32c` with P·V as CO-Dither serves it: P dithered with a fresh bit under the w64 peel; `co32c-pvN`
in blocks of N = 256, 512 or 1,024 keys; `adopted-bd-lm-pv` is `adopted-bd-lm` with that P·V, `adopted-bd-lm-pvN` in
blocks), and `ncp` (NCP-INT's: the exact product of 7-bit operands) and `cb` (`ncp`'s linears with `co`'s attention),
kept as records now that NCP-INT is no longer a design candidate. A design names its arm as `ppl` in the design registry
(`benchmarks/pouw/machine_model/data/designs/`, decision 79's names): Dither-1S `adopted`, Dither-2S `adopted-bd-lm`,
Dither-2S-pv `adopted-bd-lm-pv`, Dither-C `co32c-pv` and Dither-C-peel `co32c`.
`adopted-bd-lm-dB` and `co32c-dB` (B = 2, 3, 4) are `adopted-bd-lm` and `co32c` with a B-bit dither on the linears,
lm_head, Q and K (note:20261017T0518Z-draft-ppl-dither-size).
One-sided dither against `adopted-bd-lm`'s both-sided: `side-lin-act` and `side-lin-w` round the linears' and
lm_head's weights, or their activations, on ±127; `side-qk-q` and `side-qk-k` round K, or Q, in Q Kᵀ; each `-dB`
(B = 2, 3, 4) is `adopted-bd-lm-dB` with the same operand rounded; `side-act-k[-d2]` rounds both the weights and Q
(the record's sides without U). `dither_plot.py` draws perplexity against B per side choice.
`co32c-uN` (N = 8, 16, 32, 64) is `adopted-bd-lm` with the linears' and lm_head's weights capped at ±(127 − N): co32c's
linears with the peeled U on ±N (`benchmarks/pouw/theory/narrow_u.py`) and design A's attention.
`co32c-u16-cohalf` is `co32c-u16` with Dither-C's cohalf Q Kᵀ at U on ±16 (Q and K dithered within ±111).
PPL-LUT's arms (note:20261017T2111Z-draft-ppl-lut) measure the integer softmax of `benchmarks/pouw/edited_gemm/fattn.cu`
(`fattn_ref.py`), `attention.int8`'s `softmax = "lut"`. Per query row it computes:

- the fixed-point A = rint(s_q s_k log2(e) `lut_res` / √d 2³²), which needs one K scale per (layer, KV head);
- the max m of S = Q Kᵀ over keys 0..pos;
- t = min(⌊(m − S) A / 2³²⌋, `lut_size` − 1) and e = rint(127 · 2^(−t/`lut_res`)), whose last entry must be 0;
- P = e at keys 1..pos, and key 0 as z_q V₀, with s_q = f32(1)/f32(L), z_q = f32(e₀)/f32(L) and L = Σ e.

`lut_sink` takes key 0 out of the max. Its weight above the max over keys 1..pos, at u = −t, is H[u mod `lut_res`] ·
2^(u div `lut_res`) with H[r] = rint(127 · 2^(r/`lut_res`)). `lut_round` takes t to nearest, ⌊((m − S) A + 2³¹) / 2³²⌋.
`lut_norm` = b sums L over a table 2^b finer, rint(127 · 2^b · 2^(−t/`lut_res`)) (key 0 above the max: its weight
times 2^b), with s_q = f32(2^b)/f32(L) and P = e as before, so keys whose e rounds to 0 still count in L.

The arms:
- `vllm-int8` is vLLM's plain step as VLLM-18 serves it: int8 everywhere, rounded, unrotated, lm_head bf16, float
  softmax.
- `vllm-int8-lm` and `vllm-int8-lut-cal-lm` are `vllm-int8` and `vllm-int8-lut-cal` with lm_head on the same int8 grid,
  as recipe `verity-int8-channel127/v2` serves it (VLLM-49).
- `vllm-int8-khead` changes only the K scale, to one per (layer, KV head).
- `-khead-sink` is `-lut-sink`'s float twin: the same K scales, float softmax, and key 0 in a P block of its own
  (`p_sink`).
- `-lut` is the kernel's form; `-lut-sinkm` takes key 0 out of the max; `-lut-sink` also gives key 0's K its own scale;
  `-lut-sink-rnd` also rounds t; `-lut-sink-norm` sums L over a table 2⁷ finer.
- `-lut32` is 2^(−t/32) with 512 entries; `-lut-cal` calibrates the K scale; `-lut-sink-cal`, `-lut-sink-norm-cal`
  and `-lut32-sink` combine them.
- `co32c-u16-cohalf-*` are the same on H1's numerics.
`frontier_plot.py` draws perplexity against core ε (`machine-model eps --checking full`) under each reading of the table
route, and writes the table.
PPL-COMPILE's arms (`pc-*`) cost running Dither-2S (`adopted-bd-lm`) on FP8 and FP4 hardware, every matmul including
lm_head, after the same SmoothQuant, rotations and key centring: `pc-int8` and the format baselines `pc-fp8-tensor`,
`pc-fp8-channel`, `pc-mxfp8`, `pc-nvfp4` and `pc-mxfp4` are plain quantization with no PoUW, and each one's `-lin`
twin quantizes only the linears in the format and keeps `pc-int8`'s attention (C2's and C3's scope); `pc-c1-ROUTE` is
`adopted-bd-lm` with every int8 product emulated from limbs (its perplexity is `adopted-bd-lm`'s bit for bit);
`pc-c2-GRID-2s|1s` puts the linears on a format's integer grid (`int5`, `e4m3i`, `e4m3i-t` per tensor, `mx8`, `nvfp4`
with odd scales, `mxfp4`, on E2M1's integers; `nvfp4x2`, `mxfp4x2` on twice E2M1's values), both operands dithered or
the activations only (`int5-1s` is `fp8-peel-w15`'s design), and `pc-c3-fp8|fp4|fp4x2` keeps Dither's int8 activations
against native dithered weights; C2 and C3 keep Dither-2S's attention.
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
- `mm.cached(name, fn)`: once per matmul and chain prefix, shared across windows and arms (and across seeds with
  `coinless=True`, when fn reads none of the step's own coins).
- `mm.shared(key, fn)`: once per run.
- `mm.coins(shape)`: fresh bits from the arm's seed, the layer and the window.
- `mm.checking`: this is the check window. `mm.gate(name, ok)` records a gate; `mm.int_product(x, w)` is a gated exact product.
- `mm.kind`, `mm.name`, `mm.layer`, `mm.window`, `mm.seed`, `mm.extra`: where the matmul is. Attention's extra holds
  the KV group, the query rows, the causal mask, and `rep`, `s` and `d`.

Declare `seeded=True` (or a function of the parameters) when the step's effect depends on the seed, so the arm runs
every seed, and `calib=True` when it reads `mm.run.calib`. A `check=` function refuses bad parameters at parse time.
`source=` names the harnesses or notes it comes from (`run.py tables steps`' last column).
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

DeepSeek-V2-Lite (`dsv2.py`) is HF's `DeepseekV2DecoderLayer` op for op: MLA with no q LoRA (16 heads of 192-wide
keys, 128-wide values), YaRN RoPE as complex numbers on interleaved pairs with its mscale² on the softmax scale, layer 0
dense and the rest 64 routed experts (greedy softmax top-6) plus two shared, eager experts, checked by `hf_reference`.
Arms reach every linear, each routed expert's on its rows, and attention, with Q and K zero-padded to 256 for
`rotate`'s Hadamard. SmoothQuant's groups are the layer's: one m per input its linears share (q_proj with
kv_a_proj_with_mqa, the shared experts' gate and up with every routed expert's gate_up), and a routed expert no
calibration token chose is not smoothed. It is the V4 proxy's model: `scale_error` scales the error of the chain after
it by a factor per matmul, so its matmuls carry the per-matmul error another model's shapes would.

`error_probe.py measure` gives those factors' inputs: an arm's relative error per linear class on synthetic operands of
each model's (k, n), and per attention shape (whole, Q K^T alone, P V alone) over a grid of logit spreads and sink
logits. `error_probe.py factors PROBE.json --fit RESULTS.json --spec specs/NAME.toml [--upper]` fits the attention
grid to a V2-Lite run's measured `attn_rel_err_w0` of the arms that differ only in P V's block (the synthetic Q and K
have no outlier channels, so int8-all is not a target), checks the fit's context scaling on Llama-3.1-8B (2K fitted, 8K
predicted against H1's measured errors), and writes the V4-scaled arm, at the fit or at the upper end of the attention
factors over the grid points near it.

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

`run.py submit --device cpu [--threads T] [--on NODE] [--mem-gb G]` runs it on a CPU node instead (vy-nebius-cpu-2 by
default; `cpu_job.sh`, a venv of CPU torch and transformers in scratch, the snapshot read from the node's HF cache
offline): one process per group of 4 windows, T threads each, SDPA's math kernel (its flash kernel is not deterministic
on the CPU). oneDNN's bf16 and int8 GEMMs are exact and deterministic on vy-nebius-cpu-2's AMD cores at any thread
count (r20261026-201043-5405), which `hf_reference` and `product_exact` check on every run; they are not on an AMX
agent VM.

## Baseline

The adopted design (dither, w64 peel, psink, kc, SmoothQuant α 0.5, weights ±63), its halves, the design without
SmoothQuant, and the plain int8 arms, locked in `baseline.json` per point the bench has run. "tonight" is the same arm
as 16 Oct's harnesses measured it on the same windows (D4-70B's `linears_layerwise.py` on C4, D4-ATTN's `attention.py`
on WikiText-2), read from their runs' `results.jsonl`.

Two measurements of an arm reproduce each other when their seed ranges, widened by 0.1% (`methodology.reproduces`,
`REPRODUCE_FLOOR`), overlap. `run.py baseline RESULTS...` checks new results against the lock that way, and
`run.py baseline --lock RESULTS...` replaces their points in `baseline.json`. `run.py tables baseline` (or `run.py
baseline` alone) prints the lock as a table: per point and arm, the bench's perplexity with its seeds' range, its Δ%
against bf16 and against plain int8, tonight's measurement and method, their Δ%, whether they reproduce, and each
point's runs. The floor covers what moves a deterministic arm between harnesses:

- RoPE tables. Tonight's harnesses built them on the GPU (`LlamaRotaryEmbedding(config, device=cuda)`), whose
  `inv_freq` differs from the CPU's in 5 of 64 entries. HF's `from_pretrained` computes them on the CPU, as the bench
  does. HF's own `LlamaForCausalLM` (SDPA, 4 windows per forward) is the bench's bf16 end to end on Llama C4 2K, 9.15834,
  and with GPU-built tables it is tonight's 9.15903 (`r20261016-182515-16e9`).
- Kernel shapes. A bf16 GEMM's rounding depends on its rows: the same model at 1 window per forward is 9.16174
  (+0.04%). The bench's arms run 1 window per forward; D4-70B's ran 4.
- The coins. D4-70B drew a linear's coins per batch of 4 windows, the bench per window, so seeded arms agree in
  distribution, not bit for bit.
- The reference attention under a linears-only arm: D4-ATTN's were eager (Llama) or FP32 (Qwen2.5), marked.

## Tests

`uv run tools/verity/check/suites.py benchmarks/ppl` runs the CPU tests on tiny random Llama, Qwen2, block-FP8 GLM and DeepSeek-V2 models: spec
parsing, the window recipe and lock, the significance rule, the engine's bf16 stream against an independent float64
forward, every gate on the adopted arms, gates that fail on sabotage reporting nothing, sharding, seeds, and HF parity
where transformers is importable.
