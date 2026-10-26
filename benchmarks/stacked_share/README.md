# The stacked share of live KV

How much of a node's live KV a prover could hold as encoded carriers inside the honest compute budget the audit allows. That's
the stacking law, read in bytes:

    stock(L) = 128 R_cap L / E + R_in L        f = stock(L) / M_dyn

- R_cap is the honest Program's property/v1 units a second.
- E is the units an encode of one 1024-bit word costs.
- L is a carrier's lifetime.
- R_in is the node's run-input bytes a second.
- M_dyn is vLLM's KV pool.

At f ≥ 1 no KV is credited.

- `units.py DESCRIPTOR.json.gz --out units.json [--as SPEC.json --as-out OUT.json]` takes a Build's Program, decoded through
  vLLM's registry, so it runs where `verity_vllm` imports. It produces:
  - property/v1's units (`verity.ml.circuits.stacking`) by callee, class and census/v1 step, checked against
    `stacking.analyze`;
  - the per-token and per-sample units;
  - the attention window's units: one request-step-layer's G `Int8KvQuant_v1` calls (a KV head's window of S keys) and H
    `Int8LutAttentionRow_v1` calls a row (`--kv-heads`, `--heads`), fitted exactly as
    `r0 N + b_q R + b_kv S + w0 + w_start [N = S]` and held to the Program's own calls;
  - property/v1's refusals but the budgets' (operand bits past B, unproven s8 windows, unlisted primitives);
  - what each lever acts on, read off the Program's Definitions;
  - the per-token, per-sample and attention units by primitive, every composite expanded at the ranges its call gives it
    and held to the callee's own count.

  `--as` models another model from the Program's callees at one rank's statics, for example `llama31-70b-tp8.json`: a
  statics key is a callee's id, a `Family{k=v,..}` its family and statics match, or a family, and a key that names no callee
  is refused.
- `share.py --model NAME=UNITS.json,MIX.json ... --out share.json` combines units with a served trace
  (the serve-bandwidth benchmark's `mix.json`). It computes R_cap per node (busy-time mean and peak 1 s window), M_dyn, and f
  for each E, L and R_in, as built and under each lever, cumulatively:
  - K' quantized once at the KV write (its cost leaves every key a step reads for one write a token row);
  - V' stored once with a static scale (likewise);
  - an int8 LM head;
  - static activation scales.

  It also gives the units a token by primitive over the trace, held to the step model's total.
- `job.sh` builds the exact-int8 8B request Program (LP=256, T=31) on one GPU and runs `units.py` on it (`DESCRIPTOR=PATH`
  skips the Build and needs no GPU).

What is measured and what is modeled:

| quantity | source |
|---|---|
| units/callee, 8B int8 | measured: property/v1 on the Build's Program |
| units/callee, 70B TP8 | modeled: the 8B callees bound at one rank's statics, plus the TP all-reduces |
| T, K, decoding requests, step spans | measured: the served trace (vLLM's scheduler output) |
| Q, n, X | exact from the trace's requests; prefill per step at the trace's means; split prompts bounded |
| M_dyn | measured: vLLM's KV blocks × block × bytes a token × tp |
| E | the four values in `share.E_WORD` (proved, conjectured, the red team's cheapest under #2420 and with hole B open) |

The served rate comes from stock FP8 serving (vLLM's FP8 linears, BF16 KV and LM head): the int8 Program's units are counted at
a throughput int8 serving hasn't been measured at.

## Frontier models

The same law for models no Program has been built for (DeepSeek-V3/R1, DeepSeek-V3.2-Exp, Kimi K2, Qwen3-235B-A22B,
Qwen3-Next, GLM-4.5/4.6, GLM-5.3, MiniMax-M1/M2, with the Llama-3.1 baseline), each costed as the exact-int8 profile would
serve it:

- `frontier_models.json`: each model's shapes from its published config (revision recorded), the reference node and workload,
  and the 8B census it is anchored to.
- `atoms.py --out atoms.json` counts each Definition a composition reads (vLLM's registry, bound at the model's statics) by
  property/v1, and fits the attention row and KV window exactly in S. It runs where `verity_vllm` imports, without torch.
- `frontier.py --atoms atoms.json --out frontier.json` composes a decode token-step and a prefill token per model and layer
  kind (GQA, MLA with or without DSA, Gated DeltaNet, lightning attention, dense or MoE FFN) from the atoms, with the KV
  quantized as built or once at its write. It refuses atoms that don't reproduce the census's measured 8B token units. It counts
  in nonlinear units (#4622's gate model): property/v1's units plus one a data product, a product of two run values that
  property/v1 left free as s8 lanes (attention's QKᵀ and PV, DSA's indexer dots, linear attention's chunked prefill products),
  with E a word in the same units (`E_WORD`). It gives:
  - the nonlinear units and honest operations by class, and the data products a token by kind;
  - the dynamic memory's kind, bytes, write path and cut widths;
  - the roofline serving model's R_cap, M_dyn and turnover;
  - f at each E and L, and again with the softmax's work a key at property/v1's floor (`softmax_floor`), with every
    requantization and linear epilogue at one rounding an element as well (`exact_floor`), and before #4622 (property/v1's
    units, E 364,497);
  - f with R_cap and with the decode rate alone as the pool's context varies (`sweep`, up to each model's max_context).

  It also writes the tables as `frontier.md`.
- `frontier_job.sh` runs both on a CPU node.

| quantity | source |
|---|---|
| units per atom | measured: property/v1 on the registry's Definitions at the model's statics |
| units per token | modeled: the atoms composed per layer, checked exact against the 8B census |
| linear attention, DeltaNet and routed-weight arithmetic | modeled: one rounding per FP output (`MODELED`) |
| DSA's indexer scores, top-k and row selection | modeled: a unit per operation (`MODELED`); the served Waksman form is refused by property/v1 |
| the floors | modeled: the softmax at one rounding a head-key; a requantization or an epilogue at the measured count of its essential rounding (`ESSENTIAL_ROUNDING`) |
| data products | shapes: a unit each (`PRODUCT`); the attention row's 2 S D a head matches `products_pass.py`'s measured 16,384 at D = 128, S = 64 |
| E a word | the conjecture (`SquaringFloor`, `ChainSum` at 2^-40) and the best construction found (the seam square), in nonlinear units (note:20261026T0110Z-report-s5-lowerbounds) |
| shapes, experts, KV and state sizes | config: each model's `config.json` at its recorded revision |
| R_cap, batch, t_step | modeled: roofline on the reference node (`node` in `frontier_models.json`) |
