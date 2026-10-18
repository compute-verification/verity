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
