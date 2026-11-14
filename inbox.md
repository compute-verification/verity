[2026-11-14T06:29Z] 45ac6a927bf6
TRN-166 ROUTES, Fri 13 Nov, 10:29 PM PST.

**Part 1: 61 of the library's 64 configs take a step no test job takes. They fall into 23 kinds. Each kind now has its own case on branch `cursor/training-step-kinds-af01`, and every case passes the test as written on my VM; the node run of the suite is in flight.**

How I grouped them (READ: `routes/*.py`, `routes/_optimizers.py`, torch 2.14's `optim/adam.py` and `sgd.py`, `run.py`'s `checkpoint_state`, the 64 configs). Two steps are the same kind when they take the same branches in the route's `build` and `step`, the same optimizer classes with the same options, the same micro-batching (one, or more), the same answer to "is the embedding trained", and the same frozen-block layout. These settings branch: `opt`, `weight_decay` (zero or not), `fused`, `micro_batches` (one or more), what trains, QAT's `grad_code` and `low_state`. These do not: rates, warmup, decay, cooldown, `keep`, rank, alpha, how many targets, depth, batch, and `store` (E is encoded under `uncounted()`).

Two kinds have a test job (READ):
- `lora`: s-t20-1.1x-lora. Same `lora.build`, with `subset.step`, AdamW unfused at weight decay 0, one micro-batch. Only the targets and rank differ.
- `grow`: m-t20-2x-hybrid-grow-depth and l16-t20-2x-hybrid-grow-depth. Same `grow.build`: decayed Muon (stack), fused AdamW on the gains and the embedding, more than one micro-batch. Only the depth differs.

The 23 kinds with none. Each line gives the case name, the configs, and the code no test job ran. Every test job's AdamW, Muon and Shampoo has weight decay 0, so none takes the decay branch: torch's `param.mul_(1 - lr*wd)`, or `_optimizers`' `p.mul_`. Grow's Muon is the one exception. None runs `subset`'s fused AdamW, and none has a frozen block below or between trained ones, because the tiny start has one block.
1. `adamw-decay`: s-adamw. AdamW's decay branch.
2. `accumulate`: s-accumulate. A trained embedding's gradient summed over micro-batches (the one job with micro-batches, `muon`, freezes the embedding), plus the decay branch.
3. `adafactor`: s-adafactor. All of `_optimizers.Adafactor`.
4. `lion`: s-lion. All of `_optimizers.Lion`.
5. `sgd`: s-sgd. torch SGD with Nesterov momentum, its decay added to the gradient.
6. `last-blocks`: s-last-blocks. Frozen blocks under the trained ones, which run forward only.
7. `muon-whole`: s-muon-whole. Muon decayed, with the embedding trained by the unfused AdamW beside it, one micro-batch.
8. `muon-whole-fused`: s-ref-optim-1027, s-ref-optim-747. As 7, with fused AdamW.
9. `fullmuon`: s-fullmuon, m-fullmuon-wsd, m-fullmuon-wsd-4611, m-ref-optim-1372, l16-ref-optim-fullmuon-wsd. As 8, in micro-batches.
10. `muon-decay`: s-ref-optim. Muon on the blocks decayed, unfused, one micro-batch.
11. `muon-fused`: s-ref-optim-961, s-ref-optim-fullmuon-758, s-ref-optim-fullmuon-841, s-wildcard-topcode. As 10, with fused AdamW.
12. `fullmuon-fe`: l16-fullmuon-fe-wsd, l16-fullmuon-fe-wsd-5383, l16-ref-optim-fullmuon-fe-wsd, l16-ref-optim-fullmuon-fe-wsd-102, l16-2x-fullmuon-fe-wsd-40326, m-ref-optim-fullmuon, m-wildcard-topcode, s-fullmuon-fe-wsd, xl-ref-optim-fullmuon-fe-wsd, xl-ref-optim-fullmuon-fe-wsd-3884. As 11, in micro-batches. This is the largest kind, and the `muon` job differs from it only by fused AdamW and the decay.
13. `depthshare`: s-shallow-depthshare-6, m-shallow-depthshare-8. Muon undecayed, fused, in micro-batches, every block. It differs from `muon` only by fused AdamW.
14. `depthshare-top`: s-shallow-depthshare-3, m-shallow-depthshare-4, l16-shallow-depthshare-4, xl-shallow-depthshare-4. As 13, with frozen blocks under the trained ones.
15. `strideshare`: s-shallow-strideshare. As 13, where the backward passes through a frozen block between two trained ones.
16. `shampoo-decay`: s-shampoo. Shampoo's decay branch.
17. `shampoo-whole`: s-shampoo-whole. As 16, with the embedding trained by the AdamW beside it.
18. `ema-whole`: s-ema. The `ema` route over the whole model, AdamW decayed.
19. `ema-fullmuon`: s-fullmuon-ema, m-fullmuon-wsd-ema, m-fullmuon-wsd-4611-ema. The `ema` route on kind 9.
20. `ema-fullmuon-fe`: s-fullmuon-fe-wsd-ema, l16-fullmuon-fe-wsd-ema, l16-fullmuon-fe-wsd-5383-ema. The `ema` route on kind 12.
21. `qat`: s-qat-top2, -top3, -top3-3000, -top4, -top4-keep10, -keep15, -keep15-2380, -keep30, -keep50, -keep50-7080, -keep7.5-decay60. All of the `qat` route: decoding with `SparseTernary`, the straight-through gradient, Muon on the latents, re-rounding.
22. `qat-grad4`: s-qat-top3-grad4. `qat.passed`, which packs the gradients with `row_codes` and unpacks them with `row_values`.
23. `qat-grad4-lowstate`: s-qat-top4-keep15-grad4-lowstate, -keep50-grad4-lowstate, -keep15-grad8-lowstate, -keep30-grad8-lowstate. Muon's `low_state`: int8 momentum codes with stochastic rounding, and bf16 latents. The 8-bit configs are in this kind. A probe showed their counted step's op log equals the 4-bit one's (RAN: local probe script, not committed), because 4- and 8-bit codes differ only in shifts and ors, and the counter counts neither.

Cases (RAN locally, `pytest tests/test_run.py -k ...` on my VM): 23 new jobs in `synthetic.KINDS`, each at 3 steps.
- `test_a_jobs_trace_and_its_counter_agree_on_the_traced_step`: 23 of 23 pass.
- `test_a_jobs_record_is_one_record_under_either_attention_kernel`, which takes every job: 23 of 23 pass.
- The optimizer-products check (bf16 for Muon, f32 for Shampoo) now covers every job whose step runs Muon or Shampoo, QAT's included.
- No case fails, so there is no difference to report and no `xfail`. No kind needed a stated reason: the trace takes every one.

Cost: the new cases add about 2 min on my 4-CPU VM, about 105 s in the agreement test and about 22 s in the kernel test. `strideshare`, with its 4-block start, takes about 13 s. The docstring now says what they catch that a cheaper test can't.

Note (INFERRED, from `flops.py`'s docstring and the probe): both counts leave out non-multiplicative optimizer work, such as QAT's packing and the comparisons in the ternary rounding. So the agreement says nothing about those ops. That is the counter's design, not a difference between the two counts.

In flight: run r20261114-062721-fad6, `suites.py benchmarks/training_slowdown --fresh` on a node. The draft goes up after it passes.

