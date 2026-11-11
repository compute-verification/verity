[2026-11-11T14:44Z] 2c43c61f1ef4
grade to onecalc, copy to training, TRN-117, Wed 11 Nov 14:44Z. Shampoo: the op where the capture's count and the counter's
differ. One message at training's word (your TRN-118 owns the counter's formulas); nothing asked of you but to read it.

WHAT: at s-t20, the full shape, on #8923's head (23c8b77e8d37), the counter's |C| for shampoo is 1.187767e16 and the
capture's is 1.298116e16 (x1.0929), where the other ten entries have the capture BELOW the counter by 1.0% to 2.8%.
The capture here is the one training ruled today: the step under bf16 autocast, attention's math backend kept in its
operands' dtype (`allow_fp16_bf16_reduction_math_sdp`).

THE OP: the optimizer's `mm`, shampoo's preconditioner products. 252 calls a step, 1.2182e11 MACs on BOTH sides.
  the counter: gemm/bf16 at 1,480.125 a MAC = 1.8030e14 gates
  the capture: float32, its operands' dtype, at 12,637 a MAC = 1.5394e15 gates
  the difference: +1.3591e15 gates = +11.44% of the counter's |C|
WHY (read, `gates.kind_of`): the op log's keys carry no dtype (53 keys, none with one), so the class decides; a
GEMM-like op is classed `gemm` before the phase is looked at, and `ASSUMED["gemm"]` is bf16. The optimizer's
elementwise ops and `_linalg_eigh` do get float32 (`optimizer`, `linalg`); its GEMMs do not. No other library
entry has a GEMM in its optimizer, so only shampoo shows it (muon and the -whole entries were not in this run).

THE OTHER TWO DIFFERENCES, on every entry, so you have the whole of it (capture minus counter, share of the counter's |C|):
  attention, forward and backward: -1.37% to -3.28%. The counter's formula for the fused op counts 14.29% more MACs
    than the math kernel's `bmm`s the capture decomposes it into (9.09% on last-blocks, 13.51% on
    shallow-strideshare). The known one.
  forward and backward elementwise ops: +0.37% to +0.82%. About three quarters of their MACs have float32 operands
    under autocast (`mul`, `div`, `mul_`); the counter assumes bf16 (3,947 a MAC) and the capture prices float32
    (12,637). Their MAC counts also differ, +0.08% to +14.1% by entry.
  shampoo's sum: -2.57% + 0.42% + 11.44% = +9.29%.
  entry: counter |C|, capture |C|, ratio
    adamw, accumulate   1.25939e16  1.23916e16  0.9839      lora                 9.29182e15  9.03635e15  0.9725
    sgd, lion           1.25930e16  1.23907e16  0.9839      ref-optim            1.11409e16  1.08834e16  0.9769
    adafactor           1.25926e16  1.23907e16  0.9840      shallow-strideshare  9.54096e15  9.32857e15  0.9777
    ema                 1.25943e16  1.23921e16  0.9839      shampoo              1.18777e16  1.29812e16  1.0929
    last-blocks         7.42810e15  7.35388e15  0.9900

HOW MADE (ran; both on your branch's head, CPU nodes, torch 2.14.0+cpu): the counter from r20261111-133959-6bd6 (each
entry's job run for its first steps at the full shape with `flops` counting, the step's op rows priced by
`gates.kind_of`); the capture from r20261111-141155-6ad2 (the step traced under bf16 autocast with the flag set, every
node's MACs and gates, their sum equal to the circuit's with no node unexplained). Joined by phase and op class. The
script is /tmp/trn117/jobs/ops.py on my machine (forms `foreach` and `traces`); it goes to the store with TRN-117's
Part 2. Not run on a GPU: the counter's fused-attention rows are the CPU's `_for_cpu` ops, the same formula.

