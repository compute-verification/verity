circuits' worker FP8READ, Thu 12 Nov 10:57 AM PST (18:57Z), second comment: what the card showed.

**FlashMLA's FP8 sparse read ran alone on an H200, called as vLLM's pin calls it, and every output word equals the new Definitions' reference: 48,234,496 words over 1,472 tokens, 0 differ. This kernel and the dense prefill's read-back do read one cache row differently, as the hand example said: of 144 probed rows that vLLM's own write filled, 134 differ, in 16,171 of 73,728 words, each by one bf16 step. No name, static or port of my first comment moved. Sat 14 Nov 8 AM PST holds.** This is the kernel alone on random and planted rows. Nothing was served.

**1. The run** [read: `r20261112-181549-6fce` and `r20261112-183549-43fa` on node A's card 1; the two saved the same inputs and outputs, byte for byte]

- **Called:** `FlashMLASparseImpl._forward_fp8_kv_mixed_batch` itself, 8 heads padded to 64. vLLM's own `fused_norm_rope` wrote 163,354 cache rows; 128 more are planted (the hand row, NaN bytes, scales on a bf16 tie, infinite and subnormal scales).
- **30 steps:** 24 decode steps of 1 to 133 tokens, one query row a request; prefill-shaped steps of 320 and 45 tokens; edge rows (indices of -1 scattered, in whole blocks, everywhere); probes.
- **Against the reference function:** the 8 real heads and one pad head, 6,782,976 words, 0 differ. The other 55 pad heads against that one: 41,451,520 words, 0 differ.
- **Against `FlashMlaV32Decode_v1`'s own evaluator:** 73 tokens, 299,008 words, 0 differ. The other 1,399 tokens are queued off the card (run `r20261112-183708-9a9a`).
- **The planner's cut equals my table at every step:** 32 splits at T ≤ 3 tokens, 16 at 4 to 6, 11 at 7 to 10, 8 at 11 to 13, 7 at 14 to 16, 6 at 17 to 18, 5 at 19 to 22, 4 at 23 to 33 (two cuts), 3 at 34 to 44, 2 at 45 to 66, one from 67.
- **Saved in each run:** everything the kernel read and wrote, and this instance's sm_90a SASS (4,296 instructions).

**2. The five registered Definitions, from that SASS and the run.** None holds with new statics alone. Three bodies are reused.

- **`FlashMlaRowValid_v1`: no.** This instance compiles the length test out. A row is valid iff its index is not -1.
- **`FlashMlaKvRow_v1`: no.** The SASS loads four f32 scales at byte 512, rounds each to bf16, then multiplies once in bf16. The registered scale is one UE8M0 byte.
- **`FlashMlaBlock_v1`: its arithmetic, at two widths.** 36 score steps where the registered instance has 32; the same 8 value steps over 512 columns. Beyond the row, that chain and the index divisions, the two instances' floating-point instructions are equal count for count (a census, not a line-by-line diff).
- **`FlashMlaSplit_v1`: its end, on a constant sink of -inf.** No sink and a sink of -inf wrote equal words at all 30 steps.
- **`FlashMlaCombine_v1`: its arithmetic without the sink term.** The combine and the planner are the compiled functions the registered instance calls.
- **The scheduler** sees 64 heads, so the cut is a function of T alone.
- **The 56 pad heads: no Definition.** Their outputs equal the reference on a zero query, and vLLM slices them off.

**5. The two reads of one cache row.** A probe is a token with one valid row read through a pad head: its output is the row as this kernel staged it.

- **The hand row:** this kernel gives 1.765625, the read-back 1.7578125, as computed by hand.
- **Rows vLLM's write filled:** 134 of 144 differ, in up to 253 of a row's 512 NoPE words. The 10 that agree hold only zeros [computed from the saved capture].
- **The 64 RoPE words of a row never differ** (22,528 compared).
- **Each read equals its model:** the read-back on all 352 probed rows, this kernel on the 338 without a NaN or an infinity. The read-back is the f32 product rounded once to bf16. This kernel rounds the scale to bf16 first, then the product.
- So one cache row has two bf16 readings, and neither read's Definition can stand in for the other's.

**Changed since the first comment**

- **The module is `verity/definitions/registry/flashmla_dsv32.py`**, not `flashmla_v32.py`: the adapter's names lint refuses `v32` in a module's name. Every Definition's name is as posted.
- **T under a CUDA graph** [read at `gpu_model_runner.py:2490-2505` and `:4466-4492`, not run]: in a full graph the attention's token count is the graph's padded size. So the T that picks the splits is the padded one, and a step's record must carry it.

**Not done:** the pull request is not open. The evaluator over every token, and circuit-check and the suites on the final head, are still to run. The capture is done: I expect no further run on a card.
