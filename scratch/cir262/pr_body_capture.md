
## The capture: the kernel alone on a card (nothing here is served)

Runs `r20261112-181549-6fce` and `r20261112-183549-43fa`, node A's card 1 (an H200, driver 595.91.07), both at head
`1f6f0969a9`, both rc 0. Each ran the script's three stages: the library's SASS, the launches under one lease, the
comparison on the node's cores. Each run's output directory holds everything the kernel read and wrote, so the comparison
runs again off the card. What I read in both:

- **The instance.** `_flashmla_C.abi3.so` (sha256 `11b226b5…bb3ac3`) holds 23 sm_90a images; the V32 instance at 64 heads
  is 4,296 instructions, kept whole in the run with the registered instance's listing beside it.
- **The call.** `FlashMLASparseImpl._forward_fp8_kv_mixed_batch` itself, 8 heads padded to 64, on a cache of 2,828 blocks
  for 134 requests. vLLM's own write (`fused_norm_rope`) filled 163,354 rows, 10,109 of them with a scale at the floor.
  128 more rows are planted bytes: the hand example, NaN codes, scales on a bf16 tie, infinite and subnormal scales.
- **30 cases, 1,472 tokens.** 24 decode steps of 1 to 133 tokens, one query row a request, at both ends of every range of
  the step's token count where the planner's cut of the 32 blocks changes. Two prefill shapes: 320 and 45 tokens of one
  request. The planted rows with indices of -1 scattered, in whole blocks and everywhere, and a request whose rows are
  not a multiple of the block. The probes.
- **Every launch check held in every case**: one call into FlashMLA with the pin's arguments, the pad heads' queries
  zero, every output word written (two sentinels), the same words on a second call, the same words with a sink of -inf,
  and the planner's metadata equal to the reference's.
- **48,234,496 output words compared, 0 differ.** 6,782,976 of them are the 8 real heads and one pad head against the
  reference function (0 differ). 41,451,520 are the other 55 pad heads against that pad head's words, kernel against
  kernel (0 differ).
- **`FlashMlaV32Decode_v1`'s own evaluator, in the card job:** 73 tokens, 299,008 words, 0 differ.
- The two runs saved the same bytes: 246 of 247 files are identical, and `launch.json` differs only in the output path.

**Every token against the Definition's evaluator, off the card.** Run `r20261112-185922-3b8c` (a check slot, head `1f6f0969a9`) took
the first card run's saved capture and held every token's 8 real heads to `FlashMlaV32Decode_v1` bound at GLM-5.3's
statics and the step's SPLITS: 1,472 tokens, 6,029,312 words, 0 differ, over the 12 cuts the planner made.

**The captures of record ran at `1f6f0969a9`; this head is `@@HEAD@@`.** Between them lie one commit of mine
(`ce254bd223`) and the merge of `main`, which touches none of the nine files. The commit removes seven catch-all handlers
from the script, which the adapter's lint ratchet refuses: a step that fails is now the script's own traceback and
non-zero exit, and it leaves no launch file and no record, where before it left an entry under `errors` and a record that
was not ok. A missing SASS listing is still recorded, as `sass.ok` false. Nothing the capture computes changed:

- **The comparison, run again at this head.** Run `r20261112-231418-ada0` took the first card run's saved capture and
  compared it with this head's script: 48,234,496 words, 0 differ; the evaluator's 73 tokens, 299,008 words, 0 differ; and
  the record it wrote is byte for byte the card run's record (digest `1002e7be…bb9a96e`).
- **The launch stage at this head.** @@LAUNCH@@

## The two reads of one cache row

A probe is a token with one valid row, read through a pad head: a zero query gives that row weight 1, so the output is the
row as this kernel staged it. The same rows were read by the dense prefill's read-back
(`cp_gather_and_upconvert_fp8_kv_cache`) in the same job. From the runs above:

- **They disagree, as the hand example says.** At scale 1 + 2^-8 + 2^-20 and value 1.75 this kernel gives `0x3FE2`
  (1.765625) and the read-back `0x3FE1` (1.7578125).
- **Rows vLLM's own write filled:** 144 probed, 134 differ, in 16,171 of 73,728 NoPE words, every one by exactly one bf16
  step, up to 253 of a row's 512 words. The 10 rows that agree hold only zero bytes at the floor scale.
- **All probed rows:** 352, of which 14 hold a NaN or an infinity and are left out; 326 of the other 338 differ. 173,056
  words, 41,766 differ: 41,520 by one bf16 step and 246 by more, all 246 in planted rows. A -0 this kernel returns as +0
  is counted apart (3,491 words), since the probe cannot show a zero's sign.
- **The 64 RoPE words of a row never differ** (22,528 compared, 0 differ).
- **Each read equals its model.** The read-back is the f32 product rounded once to bf16 (0 words differ on all 352 rows).
  This kernel rounds the scale to bf16, then rounds the bf16 product (0 words differ on the 338 rows).

So one cache row has two bf16 readings, and neither read's Definition can stand in for the other's.
