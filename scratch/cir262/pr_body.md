Lane circuits
Refs CIR-262
Goal vllm/production-vllm-reproducible

## What changed and why

vLLM's recipe for GLM-5.3 serves with an FP8 cache (`fp8_ds_mla`). At the pin (vLLM `d9105ea80`, FlashMLA `6bc49418`) a rank
with 8 heads then reads the cache through FlashMLA's sparse FP8 decode kernel at its V32 instance, 64 heads
(`flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<V32, 64>>`), called by `_forward_fp8_kv_mixed_batch`
(`flashmla_sparse.py:826`). No GLM-5.3 Definition modelled that read. This PR adds it.

- `verity/definitions/registry/flashmla_dsv32.py`: seven Definitions, all `_v1`, and GLM-5.3's statics (`FLASHMLA_GLM53`).
  `FlashMlaV32KvRow` is the staged row (512 E4M3 bytes, four f32 scales each rounded to bf16 before it multiplies, 64 bf16
  words; valid when its index is not -1). `FlashMlaV32Block`, `FlashMlaV32SplitEnd`, `FlashMlaV32CombineWeights` and
  `FlashMlaV32CombineValues` are the registered block, split end and combine with the value 512 wide beside a 576-wide key,
  and with no sink. `FlashMlaV32DecodeHead` and `FlashMlaV32Decode` are a head and a token, whose bodies hold nothing but
  the calls a token is cut into, so each unit fits a Flock block of 2^27 bits.
- `verity/definitions/numerics/flashmla_dsv32_twin.py`: the numpy twin, over `flashmla_twin`'s parts, and the splits a
  step's token count gives (`step_splits`).
- `verity/definitions/registry/flashmla.py` and `numerics/flashmla_twin.py`: `block()` takes the value's width, and a
  split's end and the combine's scale are functions of their own, so the new module calls the registered arithmetic
  instead of writing it again. No registered Definition's program changed.
- `adapters/vllm/tests/properties/glm53_flashmla_fp8_capture_gpu.py` and its test: the capture on a card. It keeps the
  kernel's SASS, calls the pin's own method on a cache vLLM's own write filled (plus planted rows), saves every input and
  output, and compares every output word with the twin and a sample with the Definition's own evaluator; it also reads the
  same rows through the dense prefill's read-back and counts where the two reads differ. No stage catches a failure: a
  step that does not finish is the script's traceback and non-zero exit, and it leaves no record.

The branch holds `main` as of `4175265019` merged in; against `main` the diff is these nine files.

## The Definitions

Each is `_v1`. The values are GLM-5.3's at 8 heads a rank on the H200 (`FLASHMLA_GLM53`); N = 2048 rows a position.

| Definition | statics (bound) | in | out | reuses |
| --- | --- | --- | --- | --- |
| `FlashMlaV32KvRow` | `DN`=512, `DR`=64, `QT`=128 | `nope`: DN e4m3; `scale`: DN/QT f32; `rope`: DR bf16; `idx`: i32 | a staged row: DN+DR bf16 and a valid bit | registered primitives; the row itself is new |
| `FlashMlaV32Block` | `DK`=576, `DV`=512, `BN`=64, `FIRST`, `DOT`=`HopperBF16WgmmaDot16_v1`, `SCALE`=0.0625 | `q`: DK bf16; `k`: BN staged rows; unless `FIRST` the state (`nm_prev`, `l_prev`: 4 f32, `o_prev`: DV f32) | the state (`nm`: f32, `l`: 4 f32, `o`: DV f32) | `FlashMlaBlock_v1`'s body (`flashmla.block`) at a value width |
| `FlashMlaV32SplitEnd` | `DV`=512, `PARTIAL` | `nm`, `l`, `o` | DV bf16, or (`PARTIAL`) `oacc`: DV f32 and `lse`: f32 | `FlashMlaSplit_v1`'s end (`flashmla.split_end`) on a sink of -inf |
| `FlashMlaV32CombineWeights` | `NS` (2 to 32) | `lse`: NS f32 | NS f32 | `FlashMlaCombine_v1`'s scale (`flashmla.combine_scale`) with no sink |
| `FlashMlaV32CombineValues` | `NS`, `G`=256 | `w`: NS f32; `oacc`: NS x G f32 | G bf16 | none: `FlashMlaCombine_v1`'s weighing loop written again over G columns (one FFMA a split, then the bf16) |
| `FlashMlaV32DecodeHead` | `SPLITS`, `DK`, `DV`, `BN`, `DOT`, `SCALE` | `q`: DK bf16; `k`: N staged rows | DV bf16 | calls only |
| `FlashMlaV32Decode` | `NH`=8, `SPLITS`, `DN`, `DR`, `QT`, `BN`, `DOT`, `SCALE` | `q`: NH x (DN+DR) bf16; `nope`: N x DN e4m3; `scale`: N x DN/QT f32; `rope`: N x DR bf16; `idx`: N i32 | NH x DN bf16 | calls only |

`SPLITS` is the step's, not the row's: the planner cuts each position's 32 blocks by the step's token count T alone, 32
splits at T of 1 to 3 down to one split from T = 67. The capture confirmed the table at both ends of every range.

Of the five registered Definitions none holds here with new statics alone. `FlashMlaRowValid_v1` tests a length this
instance compiles out. `FlashMlaKvRow_v1`'s scale is one UE8M0 byte; here the SASS loads four f32 scales and rounds each
to bf16 before one bf16 multiply. `FlashMlaBlock_v1` has one width for key and value. `FlashMlaSplit_v1` and
`FlashMlaCombine_v1` carry a sink the call does not pass, and at 32 splits over 512 columns they are past one Flock block.

## circuit-check

Run `r20261112-231418-ada0` at this head, a check slot, rc 0 for each pass. circuit-check's own update pass
(`--update-pins`) on the sidecar ran first: 12 targets, 0 failures, and it changed no file (the module defines no new
piece to pin). Then the three passes below. The verdicts and sizes equal those of run `r20261112-173637-e88d` on
`1b61bd883e`, before `main` was merged in.

The sidecar (`verity/definitions/registry/flashmla_dsv32.circuit-check.py`): 12 targets, 0 failures, 0 unchecked, all 12
compared with the reference on random vectors.

~~~text
ok  FlashMlaV32KvRow_v1{DN=512,DR=64,QT=128}                                     failures=0 unchecked=0 q_call=2118u/42050b
ok  FlashMlaV32Block_v1{DK=576,DV=512,BN=64,FIRST=True,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}   failures=0 unchecked=0 q_call=2495u/78816b
ok  FlashMlaV32Block_v1{DK=576,DV=512,BN=64,FIRST=False,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}  failures=0 unchecked=0 q_call=3014u/95424b
ok  FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=False}                                 failures=0 unchecked=0 q_call=1034u/24865b
ok  FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=True}                                  failures=0 unchecked=0 q_call=522u/16673b
ok  FlashMlaV32CombineWeights_v1{NS=2}                                           failures=0 unchecked=0 q_call=18u/545b
ok  FlashMlaV32CombineWeights_v1{NS=32}                                          failures=0 unchecked=0 q_call=198u/6305b
ok  FlashMlaV32CombineValues_v1{NS=2,G=256}                                      failures=0 unchecked=0 q_call=768u/20480b
ok  FlashMlaV32CombineValues_v1{NS=32,G=256}                                     failures=0 unchecked=0 q_call=8448u/266240b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,2),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}    failures=0 unchecked=0 q_call=6543u/199105b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,1,3),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}  failures=0 unchecked=0 q_call=10602u/327907b
ok  FlashMlaV32Decode_v1{NH=2,SPLITS=(0,1,3),DN=512,DR=64,QT=128,BN=64,DOT=…,SCALE=0.0625}  failures=0 unchecked=0 q_call=427860u/8729414b
circuit-check: 12 target(s), 0 failure(s), 0 unchecked
~~~

Heads at served sizes (32, 16 and 1 splits) and two combines: 5 targets, 0 failures, 0 unchecked, all on random vectors.

~~~text
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,1,…,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}   failures=0 unchecked=0 q_call=113638u/3594433b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,2,…,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}   failures=0 unchecked=0 q_call=105302u/3328177b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}       failures=0 unchecked=0 q_call=96963u/3061825b
ok  FlashMlaV32CombineValues_v1{NS=16,G=256}                                          failures=0 unchecked=0 q_call=4352u/135168b
ok  FlashMlaCombine_v1{NS=16,D=512}                                                   failures=0 unchecked=0 q_call=8817u/273859b
circuit-check: 5 target(s), 0 failure(s), 0 unchecked
~~~

One whole token at 16 splits: 0 failures, 1 unchecked. The warning is `vectors/edge-only`: at this size the token was
compared with the reference on edge vectors only, and the partition check did not run on the whole. Its units are the
targets above.

~~~text
unchecked  FlashMlaV32Decode_v1{NH=8,SPLITS=(0,2,…,32),DN=512,DR=64,QT=128,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}  failures=0 unchecked=1
unchecked vectors/edge-only: 1 over 1 target(s)
circuit-check: 1 target(s), 0 failure(s), 1 unchecked (warnings, which never fail)
~~~

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

**The captures of record ran at `1f6f0969a9`; this head is `70a27d97ec`.** Between them lie one commit of mine
(`ce254bd223`) and the merge of `main`, which touches none of the nine files. The commit removes seven catch-all handlers
from the script, which the adapter's lint ratchet refuses: a step that fails is now the script's own traceback and
non-zero exit, and it leaves no launch file and no record, where before it left an entry under `errors` and a record that
was not ok. A missing SASS listing is still recorded, as `sass.ok` false. Nothing the capture computes changed:

- **The comparison, run again at this head.** Run `r20261112-231418-ada0` took the first card run's saved capture and
  compared it with this head's script: 48,234,496 words, 0 differ; the evaluator's 73 tokens, 299,008 words, 0 differ; and
  the record it wrote is byte for byte the card run's record (digest `1002e7be…bb9a96e`).
- **The launch stage at this head.** It has not run on a card. Its lines changed only by the removed handlers, and in both runs
  of record every check was true and no step recorded an error, so the expressions that ran there are the ones this
  head runs. A capture at the landed head would close this.

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

## Suites and lints

Run `r20261112-231418-ada0` at this head; `out/suites.json` and `out/suites-repository.json` are in the run.

- **`suites.py --changed 4175265019 --quick`**, the suites that read the nine files: 57 suites, 57 passed (8 of them a
  cached pass reused), 0 failed. Among them `verity-vllm` (14,366 passed, 401 skipped), `verity/definitions` (906 passed),
  `circuit-check` (376 passed, 2 skipped) and `repository` (352 passed).
- **`suites.py repository`, whole:** 356 passed.
- **The retired-terms lint** (`tests/lint/test_repo_retired_terms.py`) on the nine changed files: 0 sites in each [run on
  my VM].
- **The adapter's lint ratchet** (`adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json`): `failures: []` [run on
  my VM].
- **The registered FlashMLA Definitions are unchanged:** the program digests of 13 registered bindings at this head equal
  those computed before the edit of `registry/flashmla.py` [computed on my VM].

## Runs

| run | head | what I read in it |
| --- | --- | --- |
| `r20261112-181549-6fce` | `1f6f0969a9` | Capture of record, node A's card 1: 30 cases, 1,472 tokens, 48,234,496 words compared, 0 differ; evaluator 73 tokens, 299,008 words, 0 differ; the probes; the SASS. rc 0. |
| `r20261112-183549-43fa` | `1f6f0969a9` | The same capture again: the same counts, and the same saved bytes but for the output path in `launch.json`. rc 0. |
| `r20261112-185922-3b8c` | `1f6f0969a9` | Off the card, from the first run's saved capture: every token to the Definition's evaluator, 1,472 tokens, 6,029,312 words, 0 differ. rc 0. |
| `r20261112-231418-ada0` | this head | The saved capture compared again (the card run's record byte for byte); the pins update pass (no file changed); circuit-check (above); the suites (above). rc 0. |
| `r20261112-173637-e88d` | `1b61bd883e` | circuit-check before the merge of `main`: the same verdicts and sizes as above. |
| `r20261112-170240-ac18` | `1b61bd883e` | `verity-vllm`, `verity/definitions` and `circuit-check` suites, quick tier: 3 passed. Superseded by the run at this head. |
| `r20261112-153801-6d0c` | `55d7a918bb` | The pre-mark suites on an earlier head: 56 of 57 passed; the one failure was the adapter's names lint on the module's first name, since renamed. Superseded. |
| `r20261112-152809-d14e` | earlier head | The pins update pass: no file changed. Superseded. |
| `r20261112-150200-2bd9` | earlier head | The first circuit-check at these sizes: 18 targets, 0 failures. Superseded. |
| `r20261112-191237-3c6f` | `1f6f0969a9` | Node A's cores, no card: one token's evaluation by the Definition's evaluator peaks at 919 MiB, 4,096 words, 0 differ. |
| `r20261112-180535-23a8`, `r20261112-180918-ecaf` | earlier heads | A quick first pass on the card and a SASS-only pass, from which the listing was fixed to be made image by image. Superseded by the two captures of record. |
| `r20261112-183708-9a9a` | `1f6f0969a9` | Cancelled in the queue before it took a slot; `r20261112-185922-3b8c` replaced it. |

## Not run

- **Nothing was served.** The capture is the kernel alone, called by the pin's method on a stand-in for the backend, on
  random and planted rows. No request went through vLLM's engine, and no checkpoint was read.
- **Which T a served step has** is read at the pin's source, not run: the decode and prefill routing, and that under a full
  CUDA graph the attention's token count is the graph's padded size (`gpu_model_runner.py:2490-2505`, `:4466-4492`).
- **The whole token at served size** was compared by circuit-check on edge vectors only, and its partition check did not
  run on the whole; its units were each checked on random vectors.
- **The SASS comparison with the registered instance** is a census of floating-point opcodes, not a line-by-line diff.
- **The index conversion** from a request's positions to cache slots is not modelled, and the 56 pad heads have no
  Definition (the capture holds their outputs to the reference on a zero query).
- **No Boolean `_v2` forms** of the new Definitions.
- **Not wired, by the brief:** the capture's record is not in the property registry, its test is not in
  `suite_parts.SECONDS`, and `serving_templates` does not stage the cut. Those files are the adapter's.
- **The write and the dense read-back** are other issues' Definitions; this PR states the row's layout from the pin's
  source and reads neither.
