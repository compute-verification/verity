[2026-11-11T22:53Z] e380f0857ecd
SEAT (training), 2:52 PM PST (22:52Z). TRN-70, job A: the table. Nothing is deleted.

**Outcome.** The steps I could derive are the reference model's alone. They fire 17 of my files' 33 training rules (13 on the grader's steps, 4 more only under the model's own cross-entropy loss) and 3 of their 16 reference rules. 16 training rules fire on no step I derived. I derived none of the 29 searched programs of record and read none of their hand-overs, and 9 of the 16 lower an op the harness's model, routes or codecs call [READ]. So by your rule (a rule only they might fire is not deleted this week) I count all 16 as not deletable until a searched program's step is derived. Hand-in 1 is on VLLM-340 (22:30Z).

Every figure is RAN (the runs at the end) unless marked READ or INFERRED. A rule fires when the translator asks it about a node, it accepts, and its `emit` returns.

**What I derived and what I could not**

(a) The reference step.
- The `torch_step` pins: nothing to derive. `verity/definitions/registry/torch_step.py` is written by hand and imports nothing of the adapter [READ, and RAN `rg`], so the pins fire no rule.
- The grader's steps: derived, all 5 its suite derives. Each is the tiny reference model under `square_loss`: f32 at 2 rows of 17 tokens, f32 at 3 rows of 5, bf16 at 2x17, bf16 at 3x5, and bf16 at 2x17 at cc 12.0 on the suite's stand-in routes.
- 10 more settings of the reference step: derived whole. Seven are the tiny size at 2 rows of 16 tokens and a vocabulary of 64 (f32 and bf16, `square_loss` and the model's own `loss_fn`, with and without cc 12.0; three of them repeat a grader's setting at another shape); three are the `s` size at the same 2x16 tokens and vocabulary.
- The `s` step at its own shape (2 rows of 1,024 tokens, vocabulary 50,304): not derived. In f32 it passed the 30 GiB of address space I capped it at after 154 s, having fired the same 15 training rules and 2 reference rules as the tiny step of that setting. In bf16 at cc 12.0 (vocabulary 32,000, the shape the route capture covers) it raised `MemoryError` under the same cap.

(b) The steps the harness derives through the translator: there are none. At `454db30ccc2c` the harness package imports nothing of `verity_vllm` [RAN `rg`]. The routes' captures and the port check's 29 entries go through the harness's own tracer and counter. Its suite under the counter: 432 passed, 0 rule fires. Outside the adapter only the grader names the translator, and the grader reads hand-overs; it derives only in its tests, which is (a).

(c) The runs of record: none derived, none read. The 29 entries' records are counter records, not Programs. A step of theirs meets the translator only when a hand-over is made, and the hand-overs came from scratch jobs outside the tree [READ: the hill-climb inventory note, row 27]. I found no run or artifact id for one in the lane's notes, and this VM has no store index. The tree's own entry cannot stand in: `step_program` takes only a `torch.optim.AdamW` with one parameter group [READ], and the routes step with Muon, Shampoo and an average.

**Searched programs of record I could not derive: all 29** [READ `of_record.toml`]. `l16-t20-1.1x-fullmuon-fe-wsd`, `l16-t20-1.1x-fullmuon-fe-wsd-5383`, `l16-t20-1.1x-fullmuon-fe-wsd-5383-ema`, `l16-t20-1.1x-fullmuon-fe-wsd-ema`, `l16-t20-1.1x-ref-optim-fullmuon-fe-wsd`, `l16-t20-1.1x-ref-optim-fullmuon-wsd`, `m-t20-1.1x-fullmuon-wsd`, `m-t20-1.1x-fullmuon-wsd-4611`, `m-t20-1.1x-fullmuon-wsd-4611-ema`, `m-t20-1.1x-fullmuon-wsd-ema`, `m-t20-1.1x-ref-optim-1372`, `m-t20-1.1x-ref-optim-fullmuon`, `m-t20-1.1x-shallow-depthshare-4`, `m-t20-1.1x-shallow-depthshare-8`, `m-t20-1.1x-wildcard-topcode`, `s-t20-1.1x-fullmuon`, `s-t20-1.1x-fullmuon-ema`, `s-t20-1.1x-fullmuon-fe-wsd`, `s-t20-1.1x-fullmuon-fe-wsd-ema`, `s-t20-1.1x-ref-optim-1027`, `s-t20-1.1x-ref-optim-747`, `s-t20-1.1x-ref-optim-961`, `s-t20-1.1x-ref-optim-fullmuon-758`, `s-t20-1.1x-ref-optim-fullmuon-841`, `s-t20-1.1x-shallow-depthshare-3`, `s-t20-1.1x-shallow-depthshare-6`, `s-t20-1.1x-shallow-strideshare`, `s-t20-1.1x-wildcard-topcode`, `xl-t20-1.1x-ref-optim-fullmuon-fe-wsd`.

**Training rules a derived step fires: 17 of 33**

| rule | grader's steps (of 5) | other settings (of 10) | one step | my test functions that fire it; that only see it refuse | ops of it only tests fire |
|---|---|---|---|---|---|
| `Addcmul` | 5 | 10 | `grader f32 2x17` | 7; 0 | none |
| `Binary` | 5 | 10 | `grader f32 2x17` | 23; 3 | none |
| `EmbeddingDenseBackward` | 5 | 10 | `grader f32 2x17` | 7; 1 | none |
| `Lerp` | 5 | 10 | `grader f32 2x17` | 8; 0 | `_foreach_lerp.Scalar` |
| `NllLossBackward` | 0 | 7 | `tiny-f32-xent` | 3; 0 | none |
| `NllLossForward` | 0 | 7 | `tiny-f32-xent` | 6; 0 | none |
| `Reduce` | 5 | 10 | `grader f32 2x17` | 9; 0 | `mean.default` |
| `Unary` | 5 | 10 | `grader f32 2x17` | 16; 2 | `abs`, `exp`, `floor`, `reciprocal` |
| `Cast` | 1 | 3 | `grader bf16 2x17 cc12.0` | 12; 1 | none |
| `Matmul` | 5 | 10 | `grader f32 2x17` | 17; 2 | `mv` |
| `FullReduce` | 1 | 1 | `grader bf16 2x17 cc12.0` | 9; 2 | `linalg_vector_norm`, `sum.dim_IntList` |
| `LogSoftmax` | 0 | 7 | `tiny-f32-xent` | 5; 0 | none |
| `LogSoftmaxBackward` | 0 | 7 | `tiny-f32-xent` | 4; 0 | none |
| `SafeSoftmax` | 5 | 10 | `grader f32 2x17` | 8; 0 | none |
| `SoftmaxBackward` | 5 | 10 | `grader f32 2x17` | 7; 0 | none |
| `BoxPermute` | 5 | 10 | `grader f32 2x17` | 7; 0 | none |
| `StaticFloats` | 5 | 10 | `grader f32 2x17` | 21; 2 | 16 of its 21 ops |

`NllLossForward`, `NllLossBackward`, `LogSoftmax` and `LogSoftmaxBackward` fire only under the reference model's own loss; the grader's steps use `square_loss`. `Cast` fires only on a bf16 step at a cc, and `FullReduce` only on `square_loss` at a cc.

**Training rules no derived step fires: 16 of 33, 646 lines of rule classes**

| rule | file | what it lowers [READ] | lines | my test functions that fire it; that only see it refuse | what in the harness runs that op [READ the line; INFERRED that its step would fire the rule] |
|---|---|---|---|---|---|
| `Addcdiv` | `training.py` | `addcdiv` of f32 | 34 | 2; 0 | torch's own Adam step, unfused |
| `ForeachAdd` | `training.py` | `_foreach_add.Scalar` | 30 | 2; 0 | an optimizer's step counts |
| `FusedAdamW` | `training.py` | `_fused_adamw` at a cc | 86 | 1; 0 | the `fused` dial, `routes/subset.py` line 53 |
| `Log` | `training.py` | `log` at a cc | 27 | 1; 1 | none found |
| `SiluBackward` | `training.py` | `silu_backward` | 39 | 3; 0 | the harness's model, `model.py` line 60 |
| `FlashSdpa` | `training_attention.py` | `_scaled_dot_product_flash_attention` of bf16 at a cc | 62 | 2; 1 | the harness's model on a GPU, `model.py` line 57 |
| `AMax` | `training_masks.py` | `amax` of f32 | 21 | 1; 1 | `ScaledQuant`, `codecs/__init__.py` line 228 |
| `Clamp` | `training_masks.py` | `clamp`, `clamp_min`, `clamp_max` | 56 | 6; 1 | `routes/_optimizers.py` lines 98 to 114; `codecs/__init__.py` line 231 |
| `Compare` | `training_masks.py` | `gt`, `ge`, `lt`, `le`, `eq`, `ne` | 42 | 7; 1 | `codecs/__init__.py` lines 227 and 231 |
| `Copy` | `training_masks.py` | `copy` (a functionalized `copy_`) | 39 | 3; 1 | `routes/ema.py` line 61 |
| `Index` | `training_masks.py` | `index.Tensor` of a 1-D f32 table | 38 | 5; 0 | `QuantDelta`, `codecs/__init__.py` line 167 |
| `IntOps` | `training_masks.py` | int64 `+`, `<<`, `>>`, `&` | 25 | 3; 3 | none found |
| `IntSum` | `training_masks.py` | `sum` of bool or int64 | 26 | 4; 0 | none found |
| `KthValue` | `training_masks.py` | `kthvalue` at a cc | 67 | 5; 1 | none found |
| `Logic` | `training_masks.py` | bool and, or, not | 24 | 3; 0 | none found |
| `Where` | `training_masks.py` | `where.self` | 30 | 4; 0 | `codecs/__init__.py` lines 227 and 231 |

Steps that fire each of the 16: none. The ten of `training_masks.py` are written for QAT's sparse quantizer [READ `test_train_quant.py`'s docstring]; which program of record trains with it I did not find.

**Reference rules: 3 of 16 fire on a training step**

| rule | grader's steps (of 5) | other settings (of 10) | one step | my test functions that fire it; that only see it refuse | the adapter's other test files that fire it |
|---|---|---|---|---|---|
| `ArgmaxRefRule` | 0 | 0 | none | 0; 0 | 4 |
| `BroadcastBiasAddRefRule` | 0 | 0 | none | 0; 3 | 2 |
| `CastRefRule` | 3 | 6 | `grader bf16 2x17 cc12.0` | 5; 1 | 2 |
| `ElementwiseRefRule` | 5 | 10 | `grader f32 2x17` | 10; 3 | 7 |
| `EmbeddingRefRule` | 5 | 10 | `grader f32 2x17` | 6; 0 | 4 |
| `IntCatNarrowRule` | 0 | 0 | none | 0; 0 | 3 |
| `LayerNormRefRule` | 0 | 0 | none | 0; 0 | 3 |
| `LinearRefRule` | 0 | 0 | none | 0; 0 | 6 |
| `MatmulRefRule` | 0 | 0 | none | 0; 0 | 2 |
| `RMSNormRefRule` | 0 | 0 | none | 0; 0 | 1 |
| `SdpaRefRule` | 0 | 0 | none | 0; 0 | 2 |
| `SoftmaxRefRule` | 0 | 0 | none | 0; 0 | 2 |
| `EagerAttentionPatternRule` | 0 | 0 | none | 0; 0 | 2 |
| `HFRMSNormPatternRule` | 0 | 0 | none | 0; 2 | 2 |
| `HFRoPEPatternRule` | 0 | 0 | none | 0; 2 | 2 |
| `HFRotaryTableRule` | 0 | 0 | none | 0; 1 | 2 |

The 13 that fire on no training step fire in eight other test files of `tests/program/` [RAN, the partial run], none of them mine: `test_derive.py`, `test_derive_hf5.py`, `test_derive_transfer.py`, `test_derive_negative.py`, `test_registered_value_rules.py`, `test_harden_moe.py`, `test_harden_guards.py` and `test_boolean_switch.py`. Every fire of a reference rule there is in mode `reference`, and in the adapter's library only `train_step.py` names that mode [RAN `rg`]. `LinearRule`, also in `reference.py`, is in no rule list itself: it is the `emit` of profile's `ProfileLinearRule`, which 19 test files fire. No test file outside my eight fires a training rule.

**Tests: 425 cases in my 8 files, all passed under the counter**

| file | cases | only rules a step fires | rules of both kinds | only rules no step fires | no rule of mine |
|---|---|---|---|---|---|
| `test_train_draws.py` | 30 | 4 | 0 | 0 | 26 |
| `test_train_embedding.py` | 4 | 4 | 0 | 0 | 0 |
| `test_train_log.py` | 3 | 0 | 0 | 3 | 0 |
| `test_train_matmul.py` | 8 | 8 | 0 | 0 | 0 |
| `test_train_quant.py` | 77 | 12 | 42 | 23 | 0 |
| `test_train_reduce.py` | 24 | 24 | 0 | 0 | 0 |
| `test_train_sdpa.py` | 11 | 0 | 0 | 10 | 1 |
| `test_train_step.py` | 268 | 239 | 12 | 15 | 2 |
| all | 425 | 291 | 54 | 51 | 29 |

A case counts for a rule when the rule fired in it, or was asked about the node its derivation was refused at. Test functions with cases only on rules no step fires (17 functions wholly, 35 cases; the rest in part):
- `test_train_log.py`: `test_cpu_log_is_refused` (all 1: Log); `test_gpu_log_is_logf_without_flush_to_zero` (all 2: Log)
- `test_train_quant.py`: `test_a_value_table_lookup` (all 2: Index); `test_a_zero_bound_is_the_gpus` (all 1: Clamp); `test_a_zero_bound_is_torch_cpus` (all 1: Clamp); `test_f32_clamps` (all 2: Clamp); `test_int64_clamps` (all 2: Clamp); `test_integer_sums` (all 2: Compare, IntSum); `test_kthvalue_is_the_gpus_radix_select` (all 5: KthValue); `test_kthvalue_is_torch_cpus_on_magnitudes` (all 1: KthValue); `test_kthvalue_of_many_slices_is_the_gpus_radix_select` (all 1: KthValue); `test_refusals` (4 of 19: Clamp, IntOps, KthValue); `test_static_amounts_across_a_wide_tensor` (all 2: IntOps)
- `test_train_sdpa.py`: `test_flash_forward_is_one_batch_of_the_captured_head` (all 4: FlashSdpa); `test_flash_forward_refusals` (all 5: FlashSdpa); `test_flash_forward_wires_each_heads_o_and_lse` (all 1: FlashSdpa)
- `test_train_step.py`: `test_copy_and_static_arithmetic_refusals` (1 of 4: Copy); `test_gpu_binary_is_the_gpus_arithmetic` (2 of 15: Addcdiv, ForeachAdd); `test_gpu_fused_adamw_is_its_kernels_arithmetic` (all 1: FusedAdamW); `test_gpu_silu_backward_is_its_stated_arithmetic` (all 2: SiluBackward); `test_rule_bit_exact` (3 of 132: Addcdiv, ForeachAdd); `test_rule_close` (3 of 40: SiluBackward); `test_rule_is_its_stated_arithmetic` (3 of 40: SiluBackward)

Test functions with cases on rules of both kinds, which a deletion would cut into:
- `test_train_quant.py`: `test_a_codebook_lookup` (2); `test_a_static_arange_with_a_row_scattered_in` (2); `test_amax` (2); `test_an_index_out_of_range_reads_a_nan` (1); `test_bit_planes_through_bools` (2); `test_comparisons_are_ieees` (12); `test_f16_comparisons_and_selects` (2); `test_int64_adds_shifts_and_ands` (2); `test_logic` (2); `test_qats_sparse_quantizer_definitions` (1); `test_qats_sparse_quantizer_is_the_gpus` (1); `test_refusals` (11); `test_where` (2)
- `test_train_step.py`: `test_an_allocation_read_before_a_copy_writes_it_is_refused` (1); `test_bf16_rounding_in_between_is_refused` (3); `test_copy_and_static_arithmetic_refusals` (2); `test_gpu_binary_definitions_and_refusals` (1); `test_gpu_binary_is_the_gpus_arithmetic` (1); `test_rule_bit_exact` (4)

Every other case tests only rules a step fires, or none of mine (26 of the 29 are `test_train_draws.py`'s refused random ops).

**Not checked**
- The searched programs' steps: none derived, none read. So "fires on no derived step" speaks of the reference model's steps only.
- The `s` step at its own shape. INFERRED that it fires no further rule: the `s` size at 2x16 tokens fired the same rules as the tiny size.
- Anything on a GPU. Every derivation ran on a CPU node; the steps at a cc are the exports the tests and the grader's suite take.
- Below a rule. The count is per rule and per op, not per branch of a rule's `emit`.
- Who a refusal test tests. A refusal counts for every rule asked at the refused node, so a rule standing by can put a case under both kinds. An `emit` that raises is no fire.
- The rest of the adapter's suite under the counter is partial: 14,922 cases ran (98% by pytest's progress line [READ]; 14,346 passed, 552 skipped, 24 failed) before I cancelled the run at 70.8 GiB against its 32 GiB reservation. The 24 are the scratch tree's, not my branch's: 23 want fixtures a bare `pytest` does not fetch [RAN four of them without the counter: the same failures; INFERRED for the other 19], 1 is ruff on my scratch files [RAN]. The last column of the reference table and "no other file fires a training rule" rest on that run.
- Whether a derivation of record outside training runs in mode `reference`. I read no caller that passes the mode as a variable.

**Asks**
1. The four scratch scripts and the route list that made the hand-overs, or a run id that holds them. With them I derive the searched programs' steps under the same counter, and the 16 get an answer. A hand-over alone tells me only what its annotations keep.
2. Your word on the 16. Until it comes I delete nothing and work the lists (71 entries) on the files of the rules that fire.

**Runs** (on nodes, from a scratch merge of the refactor head `0bfe402a6cf2` with `454db30ccc2c`; its three commits are on no remote [RAN `git ls-remote`]): my 8 test files `r20261111-214755-3b73` (425 passed); the grader's suite (28 passed, 3 skipped) and the harness's (432 passed) `r20261111-214806-a743`; the reference step's settings `r20261111-214644-f919`, `r20261111-221045-b9f7`, `r20261111-221114-6914`; the rest of the adapter's suite `r20261111-214825-9aed` (cancelled, partial). Every test function's row is in my note `lanes/training/20261111T2252Z-report-trn70-rules-the-reference-steps-fire.md`.

