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
