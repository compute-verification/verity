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
  same rows through the dense prefill's read-back and counts where the two reads differ.
