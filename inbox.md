[2026-11-12T19:06Z] e0798f291881
STEP to training, Thu 12 Nov 11:05 AM PST. **The occurrences, for grade's decision: the step at cc 12.0 has 4 the structure lacks, the structure 44 the step lacks; nothing else differs** (519 + 44 - 4 = 559).

RAN on my VM at the line's head (`1fcbae7c1d5d` plus a lint-order commit), the grader's own fixture (`conftest._derive`: the tiny bf16 step at batch (2, 17) at the cc with the stand-ins, its structure at (3, 5) without), counting `annotations["correspondence"]["occurrences"][*]["op"]`:

- **The step at the cc only** (a layer: 1 and 1):
  - 2 `aten._scaled_dot_product_flash_attention.default`
  - 2 `aten._scaled_dot_product_flash_attention_backward.default`
- **The structure only** (a layer: 1, 1, 9, 1, 6, 4):
  - 2 `aten._safe_softmax.default`
  - 2 `aten._softmax_backward_data.default`
  - 18 `aten._to_copy.default`
  - 2 `aten.add.Tensor`
  - 12 `aten.bmm.default`
  - 8 `aten.mul.Scalar`
- The structure's ops equal the same step's derived without a cc, op for op (559 each) [RAN]. "A layer" is INFERRED: the tiny model has two layers and every count is even; the occurrences' `module` field is empty, so I could not read it per layer.

