[2026-11-12T19:07Z] 57e383b668b7
training to grade, Thu 12 Nov 11:07 AM PST. STEP's occurrence diff, for the structure rule in your 1 PM report (in case it reached only me).

The grader's own fixture (`conftest._derive`: the tiny bf16 step at batch (2, 17) at cc 12.0 with the stand-ins, its structure at (3, 5) without), counting `annotations["correspondence"]["occurrences"][*]["op"]` [STEP RAN, at its line's head 1fcbae7c1d5d plus a lint commit]:

- The step at the cc only, 4 (two layers, 1 and 1 each): 2 `aten._scaled_dot_product_flash_attention.default`, 2 `aten._scaled_dot_product_flash_attention_backward.default`.
- The structure only, 44 (a layer: 1, 1, 9, 1, 6, 4): 2 `aten._safe_softmax.default`, 2 `aten._softmax_backward_data.default`, 18 `aten._to_copy.default`, 2 `aten.add.Tensor`, 12 `aten.bmm.default`, 8 `aten.mul.Scalar`.
- Nothing else differs: 519 + 44 - 4 = 559. The structure's ops equal the same step's derived without a cc, op for op.

So per layer the fused pair stands where 22 decomposed occurrences stand, 6 of them `bmm`. "A layer" is STEP's inference: the `module` field is empty.

