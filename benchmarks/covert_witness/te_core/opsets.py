"""The aten ops te_core counts beyond matmuls, by overload-packet name: one source for the harness's counter (`flops.py`, which
resolves them on `torch.ops.aten`) and the graph capture (`capture.structure`, which matches traced nodes by name), so both
count the same work."""

#: one multiplication per output element
ELEMENTWISE = frozenset({"mul", "mul_", "addcmul", "addcmul_", "addcdiv", "addcdiv_", "lerp", "lerp_", "_foreach_mul",
                         "_foreach_mul_", "_foreach_addcmul", "_foreach_addcmul_", "_foreach_addcdiv", "_foreach_addcdiv_",
                         "_foreach_lerp", "_foreach_lerp_"})
#: one multiplication per output element when scaled (alpha ≠ 1), none otherwise
SCALED = frozenset({"add", "add_", "sub", "sub_", "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_"})
#: fused optimizer steps, at the multiplications per parameter their unfused forms make
FUSED_OPTIMIZERS = {"_fused_adam_": 7, "_fused_adamw_": 8, "_fused_sgd_": 2, "_fused_adagrad_": 4}
#: dense linear algebra, at n³ for an n × n operand
LINALG = frozenset({"_linalg_eigh", "linalg_eigh", "_linalg_svd", "linalg_svd", "linalg_qr", "linalg_cholesky_ex",
                    "linalg_inv_ex", "_linalg_solve_ex", "linalg_solve_triangular", "linalg_lu_factor_ex", "cholesky_solve",
                    "triangular_solve"})
