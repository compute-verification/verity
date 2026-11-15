"""The op sets training_slowdown counts, by aten name, with no torch: training_slowdown.flops and training_slowdown.capture build their aten sets from
them, and training_slowdown.gates prices a record's ops by them."""
from __future__ import annotations

#: Parts of the names of ops that compute products. An op whose name has one, and that neither the counter nor
#: training_slowdown.capture has a formula for (an RNN kernel, `_trilinear`, a packed or sparse matmul), makes a run invalid or a
#: capture refused, since its work would count as zero.
GEMM_LIKE = ("mm", "matmul", "conv", "linear", "attention", "einsum", "tensordot", "bilinear", "lstm", "gru", "rnn",
             "mv", "addr", "dot", "dist", "embedding_bag", "kron")
#: names with one of those parts that compute no products
NOT_GEMM = frozenset({"cummax", "cummin", "digamma", "digamma_", "igamma", "igamma_", "igammac", "igammac_", "lgamma",
                      "lgamma_", "mvlgamma", "mvlgamma_", "polygamma", "polygamma_", "_standard_gamma",
                      "_convert_weight_to_int4pack", "_convert_weight_to_int4pack_for_cpu", "upsample_linear1d",
                      "upsample_bilinear2d", "_upsample_bilinear2d_aa", "_upsample_bilinear2d_aa_backward",
                      "upsample_trilinear3d", "upsample_linear1d_backward", "upsample_bilinear2d_backward",
                      "upsample_trilinear3d_backward", "_record_function_enter_new", "_record_function_exit",
                      "conj_physical", "native_dropout_backward"})
#: The non-GEMM op sets by aten name, which training_slowdown.capture's static count (`capture.structure`) reads too. Division is
#: a multiplication by a reciprocal.
ELEMENTWISE_NAMES = frozenset({"mul", "mul_", "div", "div_", "addcmul", "addcmul_", "addcdiv", "addcdiv_", "lerp", "lerp_",
                               "_foreach_mul", "_foreach_mul_", "_foreach_div", "_foreach_div_", "_foreach_addcmul",
                               "_foreach_addcmul_", "_foreach_addcdiv", "_foreach_addcdiv_", "_foreach_lerp",
                               "_foreach_lerp_"})
#: counted as elementwise multiplications only with a scale (`alpha` other than 1)
SCALED_NAMES = frozenset({"add", "add_", "sub", "sub_", "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_"})
#: fused optimizer steps, each at a constant per parameter, never below the multiplications its unfused form makes at
#: any setting of the optimizer's options: above AdamW's and Adam's 6, and SGD's and Adagrad's own largest
FUSED_OPTIMIZER_MACS = {"_fused_adam_": 7, "_fused_adamw_": 8, "_fused_sgd_": 4, "_fused_adagrad_": 4}
LINALG_NAMES = frozenset({"_linalg_eigh", "linalg_eigh", "_linalg_svd", "linalg_svd", "linalg_qr", "linalg_cholesky_ex",
                          "linalg_inv_ex", "_linalg_solve_ex", "linalg_solve_triangular", "linalg_lu_factor_ex",
                          "cholesky_solve", "triangular_solve"})
#: ops whose result is gathered from their first argument: their work is the result, not the whole source
GATHER_NAMES = ("embedding", "index", "index_select", "take", "gather")


def gemm_like(name: str) -> bool:
    """Whether an aten op's name says it computes products (`GEMM_LIKE`)."""
    return name not in NOT_GEMM and any(s in name for s in GEMM_LIKE)
