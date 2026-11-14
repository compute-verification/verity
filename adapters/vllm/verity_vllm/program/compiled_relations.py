"""The argument roles of every Inductor-generated kernel of the production compiled Llama execution, and the one-thread warm-up of
MKL's VML.

Argument roles per kernel (`ROLES`: hook arg index -> role; `WEIGHT_ARGS`: registered inputs, committed by reference only) are read off
the wrapper `call()` signatures of the generated artifacts.
"""
from __future__ import annotations

import importlib.util
import re

__all__ = ["OUTPUT_ROLES", "pointer_arity", "roles_of", "warm_vml"]

# hook/wrapper argument index -> committed member name (roles), per kernel-name prefix; registered weights are not committed.
# a kernel NAME does not fix its argument list: Inductor names pointwise kernels `triton_poi_fused_<n>` in both the
# Llama RoPE (in_ptr0..2, out_ptr0..1: qkv, positions, cos|sin -> k, q) and the Qwen2.5 RoPE-with-QKV-bias (in_ptr0..3, out_ptr0..1: qkv,
# bias, positions, cos|sin -> k (rotated from the UNROUNDED f32 qkv+bias), qkv+bias (bf16)), and a name-keyed table applies the Llama
# roles to the Qwen kernel (the cos|sin table taken for the k output; 84 undefined dataflow edges).  The role table is therefore keyed by
# (name prefix, POINTER ARITY of the kernel's signature = the number of in_ptr*/out_ptr*/in_out_ptr* parameters, read from the generated
# source in the wrapper); a name whose arity has no entry is UNMODELLED (refused by the manifest as incomplete), never mis-roled.
ROLES: dict[str, dict[int, dict[int, str]]] = {
    "matmul_kernel_persistent": {4: {0: "a", 2: "out"}},                 # a_ptr, b_ptr, c_ptr, bias_ptr (-1: no bias; vLLM batch-invariant GEMM)
    "triton_red_fused__to_copy_embedding_rms_norm": {5: {0: "ids", 3: "emb", 4: "normed"}},
    "triton_red_fused_fused_add_rms_norm_t_0": {4: {0: "x0", 1: "x1", 3: "normed"}},
    "triton_red_fused_fused_add_rms_norm_t_2": {6: {0: "x0", 1: "x1", 2: "x2", 4: "res_out", 5: "normed"}},
    "triton_red_fused_fused_add_rms_norm_2": {5: {0: "x0", 1: "x1", 2: "x2", 4: "out"}},
    "triton_poi_fused_mul_silu_slice": {2: {0: "x", 1: "out"}},
    # Llama RoPE (q and k in one kernel)                       Qwen2.5 RoPE(k) + QKV bias (k from the f32 sum; qkv+bias stored bf16)
    "triton_poi_fused_": {5: {0: "qkv", 1: "pos", 3: "k", 4: "q"}, 6: {0: "qkv", 1: "bias", 2: "pos", 4: "k", 5: "qkv_b"}},
    # Qwen2.5 RoPE(q) on the bf16 qkv+bias buffer of the previous launch
    "triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_": {4: {0: "qkv_b_in", 1: "pos", 3: "q"}},   # arg 0 READS the bias-added buffer
}
WEIGHT_ARGS: dict[str, dict[int, set[int]]] = {
    "matmul_kernel_persistent": {4: {1}}, "triton_red_fused__to_copy_embedding_rms_norm": {5: {1, 2}}, "triton_red_fused_fused_add_rms_norm_t_0": {4: {2}},
    "triton_red_fused_fused_add_rms_norm_t_2": {6: {3}}, "triton_red_fused_fused_add_rms_norm_2": {5: {3}}, "triton_poi_fused_mul_silu_slice": {2: set()},
    "triton_poi_fused_": {5: {2}, 6: {1, 3}},
    "triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_": {4: {2}},
}
# roles that are OUTPUTS of their launch (the others are inputs produced outside the launch: by an earlier launch, the eager attention, or the engine)
OUTPUT_ROLES = {"out", "emb", "normed", "res_out", "k", "q", "qkv_b"}
_PTR_PARAM_RE = None                                        # parameter names that are pointers: in_ptr0 / out_ptr1 / in_out_ptr0 / a_ptr / c_ptr ...


def kernel_key(kernel: str) -> str | None:
    """Longest ROLES prefix matching an Inductor kernel name (`triton_red_fused_fused_add_rms_norm_t_2` before `triton_poi_fused_`)."""
    best = None
    for k in ROLES:
        if kernel.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    return best


def pointer_arity(signature: list[str] | None) -> int | None:
    """Number of pointer parameters of a generated kernel's `def` (Inductor's in_ptr*/out_ptr*/in_out_ptr*, the GEMM's a_ptr/b_ptr/c_ptr);
    None when no signature is known."""
    global _PTR_PARAM_RE
    if signature is None:
        return None
    if _PTR_PARAM_RE is None:
        _PTR_PARAM_RE = re.compile(r"^\w*_?ptr\d*$")
    return sum(1 for p in signature if _PTR_PARAM_RE.match(str(p).strip().split(":")[0].split("=")[0].strip()))


def roles_of(kernel: str, signature: list[str] | None = None, *, arity: int | None = None) -> tuple[dict[int, str], set[int]]:
    """(arg index -> role, weight arg indices) of a launch of `kernel`.  With the kernel's signature (or its pointer arity) the entry of that
    arity is returned -- none for that arity = UNMODELLED ({}, set()).  Without a signature the name's single entry is returned (the Llama
    callers: every Llama kernel name has exactly one arity); a name with several arities and no signature is unmodelled too (never a guess)."""
    k = kernel_key(kernel)
    if k is None:
        return {}, set()
    n = arity if arity is not None else pointer_arity(signature)
    table = ROLES[k]
    if n is None:
        if len(table) != 1:
            return {}, set()
        n = next(iter(table))
    if n not in table:
        return {}, set()
    return table[n], WEIGHT_ARGS[k][n]


_VML_WARM = False


def warm_vml() -> bool:
    """Make this process's first MKL VML call on one thread; False without torch, and a no-op after the first call.  torch's CPU `exp`,
    `log`, `tanh`, `erf` and the other VML functions run in the MKL torch links statically, whose first-call initialisation races
    when several intra-op threads call at once: one thread's whole chunk then comes out at about 2e-4 relative error (1771 ulp).  One
    call of any VML function initialises them all, so every entry point that replays on torch's CPU math calls this before its first
    torch op."""
    global _VML_WARM
    if _VML_WARM:
        return True
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.exp(torch.zeros(1))
    finally:
        torch.set_num_threads(n)
    _VML_WARM = True
    return True
