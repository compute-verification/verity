"""P_compiled relations: the numerics of every Inductor-generated kernel of the production compiled Llama execution, read from the
kernel sources (each relation reproduces the implementation's materialized outputs bit-for-bit on every launch of the B1 row).

Kernel -> relation (`relation()`): matmul_kernel_persistent = GEMM (existing Definition, vLLM's batch-invariant Triton GEMM);
triton_red_fused__to_copy_embedding_rms_norm_t_0 = EmbedRMSNormTriton; triton_red_fused_fused_add_rms_norm_t_0 = AddRMSNormTriton (the
eager bf16 residual rounding elided, no residual store); triton_red_fused_fused_add_rms_norm_t_2 / _2 = AddAddRMSNormTriton (residual
recomputed unrounded from three inputs; one store rounding); triton_poi_fused_mul_silu_slice_t_1 = SiluMulTriton; triton_poi_fused_1/_3 =
RoPETriton (q and k in one kernel, bf16 cos|sin table, f32 math, one rounding).  The norms' tl.sum order is a MODEL SET
(`_triton_sum_models`): the one exact on every S2 launch is "8 contiguous positions per thread, xor-butterfly over 32 lanes, then over 8
warps" (labels `r0=2048 warps=8 chunk=8 sep|fma`; sep/fma coincide on bf16-derived data).

The norm relation OF RECORD is the literal Triton transcription of the generated source launched with the executed config
(compiled_norm_literal.py, labels `triton-literal XBLOCK=.. R0_BLOCK=.. warps=.. (executed)`); the torch model set stays as witnesses.

Argument roles per kernel (`ROLES`: hook arg index -> role; `WEIGHT_ARGS`: registered inputs, committed by reference only) are read off
the wrapper `call()` signatures of the same artifacts.
"""
from __future__ import annotations

import importlib.util

from verity_vllm.program.compiled_norm_literal import literal_variants

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
    import re

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
    """Make this process's first MKL VML call on one thread; False without torch, and a no-op after the first call.  torch's CPU `exp`
    (the SiLU relation's), `log`, `tanh`, `erf` and the other VML functions run in the MKL torch links statically, whose first-call
    initialisation races when several intra-op threads call at once: one thread's whole chunk then comes out at about 2e-4 relative
    error (1771 ulp).  One call of any VML function initialises them all, so every entry point that may evaluate these relations calls
    this before its first torch op."""
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


def _fma(a, b, c):
    """fma(a,b,c) in f32 emulated through f64 (a*b exact in f64; one f64 rounding then f32 -- double rounding possible in rare halfway cases)."""
    return (a.double() * b.double() + c.double()).float()


def _triton_sum_models(r, labels=None):
    """Candidate orders for the norm kernels' sum of squares over a [M, 2048] f32 row: Inductor's loop accumulates x*x per tile position
    across the r0 iterations (separate mul+add, or contracted into an fma), then tl.sum reduces the R0_BLOCK positions: each thread
    accumulates the positions it holds (blocked layout: chunks of k contiguous positions per thread), a butterfly (xor) shuffle tree
    across the 32 lanes of a warp, then across warps.  Yields (label, sums[M]); `labels` restricts to a subset."""
    if labels is None or "torch.sum" in labels:
        yield "torch.sum", (r * r).sum(-1)
    N = int(r.shape[1])
    # the reduction tile r0 is R0_BLOCK: a power of two (Inductor's reduction heuristic: size hint 2048 -> R0_BLOCK 2048,
    # `r0_mask = r0_index < r0_numel` zero-masks the tail) -- for Qwen2.5-1.5B (row 1536) the EXECUTED config is XBLOCK=2 R0_BLOCK=2048
    # num_warps=16 (loaded_code_objects executed_configs), i.e. the same [2, 2048] tile as Llama's with positions 1536..2047 held as zeros;
    # divisors of the row are kept as candidates for a looped (R0_BLOCK < row) config.
    r0s = sorted({x for x in (N, N // 2, N // 4, N // 8, 2048, 1024, 512, 256) if x > 0 and x <= N and N % x == 0}
                 | {p for p in (256, 512, 1024, 2048) if p >= N}, reverse=True)
    for r0 in r0s:
        for nw in (1, 2, 4, 8, 16):
            T = 32 * nw
            if r0 % T:
                continue
            e = r0 // T
            for chunk in sorted(c for c in {1, 2, 4, 8, e} if 1 <= c <= e and e % c == 0):     # contiguous positions per thread must tile the thread's e positions
                for acc in ("sep", "fma"):
                    label = f"r0={r0} warps={nw} chunk={chunk} {acc}"
                    if labels is not None and label not in labels:
                        continue
                    yield label, _triton_sum(r, r0, nw, chunk, acc)


def _triton_sum(r, r0, nw, chunk, acc_mode):
    import torch

    M, N = r.shape
    T = 32 * nw
    e = r0 // T
    acc = torch.zeros((M, r0), dtype=torch.float32, device=r.device)
    for off in range(0, N, r0):
        blk = r[:, off: off + r0]
        if blk.shape[1] < r0:                                          # R0_BLOCK > row: the masked tail loads `other=0.0` -> the position adds an exact 0
            blk = torch.nn.functional.pad(blk, (0, r0 - blk.shape[1]))
        acc = _fma(blk, blk, acc) if acc_mode == "fma" else acc + blk * blk
    # positions held by thread t: chunks of `chunk` contiguous positions, strided by T*chunk:  pos = c*(T*chunk) + t*chunk + j
    per = acc.reshape(M, e // chunk, T, chunk).permute(0, 2, 1, 3).reshape(M, T, e)
    th = per[:, :, 0]
    for j in range(1, e):
        th = th + per[:, :, j]
    v = th.reshape(M, nw, 32)
    for o in (16, 8, 4, 2, 1):
        idx = torch.arange(32, device=r.device) ^ o
        v = v + v[:, :, idx]
    w = v[:, :, 0]
    o = nw // 2
    while o >= 1:
        idx = torch.arange(nw, device=r.device) ^ o
        w = w + w[:, idx]
        o //= 2
    return w[:, 0]



_SILU_KERNEL = None


def _silu_mul_triton(x: "torch.Tensor"):
    """The generated `triton_poi_fused_mul_silu_slice_t_1` body, transcribed statement for statement (S1: the relation is the source):
    tmp0/tmp8 = the two halves of a row in f32; tmp3 = libdevice.exp(-tmp1); tmp6 = tmp1 / (tmp3 + 1); out = (tmp6 * tmp8) stored bf16.
    None when Triton is unavailable or x is not on CUDA (the torch variant stands alone; the report names the variant that matched)."""
    global _SILU_KERNEL
    import torch
    if not x.is_cuda:
        return None
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001
        return None
    if _SILU_KERNEL is None:
        @triton.jit
        def _k(in_ptr0, out_ptr0, xnumel, H: tl.constexpr, XBLOCK: tl.constexpr):
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex % H
            x1 = xindex // H
            tmp0 = tl.load(in_ptr0 + (x0 + 2 * H * x1), xmask).to(tl.float32)
            tmp8 = tl.load(in_ptr0 + (H + x0 + 2 * H * x1), xmask).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = -tmp1
            tmp3 = libdevice.exp(tmp2)
            tmp4 = tl.full([1], 1.0, tl.float32)
            tmp5 = tmp3 + tmp4
            tmp6 = (tmp1 / tmp5)
            tmp7 = tmp6.to(tl.float32)
            tmp9 = tmp7 * tmp8
            tl.store(out_ptr0 + xindex, tmp9, xmask)
        _SILU_KERNEL = _k
    xc = x.contiguous()
    H = xc.shape[1] // 2
    out = torch.empty((xc.shape[0], H), dtype=torch.bfloat16, device=xc.device)
    n = out.numel()
    try:
        _SILU_KERNEL[(triton.cdiv(n, 1024),)](xc, out, n, H=H, XBLOCK=1024)
    except Exception:  # noqa: BLE001
        return None
    return out

def rms_eps_of(model, default: float = 1e-5) -> tuple[float, str]:
    """The model's RMSNorm epsilon -- the generated norm kernels bake it as a literal (`tl.full([1, 1], 1e-06, ...)`: Llama-3.2 1e-5,
    Qwen2.5 1e-6), so the relation must use the model's, not a default: from the norm modules' variance_epsilon (vLLM RMSNorm), else
    config.rms_norm_eps, else `default` (source named; a default is a finding, not a value of record)."""
    for m in getattr(model, "modules", lambda: [])():
        e = getattr(m, "variance_epsilon", None)
        if e is not None:
            return float(e), "RMSNorm.variance_epsilon"
    e = getattr(getattr(model, "config", None), "rms_norm_eps", None)
    if e is not None:
        return float(e), "config.rms_norm_eps"
    return float(default), "default (no RMSNorm module / config found)"


def relation(kernel: str, ins: list, outs_meta: list, eps: float = 1e-5, labels=None) -> dict:
    """Evaluate the registered relation for one launch from its materialized inputs.  Returns {"kind", "outputs": [tensor...] or
    "variants": {label: [tensor...]}}."""
    import torch

    k = kernel
    if "matmul_kernel_persistent" in k:
        a, b = ins[0], ins[1]
        return {"kind": "gemm", "outputs": [torch.mm(a, b)]}
    # Norm family: the relation OF RECORD is the literal Triton transcription of the generated source launched with the executed config
    # (compiled_norm_literal: the same lowering of the same statements -- the Qwen run-8 residue of 3 launches x 1 element x 1 bf16 ulp was
    # the compiler's fma(s, 1/N, eps) contraction of `s / N + eps` (one rounding) against the torch model's two roundings, exact only when
    # N is a power of two); the torch models (hand-modelled order, torch division and rsqrt) remain as witnesses under their labels.
    if "embedding_rms_norm" in k:
        ids, table, w = ins[0], ins[1], ins[2]
        x = table[ids.long()]                                                 # bf16 rows
        xf = x.float()
        variants = dict(literal_variants("embed_norm", k, ins, eps))
        for label, s in _triton_sum_models(xf, labels):
            inv = torch.rsqrt(s / xf.shape[1] + eps)
            variants[label] = [x, ((xf * inv[:, None]) * w.float()).to(torch.bfloat16)]
        return {"kind": "embed_norm", "variants": variants}
    if "rms_norm" in k:
        tens = [t for t in ins if t is not None]
        w = [t for t in tens if t.dim() == 1][-1]
        xs = [t for t in tens if t.dim() != 1]
        if len(xs) == 2:
            r = xs[0].float() + xs[1].float()
            kind = "add1_norm"
        else:
            r = xs[0].float() + (xs[1].float() + xs[2].float())
            kind = "add2_norm" if len(outs_meta) == 2 else "final_norm"
        variants = dict(literal_variants(kind, k, ins, eps))
        for label, s in _triton_sum_models(r, labels):
            inv = torch.rsqrt(s / r.shape[1] + eps)
            out = ((r * inv[:, None]) * w.float()).to(torch.bfloat16)
            variants[label] = [r.to(torch.bfloat16), out] if kind == "add2_norm" else [out]
        return {"kind": kind, "variants": variants}
    if "silu" in k:
        x = ins[0]
        H = x.shape[1] // 2
        g = x[:, :H].float()
        u = x[:, H:].float()
        # two evaluations of the SAME expression tree as the generated source (tmp1 / (libdevice.exp(-tmp1) + 1) * tmp8, one bf16 store
        # rounding): torch f32 (IEEE div.rn, CUDA expf) and the LITERAL transcription run through Triton (Triton lowers f32 `/` to
        # div.full.f32 -- not correctly rounded -- and libdevice.exp to __nv_expf; the B8 prefill showed 1 element in 6.3 M where the two
        # differ by one bf16 ulp).  The kernel's numerics ARE the Triton lowering: that variant is the relation of record; torch is a witness.
        variants = {"torch-f32": [((g / (torch.exp(-g) + 1.0)) * u).to(torch.bfloat16)]}
        lit = _silu_mul_triton(x)
        if lit is not None:
            variants["triton-literal"] = [lit]
        return {"kind": "silu_mul", "variants": variants}
    # RoPE family (neox rotate-half, bf16 cos|sin table row per position, f32 math, one bf16 store rounding).  Three generated forms:
    #   Llama   [qkv, pos, cache]        -> k [M,nh_k,hd], q [M,nh_q,hd]           (one kernel, both heads sets)
    #   Qwen2.5 [qkv, bias, pos, cache]  -> k [M,nh_k,hd], qkv_b [M,W]              (k rotated from the UNROUNDED f32 qkv+bias; qkv+bias stored bf16)
    #   Qwen2.5 [qkv_b, pos, cache]      -> q [M,nh_q,hd]                           (q rotated from the bf16 qkv+bias buffer)
    # Column layout of the qkv buffer: q [0, nh_q*hd) | k [q_size, q_size+kv) | v; q_size = W - 2*kv, kv = nh_k*hd (GQA or MHA alike).
    ptrs = [t for t in ins if t is not None]
    pos_i = next((i for i, t in enumerate(ins) if t is not None and t.dtype == torch.int64 and t.dim() == 1), None)
    if pos_i is not None and len(ptrs) >= 3 and ins[0] is not None and ins[0].dim() == 2:
        cache = next((t for t in ins[pos_i + 1:] if t is not None and t.dim() == 2 and t.dtype != torch.int64), None)
        bias = next((t for t in ins[1:pos_i] if t is not None and t.dim() == 1 and t.dtype != torch.int64), None)
        if cache is not None:
            qkv, pos = ins[0], ins[pos_i]
            hd = int(cache.shape[-1])
            W = int(qkv.shape[1])
            heads = [om for om in outs_meta if len(om["shape"]) == 3]
            has_full = any(len(om["shape"]) == 2 for om in outs_meta)         # the bias-added qkv buffer is an output: the Qwen k kernel
            # which head output is q / k: by role name when the caller gives it; else two outputs -> the larger head count is q; one
            # output -> k when the kernel also stores the full qkv+bias buffer (Qwen form 2), q otherwise (Qwen form 3)
            def _is_q(om) -> bool:
                if om.get("role") in ("q", "k"):
                    return om["role"] == "q"
                if len(heads) == 2:
                    return om["shape"][1] == max(h["shape"][1] for h in heads)
                return not has_full
            kv = None
            for om in heads:
                if not _is_q(om):
                    kv = int(om["shape"][1]) * int(om["shape"][2])
            if kv is None and len(heads) == 1:                                # only q is stored: q_size = nh_q*hd, the rest splits in k | v
                kv = (W - int(heads[0]["shape"][1]) * int(heads[0]["shape"][2])) // 2
            if kv is not None and 0 < kv <= W:
                q_size = W - 2 * kv
                base = qkv.float() + bias.float() if bias is not None else qkv.float()
                row = cache[pos.long()].float()                               # [M, hd] = cos[0:hd/2] | sin[0:hd/2]
                half = hd // 2
                cos, sin = row[:, None, :half], row[:, None, half:]
                variants = {}
                for fma_mode in ("plain", "fma_first", "fma_second"):
                    outs = []
                    for om in outs_meta:                                      # in signature order
                        if len(om["shape"]) != 3:                             # qkv_b: the bias-added buffer, bf16
                            outs.append(base.to(torch.bfloat16))
                            continue
                        nh = int(om["shape"][1])
                        col0 = 0 if _is_q(om) else q_size
                        x = base[:, col0: col0 + nh * hd].reshape(-1, nh, hd)
                        x1, x2 = x[..., :half], x[..., half:]
                        if fma_mode == "plain":
                            lo = x1 * cos - x2 * sin
                            hi = x2 * cos + x1 * sin
                        elif fma_mode == "fma_first":
                            lo = _fma(x1, cos, -(x2 * sin))
                            hi = _fma(x2, cos, x1 * sin)
                        else:
                            lo = _fma(-x2, sin, x1 * cos)
                            hi = _fma(x1, sin, x2 * cos)
                        outs.append(torch.cat([lo, hi], -1).to(torch.bfloat16))
                    variants[fma_mode] = outs
                return {"kind": "rope_bias" if bias is not None else "rope", "variants": variants}
    return {"kind": "unknown", "outputs": None}


# ------------------------------------------------------------------------------------------------------------------------------ replay
