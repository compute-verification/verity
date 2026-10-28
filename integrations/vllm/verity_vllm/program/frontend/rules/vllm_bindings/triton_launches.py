"""Direct Triton launches: the batch-invariant RMSNorm and GEMM, and the fused q / kv RMSNorm."""

from __future__ import annotations

import sys

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, norm_kind, rows_of, shape
from verity_vllm.program.frontend.rules.profile import BF16, _bias_shape
from verity_vllm.program.frontend.rules.triton_iface import launch_meta, launch_tensor, triton_launch
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value, _strides, _untransposed
from verity_vllm.program.frontend.rules.vllm_bindings.pins import (GEMM_PINS, QKV_RMS_PINS, RMS_PINS, _pins_ok, _target_kinds, active_gemm_target, gemm_bias_role,
                                                                   gemm_pins_for, lookup_key)
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity.ml.definitions.registry.dsa_norm import check_block


# -- direct Triton launches -----------------------------------------------------------------------------------------------


class TritonRmsNormRule(InPlaceRule):
    """`_rms_norm_kernel[grid](input_2d, weight, output, in_stride, out_stride, n_cols, eps, BLOCK_SIZE, HAS_WEIGHT)` launched by
    `rms_norm_batch_invariant` -> `RMSNormTriton{N,EPS}` per row; `output` (an allocation) is rebound."""
    modes = frozenset({"profile"})
    info = RuleInfo("TritonRmsNorm[vllm-bi]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of batch_invariant._rms_norm_kernel (kernel name + constexprs pinned)",
                    preconditions=("kernel qualified name is the profile's `_rms_norm_kernel`", "BLOCK_SIZE=1024, HAS_WEIGHT=True (profile pins)",
                                   "input is bf16 [R, N] with unit column stride (row stride = the launch's input_row_stride, >= N: a column slice of a wider "
                                   "row is allowed), weight bf16 [N], output an unwritten allocation [R, N] (row stride = output_row_stride)",
                                   "n_cols == N; eps is a static float; grid == (R,)"),
                    evidence_domain="structural; numerics: RMSNormTriton_v1 (the same Triton kernel, W11 model)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != RMS_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {RMS_PINS['kernel']}"
        sc = meta.get("scalars", {})
        bad = _pins_ok(meta, RMS_PINS)
        if bad:
            return bad
        x, w, out = (launch_tensor(node, meta, n) for n in ("input_ptr", "weight_ptr", "output_ptr"))
        if x is None or w is None or out is None:
            return "tensor arguments input_ptr/weight_ptr/output_ptr not all present"
        r = _row_value(ctx, x, "input") or _row_value(ctx, w, "weight")
        if r:
            return r
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if str(ctx.dtype_of(out)) != RMS_PINS["dtype"]:
            return f"numerics_unregistered: output dtype {ctx.dtype_of(out)}; RMSNormTriton_v1 is the bf16 kernel"
        xs = shape(ctx, x)
        if len(xs) != 2 or shape(ctx, w) != (xs[1],) or shape(ctx, out) != xs:
            return f"shapes input {xs} weight {shape(ctx, w)} output {shape(ctx, out)}"
        if sc.get("n_cols") != xs[1]:
            return f"n_cols {sc.get('n_cols')} vs N={xs[1]}"
        # the kernel reads row r at `input_ptr + r*input_row_stride` (columns contiguous) and writes
        # `output_ptr + r*output_row_stride`: the launch's strides must be the OPERANDS' row strides, which need not be N -- OLMoE's q_norm /
        # k_norm normalise column slices of the fused qkv GEMM output ([M, 2048] rows of stride 6144).  The operand's value collection is the
        # strided view's, so the emitted rows are exactly the rows the kernel reads; only the stride facts are checked here.
        for nm, t, st_key in (("input", x, "input_row_stride"), ("output", out, "output_row_stride")):
            st = _strides(t)
            if st is None or len(st) != 2 or st[1] != 1:
                return f"{nm} strides {st} are not (row_stride, 1)"
            if sc.get(st_key) != st[0] or st[0] < xs[1]:
                return f"{st_key}={sc.get(st_key)} vs the {nm} operand's row stride {st[0]} (N={xs[1]})"
        if not isinstance(sc.get("eps"), float):
            return "eps is not a static float"
        if meta.get("grid") not in ([xs[0]], "<callable>"):
            return f"grid {meta.get('grid')} is not (R,)"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        x, w, out = (launch_tensor(node, meta, n) for n in ("input_ptr", "weight_ptr", "output_ptr"))
        xc, wc = ctx.coll(x), ctx.coll(w)
        rows, lead = rows_of(xc)
        N, eps = rows[0].type.n, float(sc["eps"])
        fn, extra = norm_kind(ctx, "rmsnorm", node, N, eps, TARGET=active_target_profile(), M=len(rows))
        outs = [ctx.call(fn, r, wc, *extra, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(out, from_rows(outs, lead), self, what="output")
        ctx.bind_weight_role(w, self, "rmsnorm.weight", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "rows": len(rows), "lookup_key": lookup_key(meta), "pins": RMS_PINS}, emitted=f"{fn.id} x{len(rows)}")


#: Triton specializes an integer argument divisible by 16 and a pointer aligned to 16 bytes (`tt.divisibility`); the PTX of
#: `_fused_q_kv_rmsnorm_kernel` that `registry.dsa_norm` reads is the specialization where every stride and pointer is
TRITON_DIVISIBILITY = 16
QKV_ARGS = (("q", "q_ptr", "q_out_ptr", "q_weight_ptr", "q_in_stride", "q_out_stride", "Q_SIZE"),
            ("kv", "kv_ptr", "kv_out_ptr", "kv_weight_ptr", "kv_in_stride", "kv_out_stride", "KV_SIZE"))


def _aligned(t) -> bool:
    v = t.meta["val"]
    return v.storage_offset() * v.element_size() % TRITON_DIVISIBILITY == 0


def qkv_rms_launch(sizes: list[int]) -> tuple[int, int]:
    """`fused_q_kv_rmsnorm`'s BLOCK_SIZE and num_warps at (Q_SIZE, KV_SIZE): the next power of two of the larger, 8 warps from 2048."""
    block = 1 << (max(sizes) - 1).bit_length()
    return block, 8 if block >= 2048 else 4


class FusedQKvRmsNormRule(InPlaceRule):
    """`_fused_q_kv_rmsnorm_kernel[(T, 2)](q, q_out, q_weight, q_in_stride, q_out_stride, kv, kv_out, kv_weight, kv_in_stride,
    kv_out_stride, eps, Q_SIZE, KV_SIZE, BLOCK_SIZE, launch_pdl, num_warps)` launched by `fused_q_kv_rmsnorm` -> per token
    `TritonBlockRMSNorm{N=Q_SIZE,BLOCK,WARPS,EPS}` of the q row and `{N=KV_SIZE}` of the kv row; q_out and kv_out (allocations) are
    rebound.  `launch_pdl` is either value: it adds the PDL wait and launch and no arithmetic."""
    modes = frozenset({"profile"})
    info = RuleInfo("TritonQKvRmsNorm[vllm-default]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of fused_qk_rmsnorm._fused_q_kv_rmsnorm_kernel (kernel name + source pinned)",
                    preconditions=("kernel qualified name and source sha256 are the pinned ones (QKV_RMS_PINS)",
                                   "BLOCK_SIZE = next_pow2(max(Q_SIZE, KV_SIZE)) and num_warps 8 from BLOCK_SIZE 2048, else 4 (the wrapper's)",
                                   "q, kv bf16 [T, SIZE] with unit column stride and the launch's row stride; weights bf16 [SIZE]; outputs "
                                   "two unwritten allocations [T, SIZE], packed, neither an input",
                                   "every row stride divisible by 16 and every pointer 16-byte aligned (Triton's specialization the PTX is)",
                                   "TritonBlockRMSNorm's statics (check_block); eps a static float; grid (T, 2)"),
                    evidence_domain="structural; numerics: TritonBlockRMSNorm_v1 (registry.dsa_norm, the kernel's cc 9.0 PTX)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != QKV_RMS_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {QKV_RMS_PINS['kernel']}"
        if meta.get("src_sha256") != QKV_RMS_PINS["src_sha256"]:
            return f"kernel source sha256 {meta.get('src_sha256')} differs from the pin {QKV_RMS_PINS['src_sha256']}"
        if not ctx.vocab.has("triton_block_rmsnorm"):
            return "numerics_unregistered: the vocabulary has no triton_block_rmsnorm"
        sc, nw = meta.get("scalars", {}), (meta.get("geometry") or {}).get("num_warps")
        sizes = [sc.get(k[-1]) for k in QKV_ARGS]
        if not all(isinstance(n, int) and n > 0 for n in sizes) or not isinstance(sc.get("eps"), float):
            return f"Q_SIZE, KV_SIZE {sizes} and eps {sc.get('eps')!r} are not static positive sizes and a float"
        block, warps = qkv_rms_launch(sizes)
        if sc.get("BLOCK_SIZE") != block or nw != warps:
            return f"BLOCK_SIZE {sc.get('BLOCK_SIZE')}, num_warps {nw}: the wrapper launches {block}, {warps}"
        try:
            for N in sizes:
                check_block(N, block, warps, sc["eps"])
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        tensors = [[launch_tensor(node, meta, n) for n in k[1:4]] for k in QKV_ARGS]
        if any(t is None for ts in tensors for t in ts):
            return f"tensor arguments {[k[1:4] for k in QKV_ARGS]} not all present"
        outs = {ts[1] for ts in tensors}
        if len(outs) != 2 or outs & {t for ts in tensors for t in (ts[0], ts[2])}:
            return "q_out and kv_out are not two tensors apart from the inputs"
        T = None
        for (name, _, _, _, in_stride, out_stride, size_key), (x, out, w) in zip(QKV_ARGS, tensors):
            r = _row_value(ctx, x, name) or _row_value(ctx, w, f"{name} weight")
            if r:
                return r
            if not isinstance(ctx.val(out), Alloc) or not dtype_is(ctx, out, BF16):
                return f"{name} output is not an unwritten bf16 allocation"
            N, xs = sc[size_key], shape(ctx, x)
            if len(xs) != 2 or xs[1] != N or shape(ctx, w) != (N,) or shape(ctx, out) != xs or (T is not None and xs[0] != T):
                return f"{name} shapes {xs}, weight {shape(ctx, w)}, output {shape(ctx, out)} vs {size_key}={N}"
            T = xs[0]
            st, ost = _strides(x), _strides(out)
            if st is None or st[1] != 1 or st[0] != sc.get(in_stride) or ost != (N, 1) or sc.get(out_stride) != N:
                return f"{name} strides {st}, output {ost}: not ({in_stride}={sc.get(in_stride)}, 1) and packed ({out_stride}={sc.get(out_stride)})"
            if st[0] % TRITON_DIVISIBILITY or N % TRITON_DIVISIBILITY or not all(_aligned(t) for t in (x, out, w)):
                return (f"{name}: row strides {st[0]}, {N} or a pointer offset not divisible by {TRITON_DIVISIBILITY}, another Triton "
                        f"specialization than the PTX")
        if meta.get("grid") != [T, 2]:
            return f"grid {meta.get('grid')} is not ({T}, 2)"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc, eps = meta["scalars"], float(meta["scalars"]["eps"])
        S = {"BLOCK": sc["BLOCK_SIZE"], "WARPS": meta["geometry"]["num_warps"], "EPS": eps}
        emitted = []
        for name, x_arg, out_arg, w_arg, _, _, size_key in QKV_ARGS:
            x, out, w = (launch_tensor(node, meta, n) for n in (x_arg, out_arg, w_arg))
            N = sc[size_key]
            fn = ctx.kind("triton_block_rmsnorm", node, N=N, **S)
            rows, lead = rows_of(ctx.coll(x))
            wc = ctx.coll(w)
            outs = [ctx.call(fn, r, wc, node=node, rule=self, name=f"{ctx.hint_name(node)}/{name}[{i}]") for i, r in enumerate(rows)]
            ctx.rebind(out, from_rows(outs, lead), self, what=f"{name} output")
            ctx.bind_weight_role(w, self, f"rmsnorm.{name}.weight", N=N, EPS=eps)
            emitted.append(f"{fn.id} x{len(rows)}")
        ctx.record(self, node, {"Q_SIZE": sc["Q_SIZE"], "KV_SIZE": sc["KV_SIZE"], **S, "rows": len(rows), "pins": QKV_RMS_PINS},
                   emitted=" + ".join(emitted))


class TritonGemmRule(InPlaceRule):
    """`matmul_kernel_persistent[grid](a, b, c, bias, M, N, K, strides…, NUM_SMS, …constexprs)` launched by
    `matmul_persistent` from vLLM's `aten::linear` batch-invariant override -> `Gemm{K,N}` per row of `a`; `c` is rebound.
    `b` is `W.t()` (a view of the [N, K] parameter: stride_bk == 1, stride_bn == K); the kind takes W as [N, K].  When the only reader
    of `c` is the linear's bias add (`_bias_add_reader`), the pair is one `GemmBias_v1{K,N}(a row, W, b)` per row bound to the add's
    value instead, on the target whose GEMM is `Gemm_v1`, and one `GemmBias_v2{K,N,DOT}` on a target whose record says `gemm_bias_dot`
    (`frontend.targets.gemm_bias_spec`: cc 9.0 and cc 12.0).  Under `pouw:ncp-v2` a linear layer's launch is its call's served rows:
    `NcpLinearRow_v1{K,N}`, or `NcpLinearRowBias_v1{K,N}` bound to the bias add's value on any target (`_pouw_scheme`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("TritonGemm[vllm-bi]", 3, "in-place-binding",
                    applies_to="verity_cba::triton_launch of batch_invariant.matmul_kernel_persistent (kernel name + constexprs pinned)",
                    preconditions=("kernel qualified name is the profile's `matmul_kernel_persistent`; source sha256 pinned",
                                   "the active TargetProfile's capability has a REGISTERED GEMM specialization (verity_vllm.program.frontend.gemm_targets: ada cc 8.9 with "
                                   "vLLM's tuned sm_89 table, ampere cc 8.0 default tiling only; anything else -> numerics_unregistered)",
                                   "tiling = that target's launch geometry for (M, N, K) (GemmTarget.launch_config: the default 128/128/64, 8 warps, 3 stages, or the "
                                   "M-bucket row of the target's pinned tuned table); GROUP_SIZE_M 8, *_LARGE False, HAS_BIAS False, "
                                   "NUM_SMS = the active TargetProfile's num_sms (default 128)",
                                   "a is bf16 [M, K] row-major (stride_am K, stride_ak 1); b is bf16 [K, N] with stride_bk 1, stride_bn K (= W[N,K].t())",
                                   "c is an unwritten allocation [M, N] row-major; bias_ptr is None; M, N, K equal the shapes",
                                   "fused bias (Gemm_v1 targets, and targets whose record says gemm_bias_dot): c's only other reader is "
                                   "aten.add.Tensor(c, b) (either order, alpha 1), the next computing node, in the launch's module; b bf16 [N] or "
                                   "[1, N]; the sum bf16 [M, N]"),
                    evidence_domain="structural; numerics: Gemm_v1{K,N} / Gemm_v2{K,N,DOT} (the same Triton kernel; f32 accumulator over "
                                    "BLOCK_SIZE_K tiles); GemmBias_v1{K,N} / GemmBias_v2{K,N,DOT} = BiasAdd_v1 of that word "
                                    "(linear_batch_invariant's `output + bias` in bf16)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {GEMM_PINS['kernel']}"
        sc = meta.get("scalars", {})
        from verity_vllm.program.frontend.gemm_targets import UnregisteredGemmTarget
        try:
            pins, _src = gemm_pins_for(sc.get("M"), sc.get("N"), sc.get("K"))
        except UnregisteredGemmTarget as e:   # fail closed: no registered Gemm Definition for the target, whatever tiling the launch shows
            return str(e)
        bad = _pins_ok(meta, pins)
        if bad:
            return bad
        if sc.get("bias_ptr") is not None or launch_tensor(node, meta, "bias_ptr") is not None:
            return "bias_ptr is set (the fused bias epilogue is a different numeric than Gemm + BiasAdd)"
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        if a is None or b is None or c is None:
            return "tensor arguments a_ptr/b_ptr/c_ptr not all present"
        r = _row_value(ctx, a, "a") or _row_value(ctx, b, "b")
        if r:
            return r
        if not isinstance(ctx.val(c), (Alloc, Coll)):
            return "c is not an allocation/value"
        if str(ctx.dtype_of(c)) != GEMM_PINS["dtype"]:
            return f"numerics_unregistered: c dtype {ctx.dtype_of(c)}; Gemm_v1 is the bf16 kernel"
        as_, bs, cs = shape(ctx, a), shape(ctx, b), shape(ctx, c)
        if len(as_) != 2 or len(bs) != 2 or len(cs) != 2:
            return f"ranks a{as_} b{bs} c{cs}"
        M, K = as_
        K2, N = bs
        if K != K2 or cs != (M, N) or (sc.get("M"), sc.get("N"), sc.get("K")) != (M, N, K):
            return f"shapes a{as_} b{bs} c{cs} vs scalars M={sc.get('M')} N={sc.get('N')} K={sc.get('K')}"
        if (sc.get("stride_am"), sc.get("stride_ak")) != (K, 1):
            return f"a strides {sc.get('stride_am')},{sc.get('stride_ak')} are not row-major [M, K]"
        if (sc.get("stride_bk"), sc.get("stride_bn")) != (1, K):
            return f"b strides {sc.get('stride_bk')},{sc.get('stride_bn')} are not W[N, K].t()"
        if (sc.get("stride_cm"), sc.get("stride_cn")) != (N, 1):
            return f"c strides {sc.get('stride_cm')},{sc.get('stride_cn')} are not row-major [M, N]"
        if K % 16:
            return f"K={K} is not a multiple of the 16-wide k-step"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        a, b, c = (launch_tensor(node, meta, n) for n in ("a_ptr", "b_ptr", "c_ptr"))
        ac, bc = ctx.coll(a), ctx.coll(b)
        rows, lead = rows_of(ac)
        K, N = shape_of(bc.type)[0]
        w_nk = _untransposed(ctx, b, K, N, ctx.hint_name(node) + "/W")  # b = W.t(): the kind's W is [N, K] — a reference rearrangement
        accepted, dot = _target_kinds()   # the target's registered k16 step selects Gemm_v1 (accepted) or Gemm_v2{DOT}
        pouw = _pouw_scheme(node)
        bias_role = ("pouw_linear_row_bias", {}) if pouw is not None else gemm_bias_role()
        fused = _bias_add_reader(node, c, ctx, len(rows), N) if bias_role is not None else None
        if fused is not None:
            add, bn = fused
            bias = ctx.coll(bn, "b")
            if len(shape_of(bias.type)[0]) != 1:
                bias = bias.reshape(_tensor_type((N,), 16))   # [1, N] -> [N]: the same leaves, the same broadcast
            role, extra = bias_role
            fn = ctx.kind(role, add, K=K, N=N, **extra)
            outs = [ctx.call(fn, r, w_nk, bias, node=add, rule=self, name=f"{ctx.hint_name(add)}[{i}]") for i, r in enumerate(rows)]
            ctx.define(add, from_rows(outs, lead))
            if ctx.weight_of(bn) is not None:
                ctx.bind_weight_role(bn, self, "linear.bias", N=N)
        else:
            if pouw is not None:
                fn = ctx.kind("pouw_linear_row", node, K=K, N=N)
            else:
                fn = ctx.kind("gemm", node, K=K, N=N) if accepted else ctx.kind("gemm_dot", node, K=K, N=N, DOT=dot)
            outs = [ctx.call(fn, r, w_nk, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
            ctx.rebind(c, from_rows(outs, lead), self, what="c")
        ctx.bind_weight_role(b, self, "linear.weight", K=K, N=N)
        wl = ctx.weight_of(b)
        pins, src = gemm_pins_for(sc.get("M"), sc.get("N"), sc.get("K"))
        tgt = active_gemm_target()
        ctx.record(self, node, {"K": K, "N": N, "rows": len(rows), "lookup_key": lookup_key(meta), "pins": pins, "pin_source": src,
                                "gemm_target": tgt.key, "gemm_target_dot": tgt.dot, "W_is_parameter": wl.name if wl else None,
                                "bias_add": fused[0].name if fused else None, "pouw": pouw}, emitted=f"{fn.id} x{len(rows)}")


def linear_types() -> tuple[type, ...]:
    """The classes of a PoUW linear layer: vLLM's ``LinearBase``, the PoUW adapter's executed site (lm_head is not one)."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def module_type(cls) -> type | None:
    """A module stack entry's class. `torch.export` records it as its qualified name (``module.QualName``), resolved in the
    already imported module; an fx trace records the type itself."""
    if isinstance(cls, type):
        return cls
    if not isinstance(cls, str):
        return None
    parts = cls.split(".")
    for i in range(len(parts) - 1, 0, -1):
        obj = sys.modules.get(".".join(parts[:i]))
        if obj is None:
            continue
        for name in parts[i:]:
            obj = getattr(obj, name, None)
        return obj if isinstance(obj, type) else None
    return None


def _pouw_scheme(node) -> str | None:
    """The PoUW scheme a launch is lowered under: the active target's ``pouw:<scheme>`` when that scheme has a circuit
    (`verity.ml.protocols.pouw.circuit.SCHEMES`) and the launch belongs to a linear layer (`linear_types`, by the class of the innermost
    module in the node's module stack)."""
    from verity.ml.protocols.pouw import circuit

    entries = dict(e.split(":", 1) for e in (active_target_profile().protocols or ()))
    scheme = entries.get("pouw")
    if scheme not in circuit.SCHEMES:
        return None
    stack = node.meta.get("nn_module_stack") or {}
    cls = module_type(list(stack.values())[-1][1]) if stack else None
    return scheme if cls is not None and issubclass(cls, linear_types()) else None


def _bias_add_reader(launch, c, ctx, M: int, N: int):
    """`linear_batch_invariant`'s `output + bias` when it is the launch output's ONLY reader: `aten.add.Tensor(c, b)` (either order,
    alpha 1) in the launch's module, the first computing node after the launch, `b` a bf16 `[N]` (or `[1, N]`) value, the sum bf16
    `[M, N]` -> (add node, b node); else None and the pair stays the GEMM + `BiasAdd_v1`.  The pre-bias word then never leaves its
    coordinate (`GemmBias_v1` / `GemmBias_v2`), and `c` is never written: any other read of it is refused as an uninitialised allocation."""
    readers = [u for u in c.users if u is not launch]
    if len(readers) != 1:
        return None
    add = readers[0]
    if add.op != "call_function" or add.target is not torch.ops.aten.add.Tensor or len(add.args) != 2 or add.kwargs.get("alpha", 1) != 1:
        return None
    nxt = launch.next
    while nxt is not add and nxt.op != "call_function" and nxt.op != "output":
        nxt = nxt.next
    if nxt is not add or ctx.stack_path(add) != ctx.stack_path(launch):
        return None
    bn = add.args[1] if add.args[0] is c else add.args[0]
    if bn is c or not isinstance(bn, torch.fx.Node) or not isinstance(ctx.val(bn), Coll):
        return None
    if not (dtype_is(ctx, bn, BF16) and dtype_is(ctx, add, BF16)) or _bias_shape(shape(ctx, bn)) != N or shape(ctx, add) != (M, N):
        return None
    if len(shape(ctx, bn)) > 2:
        return None
    return add, bn
