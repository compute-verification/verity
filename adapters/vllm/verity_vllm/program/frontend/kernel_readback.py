"""READ BACK what an instantiated vLLM model's layers execute, for `target_profile.applied_kernels` / `profile_mismatches`.

Structural scaffolding over `model.modules()`, each module's `quant_method` and each attention layer's `impl`, no model fact;
torch-free (any object with `modules()` works).  Each read-back is one value over every layer it reads: None when no layer has
one, '<mixed>' when the layers disagree (both are mismatches against a declaration).  `target_profile` re-exports every name.
"""
from __future__ import annotations

#: the arm an exact-int8 linear method or attention impl reports (`int8_arm`): steps.py's plain int8 linear, the LUT-softmax attention
INT8_ARM = "plain"


def fp8_block_gemm_kernel_of(model) -> str | None:
    """The block-scaled FP8 linear kernel class the instantiated model carries: the class name of
    `<layer>.quant_method.fp8_linear` (vLLM `Fp8LinearMethod`), one value over all FP8 linear layers."""
    return _one_name({type(k).__name__ for k in _quant_attrs(model, "fp8_linear")})


def moe_experts_class_of(model) -> str | None:
    """The experts class of every fused-MoE layer: `<layer>.quant_method.experts_cls` (vLLM's `Fp8MoEMethod` and
    `UnquantizedFusedMoEMethod`), one name over all layers."""
    return _one_name({getattr(c, "__name__", repr(c)) for c in _quant_attrs(model, "experts_cls")})


def int8_exact_of(model) -> str | None:
    """The int8 scope the layers run (`TargetProfile.INT8_EXACT_SCOPES`): which of the linears (`quant_method.int8_arm` of every
    module whose quant_method is a vLLM linear method or fused-MoE method, a class named `LinearMethodBase` or
    `FusedMoEMethodBase` in its MRO, but the MoE routers, `routers`, and the linears the checkpoint keeps bf16, `kept`) and the
    attention layers (`impl.int8_arm` of every module whose `impl` has a class named `AttentionImpl` in its MRO) all run the
    plain arm.  None when neither does, '<mixed>' when some but not all layers of a kind do (or one runs another arm)."""
    parts = []
    linears = _arms(model, "quant_method", ("LinearMethodBase", "FusedMoEMethodBase"), routers(model) | kept(model))
    for kind, arms in (("linears", linears), ("attention", _arms(model, "impl", ("AttentionImpl",)))):
        if arms == {INT8_ARM}:
            parts.append(kind)
        elif arms - {"<none>"}:
            return "<mixed>"
    return {("linears",): "linears", ("attention",): "attention", ("linears", "attention"): "all"}.get(tuple(parts))


def int8_k_scales_of(model) -> dict[str, list[int]]:
    """{layer_name: the K scales as float32 bit patterns, one per KV head} of every attention layer whose impl carries them
    (`engine.int8_attention_vllm.Int8AttentionImpl.k_scale_bits`)."""
    out = {}
    for name, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if is_attention_impl(impl) and callable(kb := getattr(impl, "k_scale_bits", None)):
            ln = getattr(mod, "layer_name", name)
            out[ln] = [b & 0xFFFFFFFF for b in kb(ln)]
    return out


def is_attention_impl(obj) -> bool:
    return obj is not None and any(c.__name__ == "AttentionImpl" for c in type(obj).__mro__)


def applied_kernels(profile, model) -> dict:
    """The kernel read-backs `profile_mismatches` judges: the FP8 linears' always, the experts class under a declared `moe_backend`,
    the int8 scope under a declared `int8_exact`."""
    out = {"fp8_block_gemm_kernel": fp8_block_gemm_kernel_of(model)}
    if profile.moe_backend is not None:
        out["moe_experts_class"] = moe_experts_class_of(model)
    if profile.int8_exact is not None:
        out["int8_exact"] = int8_exact_of(model)
    return out


def routers(model) -> set[int]:
    """The ids of the MoE routers: each linear module (a quant method with a class named `LinearMethodBase` in its MRO) built
    with no quantization config (``quant_config`` None) whose ``output_size`` is the expert count of a MoE layer (a module
    with ``routed_experts``) it sits beside (a child of the same module) or that the layer holds as its ``gate``.  A router
    stays bf16 under the int8 profile, as the model builds it."""
    out = set()
    for p in model.modules():
        kids = list(p.children()) if callable(getattr(p, "children", None)) else []
        for moe in (c for c in [p, *kids] if getattr(c, "routed_experts", None) is not None):
            experts = getattr(moe.routed_experts, "global_num_experts", None)
            for c in [*kids, getattr(moe, "gate", None)]:
                if c is not None and getattr(c, "quant_config", 0) is None and getattr(c, "output_size", None) == experts \
                        and _is(getattr(c, "quant_method", None), ("LinearMethodBase",)):
                    out.add(id(c))
    return out


#: Linears vLLM fuses in its weight loader without naming them in the model's ``packed_modules_mapping``: the fused leaf and its
#: shards (vLLM's ``DeepseekV32Indexer``: ``wk_weights_proj`` = ``wk`` and ``weights_proj``).
FUSED_UNPACKED = {"wk_weights_proj": ["wk", "weights_proj"]}


def kept(model) -> set[int]:
    """The ids of the linears the checkpoint's ``modules_to_not_convert`` keeps bf16: each module whose linear method says
    so (``int8_kept``, `engine.int8_vllm.Int8KeptLinearMethod`), and each linear the model builds with no quantization config
    (``quant_config`` None, so no quantization method reaches it: vLLM's ``DeepseekV32Indexer.wk_weights_proj``) whose every
    shard (the model's ``packed_modules_mapping``, else `FUSED_UNPACKED`) the checkpoint's ``modules_to_not_convert`` names."""
    marked = {id(m) for m in model.modules() if _is(qm := getattr(m, "quant_method", None), ("LinearMethodBase",))
              and getattr(qm, "int8_kept", False) is True}
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    names = set((qc.get("modules_to_not_convert") if isinstance(qc, dict) else getattr(qc, "modules_to_not_convert", None)) or ())
    if not names or not callable(getattr(model, "named_modules", None)):
        return marked
    packed = {**FUSED_UNPACKED, **(getattr(model, "packed_modules_mapping", None) or {})}
    for name, m in model.named_modules():
        if getattr(m, "quant_config", 0) is None and _is(getattr(m, "quant_method", None), ("LinearMethodBase",)):
            head, _, leaf = name.rpartition(".")
            if all((f"{head}.{s}" if head else s) in names for s in packed.get(leaf, [leaf])):
                marked.add(id(m))
    return marked


def _is(obj, bases: tuple[str, ...]) -> bool:
    return obj is not None and any(c.__name__ in bases for c in type(obj).__mro__)


def _arms(model, attr: str, bases: tuple[str, ...], skip: set[int] = frozenset()) -> set[str]:
    return {getattr(v, "int8_arm", None) or "<none>" for m in model.modules()
            if id(m) not in skip and _is(v := getattr(m, attr, None), bases)}


def _quant_attrs(model, attr: str) -> list:
    return [v for m in model.modules() if (v := getattr(getattr(m, "quant_method", None), attr, None)) is not None]


def _one_name(names: set[str]) -> str | None:
    return None if not names else names.pop() if len(names) == 1 else "<mixed>"
