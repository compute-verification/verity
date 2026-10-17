"""READ BACK what an instantiated vLLM model's layers execute, for `target_profile.applied_kernels` / `profile_mismatches`.

Structural scaffolding over `model.modules()` and each module's `quant_method`, no model fact; torch-free (any object with
`modules()` works).  Each read-back is one value over every layer it reads: None when no layer has one, '<mixed>' when the
layers disagree (both are mismatches against a declaration).  `target_profile` re-exports every name.
"""
from __future__ import annotations


def fp8_block_gemm_kernel_of(model) -> str | None:
    """The block-scaled FP8 linear kernel class the instantiated model carries: the class name of
    `<layer>.quant_method.fp8_linear` (vLLM `Fp8LinearMethod`), one value over all FP8 linear layers."""
    return _one_name({type(k).__name__ for k in _quant_attrs(model, "fp8_linear")})


def moe_experts_class_of(model) -> str | None:
    """The experts class of every fused-MoE layer: `<layer>.quant_method.experts_cls` (vLLM's `Fp8MoEMethod` and
    `UnquantizedFusedMoEMethod`), one name over all layers."""
    return _one_name({getattr(c, "__name__", repr(c)) for c in _quant_attrs(model, "experts_cls")})


def int8_exact_arm_of(model) -> str | None:
    """The int8 arm every linear layer runs: `quant_method.int8_arm` of every module whose quant_method is a vLLM linear method
    (a class named `LinearMethodBase` in its MRO), '<none>' for one without an arm, so a single linear on another method makes
    the value '<mixed>'."""
    arms = {getattr(qm, "int8_arm", None) or "<none>" for m in model.modules()
            if (qm := getattr(m, "quant_method", None)) is not None and any(c.__name__ == "LinearMethodBase" for c in type(qm).__mro__)}
    return _one_name(arms)


def _quant_attrs(model, attr: str) -> list:
    return [v for m in model.modules() if (v := getattr(getattr(m, "quant_method", None), attr, None)) is not None]


def _one_name(names: set[str]) -> str | None:
    return None if not names else names.pop() if len(names) == 1 else "<mixed>"
