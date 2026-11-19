"""What a Commit attaches beside the record's sources, from the capture plan's classes (`check.capture_plan.of_build`, taken as
data): the exact-int8 quantizers' results (`int8_source`, which claims its `call_boundary` classes) and the blockwise FP8 GEMM's scale
products (`scale_products_source`); a plan with a call boundary no source claims is refused by name.  `attached` names what `attach`
attaches for a plan, constructing nothing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity_vllm.acquire.sources import call_boundary_source, int8_source, scale_products_source
from verity_vllm.log import init_logger
from verity_vllm.records.capture import SCALE_PRODUCTS

logger = init_logger(__name__)

def attach(com: Any, plan: Sequence[Mapping[str, Any]], required_classes: tuple[str, ...], *,
           rows: Sequence[Mapping[str, Any]] | None = None, explicit_classes: bool = False, model: Any = None) -> tuple[str, ...]:
    """Attach to `com` the sources of the capture plan's classes (`check.capture_plan.of_build`): the scale-products source
    when the plan names `scale_products` classes (its block geometry from the plan's `rows`, `capture_plan.rows_of_build`, without
    which it is refused) and the int8 source when it names Int8RowQuant call boundaries or Int8KvQuant module
    outputs.  A plan with a `call_boundary` class no source claims is refused by name (exit 3).  Returns the required classes with the attached
    sources' capture classes added unless the caller gave them explicitly."""
    refused = [f"{c['op_path']}/{c['member']}: a call boundary no attached source claims" for c in call_boundary_source.sites(plan)]
    if refused:
        logger.error("[delta] REFUSE: %d classes of the capture plan have no acquisition source: %s", len(refused), refused[:4])
        raise SystemExit(3)
    required_classes, _src = scale_products_source.attach(com, any(c["capture"] == SCALE_PRODUCTS for c in plan), plan, required_classes,
                                                          rows=rows, explicit_classes=explicit_classes, model=model)
    required_classes, _src = int8_source.attach(com, int8_source.applies(plan), plan, required_classes,
                                                explicit_classes=explicit_classes, model=model)
    return required_classes


def warmup(com: Any, on: bool) -> None:
    """Around the Commit's warm-up instrumented run, which is never a record (its root is discarded): bounded staging learns only
    (`learn_only`) and every source with placeholder words (`placeholder`) commits them instead of evaluating."""
    if getattr(com, "bounded", False):
        com.learn_only = on
    for src in getattr(com, "sources", None) or ():
        if hasattr(src, "placeholder"):
            src.placeholder = on


def attached(plan: Sequence[Mapping[str, Any]]) -> list[str]:
    """The `short_name`s of the sources `attach` attaches for a capture plan's classes, constructing none: the scale-products source
    and the int8 source, each when the plan names a class of its."""
    return [*([scale_products_source.ScaleProductsSource.short_name] if any(c["capture"] == SCALE_PRODUCTS for c in plan) else []),
            *([int8_source.SHORT_NAME] if int8_source.applies(plan) else [])]


__all__ = ["attach", "warmup", "attached"]
