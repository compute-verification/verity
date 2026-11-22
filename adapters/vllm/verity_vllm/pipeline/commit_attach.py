"""The Commit's acquisition on a built committer: every source, tap and hook the capture plan of the row's Build of record asks
for (`attach_of_plan`), which serve-openai's rows-sha512 Commit (`pipeline.serve_seal.committer`) builds its own through."""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.acquire import fa2_plane_classes as FP
from verity_vllm.acquire import install as INSTALL
from verity_vllm.acquire import stage as STAGE
from verity_vllm.acquire.sources import moe_source as MOE
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.log import init_logger
from verity_vllm.records.capture import CALL_BOUNDARY, RUN_VALUE, SCALE_PRODUCTS

__all__ = ["attach_of_plan"]

logger = init_logger(__name__)


#: the capture classes a source inside a module acquires (`taps.attach`), not a module hook
TAPPED = (CALL_BOUNDARY, SCALE_PRODUCTS, RUN_VALUE)


def attach_of_plan(com: Any, eng: Any, args: Any, classes: Sequence[Mapping[str, Any]], out: Path, *, kind: str,
                   rows: Sequence[Mapping[str, Any]]) -> SimpleNamespace:
    """The served Commit's acquisition on ``com`` for the row's plan classes (`capture_plan.of_build`) and plan rows
    (`capture_plan.rows_of_build`): the sources `taps.attach` attaches for them (those of the classes a module's own return doesn't carry, `TAPPED`, among them), the X-03 attention plane classes from the Build's
    record, and the hook sets of the acquisition plan, written into ``out``. An MoE model is refused by name (its block stream
    is `capture_plan.NOT_SERVED`), and so is a class with no site (exit 3)."""
    model = getattr(com, "model", None)
    if model is None or not hasattr(com, "_modules"):
        logger.error("[delta] REFUSE: the acquisition plan needs a hooking committer with a model (got %s)", kind)
        sys.exit(3)
    if MOE.MoeBlockSource(com, model).layers:
        from verity_vllm.check.capture_plan import NOT_SERVED

        logger.error("[delta] REFUSE: the model has MoE layers, and %s", NOT_SERVED["moe_block_stream"])
        sys.exit(3)
    tapped = [c for c in classes if c["capture"] in TAPPED]
    TAPS.attach(com, list(classes), (), rows=rows, model=model)
    _impls, _impls_path = FP.attention_impls_of_record(program_dir=None, programs_root=args.programs_root)
    if _decls := (FP.declarations_from_attention_impls(_impls) if _impls else []):
        com.attention_layer_decls = _decls
    try:
        docs = STAGE.plan_at_stage(model, getattr(eng, "runner", None), classes, out, source=f"commit {kind}")
        for ln in docs.log_lines("delta"):
            logger.info("%s", ln)
        if docs.plan is None:
            raise INSTALL.IncompletePlan("the stage drew no acquisition plan")
        rec = INSTALL.apply_plan_to_committer(com, docs.plan)
    except INSTALL.IncompletePlan as e:
        logger.error("[delta] REFUSE: acquisition plan: %s", e)
        sys.exit(3)
    (out / "acquisition_engine.json").write_text(json.dumps({"engine": "v2", "plan_digest": docs.plan.digest, **rec}, indent=1) + "\n")
    logger.info("[delta] committer %s: %s module output hooks from the plan's %s classes (%s tapped), consumer inputs %s", com.name,
                len(rec["modules_after"]), len(classes), len(tapped), rec["plan_inputs"])
    return SimpleNamespace(plan=docs.plan, runtime=docs.runtime, tapped=len(tapped), attention_decls=len(_decls))
