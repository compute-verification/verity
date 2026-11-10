"""The fold's token-select fallback (`fold._Fold._token_select_output`): which of a step's instances produced a sampled token when
the sampler view is absent or unresolved."""

from __future__ import annotations

from typing import Any

from verity.ml.circuits.refs import Affine, Refs
from verity_vllm.observe.events import StepStart, ViewDesc
from verity_vllm.observe.fold.views import same_element
from verity_vllm.program import sampling_event as SE


def token_select_output(instances: list[dict[str, Any]], start: StepStart | None, step: int, row: int,
                        view: ViewDesc | None = None) -> Refs | None:
    """The step's instance whose output view IS the sampled row's element (`view`, family-agnostic: any token-selection definition),
    else the step's sampling-event instance (`sampling_event.is_sampling_event`: greedy or Gumbel token selection) for logits row
    `row`: the one whose attribution is that row's (request, position), or, unattributed, row `row` of a select spanning every logits
    row.  An instance's `row` is its Compute's: the greedy select is one Compute over the logits rows, a Gumbel select one Compute per
    logits row (each its own row 0), so the output element is matched over the whole step before any row is.  `instances` are the
    fold's, newest last, and `start` the step's StepStart."""
    step_instances = []
    for m in reversed(instances):
        if m["step"] != step:
            break
        step_instances.append(m)
    if view is not None:
        for m in step_instances:
            if any(ov is not None and same_element(ov, view) for ov in m.get("output_views", ())):
                return Affine("n", m["node"], 0, 1)
    logits_rows = list(start.logits_rows) if start is not None else []
    want = (str(logits_rows[row][0]), int(logits_rows[row][1])) if row < len(logits_rows) else None
    for m in step_instances:
        if not SE.is_sampling_event(m["spec"]):    # the ONE sampling-event vocabulary (verity_vllm.program.sampling_event)
            continue
        if m.get("request") is not None:
            if want is not None and (m["request"], m["position"]) == want:
                return Affine("n", m["node"], 0, 1)
        elif m["row"] == row and m.get("rows") == len(logits_rows):
            return Affine("n", m["node"], 0, 1)
    return None
