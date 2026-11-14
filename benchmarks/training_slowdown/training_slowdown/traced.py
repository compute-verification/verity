"""A config's training step as the capture measures it: what a job records as its `trace` (`trace_step`,
`training_slowdown.run`) and, beside the capture's graph, what the library reads an entry's bound on (`traced_step`,
`training_slowdown.library`). The step is traced on tensors without data (`training_slowdown.capture`), so tracing it
trains nothing and reads no weights and no data. The module is no part of the job's machinery (`run`, the
sandbox), so what reads a traced step imports none of it."""
from __future__ import annotations

import torch

from . import VERSION, config

#: the steps the trace's capture takes, the run's last (`trace_step`): the measured one, the one before it, which
#: wrote the values it reads, and the one after it, which reads the values it makes
TRACED_STEPS = 3


def trace_step(cfg: config.Config, route, model, optimizers, device: str) -> dict:
    """The record's `trace`: the route's `step` on its model and fresh copies of its optimizers, at the job's batch
    and the start's context, as the capture measures it (`capture_steps`, `capture.products.raw_on_dial`). The
    capture takes the run's last three steps in one graph and measures the middle one, which reads what the step
    before it wrote and makes what the step after it reads (`TRACED_STEPS`; a run of two steps is measured at its
    second, a run of one at its only step, which reads no update). Where the three aren't the run's first steps the
    capture is told so (`mid_run`): the first of them reads the run's own weights, not θ's. A route's `output` is
    traced as the end of every step, so the products it computes are the measured step's, charged like any other,
    though a run calls it at its checkpoints only.

    It keeps the capture's and torch's versions, the version of the code that took the trace (`capture_code`, this
    tree's: a record whose trace is taken again keeps its job's `training_slowdown`), the step and the token axes it
    was traced at, the forward's updated MAC nodes (`StepGraph.structure`'s `updated`), the product rows with their
    operands' dtypes and no price at each k of the console's dial, the ops that break P-match, the products written
    with elementwise ops, the read-backs the capture lists (`data_dependent`, `carried_scalars`; a step the capture
    takes as a function has none, since it refuses one), and the measured step's circuit as the capture measures it
    (`circuit`, `capture.circuit.measure`): the gradient's cut and one token group's partial, the carried state no
    token op reads, the update's profile and its cut, τ, a, and the shares of the step's gates in token ops (σ) and
    in ops an update reaches, at the gate prices of the code that took the trace. Nothing the formalism derives is in
    it. A step the capture refuses gives {"refused": {"reason", "op", "message"}}; any other error is raised, and the
    job records it in the trace's place ({"failed": why}, `run.inside`).

    The batch is traced as its inputs and each window's last target apart, so the token axes are the batch and the
    context, the tokens the counter counts a step."""
    return traced_step(cfg, route, model, optimizers, device)[0]


def traced_step(cfg: config.Config, route, model, optimizers, device: str) -> tuple[dict, object]:
    """`trace_step`'s trace beside the capture's graph it was read from (a `StepGraph`), which holds what the record
    doesn't keep: the parameters the step writes and the ops the capture prices by a rule of thumb. No graph, None,
    for a step the capture refuses."""
    from .capture import CaptureRefused, capture_steps, products

    n, batch, ctx = cfg.budget["steps"], cfg.budget["batch"], int(cfg.start.shape["ctx"])
    ts = range(max(n - TRACED_STEPS, 0), n)
    at = min(1, len(ts) - 1)

    def step(t: int):
        def traced(m, rows, opts):
            route.step(m, torch.cat(rows, dim=1), opts, cfg.dials, t, n)
            if hasattr(route, "output"):
                with torch.no_grad():
                    route.output(m, cfg.dials)
        return traced

    example = (torch.zeros(batch, ctx, dtype=torch.long), torch.zeros(batch, 1, dtype=torch.long))
    try:
        g = capture_steps([step(t) for t in ts], model, example, measured=at, optimizers=optimizers,
                          token_dims=((0, 1), (0,)), device=device, mid_run=ts[0] > 0)
    except CaptureRefused as e:
        return {"refused": {"reason": e.reason, "op": e.op, "message": str(e)}}, None
    raw = products.raw_on_dial(g)
    read_backs = g.circuit.read_backs or {}
    return {"capture": raw["capture"], "capture_code": VERSION, "torch": torch.__version__, "step": ts[at],
            "axes": g.axes, "tokens": g.tokens, "order": raw["order"], "updated": g.structure()["updated"],
            "rows": raw["rows"], "k": raw["k"], "p_match": raw["p_match"], "at": raw["at"],
            "elementwise_products": raw["elementwise_products"],
            "data_dependent": list(read_backs.get("data_dependent", [])),
            "carried_scalars": list(read_backs.get("carried_scalars", [])),
            "circuit": {k: v for k, v in g.formalism.items() if k not in read_backs}}, g
