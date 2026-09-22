"""[R8 C-codec] Interchange with lane C-builder's hint schema `verity-aot/hints/v1` (verity_capture/experimental/aot/hints.py).

C-builder's `h` is the *residual* against their predictor `dense-eager-fcfs/v1` (served-length overrides + whole-step row overrides);
this lane's ScheduleHint is the *absolute* structural fact set (every step's rows + logits + per-request lengths).  Both describe the same
execution, so they must convert exactly in both directions given phi and their predictor:

    to_builder_hints(phi, h)    -> builder Hints JSON   (minimal residual: greedy first-mismatch overrides, like their encode.py)
    from_builder_hints(phi, hj) -> ScheduleHint         (their predictor run with the hints, re-expressed as absolute facts)

    from_builder_hints(phi, to_builder_hints(phi, h)) == h   is asserted by tests/test_builder_adapter.py on the corpus hints.

The builder tree is read-only for this lane: it is imported, never edited.  Resolution order: the merged package
`verity_capture.experimental.aot`; else the sibling worktree named by $AOT_BUILDER_TREE (default /private/tmp/w-integ), loaded under a
private package name so nothing of this lane's `verity_capture` is shadowed.  Absent both -> `BuilderUnavailable`.

What the v1 schema cannot say (each is reported, not hidden):
  * finish_reason other than length/stop (abort/other) -- v1 has `served` only (structurally equivalent; the label is record fidelity);
  * an idle engine gap (their predictor counts idle step numbers, the harness collapses them -- arrivals.py) -> step numbering drifts;
  * a request that is never scheduled (arrive_step -1).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

from .facts import SCHEMA


class BuilderUnavailable(RuntimeError):
    pass


class NotExpressibleInV1(ValueError):
    pass


_aot = None


def load_builder():
    """The builder package (hints, predict, phi modules) or raise BuilderUnavailable."""
    global _aot
    if _aot is not None:
        return _aot
    try:
        _aot = importlib.import_module("verity_capture.experimental.aot")
        importlib.import_module("verity_capture.experimental.aot.predict")
        return _aot
    except Exception:
        pass
    tree = Path(os.environ.get("AOT_BUILDER_TREE", "/private/tmp/w-integ")) / "verity_capture" / "experimental" / "aot"
    if not (tree / "predict.py").exists():
        raise BuilderUnavailable(f"builder package not importable and {tree} not found")
    name = "_aot_builder_ro"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, tree / "__init__.py", submodule_search_locations=[str(tree)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    _aot = sys.modules[name]
    for sub in ("hints", "phi", "predict"):
        importlib.import_module(f"{name}.{sub}")
    return _aot


def _mods():
    aot = load_builder()
    return sys.modules[f"{aot.__name__}.hints"], sys.modules[f"{aot.__name__}.phi"], sys.modules[f"{aot.__name__}.predict"]


def builder_phi(phi: dict, arrive: list[int] | None = None):
    """Their Phi from this lane's phi dict: the predictor reads len(prompt_token_ids), max_tokens, arrive_step and the engine facts only.
    Prompt token ids are not part of this lane's phi (only prompt_len is structural) -> placeholder ids of the right length."""
    _, phi_mod, _ = _mods()
    reqs = []
    for i, q in enumerate(phi["requests"]):
        a = arrive[i] if arrive is not None else q.get("arrive_step", 0)
        reqs.append({"request_id": q.get("request_id", f"r{i}"), "prompt_token_ids": [0] * int(q["prompt_len"]), "max_tokens": int(q["max_tokens"]),
                     "arrive_step": int(a)})
    eng = phi["engine"]
    return phi_mod.Phi(engine={"max_num_seqs": eng["max_num_seqs"], "max_num_batched_tokens": eng["max_num_batched_tokens"],
                               "enable_chunked_prefill": eng.get("chunked_prefill", True), "enable_prefix_caching": eng.get("prefix_caching", False),
                               "block_size": eng.get("block_size", 16)},
                       workload={"requests": reqs})


def _expand(rows: list[list[int]]) -> list[tuple[int, int]]:
    return [(i, c + j) for i, c, n in rows for j in range(n)]


def _runs(rows: list[list[int]]):
    hints_mod, _, _ = _mods()
    return tuple(hints_mod.RowRun(int(i), int(c), int(n)) for i, c, n in rows)


def _my_logits_as_positions(rows: list[list[int]], lg: list[list[int]]) -> list[tuple[int, int]]:
    """This lane's logits rows [[req, is_final]] (one per sampled row, row-group order) -> their (request, position) pairs."""
    last = {}
    for i, c, n in rows:
        last[i] = c + n - 1
    return [(int(i), last[i]) for i, _ in lg]


def to_builder_hints(phi: dict, h: dict, *, max_rounds: int | None = None) -> dict:
    """Minimal v1 residual of `h` against their predictor: served overrides for every n_generated != max_tokens, then greedy step overrides
    at the first mismatching step, re-predicting after each (their encode.py strategy), until the whole plan equals `h`."""
    hints_mod, _, predict_mod = _mods()
    for i, q in enumerate(h["requests"]):
        if q["finish_reason"] not in ("length", "stop"):
            raise NotExpressibleInV1(f"request {i}: finish_reason {q['finish_reason']!r} has no v1 field (served length only)")
        if q["arrive_step"] < 0:
            raise NotExpressibleInV1(f"request {i}: never scheduled")
    bphi = builder_phi(phi, [q["arrive_step"] for q in h["requests"]])
    hints = hints_mod.Hints(predictor=predict_mod.PREDICTOR_ID)
    # v1 `served` counts *sampled rows*: under async scheduling an EOS stop at the k-th generated token still gets one placeholder decode
    # row (k+1 rows; MEASURED on gen/eos4), so served = n_generated + 1 for "stop"; an EOS at max_tokens-1 is then structurally identical
    # to a length stop (same rows, same Program) and needs no served entry -- from_builder_hints returns it as (max_tokens, length).
    hints.served = [hints_mod.ServedLength(i, _served_rows(phi, q)) for i, q in enumerate(h["requests"]) if _served_rows(phi, q) != q["max_tokens"]]
    T = len(h["steps"])
    rounds = 0
    while True:
        plans = predict_mod.predict_schedule(bphi, hints)
        bad = None
        for t in range(min(T, len(plans))):
            p = plans[t]
            if p.step != t:
                raise NotExpressibleInV1(f"predictor step numbering drifts at plan {t} (step {p.step}): idle gap not expressible in v1")
            if p.rows != _expand(h["steps"][t]["rows"]) or p.logits_rows != _my_logits_as_positions(h["steps"][t]["rows"], h["logits"][t]):
                bad = t
                break
        if bad is None:
            if len(plans) != T:
                raise NotExpressibleInV1(f"predictor emits {len(plans)} steps for the hinted run, record has {T}")
            break
        rows = h["steps"][bad]["rows"]
        mine = _my_logits_as_positions(rows, h["logits"][bad])
        default = predict_mod.StepPlan(bad, _expand(rows), [], []).default_logits_rows()
        override = hints_mod.StepRows(bad, _runs(rows), None if mine == default else tuple(mine))
        hints.steps = [s for s in hints.steps if s.step != bad] + [override]
        hints.steps.sort(key=lambda s: s.step)
        rounds += 1
        if max_rounds is not None and rounds > max_rounds:
            raise NotExpressibleInV1(f"more than {max_rounds} override rounds")
        if rounds > T + 1:
            raise RuntimeError("override loop did not converge")
    return hints.to_json()


def from_builder_hints(phi: dict, hj: dict) -> dict:
    """Their predictor, run with their hints, re-expressed as this lane's absolute ScheduleHint (arrive_step = first step scheduled)."""
    hints_mod, _, predict_mod = _mods()
    hints = hints_mod.Hints.from_json(hj)
    bphi = builder_phi(phi)  # arrivals: the builder-side phi carries arrive_step per request (their workload dict)
    plans = predict_mod.predict_schedule(bphi, hints)
    targets = {i: int(q["max_tokens"]) for i, q in enumerate(phi["requests"])}
    for s in hints.served:
        targets[s.request] = s.served
    steps, logits, first = [], [], {}
    for k, p in enumerate(plans):
        rows = [[r.request, r.first, r.count] for r in p.runs()]
        for i, _, _ in rows:
            first.setdefault(i, k)
        steps.append({"rows": rows})
        logits.append([[int(r), 1 if served else 0] for (r, _), served in zip(p.logits_rows, p.served)])
    reqs = []
    async_sched = bool(phi["engine"].get("async_scheduling", True))
    for i, q in enumerate(phi["requests"]):
        rows_served = targets[i]
        if rows_served == int(q["max_tokens"]):
            g, fr = int(q["max_tokens"]), "length"
        else:
            g, fr = (rows_served - 1 if async_sched else rows_served), "stop"
        reqs.append({"prompt_len": int(q["prompt_len"]), "max_tokens": int(q["max_tokens"]), "arrive_step": first.get(i, -1), "n_generated": g,
                     "finish_reason": fr})
    return {"schema": SCHEMA, "requests": reqs, "steps": steps, "logits": logits}


def _served_rows(phi: dict, q: dict) -> int:
    """Sampled rows of a request = n_generated (+1 placeholder row for an EOS stop under async scheduling)."""
    extra = 1 if (q.get("finish_reason") == "stop" and bool(phi["engine"].get("async_scheduling", True))) else 0
    return int(q["n_generated"]) + extra


def naive_bits(phi: dict, hj: dict) -> int:
    """Their builder-side fixed-width bound for the same hints (label: calculated), for the side-by-side table."""
    hints_mod, _, _ = _mods()
    hints = hints_mod.Hints.from_json(hj)
    maxL = max(int(q["prompt_len"]) for q in phi["requests"])
    maxM = max(int(q["max_tokens"]) for q in phi["requests"])
    return hints.naive_bits(max_tokens=maxM, n_requests=len(phi["requests"]), max_position=maxL + maxM)
