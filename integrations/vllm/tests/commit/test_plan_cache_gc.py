"""A plan the worker caches costs the cyclic collector one tracked object, whatever the plan's tensor count.

Every full collection walks every object the collector tracks. The worker keeps each plan's record for two runs
(`plan_cache`), and every committed step's layout is its plan's record. A record that held one `TensorMeta` per tensor made
two runs' plans about 420k tracked objects at Llama-3.2-1B's 197 tensors per step, and every full collection of the serving
process walked them (circuits' heap census of the network_traces Commit chunk, art:ef0c3e05). Here the committer of
`test_native_collect_plan_cache` commits runs of 9 and of 200 tensors per step, with the identity check on, and after a
collection counts what the cache keeps tracked per plan and what the layouts keep per step, before and after `finalize`.
"""
from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")

from tests.commit import test_native_collect_plan_cache as P  # noqa: E402

RUNS = 3                                                       # the third run retires the first run's plans
BASE = P.step_tensors


def step_tensors(count: int):
    """`step_tensors` grown to `count` tensors per step by more MLP outputs, sized by the step's token count."""

    def tensors(kv: tuple[int, ...]) -> list[tuple[str, torch.Tensor]]:
        out = BASE(kv)
        for i in range(count - len(out)):
            v = (torch.arange(len(kv) * P.H, dtype=torch.int64) * 3 + sum(kv) + i) % 251 - 125
            out.append((f"model.layers.{P.LAYERS + i}.mlp.down_proj", v.to(torch.bfloat16).reshape(len(kv), P.H)))
        return out

    return tensors


def counted(monkeypatch, native_worker: bool, count: int) -> dict[str, list[int]]:
    """Over RUNS runs whose every step learns a new plan, after a collection: the objects the layouts keep tracked at the end
    of each run's steps, and those the layouts and the cache (both generations) keep after its `finalize`."""
    monkeypatch.setattr(P, "step_tensors", step_tensors(count))
    com = P.committer(monkeypatch, native_worker=native_worker)
    modules = sorted({com._module_of(n) for n, _t in P.step_tensors((40, 45))} - {"fa2.m1"})
    com.expected_manifest = lambda: {"modules": modules, "attention_layers": [], "fa2_hidden_per_layer": 0, "sampler_ids": 1}
    seen: dict[str, list[int]] = {"cache": [], "steps": [], "finalized": [], "plans": []}
    finalize = com.finalize

    def counting_finalize():
        gc.collect()
        seen["steps"].append(P.tracked(com._layouts))
        rc = finalize()
        gc.collect()
        seen["finalized"].append(P.tracked(com._layouts))
        seen["cache"].append(P.retained(com))
        seen["plans"].append(len(com._metas_cache))
        assert rc.extra["identity_check"]["ok"] is True and rc.extra["identity_check"]["steps_checked"] == P.STEPS
        return rc

    com.finalize = counting_finalize
    for r in range(RUNS):
        P.run(com, P.schedule(40 + r * P.STEPS, P.STEPS))
    assert com.errors == [] and len(P.step_tensors((40, 45))) == count
    return seen


@pytest.mark.parametrize("native_worker", [False, True], ids=["python-worker", "native-worker"])
def test_a_cached_plan_keeps_one_object_tracked_whatever_its_tensor_count(monkeypatch, native_worker):
    """At 9 and at 200 tensors per step, the cache keeps one tracked object per plan (its record) beside at most four of its
    own (the list `retained` walks, the cache and its two generations), each step's layout entry is its plan's record beside
    the layouts' dict, and `finalize` builds no `TensorMeta` that stays."""
    small, large = counted(monkeypatch, native_worker, 9), counted(monkeypatch, native_worker, 200)
    for seen in (small, large):
        assert seen["plans"] == [P.STEPS, 2 * P.STEPS, 2 * P.STEPS]
        assert all(kept - plans <= 4 for kept, plans in zip(seen["cache"], seen["plans"])), seen
        assert seen["steps"] == seen["finalized"] == [P.STEPS + 1] * RUNS, seen
    assert large == small, f"tracked at 9 tensors per step {small}, at 200 {large}"
