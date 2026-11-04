"""A plan the worker caches costs the cyclic collector one tracked object, whatever the plan's tensor count.

Every full collection walks every object the collector tracks. The worker keeps each plan's record for two runs
(`plan_cache`), and every committed step's layout is its plan's record. A record that held one `TensorMeta` per tensor made
two runs' plans about 420k tracked objects at Llama-3.2-1B's 197 tensors per step, and every full collection of the serving
process walked them (a heap census of the network_traces Commit chunk). Here the committer of
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


def test_a_record_answers_its_readers_as_its_metas_do_without_building_them():
    """The readers that take a record's columns (the retain-exclude ranges, the layout summary, the names) or a pass-only
    `scan()` (the layout digest, the step's sequences) get what the plan's `TensorMeta`s give, and the record builds none;
    iterating it builds them once, so every pass sees the same objects."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter as NH

    record, metas = _record(), list(_record())
    ranges = P.NC._excluded_chunk_ranges(record, {"fa2_hidden_m1"}, 256)
    assert ranges == P.NC._excluded_chunk_ranges(metas, {"fa2_hidden_m1"}, 256) and len(ranges) == 2
    assert record.summary() == [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes] for m in metas]
    assert list(record.names) == [m.name for m in metas] and record.nbytes == sum(m.nbytes for m in metas)
    assert NH.layout_digest(record.scan()) == NH.layout_digest(metas)
    assert repr(NH._step_sequences(P.SimpleNamespace(), 0, record.scan())) == repr(NH._step_sequences(P.SimpleNamespace(), 0, metas))
    assert record._metas is None
    assert [id(m) for m in record] == [id(m) for m in record] and record[2] is record.scan()[2] and len(record) == len(metas)


NAMES = ["model.embed_tokens/in", "fa2.m1/L0/acc", "model.layers.0.mlp.down_proj", "fa2.m1/L1/acc", "runner.sampler/sampled_token_ids"]


def _record():
    """A plan's record of five tensors, two of them attention interiors, as the collector exports it."""
    from verity_vllm.commit.committer.plan_cache import PlanMetas

    entries, off = [], 0
    for i, (dtype, shape, nbytes) in enumerate([(torch.int32, (5,), 20), (torch.float32, (300, 4), 4800), (torch.bfloat16, (5, 24), 240),
                                                (torch.float32, (200, 4), 3200), (torch.int32, (), 4)]):
        gap = -off % 4
        entries.append((i, dtype, shape, nbytes, off + gap, gap))
        off += gap + nbytes
    return PlanMetas.of_entries(entries, NAMES, 256)


def test_a_records_columns_rebuild_it_and_a_committed_read_through_them_builds_no_metas():
    """`PlanMetas.columns`, what the seal's process is sent for each step (`seal_process`), rebuild the record there with the same
    metas, and `served_record.committed_raw` reads a committed tensor's bytes through the record's columns as `committed_ints`
    read them through its metas, building none."""
    import pickle

    from verity_vllm.commit.committer.plan_cache import PlanMetas
    from verity_vllm.commit.served_record import committed_ints, committed_raw

    record = _record()
    record.digest = b"\x07" * 32
    again = PlanMetas.of_columns(pickle.loads(pickle.dumps(record.columns())))
    assert record._metas is None and list(again) == list(record) and (again.nbytes, again.digest) == (record.nbytes, record.digest)
    fresh = _record()
    blk = (torch.arange(fresh.nbytes + 8, dtype=torch.int64) % 251).to(torch.uint8)
    com = P.SimpleNamespace(_layouts={0: fresh}, _gpu_blocks={0: (blk, fresh.nbytes)})
    built = P.SimpleNamespace(_layouts={0: list(_record())}, _gpu_blocks={0: (blk, fresh.nbytes)})
    for name in NAMES:
        assert committed_raw(com, 0, name) == committed_raw(built, 0, name) is not None
    assert committed_ints(com, 0, NAMES[0]) == committed_ints(built, 0, NAMES[0])
    assert committed_raw(com, 0, "model.norm") is None and committed_raw(com, 1, NAMES[0]) is None and fresh._metas is None


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
