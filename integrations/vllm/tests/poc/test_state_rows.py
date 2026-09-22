"""State rows R2/R3/R4 (W10): position-determined phase, cached-prefix binding, recompute units."""

from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm import identity
from verity_vllm.capture import (
    Boundary,
    CachedPrefix,
    CaptureReader,
    RunWriter,
    StepRecord,
)
from verity_vllm.description import Description
from verity_vllm.synthetic import toy_description
from verity_vllm_adapter import state_rows

from _helpers import TINY_MANIFEST

PID = "0123456789ab"


def _boundary(rid, prompt, outputs, steps, cached=()):
    return Boundary.build(
        rid, prompt, outputs, {"max_tokens": len(outputs)}, seed=None, finish_reason="length", steps=steps,
        cached_prefix=list(cached),
    )


# --- R2: a prefill split over several steps is still one identity space --------------------------


def test_chunked_prefill_steps_cover_positions_once():
    b = _boundary("r", list(range(10, 18)), [1, 2], [
        StepRecord(0, "prefill", (0, 1, 2)), StepRecord(1, "prefill", (3, 4, 5)), StepRecord(2, "prefill", (6, 7)),
        StepRecord(3, "decode", (8,)),
    ])
    assert b.validate() == []
    assert b.step_of(4).step == 1 and b.phase_at(4) == "prefill" and b.phase_at(8) == "decode"
    desc = Description(toy_description(PID, layers=1, hidden=4, vocab=8))
    ids = desc.expected_ids(b, PID)
    # every processed position once; a sampled row only at the end of the last chunk and the decode
    assert sum("/l=sampler/" in c for c in ids) == 2
    assert sorted({int(c.split("/t=")[1].split("/")[0]) for c in ids}) == list(range(9))


# --- R4: recompute after preemption -------------------------------------------------------------


def test_recompute_step_keeps_identities_and_mixed_phase():
    # prompt 4, two outputs sampled (t=3, t=4), preempted, recomputed 0..4 in one chunk (mixed), then decode
    b = _boundary("r", [1, 2, 3, 4], [5, 6, 7], [
        StepRecord(0, "prefill", (0, 1, 2, 3)), StepRecord(1, "decode", (4,)),
        StepRecord(2, "mixed", (0, 1, 2, 3, 4, 5), recompute=True),
    ])
    assert b.validate() == []
    assert b.recomputed_positions() == {2: [0, 1, 2, 3, 4]}
    assert b.step_of(2).step == 0 and b.phase_at(4) == "decode"
    desc = Description(toy_description(PID, layers=1, hidden=4, vocab=8))
    ids = desc.expected_ids(b, PID)
    ts = sorted(int(c.split("/t=")[1].split("/")[0]) for c in ids if "/op=embed_tokens/" in c)
    assert ts == [0, 1, 2, 3, 4, 5]  # listed once each although step 2 repeated 0..4
    sampled = sorted(int(c.split("/t=")[1].split("/")[0]) for c in ids if "/l=sampler/" in c)
    assert sampled == [3, 4, 5]
    # a mixed step that is not a recompute, or a recompute whose repeated positions are not the chunk's prefix, is refused
    bad = _boundary("r", [1, 2, 3, 4], [5, 6, 7], [
        StepRecord(0, "prefill", (0, 1, 2, 3)), StepRecord(1, "decode", (4,)), StepRecord(2, "mixed", (0, 1, 2, 3, 4, 5)),
    ])
    assert any("phase 'mixed'" in p for p in bad.validate())
    bad = _boundary("r", [1, 2, 3, 4], [5, 6, 7], [
        StepRecord(0, "prefill", (0, 1, 2, 3)), StepRecord(1, "decode", (4,)), StepRecord(2, "mixed", (5, 0, 1, 2, 3, 4), recompute=True),
    ])
    assert any("not the chunk's prefix" in p for p in bad.validate())


def _run_with_recompute(run_dir, *, stale: bool):
    run = RunWriter(run_dir, TINY_MANIFEST)
    pid = run.profile_id

    def emb(t, ph):
        return str(identity.value_id(pid, "req-0", t, ph, "embed", "embed_tokens", 0))

    ru = run.ru("req-0", 0, phase="prefill", positions=[0, 1, 2])
    for t in range(3):
        ru.add(emb(t, "prefill"), "bfloat16", [4], np.full(4, 0x3F80 + t, np.uint16))
    ru.add(str(identity.value_id(pid, "req-0", 2, "prefill", "sampler", "sampler.select", 0)), "int32", [], np.array(7, np.int32))
    ru.finish()
    ru = run.ru("req-0", 1, phase="mixed", positions=[0, 1, 2, 3])
    for t in range(3):
        val = 0x3F80 + t + (1 if (stale and t == 1) else 0)
        ru.add(emb(t, "prefill"), "bfloat16", [4], np.full(4, val, np.uint16))
    ru.add(emb(3, "decode"), "bfloat16", [4], np.full(4, 0x3F83, np.uint16))
    ru.add(str(identity.value_id(pid, "req-0", 3, "decode", "sampler", "sampler.select", 0)), "int32", [], np.array(9, np.int32))
    ru.finish()
    b = _boundary("req-0", [1, 2, 3], [7, 9], [StepRecord(0, "prefill", (0, 1, 2)), StepRecord(1, "mixed", (0, 1, 2, 3), recompute=True)])
    assert b.validate() == []
    run.write_boundary(b)
    w = run.weights(store_values=True)
    w.add("embed_tokens.weight", "bfloat16", [16, 4], np.arange(64, dtype=np.uint16))
    w.finish()
    run.finish()
    return CaptureReader(run_dir)


def test_recompute_unit_must_repeat_the_first_units_bytes(tmp_path):
    reader = _run_with_recompute(tmp_path / "ok", stale=False)
    assert reader.recompute_problems("req-0") == []
    assert reader.recompute_problems("req-0", check_bytes=True) == []
    # locate binds the identity to the FIRST unit that computed it
    pid = reader.profile_id
    assert reader.locate(str(identity.value_id(pid, "req-0", 1, "prefill", "embed", "embed_tokens", 0))) == ("req-0", 0)
    stale = _run_with_recompute(tmp_path / "stale", stale=True)
    problems = stale.recompute_problems("req-0")
    assert problems and "different bytes" in problems[0]
    # the post-hoc report agrees
    rep = state_rows.recompute_consistency(tmp_path / "stale")
    assert rep["recomputed_leaves_compared"] == 3 and rep["recomputed_leaves_identical"] == 2 and not rep["all_identical"]
    rep = state_rows.recompute_consistency(tmp_path / "ok")
    assert rep["all_identical"] and rep["requests_with_recompute"] == 1


def test_n05_recompute_negative_perturbs_only_the_recompute_copy(tmp_path):
    """The r4 row's first N05 run accepted its own perturbation: a plain PatchedReader patches an id in
    every unit holding it (both copies stay equal). The unit-scoped reader changes one copy."""
    from verity_vllm.capture import Entry, Value
    from verity_vllm.negatives import PatchedReader
    from verity_vllm.negatives_state import UnitPatchedReader

    reader = _run_with_recompute(tmp_path / "ok", stale=False)
    pid = reader.profile_id
    cid = str(identity.value_id(pid, "req-0", 1, "prefill", "embed", "embed_tokens", 0))
    new_bytes = np.full(4, 0x3F81 + 1, np.uint16).tobytes()
    # the bug: both copies patched -> no problem found
    both = PatchedReader(reader, values={cid: new_bytes})
    assert both.recompute_problems("req-0", check_bytes=True) == []
    # stale bytes in the recompute copy only: the byte-level audit sees it, the index level does not
    one = UnitPatchedReader(reader, unit=("req-0", 1), values={cid: new_bytes})
    assert one.recompute_problems("req-0") == []
    assert "different bytes" in one.recompute_problems("req-0", check_bytes=True)[0]
    assert one.ru("req-0", 0).value(cid).data == reader.ru("req-0", 0).value(cid).data
    # a consistently committed different copy: the verifier's index-level check rejects
    e = reader.ru("req-0", 1).entry(cid)
    new_entry = Entry(cid, e.dtype, e.shape, e.offset, e.nbytes, e.origin, Value(cid, e.dtype, e.shape, new_bytes).leaf().hex())
    committed = UnitPatchedReader(reader, unit=("req-0", 1), values={cid: new_bytes}, entries={cid: new_entry})
    assert "different bytes" in committed.recompute_problems("req-0")[0]
    assert committed.ru("req-0", 0).entry(cid).leaf == e.leaf  # the original copy is untouched


# --- R3: cached prefix bound to the source request -------------------------------------------------


def test_cached_prefix_binds_reads_to_the_source():
    src = _boundary("A", [1, 2, 3, 4, 5, 6], [7, 8], [StepRecord(0, "prefill", (0, 1, 2, 3, 4, 5)), StepRecord(1, "decode", (6,))])
    hit = _boundary("B", [1, 2, 3, 4, 9, 9], [7, 8], [StepRecord(0, "prefill", (4, 5)), StepRecord(1, "decode", (6,))],
                    cached=[CachedPrefix("A", 6, (0, 1, 2, 3))])
    assert hit.validate() == []
    assert hit.cached_positions() == {0, 1, 2, 3} and hit.cached_source(2).source_request_id == "A" and hit.cached_source(4) is None
    assert hit.cached_prefix_problems(lambda rid: {"A": src}.get(rid)) == []
    desc = Description(toy_description(PID, layers=1, hidden=4, vocab=8))
    ids = desc.expected_ids(hit, PID)
    assert sorted({int(c.split("/t=")[1].split("/")[0]) for c in ids}) == [4, 5, 6]  # B never computed 0..3
    # an arg over all positions <= t resolves the cached ones to A's identities
    node = desc.node("layers.0.mlp.gemm")
    inst = desc.instance_of(str(identity.value_id(PID, "B", 5, "prefill", "0", node.op, 0)), hit)
    from verity_vllm.description import Arg

    arg = Arg(name="kv", source="node", op="embed_tokens", port=0, t="all_le")
    refs = desc.resolve_arg(arg, inst, hit, PID)
    assert [identity.parse(r).request_id for r in refs] == ["A", "A", "A", "A", "B", "B"]
    assert all(identity.parse(r).t == i for i, r in enumerate(refs))
    # the binding is refused when the source's tokens differ on 0..t, or when the source never computed t
    other = _boundary("C", [1, 2, 0, 4, 5, 6], [7, 8], [StepRecord(0, "prefill", (0, 1, 2, 3, 4, 5)), StepRecord(1, "decode", (6,))])
    problems = hit.cached_prefix_problems(lambda rid: {"A": other}.get(rid))
    assert problems and "tokens differ at t=2" in problems[0]
    short = _boundary("A", [1, 2, 3, 4, 5, 6], [7, 8], [StepRecord(0, "prefill", (2, 3, 4, 5)), StepRecord(1, "decode", (6,))],
                      cached=[CachedPrefix("Z", 6, (0, 1))])
    problems = hit.cached_prefix_problems(lambda rid: {"A": short}.get(rid))
    assert problems and "no step computed t=0" in problems[0]
    # structural: cached positions must be the leading prefix and never recomputed
    bad = _boundary("B", [1, 2, 3, 4, 9, 9], [7, 8], [StepRecord(0, "prefill", (2, 3, 4, 5)), StepRecord(1, "decode", (6,))],
                    cached=[CachedPrefix("A", 6, (0, 1, 2, 3))])
    assert bad.validate()
    doc = hit.to_json()
    again = Boundary.from_json(doc)
    assert again.cached_prefix == hit.cached_prefix and again.validate() == []


# --- declared engine overrides ---------------------------------------------------------------------


def test_engine_overrides_round_trip():
    ov = state_rows.parse_engine_args(["max_num_batched_tokens=256", "enable_prefix_caching=false", "gpu_memory_utilization=0.5", "num_gpu_blocks_override=300"])
    assert ov == {"max_num_batched_tokens": 256, "enable_prefix_caching": False, "gpu_memory_utilization": 0.5, "num_gpu_blocks_override": 300}
    with pytest.raises(SystemExit):
        state_rows.parse_engine_args(["enforce_eager=false"])
    manifest = {"engine_overrides": dict(ov, enable_prefix_caching=True)}
    assert state_rows.replay_engine_overrides(manifest) == {"max_num_batched_tokens": 256, "gpu_memory_utilization": 0.5, "num_gpu_blocks_override": 300}
    assert state_rows.replay_engine_overrides(manifest, prefix_caching=True)["enable_prefix_caching"] is True
    assert state_rows.replay_engine_overrides({}) == {}


def test_shared_prefix_workload(tmp_path):
    from verity_vllm_sampler.workload import SCHEMA, Request, Workload, base_request_id

    reqs = []
    for i, n in enumerate((32, 128, 512, 2048)):
        ids = [(i * 1000 + k) % 5000 + 3 for k in range(n)]
        reqs.append(Request(base_request_id("m", ids, 16), i, n, 16, ids, "", False, i))
    Workload(SCHEMA, "m", None, 0, reqs).save(str(tmp_path / "base.json"))
    design = state_rows.build_shared_prefix_workload(tmp_path / "base.json", tmp_path / "r3.json", block_size=16, tail=64)
    out = json.loads((tmp_path / "r3.json").read_text())
    assert design["hits"] == 2 and design["requests"] == 6 and len(out["requests"]) == 6
    by_id = {r["request_id"]: r for r in out["requests"]}
    for pair in design["pairs"]:
        s, h = by_id[pair["source"]], by_id[pair["hit"]]
        k = pair["shared_tokens"]
        assert s["prompt_token_ids"][:k] == h["prompt_token_ids"][:k] and s["prompt_token_ids"] != h["prompt_token_ids"]
        assert k % 16 == 0 and pair["expected_num_cached_tokens"] == k
    # sources come before hits in the request order
    order = [r["request_id"] for r in out["requests"]]
    assert all(order.index(p["source"]) < order.index(p["hit"]) for p in design["pairs"])
