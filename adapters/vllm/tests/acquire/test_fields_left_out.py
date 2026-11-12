"""A field a document may leave out reads as empty whatever false value holds its place: left out, null, an empty list,
dict or string, zero and False read alike.  The documents are the ones `acquire` reads from files: a manifest (its rows,
under any of the three names a writer gave them, and a row's consumers), a plan document's hooks, a kernel's launch record,
a launch config, the fused-MoE record and a compile artifact's parts."""
from __future__ import annotations

from typing import Any

import pytest

from tests.acquire.test_plan import manifest, row
from tests.acquire.test_served_moe import RECORD, SMALL, _wired
from verity_vllm.acquire import gate as G
from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire import plan as P
from verity_vllm.acquire import served_artifact as SA
from verity_vllm.acquire import served_moe as SM
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.acquire.sources import call_boundary_source as CB

LEFT_OUT = object()
EMPTY = [LEFT_OUT, None, [], {}, "", 0, False]
IDS = ["left out", "null", "[]", "{}", "''", "0", "False"]
empty = pytest.mark.parametrize("held", EMPTY, ids=IDS)


def holding(doc: dict[str, Any], key: str, held: Any) -> dict[str, Any]:
    """`doc` with `held` in `key`'s place."""
    rest = {k: v for k, v in doc.items() if k != key}
    return rest if held is LEFT_OUT else {**rest, key: held}


def _rows() -> list[dict[str, Any]]:
    return [row("instance_outputs", "body.blocks.0.act", "out", spec="Act_v1", consumers=["body.blocks.0.norm_out"]),
            row("instance_outputs", "body.embed", "out", spec="Embedding_v1")]


NAMES = ("identities", "entries", "rows")


@pytest.mark.parametrize("name", NAMES)
@empty
def test_a_manifests_rows_are_read_under_the_first_name_that_holds_any(name: str, held: Any) -> None:
    want = P.value_classes(manifest(_rows()))
    assert len(want) == 2
    doc = holding(manifest(_rows()), "identities", LEFT_OUT)
    for earlier in NAMES[:NAMES.index(name)]:
        doc = holding(doc, earlier, held)
    assert P.value_classes({**doc, name: _rows()}) == want


@empty
def test_a_manifest_without_rows_has_no_value_and_no_call_boundary(held: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    man = holding(manifest(_rows()), "identities", held)
    assert P.value_classes(man) == []
    monkeypatch.setattr(G.taps, "attached", lambda manifest: [])            # the sources' own reading of a manifest is theirs
    monkeypatch.setattr(CB, "unattached", lambda manifest, on: [])
    assert G.call_boundaries_of(man, None)["identities"] == 0


@empty
def test_a_row_names_no_consumer(held: Any) -> None:
    (v,) = P.value_classes(manifest([holding(_rows()[0], "consumers", held)]))
    assert (v.value_id, v.consumers) == ("body.blocks.0.act/out", [])


@empty
def test_a_kernel_recorded_without_candidates_ran_its_one_config(held: Any) -> None:
    ran = {"kwargs": {"XBLOCK": 2}, "num_warps": 16}
    record = {T.source_sha256("kernel"): holding({"candidates": [ran], "executed": [ran]}, "candidates", held)}
    assert T.launch_config(record, "kernel") == ran


@empty
def test_a_kernel_recorded_without_an_executed_config_is_refused(held: Any) -> None:
    record = {T.source_sha256("kernel"): holding({"candidates": [], "executed": [{"num_warps": 16}]}, "executed", held)}
    with pytest.raises(Unmodelled, match=r"0 candidate configs, its autotuners hold \[\]"):
        T.launch_config(record, "kernel")


@empty
def test_a_launch_config_without_kwargs_names_no_block_size(held: Any) -> None:
    t = T.Template("STATIC", "pointwise", {}, "def KERNEL():", lambda lit: True, (), lambda lit: {}, config=True)
    got = T.instantiate("kernel", t, {}, holding({"kwargs": {"XBLOCK": 2}, "num_warps": 16}, "kwargs", held))
    assert got["literals"] == {"XBLOCK": None, "R0_BLOCK": None, "NUM_WARPS": 16}


@empty
def test_a_moe_record_without_configs_has_no_launch_at_any_row_count(held: Any) -> None:
    plan = SM.expand(_wired(), holding(RECORD, "configs", held))[1]
    with pytest.raises(Unmodelled, match=r"no fused_moe_kernel launch at 8 rows \(MUL_ROUTED_WEIGHT (True|False)\) was recorded"):
        plan(8)


@pytest.mark.parametrize("held", EMPTY[1:], ids=IDS[1:])
def test_a_row_count_holding_nothing_has_no_launch(held: Any) -> None:
    plan = SM.expert_plans({SM.plan_key(8, 32, 32, False)}, {"8": held, "256": {"False": SMALL, "True": SMALL}})
    with pytest.raises(Unmodelled, match=r"no fused_moe_kernel launch at 8 rows \(MUL_ROUTED_WEIGHT False\) was recorded"):
        plan(8)


@pytest.mark.parametrize("part", SA.PARTS)
@empty
def test_an_artifact_without_a_part_differs_by_each_name_of_the_other(part: str, held: Any) -> None:
    whole = {"kernels": {"aa11": {"num_warps": 4}}, "winners": {"pp/9eef.best_config": {"R0_BLOCK": 32}}}
    (name,) = whole[part]
    assert SA.differences(whole, holding(whole, part, held)) == [f"{part} {name}: pinned {whole[part][name]}, loaded absent"]
    assert SA.differences(holding(whole, part, held), whole) == [f"{part} {name}: pinned absent, loaded {whole[part][name]}"]
