"""What the sources read from a record that leaves an optional field out: absent, null and empty read alike, and a stated one is read.
A field that holds another empty value (`OTHER_EMPTY`) in its block's place reads as the absent one too.

Each reader below takes a record whose writer may leave a field out: a step's layout and a capture's stored tensors
(`verity_vllm.acquire.sources.partial_source`).  CPU only."""
from __future__ import annotations

import json
import types
from typing import Any

import pytest
import torch

from tests.check.opened import commit_steps
from verity_vllm.acquire.sources import partial_source as P
from verity_vllm.check import oracle_compare as OC
from verity_vllm.commit import capture_map as CM

#: a manifest that names no identity: none given, the field absent, null and empty
NO_IDENTITIES: list[dict[str, Any] | None] = [None, {}, {"identities": None}, {"identities": []}]
#: the other empty values a writer may leave in an optional block's place; each reads as the absent block
OTHER_EMPTY: list[Any] = [0, "", False, 0.0, []]
NARROWED = {"narrowed_of": "0", "op_path": "model.layers.1.attn.qkv", "output_member": "0@4", "columns": [2, 6], "width": 4, "dtype": "bf16"}
NARROWED_SITES = {"model.layers.1.attn.qkv": {"0@4": {"columns": (2, 6), "bytes": 2}}}
OPROJ = "model.layers.0.self_attn.o_proj"














def _body(rows: int = 4) -> Any:
    """A body's size as `_arena_bound` reads it: one bf16 target of 8 words a row, at one position."""
    return types.SimpleNamespace(parts=[types.SimpleNamespace(rows=rows)],
                                 targets=[types.SimpleNamespace(shared=False, at=(0,), lo=0, hi=8, dtype="bf16")])














def _made(states: list[Any]) -> list[dict[str, Any]]:
    """Each rank's record with what it recorded of its sources: `...` leaves the field out."""
    return [{"rank": r} if s is ... else {"rank": r, "tap_sources": s} for r, s in enumerate(states)]






def _map(name: str, **fields: Any) -> dict[str, Any]:
    return {"schema": CM.SCHEMA, "id": name, "route_site": {"module": "m"}, "target": {"tensor_parallel_size": 2}, **fields}












def test_a_name_a_step_committed_nothing_under_has_no_occurrence():
    """`verity_vllm.acquire.sources.partial_source._nth`: occurrence k of a name in a step's layout, None past its last and for a name
    the layout does not hold."""
    a, b = object(), object()
    by: dict[str, Any] = {"m/out": [a, b], "m/none": [], "m/null": None, **{f"m/empty{i}": v for i, v in enumerate(OTHER_EMPTY)}}
    assert P._nth(by, "m/out", 0) is a and P._nth(by, "m/out", 1) is b and P._nth(by, "m/out", 2) is None
    assert P._nth(by, "m/absent", 0) is None and P._nth(by, "m/none", 0) is None and P._nth(by, "m/null", 0) is None
    assert [P._nth(by, f"m/empty{i}", 0) for i in range(len(OTHER_EMPTY))] == [None] * len(OTHER_EMPTY)


class RowParallelLinear:
    reduce_results, tp_size = True, 2

    def parameters(self):
        return iter(())


@pytest.mark.parametrize("stored", [{}, {"part": None, "out": None}, {"part": [], "out": []}, *({"part": v, "out": v} for v in OTHER_EMPTY)])
def test_a_capture_step_that_stored_no_tensors_is_named_inconsistent_with_its_record(tmp_path, stored):
    """`verity_vllm.acquire.sources.partial_source.TPPartialSource.compare_match_oracle`: the step's `.pt` leaves `part` and `out` out
    while the capture's record names one collective at the step."""
    part = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
    (tmp_path / "coll.rank0.jsonl").write_text(
        json.dumps({"rank": 0, "step": 0, "k": 0, "kind": "all_reduce", "module": OPROJ, "shape": list(part.shape)}) + "\n")
    torch.save({"step": 0, **stored}, tmp_path / "coll.rank0.step0.pt")
    com, run = commit_steps({0: [(f"{OPROJ}/part_rank0", P._bytes(part))]})
    src = P.TPPartialSource(com, types.SimpleNamespace(named_modules=lambda: [(OPROJ, RowParallelLinear())]), rank=0, world=2)
    res = src.compare_match_oracle(str(tmp_path), OC.OpenedReader(com, run))
    assert res["verdict"] == "FAIL" and res["oracle_inconsistent"] == [{"step": 0, "jsonl_names": 1, "pt_part": 0, "pt_out": 0}]
    assert res["tensors"] == 0 and "error" not in res


#: a tap's `commit` block that names no binder, or another source's: left out (`...`), null, empty, another empty value, or a block
NOT_THE_SOURCES: list[Any] = [..., None, {}, *OTHER_EMPTY, {"name": "a/b"}, {"binder": None, "name": "a/b"}, {"binder": "another", "name": "a/b"}]


def _tap(block: Any) -> dict[str, Any]:
    tap: dict[str, Any] = {"id": "t", "role": "r", "site": {"function": "f", "module": "m", "output": 0}}
    return tap if block is ... else {**tap, "commit": block}


