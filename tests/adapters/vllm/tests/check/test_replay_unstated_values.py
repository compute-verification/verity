"""What `verity_vllm.check.replay` does with a value its caller's contract leaves unstated, on the sampled replay's CPU-served
world (`tests.check.test_sampled_replay`): an evaluator that returns neither outputs nor a reason, a cached-prefix entry that is
no mapping beside a Program that takes no cached prefix, and a schedule reader's row with no member at its port 0.  None is
checked ahead of its use, so each ends where the value is first read."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.check import test_sampled_replay as T
from verity_vllm.check.replay import checks as CK
from verity_vllm.check.replay import driver as DR
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay import request_replay as RR
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.query.manifest import format as RM


@pytest.fixture(scope="module")
def world() -> Any:
    inst, woff = T._program()
    weights = T._bf16(np.random.default_rng(7), max(hi for _lo, hi in woff.values()), 0.5)
    return T.Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


@pytest.fixture(scope="module")
def store(world: Any) -> CommittedStore:
    return CommittedStore(*world.bmap_and_reader())


@pytest.fixture(scope="module")
def gemm(world: Any) -> int:
    """A row the population samples, whose operand another row of its step produces."""
    return next(int(r["i"]) for r in world.pi.rows if (d := world.pi.describe(r)) is not None and d["members"] and d["family"].startswith("Gemm"))


@pytest.fixture
def no_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """An evaluator installed over the replay's (`evaluate.ROW_EVALUATORS`) that returns nothing for every row."""
    monkeypatch.setattr(EV, "ROW_EVALUATORS", [lambda pi, r, d, operands: None])


def test_a_row_whose_evaluator_gives_no_outputs_and_no_reason_is_no_record_of_the_row(world, store, gemm, no_outputs):
    """`evaluated` gives a reason for a row it could not evaluate; with none, its readers go on to the outputs."""
    weights = T._weights(world)
    assert EV.evaluated(world.pi, store, weights, world.pi.by_i[gemm], "r0", 0, []) == (None, None)
    with pytest.raises(TypeError, match="object of type 'NoneType' has no len"):
        EV._replay_vu(world.pi, store, weights, {"i": gemm, "request_id": "r0", "phase": "prefill"}, 0, {})
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'values'"):
        CK._check(world.pi, store, weights, "r0", 0, None, "a check", gemm, 0, 1, 1, {})
    with pytest.raises(LookupError, match=r"of request r0: evaluator error: TypeError\(\"object of type 'NoneType' has no len\(\)\"\)"):
        RR.Replay(world.pi, "r0", world.prompts["r0"], weights)._evaluate(gemm)


def test_the_same_row_under_the_replay_s_own_evaluator_is_recorded(world, store, gemm):
    weights = T._weights(world)
    record = EV._replay_vu(world.pi, store, weights, {"i": gemm, "request_id": "r0", "phase": "prefill"}, 0, {})
    assert record["result"] is True and "why" not in record
    replay = RR.Replay(world.pi, "r0", world.prompts["r0"], weights)
    replay._evaluate(gemm)
    assert [member for _op, member in replay.out[gemm]] == ["out"]


def test_a_row_its_index_does_not_describe_ends_at_the_first_read_of_its_description(world, store, gemm, monkeypatch):
    """A population holds only the rows its index describes and a replay evaluates only those (`Replay._ensure`); one handed
    another is not refused ahead of the read."""
    weights = T._weights(world)
    replay = RR.Replay(world.pi, "r0", world.prompts["r0"], weights)
    monkeypatch.setattr(world.pi, "describe", lambda r: None)
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        EV._replay_vu(world.pi, store, weights, {"i": gemm, "request_id": "r0", "phase": "prefill"}, 0, {})
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        replay._evaluate(gemm)
    replay._ensure(gemm)
    assert replay.out == {} and replay.rows == 0


@pytest.mark.parametrize("entry, named", [(None, "'NoneType' object has no attribute 'get'"), ([], "'list' object has no attribute 'get'")])
def test_a_cached_prefix_entry_that_is_no_mapping_is_read_whatever_its_request_s_program_takes(world, entry, named):
    programs = {rid: world.inst for rid in world.prompts}

    def populate(cached: dict) -> list:
        return DR.populate(programs, [0], {}, None, None, attention_paths=None, tp=None, program_dirs=None, manifest=None,
                           cached_prefixes=cached)[0]

    served_cold = populate({})
    assert served_cold and populate({"r0": {"cached": 0}, "a request with no Program": entry}) == served_cold
    with pytest.raises(AttributeError, match=named):
        populate({"r0": entry})


def test_a_schedule_reader_with_no_member_at_port_0_reads_the_member_its_description_names_none():
    """No description names a member None; one that did, of the slice's width, is the member the row is keyed by."""
    reader = sorted(RM.SCHEDULE_READERS)[0]
    index: Any = SimpleNamespace(member_of_port=lambda i: {})
    read: list[Any] = []

    class Store:
        def words(self, rid: str, step: int, op_path: str, member: Any, lo: int, hi: int) -> tuple[np.ndarray, str]:
            read.append(member)
            return np.arange(lo, hi, dtype=np.uint32), "u32"

    def witness(members: list) -> np.ndarray:
        d = {"family": reader, "members": members, "row": 0, "step": 0, "op_path": "moe/L0"}
        store: Any = Store()
        return EV._schedule_witness(index, store, {"i": 0}, d, "r0", 0, RM.SCHEDULE_PARAM, 0, 4, [])

    assert witness([(None, 4, "u32")]).tolist() == [0, 1, 2, 3] and read == [None]
    for members in ([("rank", 4, "u32")], [(None, 8, "u32")], []):
        with pytest.raises(EV._Unresolved, match="row's own committed rank"):
            witness(members)
