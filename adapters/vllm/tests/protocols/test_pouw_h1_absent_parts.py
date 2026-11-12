"""What H1's readers make of a part their input leaves out: a rank's release whose committer states no errors, a pull
from a rank that shipped no acquisitions, a window's record with no commitment object, a slot of a round the round
log does not hold, and an engine that holds no executor in its own process."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from verity.protocols.spot.device.challenger import OPENING_FORM, LiveCommitment

from tests.protocols.test_pouw_h1_served import _unit
from verity_vllm.protocols.pouw.h1.challenger import drawn_of
from verity_vllm.protocols.pouw.h1.commit import ROOT_BYTES, SCHEME
from verity_vllm.protocols.pouw.h1.ranks import Holder, RankExtension, executor_of
from verity_vllm.protocols.pouw.h1.requests import RoundOpening, slot_reason
from verity_vllm.protocols.pouw.h1.served import declared, descriptor

#: what a rank's release hands on of its committer's finalize beside the errors (`h1rank_release`)
COUNTS = {"expected_tensors": 2, "committed_tensors": 2, "errors_n": 0}


class _Committer:
    """What `RankExtension.h1rank_release` reads of a rank's committer: rows-sha512's scheme, its name and its finalize's
    ``extra``."""

    commit_scheme, name = SCHEME, "stand-in"

    def __init__(self, extra: dict[str, Any]) -> None:
        self.extra, self.closed = extra, False

    def finalize(self) -> SimpleNamespace:
        return SimpleNamespace(run_root=bytes(ROOT_BYTES), extra=self.extra)

    def close(self) -> None:
        self.closed = True


def _released(extra: dict[str, Any]) -> dict[str, Any]:
    """What the rank at linear 3 releases when its committer's finalize states `extra`; the committer closed, the tap off
    and the hooks out, in that order, and the rank left with no window."""
    com, done = _Committer(extra), []
    rank = RankExtension()
    rank._h1rank = {"run": SimpleNamespace(lo=3), "sent": 0, "hooks": SimpleNamespace(uninstall=lambda: done.append("hooks")),
                    "commit": (com, SimpleNamespace(untap=lambda: done.append("tap")))}
    out = rank.h1rank_release()
    assert out is not None and com.closed and done == ["tap", "hooks"] and rank._h1rank is None
    return out


@pytest.mark.parametrize("stated", [{}, {"errors": None}, {"errors": []}, {"errors": 0}],
                         ids=["absent", "null", "empty", "a false value"])
def test_a_rank_whose_committer_states_no_errors_releases_none(stated):
    assert _released({**COUNTS, **stated}) == {"lo": 3, "run_root": bytes(ROOT_BYTES).hex(), "commit_scheme": SCHEME,
                                               "name": "stand-in", **COUNTS, "errors": []}


def test_a_ranks_release_carries_its_first_eight_errors_each_cut_to_300_characters():
    errors = [f"{i}: " + "x" * 400 for i in range(10)]
    assert _released({**COUNTS, "errors": errors, "errors_n": 10})["errors"] == [e[:300] for e in errors[:8]]


class _Tap:
    """What `Holder.pull` and `ranks.feed` use of the window's tap: its port's name, what it took (the name, the engine
    step and the count of ids) and what it refused."""

    port = "ids"

    def __init__(self) -> None:
        self.took: list[tuple[str, int, int | None]] = []
        self.refused: list[str] = []
        self._ids: dict[int, Any] = {}

    def take(self, name: str, t: Any, step: int, n: int | None = None) -> None:
        self.took.append((name, step, n))


def _rank(lo: int, **more: Any) -> dict[str, Any]:
    """A rank's answer to ``h1rank_rows`` with no call since the last pull."""
    return {"lo": lo, "rows": [], "refused": [], "order_ok": True, "unopened": 0, **more}


def _pulled(*outs: dict[str, Any]) -> _Tap:
    """The window's tap after one pull that the ranks answer with `outs`."""
    tap = _Tap()
    run = SimpleNamespace(served=SimpleNamespace(tap=tap), shapes_=[(1, 1)])
    holder = Holder(run, SimpleNamespace(collective_rpc=lambda method: list(outs)), [{"decode_query_len": 1}], None)
    assert holder.pull() == 0 and sorted(holder.ranks) == sorted(o["lo"] for o in outs)
    return tap


SHIPPED = [("site/0", 2, b"x_q"), ("ids", 2, [5, 6])]


def test_the_first_ranks_acquisitions_are_the_windows_and_its_ids_the_steps():
    tap = _pulled(_rank(0, acquired=SHIPPED), _rank(4, acquired=[]))
    assert (tap.took, tap.refused) == ([("site/0", 2, None), ("ids", 2, 2)], [])


@pytest.mark.parametrize("none", [{}, {"acquired": None}, {"acquired": []}, {"acquired": 0}],
                         ids=["absent", "null", "empty", "a false value"])
def test_a_rank_that_shipped_no_acquisitions_feeds_nothing_and_is_still_the_first_rank(none):
    """The rank holding linear 0 answers with no ``acquired`` (it has no Commit): the tap takes nothing of it, and the
    next rank's ids are still not the first rank's."""
    tap = _pulled(_rank(0, **none), _rank(4, acquired=SHIPPED))
    assert (tap.took, tap.refused) == ([("site/0", 2, None)], ["engine step 2: a rank's input ids are not the first rank's"])


def _descriptor() -> bytes:
    """The descriptor bytes of a declared window of one call (N = 64 over K = 64, 32 slots, one step) on the served
    test's stand-in proof block: a window whose layout core reads, so a refusal below is the record's."""
    return descriptor(declared([(64, 64)], {0: "q"}, 32, 1, unit=_unit()))


def _sampled(record: dict[str, Any]) -> tuple[Any, Any]:
    """What core samples from the window's `record` at K = 1 over that descriptor, or its refusal."""
    return drawn_of(record, {"root": "00"}, 1, _descriptor())


def _refusal(why: str) -> tuple[None, tuple[str, str]]:
    """The refusal of a record whose sampling coin cannot be read, for the reason `why`."""
    return None, ("draw", f"the window's draw coin is malformed ({why})")


@pytest.mark.parametrize("record", [{}, {"commitment": None}, {"commitment": []}, {"commitment": "a root"}])
def test_a_record_with_no_commitment_object_is_refused_at_its_sampling_coin(record):
    assert _sampled(record) == _refusal("the window's record carries no commitment object")


@pytest.mark.parametrize("sample", [..., None, {}, {"coin": {}}])
def test_a_commitment_that_holds_no_opened_sampling_coin_is_refused_at_it(sample):
    """The commitment object states no sample (``...``), or one that is no object or holds no opened coin."""
    commitment = {} if sample is ... else {"draw": sample}
    assert _sampled({"commitment": commitment}) == _refusal(OPENING_FORM)


def test_a_slot_of_a_round_the_log_does_not_hold_is_refused_before_anything_is_opened():
    """`slot_reason` hands on the round's own refusal: the coins' commitment, the salt and the step's calls go unread."""
    coins = LiveCommitment("pouw-h1/rounds", 0, (("round", 2),), bytes(64))
    slot = RoundOpening(round=3, position=0, value=b"", path=())
    assert slot_reason([{"round": 0}, {"round": 1}], coins, b"salt", 4, [], slot) == "the round log has no round 3"


def _engine(**holds: Any) -> SimpleNamespace:
    """What `executor_of` reads of vLLM's ``LLM``: its engine, which holds `holds`."""
    return SimpleNamespace(llm_engine=SimpleNamespace(**holds))


def test_the_executor_is_the_engine_cores_held_by_the_engine_or_by_its_client():
    executor = SimpleNamespace(collective_rpc=lambda method: [])
    core = SimpleNamespace(model_executor=executor)
    assert executor_of(_engine(engine_core=core)) is executor
    assert executor_of(_engine(engine_core=SimpleNamespace(engine_core=core))) is executor


@pytest.mark.parametrize("holds", [
    {}, {"engine_core": None}, {"engine_core": SimpleNamespace()}, {"engine_core": SimpleNamespace(model_executor=None)},
    {"engine_core": SimpleNamespace(model_executor=SimpleNamespace())},
    {"engine_core": SimpleNamespace(model_executor=SimpleNamespace(collective_rpc=None))},
], ids=["no core", "a null core", "a client of a core in another process", "a null executor", "no collective_rpc",
        "a collective_rpc that is no call"])
def test_an_engine_with_no_executor_that_reaches_its_ranks_is_refused(holds):
    with pytest.raises(RuntimeError, match="no in-process executor with collective_rpc"):
        executor_of(_engine(**holds))
