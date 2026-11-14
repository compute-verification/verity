"""`verity_vllm.pipeline.tp.rank_replay.in_process`: what a rank's replay is told of its weights, and what it leaves on the
rank's committer.  Before the relations are evaluated the rank registers its live parameters again and holds their root to
the root the run committed; the record it hands the replay (`weights_check`) says whether the two are equal, which root it
held them to, or why it could not tell.  The committer, the run's commitment, the rank's Programs and the evaluation are
stand-ins: no committed value is read and no relation is evaluated."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.check import oracle_compare as OC
from verity_vllm.check.replay import driver as SR
from verity_vllm.pipeline.tp import rank_replay as RR

RUN_ROOT = bytes(range(64))
LIVE = "the engine's parameters"
SAMPLED, LINKED = {"result": "PASS", "grade": "evaluated"}, {"result": True, "pairs": 4, "linked": 4}
UNVERIFIED = {"equal": None, "self_consistent": None, "pin_of_record": None, "pinned": None}


class _Committer:
    """A committer whose `register_weights` notes the model it registers and, like the real one, keeps the registration it
    returns as its `weights`; with `fails`, it raises before it keeps any."""
    def __init__(self, replay_root: str | None, *, saved: dict | None = None, fails: BaseException | None = None):
        self.model = "the committer's own model"
        self.registered: list[Any] = []
        self._replay_root, self._fails = replay_root, fails
        if saved is not None:
            self.weights = saved

    def register_weights(self) -> dict:
        self.registered.append(self.model)
        if self._fails is not None:
            raise self._fails
        self.weights = {"root": self._replay_root, "tensors": 7}
        return self.weights


def _replay(monkeypatch, com: _Committer, *, registration: Any = None, replay: dict | None = None, world: int = 2,
            manifest: Any = None) -> tuple[dict, dict]:
    """Rank 1's `in_process` over `com` and a run whose commitment holds the weight registration `registration`: (its record
    without the wall, what the evaluation was given).  `manifest` is the required manifest the rank was started with."""
    given: dict[str, Any] = {}

    def evaluate(p, cfg, **kw):
        given.update(kw, p=p, cfg=cfg)
        return dict(SAMPLED), dict(LINKED)

    monkeypatch.setattr(RR, "load", lambda cfg, rank, world_, **kw: SimpleNamespace(world=world_, cfg=cfg, rank=rank))
    monkeypatch.setattr(RR, "xrank_facts", lambda st, p: st.update(facts_of=p))
    monkeypatch.setattr(RR, "index_memo", lambda p, man: {"memo": man})
    monkeypatch.setattr(RR, "evaluate", evaluate)
    monkeypatch.setattr(OC, "OpenedReader", lambda com_, rc: ("reader", com_, rc))
    monkeypatch.setattr(SR, "default_workers", lambda processes=1: 100 + processes)
    st = {"rank": 1, "world": world, "bmap": {"digest": "dd"}, "com": com, "weights": LIVE, "replay": replay, "manifest_doc": manifest}
    rc = SimpleNamespace(weight_registration=registration, run_root=RUN_ROOT)
    record = RR.in_process(st, rc)
    assert isinstance(record.pop("wall_s"), float)
    given["rc"], given["st"] = rc, st
    return record, given


def test_a_rank_whose_parameters_register_to_the_committed_root_replays_with_the_two_roots_named_equal(monkeypatch):
    com = _Committer("aa" * 64)
    record, given = _replay(monkeypatch, com, registration={"root": "aa" * 64, "tensors": 7})
    assert record == {"schema": RR.SCHEMA, "rank": 1, "method": RR.METHOD, "sampled_replay": SAMPLED, "boundary_linkage": LINKED}
    check = given["weights_check"]
    assert (check["equal"], check["self_consistent"], check["why"]) == (True, True, None)
    assert (check["committed_root"], check["replay_root"], check["tensors"]) == ("aa" * 64, "aa" * 64, 7)
    assert (check["pin_of_record"], check["pinned"]) == (None, None), "equal roots are self-consistency, never a pin of record"
    assert check["method"].startswith("rank 1: live parameters (weight SHARDS) re-registered at replay == the run's committed weights root")
    assert given["run_root"] == RUN_ROOT and given["committed"] == ("reader", com, given["rc"]) and given["bmap"] == {"digest": "dd"}
    assert given["memo"] == {"memo": None} and given["st"]["facts_of"] is given["p"]


def test_the_ranks_memo_is_of_its_own_identities_and_the_replay_is_given_the_whole_manifest(monkeypatch):
    manifest = {"ranks": 2, "identities": [{"rank": 0, "op": "a"}, {"rank": 1, "op": "b"}, {"op": "c"}, {"rank": 1, "op": "d"}]}
    given = _replay(monkeypatch, _Committer("aa"), registration={"root": "aa"}, manifest=manifest)[1]
    assert given["memo"] == {"memo": {"ranks": 2, "identities": [{"rank": 1, "op": "b"}, {"rank": 1, "op": "d"}]}}
    assert given["manifest_doc"] is manifest


def test_the_parameters_registered_again_are_the_engines_and_the_committers_model_is_put_back(monkeypatch):
    com = _Committer("aa")
    _replay(monkeypatch, com, registration={"root": "aa"})
    assert com.registered == [LIVE] and com.model == "the committer's own model"


def test_a_rank_whose_parameters_register_to_another_root_is_told_the_root_differs(monkeypatch):
    check = _replay(monkeypatch, _Committer("bb"), registration={"root": "aa"})[1]["weights_check"]
    assert (check["equal"], check["self_consistent"], check["why"]) == (False, False, "root differs")
    assert (check["committed_root"], check["replay_root"]) == ("aa", "bb")


def test_the_committers_own_registration_is_kept_through_the_replays(monkeypatch):
    saved = {"root": "aa", "tensors": 7, "of": "the Commit"}
    com = _Committer("bb", saved=saved)
    _replay(monkeypatch, com, registration={"root": "aa"})
    assert com.weights is saved, "the replay's registration does not replace the one the run committed"


@pytest.mark.parametrize("registration", [None, "aa", ["aa"]], ids=["none", "a text", "a list"])
def test_a_commitment_that_holds_no_registration_is_held_to_the_committers_own(monkeypatch, registration):
    com = _Committer("aa", saved={"root": "aa"})
    check = _replay(monkeypatch, com, registration=registration)[1]["weights_check"]
    assert (check["equal"], check["committed_root"], check["why"]) == (True, "aa", None)
    other = _replay(monkeypatch, _Committer("aa", saved={"root": "cc"}), registration=registration)[1]["weights_check"]
    assert (other["equal"], other["committed_root"], other["replay_root"], other["why"]) == (False, "cc", "aa", "root differs")


@pytest.mark.parametrize("registration, saved", [(None, None), ({}, None), ({"root": None}, {"root": "aa"}), ({"root": ""}, None), (None, {"tensors": 7})],
                         ids=["neither registration", "an empty registration", "a registration with no root beside a committer's with one",
                              "an empty root", "a committer's registration with no root"])
def test_a_run_with_no_committed_weights_root_is_neither_equal_nor_different(monkeypatch, registration, saved):
    check = _replay(monkeypatch, _Committer("aa", saved=saved), registration=registration)[1]["weights_check"]
    assert (check["equal"], check["self_consistent"], check["why"]) == (None, None, "no committed weights root on the run")
    assert check["committed_root"] in (None, "") and check["replay_root"] == "aa"


def test_a_rank_whose_parameters_cannot_be_registered_replays_by_reference_and_says_what_was_raised(monkeypatch):
    com = _Committer("aa", saved={"root": "aa"}, fails=RuntimeError("no parameters: " + "x" * 300))
    record, given = _replay(monkeypatch, com, registration={"root": "aa"})
    note = "by reference, unverified at replay: RuntimeError('no parameters: " + "x" * 300 + "')"
    assert given["weights_check"] == UNVERIFIED | {"note": note[:200]} and len(given["weights_check"]["note"]) == 200
    assert record["sampled_replay"] == SAMPLED and record["boundary_linkage"] == LINKED, "the relations are still evaluated"
    assert com.model == "the committer's own model" and com.weights == {"root": "aa"}


# the replay's workers

@pytest.mark.parametrize("replay, world, workers", [(None, 2, 102), ({}, 4, 104), ({"ranks_at_once": 3}, 8, 103), ({"ranks_at_once": 8}, 2, 102),
                                                    ({"ranks_at_once": 0}, 4, 104), ({"workers": 5, "ranks_at_once": 3}, 8, 5), ({"workers": 0}, 8, 0)],
                         ids=["no config: the world's ranks at once", "an empty config", "fewer ranks at once than the world", "more ranks at once than the world",
                              "no bound on the ranks at once", "the driver's count", "the driver's count of zero"])
def test_a_replay_given_no_worker_count_shares_the_cores_among_the_ranks_that_replay_at_once(monkeypatch, replay, world, workers):
    given = _replay(monkeypatch, _Committer("aa"), registration={"root": "aa"}, replay=replay, world=world)[1]
    assert given["cfg"]["workers"] == workers
    assert {k: v for k, v in given["cfg"].items() if k != "workers"} == {k: v for k, v in (replay or {}).items() if k != "workers"}
