"""Which runs of a `--replay-deferred` Commit await their replay (`replay_bundle.awaits_replay`, `pass_of`, `seal_all`), on rows
that hold no Commit record, or one with no sampled replay: such a run awaits nothing, so its bundle is not sealed.  CPU only;
the verdict of a deferred run and a sealed bundle's files are `test_replay_bundle.py`."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any


import verity_vllm.pipeline.replay_bundle as RB

Record = dict[str, Any]
DEFERRED, _ = RB.deferred_records(RB.path("/o", 1))
AWAITING: Record = {"commit_pass": None, "sampled_replay": DEFERRED}
#: an instrumented run's row with no record that awaits a replay: none at all, or one whose sampled replay is not a deferred one
NOT_AWAITING: list[Record] = [{}, {"commit": None}, {"commit": {}}, {"commit": {"commit_pass": None}},
                              {"commit": {"commit_pass": None, "sampled_replay": None}},
                              {"commit": {"commit_pass": None, "sampled_replay": {}}},
                              {"commit": {"commit_pass": None, "sampled_replay": {"grade": "sampled"}}},
                              {"commit": {"commit_pass": False, "sampled_replay": DEFERRED}}]


class Spans:
    """The timeline's spans, kept in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Record]] = []

    @contextlib.contextmanager
    def span(self, name: str, **kw: Any) -> Iterator[None]:
        self.calls.append((name, kw))
        yield


def test_a_run_awaits_its_replay_while_it_has_no_pass_and_its_sampled_replay_is_the_deferred_one() -> None:
    assert RB.awaits_replay(AWAITING) and RB.awaits_replay({"sampled_replay": DEFERRED})
    assert not RB.awaits_replay(AWAITING | {"commit_pass": True}), "a run that passed awaits nothing"
    assert [RB.awaits_replay(r["commit"]) for r in NOT_AWAITING if r.get("commit") is not None] == [False] * 6


def test_a_commit_that_deferred_nothing_seals_nothing_and_its_verdict_names_no_bundle(tmp_path: Path) -> None:
    verdict: Record = {"commit_pass": True}
    RB.seal_all(tmp_path, {}, [{"pair": 1, "arm": "instrumented", "commit": {"commit_pass": True}}], verdict, [], tl=Spans())
    assert verdict == {"commit_pass": True} and list(tmp_path.iterdir()) == []
