"""Which runs of a `--replay-deferred` Commit await their replay (`replay_bundle.awaits_replay`, `pass_of`, `seal_all`), on rows
that hold no Commit record, or one with no sampled replay: such a run awaits nothing, so its bundle is not sealed.  CPU only;
the verdict of a deferred run and a sealed bundle's files are `test_replay_bundle.py`."""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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


@pytest.mark.parametrize("row", NOT_AWAITING)
def test_a_run_with_no_record_that_awaits_a_replay_fails_the_commit_s_pass(row: Record) -> None:
    passed, awaiting = {"arm": "instrumented", "commit": {"commit_pass": True}}, {"arm": "instrumented", "commit": AWAITING}
    assert RB.pass_of([passed, awaiting]) is None
    assert RB.pass_of([passed, awaiting, {"arm": "instrumented"} | row]) is False
    assert RB.pass_of([passed, awaiting, {"arm": "control"} | row]) is None, "a control run has no Commit to pass"
    assert RB.pass_of([{"arm": "control"} | row]) is False, "no instrumented run, no pass"


@pytest.mark.parametrize("row", NOT_AWAITING)
def test_only_the_bundle_of_a_run_that_awaits_its_replay_is_sealed(tmp_path: Path, caplog: pytest.LogCaptureFixture, row: Record) -> None:
    """Pair 1 awaits its replay, pair 2's row holds no record that does, and pair 3 has no instrumented row at all."""
    caplog.set_level(logging.INFO, logger=RB.__name__)
    for pair in (1, 2, 3):
        (RB.partial(tmp_path, pair) / "store").mkdir(parents=True)
        (RB.partial(tmp_path, pair) / "store" / "store.json").write_text("{}")
    (tmp_path / "binding_map_p1.json").write_text('{"entries": []}')
    rows = [{"pair": 1, "arm": "instrumented", "commit": dict(AWAITING)}, {"pair": 2, "arm": "instrumented"} | row,
            {"pair": 3, "arm": "control", "commit": dict(AWAITING)}]
    verdict: Record = {"commit_pass": None}
    tl = Spans()
    RB.seal_all(tmp_path, {p: {"pair": p, "run_root": "ab" * 32} for p in (1, 2, 3)}, rows, verdict, [1], tl=tl)
    sealed = RB.path(tmp_path, 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["binding_map_p1.json", sealed.name], "the other two partial bundles are gone"
    opened = RB.open_bundle(sealed)
    assert opened["context"] == {"schema": RB.SCHEMA, "pair": 1, "run_root": "ab" * 32, "row": rows[0], "verdict": {"commit_pass": None}}
    assert verdict == {"commit_pass": None, "replay_deferred": {"pending_pairs": [1], "bundles": [{
        "pair": 1, "path": str(sealed), "manifest_sha256": opened["manifest_sha256"], "bytes": opened["manifest"]["bytes"], "files": 3}]}}
    assert tl.calls == [("closure.replay_bundle", {"pair": 1})]
    assert [r.getMessage() for r in caplog.records if r.name == RB.__name__] == [
        f"[delta] REPLAY BUNDLE (pair 1): {sealed} sealed, 3 files, 0 MiB, manifest sha256 {opened['manifest_sha256'][:16]}"]


def test_a_commit_that_deferred_nothing_seals_nothing_and_its_verdict_names_no_bundle(tmp_path: Path) -> None:
    verdict: Record = {"commit_pass": True}
    RB.seal_all(tmp_path, {}, [{"pair": 1, "arm": "instrumented", "commit": {"commit_pass": True}}], verdict, [], tl=Spans())
    assert verdict == {"commit_pass": True} and list(tmp_path.iterdir()) == []
