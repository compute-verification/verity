"""`verity_vllm.acquire.h1_commit.committer` on CPU, with a stand-in for the windowed collector's class that keeps what it is
built with and what the environment says of retention as it is built.

It catches H1's committer built another way than the declared Commit's collector (`pipeline.commit`'s native_collect_v2b, with
its wrap and no arena size, as `pipeline.served_match` builds it), on another engine or run than the one given, or under
another scheme than rows-sha512, whose roots H1 binds; and a collector built before retention is set, or over a retention the
environment names (an empty one too, which the collector refuses by name): left to itself H1's collector retains its values
nowhere."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm import settings
from verity_vllm.acquire import h1_commit as HC
from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.pipeline import commit as PC


class Collector:
    """In `NativeCollectCommitter`'s place: what it was built with, and VERITY_RETAIN as the environment held it then."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        self.args, self.kw, self.retain = args, kw, settings.raw("VERITY_RETAIN")


@pytest.mark.parametrize("held,retain", [(None, "none"), ("host", "host"), ("", "")], ids=["unset", "host", "empty"])
def test_h1s_committer_is_the_declared_commits_collector_under_rows_sha512_and_retains_nowhere_unless_told(
        monkeypatch: pytest.MonkeyPatch, held: str | None, retain: str) -> None:
    monkeypatch.setattr(NC, "NativeCollectCommitter", Collector)
    monkeypatch.setenv("VERITY_RETAIN", held or "")                   # so the test leaves the variable as it found it
    if held is None:
        monkeypatch.delenv("VERITY_RETAIN")
    engine = object()
    com = HC.committer(engine, "run-7")
    assert type(com) is Collector and com.args == (engine,) and com.retain == retain
    declared = PC.make_committer("native_collect_v2b", engine, SimpleNamespace(arena_mb=0, commit_scheme=HC.SCHEME), "run-7")
    assert HC.SCHEME == "rows-sha512" and declared.args == (engine,) and declared.kw.pop("populate_threads") is None
    assert com.kw == {**declared.kw, "wrap": True}
